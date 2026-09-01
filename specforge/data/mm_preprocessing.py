"""Image+text prompt preparation for online target capture.

The text path tokenizes conversations with a plain tokenizer. A vision-language
target instead needs its ``AutoProcessor``: the chat template emits an image
placeholder that the processor expands into the exact number of image tokens the
target model consumes. Capture alignment depends on that expansion, because the
producer sends ``input_ids`` to the server and every captured per-token feature
(``hidden_states``, ``loss_mask``) must have the same length.

Records follow the ShareGPT4V layout produced by ``scripts/prepare_data_mm.py``::

    {"id": ..., "image": "/path/to/image.jpg",
     "conversations": [{"role": "user", "content": "<image>\\nWhat ...?"},
                       {"role": "assistant", "content": "..."}]}

Loss masking marks only assistant reply tokens. Prompt tokens, and therefore all
image tokens, stay at 0.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

IMAGE_PLACEHOLDER = "<image>"


def load_mm_processor(config: Any):
    """Load the target model's multimodal processor (tokenizer + image processor)."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        config.model.target_model_path,
        cache_dir=config.model.cache_dir,
        trust_remote_code=config.model.trust_remote_code,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError(
            "AutoProcessor for "
            f"{config.model.target_model_path!r} exposes no tokenizer; "
            "the image modality requires a vision-language target model"
        )
    pad_override = getattr(config.model, "tokenizer_pad_token_id", None)
    if pad_override is not None:
        tokenizer.pad_token_id = pad_override
    elif tokenizer.pad_token_id is None:
        fallback = tokenizer.eos_token_id
        if isinstance(fallback, (list, tuple)):
            fallback = fallback[0] if fallback else None
        tokenizer.pad_token_id = fallback

    _limit_image_tokens(processor, resolve_image_token_cap(config))
    return processor


def resolve_image_token_cap(config) -> int:
    """Visual-token budget per image; 0 keeps the checkpoint's own resolution.

    Capping here only works if the capture server applies the same limit. It
    re-expands ``image_data`` itself, so a client-only cap desynchronizes the two
    token counts and the server rejects the request.
    """
    return int(getattr(config.data, "image_max_tokens", 0) or 0)


def pixels_per_visual_token(processor) -> int:
    """Pixels collapsed into one visual token (patch_size * merge_size squared)."""
    image_processor = getattr(processor, "image_processor", None)
    patch = int(getattr(image_processor, "patch_size", 14) or 14)
    merge = int(getattr(image_processor, "merge_size", 2) or 2)
    return (patch * merge) ** 2


def _limit_image_tokens(processor, max_tokens: int) -> None:
    """Bound an image to ``max_tokens`` visual tokens by capping its pixel budget.

    Without this an image is resized only by the checkpoint's native limits, so a
    high-resolution photo can occupy more positions than ``data.max_length`` and
    crowd out the assistant reply the draft is supposed to learn.
    """
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or max_tokens <= 0:
        return
    max_pixels = max_tokens * pixels_per_visual_token(processor)

    # Newer processors carry a SizeDict; older ones expose max_pixels directly.
    size = getattr(image_processor, "size", None)
    if size is not None:
        longest = (
            size.get("longest_edge")
            if isinstance(size, dict)
            else getattr(size, "longest_edge", None)
        )
        if longest is not None and max_pixels < int(longest):
            if isinstance(size, dict):
                size["longest_edge"] = max_pixels
            else:
                size.longest_edge = max_pixels
    if getattr(image_processor, "max_pixels", None) is not None:
        image_processor.max_pixels = min(image_processor.max_pixels, max_pixels)


def resolve_image_path(image: Any, image_root: str) -> str | None:
    """Resolve one record's image reference against the configured root.

    ``None`` for a text-only record. A blend such as LLaVA-OneVision mixes
    text-only rows into an otherwise multimodal file, and those rows train the
    draft just as well -- they simply carry no picture.
    """
    if image is None:
        return None
    if not isinstance(image, str) or not image:
        raise ValueError(
            f"record image must be a non-empty string or None, got {image!r}"
        )
    if image_root and not os.path.isabs(image):
        return os.path.join(image_root, image)
    return image


def to_chat_messages(conversations: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Convert stored conversations into processor chat messages.

    ``<image>`` placeholders become structured image parts, which is what
    vision chat templates expect. Text-only turns keep a plain string so the
    template renders them exactly like the text path does.
    """
    messages: list[dict] = []
    for turn in conversations:
        role = turn.get("role")
        content = turn.get("content", "")
        if role is None:
            raise ValueError(f"conversation turn is missing a role: {turn!r}")
        if not isinstance(content, str):
            raise ValueError(f"conversation content must be a string, got {content!r}")
        if IMAGE_PLACEHOLDER in content:
            parts: list[dict] = []
            for index, chunk in enumerate(content.split(IMAGE_PLACEHOLDER)):
                if index:
                    parts.append({"type": "image"})
                chunk = chunk.strip()
                if chunk:
                    parts.append({"type": "text", "text": chunk})
            messages.append({"role": role, "content": parts})
        else:
            messages.append({"role": role, "content": content})
    return messages


_ASSISTANT_PROBE_MARKER = "ZZASSISTANTREPLYMARKERZZ"


def _assistant_header_ids(processor) -> list[int]:
    """Derive the text that precedes an assistant reply, as token ids.

    Derived from a rendered conversation rather than from
    ``add_generation_prompt=True``: reasoning templates (Qwen3-style) append an
    extra ``<think>`` opener to the *generation* prompt that never appears in a
    stored assistant turn, so a generation-prompt diff finds a header that
    matches no training record.

    Rendering a probe reply and locating the marker isolates exactly the control
    tokens that precede assistant content in real data, for any template.
    """
    probe_user = [{"role": "user", "content": "x"}]
    probe_full = probe_user + [
        {"role": "assistant", "content": _ASSISTANT_PROBE_MARKER}
    ]
    text_user = processor.apply_chat_template(
        probe_user, tokenize=False, add_generation_prompt=False
    )
    text_full = processor.apply_chat_template(
        probe_full, tokenize=False, add_generation_prompt=False
    )
    marker_index = text_full.find(_ASSISTANT_PROBE_MARKER)
    if marker_index < 0:
        raise ValueError(
            "chat template did not render the probe assistant reply; cannot "
            "derive the assistant header for loss masking"
        )
    common = os.path.commonprefix([text_user, text_full[:marker_index]])
    header_text = text_full[len(common) : marker_index]
    if not header_text:
        raise ValueError(
            "chat template produced an empty assistant header; cannot locate "
            "assistant replies for loss masking"
        )
    header_ids = processor.tokenizer(header_text, add_special_tokens=False)["input_ids"]
    if not header_ids:
        raise ValueError("assistant header tokenized to an empty sequence")
    return list(header_ids)


def _end_token_ids(processor) -> set[int]:
    """Token ids that terminate an assistant reply."""
    tokenizer = processor.tokenizer
    ids: set[int] = set()
    eos = tokenizer.eos_token_id
    if isinstance(eos, (list, tuple)):
        ids.update(int(value) for value in eos if value is not None)
    elif eos is not None:
        ids.add(int(eos))
    for token in ("<|im_end|>", "<|eot_id|>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:  # pragma: no cover - tokenizer without this token
            continue
        if isinstance(token_id, int) and token_id >= 0:
            ids.add(token_id)
    return ids


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> list[int]:
    """Return every start index where ``needle`` occurs in ``haystack``."""
    if not needle or len(needle) > len(haystack):
        return []
    first = needle[0]
    width = len(needle)
    starts: list[int] = []
    for index in range(len(haystack) - width + 1):
        if haystack[index] == first and list(haystack[index : index + width]) == list(
            needle
        ):
            starts.append(index)
    return starts


def build_loss_mask(
    input_ids: Sequence[int],
    *,
    header_ids: Sequence[int],
    end_ids: set[int],
    train_only_last_turn: bool,
) -> list[int]:
    """Mark assistant reply tokens with 1 and everything else with 0.

    A reply runs from the end of an assistant header to its terminating token
    (inclusive, so the draft also learns where a reply stops).
    """
    loss_mask = [0] * len(input_ids)
    starts = _find_subsequence(input_ids, header_ids)
    if not starts:
        return loss_mask
    if train_only_last_turn:
        starts = starts[-1:]
    for start in starts:
        cursor = start + len(header_ids)
        while cursor < len(input_ids):
            loss_mask[cursor] = 1
            if input_ids[cursor] in end_ids:
                break
            cursor += 1
    return loss_mask


def encode_mm_record(
    record: Mapping[str, Any],
    processor,
    *,
    image_root: str,
    max_length: int,
    train_only_last_turn: bool,
    header_ids: Sequence[int],
    end_ids: set[int],
) -> dict[str, Any] | None:
    """Encode one image+text record into JSON-safe capture payload fields.

    Returns ``None`` when the record carries no trainable assistant token, which
    matches how the text path drops unusable rows.
    """
    from PIL import Image

    conversations = record.get("conversations")
    if not conversations:
        raise ValueError(f"record {record.get('id')!r} has no conversations")

    image_path = resolve_image_path(record.get("image"), image_root)
    # Checked before templating: to_chat_messages turns every placeholder into a
    # structured image part, so by then they are gone from the text. One part is
    # emitted per occurrence, but only `image_path` is ever handed to the
    # processor, so a record carrying more placeholders than pictures makes the
    # processor index a grid row that does not exist. That happens for real: an
    # OCR reply transcribing a web page reproduces its literal <image> markup.
    placeholders = sum(
        turn["content"].count(IMAGE_PLACEHOLDER)
        for turn in conversations
        if isinstance(turn.get("content"), str)
    )
    available = 0 if image_path is None else 1
    if placeholders > available:
        raise ValueError(
            f"record {record.get('id')!r} carries {placeholders} "
            f"{IMAGE_PLACEHOLDER} placeholder(s) but {available} image(s)"
        )

    messages = to_chat_messages(conversations)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    if image_path is None:
        # A text-only row: the processor must not be handed an empty image
        # list, which some implementations turn into a zero-length pixel batch
        # rather than a plain text encode.
        batch = processor(text=[text], return_tensors="pt")
    else:
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            batch = processor(text=[text], images=[image], return_tensors="pt")

    input_ids = batch["input_ids"][0].tolist()
    if len(input_ids) > max_length:
        # Drop rather than truncate: the capture server rebuilds the sequence
        # from image_data at full resolution, so a shortened prompt no longer
        # matches what it produces and the request is refused. Truncating also
        # tends to cut away the assistant reply, leaving nothing to train on.
        return None

    loss_mask = build_loss_mask(
        input_ids,
        header_ids=header_ids,
        end_ids=end_ids,
        train_only_last_turn=train_only_last_turn,
    )
    if not any(loss_mask):
        return None

    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        # None for a text-only row; the capture request carries no picture for
        # it and the server builds a plain text sequence.
        "image": image_path,
    }


__all__ = [
    "IMAGE_PLACEHOLDER",
    "build_loss_mask",
    "encode_mm_record",
    "load_mm_processor",
    "resolve_image_path",
    "to_chat_messages",
]
