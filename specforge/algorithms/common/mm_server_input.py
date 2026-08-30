"""Image+text ``ServerInputAdapter`` for DFlash-family online capture.

The runtime routes every non-text modality through this port
(:class:`specforge.algorithms.common.providers.ServerInputAdapter`). The adapter
owns three things and nothing else:

``load_input_tools``
    Load the target's ``AutoProcessor`` instead of a bare tokenizer.
``prepare_prompts``
    Turn image+conversation records into JSON-safe capture payloads whose
    ``input_ids`` already contain the expanded image tokens.
``build_request_inputs``
    Attach the images to one batched SGLang ``/generate`` request.

Transport keys (``extra_key``, ``sampling_params``, ``spec_capture``) stay
runtime-owned; this adapter returns model inputs only.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

# Per-worker state. Encoding one record costs an image decode plus a processor
# pass, so prompts are built in a process pool; each worker loads the processor
# once and keeps torch single-threaded to avoid oversubscribing the box.
_WORKER: dict[str, Any] = {}


def _init_worker(config: Any) -> None:
    import torch

    from specforge.data.mm_preprocessing import (
        _assistant_header_ids,
        _end_token_ids,
        load_mm_processor,
    )

    torch.set_num_threads(1)
    processor = load_mm_processor(config)
    _WORKER["config"] = config
    _WORKER["processor"] = processor
    _WORKER["header_ids"] = _assistant_header_ids(processor)
    _WORKER["end_ids"] = _end_token_ids(processor)


def _encode_worker(record: Mapping[str, Any]):
    """Encode one record inside a pool worker; None drops an unusable row."""
    from specforge.data.mm_preprocessing import encode_mm_record

    config = _WORKER["config"]
    try:
        return encode_mm_record(
            record,
            _WORKER["processor"],
            image_root=config.data.image_root,
            max_length=config.data.max_length,
            train_only_last_turn=config.data.train_only_last_turn,
            header_ids=_WORKER["header_ids"],
            end_ids=_WORKER["end_ids"],
        )
    except (OSError, ValueError):
        # unreadable image or malformed record: drop it like the text path does
        return None


class ImageServerInputAdapter:
    """Prepare image+text prompts and image-bearing capture requests."""

    def __init__(self, config: Any) -> None:
        self._config = config

    # -- input tooling --------------------------------------------------------
    def load_input_tools(self, config: Any) -> Any:
        from specforge.data.mm_preprocessing import load_mm_processor

        return load_mm_processor(config)

    # -- prompt preparation ---------------------------------------------------
    def prepare_prompts(
        self,
        config: Any,
        input_tools: Any,
        *,
        draft_config: Any,
    ) -> list[dict[str, Any]]:
        from concurrent.futures import ProcessPoolExecutor

        from specforge.algorithms.model_providers import dflash_min_loss_tokens
        from specforge.data.prompt_builder import _iter_records

        del input_tools  # each pool worker loads its own processor

        source_path = config.data.train_data_path or config.data.prompts_path
        if not source_path:
            raise ValueError("prompt preparation requires a non-empty data path")

        min_loss_tokens = dflash_min_loss_tokens(config, draft_config)
        max_prompts = config.data.max_prompts or None

        # Only read as many records as can survive filtering; without a cap this
        # walks the whole corpus, which for image data is hours of decoding.
        records = []
        for _line_number, record in _iter_records(source_path):
            records.append(record)
            if max_prompts is not None and len(records) >= max_prompts:
                break

        workers = max(1, int(config.data.build_dataset_num_proc))
        print(
            f"[mm-prompts] encoding {len(records)} image records with "
            f"{workers} workers (max_prompts={max_prompts})",
            flush=True,
        )

        prompts: list[dict[str, Any]] = []
        dropped = 0
        started = time.monotonic()
        report_every = max(1, len(records) // 20)
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(config,)
        ) as pool:
            for done, payload in enumerate(
                pool.map(_encode_worker, records, chunksize=8), start=1
            ):
                if payload is None or sum(payload["loss_mask"]) < min_loss_tokens:
                    dropped += 1
                else:
                    prompts.append({"payload": payload})
                if done % report_every == 0 or done == len(records):
                    rate = done / max(time.monotonic() - started, 1e-6)
                    remaining = (len(records) - done) / rate if rate else 0.0
                    print(
                        f"[mm-prompts] {done}/{len(records)} encoded "
                        f"({rate:.1f}/s, ~{remaining / 60:.1f} min left, "
                        f"kept={len(prompts)}, dropped={dropped})",
                        flush=True,
                    )

        if not prompts:
            raise ValueError(
                f"no usable image prompts were produced from {source_path!r}; "
                "check the image paths, chat template, and loss-mask coverage"
            )
        print(f"[mm-prompts] ready: {len(prompts)} prompts", flush=True)
        return prompts

    # -- request construction -------------------------------------------------
    def build_request_inputs(
        self,
        tasks: Sequence[Any],
    ) -> Mapping[str, Any]:
        """Send pre-expanded token ids together with their images.

        ``input_ids`` already carry the processor's image-token expansion, so the
        captured features line up one-to-one with the payload the trainer reads.

        A blend such as LLaVA-OneVision mixes text-only rows into an otherwise
        multimodal file, so a batch can hold both; dropping the text-only third
        is not an option. Every entry is a list -- ``[path]`` for a picture and
        ``[]`` for none -- because that is the shape SGLang reads per batch
        position: a flat list makes it label *every* entry "image", including
        the ones with nothing to load, while a list of lists lets it assign no
        modality to the empty ones. When the whole batch is text, ``image_data``
        is left out and the request is an ordinary text one.
        """
        input_ids: list[list[int]] = []
        image_data: list[list[Any]] = []
        for task in tasks:
            payload = task.payload
            input_ids.append(list(payload["input_ids"]))
            image = payload.get("image")
            image_data.append([image] if image else [])
        if not any(image_data):
            return {"input_ids": input_ids}
        return {"input_ids": input_ids, "image_data": image_data}


def build_image_input_adapter(config: Any) -> ImageServerInputAdapter:
    """Factory registered as ``ServerStreamingProvider.build_input_adapter``."""
    return ImageServerInputAdapter(config)


__all__ = ["ImageServerInputAdapter", "build_image_input_adapter"]
