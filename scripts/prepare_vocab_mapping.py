#!/usr/bin/env python
"""Build only the EAGLE3 ``t2d``/``d2t`` vocab mapping from a training config.

``scripts/prepare_hidden_states.py`` also produces a mapping, but it gets there
by running the target model over the corpus and writing every hidden state to
disk -- hundreds of GB for a corpus this size, all of it useless once the
mapping exists. Nothing about the mapping needs the model: it is a frequency
ranking over the token ids the loss mask supervises, so tokenized text is
enough.

That also makes it modality-agnostic. Image tokens live in the prompt, where
``loss_mask`` is 0, so they never enter the counts; the images themselves are
never opened and the vision tower is never built. A multimodal corpus is
therefore processed exactly like a text one, at tokenizer speed.

Usage::

    python scripts/prepare_vocab_mapping.py \\
        --config scripts/mmtraining_configs/qwen3.5-4b-eagle3.yaml

The output path defaults to ``model.vocab_mapping_path`` from the config, which
is the file a disaggregated EAGLE3 run requires up front.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Any, Iterator, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="training YAML the mapping is built for"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="where to write the mapping; defaults to model.vocab_mapping_path",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="corpus to count over; defaults to data.train_data_path",
    )
    parser.add_argument(
        "--target-path",
        default=None,
        help=(
            "local snapshot to read the tokenizer/config from, bypassing the "
            "Hub. Use it when data.cache_dir does not already hold the target "
            "(only small config files would be fetched, never weights)."
        ),
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=None,
        help="stop after this many records (a quick sanity run)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="records rendered and tokenized per batch",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rebuild even when the output file already exists",
    )
    return parser.parse_args()


class _TokenizerOnlyProcessor:
    """Give a bare tokenizer the two attributes the mm helpers reach for."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.apply_chat_template(*args, **kwargs)


def load_processor(cfg: Any) -> Any:
    """The target's processor for image corpora, its tokenizer otherwise."""
    if cfg.model.input_modality == "image":
        from specforge.data.mm_preprocessing import load_mm_processor

        return load_mm_processor(cfg)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir or None,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    return _TokenizerOnlyProcessor(tokenizer)


def resolve_vocab_sizes(cfg: Any) -> tuple[int, int]:
    """(target_vocab_size, draft_vocab_size) for the configured pair."""
    import json

    from specforge.modeling.target.target_utils import (
        load_target_config,
        target_vocab_size,
    )

    target_config = load_target_config(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir or None,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    target_size = int(target_vocab_size(target_config))

    draft_source = cfg.model.draft_model_config
    if not draft_source:
        raise ValueError(
            "model.draft_model_config is required: the draft vocab size decides "
            "how many entries the mapping holds"
        )
    path = draft_source
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    if not os.path.isfile(path):
        raise ValueError(
            f"cannot read draft config {draft_source!r}; point "
            "model.draft_model_config at a local JSON file or directory"
        )
    with open(path, encoding="utf-8") as handle:
        draft_config = json.load(handle)
    draft_size = draft_config.get("draft_vocab_size")
    if not draft_size:
        raise ValueError(
            f"{path} has no draft_vocab_size; only draft models with a reduced "
            "vocabulary (EAGLE3-family) need a mapping"
        )
    return target_size, int(draft_size)


def iter_batches(path: str, size: int, limit: Optional[int]) -> Iterator[list[dict]]:
    """Yield record batches from a JSON array or JSONL corpus."""
    from specforge.data.prompt_builder import _iter_records

    batch: list[dict] = []
    seen = 0
    for _line_number, record in _iter_records(path):
        batch.append(record)
        seen += 1
        if len(batch) >= size:
            yield batch
            batch = []
        if limit is not None and seen >= limit:
            break
    if batch:
        yield batch


def count_supervised_tokens(
    records: list[dict],
    processor: Any,
    *,
    header_ids: list[int],
    end_ids: set[int],
    train_only_last_turn: bool,
    counter: Counter,
) -> tuple[int, int]:
    """Add one batch's supervised token ids to ``counter``.

    Returns ``(kept, dropped)``. The conversation is rendered and tokenized
    without its images: an unexpanded ``<|image_pad|>`` shifts positions but
    never carries loss, so the supervised ids -- and therefore the counts --
    are the ones training would see.
    """
    from specforge.data.mm_preprocessing import build_loss_mask, to_chat_messages

    texts = []
    for record in records:
        conversations = record.get("conversations")
        if not conversations:
            continue
        try:
            messages = to_chat_messages(conversations)
            texts.append(
                processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    if not texts:
        return 0, len(records)

    tokenizer = processor.tokenizer
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]

    kept = 0
    for input_ids in encoded:
        loss_mask = build_loss_mask(
            input_ids,
            header_ids=header_ids,
            end_ids=end_ids,
            train_only_last_turn=train_only_last_turn,
        )
        supervised = [
            token for token, keep in zip(input_ids, loss_mask) if keep
        ]
        if not supervised:
            continue
        counter.update(supervised)
        kept += 1
    return kept, len(records) - kept


def main() -> int:
    args = parse_args()

    import torch
    import yaml

    from specforge.config.schema import Config
    from specforge.data.mm_preprocessing import _assistant_header_ids, _end_token_ids
    from specforge.data.preprocessing import process_token_dict_to_mappings

    with open(args.config, encoding="utf-8") as handle:
        cfg = Config(**yaml.safe_load(handle))

    output = args.output or cfg.model.vocab_mapping_path
    if not output:
        raise ValueError(
            "no output path: pass --output or set model.vocab_mapping_path"
        )
    if os.path.exists(output) and not args.overwrite:
        print(f"vocab mapping already exists at {output}; pass --overwrite to rebuild")
        return 0

    data_path = args.data_path or cfg.data.train_data_path or cfg.data.prompts_path
    if not data_path:
        raise ValueError("no corpus: pass --data-path or set data.train_data_path")

    if args.target_path:
        # Read the tokenizer/config straight from a local snapshot. Nothing here
        # loads weights either way; this only avoids a Hub round trip when the
        # configured cache_dir has not seen the target yet.
        cfg.model.target_model_path = args.target_path

    target_size, draft_size = resolve_vocab_sizes(cfg)
    print(
        f"target vocab {target_size}, draft vocab {draft_size}, "
        f"modality {cfg.model.input_modality!r}"
    )
    print(f"counting supervised tokens in {data_path}")

    processor = load_processor(cfg)
    header_ids = _assistant_header_ids(processor)
    end_ids = _end_token_ids(processor)

    counter: Counter = Counter()
    kept = dropped = 0
    for batch in iter_batches(data_path, args.batch_size, args.num_records):
        batch_kept, batch_dropped = count_supervised_tokens(
            batch,
            processor,
            header_ids=header_ids,
            end_ids=end_ids,
            train_only_last_turn=cfg.data.train_only_last_turn,
            counter=counter,
        )
        kept += batch_kept
        dropped += batch_dropped
        print(
            f"\r  {kept} records counted ({dropped} unusable), "
            f"{len(counter)} distinct tokens",
            end="",
            flush=True,
        )
    print()

    if not counter:
        raise ValueError(
            f"no supervised tokens found in {data_path}; check the chat template "
            "and that the records carry assistant turns"
        )
    if len(counter) < draft_size:
        print(
            f"WARNING: only {len(counter)} distinct supervised tokens for a draft "
            f"vocab of {draft_size}; the remainder is padded with unused ids"
        )

    d2t, t2d = process_token_dict_to_mappings(counter, draft_size, target_size)

    directory = os.path.dirname(os.path.abspath(output))
    os.makedirs(directory, exist_ok=True)
    torch.save({"d2t": d2t, "t2d": t2d}, output)
    print(f"saved vocab mapping to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
