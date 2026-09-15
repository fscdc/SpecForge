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


#: Record key under which ``prepare_prompts`` attaches the joined sidecar entry
#: (``(n_tokens, fingerprint, g)`` or None) before handing it to a worker.
VISUAL_SCORE_RECORD_KEY = "_visual_score"


def _wants_visual_score(config: Any) -> bool:
    """Whether the run's strategy consumes the ``visual_score`` channel.

    Only MMFlash lists it in its capture layout, and the capture adapter
    refuses a payload that lacks a listed key, so an MMFlash run must always
    carry the channel -- with or without a sidecar. Other image strategies do
    not list it and should not pay for a third per-token array in memory.
    """
    return getattr(getattr(config, "training", None), "strategy", None) == "mmflash"


def _encode_worker(record: Mapping[str, Any]):
    """Encode one record inside a pool worker.

    Returns ``(payload, visual_status)``; ``payload`` is None for an unusable
    row. ``visual_status`` reports what happened to the visual-score channel
    (``"text_only"``, ``"scored"``, ``"missing"``, ``"misaligned"``, or
    ``"off"`` when the strategy does not use it) so the caller can print counts.
    """
    from specforge.data.mm_preprocessing import encode_mm_record
    from specforge.data.visual_score import (
        expand_visual_score,
        text_only_visual_score,
    )

    config = _WORKER["config"]
    try:
        payload = encode_mm_record(
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
        return None, "dropped"
    if payload is None:
        return None, "dropped"
    if not _wants_visual_score(config):
        return payload, "off"
    num_tokens = len(payload["input_ids"])
    if payload["image"] is None:
        payload["visual_score"] = text_only_visual_score(num_tokens)
        return payload, "text_only"
    channel, status = expand_visual_score(
        payload["loss_mask"],
        record.get(VISUAL_SCORE_RECORD_KEY),
        input_ids=payload["input_ids"],
    )
    payload["visual_score"] = channel
    return payload, status


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

        # Loaded before the corpus is walked: a sidecar that is missing or
        # points at the wrong place should fail in seconds, not after reading
        # three gigabytes of jsonl.
        score_table = self._load_visual_scores(config)

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

        # Joined here, once, in the main process: keyed by record id, so it
        # must happen before the records are handed to workers that never see
        # the id again.
        records, join_counts = self._attach_visual_scores(config, records, score_table)

        prompts: list[dict[str, Any]] = []
        dropped = 0
        visual_counts: dict[str, int] = {}
        started = time.monotonic()
        report_every = max(1, len(records) // 20)
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(config,)
        ) as pool:
            for done, (payload, visual_status) in enumerate(
                pool.map(_encode_worker, records, chunksize=8), start=1
            ):
                if payload is None or sum(payload["loss_mask"]) < min_loss_tokens:
                    dropped += 1
                else:
                    prompts.append({"payload": payload})
                    visual_counts[visual_status] = visual_counts.get(visual_status, 0) + 1
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
        self._report_visual_scores(config, join_counts, visual_counts, len(prompts))
        return prompts

    # -- visual-dependency scores ----------------------------------------------
    @staticmethod
    def _load_visual_scores(config: Any):
        """The sidecar named by ``data.visual_score_path``, or None."""
        if not _wants_visual_score(config):
            return None
        from specforge.data.visual_score import VisualScoreTable

        path = getattr(config.data, "visual_score_path", "") or ""
        if not path:
            print(
                "[visual-score] data.visual_score_path is empty: g=0 everywhere, so "
                f"every row trains on the plain training.loss_type={config.training.loss_type!r} "
                "objective (the visual multiplier is 1)",
                flush=True,
            )
            return None
        started = time.monotonic()
        table = VisualScoreTable.load(
            path,
            transform=config.data.visual_score_transform,
            binary_threshold=config.data.visual_score_binary_threshold,
            confidence_gate=getattr(config.data, "visual_score_confidence_gate", True),
        )
        print(
            f"[visual-score] loaded {path!r} in {time.monotonic() - started:.0f}s: "
            f"{table.stats.describe()}",
            flush=True,
        )
        return table

    @staticmethod
    def _attach_visual_scores(
        config: Any, records: list, table
    ) -> tuple[list, dict[str, int]]:
        """Attach the sidecar entry to each record (see ``specforge.data.visual_score``)."""
        if not _wants_visual_score(config):
            return records, {}
        from specforge.data.visual_score import join_visual_scores

        return join_visual_scores(records, table, key=VISUAL_SCORE_RECORD_KEY)

    @staticmethod
    def _report_visual_scores(
        config: Any,
        join_counts: dict[str, int],
        visual_counts: dict[str, int],
        kept: int,
    ) -> None:
        if not _wants_visual_score(config):
            return
        print(
            "[visual-score] join over raw records: "
            + ", ".join(f"{k}={v}" for k, v in sorted(join_counts.items())),
            flush=True,
        )
        print(
            f"[visual-score] channel over the {kept} kept prompts: "
            + ", ".join(f"{k}={v}" for k, v in sorted(visual_counts.items())),
            flush=True,
        )
        image_rows = sum(v for k, v in visual_counts.items() if k != "text_only")
        bad = visual_counts.get("misaligned", 0)
        if bad:
            share = bad / max(image_rows, 1)
            level = "ERROR" if share > 0.01 else "WARNING"
            print(
                f"[visual-score] {level}: {bad} image rows ({share:.2%}) have a sidecar "
                "entry whose token count does not match this run's tokenisation; "
                "they fall back to g=0. Re-score with the same processor/template/"
                "max_length as training if this is more than a handful.",
                flush=True,
            )
        if getattr(config.data, "visual_score_path", "") and visual_counts.get("missing", 0):
            share = visual_counts["missing"] / max(image_rows, 1)
            # A sidecar is configured, so rows without an entry are rows the
            # scoring pass has not reached: a handful is normal (a row it
            # skipped), most of them means it was interrupted and this run
            # would train almost the whole corpus at g=0 while looking fine.
            level = "ERROR" if share > 0.2 else "note"
            print(
                f"[visual-score] {level}: {visual_counts['missing']} image rows "
                f"({share:.2%}) have no sidecar entry and fall back to g=0"
                + (
                    ". The scoring pass looks incomplete -- finish it "
                    "(re-run scripts/score_visual_kl_hpc.sh, it resumes) or set "
                    "data.visual_score_path: \"\" to train the no-visual-term ablation "
                    "deliberately."
                    if level == "ERROR"
                    else ""
                ),
                flush=True,
            )

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
