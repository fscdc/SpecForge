"""Convert the ShareGPT4V multimodal dataset to SpecForge conversation JSONL.

Rows carry a stable id, a resolved local image path, and a conversations list
containing an `<image>` placeholder. Heavy dataset dependencies stay behind
loader functions so row conversion helpers and their tests remain usable in
lightweight environments.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SUPPORTED_DATASETS = ("sharegpt4v", "sharegpt4v-pt")
DEFAULT_OUTPUT_DIRECTORY = "/local_home3/fengsicheng/specforge/data/"
SUPPORTED_DATA_PATH_SUFFIXES = {".json", ".jsonl"}
IMAGE_PLACEHOLDER = "<image>"
# Lin-Chen/ShareGPT4V hosts both subsets in one repo, distinguished by the
# `datasets.load_dataset` config name (its second positional argument).
SHAREGPT4V_HF_CONFIGS = {
    "sharegpt4v": "ShareGPT4V",
    "sharegpt4v-pt": "ShareGPT4V-PT",
}

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}

ProcessedRow = tuple[dict[str, Any] | None, int]
RowProcessor = Callable[[Mapping[str, Any], str | None], ProcessedRow]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the ShareGPT4V multimodal training dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset preset to prepare.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help=(
            "Directory containing the images referenced by each row's "
            "relative `image` path (e.g. the directory holding coco/, sam/, ...)."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for <dataset>_train.jsonl (default: cache/dataset).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Custom ShareGPT4V-format JSON or JSONL file instead of the hosted preset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Maximum number of source rows to process.",
    )
    parser.add_argument(
        "--split-eval",
        action="store_true",
        help="Write a deterministic 5% evaluation split.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if (
        args.data_path is not None
        and args.data_path.suffix.lower() not in SUPPORTED_DATA_PATH_SUFFIXES
    ):
        parser.error("--data-path must point to a .json or .jsonl file")
    if args.sample_size is not None and args.sample_size <= 0:
        parser.error("--sample-size must be greater than zero")

    return args


def _load_hf_dataset(*args: Any, **kwargs: Any) -> Any:
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def load_dataset_from_path(data_path: Path) -> Any:
    suffix = data_path.suffix.lower()
    if suffix not in SUPPORTED_DATA_PATH_SUFFIXES:
        raise ValueError(
            f"Unsupported ShareGPT4V data file {data_path}; expected .json or .jsonl"
        )
    return _load_hf_dataset("json", data_files=str(data_path), split="train")


def process_sharegpt4v_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    conversations = []
    skipped_count = 0
    for message in row["conversations"]:
        role = ROLE_MAPPING.get(message["from"])
        if role is None:
            skipped_count += 1
            continue
        conversations.append({"role": role, "content": message["value"]})

    has_image_placeholder = any(
        IMAGE_PLACEHOLDER in message["content"] for message in conversations
    )
    if not has_image_placeholder:
        return None, skipped_count + 1

    return (
        {
            "id": str(row["id"]),
            "image": row["image"],
            "conversations": conversations,
        },
        skipped_count,
    )


def load_dataset_preset(
    dataset_name: str,
    *,
    data_path: Path | None = None,
) -> tuple[Any, RowProcessor]:
    """Load one named preset and return its row processor."""
    if dataset_name in SHAREGPT4V_HF_CONFIGS:
        dataset = (
            load_dataset_from_path(data_path)
            if data_path is not None
            else _load_hf_dataset(
                "Lin-Chen/ShareGPT4V",
                SHAREGPT4V_HF_CONFIGS[dataset_name],
                split="train",
            )
        )
        return dataset, process_sharegpt4v_row
    raise ValueError(
        f"Unsupported dataset preset {dataset_name!r}; choose from {SUPPORTED_DATASETS}"
    )


def resolve_image_path(image_root: Path, relative_path: str) -> Path | None:
    resolved = (image_root / relative_path).resolve()
    return resolved if resolved.is_file() else None


def _write_split(
    dataset: Iterable[Mapping[str, Any]],
    output_path: Path,
    processor: RowProcessor,
    dataset_name: str,
    image_root: Path,
) -> tuple[int, int]:
    skipped_messages = 0
    missing_images = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for item in dataset:
            row, skipped_count = processor(item, dataset_name)
            if row is None:
                skipped_messages += skipped_count
                continue

            resolved_image = resolve_image_path(image_root, row["image"])
            if resolved_image is None:
                missing_images += 1
                continue

            row["image"] = str(resolved_image)
            skipped_messages += skipped_count
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return skipped_messages, missing_images


def process_and_save_dataset(
    dataset: Iterable[Mapping[str, Any]],
    output_directory: Path,
    processor: RowProcessor,
    dataset_name: str,
    image_root: Path,
    *,
    eval_dataset: Iterable[Mapping[str, Any]] | None = None,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    train_output_path = output_directory / f"{dataset_name}_train.jsonl"
    if train_output_path.exists():
        print(f"Dataset already exists at {train_output_path}; skipping conversion.")
        return train_output_path

    skipped_messages, missing_images = _write_split(
        dataset,
        train_output_path,
        processor,
        dataset_name,
        image_root,
    )
    if eval_dataset is not None:
        eval_output_path = output_directory / f"{dataset_name}_test.jsonl"
        eval_skipped, eval_missing = _write_split(
            eval_dataset,
            eval_output_path,
            processor,
            dataset_name,
            image_root,
        )
        skipped_messages += eval_skipped
        missing_images += eval_missing

    if skipped_messages:
        print(
            f"Skipped {skipped_messages} unsupported or image-less messages while "
            f"processing {dataset_name}."
        )
    if missing_images:
        print(f"Skipped {missing_images} rows with images missing under {image_root}.")
    print(f"Saved {dataset_name} training data to {train_output_path}.")
    return train_output_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset, processor = load_dataset_preset(args.dataset, data_path=args.data_path)

    if args.sample_size is not None and args.sample_size < len(dataset):
        dataset = dataset.select(range(args.sample_size))
        print(f"Processing {args.sample_size} samples from {args.dataset}.")

    eval_dataset = None
    if args.split_eval:
        split = dataset.train_test_split(test_size=0.05, seed=42)
        dataset = split["train"]
        eval_dataset = split["test"]

    process_and_save_dataset(
        dataset,
        args.output_path,
        processor,
        args.dataset,
        args.image_root,
        eval_dataset=eval_dataset,
    )


if __name__ == "__main__":
    main()
