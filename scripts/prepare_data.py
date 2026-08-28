"""Convert supported text training datasets to SpecForge conversation JSONL.

All presets produce rows with a stable id and a conversations list. Heavy
dataset dependencies stay behind loader functions so row conversion helpers and
their tests remain usable in lightweight environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SUPPORTED_DATASETS = (
    "ultrachat",
    "sharegpt",
    "eaglechat",
    "perfectblend",
    "perfectblend-llama3.1-8b-instruct",
    "perfectblend-llama3.3-70b-instruct",
    "perfectblend-llama4-scout-instruct",
    "perfectblend-llama4-maverick-instruct",
    "magpie-qwen2.5-pro-1m-v0.1",
    "opc",
    "gsm8k",
    "hendrycks_math",
    "math_qa",
    "codealpaca-20k",
    "opencodeinstruct",
    "magicoder-evol-instruct",
    "sciq",
    "camel",
    "nebius-llama31-8b-infinity-instruct",
)
UNSUPPORTED_VLM_DATASETS = frozenset({"sharegpt4v", "allava4v"})
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent.parent / "cache" / "dataset"
SUPPORTED_DATA_PATH_SUFFIXES = {".json", ".jsonl"}
OPC_SUBSETS = (
    "largescale_diverse_instruct",
    "filtered_infinity_instruct",
    "realuser_instruct",
)

SOURCE_COLUMN = "source"

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
        description="Prepare a supported SpecForge training dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset preset to prepare.",
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
        help="Custom ShareGPT JSON or JSONL file instead of the hosted preset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Maximum number of source rows to process.",
    )
    parser.add_argument(
        "--split-eval",
        action="store_true",
        help="Write a deterministic 5%% evaluation split.",
    )
    parser.add_argument(
        "--opc-subset",
        choices=(*OPC_SUBSETS, "all"),
        default=OPC_SUBSETS[0],
        help="OpenCoder opc-sft-stage1 subset, or all supported subsets.",
    )
    parser.add_argument(
        "--source",
        action="append",
        metavar="NAME[=COUNT]",
        help=(
            "Keep only this source of a blended dataset, optionally capped at "
            "COUNT randomly drawn rows (omit COUNT, or use 'all', to keep every "
            "row of it). Repeat the flag to build a mixture, e.g. "
            "--source metamathqa=50000 --source ultrainteract=30000. NAME is "
            "matched case- and punctuation-insensitively against the dataset's "
            "source column, so 'metamathqa' finds 'meta-math/MetaMathQA'. Use "
            "--list-sources to see what a preset actually contains."
        ),
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Print the source column's values and row counts, then exit.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for the per-source random draw (default: 42).",
    )
    parser.add_argument(
        "--output-name",
        help=(
            "Base name for the output files instead of the preset name, so "
            "several source mixtures can live side by side as "
            "<name>_train.jsonl."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Rewrite the output even if it exists. Without it an existing file "
            "is kept, which silently ignores a changed --source mixture."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.data_path is not None:
        if args.dataset != "sharegpt":
            parser.error("--data-path is only supported with --dataset sharegpt")
        if args.data_path.suffix.lower() not in SUPPORTED_DATA_PATH_SUFFIXES:
            parser.error("--data-path must point to a .json or .jsonl file")
    if args.sample_size is not None and args.sample_size <= 0:
        parser.error("--sample-size must be greater than zero")
    if args.source:
        try:
            parse_source_specs(args.source)
        except ValueError as error:
            parser.error(str(error))

    return args


def _load_hf_dataset(*args: Any, **kwargs: Any) -> Any:
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _concatenate_hf_datasets(datasets: Sequence[Any]) -> Any:
    from datasets import concatenate_datasets

    return concatenate_datasets(list(datasets))


def _stable_id(*parts: str) -> str:
    return hashlib.md5("".join(parts).encode()).hexdigest()


def _conversation_row(
    row_id: Any,
    user_content: str,
    assistant_content: str,
) -> dict[str, Any]:
    return {
        "id": str(row_id),
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def process_ultrachat_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    conversations = []
    for message in row["messages"]:
        role = message["role"]
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported UltraChat role: {role!r}")
        conversations.append({"role": role, "content": message["content"]})

    return {
        "id": str(row["prompt_id"]),
        "conversations": conversations,
    }, 0


def process_sharegpt_row(
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

    return {
        "id": str(row["id"]),
        "conversations": conversations,
    }, skipped_count


def process_perfectblend_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    """Convert a blend row, keeping the source column the plain path drops.

    scripts/regenerate_train_data.py reads the source back to decide which rows
    are math and therefore need the benchmark's prompt wording; without it a
    prepared blend is indistinguishable from any other ShareGPT file.
    """
    processed, skipped_count = process_sharegpt_row(row, dataset_name)
    source = row.get(SOURCE_COLUMN)
    if processed is not None and source is not None:
        processed[SOURCE_COLUMN] = source
    return processed, skipped_count


def process_nebius_infinity_instruct(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    conversation = row["conversation"][0]
    generated_message = row["generated_message"]
    return (
        _conversation_row(
            row["id"],
            conversation["content"],
            generated_message["content"],
        ),
        0,
    )


def process_opc_sft_stage1(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    instruction = row["instruction"]
    output = row["output"]
    return (
        _conversation_row(
            _stable_id(instruction, output),
            instruction,
            output,
        ),
        0,
    )


def process_codealpaca_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    return process_opc_sft_stage1(row, dataset_name)


def process_opencodeinstruct_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    row_id = row.get("id") or _stable_id(row["input"], row["output"])
    return _conversation_row(row_id, row["input"], row["output"]), 0


def process_magicoder_evol_instruct_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    instruction = row["instruction"]
    response = row["response"]
    return (
        _conversation_row(
            _stable_id(instruction, response),
            instruction,
            response,
        ),
        0,
    )


def process_gsm8k_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    question = row["question"]
    answer = row["answer"]
    return _conversation_row(_stable_id(question, answer), question, answer), 0


def process_hendrycks_math_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    problem = row["problem"]
    solution = row["solution"]
    return _conversation_row(_stable_id(problem, solution), problem, solution), 0


def process_math_qa_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    user_content = f"{row['Problem']}\n{row['options']}"
    rationale = row["Rationale"]
    return (
        _conversation_row(
            _stable_id(user_content, rationale),
            user_content,
            rationale,
        ),
        0,
    )


def process_sciq_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    answers = [
        row["distractor3"],
        row["distractor1"],
        row["distractor2"],
        row["correct_answer"],
    ]
    random.shuffle(answers)
    labels = ("a", "b", "c", "d")
    options = list(zip(labels, answers))
    correct_label = next(
        label for label, answer in options if answer == row["correct_answer"]
    )
    options_text = "\n".join(f"{label}) {answer}" for label, answer in options)
    user_content = f"{row['question']}\n{options_text}"
    assistant_content = (
        f"{row['support']}\nanswer: {correct_label}) {row['correct_answer']}"
    )
    return (
        _conversation_row(
            _stable_id(user_content, assistant_content),
            user_content,
            assistant_content,
        ),
        0,
    )


def process_camel_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    user_content = row["message_1"]
    assistant_content = row["message_2"]
    return (
        _conversation_row(
            _stable_id(user_content, assistant_content),
            user_content,
            assistant_content,
        ),
        0,
    )


def _identity_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    return dict(row), 0


def _normalize_source(name: str) -> str:
    """Fold a source name to letters and digits, so short handles still match.

    ``meta-math/MetaMathQA`` becomes ``metamathmetamathqa``, which contains the
    handle ``metamathqa``; without the fold a user would have to type the exact
    Hugging Face path, punctuation included.
    """
    return "".join(character for character in name.lower() if character.isalnum())


def parse_source_specs(specs: Sequence[str]) -> list[tuple[str, int | None]]:
    """Turn ``["metamathqa=50000", "ultrainteract"]`` into (name, count) pairs.

    A missing count, or the literal ``all``, means "every row of that source".
    Raises ValueError on anything malformed so the CLI can report it.
    """
    parsed: list[tuple[str, int | None]] = []
    for spec in specs:
        name, separator, raw_count = spec.partition("=")
        name = name.strip()
        if not name:
            raise ValueError(f"--source {spec!r} is missing a source name")
        if not separator or raw_count.strip().lower() == "all":
            parsed.append((name, None))
            continue
        try:
            count = int(raw_count)
        except ValueError:
            raise ValueError(
                f"--source {spec!r}: COUNT must be an integer or 'all'"
            ) from None
        if count <= 0:
            raise ValueError(f"--source {spec!r}: COUNT must be greater than zero")
        parsed.append((name, count))
    return parsed


def source_counts(dataset: Any) -> dict[str, int]:
    """Row count per value of the source column, read as a column not row by row."""
    if SOURCE_COLUMN not in getattr(dataset, "column_names", []):
        raise ValueError(
            f"this dataset has no {SOURCE_COLUMN!r} column, so it cannot be "
            "split by source"
        )
    counts: dict[str, int] = {}
    for value in dataset[SOURCE_COLUMN]:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _resolve_source(name: str, available: Iterable[str]) -> str:
    """Map one user-supplied handle onto exactly one real source value."""
    wanted = _normalize_source(name)
    matches = [value for value in available if wanted in _normalize_source(value)]
    if len(matches) == 1:
        return matches[0]
    listing = ", ".join(sorted(available))
    if not matches:
        raise ValueError(f"--source {name!r} matches none of: {listing}")
    raise ValueError(
        f"--source {name!r} is ambiguous, it matches {sorted(matches)}; "
        f"use a longer handle. Available: {listing}"
    )


def select_sources(
    dataset: Any,
    specs: Sequence[tuple[str, int | None]],
    *,
    seed: int = 42,
) -> Any:
    """Keep the requested sources, each capped at its own random sample.

    Row indices are gathered from the source column and sampled per source, so
    a blend of a few tens of thousands of rows never materializes the whole
    dataset. The kept indices are returned in dataset order, which keeps the
    output reproducible for a given seed.
    """
    counts = source_counts(dataset)
    column = dataset[SOURCE_COLUMN]
    rng = random.Random(seed)

    indices: list[int] = []
    for name, limit in specs:
        resolved = _resolve_source(name, counts)
        available = [i for i, value in enumerate(column) if value == resolved]
        if limit is None or limit >= len(available):
            if limit is not None and limit > len(available):
                print(
                    f"  {resolved}: asked for {limit}, only {len(available)} "
                    "rows exist; keeping all of them"
                )
            chosen = available
        else:
            chosen = rng.sample(available, limit)
        print(f"  {resolved}: {len(chosen)} of {len(available)} rows")
        indices.extend(chosen)

    indices.sort()
    return dataset.select(indices)


def add_index(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    indexed = dict(row)
    indexed["id"] = index
    return indexed


def load_dataset_from_path(data_path: Path) -> Any:
    suffix = data_path.suffix.lower()
    if suffix not in SUPPORTED_DATA_PATH_SUFFIXES:
        raise ValueError(
            f"Unsupported ShareGPT data file {data_path}; expected .json or .jsonl"
        )
    return _load_hf_dataset("json", data_files=str(data_path), split="train")


def _train_split(*args: Any, **kwargs: Any) -> Any:
    return _load_hf_dataset(*args, **kwargs)["train"]


def _indexed(dataset: Any) -> Any:
    return dataset.map(add_index, with_indices=True)


def load_dataset_preset(
    dataset_name: str,
    *,
    data_path: Path | None = None,
    opc_subset: str = OPC_SUBSETS[0],
) -> tuple[Any, RowProcessor | None]:
    """Load one named preset and return its row processor."""
    if dataset_name in UNSUPPORTED_VLM_DATASETS:
        raise ValueError(
            f"Dataset preset {dataset_name!r} is not supported; VLM data "
            "preparation and training are not supported"
        )
    if dataset_name == "ultrachat":
        return (
            _load_hf_dataset(
                "HuggingFaceH4/ultrachat_200k",
                split="train_sft",
            ),
            process_ultrachat_row,
        )
    if dataset_name == "sharegpt":
        dataset = (
            load_dataset_from_path(data_path)
            if data_path is not None
            else _load_hf_dataset(
                "Aeala/ShareGPT_Vicuna_unfiltered",
                split="train",
            )
        )
        return dataset, process_sharegpt_row
    if dataset_name == "eaglechat":
        return _train_split("zhaode/EagleChat"), _identity_row
    if dataset_name == "perfectblend":
        return (
            _indexed(_train_split("mlabonne/open-perfectblend")),
            process_perfectblend_row,
        )

    regenerated_presets = {
        "perfectblend-llama3.1-8b-instruct": (
            "frankleeeee/PerfectBlend-Regenerated-Llama-3.1-8B-Instruct"
        ),
        "perfectblend-llama3.3-70b-instruct": (
            "frankleeeee/PerfectBlend-Regenerated-Llama-3.3-70B-Instruct"
        ),
        "perfectblend-llama4-scout-instruct": (
            "frankleeeee/PerfectBlend-Regenerated-Llama-4-Scout-17B-16E-Instruct"
        ),
        "perfectblend-llama4-maverick-instruct": (
            "frankleeeee/PerfectBlend-Regenerated-Llama-4-Maverick-17B-128E-Instruct"
        ),
    }
    if dataset_name in regenerated_presets:
        return _indexed(_train_split(regenerated_presets[dataset_name])), _identity_row

    if dataset_name == "magpie-qwen2.5-pro-1m-v0.1":
        dataset = _train_split("Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1")
        return dataset.rename_column("uuid", "id"), process_sharegpt_row
    if dataset_name == "nebius-llama31-8b-infinity-instruct":
        dataset = _load_hf_dataset(
            "nebius/Llama-3.1-8B-Instruct-Infinity-Instruct-0625",
            split="train",
        )
        return _indexed(dataset), process_nebius_infinity_instruct
    if dataset_name == "opc":
        if opc_subset == "all":
            dataset = _concatenate_hf_datasets(
                [
                    _train_split("OpenCoder-LLM/opc-sft-stage1", subset)
                    for subset in OPC_SUBSETS
                ]
            )
        else:
            dataset = _train_split("OpenCoder-LLM/opc-sft-stage1", opc_subset)
        return dataset, process_opc_sft_stage1
    if dataset_name == "gsm8k":
        return _train_split("openai/gsm8k", "main"), process_gsm8k_row
    if dataset_name == "hendrycks_math":
        subjects = (
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        )
        return (
            _concatenate_hf_datasets(
                [
                    _train_split("EleutherAI/hendrycks_math", subject)
                    for subject in subjects
                ]
            ),
            process_hendrycks_math_row,
        )
    if dataset_name == "math_qa":
        return (
            _train_split("allenai/math_qa", trust_remote_code=True),
            process_math_qa_row,
        )
    if dataset_name == "codealpaca-20k":
        return (
            _train_split("sahil2801/CodeAlpaca-20k", trust_remote_code=True),
            process_codealpaca_row,
        )
    if dataset_name == "opencodeinstruct":
        return (
            _train_split("nvidia/OpenCodeInstruct", trust_remote_code=True),
            process_opencodeinstruct_row,
        )
    if dataset_name == "magicoder-evol-instruct":
        return (
            _train_split(
                "ise-uiuc/Magicoder-Evol-Instruct-110K",
                trust_remote_code=True,
            ),
            process_magicoder_evol_instruct_row,
        )
    if dataset_name == "sciq":
        return (
            _train_split("allenai/sciq", trust_remote_code=True),
            process_sciq_row,
        )
    if dataset_name == "camel":
        return (
            _concatenate_hf_datasets(
                [
                    _load_hf_dataset(f"camel-ai/{subject}", split="train")
                    for subject in ("biology", "chemistry", "physics")
                ]
            ),
            process_camel_row,
        )
    raise ValueError(
        f"Unsupported dataset preset {dataset_name!r}; choose from {SUPPORTED_DATASETS}"
    )


def load_canonical_dataset(dataset_name: str, data_path: Path | None = None) -> Any:
    """Compatibility wrapper returning only the dataset for a named preset."""
    dataset, _ = load_dataset_preset(dataset_name, data_path=data_path)
    return dataset


def _write_split(
    dataset: Iterable[Mapping[str, Any]],
    output_path: Path,
    processor: RowProcessor | None,
    dataset_name: str,
) -> int:
    skipped_messages = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for item in dataset:
            if processor is None:
                row, skipped_count = dict(item), 0
            else:
                row, skipped_count = processor(item, dataset_name)
            if row is None:
                continue
            skipped_messages += skipped_count
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return skipped_messages


def process_and_save_dataset(
    dataset: Iterable[Mapping[str, Any]],
    output_directory: Path,
    processor: RowProcessor | None,
    dataset_name: str,
    *,
    eval_dataset: Iterable[Mapping[str, Any]] | None = None,
    overwrite: bool = False,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    train_output_path = output_directory / f"{dataset_name}_train.jsonl"
    if train_output_path.exists() and not overwrite:
        print(
            f"Dataset already exists at {train_output_path}; skipping conversion "
            "(pass --overwrite to rebuild it, e.g. after changing --source)."
        )
        return train_output_path

    skipped_messages = _write_split(
        dataset,
        train_output_path,
        processor,
        dataset_name,
    )
    if eval_dataset is not None:
        eval_output_path = output_directory / f"{dataset_name}_test.jsonl"
        skipped_messages += _write_split(
            eval_dataset,
            eval_output_path,
            processor,
            dataset_name,
        )

    if skipped_messages:
        print(
            f"Skipped {skipped_messages} unsupported messages while "
            f"processing {dataset_name}."
        )
    print(f"Saved {dataset_name} training data to {train_output_path}.")
    return train_output_path


def process_and_save_ds(
    train_ds: Iterable[Mapping[str, Any]],
    test_ds: Iterable[Mapping[str, Any]] | None,
    output_path: Path,
    proc_fn: RowProcessor | None,
    dataset_name: str,
) -> Path:
    """Backward-compatible wrapper for external data preparation callers."""
    return process_and_save_dataset(
        train_ds,
        output_path,
        proc_fn,
        dataset_name,
        eval_dataset=test_ds,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset, processor = load_dataset_preset(
        args.dataset,
        data_path=args.data_path,
        opc_subset=args.opc_subset,
    )

    if args.list_sources:
        counts = source_counts(dataset)
        total = sum(counts.values())
        print(f"{args.dataset} sources ({total} rows):")
        for name, count in sorted(counts.items(), key=lambda item: -item[1]):
            print(f"  {count:>9}  {100 * count / total:5.1f}%  {name}")
        return

    if args.source:
        print(f"Selecting sources from {args.dataset} (seed {args.sample_seed}):")
        dataset = select_sources(
            dataset,
            parse_source_specs(args.source),
            seed=args.sample_seed,
        )
        print(f"Kept {len(dataset)} rows in total.")

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
        args.output_name or args.dataset,
        eval_dataset=eval_dataset,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
