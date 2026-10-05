"""Convert multimodal datasets to SpecForge conversation JSONL.

Rows carry a stable id, a resolved local image path, and a conversations list
containing an `<image>` placeholder. Heavy dataset dependencies stay behind
loader functions so row conversion helpers and their tests remain usable in
lightweight environments.

The `llava-video-178k` preset writes video rows: `image` is then a *list* of
frame paths (48 by default) and the user turn opens with one `<image>` per
frame, exactly the frames-then-text layout the video benchmarks send. Its
prompts are built by the benchmarks' own formatters so that a training prompt
is byte-for-byte what `bench_mm.py` asks on LongVideoBench, MVBench and
MovieChat. See `prepare_llava_video` for the sampling scheme.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

LLAVA_OV_PRESET = "llava-onevision-1.5"
LLAVA_VIDEO_PRESET = "llava-video-178k"
SUPPORTED_DATASETS = ("sharegpt4v", "sharegpt4v-pt", LLAVA_OV_PRESET, LLAVA_VIDEO_PRESET)
DEFAULT_OUTPUT_DIRECTORY = "/local_home1/fengsicheng/specforge/data/"
SUPPORTED_DATA_PATH_SUFFIXES = {".json", ".jsonl"}
IMAGE_PLACEHOLDER = "<image>"
#: The benchmark implementations whose prompt builders the video preset reuses.
BENCHMARKS_DIRECTORY = Path(__file__).resolve().parents[1] / "benchmarks"

LLAVA_VIDEO_REPO = "lmms-lab/LLaVA-Video-178K"
#: Pinned for the same reason as LLAVA_OV_REVISION below.
LLAVA_VIDEO_REVISION = "6d8c562dc26d70042a0d9704d1cae58c94b89098"
#: 2-3 minute YouTube videos: 24,685 of them in 98 ~5 GiB archives (~250 each),
#: 9,999 with both multiple-choice and open-ended questions. 48 frames of a 2-3
#: minute video are a few seconds apart, which is the regime of the three video
#: benchmarks; the 1_2_m_youtube_v0_1 subset has the same layout.
LLAVA_VIDEO_DEFAULT_SUBSET = "2_3_m_youtube_v0_1"
#: The three benchmarks are run at 48 frames rescaled to (or natively at) a
#: 1280x720 frame's area, 880 tokens a frame on Qwen3.5, ~42k a prompt.
LLAVA_VIDEO_NUM_FRAMES = 48
LLAVA_VIDEO_FRAME_PIXELS = "1280x720"
#: Records per video and the open-ended share among them: 7 multiple-choice
#: (the LongVideoBench and MVBench format, alternating) + 3 open-ended (the
#: MovieChat format) mirrors two multiple-choice benchmarks against one
#: open-ended one, and one 42k-token prefill serves all ten at regeneration.
LLAVA_VIDEO_PER_VIDEO = 10
LLAVA_VIDEO_OE_SHARE = 0.3
#: Every multiple-choice question in the subset ends with one of these two
#: lines (169,870 of 169,870 checked). They are dropped: the benchmark's own
#: closing line goes on instead.
LLAVA_VIDEO_MC_TAILS = frozenset(
    {
        "Please respond with only the letter of the correct answer.",
        "Please provide your answer by stating the letter followed by the full option.",
    }
)
#: Which benchmark's prompt a video record carries, kept in its `style` field.
VIDEO_PROMPT_STYLES = ("longvideobench", "mvbench", "moviechat")
_MC_OPTION_LINE = re.compile(r"^([A-F])\.\s*(.*)$")
_VIDEO_ARCHIVE = re.compile(r"_videos_(\d+)\.tar\.gz$")
#: Repo file listing, fetched once and reused.
_LLAVA_VIDEO_FILES: list[str] | None = None
# Lin-Chen/ShareGPT4V hosts both subsets in one repo, distinguished by the
# `datasets.load_dataset` config name (its second positional argument).
SHAREGPT4V_HF_CONFIGS = {
    "sharegpt4v": "ShareGPT4V",
    "sharegpt4v-pt": "ShareGPT4V-PT",
}

LLAVA_OV_REPO = "mvp-lab/LLaVA-OneVision-1.5-Instruct-Data"
LLAVA_OV_SIZE_ENDPOINT = (
    "https://datasets-server.huggingface.co/size?dataset=mvp-lab%2F"
    "LLaVA-OneVision-1.5-Instruct-Data"
)

# Every config of the blend, by task family. Sharded configs
# (`svit-part-00-of-10`, ...) share their base name's family, so only the base
# is listed here. Families are what --exclude-family selects on.
LLAVA_OV_FAMILIES: dict[str, str] = {
    name: family
    for family, names in {
        "text-math": "openmathinstruct cn_k12 mathinstruct_262k synthetic_math "
        "olympiads mathqa synthetic_amc aops_forum gsm8k math amc_aime",
        "text-general": "orca_agentinstruct magpie_pro open_orca magpie_ultra "
        "orca_994k Evol-Instruct-GPT4-Turbo wizardlm",
        "text-instruction": "ifeval_like code_feedback_66k",
        "caption": "wikipedia_2m llava_instruct sharegpt4v svit allava cambrian "
        "coco sharegpt4o image_textualization laion_220k vision_flan vflan "
        "textcaps llava_wild gpt4o gpt4v llrv_gpt4v visual_chat vision_oritented "
        "alfredplpl textocr_gpt4v allava_instruct_laion4v allava_instruct_vflan4v "
        "wit viquae sherlock vg",
        "chart": "tinychart_train ureader_chart dvqa FigureQA chart2text mapqa "
        "plotqa chartqa vistext oroikon_chart_captioning lrv_chart",
        "table": "robut_wikisql robut_wtq robut_sqa tabmwp hitab tat_qa finqa",
        "document": "ocrvqa Docmatix ocr ureader_qa ureader_tr ureader_cap "
        "ureader_kg ureader_ie ureader_ocr allenai_pixmo_docs rootsautomation "
        "latex_ocr hme100k docvqa_train infographic_vqa infographic_azuregpt4v "
        "sroie_data invoices-and-receipts_ocr visualmrc OmniDocBench_train "
        "st_vqa textvqa llavar iam iiit rects chrome_writting rendered_text",
        "geometry": "geo170k_qa geo170k_align mavis_math_metagen GeoQA+ "
        "Geometry3K unigeo geomverse geo3k intergps GEOS",
        "synthetic": "CLEVR Super-CLEVR CLEVR-Math raven IconQA tallyqa",
        "science": "tqa arxivqa arxiv_figs ai2d scienceqa diagram datikz websight",
        "vqa": "gqa aokvqa visual7w vqaas VizWiz vsr oodvqa hateful_memes idk",
        "cot": "llava_cot_100k",
        "gui": "screen_qa screen2words alfworld VisualWebInstruct",
        "medical": "PMC-VQA pathvqa vqarad",
    }.items()
    for name in names.split()
}
LLAVA_OV_DEFAULT_EXCLUDED = ("gui", "medical")
#: Pinned so the same command on another machine reads the same bytes. `main`
#: is resolved at call time, so leaving it unpinned would silently re-sample if
#: the blend were ever re-uploaded.
LLAVA_OV_REVISION = "0efb7ad47f8ba36e1d262545cadec4cd8960dc5f"
#: Manifests live beside the script rather than with the data: they are the
#: record of how a mixture was built, small enough to keep in the repo, and
#: still useful once the dataset itself has been moved or deleted.
MANIFEST_DIRECTORY = Path(__file__).resolve().parent / "data_reproduce"
#: Streaming reads whole parquet row groups, and this repo writes them at
#: roughly this size, so it is the floor on what touching one config costs.
LLAVA_OV_ROW_GROUP_BYTES = 100 * 1024**2
#: Typical size of one parquet shard in this repo, the unit --fetch shards
#: transfers.
LLAVA_OV_SHARD_BYTES = 500 * 1024**2
#: Repo file listing, fetched once and reused across configs.
_LLAVA_OV_FILES: list[str] | None = None

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}

#: The blend mixes two turn schemas: ShareGPT's ``from``/``value`` with
#: human/gpt, and OpenAI's ``role``/``content`` with user/assistant. Some
#: configs carry all four keys and leave the unused pair null, so a turn has to
#: be read by whichever pair is actually populated rather than by config.
LLAVA_OV_ROLES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}

ProcessedRow = tuple[dict[str, Any] | None, int]


def read_turn(message: Mapping[str, Any]) -> tuple[str, str] | None:
    """Role and text of one turn, whichever of the two schemas it uses."""
    for role_key, text_key in (("from", "value"), ("role", "content")):
        role = LLAVA_OV_ROLES.get(message.get(role_key))
        text = message.get(text_key)
        if role is not None and isinstance(text, str):
            return role, text
    return None
RowProcessor = Callable[[Mapping[str, Any], str | None], ProcessedRow]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a multimodal training dataset for SpecForge."
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
        help=(
            "Root the records' relative `image` paths resolve against, the "
            "same directory the training config's `image_root` points at. The "
            "sharegpt4v presets need it to already hold the images (coco/, "
            f"sam/, ...). For {LLAVA_OV_PRESET} it is where the extracted "
            "images go, under <image-root>/<output-name>/; the records always "
            "carry absolute paths, so the training config needs no image_root."
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
        help="Write a deterministic 5%% evaluation split.",
    )
    parser.add_argument(
        "--output-name",
        help=(
            "Base name for the output files instead of the preset name, so "
            "several mixtures can live side by side as <name>_train.jsonl."
        ),
    )

    onevision = parser.add_argument_group(
        f"{LLAVA_OV_PRESET} (streamed sampling)"
    )
    onevision.add_argument(
        "--image-dir",
        type=Path,
        help=(
            "Where the embedded images are written out to "
            "(default: <output-path>/<name>_images). One subdirectory per config."
        ),
    )
    onevision.add_argument(
        "--exclude-family",
        action="append",
        default=None,
        metavar="FAMILY",
        help=(
            "Drop a task family before sampling; repeatable. Defaults to "
            f"{' and '.join(LLAVA_OV_DEFAULT_EXCLUDED)}. Pass --exclude-family "
            "none to keep everything."
        ),
    )
    onevision.add_argument(
        "--fetch",
        choices=("stream", "shards"),
        default="stream",
        help=(
            "How the parquet is read. 'stream' pulls byte ranges, so it "
            "transfers the least and suits small samples, but shows no "
            "progress and restarts from nothing when interrupted. 'shards' "
            "downloads whole files: about a tenth more transfer, with a "
            "progress bar, resumable downloads, and per-config resume -- use "
            "it for runs long enough to hit a walltime (default: %(default)s)."
        ),
    )
    onevision.add_argument(
        "--keep-parquet",
        action="store_true",
        help=(
            "With --fetch shards, never delete a downloaded shard. By "
            "default a config's shards are dropped once it finishes, so disk "
            "holds at most one config's worth (~10 GiB) and an interrupted "
            "config still retries for free. Keeping them all costs the full "
            "size of every shard touched -- worth it only to grow the sample "
            "later without re-downloading the overlap."
        ),
    )
    onevision.add_argument(
        "--revision",
        default=None,
        help=(
            "Dataset commit to read, pinned so the same command reproduces the "
            f"same sample on another machine (default: {LLAVA_OV_REVISION[:12]} "
            f"for {LLAVA_OV_PRESET}, {LLAVA_VIDEO_REVISION[:12]} for "
            f"{LLAVA_VIDEO_PRESET})."
        ),
    )
    onevision.add_argument(
        "--manifest",
        help=(
            "Replay a previous run's <name>_manifest.json: its per-config "
            "quota, revision, seed and shuffle buffer are used verbatim "
            "instead of being recomputed, which is what makes a re-run on "
            "another machine byte-for-byte identical."
        ),
    )
    onevision.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the per-config stream shuffle (default: 42).",
    )
    onevision.add_argument(
        "--min-samples-per-config",
        type=int,
        default=64,
        help=(
            "Smallest draw worth paying a config's ~100 MB parquet row group "
            "for. Configs are picked at random, weighted by size, until the "
            "sample can give each one this many records, so a small "
            "--sample-size costs bandwidth proportional to itself instead of "
            "to the whole 172-config blend. Raise it for less transfer, lower "
            "it for a wider mix; 0 keeps every config (default: 64)."
        ),
    )
    onevision.add_argument(
        "--shuffle-buffer",
        type=int,
        default=1000,
        help=(
            "Rows each config buffers before yielding. The shard a config is "
            "read from is randomised regardless; this only reorders rows "
            "within it, and every buffered row is downloaded. Pass 1 to keep "
            "shard-level randomness at no extra transfer (default: 1000)."
        ),
    )

    video = parser.add_argument_group(
        f"{LLAVA_VIDEO_PRESET} (videos as frames, benchmark-format prompts)",
        description=(
            "Downloads whole ~5 GiB video archives of one subset in order until "
            "--sample-size records are written, decodes each used video into "
            "--num-frames JPEG frames under the image directory, and gives each "
            "record the prompt of one of the three video benchmarks: multiple-"
            "choice questions alternate between LongVideoBench's and MVBench's "
            "layout, open-ended ones take MovieChat's. --image-root/--image-dir, "
            "--seed, --manifest and --revision apply as above."
        ),
    )
    video.add_argument(
        "--subset",
        default=LLAVA_VIDEO_DEFAULT_SUBSET,
        help=(
            "Folder of the repo to draw from; it must hold both a *_mc_* and a "
            "*_oe_* qa_processed.json (default: %(default)s)."
        ),
    )
    video.add_argument(
        "--per-video",
        type=int,
        default=LLAVA_VIDEO_PER_VIDEO,
        metavar="N",
        help=(
            "Records taken from each video; all of them share its frames and, "
            "sent together, its prefill (default: %(default)s)."
        ),
    )
    video.add_argument(
        "--oe-share",
        type=float,
        default=LLAVA_VIDEO_OE_SHARE,
        help=(
            "Share of a video's records that are open-ended (MovieChat-style); "
            "the rest are multiple-choice (default: %(default)s, i.e. 3 of 10)."
        ),
    )
    video.add_argument(
        "--num-frames",
        type=int,
        default=LLAVA_VIDEO_NUM_FRAMES,
        help="Evenly spaced frames per video (default: %(default)s).",
    )
    video.add_argument(
        "--frame-pixels",
        default=LLAVA_VIDEO_FRAME_PIXELS,
        help=(
            "Area every frame is rescaled to, as WxH or a pixel count; 0 keeps "
            "the native size. The benchmarks' frames are 1280x720 or rescaled "
            "to that area, 880 tokens each (default: %(default)s)."
        ),
    )
    video.add_argument(
        "--archives",
        default=None,
        help=(
            "Comma-separated archive numbers (the N of *_videos_N.tar.gz) to "
            "take videos from, in that order; by default archives are consumed "
            "from 1 upwards until the sample is full."
        ),
    )
    video.add_argument(
        "--video-dir",
        type=Path,
        help=(
            "Where the extracted videos are kept (default: next to the frames, "
            "<image directory>_videos). Only the videos actually used are "
            "extracted; the archives themselves are dropped after use."
        ),
    )
    video.add_argument(
        "--keep-archives",
        action="store_true",
        help="Leave the downloaded archives in the Hugging Face cache.",
    )

    parser.add_argument(
        "--dump-missing-prefix",
        action="append",
        default=[],
        metavar="PREFIX",
        help=(
            "For missing images whose directory prefix matches, print each "
            "full relative image path (e.g. web-celebrity/images). Repeatable."
        ),
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

    sampled = args.dataset in (LLAVA_OV_PRESET, LLAVA_VIDEO_PRESET)
    if sampled:
        if args.sample_size is None:
            parser.error(
                f"--sample-size is required for {args.dataset}: the source is "
                "terabytes, so how much to take has to be stated"
            )
        if args.data_path is not None:
            parser.error(f"--data-path is not supported for {args.dataset}")
        if args.split_eval:
            parser.error(
                f"--split-eval is not supported for {args.dataset}; sample "
                "a separate evaluation set with a different --seed instead"
            )
        if args.image_root is not None and args.image_dir is not None:
            parser.error(
                "--image-root and --image-dir both say where images go; "
                "--image-root adds an <output-name> subdirectory under it, "
                "--image-dir is used as given"
            )
    if args.dataset == LLAVA_OV_PRESET:
        if args.revision is None:
            args.revision = LLAVA_OV_REVISION
        if args.exclude_family is None:
            args.exclude_family = list(LLAVA_OV_DEFAULT_EXCLUDED)
        elif args.exclude_family == ["none"]:
            args.exclude_family = []
        if args.shuffle_buffer < 1:
            parser.error("--shuffle-buffer must be at least 1")
    elif args.dataset == LLAVA_VIDEO_PRESET:
        if args.revision is None:
            args.revision = LLAVA_VIDEO_REVISION
        if args.per_video < 1:
            parser.error("--per-video must be at least 1")
        if not 0.0 <= args.oe_share <= 1.0:
            parser.error("--oe-share must be between 0 and 1")
        if args.num_frames < 1:
            parser.error("--num-frames must be at least 1")
        if not re.fullmatch(r"\d+(x\d+)?", str(args.frame_pixels).lower()):
            parser.error("--frame-pixels must be WxH or a pixel count (0 = native)")
        if args.archives is not None:
            try:
                args.archives = [int(part) for part in str(args.archives).split(",") if part.strip()]
            except ValueError:
                parser.error("--archives must be comma-separated archive numbers")
            if not args.archives or min(args.archives) < 1:
                parser.error("--archives must name archive numbers from 1 upwards")
    elif args.image_root is None:
        parser.error(f"--image-root is required for --dataset {args.dataset}")

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


def _image_prefix(relative_path: str) -> str:
    return relative_path.rsplit("/", 1)[0] if "/" in relative_path else "(no directory)"


def _write_split(
    dataset: Iterable[Mapping[str, Any]],
    output_path: Path,
    processor: RowProcessor,
    dataset_name: str,
    image_root: Path,
    dump_missing_prefixes: Sequence[str] = (),
) -> tuple[int, int, Counter[str], list[str]]:
    skipped_messages = 0
    missing_images = 0
    missing_prefixes: Counter[str] = Counter()
    dumped_missing_paths: list[str] = []
    with output_path.open("w", encoding="utf-8") as output_file:
        for item in dataset:
            row, skipped_count = processor(item, dataset_name)
            if row is None:
                skipped_messages += skipped_count
                continue

            resolved_image = resolve_image_path(image_root, row["image"])
            if resolved_image is None:
                missing_images += 1
                prefix = _image_prefix(row["image"])
                missing_prefixes[prefix] += 1
                if prefix in dump_missing_prefixes:
                    dumped_missing_paths.append(row["image"])
                continue

            row["image"] = str(resolved_image)
            skipped_messages += skipped_count
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return skipped_messages, missing_images, missing_prefixes, dumped_missing_paths


# ---------------------------------------------------------------------------
# LLaVA-OneVision 1.5: streamed, quota-sampled, images embedded in the parquet
# ---------------------------------------------------------------------------


def _base_config(config: str) -> str:
    """`svit-part-03-of-10` and `svit` are the same subset for family purposes."""
    import re

    return re.sub(r"-part-\d+-of-\d+$", "", config)


def llava_ov_config_sizes() -> dict[str, int]:
    """Row count of every config, from the Hugging Face datasets-server.

    Fetched rather than hard-coded so the split stays correct if the blend gains
    a subset, and needed at all because the sample has to be spread over the
    configs in proportion to their sizes.
    """
    import urllib.request

    with urllib.request.urlopen(LLAVA_OV_SIZE_ENDPOINT, timeout=120) as response:  # noqa: E501
        payload = json.loads(response.read())
    sizes = {
        entry["config"]: int(entry["num_rows"])
        for entry in payload["size"]["configs"]
        if entry["config"] != "default"
    }
    if not sizes:
        raise RuntimeError(f"{LLAVA_OV_SIZE_ENDPOINT} returned no configs")
    return sizes


def llava_ov_eligible_configs(
    excluded_families: Sequence[str],
) -> list[tuple[str, str, int]]:
    """(config, family, rows) for every config left after the exclusions."""
    sizes = llava_ov_config_sizes()
    unknown = sorted(
        {_base_config(name) for name in sizes} - set(LLAVA_OV_FAMILIES)
    )
    if unknown:
        raise RuntimeError(
            f"{LLAVA_OV_REPO} has configs this script cannot classify: {unknown}. "
            "Add them to LLAVA_OV_FAMILIES before sampling, so nothing is "
            "silently left out of the blend."
        )
    excluded = set(excluded_families)
    unknown_families = excluded - set(LLAVA_OV_FAMILIES.values())
    if unknown_families:
        raise ValueError(
            f"unknown family/families {sorted(unknown_families)}; choose from "
            f"{sorted(set(LLAVA_OV_FAMILIES.values()))}"
        )
    eligible = [
        (name, LLAVA_OV_FAMILIES[_base_config(name)], rows)
        for name, rows in sorted(sizes.items())
        if LLAVA_OV_FAMILIES[_base_config(name)] not in excluded and rows > 0
    ]
    if not eligible:
        raise ValueError("every family was excluded; nothing left to sample")
    return eligible


def select_configs(
    configs: Sequence[tuple[str, str, int]],
    sample_size: int,
    *,
    min_rows: int,
    seed: int,
) -> list[tuple[str, str, int]]:
    """Narrow the config set so every config touched pays back its row group.

    Streaming's floor is one parquet row group per config read, ~100 MB here,
    which is charged whether the config yields one row or five hundred. Spread
    a small sample proportionally over all 172 configs and the download is
    dominated by that floor: 1,000 rows would cost ~13 GB to write ~190 MB.

    Drawing the configs themselves at random, weighted by row count, keeps the
    sample unbiased in expectation -- a row's chance of being picked is still
    proportional to nothing but the corpus -- and trades per-draw family
    diversity, which a small sample cannot afford anyway, for bandwidth. Large
    samples reach every config and this becomes a no-op.
    """
    if min_rows <= 0 or sample_size >= min_rows * len(configs):
        return list(configs)

    wanted = max(1, math.ceil(sample_size / min_rows))
    rng = random.Random(seed)
    pool = list(configs)
    weights = [rows for _name, _family, rows in pool]
    chosen: list[tuple[str, str, int]] = []
    held = 0
    # Draw the target count, then keep drawing while the picks cannot cover the
    # sample: a run of small configs would otherwise silently write short.
    while pool and (len(chosen) < wanted or held < sample_size):
        index = rng.choices(range(len(pool)), weights=weights, k=1)[0]
        chosen.append(pool.pop(index))
        held += weights.pop(index)
    return sorted(chosen)


def allocate_quota(
    configs: Sequence[tuple[str, str, int]], sample_size: int
) -> dict[str, int]:
    """Split `sample_size` over the configs in proportion to their row counts.

    Largest-remainder, so the quotas sum to exactly `sample_size` and the blend
    keeps the corpus's own mix rather than over-representing small subsets. A
    config never gets more than it holds; what it cannot take is redistributed.
    """
    total = sum(rows for _, _, rows in configs)
    quota: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for name, _family, rows in configs:
        exact = sample_size * rows / total
        quota[name] = min(int(exact), rows)
        remainders.append((exact - int(exact), name))

    # hand out the rounding slack, largest fractional part first
    sizes = {name: rows for name, _family, rows in configs}
    for _fraction, name in sorted(remainders, reverse=True):
        if sum(quota.values()) >= sample_size:
            break
        if quota[name] < sizes[name]:
            quota[name] += 1
    # a config capped at its own size leaves slack; spread it over the rest
    while sum(quota.values()) < sample_size:
        headroom = [n for n in quota if quota[n] < sizes[n]]
        if not headroom:
            break
        for name in sorted(headroom, key=lambda n: -sizes[n]):
            if sum(quota.values()) >= sample_size:
                break
            quota[name] += 1
    return {name: count for name, count in quota.items() if count > 0}


_IMAGE_MAGIC = (
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"GIF8", ".gif"),
    (b"RIFF", ".webp"),
    (b"BM", ".bmp"),
)


def _image_suffix(data: bytes, fallback: str | None) -> str:
    """Extension for embedded image bytes, sniffed rather than trusted.

    The parquet's own `path` is often absent or meaningless, and the bytes are
    written straight through without a decode/re-encode round trip, so the
    suffix has to come from the magic number to stay honest.
    """
    for magic, suffix in _IMAGE_MAGIC:
        if data.startswith(magic):
            return suffix
    if fallback:
        suffix = Path(fallback).suffix.lower()
        if suffix in {value for _magic, value in _IMAGE_MAGIC}:
            return suffix
    return ".img"


def write_llava_ov_image(
    row: Mapping[str, Any], config: str, index: int, image_directory: Path
) -> str | None:
    """Write one row's embedded image out, returning its absolute path.

    The images live inside the parquet as bytes, so unlike the ShareGPT4V path
    there is nothing to resolve against an image root. A text-only subset has no
    image at all and its rows keep a null in that field.
    """
    image = row.get("image")
    if not isinstance(image, Mapping) or not image.get("bytes"):
        return None
    data = image["bytes"]
    directory = image_directory / config
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{index:08d}{_image_suffix(data, image.get('path'))}"
    destination.write_bytes(data)
    return str(destination)


#: Below this, two strings are too short for containment to mean anything --
#: a one-word answer often appears verbatim in its own question.
LEAKED_ANSWER_MIN_CHARS = 40
#: How close in length the pair must be before containment counts as a leak.
LEAKED_ANSWER_MIN_RATIO = 0.8


def leaks_answer(question: str, answer: str) -> bool:
    """Whether a pair is one blob of text split badly across both turns.

    Some `visual7w` rows in the blend carry the whole "Question: ... Answer:
    ..." text in *both* turns, with a couple of leading characters lost from
    one of them, so the answer is handed to the model inside its own prompt.
    Training on those teaches copying rather than answering.

    Detected by containment rather than a diff: the two differ only by a
    truncated prefix, so the shorter sits inside the longer, and requiring
    similar lengths keeps a genuinely short answer that happens to echo a word
    of its question from tripping it.
    """
    left = " ".join(question.split())
    right = " ".join(answer.split())
    if min(len(left), len(right)) < LEAKED_ANSWER_MIN_CHARS:
        return False
    shorter, longer = sorted((left, right), key=len)
    if len(shorter) / len(longer) < LEAKED_ANSWER_MIN_RATIO:
        return False
    return shorter in longer


def conversation_pairs(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[tuple[str, str]], int, bool]:
    """Split a row's turns into (system prompts, question/answer pairs, skipped, has image).

    Shared by the LLaVA-OneVision and LLaVA-Video presets, which both pack
    several independent questions about one picture or video into a row.
    Whether the row is a visual one is a property of the row, not of a pair:
    only the first question carries the placeholder, yet every pair split out
    of the row refers to the same picture.
    """
    turns: list[tuple[str, str]] = []
    skipped_count = 0
    for message in messages:
        turn = read_turn(message)
        if turn is None:
            skipped_count += 1
            continue
        turns.append(turn)

    system = [content for role, content in turns if role == "system"]
    pairs: list[tuple[str, str]] = []
    pending: str | None = None
    for role, content in turns:
        if role == "system":
            continue
        if role == "user":
            if pending is not None:
                skipped_count += 1  # a question nobody answered
            pending = content
        elif pending is None:
            skipped_count += 1  # a reply to nothing
        else:
            pairs.append((pending, content))
            pending = None
    if pending is not None:
        skipped_count += 1
    wants_image = any(IMAGE_PLACEHOLDER in content for _role, content in turns)
    return system, pairs, skipped_count, wants_image


def process_llava_ov_row(
    row: Mapping[str, Any], config: str, index: int, image_directory: Path
) -> tuple[list[dict[str, Any]], int]:
    """One blend row into SpecForge records, one per question/answer pair.

    Most image subsets pack several independent questions about the same picture
    into a single row -- visual7w reaches nineteen, dvqa more -- to amortise the
    visual tokens. They are not a conversation: the questions neither refer to
    each other nor depend on the order they are asked in, so each becomes its
    own record, carrying a copy of the leading system prompt when the subset has
    one. Every record from a row points at the same image file on disk.
    """
    system, pairs, skipped_count, wants_image = conversation_pairs(
        row.get("conversations") or []
    )
    if not pairs:
        return [], skipped_count + 1

    image_path = write_llava_ov_image(row, config, index, image_directory)
    if image_path is None and wants_image:
        return [], skipped_count + len(pairs)  # an <image> with nothing behind it

    records: list[dict[str, Any]] = []
    for question, answer in pairs:
        if leaks_answer(question, answer):
            skipped_count += 1
            continue
        if image_path is not None:
            # Normalise the placeholder: strip it wherever the source put it and
            # give every record exactly one, in front of its own question.
            question = question.replace(IMAGE_PLACEHOLDER, "").lstrip("\n")
            question = f"{IMAGE_PLACEHOLDER}\n{question}"
        conversations = [{"role": "system", "content": text} for text in system]
        conversations.append({"role": "user", "content": question})
        conversations.append({"role": "assistant", "content": answer})
        records.append(
            {
                "id": f"{config}#{index}-{len(records)}",
                "image": image_path,
                "conversations": conversations,
            }
        )
    return records, skipped_count


def stream_llava_ov(
    quota: Mapping[str, int],
    image_directory: Path,
    *,
    seed: int,
    shuffle_buffer: int,
    revision: str,
) -> Iterable[dict[str, Any]]:
    """Yield `sum(quota)` records, pulling only as much of each config as needed.

    The quota counts records, not source rows, so a row that unpacks into
    nineteen questions covers nineteen of them; a row is cut short rather than
    overshooting, which is harmless because its questions are independent.

    Streaming is what keeps this tractable: the blend is ~3.6 TiB and only the
    parquet row groups a config's quota actually reaches are ever downloaded.

    The sample is random, not uniform-random: a stream cannot be indexed, so
    `IterableDataset.shuffle` is used, which permutes the shard order and draws
    from a rolling window. A larger window is closer to uniform and downloads
    more before the first row comes out; `shuffle_buffer` is the ceiling, and a
    config taking only a few rows uses a proportionally smaller one.
    """
    from datasets import Image, load_dataset

    # Biggest quota first: the configs that carry the sample start producing
    # immediately, instead of the run opening on a one-row config whose row
    # group costs as much to fetch as a thousand-row one.
    order = sorted(quota.items(), key=lambda item: (-item[1], item[0]))
    for position, (config, count) in enumerate(order, start=1):
        dataset = load_dataset(
            LLAVA_OV_REPO,
            config,
            split="train",
            streaming=True,
            revision=revision,
        )
        # Keep the bytes as they are: decoding to PIL only to re-encode would
        # be slower and lossy. Text-only configs have no image column.
        try:
            dataset = dataset.cast_column("image", Image(decode=False))
        except (ValueError, KeyError):
            pass
        # A config whose quota is a handful of records must not drag the full
        # buffer down with it: at ~190 KB a row, filling 1000 slots to emit one
        # row downloads 190 MB for that one row. Shard-order shuffling is a
        # separate step in `datasets` and applies whatever the buffer is, so a
        # buffer of 1 still lands at a random shard -- it only gives up the
        # reordering *within* a shard, which is what the extra rows pay for.
        buffer = max(min(shuffle_buffer, count * 4), 1)
        dataset = dataset.shuffle(
            seed=seed + zlib.crc32(config.encode()), buffer_size=buffer
        )

        taken = 0
        seen = 0
        for index, row in enumerate(dataset):
            if taken >= count:
                break
            seen += 1
            if seen >= 200 and taken == 0:
                # a config whose rows all fail to convert would otherwise be
                # streamed to the end -- gigabytes spent to emit nothing
                raise ValueError(
                    f"{config!r}: {seen} rows read and none converted; its "
                    "conversation schema is not one this script recognises"
                )
            records, _skipped = process_llava_ov_row(
                row, config, index, image_directory
            )
            for record in records[: count - taken]:
                taken += 1
                yield record
        if taken < count:
            raise ValueError(
                f"{config!r} yielded only {taken} of {count} records"
            )


def llava_ov_shards(config: str, revision: str) -> list[str]:
    """Repo paths of one config's parquet shards, in the repo's own order."""
    from huggingface_hub import HfApi

    global _LLAVA_OV_FILES
    if _LLAVA_OV_FILES is None:
        _LLAVA_OV_FILES = HfApi().list_repo_files(
            LLAVA_OV_REPO, repo_type="dataset", revision=revision
        )
    prefix = f"{config}/"
    shards = sorted(
        name
        for name in _LLAVA_OV_FILES
        if name.startswith(prefix) and name.endswith(".parquet")
    )
    if not shards:
        raise ValueError(f"{config!r} has no parquet shards at {revision[:12]}")
    return shards


def fetch_llava_ov_config(
    config: str,
    count: int,
    image_directory: Path,
    *,
    seed: int,
    revision: str,
    keep_parquet: bool,
) -> Iterable[dict[str, Any]]:
    """Yield `count` records for one config by downloading whole shards.

    Streaming reads byte ranges, which cannot be resumed and shows no progress;
    a shard download is a plain cached file transfer, so it prints a progress
    bar, resumes where it stopped, and a re-run skips what is already on disk.
    It costs about a tenth more transfer because the tail of the last shard is
    downloaded but unused.

    Shards are visited in a seeded random order, which is where the sampling
    randomness now comes from -- rows within a shard are taken in order.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    shards = llava_ov_shards(config, revision)
    # Seeded by name, not by the config's rank in the run: growing the sample
    # nudges some ranks around, and a rank-derived seed would reshuffle those
    # configs' shards so a larger run could not reuse a smaller one's cache.
    random.Random(f"{seed}:{config}").shuffle(shards)

    taken = 0
    index = 0
    fetched = 0
    downloaded = 0
    consumed: list[str] = []
    for shard in shards:
        if taken >= count:
            break
        started = time.monotonic()
        local = hf_hub_download(
            LLAVA_OV_REPO, shard, repo_type="dataset", revision=revision
        )
        # A tqdm bar is useless in a batch log, so report per shard instead:
        # enough to see it moving and to estimate what is left.
        elapsed = max(time.monotonic() - started, 1e-6)
        size = Path(local).stat().st_size
        fetched += 1
        downloaded += size
        print(
            f"      shard {fetched} ({Path(shard).name}): "
            f"{size / 1024**2:,.0f} MB at {size / elapsed / 1024**2:,.1f} MB/s"
            f" | {taken:,}/{count:,} records, {downloaded / 1024**3:,.2f} GiB",
            flush=True,
        )
        consumed.append(local)
        parquet = pq.ParquetFile(local)
        for batch in parquet.iter_batches(batch_size=32):
            if taken >= count:
                break
            for row in batch.to_pylist():
                if taken >= count:
                    break
                records, _skipped = process_llava_ov_row(
                    row, config, index, image_directory
                )
                index += 1
                for record in records[: count - taken]:
                    taken += 1
                    yield record

    if not keep_parquet:
        # Held until the config is done rather than dropped shard by shard: a
        # job killed part way through leaves them cached, so the retry re-reads
        # them locally instead of paying for the same bytes twice. Only one
        # config's shards are ever on disk, not the whole run's.
        for path in consumed:
            Path(path).unlink(missing_ok=True)
    if taken < count:
        raise ValueError(
            f"{config!r} yielded only {taken} of {count} records after all "
            f"{len(shards)} shards; its quota is larger than it can fill"
        )


def summarise_records(
    counts: Mapping[str, int],
    prompt_chars: Mapping[str, list[int]],
    answer_chars: Mapping[str, list[int]],
    family_of: Mapping[str, str],
) -> None:
    """Print what the written dataset actually contains, per family.

    Answer length is the number worth watching: a subset whose answers are two
    characters long fills most of a draft block with padding, which flatters
    every acceptance number measured on it.
    """

    def quantile(values: Sequence[int], fraction: float) -> int:
        return values[min(int(len(values) * fraction), len(values) - 1)]

    by_family: dict[str, list[int]] = defaultdict(list)
    prompts_by_family: dict[str, list[int]] = defaultdict(list)
    for config, lengths in answer_chars.items():
        by_family[family_of[config]].extend(lengths)
        prompts_by_family[family_of[config]].extend(prompt_chars[config])

    total = sum(counts.values())
    print(f"\nAnswer and prompt length by family ({total:,} records):")
    print(
        f"  {'family':<18}{'records':>10}{'share':>8}"
        f"{'prompt p50':>12}{'answer p50':>12}{'p90':>8}{'<=3 chars':>11}"
    )
    for family, lengths in sorted(by_family.items(), key=lambda kv: -len(kv[1])):
        lengths.sort()
        prompts = sorted(prompts_by_family[family])
        tiny = sum(1 for value in lengths if value <= 3) / len(lengths)
        print(
            f"  {family:<18}{len(lengths):>10,}{len(lengths) / total:>8.1%}"
            f"{quantile(prompts, 0.5):>12,}{quantile(lengths, 0.5):>12,}"
            f"{quantile(lengths, 0.9):>8,}{tiny:>11.0%}"
        )

    tiny_configs = sorted(
        (
            (sum(1 for v in lengths if v <= 3) / len(lengths), config, len(lengths))
            for config, lengths in answer_chars.items()
            if len(lengths) >= 500
        ),
        reverse=True,
    )[:8]
    if tiny_configs and tiny_configs[0][0] > 0.05:
        print("\n  Subsets whose answers are mostly a word or two:")
        for share, config, size in tiny_configs:
            if share <= 0.05:
                break
            print(f"    {config:<24}{size:>9,}{share:>8.0%} at <=3 chars")


def _write_llava_ov_dataset(
    args: argparse.Namespace,
    name: str,
    quota: Mapping[str, int],
    output_directory: Path,
    image_directory: Path,
    family_of: Mapping[str, str],
) -> Path:
    """Fetch every config's share, concatenate the parts, and report.

    Split out of `prepare_llava_onevision` so a replay, which must not rewrite
    the manifest it was handed, still runs exactly the same writer.
    """
    train_output_path = output_directory / f"{name}_train.jsonl"
    parts_directory = output_directory / f"{name}_parts"
    parts_directory.mkdir(parents=True, exist_ok=True)

    order = sorted(quota.items(), key=lambda item: (-item[1], item[0]))
    for position, (config, count) in enumerate(order, start=1):
        part = parts_directory / f"{config}.jsonl"
        if part.exists():
            done = sum(1 for _ in part.open(encoding="utf-8"))
            if done == count:
                print(
                    f"  [{position}/{len(order)}] {config}: {count:,} already "
                    "on disk, skipping",
                    flush=True,
                )
                continue
            print(
                f"  [{position}/{len(order)}] {config}: {done:,}/{count:,} on "
                "disk is incomplete, redoing",
                flush=True,
            )
        print(
            f"  [{position}/{len(order)}] {config}: fetching {count:,} "
            f"record{'s' if count != 1 else ''} ...",
            flush=True,
        )
        if args.fetch == "shards":
            records = fetch_llava_ov_config(
                config,
                count,
                image_directory,
                seed=args.seed,
                revision=args.revision,
                keep_parquet=args.keep_parquet,
            )
        else:
            records = stream_llava_ov(
                {config: count},
                image_directory,
                seed=args.seed,
                shuffle_buffer=args.shuffle_buffer,
                revision=args.revision,
            )
        # Written to a temporary name and moved into place, so a job killed
        # mid-config never leaves a part file that looks complete.
        staging = part.with_suffix(".jsonl.partial")
        with staging.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        staging.replace(part)

    written = 0
    with_image = 0
    images: set[str | None] = set()
    counts: Counter[str] = Counter()
    prompt_chars: dict[str, list[int]] = defaultdict(list)
    answer_chars: dict[str, list[int]] = defaultdict(list)
    with train_output_path.open("w", encoding="utf-8") as handle:
        for config, _count in order:
            for line in (parts_directory / f"{config}.jsonl").open(encoding="utf-8"):
                handle.write(line)
                record = json.loads(line)
                written += 1
                with_image += record["image"] is not None
                images.add(record["image"])
                counts[config] += 1
                turns = record["conversations"]
                prompt_chars[config].append(len(turns[-2]["content"]))
                answer_chars[config].append(len(turns[-1]["content"]))
    images.discard(None)

    print(
        f"\nSaved {written:,} records to {train_output_path} "
        f"({with_image:,} with an image over {len(images):,} distinct pictures, "
        f"{written - with_image:,} text-only)."
    )
    print(f"Images written under {image_directory}")
    print(f"Per-config parts kept in {parts_directory} for resuming.")
    summarise_records(counts, prompt_chars, answer_chars, family_of)
    return train_output_path


def prepare_llava_onevision(args: argparse.Namespace) -> Path:
    """Write `--sample-size` rows of the blend to one JSONL, images alongside."""
    configs = llava_ov_eligible_configs(args.exclude_family)
    families = sorted({family for _name, family, _rows in configs})
    pool = sum(rows for _name, _family, rows in configs)
    dropped = sorted(set(args.exclude_family)) or ["nothing"]
    print(
        f"{LLAVA_OV_REPO}: {len(configs)} configs over {len(families)} "
        f"families ({pool:,} rows) after excluding {', '.join(dropped)}"
    )
    if args.sample_size > pool:
        raise ValueError(
            f"--sample-size {args.sample_size} exceeds the {pool:,} rows "
            "available (a row yields one record per question, so this is a "
            "conservative bound)"
        )

    drawn = select_configs(
        configs,
        args.sample_size,
        min_rows=args.min_samples_per_config,
        seed=args.seed,
    )
    quota = allocate_quota(drawn, args.sample_size)
    by_family: Counter[str] = Counter()
    family_of = {name: family for name, family, _rows in configs}
    for name, count in quota.items():
        by_family[family_of[name]] += count
    print(f"Sampling {args.sample_size:,} records from {len(quota)} configs:")
    for family, count in by_family.most_common():
        print(f"  {count:>9,}  {count / args.sample_size:5.1%}  {family}")
    unit = (
        LLAVA_OV_SHARD_BYTES if args.fetch == "shards" else LLAVA_OV_ROW_GROUP_BYTES
    )
    what = "shard" if args.fetch == "shards" else "parquet row group"
    print(
        f"--fetch {args.fetch} reads a whole ~{unit // 1024**2} MB {what} at a "
        f"time, so expect at least {len(quota) * unit / 1024**3:,.1f} GiB of "
        "transfer; raise --min-samples-per-config to touch fewer configs, "
        "lower it for a wider blend."
    )

    output_directory = Path(args.output_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    name = args.output_name or args.dataset
    manifest_path = MANIFEST_DIRECTORY / f"{name}_manifest.json"
    if args.manifest is not None:
        recorded = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        quota = {str(k): int(v) for k, v in recorded["quota"].items()}
        args.revision = recorded["revision"]
        args.seed = recorded["seed"]
        args.shuffle_buffer = recorded["shuffle_buffer"]
        recorded_size = sum(quota.values())
        print(
            f"Replaying {args.manifest}: {recorded_size:,} records over "
            f"{len(quota)} configs at revision {args.revision[:12]}"
        )
        # The replayed record is authoritative, so say so rather than letting a
        # stale flag look as if it took effect.
        if args.sample_size != recorded_size:
            print(
                f"  NOTE: --sample-size {args.sample_size:,} is ignored; the "
                f"manifest's quota decides, and it sums to {recorded_size:,}."
            )
        replayed = sorted(recorded.get("excluded_families", []))
        if replayed != sorted(args.exclude_family):
            print(
                f"  NOTE: --exclude-family {sorted(args.exclude_family)} is "
                f"ignored; the manifest was built excluding {replayed}."
            )
    if args.image_root is not None:
        # One subdirectory per mixture, so several of them can share a root
        # without colliding.
        image_directory = Path(args.image_root) / name
    else:
        image_directory = Path(args.image_dir or output_directory / f"{name}_images")
    image_directory.mkdir(parents=True, exist_ok=True)
    train_output_path = output_directory / f"{name}_train.jsonl"
    if train_output_path.exists():
        print(f"Overwriting existing dataset at {train_output_path}.")

    import datasets as _datasets
    import numpy as _numpy

    if args.manifest is not None:
        # A replay reproduces a record that already exists; rewriting it would
        # stamp this run's --sample-size, exclusions and library versions over
        # the ones the mixture was actually built with -- and when the manifest
        # replayed is this preset's own, it would overwrite the file just read.
        print(f"Reusing {args.manifest}; not rewriting {manifest_path}")
        return _write_llava_ov_dataset(
            args, name, quota, output_directory, image_directory, family_of
        )

    MANIFEST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "repo": LLAVA_OV_REPO,
                "revision": args.revision,
                "sample_size": args.sample_size,
                "seed": args.seed,
                "shuffle_buffer": args.shuffle_buffer,
                "excluded_families": sorted(args.exclude_family),
                # Recorded rather than recomputed on replay: the row counts
                # behind it come from a live endpoint that would move if the
                # blend were ever re-uploaded.
                "quota": dict(sorted(quota.items())),
                "versions": {
                    "datasets": _datasets.__version__,
                    "numpy": _numpy.__version__,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")

    return _write_llava_ov_dataset(
        args, name, quota, output_directory, image_directory, family_of
    )


# ---------------------------------------------------------------------------
# LLaVA-Video-178K: videos as frames, prompts in the video benchmarks' own format
# ---------------------------------------------------------------------------


def parse_mc_question(text: str) -> tuple[str, list[str]] | None:
    """Question and options of one LLaVA-Video multiple-choice prompt.

    The prompts are rigid: `<image>`, one question line, two to five option
    lines `A. ...`, and one of two closing lines asking for the letter. The
    closing line is dropped because the benchmark's own goes on instead.
    Anything that does not fit the layout is refused rather than guessed at,
    so a malformed row is skipped instead of producing a prompt no benchmark
    would ask. More than five options are refused too: the benchmarks letter
    up to E.
    """
    lines = [
        line.strip()
        for line in text.replace(IMAGE_PLACEHOLDER, "").strip().split("\n")
        if line.strip()
    ]
    if len(lines) < 3:
        return None
    question = lines[0]
    options: list[str] = []
    index = 1
    while index < len(lines) and len(options) < 6:
        match = _MC_OPTION_LINE.match(lines[index])
        if match is None or match.group(1) != "ABCDEF"[len(options)]:
            break
        options.append(match.group(2).strip())
        index += 1
    tail = " ".join(lines[index:]).strip()
    if tail and tail not in LLAVA_VIDEO_MC_TAILS:
        return None
    if not 2 <= len(options) <= 5:
        return None
    return question, options


def choose_video_questions(
    video: str, num_mc: int, num_oe: int, per_video: int, oe_share: float, seed: int
) -> tuple[list[int], list[int]]:
    """Which of a video's questions become records, as sorted index lists.

    Seeded by the video's name, so the choice does not depend on how many
    videos came before it; a video short of questions gives what it has.
    """
    rng = random.Random(f"{seed}:{video}")
    want_oe = int(round(per_video * oe_share))
    want_mc = per_video - want_oe
    picked_mc = sorted(rng.sample(range(num_mc), min(want_mc, num_mc)))
    picked_oe = sorted(rng.sample(range(num_oe), min(want_oe, num_oe)))
    return picked_mc, picked_oe


def build_video_record(
    subset: str,
    video: str,
    frames: Sequence[str],
    kind: str,
    position: int,
    style: str,
    text: str,
    answer: str,
) -> dict[str, Any]:
    """One video record: frames first, then the benchmark-format text.

    `image` is the list of frame paths, in order, and the user turn carries one
    placeholder per frame -- the same frames-then-text layout the benchmarks
    send, which the chat template turns into consecutive image blocks before the
    question. The id keeps the `<config>#<row>-<pair>` shape of the other
    presets, so the regeneration script reads it the same way.
    """
    stem = Path(video).stem
    return {
        "id": f"{subset}#{stem}-{kind}{position}",
        "image": list(frames),
        "video": video,
        "style": style,
        "conversations": [
            {
                "role": "user",
                "content": IMAGE_PLACEHOLDER * len(frames) + "\n" + text,
            },
            {"role": "assistant", "content": answer},
        ],
    }


def video_benchmark_tools() -> tuple[dict[str, Callable[..., str]], Callable[..., list[str]], Callable[[Any], int]]:
    """The benchmarks' prompt builders, frame extractor and pixel parser.

    Imported from benchmarks/ rather than copied: a training prompt is then
    byte-for-byte what `bench_mm.py` sends, including the post-prompt text, the
    option lettering, the newlines, and any `*_PRE_PROMPT`/`*_POST_PROMPT`
    environment override in force; and the frames are sampled, rescaled and
    encoded by the same function. The package import pulls in sglang (the
    benchmarkers talk to a server), which is why it is deferred to here.
    """
    if str(BENCHMARKS_DIRECTORY) not in sys.path:
        sys.path.insert(0, str(BENCHMARKS_DIRECTORY))
    from mm_benchmarker import longvideobench as lvb
    from mm_benchmarker import moviechat as mc
    from mm_benchmarker import mvbench as mvb
    from mm_benchmarker.video_utils import materialize_frames

    def longvideobench(question: str, options: Sequence[str]) -> str:
        row = {"question": question}
        row.update({f"option{index}": option for index, option in enumerate(options)})
        return lvb.format_question(
            row,
            os.environ.get("LVB_PRE_PROMPT", lvb.DEFAULT_PRE_PROMPT),
            os.environ.get("LVB_POST_PROMPT", lvb.DEFAULT_POST_PROMPT),
        )

    def mvbench(question: str, options: Sequence[str]) -> str:
        return mvb.format_question(
            {"question": question, "candidates": list(options)},
            os.environ.get("MVBENCH_POST_PROMPT", mvb.DEFAULT_POST_PROMPT),
        )

    def moviechat(question: str, options: Sequence[str] = ()) -> str:
        del options
        return mc.format_question(
            question,
            os.environ.get("MOVIECHAT_PRE_PROMPT", mc.DEFAULT_PRE_PROMPT),
            os.environ.get("MOVIECHAT_POST_PROMPT", mc.DEFAULT_POST_PROMPT),
        )

    formatters = {
        "longvideobench": longvideobench,
        "mvbench": mvbench,
        "moviechat": moviechat,
    }
    return formatters, materialize_frames, mvb.parse_pixels


def llava_video_files(revision: str) -> list[str]:
    from huggingface_hub import HfApi

    global _LLAVA_VIDEO_FILES
    if _LLAVA_VIDEO_FILES is None:
        _LLAVA_VIDEO_FILES = HfApi().list_repo_files(
            LLAVA_VIDEO_REPO, repo_type="dataset", revision=revision
        )
    return _LLAVA_VIDEO_FILES


def llava_video_archives(
    subset: str, revision: str, wanted: Sequence[int] | None = None
) -> list[str]:
    """Repo paths of a subset's video archives, by number or as requested."""
    numbered = {}
    for name in llava_video_files(revision):
        match = _VIDEO_ARCHIVE.search(name)
        if name.startswith(f"{subset}/") and match:
            numbered[int(match.group(1))] = name
    if not numbered:
        raise ValueError(f"{subset!r} has no *_videos_N.tar.gz archives at {revision[:12]}")
    if wanted is None:
        return [numbered[number] for number in sorted(numbered)]
    unknown = [number for number in wanted if number not in numbered]
    if unknown:
        raise ValueError(
            f"{subset!r} has no archive(s) {unknown}; it has 1..{max(numbered)}"
        )
    return [numbered[number] for number in wanted]


def load_llava_video_questions(
    subset: str, revision: str
) -> tuple[dict[str, dict[str, list]], Counter[str]]:
    """Every question of the subset, by video path, split into mc and oe.

    A multiple-choice question is kept as (question, options, answer) with its
    closing instruction already removed; an open-ended one as (question,
    answer). The answers only fill the assistant turn of the record, which the
    regeneration discards, so they are kept as the source wrote them.
    """
    from huggingface_hub import hf_hub_download

    files = llava_video_files(revision)
    folder = f"{subset}/"
    sources = {
        kind: sorted(
            name
            for name in files
            if name.startswith(folder)
            and f"_{kind}_" in Path(name).name
            and name.endswith("qa_processed.json")
        )
        for kind in ("mc", "oe")
    }
    missing = [kind for kind, names in sources.items() if not names]
    if missing:
        raise ValueError(
            f"{subset!r} has no {' or '.join(missing)} qa_processed.json; the "
            "video preset needs both question kinds"
        )

    questions: dict[str, dict[str, list]] = defaultdict(lambda: {"mc": [], "oe": []})
    stats: Counter[str] = Counter()
    for kind, names in sources.items():
        for name in names:
            local = hf_hub_download(
                LLAVA_VIDEO_REPO, name, repo_type="dataset", revision=revision
            )
            with open(local, encoding="utf-8") as handle:
                rows = json.load(handle)
            for row in rows:
                video = row.get("video")
                if not isinstance(video, str) or not video:
                    stats[f"{kind} rows without a video"] += 1
                    continue
                _system, pairs, skipped, _wants_image = conversation_pairs(
                    row.get("conversations") or []
                )
                stats[f"{kind} turns skipped"] += skipped
                for question, answer in pairs:
                    if kind == "mc":
                        parsed = parse_mc_question(question)
                        if parsed is None:
                            stats["mc questions off-layout"] += 1
                            continue
                        questions[video]["mc"].append((parsed[0], parsed[1], answer))
                    else:
                        text = question.replace(IMAGE_PLACEHOLDER, "").strip()
                        if not text:
                            stats["oe questions empty"] += 1
                            continue
                        questions[video]["oe"].append((text, answer))
                    stats[f"{kind} questions"] += 1
    return dict(questions), stats


def _drop_cached_file(local: str) -> None:
    """Remove a hf_hub_download result: the symlink and the blob behind it."""
    path = Path(local)
    target = path.resolve()
    path.unlink(missing_ok=True)
    if target != path:
        target.unlink(missing_ok=True)


def archive_videos(
    archive: str,
    wanted: Mapping[str, str],
    video_directory: Path,
    *,
    revision: str,
    keep_archive: bool,
) -> list[str]:
    """Extract an archive's wanted videos and return their paths, in archive order.

    `wanted` maps a video file name to its path in the annotations. The
    archive's member list is cached beside the videos, so a re-run whose
    videos are all still on disk does not download 5 GiB to learn which ones
    it held. Archives are gzip streams, hence read front to back.
    """
    from huggingface_hub import hf_hub_download

    video_directory.mkdir(parents=True, exist_ok=True)
    listing = video_directory / f"{Path(archive).name}.members.json"
    members: list[str] | None = None
    if listing.exists():
        with listing.open(encoding="utf-8") as handle:
            members = json.load(handle)
        here = [name for name in members if name in wanted]
        if all((video_directory / name).exists() for name in here):
            return [wanted[name] for name in here]

    started = time.monotonic()
    local = hf_hub_download(
        LLAVA_VIDEO_REPO, archive, repo_type="dataset", revision=revision
    )
    size = Path(local).stat().st_size
    print(
        f"    {Path(archive).name}: {size / 1024**3:.2f} GiB in "
        f"{time.monotonic() - started:,.0f}s, extracting ...",
        flush=True,
    )
    members = []
    extracted: list[str] = []
    with tarfile.open(local, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name
            members.append(name)
            video = wanted.get(name)
            if video is None:
                continue
            target = video_directory / name
            if not target.exists():
                source = tar.extractfile(member)
                if source is None:
                    continue
                staging = target.with_name(target.name + ".part")
                with staging.open("wb") as sink:
                    shutil.copyfileobj(source, sink, 1 << 20)
                staging.replace(target)
            extracted.append(video)
    with listing.open("w", encoding="utf-8") as handle:
        json.dump(members, handle)
    if not keep_archive:
        _drop_cached_file(local)
    print(
        f"    {len(members)} videos in the archive, {len(extracted)} with both "
        f"question kinds kept",
        flush=True,
    )
    return extracted


def _frame_tokens(width: int, height: int) -> int:
    """Roughly what one frame costs Qwen3.5: 32x32 pixels per visual token."""
    return max(1, round(width / 32)) * max(1, round(height / 32))


def prepare_llava_video(args: argparse.Namespace) -> Path:
    """Write `--sample-size` video records whose prompts match the video benchmarks.

    Videos come from whole archives of one subset, taken in order (or as
    `--archives` says) until the sample is full; two ~5 GiB archives cover
    2,000 records at ten per video. Only videos with both multiple-choice and
    open-ended questions are used. Each is decoded once into `--num-frames`
    frames at the benchmarks' 1280x720 area, and its records are drawn with
    `choose_video_questions`: multiple-choice ones alternate between the
    LongVideoBench and MVBench layouts, open-ended ones take MovieChat's, all
    three through the benchmarks' own formatters.
    """
    formatters, materialize_frames, parse_pixels = video_benchmark_tools()
    from PIL import Image

    name = args.output_name or args.dataset
    output_directory = Path(args.output_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIRECTORY / f"{name}_manifest.json"
    archives: list[str] | None = None
    if args.manifest is not None:
        recorded = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        for key in ("subset", "seed", "per_video", "oe_share", "num_frames", "frame_pixels", "revision", "sample_size"):
            setattr(args, key, recorded[key])
        archives = list(recorded["archives"])
        print(
            f"Replaying {args.manifest}: {args.sample_size:,} records from "
            f"{len(archives)} archive(s) of {args.subset} at {args.revision[:12]}"
        )
    if args.image_root is not None:
        frames_directory = Path(args.image_root) / name
    else:
        frames_directory = Path(args.image_dir or output_directory / f"{name}_images")
    frames_directory.mkdir(parents=True, exist_ok=True)
    video_directory = Path(
        args.video_dir or frames_directory.with_name(frames_directory.name + "_videos")
    )
    pixels = parse_pixels(args.frame_pixels)

    questions, stats = load_llava_video_questions(args.subset, args.revision)
    eligible = {
        Path(video).name: video
        for video, pools in questions.items()
        if pools["mc"] and pools["oe"]
    }
    print(
        f"{LLAVA_VIDEO_REPO}/{args.subset}: {len(questions):,} videos with "
        f"questions, {len(eligible):,} with both kinds "
        f"({stats['mc questions']:,} multiple-choice, {stats['oe questions']:,} "
        "open-ended questions in all)"
    )
    dropped = {key: value for key, value in stats.items() if not key.endswith("questions") and value}
    if dropped:
        print(f"  not usable: {dropped}")
    if archives is None:
        archives = llava_video_archives(args.subset, args.revision, args.archives)
    want_oe = int(round(args.per_video * args.oe_share))
    print(
        f"Taking {args.per_video} records per video ({args.per_video - want_oe} "
        f"multiple-choice, {want_oe} open-ended) as {args.num_frames} frames at "
        f"{args.frame_pixels} until {args.sample_size:,} records are written; "
        f"archives are read from {Path(archives[0]).name} onwards."
    )

    train_output_path = output_directory / f"{name}_train.jsonl"
    if train_output_path.exists():
        print(f"Overwriting existing dataset at {train_output_path}.")
    written = 0
    videos_used = 0
    videos_undecodable = 0
    videos_short = 0
    consumed: list[str] = []
    style_counts: Counter[str] = Counter()
    prompt_chars: dict[str, list[int]] = defaultdict(list)
    frame_sizes: Counter[tuple[int, int]] = Counter()
    frames_per_video: list[int] = []
    staging = train_output_path.with_suffix(".jsonl.partial")
    with staging.open("w", encoding="utf-8") as handle:
        for archive in archives:
            if written >= args.sample_size:
                break
            print(f"  archive {Path(archive).name} ...", flush=True)
            videos = archive_videos(
                archive,
                eligible,
                video_directory,
                revision=args.revision,
                keep_archive=args.keep_archives,
            )
            consumed.append(archive)
            # which of an archive's videos are used first is random, but fixed
            # by the seed and the archive, not by what came before
            random.Random(f"{args.seed}:{archive}").shuffle(videos)
            for video in videos:
                if written >= args.sample_size:
                    break
                file_name = Path(video).name
                frames = materialize_frames(
                    str(frames_directory),
                    str(video_directory / file_name),
                    Path(video).stem,
                    args.num_frames,
                    target_pixels=pixels or None,
                )
                if not frames:
                    videos_undecodable += 1
                    continue
                if len(frames) < args.num_frames:
                    videos_short += 1
                with Image.open(frames[0]) as first:
                    frame_size = first.size
                frame_sizes[frame_size] += 1
                frames_per_video.append(len(frames))

                pools = questions[video]
                picked_mc, picked_oe = choose_video_questions(
                    video, len(pools["mc"]), len(pools["oe"]), args.per_video, args.oe_share, args.seed
                )
                records: list[dict[str, Any]] = []
                for position, index in enumerate(picked_mc):
                    style = "longvideobench" if position % 2 == 0 else "mvbench"
                    question, options, answer = pools["mc"][index]
                    text = formatters[style](question, options)
                    records.append(
                        build_video_record(args.subset, video, frames, "mc", index, style, text, answer)
                    )
                for index in picked_oe:
                    question, answer = pools["oe"][index]
                    text = formatters["moviechat"](question)
                    records.append(
                        build_video_record(args.subset, video, frames, "oe", index, "moviechat", text, answer)
                    )
                for record in records[: args.sample_size - written]:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    style_counts[record["style"]] += 1
                    prompt_chars[record["style"]].append(
                        len(record["conversations"][0]["content"]) - len(frames) * len(IMAGE_PLACEHOLDER)
                    )
                videos_used += 1
                print(
                    f"    [{videos_used}] {Path(video).stem}: {len(frames)} frames "
                    f"{frame_size[0]}x{frame_size[1]}, {len(picked_mc)} mc + "
                    f"{len(picked_oe)} oe -> {written:,}/{args.sample_size:,}",
                    flush=True,
                )
    staging.replace(train_output_path)

    if written < args.sample_size:
        print(
            f"\nWARNING: only {written:,} of {args.sample_size:,} records could be "
            f"written from {len(consumed)} archive(s); pass more via --archives."
        )
    print(
        f"\nSaved {written:,} records over {videos_used:,} videos from "
        f"{len(consumed)} archive(s) to {train_output_path}"
    )
    print(f"Frames written under {frames_directory}; videos kept under {video_directory}")
    if videos_undecodable or videos_short:
        print(
            f"  {videos_undecodable} video(s) could not be decoded and were skipped; "
            f"{videos_short} gave fewer than {args.num_frames} frames"
        )
    if frame_sizes:
        size, count = frame_sizes.most_common(1)[0]
        per_frame = _frame_tokens(*size)
        median_frames = sorted(frames_per_video)[len(frames_per_video) // 2]
        print(
            f"  frames: {size[0]}x{size[1]} for {count} of {videos_used} videos "
            f"(~{per_frame} tokens each on Qwen3.5), {median_frames} per video "
            f"-> ~{per_frame * median_frames:,} image tokens per prompt"
        )
    print("\nRecords by prompt style:")
    for style in VIDEO_PROMPT_STYLES:
        lengths = sorted(prompt_chars[style])
        if not lengths:
            continue
        print(
            f"  {style:<16}{style_counts[style]:>7,}{style_counts[style] / max(written, 1):>8.1%}"
            f"   text p50 {lengths[len(lengths) // 2]:,} chars"
        )
    if args.manifest is not None:
        print(f"\nReusing {args.manifest}; not rewriting {manifest_path}")
        return train_output_path

    import datasets as _datasets

    examples = {
        "longvideobench": formatters["longvideobench"]("<question>", ["<option A>", "<option B>"]),
        "mvbench": formatters["mvbench"]("<question>", ["<option A>", "<option B>"]),
        "moviechat": formatters["moviechat"]("<question>"),
    }
    MANIFEST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "repo": LLAVA_VIDEO_REPO,
                "revision": args.revision,
                "subset": args.subset,
                "sample_size": args.sample_size,
                "per_video": args.per_video,
                "oe_share": args.oe_share,
                "num_frames": args.num_frames,
                "frame_pixels": args.frame_pixels,
                "seed": args.seed,
                "archives": consumed,
                "records": written,
                "videos": videos_used,
                "styles": dict(style_counts),
                # the exact prompt text each style produced, for the record
                "prompt_templates": examples,
                "versions": {"datasets": _datasets.__version__},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")
    return train_output_path


def process_and_save_dataset(
    dataset: Iterable[Mapping[str, Any]],
    output_directory: Path,
    processor: RowProcessor,
    dataset_name: str,
    image_root: Path,
    *,
    eval_dataset: Iterable[Mapping[str, Any]] | None = None,
    dump_missing_prefixes: Sequence[str] = (),
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    train_output_path = output_directory / f"{dataset_name}_train.jsonl"
    # Always regenerate: overwrite any existing output instead of skipping.
    if train_output_path.exists():
        print(f"Overwriting existing dataset at {train_output_path}.")

    skipped_messages, missing_images, missing_prefixes, dumped_missing_paths = _write_split(
        dataset,
        train_output_path,
        processor,
        dataset_name,
        image_root,
        dump_missing_prefixes,
    )
    if eval_dataset is not None:
        eval_output_path = output_directory / f"{dataset_name}_test.jsonl"
        eval_skipped, eval_missing, eval_missing_prefixes, eval_dumped = _write_split(
            eval_dataset,
            eval_output_path,
            processor,
            dataset_name,
            image_root,
            dump_missing_prefixes,
        )
        skipped_messages += eval_skipped
        missing_images += eval_missing
        missing_prefixes += eval_missing_prefixes
        dumped_missing_paths += eval_dumped

    if skipped_messages:
        print(
            f"Skipped {skipped_messages} unsupported or image-less messages while "
            f"processing {dataset_name}."
        )
    if missing_images:
        print(f"Skipped {missing_images} rows with images missing under {image_root}.")
        print("Missing-image path prefixes (count, prefix):")
        for prefix, count in missing_prefixes.most_common():
            print(f"  {count}\t{prefix}")
    if dumped_missing_paths:
        print("Missing images under the requested prefixes:")
        for path in sorted(dumped_missing_paths):
            print(f"  {path}")
    print(f"Saved {dataset_name} training data to {train_output_path}.")
    return train_output_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dataset == LLAVA_OV_PRESET:
        prepare_llava_onevision(args)
        return
    if args.dataset == LLAVA_VIDEO_PRESET:
        prepare_llava_video(args)
        return

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
        args.output_name or args.dataset,
        args.image_root,
        eval_dataset=eval_dataset,
        dump_missing_prefixes=args.dump_missing_prefix,
    )


if __name__ == "__main__":
    main()
    # `datasets` streaming leaves an aiohttp session on a background thread, and
    # tearing it down at interpreter shutdown aborts with a GIL error long after
    # the output is written -- a successful run would otherwise report failure.
    # Everything is already flushed and closed by here, so skip finalization.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
