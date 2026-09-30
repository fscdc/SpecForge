"""
LongVideoBench evaluation script, ported from lmms-eval's `longvideobench_test_v`.

LongVideoBench (https://huggingface.co/datasets/longvideobench/LongVideoBench)
asks five-way multiple-choice questions about videos of 15 seconds to an hour,
grouped by duration (15 / 60 / 600 / 3600 seconds) and by one of 17 question
categories. The test split has 5,341 questions over 3,008 videos and ships
without answers (`correct_choice` is -1 throughout); the validation split has
1,337 questions with answers and can be selected with `LVB_SPLIT=validation`.

What this port keeps from lmms-eval is the prompt shape: the question, the
options lettered "A. ..." to "E. ...", and a post-prompt. What it changes is
that post-prompt. The original asks for the letter alone, which yields a
one-token answer and tells a speculative decoder nothing; the default here asks
the model to reason about the video first and end with the letter, so the
generation is long enough to measure accept length and throughput on. The
original wording is still available as `longvideobench-origin`.

What it drops is the scoring: the test split has no ground truth, and this is a
decoding benchmark. On the validation split the letter accuracy is computed for
whatever it is worth.

Videos come as a tar split into 31 parts of 5.24 GiB (161 GiB in total). They
are streamed part by part, in order, and only the videos this run needs are
written out; the stream stops as soon as enough videos are on disk to cover the
requested number of questions. A `longvideobench:50` run therefore normally
pulls the first part or two. Since the parts are pulled in archive order, the
questions of a partial run are the earliest ones in dataset order whose video
happens to be in those parts. `LVB_MAX_PARTS` caps the download (default 4
parts, ~21 GiB; 0 lifts the cap), `LVB_VIDEO_DIR` points at an existing copy,
and `LVB_AUTO_DOWNLOAD=0` refuses to download at all.

Decoded frames are cached and kept, as for VDC, under `LVB_FRAMES_DIR`.
"""

import io
import os
import re
import tarfile
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from benchmarker.utils import BenchmarkMetrics, compute_metrics
from huggingface_hub import hf_hub_download, list_repo_files

from .base import MMBenchmarker
from .registry import MM_BENCHMARKS
from .utils import create_interleaved_sgl_function, strip_reasoning
from .video_utils import (
    default_video_dir,
    env_flag,
    index_videos,
    materialize_frames,
)

DATASET_PATH = "longvideobench/LongVideoBench"
SPLIT_FILES = {
    "test": "test-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
}
DEFAULT_SPLIT = "test"

ARCHIVE_PREFIX = "videos.tar.part."
DEFAULT_MAX_PARTS = 4
DEFAULT_VIDEO_SUBDIR = "longvideobench_videos"
DEFAULT_NUM_FRAMES = 16
DEFAULT_MAX_NEW_TOKENS = 1024

OPTION_LETTERS = "ABCDE"

# lmms-eval's post-prompt, verbatim: a one-token answer
ORIGINAL_POST_PROMPT = "Answer with the option's letter from the given choices directly.\n"
# the default here: reason first, so that there is a generation to measure
DEFAULT_POST_PROMPT = (
    "Think through what happens in the video step by step first, then give your "
    "final answer as the option's letter.\n"
)
DEFAULT_PRE_PROMPT = ""

# what an answer may be prefixed with, dropped before the letter is looked for
ANSWER_PREFIXES = (
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is",
    "The correct option is",
    "Best answer:",
    "Best option:",
)


def load_split_rows(split: str) -> List[Dict[str, Any]]:
    """The questions of one split, read from its parquet alone.

    `datasets.load_dataset` on this repository would enumerate the 161 GiB of
    video archives as data files; fetching the one parquet keeps the metadata
    load at a few MiB.
    """
    import pyarrow.parquet as pq

    if split not in SPLIT_FILES:
        raise ValueError(
            f"Unknown LongVideoBench split '{split}', expected one of "
            f"{sorted(SPLIT_FILES)}"
        )
    path = hf_hub_download(DATASET_PATH, SPLIT_FILES[split], repo_type="dataset")
    return pq.read_table(path).to_pylist()


def options_of(row: Dict[str, Any]) -> List[str]:
    """The answer options of one row, in order, without the N/A padding."""
    options = []
    for index in range(5):
        option = row.get(f"option{index}")
        if option is None or str(option) == "N/A":
            break
        options.append(str(option))
    return options


def format_question(row: Dict[str, Any], pre_prompt: str, post_prompt: str) -> str:
    """lmms-eval's `longvideobench_doc_to_text`, with the post-prompt as given."""
    lines = [
        f"{OPTION_LETTERS[index]}. {option}"
        for index, option in enumerate(options_of(row))
    ]
    question = str(row["question"]) + "\n" + "\n".join(lines)
    return f"{pre_prompt}{question}\n{post_prompt}"


def parse_choice(response: str, num_options: int) -> Optional[str]:
    """
    The option letter a response settles on, or None.

    lmms-eval takes the first letter it finds, which is right for the one-token
    answers its prompt asks for. A reasoning answer mentions letters along the
    way and puts its choice last, so the last standalone letter wins here, after
    the usual "The answer is" prefixes are stripped. No random fallback: an
    unreadable answer is reported as such, not as a coin toss.
    """
    if not isinstance(response, str):
        return None
    text = strip_reasoning(response)
    for prefix in ANSWER_PREFIXES:
        text = text.replace(prefix, "")
    letters = OPTION_LETTERS[: max(1, min(num_options, len(OPTION_LETTERS)))]
    matches = re.findall(rf"(?<![A-Za-z])([{letters}])(?![A-Za-z])", text)
    return matches[-1] if matches else None


class _SplitTarStream(io.RawIOBase):
    """
    The concatenation of the archive's parts, read as one sequential stream.

    Each part is downloaded (resumably, into the Hugging Face cache) the moment
    the previous one is exhausted, so a run that stops after the first part
    never pays for the rest. `parts_read` tells how many were opened.
    """

    def __init__(self, parts: List[str], max_parts: int):
        super().__init__()
        self.parts = parts
        self.max_parts = max_parts
        self.parts_read = 0
        self._handle = None
        self._next = 0

    def readable(self) -> bool:
        return True

    def _open_next(self) -> bool:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._next >= len(self.parts):
            return False
        if self.max_parts and self._next >= self.max_parts:
            return False
        part = self.parts[self._next]
        self._next += 1
        print(f"  downloading {part} (resumable, cached under HF_HOME)...")
        path = hf_hub_download(DATASET_PATH, filename=part, repo_type="dataset")
        self._handle = open(path, "rb")
        self.parts_read += 1
        return True

    def readinto(self, buffer) -> int:
        view = memoryview(buffer)
        filled = 0
        while filled < len(view):
            if self._handle is None and not self._open_next():
                break
            count = self._handle.readinto(view[filled:])
            if not count:
                if not self._open_next():
                    break
                continue
            filled += count
        return filled

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


@MM_BENCHMARKS.register("longvideobench")
class LongVideoBenchBenchmarker(MMBenchmarker):
    """
    LongVideoBench, as a decoding benchmark.

    Args:
        num_samples: number of questions to ask, all of them when not given.
        subset: restrict the questions to one or more duration groups (15, 60,
            600, 3600) or question categories (T2E, S2O, ...), e.g.
            `longvideobench:50:3600` or `longvideobench:50:T2E,S2O`.
        split: "test" (the default, no answers) or "validation".
        num_frames: how many evenly spaced frames a video is sent as. A video
            shorter than that many seconds contributes one frame per second,
            as in lmms-eval.
        video_dir: directory holding the videos. Defaults to
            `$HF_HOME/longvideobench_videos`, which is also where the archive
            parts are unpacked to.
        auto_download: whether missing videos may be pulled out of the archive.
        max_parts: how many of the 31 archive parts may be pulled in one run;
            0 lifts the cap.
        frames_dir: where the decoded frames are cached and kept.
        pre_prompt, post_prompt: the text around the question; see the module
            docstring for why the default post-prompt asks for reasoning.

    Every argument except `num_samples` and `subset` also reads an `LVB_`
    environment variable of the same name.
    """

    ORIGINAL_PROMPT_KWARGS = {"post_prompt": ORIGINAL_POST_PROMPT}

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        split: Optional[str] = None,
        num_frames: Optional[int] = None,
        video_dir: Optional[str] = None,
        auto_download: Optional[bool] = None,
        max_parts: Optional[int] = None,
        frames_dir: Optional[str] = None,
        pre_prompt: Optional[str] = None,
        post_prompt: Optional[str] = None,
    ):
        super().__init__(num_samples, subset)
        self.split = str(split or os.environ.get("LVB_SPLIT") or DEFAULT_SPLIT).lower()
        if self.split not in SPLIT_FILES:
            raise ValueError(
                f"Unknown LongVideoBench split '{self.split}', expected one of "
                f"{sorted(SPLIT_FILES)}"
            )
        self.num_frames = int(
            num_frames or os.environ.get("LVB_NUM_FRAMES") or DEFAULT_NUM_FRAMES
        )
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be at least 1, got {self.num_frames}")
        self.video_dir = default_video_dir(
            DEFAULT_VIDEO_SUBDIR, video_dir or os.environ.get("LVB_VIDEO_DIR")
        )
        self.auto_download = (
            auto_download
            if auto_download is not None
            else env_flag("LVB_AUTO_DOWNLOAD", True)
        )
        self.max_parts = int(
            max_parts
            if max_parts is not None
            else os.environ.get("LVB_MAX_PARTS", DEFAULT_MAX_PARTS)
        )
        self.frames_dir = (
            frames_dir
            or os.environ.get("LVB_FRAMES_DIR")
            or os.path.join(".cache", "longvideobench_frames_specforge")
        )
        self.pre_prompt = (
            pre_prompt
            if pre_prompt is not None
            else os.environ.get("LVB_PRE_PROMPT", DEFAULT_PRE_PROMPT)
        )
        self.post_prompt = (
            post_prompt
            if post_prompt is not None
            else os.environ.get("LVB_POST_PROMPT", DEFAULT_POST_PROMPT)
        )

        # per-question metadata, aligned with the loaded questions; "categories"
        # is the name dump_generations() knows
        self.categories: List[str] = []
        self.l2_categories: List[str] = []
        self.question_ids: List[str] = []
        self.num_options: List[int] = []
        self.missing_videos: List[str] = []
        self.parts_read = 0

    def default_max_new_tokens(self) -> int:
        return DEFAULT_MAX_NEW_TOKENS

    # -- data ---------------------------------------------------------------
    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        rows = load_split_rows(self.split)
        if self.subset:
            rows = self._select_subset(rows)
        if not rows:
            raise ValueError("No LongVideoBench question left after filtering")

        os.makedirs(self.video_dir, exist_ok=True)
        by_name, by_stem = self._ensure_videos(rows)
        selected = self._select_rows(rows, by_name, by_stem)
        if not selected:
            raise FileNotFoundError(
                f"None of the videos of this run is under {self.video_dir}, and "
                "none could be downloaded. See the messages above."
            )

        questions: List[Dict[str, Any]] = []
        labels: List[Optional[str]] = []
        self.categories, self.l2_categories = [], []
        self.question_ids, self.num_options = [], []
        self.missing_videos = []
        for row in selected:
            video_path = self._find_video(row, by_name, by_stem)
            frames = self._frames_of(row, video_path)
            if not frames:
                self.missing_videos.append(str(row["video_path"]))
                continue
            text = format_question(row, self.pre_prompt, self.post_prompt)
            parts = [("image", path) for path in frames] + [("text", text)]
            questions.append({"parts": parts, "id": str(row["id"])})
            choice = int(row.get("correct_choice", -1) or -1)
            labels.append(OPTION_LETTERS[choice] if 0 <= choice < 5 else None)
            self.categories.append(str(row.get("duration_group", "unknown")))
            self.l2_categories.append(str(row.get("question_category", "unknown")))
            self.question_ids.append(str(row["id"]))
            self.num_options.append(len(options_of(row)))

        if self.missing_videos:
            print(
                f"Skipped {len(self.missing_videos)} questions whose video could "
                f"not be decoded, e.g. {', '.join(self.missing_videos[:3])}"
            )
        print(
            f"Loaded {len(questions)} LongVideoBench {self.split} questions over "
            f"{len(set(q['parts'][0][1].rsplit('__', 1)[0] for q in questions))} "
            f"videos, up to {self.num_frames} frames each. Frames cached in "
            f"{self.frames_dir}"
        )
        return questions, labels

    def _select_subset(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rows whose duration group or question category is in the subset."""
        wanted = {str(name).strip().lower() for name in self.subset}
        groups = {str(row.get("duration_group")).lower() for row in rows}
        categories = {str(row.get("question_category")).lower() for row in rows}
        unknown = wanted - groups - categories
        if unknown:
            raise ValueError(
                f"Unknown LongVideoBench subset(s) {sorted(unknown)}; expected a "
                f"duration group among {sorted(groups)} or a question category "
                f"among {sorted(categories)}"
            )
        return [
            row
            for row in rows
            if str(row.get("duration_group")).lower() in wanted
            or str(row.get("question_category")).lower() in wanted
        ]

    @staticmethod
    def _find_video(
        row: Dict[str, Any], by_name: Dict[str, str], by_stem: Dict[str, str]
    ) -> Optional[str]:
        name = os.path.basename(str(row.get("video_path") or ""))
        if name in by_name:
            return by_name[name]
        for stem in (str(row.get("video_id") or ""), os.path.splitext(name)[0]):
            if stem and stem in by_stem:
                return by_stem[stem]
        return None

    def _covered(
        self, rows: List[Dict[str, Any]], by_name: Dict[str, str], by_stem: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        return [row for row in rows if self._find_video(row, by_name, by_stem)]

    def _select_rows(
        self, rows: List[Dict[str, Any]], by_name: Dict[str, str], by_stem: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """The first `num_samples` rows, in dataset order, whose video is on disk."""
        covered = self._covered(rows, by_name, by_stem)
        if self.num_samples is not None:
            covered = covered[: self.num_samples]
        return covered

    def _ensure_videos(
        self, rows: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Pull archive parts, in order, until enough videos cover the run.

        The wanted set is every candidate row's video, so any of them that
        streams past is kept; the stream stops once the covered rows reach
        `num_samples` (or the archive, or the part cap, runs out).
        """
        by_name, by_stem = index_videos(self.video_dir)
        target = self.num_samples if self.num_samples is not None else len(rows)
        if len(self._covered(rows, by_name, by_stem)) >= target:
            return by_name, by_stem
        if not self.auto_download:
            raise FileNotFoundError(
                f"Only {len(self._covered(rows, by_name, by_stem))} of the "
                f"{target} requested questions have their video under "
                f"{self.video_dir} and LVB_AUTO_DOWNLOAD is off."
            )

        wanted: Set[str] = set()
        wanted_stems: Dict[str, str] = {}
        for row in rows:
            if self._find_video(row, by_name, by_stem):
                continue
            name = os.path.basename(str(row["video_path"]))
            wanted.add(name)
            wanted_stems[os.path.splitext(name)[0]] = name
            wanted_stems[str(row.get("video_id") or "")] = name

        parts = self._list_archive_parts()
        cap = f"at most {self.max_parts} of them (LVB_MAX_PARTS)" if self.max_parts else "all of them"
        print(
            f"{len(self._covered(rows, by_name, by_stem))} of the {target} requested "
            f"questions have their video under {self.video_dir}. Streaming the "
            f"dataset's {len(parts)} archive parts of ~5.2 GiB in order, {cap}, "
            "and stopping as soon as enough videos are out."
        )
        stream = _SplitTarStream(parts, self.max_parts)
        extracted = 0
        try:
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                for member in self._iter_members(tar, stream):
                    if not member.isfile():
                        continue
                    name = os.path.basename(member.name)
                    stem = os.path.splitext(name)[0]
                    row_name = name if name in wanted else wanted_stems.get(stem)
                    if row_name is None:
                        continue
                    source = tar.extractfile(member)
                    if source is None:
                        continue
                    target_path = os.path.join(self.video_dir, name)
                    with source, open(target_path + ".part", "wb") as sink:
                        while True:
                            chunk = source.read(1 << 20)
                            if not chunk:
                                break
                            sink.write(chunk)
                    os.replace(target_path + ".part", target_path)
                    wanted.discard(row_name)
                    extracted += 1
                    by_name[name] = target_path
                    by_stem[stem] = target_path
                    if len(self._covered(rows, by_name, by_stem)) >= target:
                        break
        finally:
            stream.close()
        self.parts_read = stream.parts_read
        covered = len(self._covered(rows, by_name, by_stem))
        print(
            f"  unpacked {extracted} videos out of {stream.parts_read} archive "
            f"part(s); {covered} of {target} requested questions are now covered"
        )
        if covered < target:
            print(
                "  raise LVB_MAX_PARTS (0 = no cap) to pull more of the archive, "
                "or lower the number of questions"
            )
        return index_videos(self.video_dir)

    @staticmethod
    def _iter_members(tar, stream: "_SplitTarStream"):
        """
        The archive's members, ending quietly where the part cap cuts the stream.

        A tar split into parts ends mid-entry when only the first parts are
        read, which tarfile reports as an unexpected end of data. Under the
        LVB_MAX_PARTS cap that is the expected way for the stream to stop, not
        a corrupt archive.
        """
        try:
            yield from tar
        except (tarfile.ReadError, tarfile.StreamError, EOFError) as error:
            capped = stream.max_parts and stream.parts_read >= stream.max_parts
            if capped or stream.parts_read >= len(stream.parts):
                print(f"  reached the end of the streamed archive parts ({error})")
                return
            raise

    @staticmethod
    def _list_archive_parts() -> List[str]:
        parts = sorted(
            name
            for name in list_repo_files(DATASET_PATH, repo_type="dataset")
            if name.startswith(ARCHIVE_PREFIX)
        )
        if not parts:
            raise FileNotFoundError(f"{DATASET_PATH} has no {ARCHIVE_PREFIX}* to download")
        return parts

    def _frames_of(self, row: Dict[str, Any], video_path: Optional[str]) -> List[str]:
        if video_path is None:
            return []
        try:
            duration = float(row.get("duration") or 0)
        except (TypeError, ValueError):
            duration = 0.0
        # lmms-eval samples one frame per second for videos shorter than the budget
        count = self.num_frames
        if duration > 0:
            count = max(1, min(self.num_frames, int(duration)))
        return materialize_frames(self.frames_dir, video_path, str(row["video_id"]), count)

    # -- scoring ------------------------------------------------------------
    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        return parse_choice(output, 5)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Letter accuracy where answers exist (validation); None on the test split."""
        scored = [(p, l) for p, l in zip(predictions, labels) if l is not None]
        if not scored:
            return None
        return sum(1 for p, l in scored if p == l) / len(scored)

    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, BenchmarkMetrics]]:
        """The metrics of every duration group, the axis the prefill length follows."""
        if not self.categories:
            return None
        performance = {}
        for group in sorted(set(self.categories), key=lambda g: (len(g), g)):
            indexes = [
                index
                for index, name in enumerate(self.categories)
                if name == group and index < len(states)
            ]
            if indexes:
                performance[f"duration_{group}s"] = compute_metrics(
                    [states[index] for index in indexes], latency, answer_key=answer_key
                )
        return performance

    def describe_run(self) -> Optional[Dict[str, Any]]:
        description: Dict[str, Any] = {
            "split": self.split,
            "num_frames": self.num_frames,
            "post_prompt": self.post_prompt,
            "questions": len(self.categories),
            "questions_per_duration_group": dict(sorted(Counter(self.categories).items())),
            "questions_per_category": dict(sorted(Counter(self.l2_categories).items())),
            "question_ids": self.question_ids,
            "video_dir": self.video_dir,
            "frames_dir": self.frames_dir,
            "archive_parts_read": self.parts_read,
            "scoring": (
                "letter accuracy on validation only; the test split ships no answers"
            ),
        }
        if self.missing_videos:
            description["missing_videos"] = len(self.missing_videos)
            description["missing_videos_sample"] = self.missing_videos[:10]
        generations = getattr(self, "generations", None)
        if generations:
            lengths = [len(text) for text in generations if isinstance(text, str)]
            if lengths:
                description["generated_chars_mean"] = sum(lengths) / len(lengths)
        return description

    def create_sgl_function(self):
        return create_interleaved_sgl_function(
            function_name="get_longvideobench_answer",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
            assistant_prefix=self.assistant_prefix,
        )
