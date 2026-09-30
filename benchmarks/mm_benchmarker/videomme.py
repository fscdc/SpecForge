"""
Video-MME evaluation script, ported from lmms-eval's `videomme` task.

Video-MME (https://huggingface.co/datasets/lmms-lab/Video-MME) asks four-way
multiple-choice questions about 900 YouTube videos, 300 each of three duration
groups (short: under 2 minutes, medium: 4 to 15 minutes, long: 30 to 60
minutes), three questions per video, 2,700 in all. The videos are 720p, so a
frame costs the target about 880 tokens and 48 frames make a ~40k-token prompt,
on par with VDC at 16 frames of 1080p+.

What this port keeps from lmms-eval is the prompt shape, without subtitles:
the task's own instruction sentence (verbatim, subtitles mention included),
the question, the lettered options one per line, and a post-prompt. What it
changes is that post-prompt: the original asks for the letter alone, a
one-token answer that tells a speculative decoder nothing, so the default here
asks the model to reason about the video first and end with the letter. The
original wording is still available as `videomme-origin`.

What it drops is the scoring. This is a decoding benchmark; the letter each
answer settles on is extracted for the generations dump, but no accuracy is
reported.

The videos ship as 19 zips of ~5.3 GiB (101 GiB in total). None is downloaded
whole: the archives' central directories are listed over HTTP range reads (a
few seconds each, cached as JSON afterwards), lazily and in order until every
wanted video is located, and only those members are streamed out. A
`videomme:20` run therefore fetches about a dozen videos out of one or two
archives. `VIDEOMME_VIDEO_DIR` points at an existing copy of the videos and
`VIDEOMME_AUTO_DOWNLOAD=0` refuses to download. Decoded frames are cached
under `VIDEOMME_FRAMES_DIR`.
"""

import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from benchmarker.utils import BenchmarkMetrics, compute_metrics
from huggingface_hub import hf_hub_download, list_repo_files

from .base import MMBenchmarker
from .registry import MM_BENCHMARKS
from .utils import create_interleaved_sgl_function, stratified_indices, strip_reasoning
from .video_utils import (
    RemoteZipIndex,
    default_video_dir,
    env_flag,
    index_videos,
    materialize_frames,
)

DATASET_PATH = "lmms-lab/Video-MME"
QUESTIONS_FILE = "videomme/test-00000-of-00001.parquet"
ARCHIVE_PATTERN = re.compile(r"^videos_chunked_\d+\.zip$")
DEFAULT_VIDEO_SUBDIR = "videomme_videos"
INDEX_FILE = "zip_index.json"
DEFAULT_NUM_FRAMES = 48
DEFAULT_MAX_NEW_TOKENS = 1024
DURATIONS = ("short", "medium", "long")
OPTION_LETTERS = "ABCD"

# lmms-eval's instruction sentence, verbatim (the subtitle-free task sends it too)
OPTION_PROMPT = (
    "Select the best answer to the following multiple-choice question based on "
    "the video and the subtitles. Respond with only the letter (A, B, C, or D) "
    "of the correct option."
)
# lmms-eval's post-prompt, verbatim: a one-token answer
ORIGINAL_POST_PROMPT = "Answer with the option's letter from the given choices directly."
# the default here: reason first, so that there is a generation to measure
DEFAULT_POST_PROMPT = (
    "Think through the video step by step first, then give your final answer as "
    "the option's letter."
)

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


def load_rows() -> List[Dict[str, Any]]:
    """The 2,700 questions, read from the one parquet alone (never a zip)."""
    import pyarrow.parquet as pq

    path = hf_hub_download(DATASET_PATH, QUESTIONS_FILE, repo_type="dataset")
    return pq.read_table(path).to_pylist()


def options_of(row: Dict[str, Any]) -> List[str]:
    """The options of one row as lines, already lettered ("A. Apples.")."""
    options = row.get("options")
    if isinstance(options, str):
        import ast

        try:
            options = ast.literal_eval(options)
        except (ValueError, SyntaxError):
            options = [options]
    return [str(option) for option in (options or [])]


def format_question(row: Dict[str, Any], post_prompt: str) -> str:
    """lmms-eval's `videomme_doc_to_text`, with the post-prompt as given."""
    question = str(row["question"]) + "\n" + "\n".join(options_of(row))
    return OPTION_PROMPT + "\n" + question + "\n" + post_prompt


def parse_choice(response: str, num_options: int = 4) -> Optional[str]:
    """
    The option letter a response settles on, or None.

    A reasoning answer mentions letters along the way and puts its choice last,
    so the last standalone letter wins, after the usual "The answer is"
    prefixes are stripped. No random fallback.
    """
    if not isinstance(response, str):
        return None
    text = strip_reasoning(response)
    for prefix in ANSWER_PREFIXES:
        text = text.replace(prefix, "")
    letters = OPTION_LETTERS[: max(1, min(num_options, len(OPTION_LETTERS)))]
    matches = re.findall(rf"(?<![A-Za-z])([{letters}])(?![A-Za-z])", text)
    return matches[-1] if matches else None


def list_archives() -> List[str]:
    archives = sorted(
        name
        for name in list_repo_files(DATASET_PATH, repo_type="dataset")
        if ARCHIVE_PATTERN.match(os.path.basename(name))
    )
    if not archives:
        raise FileNotFoundError(f"{DATASET_PATH} has no videos_chunked_*.zip to fetch from")
    return archives


@MM_BENCHMARKS.register("videomme")
class VideoMMEBenchmarker(MMBenchmarker):
    """
    Video-MME, as a decoding benchmark.

    Args:
        num_samples: number of questions to ask, all 2,700 when not given. The
            parquet is ordered short, medium, long, so a slice is drawn with
            the duration mix of the whole set (a third each, the first rows of
            every group) rather than from the short videos alone.
        subset: restrict to duration groups (short, medium, long), task types
            or domains, e.g. `videomme:30:long` or `videomme:30:Counting Problem`.
        num_frames: how many evenly spaced frames a video is sent as.
        video_dir: directory holding the videos, named `<videoID>.mp4`.
            Defaults to `$HF_HOME/videomme_videos`, which is also where fetched
            videos are written.
        auto_download: whether missing videos may be fetched out of the zips.
        frames_dir: where the decoded frames are cached and kept.
        post_prompt: the line after the options; see the module docstring for
            why the default asks for reasoning.

    Every argument except `num_samples` and `subset` also reads a `VIDEOMME_`
    environment variable of the same name.
    """

    ORIGINAL_PROMPT_KWARGS = {"post_prompt": ORIGINAL_POST_PROMPT}

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        num_frames: Optional[int] = None,
        video_dir: Optional[str] = None,
        auto_download: Optional[bool] = None,
        frames_dir: Optional[str] = None,
        post_prompt: Optional[str] = None,
    ):
        super().__init__(num_samples, subset)
        self.num_frames = int(
            num_frames or os.environ.get("VIDEOMME_NUM_FRAMES") or DEFAULT_NUM_FRAMES
        )
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be at least 1, got {self.num_frames}")
        self.video_dir = default_video_dir(
            DEFAULT_VIDEO_SUBDIR, video_dir or os.environ.get("VIDEOMME_VIDEO_DIR")
        )
        self.auto_download = (
            auto_download
            if auto_download is not None
            else env_flag("VIDEOMME_AUTO_DOWNLOAD", True)
        )
        self.frames_dir = (
            frames_dir
            or os.environ.get("VIDEOMME_FRAMES_DIR")
            or os.path.join(".cache", "videomme_frames_specforge")
        )
        self.post_prompt = (
            post_prompt
            if post_prompt is not None
            else os.environ.get("VIDEOMME_POST_PROMPT", DEFAULT_POST_PROMPT)
        )

        # per-question metadata, aligned with the loaded questions
        self.categories: List[str] = []
        self.l2_categories: List[str] = []
        self.question_ids: List[str] = []
        self.missing_videos: List[str] = []
        self.archives_listed = 0

    def default_max_new_tokens(self) -> int:
        return DEFAULT_MAX_NEW_TOKENS

    # -- data ---------------------------------------------------------------
    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        rows = load_rows()
        if self.subset:
            rows = self._select_subset(rows)
        rows = [rows[index] for index in stratified_indices([r["duration"] for r in rows], self.num_samples)]
        if not rows:
            raise ValueError("No Video-MME question left after filtering")

        os.makedirs(self.video_dir, exist_ok=True)
        by_name, by_stem = self._ensure_videos(rows)

        questions: List[Dict[str, Any]] = []
        labels: List[Optional[str]] = []
        self.categories, self.l2_categories, self.question_ids = [], [], []
        self.missing_videos = []
        for row in rows:
            video_id = str(row["videoID"])
            video_path = by_stem.get(video_id) or by_name.get(f"{video_id}.mp4")
            frames = (
                materialize_frames(self.frames_dir, video_path, video_id, self.num_frames)
                if video_path
                else []
            )
            if not frames:
                self.missing_videos.append(video_id)
                continue
            text = format_question(row, self.post_prompt)
            parts = [("image", path) for path in frames] + [("text", text)]
            questions.append({"parts": parts, "id": str(row["question_id"])})
            answer = str(row.get("answer") or "").strip().upper()
            labels.append(answer if answer in OPTION_LETTERS else None)
            self.categories.append(str(row.get("duration", "unknown")))
            self.l2_categories.append(str(row.get("task_type", "unknown")))
            self.question_ids.append(str(row["question_id"]))

        if self.missing_videos:
            print(
                f"Skipped {len(self.missing_videos)} questions whose video is "
                f"missing or undecodable, e.g. {', '.join(sorted(set(self.missing_videos))[:3])}"
            )
        print(
            f"Loaded {len(questions)} Video-MME questions over "
            f"{len(set(q['parts'][0][1].rsplit('__', 1)[0] for q in questions))} "
            f"videos as {self.num_frames} frames each "
            f"({dict(Counter(self.categories))}). Frames cached in {self.frames_dir}"
        )
        return questions, labels

    def _select_subset(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rows whose duration group, task type or domain is in the subset."""
        wanted = {str(name).strip().lower() for name in self.subset}

        def keys(row):
            return {
                str(row.get("duration")).lower(),
                str(row.get("task_type")).lower(),
                str(row.get("domain")).lower(),
                str(row.get("sub_category")).lower(),
            }

        known = set().union(*(keys(row) for row in rows))
        unknown = wanted - known
        if unknown:
            raise ValueError(
                f"Unknown Video-MME subset(s) {sorted(unknown)}; expected a duration "
                f"group among {DURATIONS}, a task type, a domain or a sub-category"
            )
        return [row for row in rows if keys(row) & wanted]

    def _ensure_videos(
        self, rows: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Fetch the missing videos of this run out of the dataset's zips."""
        by_name, by_stem = index_videos(self.video_dir)
        wanted = sorted(
            {
                str(row["videoID"])
                for row in rows
                if str(row["videoID"]) not in by_stem
            }
        )
        if not wanted:
            return by_name, by_stem
        if not self.auto_download:
            raise FileNotFoundError(
                f"{len(wanted)} of this run's videos are not under {self.video_dir} "
                "and VIDEOMME_AUTO_DOWNLOAD is off. Put them there as <videoID>.mp4, "
                "or leave the download on."
            )

        archives = list_archives()
        print(
            f"{len(wanted)} videos are missing from {self.video_dir}. Locating them "
            f"in the dataset's {len(archives)} archives (listed lazily over HTTP "
            "range reads, only the wanted members are fetched)."
        )
        index = RemoteZipIndex(DATASET_PATH, archives, os.path.join(self.video_dir, INDEX_FILE))
        located = index.locate([f"{video_id}.mp4" for video_id in wanted], folder="data")
        self.archives_listed = len(index._listed)
        missing = [video_id for video_id in wanted if f"{video_id}.mp4" not in located]
        if missing:
            print(
                f"  {len(missing)} videos are in none of the archives and are "
                f"skipped, e.g. {', '.join(missing[:3])}"
            )
        plan = {
            name: (zip_path, member) for name, (zip_path, member, _) in located.items()
        }
        if plan:
            fetched = index.fetch(plan, self.video_dir)
            print(f"  fetched {fetched} videos into {self.video_dir}")
        return index_videos(self.video_dir)

    # -- scoring ------------------------------------------------------------
    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        return parse_choice(output, 4)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Not scored: this is a decoding benchmark (the letters are in the dump)."""
        return None

    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, BenchmarkMetrics]]:
        """The metrics of every duration group."""
        if not self.categories:
            return None
        performance = {}
        for group in DURATIONS:
            indexes = [
                index
                for index, name in enumerate(self.categories)
                if name == group and index < len(states)
            ]
            if indexes:
                performance[f"duration_{group}"] = compute_metrics(
                    [states[index] for index in indexes], latency, answer_key=answer_key
                )
        return performance

    def describe_run(self) -> Optional[Dict[str, Any]]:
        description: Dict[str, Any] = {
            "num_frames": self.num_frames,
            "post_prompt": self.post_prompt,
            "questions": len(self.categories),
            "questions_per_duration": dict(Counter(self.categories)),
            "questions_per_task_type": dict(sorted(Counter(self.l2_categories).items())),
            "question_ids": self.question_ids,
            "video_dir": self.video_dir,
            "frames_dir": self.frames_dir,
            "archives_listed": self.archives_listed,
            "scoring": "none, the letters are extracted into the generations dump only",
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
            function_name="get_videomme_answer",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
            assistant_prefix=self.assistant_prefix,
        )
