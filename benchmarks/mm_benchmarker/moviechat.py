"""
MovieChat-1K evaluation script, ported from lmms-eval's `moviechat_global`.

MovieChat-1K (https://huggingface.co/datasets/Enxin/lmms_MovieChat_test) asks
open-ended questions about long movie clips; the test release used here holds
170 videos with three "global" questions each (510 rows) plus a "breakpoint"
mode that asks about one timestamp. Only the global mode is implemented: it is
the one whose questions read the whole video, which is what makes the prefill
heavy.

The prompt is lmms-eval's: its instruction-style pre-prompt ("You are able to
understand the visual content ... explain your answers in detail."), the
question, and an empty post-prompt. The pre-prompt and the question are joined
with a newline here for readability; `MOVIECHAT_PRE_PROMPT` and
`MOVIECHAT_POST_PROMPT` override either.

What is dropped is the scoring: lmms-eval grades every answer with
`gpt-3.5-turbo`, which says nothing about speculative decoding, so no judge is
called and no accuracy is reported. The reference answers still travel with the
run, so `--save-generations` writes a file a judge could be run over offline.

The videos ship as one 17 GiB zip. It is downloaded once (resumably, into the
Hugging Face cache) and only the videos a run needs are extracted from it;
`MOVIECHAT_VIDEO_DIR` points at an existing copy and `MOVIECHAT_AUTO_DOWNLOAD=0`
refuses to download. Decoded frames are cached under `MOVIECHAT_FRAMES_DIR`.
"""

import os
import zipfile
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from huggingface_hub import hf_hub_download

from .base import MMBenchmarker
from .registry import MM_BENCHMARKS
from .utils import create_interleaved_sgl_function, strip_reasoning
from .video_utils import (
    VIDEO_SUFFIXES,
    default_video_dir,
    env_flag,
    index_videos,
    materialize_frames,
)

DATASET_PATH = "Enxin/lmms_MovieChat_test"
MODE_FILES = {
    "global": "moviechat_global/test-00000-of-00001.parquet",
    "breakpoint": "moviechat_breakpoint/test-00000-of-00001.parquet",
}
DEFAULT_MODE = "global"
ARCHIVE = "videos_chunked_01.zip"
DEFAULT_VIDEO_SUBDIR = "moviechat_videos"
DEFAULT_NUM_FRAMES = 16
DEFAULT_MAX_NEW_TOKENS = 512

# lmms-eval's prompts, verbatim
DEFAULT_PRE_PROMPT = (
    "You are able to understand the visual content that the user provides."
    "Follow the instructions carefully and explain your answers in detail."
)
DEFAULT_POST_PROMPT = ""


def load_mode_rows(mode: str) -> List[Dict[str, Any]]:
    """The questions of one mode, read from its parquet alone (never the zip)."""
    import pyarrow.parquet as pq

    if mode not in MODE_FILES:
        raise ValueError(
            f"Unknown MovieChat mode '{mode}', expected one of {sorted(MODE_FILES)}"
        )
    path = hf_hub_download(DATASET_PATH, MODE_FILES[mode], repo_type="dataset")
    return pq.read_table(path).to_pylist()


def format_question(question: str, pre_prompt: str, post_prompt: str) -> str:
    """Instruction, question, optional post-prompt."""
    text = str(question).strip()
    if pre_prompt:
        text = f"{pre_prompt}\n{text}"
    if post_prompt:
        text = f"{text}\n{post_prompt}"
    return text


@MM_BENCHMARKS.register("moviechat")
class MovieChatBenchmarker(MMBenchmarker):
    """
    MovieChat-1K global mode, as a decoding benchmark.

    Args:
        num_samples: number of questions to ask, all 510 when not given. The
            rows come three per video, so `moviechat:30` is ten videos.
        subset: restrict to given video names or stems, e.g. `moviechat:9:1,2,3`.
        mode: "global" (the default); "breakpoint" is not ported.
        num_frames: how many evenly spaced frames a video is sent as.
        video_dir: directory holding the videos. Defaults to
            `$HF_HOME/moviechat_videos`, which is also where the zip is
            unpacked to.
        auto_download: whether the zip may be fetched to unpack missing videos.
        frames_dir: where the decoded frames are cached and kept.
        pre_prompt, post_prompt: the text around the question.

    Every argument except `num_samples` and `subset` also reads a `MOVIECHAT_`
    environment variable of the same name.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        mode: Optional[str] = None,
        num_frames: Optional[int] = None,
        video_dir: Optional[str] = None,
        auto_download: Optional[bool] = None,
        frames_dir: Optional[str] = None,
        pre_prompt: Optional[str] = None,
        post_prompt: Optional[str] = None,
    ):
        super().__init__(num_samples, subset)
        self.mode = str(mode or os.environ.get("MOVIECHAT_MODE") or DEFAULT_MODE).lower()
        if self.mode not in MODE_FILES:
            raise ValueError(
                f"Unknown MovieChat mode '{self.mode}', expected one of {sorted(MODE_FILES)}"
            )
        if self.mode != "global":
            raise NotImplementedError(
                "MovieChat breakpoint mode asks about one timestamp and needs a "
                "time-aware frame sampler; only the global mode is ported"
            )
        self.num_frames = int(
            num_frames or os.environ.get("MOVIECHAT_NUM_FRAMES") or DEFAULT_NUM_FRAMES
        )
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be at least 1, got {self.num_frames}")
        self.video_dir = default_video_dir(
            DEFAULT_VIDEO_SUBDIR, video_dir or os.environ.get("MOVIECHAT_VIDEO_DIR")
        )
        self.auto_download = (
            auto_download
            if auto_download is not None
            else env_flag("MOVIECHAT_AUTO_DOWNLOAD", True)
        )
        self.frames_dir = (
            frames_dir
            or os.environ.get("MOVIECHAT_FRAMES_DIR")
            or os.path.join(".cache", "moviechat_frames_specforge")
        )
        self.pre_prompt = (
            pre_prompt
            if pre_prompt is not None
            else os.environ.get("MOVIECHAT_PRE_PROMPT", DEFAULT_PRE_PROMPT)
        )
        self.post_prompt = (
            post_prompt
            if post_prompt is not None
            else os.environ.get("MOVIECHAT_POST_PROMPT", DEFAULT_POST_PROMPT)
        )

        self.categories: List[str] = []
        self.reference_lengths: List[int] = []
        self.missing_videos: List[str] = []

    def default_max_new_tokens(self) -> int:
        return DEFAULT_MAX_NEW_TOKENS

    # -- data ---------------------------------------------------------------
    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        rows = load_mode_rows(self.mode)
        if self.subset:
            rows = self._select_subset(rows)
        if self.num_samples is not None:
            rows = rows[: self.num_samples]
        if not rows:
            raise ValueError("No MovieChat question left after filtering")

        os.makedirs(self.video_dir, exist_ok=True)
        by_name, by_stem = self._ensure_videos(rows)

        questions: List[Dict[str, Any]] = []
        labels: List[Optional[str]] = []
        self.categories, self.reference_lengths, self.missing_videos = [], [], []
        for row in rows:
            name = str(row["video_name"])
            video_path = self._find_video(name, by_name, by_stem)
            frames = (
                materialize_frames(
                    self.frames_dir, video_path, os.path.splitext(name)[0], self.num_frames
                )
                if video_path
                else []
            )
            if not frames:
                self.missing_videos.append(name)
                continue
            text = format_question(row["question"], self.pre_prompt, self.post_prompt)
            parts = [("image", path) for path in frames] + [("text", text)]
            questions.append({"parts": parts, "video_name": name})
            reference = str(row.get("answer") or "")
            labels.append(reference or None)
            self.reference_lengths.append(len(reference))
            self.categories.append(os.path.splitext(name)[0])

        if self.missing_videos:
            print(
                f"Skipped {len(self.missing_videos)} questions whose video is "
                f"missing or undecodable, e.g. {', '.join(self.missing_videos[:3])}"
            )
        print(
            f"Loaded {len(questions)} MovieChat {self.mode} questions over "
            f"{len(set(self.categories))} videos as {self.num_frames} frames each. "
            f"Frames cached in {self.frames_dir}"
        )
        return questions, labels

    def _select_subset(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        wanted = {str(name).strip().lower() for name in self.subset}

        def keys(row):
            name = str(row["video_name"]).lower()
            return {name, os.path.splitext(name)[0]}

        kept = [row for row in rows if keys(row) & wanted]
        if not kept:
            raise ValueError(
                f"No MovieChat video matches subset {sorted(wanted)}; the videos "
                "are named like 1.mp4, 2.mp4, ..."
            )
        return kept

    @staticmethod
    def _find_video(
        name: str, by_name: Dict[str, str], by_stem: Dict[str, str]
    ) -> Optional[str]:
        if name in by_name:
            return by_name[name]
        return by_stem.get(os.path.splitext(name)[0])

    def _ensure_videos(
        self, rows: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Unpack the missing videos of this run out of the dataset's zip."""
        by_name, by_stem = index_videos(self.video_dir)
        wanted: Set[str] = {
            str(row["video_name"])
            for row in rows
            if not self._find_video(str(row["video_name"]), by_name, by_stem)
        }
        if not wanted:
            return by_name, by_stem
        if not self.auto_download:
            raise FileNotFoundError(
                f"{len(wanted)} of this run's videos are not under {self.video_dir} "
                "and MOVIECHAT_AUTO_DOWNLOAD is off. Unpack the dataset's "
                f"{ARCHIVE} there, or leave the download on."
            )

        print(
            f"{len(wanted)} videos are missing from {self.video_dir}. Fetching "
            f"{ARCHIVE} (~17 GiB, resumable, cached under HF_HOME) and unpacking "
            "only the wanted videos."
        )
        archive_path = hf_hub_download(DATASET_PATH, filename=ARCHIVE, repo_type="dataset")
        wanted_stems = {os.path.splitext(name)[0]: name for name in wanted}
        extracted = 0
        unmatched: List[str] = []
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = os.path.basename(info.filename)
                stem, suffix = os.path.splitext(name)
                if suffix.lower() not in VIDEO_SUFFIXES:
                    continue
                row_name = name if name in wanted else wanted_stems.get(stem)
                if row_name is None:
                    if len(unmatched) < 5:
                        unmatched.append(name)
                    continue
                target = os.path.join(self.video_dir, name)
                with archive.open(info) as source, open(target + ".part", "wb") as sink:
                    while True:
                        chunk = source.read(1 << 20)
                        if not chunk:
                            break
                        sink.write(chunk)
                os.replace(target + ".part", target)
                wanted.discard(row_name)
                extracted += 1
                if not wanted:
                    break
        print(f"  unpacked {extracted} videos, {len(wanted)} still missing")
        if wanted and unmatched:
            print(
                f"  the archive holds files named like {', '.join(unmatched)}; "
                "the missing ones are skipped and counted in the run description"
            )
        return index_videos(self.video_dir)

    # -- scoring ------------------------------------------------------------
    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """The answer itself; only a reasoning block is dropped."""
        if not isinstance(output, str):
            return None
        return strip_reasoning(output).strip() or None

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Not scored here: lmms-eval grades with a GPT judge, which is skipped."""
        return None

    def describe_run(self) -> Optional[Dict[str, Any]]:
        description: Dict[str, Any] = {
            "mode": self.mode,
            "num_frames": self.num_frames,
            "questions": len(self.categories),
            "videos": len(set(self.categories)),
            "questions_per_video": dict(sorted(Counter(self.categories).items())),
            "video_dir": self.video_dir,
            "frames_dir": self.frames_dir,
            "scoring": "none, the GPT judge of the original task is deliberately not run",
        }
        if self.missing_videos:
            description["missing_videos"] = len(self.missing_videos)
            description["missing_videos_sample"] = self.missing_videos[:10]
        if self.reference_lengths:
            description["reference_chars_mean"] = sum(self.reference_lengths) / len(
                self.reference_lengths
            )
        generations = getattr(self, "generations", None)
        if generations:
            lengths = [len(text) for text in generations if isinstance(text, str)]
            if lengths:
                description["generated_chars_mean"] = sum(lengths) / len(lengths)
        return description

    def create_sgl_function(self):
        return create_interleaved_sgl_function(
            function_name="get_moviechat_answer",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
            assistant_prefix=self.assistant_prefix,
        )
