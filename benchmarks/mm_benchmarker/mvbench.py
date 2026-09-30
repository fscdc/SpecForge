"""
MVBench evaluation script, ported from lmms-eval's `mvbench` task group.

MVBench (https://huggingface.co/datasets/OpenGVLab/MVBench) is 20 temporal
understanding tasks of 200 multiple-choice questions each, over short clips
drawn from Charades (STAR), CLEVRER, Perception Test, Something-Something v2,
Moments in Time, FunQA, VLN-CE, TVQA and NTU RGB+D. Two tasks are not ported:
`fine_grained_pose` (NTU RGB+D must be requested by hand) and
`episodic_reasoning` (TVQA ships as folders of 3 fps frames, not videos). The
other 18 are, and a run spreads its questions evenly over them, one task after
the other, so `mvbench:20` asks one question of each task and a second of two.

What this port keeps from lmms-eval is the prompt shape: "Question:", the
question, "Option:", the candidates as "(A) ...", and a post-prompt; no system
prompt. What it changes is that post-prompt. The original ("Only give the best
option.") yields a one-token answer that tells a speculative decoder nothing;
the default here asks the model to reason about the video first and end with
the letter. The original wording is still available as `mvbench-origin`. As in
lmms-eval, a question that names a time span of its clip (`start`/`end`, the
STAR and STA tasks) is answered from frames of that span alone.

What it drops is the scoring: the letter each answer settles on is extracted
for the generations dump, but no accuracy is reported.

The clips are small and low-resolution: Charades at 480x270, CLEVRER at
480x320, i.e. 120 to 150 tokens a frame for Qwen3.5. At native size not even
64 frames reach the ~40k-token prompts of the other video benchmarks, so by
default every frame is upscaled to the area of a 1280x720 frame (aspect kept)
before it is sent, which costs the same ~880 tokens as a real 720p frame, 48
frames = ~40k tokens. `MVBENCH_FRAME_PIXELS=0` keeps the native size; it is the
option to use when the question is about model accuracy rather than the draft
on long prompts. `MVBENCH_FRAME_PIXELS` also takes a `WxH`.

The videos ship as one zip per source under `video/` (17 GiB in all). None is
downloaded whole: a zip's central directory is listed over HTTP range reads
(a few seconds, cached as JSON afterwards) and only the wanted members are
streamed out, into `MVBENCH_VIDEO_DIR/<source>/`. `MVBENCH_AUTO_DOWNLOAD=0`
refuses to download. Decoded frames are cached under `MVBENCH_FRAMES_DIR`.
"""

import json
import os
import re
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from benchmarker.utils import BenchmarkMetrics, compute_metrics
from huggingface_hub import hf_hub_download

from .base import MMBenchmarker
from .registry import MM_BENCHMARKS
from .utils import create_interleaved_sgl_function, strip_reasoning
from .video_utils import RemoteZipIndex, default_video_dir, env_flag, materialize_frames

DATASET_PATH = "OpenGVLab/MVBench"
DEFAULT_VIDEO_SUBDIR = "mvbench_videos"
INDEX_FILE = "zip_index.json"
DEFAULT_NUM_FRAMES = 48
#: frames are rescaled to this many pixels before they are sent (0 = native)
DEFAULT_FRAME_PIXELS = 1280 * 720
DEFAULT_MAX_NEW_TOKENS = 1024
OPTION_LETTERS = "ABCDE"

# lmms-eval's post-prompt, verbatim: a one-token answer
ORIGINAL_POST_PROMPT = "Only give the best option.\n"
# the default here: reason first, so that there is a generation to measure
DEFAULT_POST_PROMPT = (
    "Think through what happens in the video step by step first, then give your "
    "final answer as the option's letter.\n"
)

#: task -> (source zip under video/, member prefix inside the zip), in
#: lmms-eval's DATA_LIST order. A question's member is prefix + its "video".
TASK_SOURCES: "OrderedDict[str, Tuple[str, str]]" = OrderedDict(
    [
        ("action_sequence", ("star", "star/Charades_v1_480/")),
        ("action_prediction", ("star", "star/Charades_v1_480/")),
        ("action_antonym", ("ssv2_video", "ssv2_video/")),
        ("fine_grained_action", ("Moments_in_Time_Raw", "Moments_in_Time_Raw/videos/")),
        ("unexpected_action", ("FunQA_test", "FunQA_test/test/")),
        ("object_existence", ("clevrer", "clevrer/video_validation/")),
        ("object_interaction", ("star", "star/Charades_v1_480/")),
        ("object_shuffle", ("perception", "perception/videos/")),
        ("moving_direction", ("clevrer", "clevrer/video_validation/")),
        ("action_localization", ("sta", "sta/sta_video/")),
        ("scene_transition", ("scene_qa", "scene_qa/video/")),
        ("action_count", ("perception", "perception/videos/")),
        ("moving_count", ("clevrer", "clevrer/video_validation/")),
        ("moving_attribute", ("clevrer", "clevrer/video_validation/")),
        ("state_change", ("perception", "perception/videos/")),
        ("character_order", ("perception", "perception/videos/")),
        ("egocentric_navigation", ("vlnqa", "vlnqa/")),
        ("counterfactual_inference", ("clevrer", "clevrer/video_validation/")),
    ]
)
#: the tasks of the original benchmark that this port leaves out, and why
UNPORTED_TASKS = {
    "fine_grained_pose": "NTU RGB+D videos must be requested by hand",
    "episodic_reasoning": "TVQA ships as folders of 3 fps frames, not videos",
}
#: a second zip holding a few extra Charades and CLEVRER clips
EXTRA_ZIPS = {"star": ["data0613"], "clevrer": ["data0613"]}

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


def parse_pixels(value: Any) -> int:
    """A pixel count, given as an integer or as `WxH`; 0 means native size."""
    if isinstance(value, str) and "x" in value.lower():
        width, height = value.lower().split("x", 1)
        return int(width) * int(height)
    return int(value or 0)


def load_task_rows(task: str) -> List[Dict[str, Any]]:
    """The 200 questions of one task, each tagged with its task name."""
    path = hf_hub_download(DATASET_PATH, f"json/{task}.json", repo_type="dataset")
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    for row in rows:
        row["task"] = task
    return rows


def format_question(row: Dict[str, Any], post_prompt: str) -> str:
    """lmms-eval's `mvbench_doc_to_text`, with the post-prompt as given."""
    options = "".join(
        f"({OPTION_LETTERS[index]}) {candidate}\n"
        for index, candidate in enumerate(row["candidates"])
    )
    return "Question:" + str(row["question"]) + "\nOption:\n" + options + post_prompt


def answer_letter(row: Dict[str, Any]) -> Optional[str]:
    """The letter of the candidate that matches the reference answer."""
    for index, candidate in enumerate(row.get("candidates") or []):
        if str(candidate) == str(row.get("answer")):
            return OPTION_LETTERS[index]
    return None


def parse_choice(response: str, num_options: int) -> Optional[str]:
    """
    The option letter a response settles on, or None.

    A reasoning answer mentions letters along the way and puts its choice last,
    so the last standalone letter wins, after the usual "The answer is"
    prefixes are stripped; "(B)" counts as B. No random fallback.
    """
    if not isinstance(response, str):
        return None
    text = strip_reasoning(response)
    for prefix in ANSWER_PREFIXES:
        text = text.replace(prefix, "")
    letters = OPTION_LETTERS[: max(1, min(num_options, len(OPTION_LETTERS)))]
    matches = re.findall(rf"(?<![A-Za-z])([{letters}])(?![A-Za-z])", text)
    return matches[-1] if matches else None


def spread_over_tasks(
    rows_by_task: "OrderedDict[str, List[Dict[str, Any]]]", num_samples: Optional[int]
) -> List[Dict[str, Any]]:
    """
    The first question of every task, then the second of every task, and so on.

    The order is fixed by the task order, so the first N of the result are
    always the same N questions, which is what a resumable, comparable run
    needs. `None` yields every question, task after task in that same order.
    """
    if num_samples is None:
        return [row for rows in rows_by_task.values() for row in rows]
    selected: List[Dict[str, Any]] = []
    depth = 0
    while len(selected) < num_samples:
        progressed = False
        for rows in rows_by_task.values():
            if depth < len(rows):
                selected.append(rows[depth])
                progressed = True
                if len(selected) >= num_samples:
                    break
        if not progressed:
            break
        depth += 1
    return selected


@MM_BENCHMARKS.register("mvbench")
class MVBenchBenchmarker(MMBenchmarker):
    """
    MVBench, as a decoding benchmark.

    Args:
        num_samples: number of questions to ask, spread evenly over the tasks
            (see `spread_over_tasks`); all 3,600 ported questions when not given.
        subset: restrict to one or more task names, e.g.
            `mvbench:40:action_count,moving_count`.
        num_frames: how many evenly spaced frames a clip (or the question's
            span of it) is sent as.
        frame_pixels: the area (pixels, or "WxH") every frame is rescaled to
            before it is sent; 0 keeps the native size. See the module docstring.
        video_dir: directory holding the videos as `<source>/<file>`. Defaults
            to `$HF_HOME/mvbench_videos`, which is also where fetched videos
            are written.
        auto_download: whether missing videos may be fetched out of the zips.
        frames_dir: where the decoded frames are cached and kept.
        post_prompt: the line after the options; see the module docstring for
            why the default asks for reasoning.

    Every argument except `num_samples` and `subset` also reads an `MVBENCH_`
    environment variable of the same name.
    """

    ORIGINAL_PROMPT_KWARGS = {"post_prompt": ORIGINAL_POST_PROMPT}

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        num_frames: Optional[int] = None,
        frame_pixels: Optional[Any] = None,
        video_dir: Optional[str] = None,
        auto_download: Optional[bool] = None,
        frames_dir: Optional[str] = None,
        post_prompt: Optional[str] = None,
    ):
        super().__init__(num_samples, subset)
        self.num_frames = int(
            num_frames or os.environ.get("MVBENCH_NUM_FRAMES") or DEFAULT_NUM_FRAMES
        )
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be at least 1, got {self.num_frames}")
        self.frame_pixels = parse_pixels(
            frame_pixels
            if frame_pixels is not None
            else os.environ.get("MVBENCH_FRAME_PIXELS", DEFAULT_FRAME_PIXELS)
        )
        self.video_dir = default_video_dir(
            DEFAULT_VIDEO_SUBDIR, video_dir or os.environ.get("MVBENCH_VIDEO_DIR")
        )
        self.auto_download = (
            auto_download
            if auto_download is not None
            else env_flag("MVBENCH_AUTO_DOWNLOAD", True)
        )
        self.frames_dir = (
            frames_dir
            or os.environ.get("MVBENCH_FRAMES_DIR")
            or os.path.join(".cache", "mvbench_frames_specforge")
        )
        self.post_prompt = (
            post_prompt
            if post_prompt is not None
            else os.environ.get("MVBENCH_POST_PROMPT", DEFAULT_POST_PROMPT)
        )

        # per-question metadata, aligned with the loaded questions
        self.categories: List[str] = []
        self.l2_categories: List[str] = []
        self.num_options: List[int] = []
        self.missing_videos: List[str] = []

    def default_max_new_tokens(self) -> int:
        return DEFAULT_MAX_NEW_TOKENS

    # -- data ---------------------------------------------------------------
    def tasks(self) -> List[str]:
        """The tasks of this run, in benchmark order."""
        if not self.subset:
            return list(TASK_SOURCES)
        wanted = {str(name).strip().lower() for name in self.subset}
        unported = wanted & set(UNPORTED_TASKS)
        if unported:
            raise ValueError(
                "MVBench task(s) not ported: "
                + "; ".join(f"{task} ({UNPORTED_TASKS[task]})" for task in sorted(unported))
            )
        unknown = wanted - set(TASK_SOURCES)
        if unknown:
            raise ValueError(
                f"Unknown MVBench task(s) {sorted(unknown)}; expected some of "
                f"{list(TASK_SOURCES)}"
            )
        return [task for task in TASK_SOURCES if task in wanted]

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        rows_by_task: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict(
            (task, load_task_rows(task)) for task in self.tasks()
        )
        rows = spread_over_tasks(rows_by_task, self.num_samples)
        if not rows:
            raise ValueError("No MVBench question left after filtering")

        os.makedirs(self.video_dir, exist_ok=True)
        self._ensure_videos(rows)

        questions: List[Dict[str, Any]] = []
        labels: List[Optional[str]] = []
        self.categories, self.l2_categories, self.num_options = [], [], []
        self.missing_videos = []
        for row in rows:
            source, _ = TASK_SOURCES[row["task"]]
            video_path = self._local_path(row)
            frames = (
                materialize_frames(
                    self.frames_dir,
                    video_path,
                    f"{source}__{os.path.splitext(os.path.basename(str(row['video'])))[0]}",
                    self.num_frames,
                    start=row.get("start"),
                    end=row.get("end"),
                    target_pixels=self.frame_pixels or None,
                )
                if video_path
                else []
            )
            if not frames:
                self.missing_videos.append(f"{row['task']}/{row['video']}")
                continue
            text = format_question(row, self.post_prompt)
            parts = [("image", path) for path in frames] + [("text", text)]
            questions.append({"parts": parts, "task": row["task"], "video": str(row["video"])})
            labels.append(answer_letter(row))
            self.categories.append(str(row["task"]))
            self.l2_categories.append(source)
            self.num_options.append(len(row.get("candidates") or []))

        if self.missing_videos:
            print(
                f"Skipped {len(self.missing_videos)} questions whose video is "
                f"missing or undecodable, e.g. {', '.join(self.missing_videos[:3])}"
            )
        print(
            f"Loaded {len(questions)} MVBench questions over {len(set(self.categories))} "
            f"tasks as {self.num_frames} frames each"
            + (f", rescaled to {self.frame_pixels} pixels each" if self.frame_pixels else "")
            + f". Frames cached in {self.frames_dir}"
        )
        return questions, labels

    def _local_path(self, row: Dict[str, Any]) -> Optional[str]:
        """Where the clip of a question lives under `video_dir`, if it does."""
        source, _ = TASK_SOURCES[row["task"]]
        name = os.path.basename(str(row["video"]))
        path = os.path.join(self.video_dir, source, name)
        if os.path.exists(path):
            return path
        stem = os.path.splitext(name)[0]
        folder = os.path.join(self.video_dir, source)
        if os.path.isdir(folder):
            for candidate in os.listdir(folder):
                if os.path.splitext(candidate)[0] == stem and not candidate.endswith(".part"):
                    return os.path.join(folder, candidate)
        return None

    def _ensure_videos(self, rows: List[Dict[str, Any]]) -> None:
        """Fetch the missing clips of this run out of their source zips."""
        missing = [row for row in rows if self._local_path(row) is None]
        if not missing:
            return
        if not self.auto_download:
            raise FileNotFoundError(
                f"{len(missing)} of this run's videos are not under {self.video_dir} "
                "and MVBENCH_AUTO_DOWNLOAD is off. Put them there as <source>/<file>, "
                "or leave the download on."
            )
        sources = sorted({TASK_SOURCES[row["task"]][0] for row in missing})
        zips = []
        for source in sources:
            for name in [source] + EXTRA_ZIPS.get(source, []):
                if f"video/{name}.zip" not in zips:
                    zips.append(f"video/{name}.zip")
        print(
            f"{len(missing)} videos are missing from {self.video_dir}. Fetching them "
            f"out of {', '.join(zips)} (central directories listed over HTTP range "
            "reads, only the wanted members are streamed)."
        )
        index = RemoteZipIndex(DATASET_PATH, zips, os.path.join(self.video_dir, INDEX_FILE))
        index.load()
        plan: Dict[str, Tuple[str, str]] = {}
        not_found: List[str] = []
        for row in missing:
            source, prefix = TASK_SOURCES[row["task"]]
            video = str(row["video"])
            expected = prefix + video
            entry = None
            for candidate in index.members.get(os.path.basename(video), []):
                if candidate[1] == expected:
                    entry = candidate
                    break
            if entry is None:
                entry = index.find(video, folder=os.path.dirname(expected))
            if entry is None:
                not_found.append(f"{row['task']}/{video}")
                continue
            target = os.path.join(source, os.path.basename(entry[1]))
            plan[target] = (entry[0], entry[1])
        if not_found:
            print(
                f"  {len(not_found)} videos are in none of the archives and are "
                f"skipped, e.g. {', '.join(not_found[:3])}"
            )
        if plan:
            fetched = index.fetch(plan, self.video_dir)
            print(f"  fetched {fetched} videos into {self.video_dir}")

    # -- scoring ------------------------------------------------------------
    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        return parse_choice(output, 5)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Not scored: this is a decoding benchmark (the letters are in the dump)."""
        return None

    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, BenchmarkMetrics]]:
        """The metrics of every video source (the axis the clip format follows)."""
        if not self.l2_categories:
            return None
        performance = {}
        for source in sorted(set(self.l2_categories)):
            indexes = [
                index
                for index, name in enumerate(self.l2_categories)
                if name == source and index < len(states)
            ]
            if indexes:
                performance[f"source_{source}"] = compute_metrics(
                    [states[index] for index in indexes], latency, answer_key=answer_key
                )
        return performance

    def describe_run(self) -> Optional[Dict[str, Any]]:
        description: Dict[str, Any] = {
            "num_frames": self.num_frames,
            "frame_pixels": self.frame_pixels or "native",
            "post_prompt": self.post_prompt,
            "questions": len(self.categories),
            "questions_per_task": dict(Counter(self.categories)),
            "questions_per_source": dict(sorted(Counter(self.l2_categories).items())),
            "unported_tasks": UNPORTED_TASKS,
            "video_dir": self.video_dir,
            "frames_dir": self.frames_dir,
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
            function_name="get_mvbench_answer",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
            assistant_prefix=self.assistant_prefix,
        )
