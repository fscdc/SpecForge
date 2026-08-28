"""
VDC (Video Detailed Captioning) benchmark evaluation script.

VDC (https://huggingface.co/datasets/wchai/Video-Detailed-Caption) is AuroraCap's
captioning benchmark: ~1k videos drawn from Panda-70M, Ego4D, Mixkit, Pixabay and
Pexels, each annotated with five structured captions -- detailed, camera, short,
background and main object. The prompts below are the ones of the lmms-eval
`detailed_test` task and its four siblings, verbatim; one of them is drawn per
video.

What this port deliberately drops is the scoring. The lmms-eval task reports
VDCScore, which needs a second LLM on localhost:30000 to first answer the
reference QA pairs off the generated caption and then grade each answer -- two
judge calls per QA pair, ~40 per video, ~200k for a full run. None of that says
anything about speculative decoding, so no judge is involved here and no accuracy
is reported. What is kept is everything the draft model is measured by: latency,
output throughput and accept length, over a workload whose generations are long
(a detailed caption runs to ~1k tokens) and whose prefill is heavy (one video
enters as `num_frames` images). That combination is what makes VDC worth running
as a speculative-decoding benchmark at all.

The reference captions still travel with the run, so `--save-generations` writes
a file a VDCScore pass could be run over offline.

Two things the other multimodal benchmarks do not need:

- **The videos are not in the parquet.** The dataset ships them as three
  tarballs under `videos/`, 15.6 + 24.8 + 34.2 GiB. They are downloaded on
  demand: the captions are loaded first, so only the archives that still carry a
  missing video are pulled, and only the wanted members are unpacked out of
  them. A `vdc:200` run therefore normally stops after the first archive. Point
  `VDC_VIDEO_DIR` at an existing copy to skip all of it, or
  `VDC_AUTO_DOWNLOAD=0` to refuse to download. Rows whose video is in no archive
  are skipped and counted in `describe_run()`.
- **The decoded frames are cached and kept.** Decoding a thousand videos is far
  more expensive than the PNG dumps the image benchmarks throw away after each
  run, so the frames live on under `VDC_FRAMES_DIR` and are reused.

The prompt is drawn per video from a seed rather than from the process-wide
`random` the lmms-eval task uses, so that two runs send byte-identical prompts
and their latencies stay comparable.
"""

import glob
import os
import random
import tarfile
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from benchmarker.utils import BenchmarkMetrics, compute_metrics
from datasets import load_dataset
from PIL import Image

from .base import MMBenchmarker
from .registry import MM_BENCHMARKS
from .utils import create_interleaved_sgl_function, strip_reasoning

DATASET_PATH = "wchai/Video-Detailed-Caption"
# the dataset has exactly one split, and it is not called "train" or "test"
SPLIT = "VDC_captions"

# the column each caption aspect is annotated in
CAPTION_COLUMNS = {
    "detailed": "detailed_caption",
    "camera": "camera_caption",
    "short": "short_caption",
    "background": "background_caption",
    "main_object": "main_object_caption",
}

# Generation budget per aspect, sized on the reference captions: a detailed one
# runs to ~3.9k characters (~1k tokens), the middle three to a few hundred, and
# the short one is a single sentence.
DEFAULT_MAX_NEW_TOKENS = {
    "detailed": 1024,
    "camera": 512,
    "background": 512,
    "main_object": 512,
    "short": 128,
}

# How many frames one video enters the prompt as. This is the single biggest
# knob on the prefill length, hence on the throughput, so it is reported in
# describe_run() and pinned rather than left to the model's own default.
DEFAULT_NUM_FRAMES = 16

# the column a --benchmark-list subset is matched against, e.g. `vdc:100:ego4d`
SUBSET_COLUMN = "video_source"

VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".avi", ".mov")

# Where the videos land when neither --video-dir nor VDC_VIDEO_DIR names a
# directory, next to the parquet the datasets library caches under the same root.
DEFAULT_VIDEO_SUBDIR = "vdc_videos"

DETAILED_CAPTION_PROMPTS = [
    "Please imagine the video based on the sequence of frames, and provide a faithfully detailed description of this video in more than three sentences.",
    "You are given a sequence of equally spaced video frames. Based on these frames, imagine the full video and provide a detailed description of what is happening in more than three sentences.",
    "The following set contains equally spaced video frames. Imagine the video from which these frames were taken and describe it in detail in at least three sentences.",
    "Below are equally spaced frames from a video. Use these frames to visualize the entire video and provide a detailed description in more than three sentences.",
    "A sequence of equally spaced video frames is presented. Please imagine the full video and write a faithfully detailed description of the events in more than three sentences.",
    "The images provided include equally spaced frames from a video. Based on these frames, imagine the video and describe it comprehensively in at least three sentences.",
    "You are given equally spaced frames from a video. Use these frames to envision the entire video and provide a detailed description of the events in more than three sentences.",
    "The sequence includes equally spaced frames from a video. Imagine the full video based on these frames and provide a detailed description in more than three sentences.",
    "The provided images contain equally spaced frames from a video. Visualize the video from these frames and describe it in detail in more than three sentences.",
    "Here are equally spaced frames from a video. Based on these frames, imagine the video and provide a detailed, faithful description of it in more than three sentences.",
    "The set of images includes equally spaced video frames. Please imagine the video these frames come from and describe it comprehensively in at least three sentences.",
    "Describe the video based on these frames in a few sentences.",
    "What is happening in the video shown in these frames?",
    "Explain the video using these frames.",
    "Imagine the video from these frames and describe it in detail in a few sentences.",
    "Based on these frames, provide a narrative of the video in more than three sentences.",
    "Describe the events in the video shown by these frames in at least three sentences.",
    "Visualize the video from these frames and explain what is happening in more than three sentences.",
    "Describe the sequence of events in the video depicted by these frames in a detailed manner.",
    "Given these equally spaced frames, imagine the entire video and provide a detailed description of the events, including the setting, characters, and actions, in more than three sentences.",
    "Visualize the video based on these frames and write a comprehensive description of what happens, describing the beginning, middle, and end in at least three sentences.",
    "Using these frames as a reference, imagine the full video and provide a thorough description of the plot, including key details and actions, in more than three sentences.",
    "Based on the sequence of these frames, describe the entire video in detail, mentioning important aspects such as the context, movements, and transitions in more than three sentences.",
    "Imagine the video that corresponds to these frames and provide an elaborate description, covering the storyline, visual elements, and any notable features in at least three sentences.",
]

CAMERA_CAPTION_PROMPTS = [
    "Summary of the view shot, camera movement and changes in shooting angles in the sequence of video frames.",
    "Describe the camera movements in these frames.",
    "What are the camera angles and movements throughout the video?",
    "Summarize the camera actions and perspectives.",
    "Describe any camera zooms, pans, or angle changes.",
    "What camera movements are present in these frames?",
    "Describe the camera's movements, including pans, zooms, and angle changes in these frames.",
    "Summarize the camera actions and changes in shooting angles during the video.",
    "Provide a detailed description of the camera's movements and perspectives.",
    "Describe the camera's actions and how it follows the main subject.",
    "What are the camera movements and angle shifts in these frames?",
    "Given these equally spaced frames, provide a comprehensive description of the camera's movements, including any pans, zooms, and changes in shooting angles.",
    "Describe the camera's movements and angles in detail, explaining how it follows the main subject and changes perspectives.",
    "Based on these frames, provide a detailed description of the camera's actions, including any pans, zooms, angle shifts, and how it captures the scene.",
    "Using these frames, describe the camera's movements, including its tracking of the main subject, changes in angles, and any zooms or pans.",
    "Provide an elaborate description of the camera movements, covering pans, zooms, and changes in shooting angles as shown in these frames.",
]

SHORT_CAPTION_PROMPTS = [
    "Write a one-sentence summary of the video.",
    "Summarize the video in one concise sentence.",
    "Provide a brief description of the video in one sentence.",
    "Describe the main action in the video in one sentence.",
    "What is the video about? Summarize it in one sentence.",
    "In one sentence, summarize the key visual elements of the video.",
    "Provide a one-sentence summary that captures the main subject and action in the video.",
    "Write a concise one-sentence description that encapsulates the essence of the video.",
    "Describe the main theme or action of the video in a single sentence.",
    "What is happening in the video? Provide a one-sentence summary.",
    "Given these frames, write a brief one-sentence summary that captures the essence of the video's visual and artistic style.",
    "Summarize the key visual and thematic elements of the video in one concise sentence.",
    "Provide a one-sentence description that highlights the main subject and action depicted in the video.",
    "In one sentence, describe the primary visual and artistic elements of the video.",
    "Write a concise one-sentence summary that encapsulates the main action and visual style of the video.",
    "Briefly one-sentence Summary of the visual, Photographic and artistic style.",
]

BACKGROUND_CAPTION_PROMPTS = [
    "The images are given containing equally spaced video frames.Summary of the background. This should also include the objects, location, weather, and time.",
    "Describe the background, including objects, location, weather, and time.",
    "Summarize the background setting of the video based on these frames.",
    "What is the environment like in these frames?",
    "Describe the location and weather in these frames.",
    "What background objects and settings are visible in these frames?",
    "Summarize the background of the video, including details about the location, objects, weather, and time.",
    "Describe the environment shown in these frames, covering objects, location, weather, and time.",
    "Provide a detailed background description based on these frames, mentioning objects, location, weather, and time.",
    "Explain the setting of the video, focusing on the background elements like objects, location, weather, and time.",
    "Describe the overall environment in these frames, including details about objects, location, weather, and time.",
    "Given these equally spaced frames, provide a comprehensive background description, covering the objects, location, weather, and time.",
    "Imagine the environment from these frames and write a detailed description of the background, including objects, location, weather, and time.",
    "Based on these frames, describe the setting in detail, mentioning the objects present, the specific location, the weather conditions, and the time of day.",
    "Provide an elaborate background description based on these frames, covering all aspects of the environment such as objects, location, weather, and time.",
    "Using these frames as a reference, give a thorough description of the background, including details about the objects, location, weather, and time.",
]

MAIN_OBJECT_CAPTION_PROMPTS = [
    "Description of the main subject actions or status sequence. This suggests including the main subjects (person, object, animal, or none) and their attributes, their action, their position, and movements during the video frames.",
    "Describe the main subject's actions and movements.",
    "What is the main object doing in these frames?",
    "Summarize the primary subject's attributes and actions.",
    "Describe the main subject's position and movements.",
    "What actions does the main object take in these frames?",
    "Describe the main subject, including their attributes and movements throughout the video.",
    "Provide a detailed description of the main object's actions and positions in these frames.",
    "Summarize the main subject's actions, attributes, and movements during the video.",
    "Describe the primary subject's movements and actions in detail.",
    "What are the main object's attributes and how do they move throughout the video?",
    "Given these equally spaced frames, provide a comprehensive description of the main subject, including their attributes, actions, positions, and movements.",
    "Describe the primary object or subject in the video, detailing their attributes, actions, positions, and movements in these frames.",
    "Based on these frames, provide a detailed description of the main subject, including their attributes, actions, positions, and how they navigate through the video.",
    "Using these frames, describe the main subject's attributes, actions, and movements, detailing their positions and how they interact with the environment.",
    "Provide an elaborate description of the main object in the video, covering their attributes, actions, positions, and movements as shown in these frames.",
]

# the prompt list each aspect draws from
CAPTION_PROMPTS = {
    "detailed": DETAILED_CAPTION_PROMPTS,
    "camera": CAMERA_CAPTION_PROMPTS,
    "short": SHORT_CAPTION_PROMPTS,
    "background": BACKGROUND_CAPTION_PROMPTS,
    "main_object": MAIN_OBJECT_CAPTION_PROMPTS,
}


def frame_indices(total_frames: int, num_frames: int) -> List[int]:
    """`num_frames` evenly spaced frame indices, both endpoints included.

    A video with fewer frames than asked for contributes all of them.
    """
    if total_frames <= 0:
        raise ValueError("the video reports no frames")
    count = min(num_frames, total_frames)
    if count == 1:
        return [0]
    step = (total_frames - 1) / (count - 1)
    return [round(index * step) for index in range(count)]


def _read_frames_decord(video_path: str, num_frames: int) -> List[Image.Image]:
    """Read the sampled frames with decord, which indexes them directly."""
    import decord

    reader = decord.VideoReader(video_path, num_threads=1)
    batch = reader.get_batch(frame_indices(len(reader), num_frames)).asnumpy()
    return [Image.fromarray(frame) for frame in batch]


def _read_frames_cv2(video_path: str, num_frames: int) -> List[Image.Image]:
    """
    Read the sampled frames with OpenCV.

    The file is scanned sequentially rather than seeked through with
    CAP_PROP_POS_FRAMES, which lands on the wrong frame for several codecs. The
    scan stops as soon as the last wanted frame has been read.
    """
    import cv2

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError(f"could not open video file {video_path}")
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise IOError(
                f"OpenCV cannot tell how many frames {video_path} has, which is "
                "what the even sampling is computed from. Install decord "
                "(pip install decord) to read this file."
            )
        wanted = frame_indices(total_frames, num_frames)
        remaining = set(wanted)
        frames: Dict[int, Image.Image] = {}
        index = 0
        while remaining:
            ok, frame = capture.read()
            if not ok:
                break
            if index in remaining:
                remaining.discard(index)
                frames[index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index += 1
        if not frames:
            raise IOError(f"no frame of {video_path} could be decoded")
        return [frames[index] for index in wanted if index in frames]
    finally:
        capture.release()


def read_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    """The sampled frames of one video, through whichever decoder is installed."""
    try:
        return _read_frames_decord(video_path, num_frames)
    except ImportError:
        pass
    try:
        return _read_frames_cv2(video_path, num_frames)
    except ImportError:
        raise ImportError(
            "reading VDC's videos needs a video decoder, and neither decord nor "
            "OpenCV is installed. Install one of them: `pip install decord` "
            "(fastest, indexes frames directly) or "
            "`pip install opencv-python-headless`."
        ) from None


def index_videos(video_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Map the video files under `video_dir` by file name and by stem.

    The tarballs unpack into a layout of their own, and a row names its video by
    file name while some copies of the corpus name it by id, so the tree is
    walked once and both keys are kept. The first match of a name wins, which
    makes the lookup independent of the order os.walk happens to return.
    """
    by_name: Dict[str, str] = {}
    by_stem: Dict[str, str] = {}
    for root, _directories, files in os.walk(video_dir):
        for name in files:
            stem, suffix = os.path.splitext(name)
            if suffix.lower() not in VIDEO_SUFFIXES:
                continue
            path = os.path.join(root, name)
            by_name.setdefault(name, path)
            by_stem.setdefault(stem, path)
    return by_name, by_stem


@MM_BENCHMARKS.register("vdc")
class VDCBenchmarker(MMBenchmarker):
    """
    VDC benchmark implementation, without the VDCScore judge.

    Args:
        num_samples: number of videos to caption, all of them when not given.
        subset: restrict the videos to one or more sources, matched
            case-insensitively against the `video_source` column, e.g.
            `vdc:100:ego4d`. All of them when not given.
        caption_type: which of the five aspects to prompt for -- "detailed"
            (the default, and what the benchmark is named after), "camera",
            "short", "background" or "main_object".
        num_frames: how many evenly spaced frames one video is sent as.
        video_dir: directory holding the unpacked videos. Defaults to
            `$HF_HOME/vdc_videos`, which is also where the archives are
            unpacked to when a video is missing.
        auto_download: whether a missing video may be fetched from the dataset
            repository's tarballs. On by default; turning it off makes a missing
            video an error instead.
        frames_dir: where the decoded frames are cached and kept between runs.
        prompt_seed: seeds the per-video prompt draw, so that two runs send the
            same prompts and remain comparable.

    Every argument except `num_samples` and `subset` also reads a `VDC_`
    environment variable of the same name, which is how the command line reaches
    them.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        caption_type: Optional[str] = None,
        num_frames: Optional[int] = None,
        video_dir: Optional[str] = None,
        auto_download: Optional[bool] = None,
        frames_dir: Optional[str] = None,
        prompt_seed: Optional[int] = None,
    ):
        super().__init__(num_samples, subset)

        self.caption_type = (
            str(caption_type or os.environ.get("VDC_CAPTION_TYPE") or "detailed")
            .strip()
            .lower()
        )
        if self.caption_type not in CAPTION_COLUMNS:
            raise ValueError(
                f"Unknown VDC caption type '{self.caption_type}', "
                f"expected any of {sorted(CAPTION_COLUMNS)}"
            )

        self.num_frames = int(
            num_frames or os.environ.get("VDC_NUM_FRAMES") or DEFAULT_NUM_FRAMES
        )
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be at least 1, got {self.num_frames}")

        self.video_dir = video_dir or os.environ.get("VDC_VIDEO_DIR")
        self.auto_download = (
            auto_download
            if auto_download is not None
            else os.environ.get("VDC_AUTO_DOWNLOAD", "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        self.frames_dir = (
            frames_dir
            or os.environ.get("VDC_FRAMES_DIR")
            or os.path.join(".cache", "vdc_frames_specforge")
        )
        self.prompt_seed = int(
            prompt_seed
            if prompt_seed is not None
            else os.environ.get("VDC_PROMPT_SEED", 0)
        )

        # per-question metadata, kept aligned with the loaded questions.
        # `categories` is the name dump_generations() knows, so the source of a
        # video lands next to its generation as "category".
        self.categories: List[str] = []
        self.reference_lengths: List[int] = []
        # rows dropped because their video is not on disk
        self.missing_videos: List[str] = []

    def default_max_new_tokens(self) -> int:
        """Room for a caption of the requested aspect, see DEFAULT_MAX_NEW_TOKENS."""
        return DEFAULT_MAX_NEW_TOKENS[self.caption_type]

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """Sample every video into frames and pair them with an aspect prompt."""
        # the captions come first: which videos are actually needed is what keeps
        # the download of the archives below down to the ones that carry them
        dataset = load_dataset(DATASET_PATH)[SPLIT]
        if self.subset:
            dataset = dataset.select(self._select_subset(dataset))
        if self.num_samples is not None:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        video_dir = self._video_dir()
        by_name, by_stem = self._ensure_videos(dataset, video_dir)
        if not by_name:
            raise FileNotFoundError(
                f"No video file found under {video_dir}, and none could be "
                "downloaded. See the message above."
            )
        print(f"Found {len(by_name)} video files under {video_dir}")

        os.makedirs(self.frames_dir, exist_ok=True)
        column = CAPTION_COLUMNS[self.caption_type]
        prompts = CAPTION_PROMPTS[self.caption_type]

        questions: List[Dict[str, Any]] = []
        labels: List[Optional[str]] = []
        self.categories = []
        self.reference_lengths = []
        self.missing_videos = []

        for index, row in enumerate(dataset):
            video_path = self._find_video(row, by_name, by_stem)
            if video_path is None:
                self.missing_videos.append(str(row.get("video_name", row["video_id"])))
                continue

            frame_paths = self._materialize_frames(video_path, str(row["video_id"]))
            if not frame_paths:
                self.missing_videos.append(str(row.get("video_name", row["video_id"])))
                continue

            # seeded on the video rather than on its position, so that the prompt
            # a video gets does not move when --num-samples or a subset changes
            prompt = random.Random(f"{self.prompt_seed}:{row['video_id']}").choice(
                prompts
            )
            parts = [("image", path) for path in frame_paths] + [("text", prompt)]
            questions.append({"parts": parts, "video_name": str(row["video_name"])})

            reference = str(row[column] or "")
            labels.append(reference or None)
            self.reference_lengths.append(len(reference))
            self.categories.append(str(row.get(SUBSET_COLUMN, "unknown")))

            if (index + 1) % 100 == 0:
                print(f"  prepared {len(questions)} of {index + 1} videos")

        if self.missing_videos:
            print(
                f"Skipped {len(self.missing_videos)} videos that are not under "
                f"{video_dir}, e.g. {', '.join(self.missing_videos[:3])}"
            )
        print(
            f"Loaded {len(questions)} videos as {self.num_frames} frames each, "
            f"prompting for the {self.caption_type} caption. "
            f"Frames cached in {self.frames_dir}"
        )
        return questions, labels

    def _video_dir(self) -> str:
        """Where the videos live, defaulting next to the Hugging Face cache."""
        if self.video_dir:
            return self.video_dir
        root = os.environ.get("HF_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface"
        )
        return os.path.join(root, DEFAULT_VIDEO_SUBDIR)

    def _ensure_videos(
        self, dataset, video_dir: str
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Make sure the videos this run needs are on disk, and index them.

        The parquet carries no video, so the dataset repository ships them as
        three tarballs of 15.6, 24.8 and 34.2 GiB. They are pulled one at a time
        and only the members this run asks for are unpacked, so a
        `--benchmark-list vdc:200` normally stops after the first archive instead
        of dragging down all 74.6 GiB.

        Downloads land in the Hugging Face cache, which makes them resumable and
        shared with any other checkout.
        """
        os.makedirs(video_dir, exist_ok=True)
        by_name, by_stem = index_videos(video_dir)
        wanted, wanted_stems = self._wanted_videos(dataset, by_name, by_stem)
        if not wanted:
            return by_name, by_stem

        if not self.auto_download:
            raise FileNotFoundError(
                f"{len(wanted)} of this run's videos are not under {video_dir} "
                "and VDC_AUTO_DOWNLOAD is off. Unpack the dataset's "
                "videos/*.tar.gz there, or leave the download on."
            )

        archives = self._list_video_archives()
        print(
            f"{len(wanted)} videos are missing from {video_dir}. Pulling them out "
            f"of the dataset's {len(archives)} video archives (~74.6 GiB in total, "
            "downloaded one at a time and only until every wanted video is out; "
            "set VDC_AUTO_DOWNLOAD=0 to turn this off)."
        )
        for archive in archives:
            extracted = self._extract_from_archive(
                archive, wanted, wanted_stems, video_dir
            )
            print(
                f"  {archive}: unpacked {extracted} videos, "
                f"{len(wanted)} still missing"
            )
            if not wanted:
                break
        if wanted:
            print(
                f"  {len(wanted)} videos are in none of the archives; they are "
                "skipped and counted in the run description."
            )
        return index_videos(video_dir)

    @staticmethod
    def _wanted_videos(
        dataset, by_name: Dict[str, str], by_stem: Dict[str, str]
    ) -> Tuple[Set[str], Dict[str, str]]:
        """
        The rows whose video is not on disk yet, by file name and by stem.

        The stems are what makes the extraction below survive an archive that
        names its members by video id, or with a different container suffix,
        rather than by the `video_name` the parquet carries.
        """
        wanted: Set[str] = set()
        wanted_stems: Dict[str, str] = {}
        for name, video_id in zip(dataset["video_name"], dataset["video_id"]):
            name = str(name)
            if name in by_name:
                continue
            if str(video_id) in by_stem or os.path.splitext(name)[0] in by_stem:
                continue
            wanted.add(name)
            wanted_stems[os.path.splitext(name)[0]] = name
            wanted_stems[str(video_id)] = name
        return wanted, wanted_stems

    @staticmethod
    def _list_video_archives() -> List[str]:
        """The repository's video tarballs, smallest first is their own order."""
        from huggingface_hub import list_repo_files

        files = list_repo_files(DATASET_PATH, repo_type="dataset")
        archives = sorted(
            name
            for name in files
            if name.startswith("videos/") and name.endswith((".tar.gz", ".tgz", ".tar"))
        )
        if not archives:
            raise FileNotFoundError(
                f"{DATASET_PATH} has no videos/*.tar.gz to download from"
            )
        return archives

    @staticmethod
    def _extract_from_archive(
        archive: str,
        wanted: Set[str],
        wanted_stems: Dict[str, str],
        video_dir: str,
    ) -> int:
        """
        Download one archive and unpack only the wanted members out of it.

        `wanted` is emptied as the videos come out, so the caller can stop as soon
        as nothing is left. Members are written under their base name alone, which
        both flattens the archive's own layout and makes a crafted path in it
        harmless.

        An archive that yields nothing prints a few of the names it does hold:
        the alternative is silently downloading the remaining tens of GiB because
        its members turned out to be named in some way this does not recognise.
        """
        from huggingface_hub import hf_hub_download

        print(f"  downloading {archive} (resumable, cached under HF_HOME)...")
        archive_path = hf_hub_download(
            DATASET_PATH, filename=archive, repo_type="dataset"
        )

        extracted = 0
        unmatched: List[str] = []
        # streamed in one sequential pass: a .tar.gz cannot be seeked into
        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar:
                if not wanted:
                    break
                if not member.isfile():
                    continue
                name = os.path.basename(member.name)
                stem = os.path.splitext(name)[0]
                # by file name, or by stem for an archive that renamed its files
                row_name = name if name in wanted else wanted_stems.get(stem)
                if row_name is None:
                    if len(unmatched) < 5:
                        unmatched.append(name)
                    continue
                source = tar.extractfile(member)
                if source is None:
                    continue
                with source, open(os.path.join(video_dir, name), "wb") as target:
                    while True:
                        chunk = source.read(1 << 20)
                        if not chunk:
                            break
                        target.write(chunk)
                wanted.discard(row_name)
                wanted_stems.pop(stem, None)
                wanted_stems.pop(os.path.splitext(row_name)[0], None)
                extracted += 1
        if not extracted and unmatched:
            print(
                f"    nothing in {archive} matched a wanted video; it holds "
                f"files named like {', '.join(unmatched)}"
            )
        return extracted

    @staticmethod
    def _find_video(
        row: Dict[str, Any], by_name: Dict[str, str], by_stem: Dict[str, str]
    ) -> Optional[str]:
        """The file of one row, by file name, by id, or by name without suffix."""
        name = str(row.get("video_name") or "")
        if name in by_name:
            return by_name[name]
        for stem in (str(row.get("video_id") or ""), os.path.splitext(name)[0]):
            if stem and stem in by_stem:
                return by_stem[stem]
        return None

    def _materialize_frames(self, video_path: str, stem: str) -> List[str]:
        """
        Decode one video into cached JPEG frames and return their paths.

        The cache is keyed by the frame count as well, so that changing
        --num-frames does not silently reuse the previous sampling. It is looked
        up by glob rather than by expected file name, since a video shorter than
        the frame count yields fewer files than were asked for.
        """
        prefix = os.path.join(self.frames_dir, f"{stem}__{self.num_frames}f_")
        cached = sorted(glob.glob(glob.escape(prefix) + "*.jpg"))
        if cached:
            return cached

        try:
            frames = read_frames(video_path, self.num_frames)
        except ImportError:
            raise
        except Exception as error:  # a single unreadable file must not stop the run
            print(f"  cannot decode {video_path}: {type(error).__name__}: {error}")
            return []

        paths = []
        for position, frame in enumerate(frames):
            path = f"{prefix}{position:02d}.jpg"
            frame.convert("RGB").save(path, "JPEG", quality=90)
            paths.append(path)
        return paths

    def _select_subset(self, dataset) -> List[int]:
        """Indices of the rows whose video source matches the requested subset."""
        wanted = {name.strip().lower() for name in self.subset}
        sources = [str(value).strip().lower() for value in dataset[SUBSET_COLUMN]]
        available = set(sources)
        unknown = wanted - available
        if unknown:
            raise ValueError(
                f"Unknown VDC subset(s) {sorted(unknown)}, expected a video "
                f"source among {sorted(available)}"
            )
        return [index for index, source in enumerate(sources) if source in wanted]

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """The caption itself is the answer; only a reasoning block is dropped."""
        if not isinstance(output, str):
            return None
        return strip_reasoning(output).strip() or None

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """
        Not scored here.

        Grading a caption is what VDCScore's two judge passes are for, and this
        port exists to measure decoding, not caption quality. Returning None
        leaves `metrics.accuracy` unset while the reference captions still reach
        the generations file for an offline VDCScore run.
        """
        return None

    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, BenchmarkMetrics]]:
        """
        Report the metrics of every video source.

        The corpora differ in resolution and length, so their prefill cost and
        their accept length do too. The latency is the one of the whole run, so a
        source's throughput is its share of the aggregate rather than a figure it
        could reach on its own.
        """
        if not self.categories:
            return None

        performance = {}
        for source in sorted(set(self.categories)):
            indexes = [
                index
                for index, name in enumerate(self.categories)
                if name == source and index < len(states)
            ]
            if not indexes:
                continue
            performance[source] = compute_metrics(
                [states[index] for index in indexes], latency, answer_key=answer_key
            )
        return performance

    def describe_run(self) -> Optional[Dict[str, Any]]:
        """Report what was captioned, how it was framed, and how long it came out."""
        description: Dict[str, Any] = {
            "split": SPLIT,
            "caption_type": self.caption_type,
            "caption_column": CAPTION_COLUMNS[self.caption_type],
            "num_frames": self.num_frames,
            "prompt_seed": self.prompt_seed,
            "videos": len(self.categories),
            "videos_per_source": dict(sorted(Counter(self.categories).items())),
            "video_dir": self._video_dir(),
            "frames_dir": self.frames_dir,
            "scoring": "none, VDCScore's judge is deliberately not run",
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
        """Create the SGL function for VDC (frames followed by the instruction)."""
        return create_interleaved_sgl_function(
            function_name="get_vdc_caption",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
            assistant_prefix=self.assistant_prefix,
        )
