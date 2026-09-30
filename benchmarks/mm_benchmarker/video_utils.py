"""
Helpers shared by the video benchmarks (VDC, LongVideoBench, MovieChat, Video-MME, MVBench).

A video enters a prompt as `num_frames` evenly spaced JPEG frames, decoded once
and kept under a cache directory, exactly as `vdc.py` does; the sampling and
the decoder selection live there and are reused here rather than copied.
"""

import glob
import json
import os
import zipfile
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image

from .vdc import VIDEO_SUFFIXES, frame_indices, index_videos, read_frames  # noqa: F401

__all__ = [
    "RemoteZipIndex",
    "VIDEO_SUFFIXES",
    "default_video_dir",
    "env_flag",
    "index_videos",
    "materialize_frames",
    "read_frames",
    "read_frames_window",
]


def env_flag(name: str, default: bool = True) -> bool:
    """A boolean environment switch; "0", "false", "no" and "off" turn it off."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def default_video_dir(subdir: str, override: Optional[str] = None) -> str:
    """Where a benchmark's videos live, next to the Hugging Face cache by default."""
    if override:
        return override
    root = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    return os.path.join(root, subdir)


def read_frames_window(
    video_path: str,
    num_frames: int,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> List[Image.Image]:
    """
    `num_frames` evenly spaced frames between `start` and `end` seconds.

    MVBench asks about a segment of a longer video (Charades, STA), so the
    frames have to come from that window rather than from the whole file. With
    no window this is `read_frames`. Needs decord for the seek; without it the
    whole video is sampled and the window is ignored.
    """
    if start is None and end is None:
        return read_frames(video_path, num_frames)
    try:
        import decord
    except ImportError:
        return read_frames(video_path, num_frames)
    reader = decord.VideoReader(video_path, num_threads=1)
    total = len(reader)
    fps = float(reader.get_avg_fps() or 0) or 30.0
    first = 0 if start is None else max(0, min(total - 1, int(round(start * fps))))
    last = total - 1 if end is None else max(first, min(total - 1, int(round(end * fps))))
    span = last - first + 1
    indices = [first + index for index in frame_indices(span, num_frames)]
    batch = reader.get_batch(indices).asnumpy()
    return [Image.fromarray(frame) for frame in batch]


def materialize_frames(
    frames_dir: str,
    video_path: str,
    stem: str,
    num_frames: int,
    *,
    start: Optional[float] = None,
    end: Optional[float] = None,
    target_pixels: Optional[int] = None,
) -> List[str]:
    """
    Decode one video into cached JPEG frames and return their paths.

    The cache is keyed by the frame count, the time window and the resize, so
    that changing any of them does not silently reuse the previous sampling. It
    is looked up by glob rather than by expected file name, since a video
    shorter than the frame count yields fewer files than were asked for. An
    undecodable file yields an empty list rather than an exception, so one
    broken video cannot stop a run.

    `target_pixels` rescales every frame to that many pixels of area (aspect
    preserved) before it is saved; MVBench uses it to bring its 480p clips up
    to the token count of a 720p frame, whatever their aspect ratio, so that
    its prompts are as long as the other benchmarks'.
    """
    os.makedirs(frames_dir, exist_ok=True)
    tag = f"{stem}__{num_frames}f"
    if start is not None or end is not None:
        tag += f"_s{0.0 if start is None else float(start):.1f}-{'end' if end is None else f'{float(end):.1f}'}"
    if target_pixels:
        tag += f"_px{int(target_pixels)}"
    prefix = os.path.join(frames_dir, tag + "_")
    # the frame number must follow the tag directly: the tag of a native-size
    # or whole-video sampling is a prefix of the tags of its rescaled or
    # windowed variants, which must not be picked up as its cache
    cached = sorted(glob.glob(glob.escape(prefix) + "[0-9]*.jpg"))
    if cached:
        return cached

    try:
        frames = read_frames_window(video_path, num_frames, start, end)
    except ImportError:
        raise
    except Exception as error:  # a single unreadable file must not stop the run
        print(f"  cannot decode {video_path}: {type(error).__name__}: {error}")
        return []

    paths = []
    for position, frame in enumerate(frames):
        frame = frame.convert("RGB")
        if target_pixels and frame.width * frame.height != target_pixels:
            scale = (target_pixels / (frame.width * frame.height)) ** 0.5
            size = (max(1, round(frame.width * scale)), max(1, round(frame.height * scale)))
            frame = frame.resize(size, Image.LANCZOS)
        path = f"{prefix}{position:02d}.jpg"
        frame.save(path, "JPEG", quality=90)
        paths.append(path)
    return paths


class RemoteZipIndex:
    """
    Which member of which zip of a Hugging Face dataset holds a given video.

    The video benchmarks that ship as zips (Video-MME: 19 x 5 GiB, MVBench:
    one per source, MovieChat: one 17 GiB) do not need the archives on disk: a
    zip's central directory sits at its end, so the Hugging Face filesystem's
    range reads can list every member in a few seconds and then stream just the
    wanted members out. Listings are cached as JSON next to the videos, and the
    zips are listed lazily, in order, only until every wanted name is located.
    """

    def __init__(self, repo: str, zip_paths: Iterable[str], cache_file: str):
        self.repo = repo
        self.zip_paths = list(zip_paths)
        self.cache_file = cache_file
        # member basename -> [(zip path, member name, size)]
        self.members: Dict[str, List[Tuple[str, str, int]]] = {}
        self._listed: List[str] = []
        self._cached: Optional[Dict[str, List[List]]] = None

    def _fs(self):
        from huggingface_hub import HfFileSystem

        return HfFileSystem()

    def _remote(self, zip_path: str) -> str:
        return f"datasets/{self.repo}/{zip_path}"

    def _open_remote(self, zip_path: str):
        return self._fs().open(self._remote(zip_path), "rb", block_size=8 * 1024 * 1024)

    def _load_cache(self) -> Dict[str, List[List]]:
        if self._cached is None:
            self._cached = {}
            if os.path.exists(self.cache_file):
                with open(self.cache_file, encoding="utf-8") as handle:
                    self._cached = json.load(handle).get("zips", {})
        return self._cached

    def _save_cache(self) -> None:
        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        temporary = self.cache_file + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump({"repo": self.repo, "zips": self._cached}, handle)
        os.replace(temporary, self.cache_file)

    def list_zip(self, zip_path: str) -> None:
        """Add one zip's members to the index, from the cache or the remote."""
        if zip_path in self._listed:
            return
        cached = self._load_cache()
        entries = cached.get(zip_path)
        if entries is None:
            print(f"  listing {zip_path} (central directory over HTTP range reads)...")
            with self._open_remote(zip_path) as handle:
                with zipfile.ZipFile(handle) as archive:
                    entries = [
                        [info.filename, int(info.file_size)]
                        for info in archive.infolist()
                        if not info.is_dir()
                    ]
            cached[zip_path] = entries
            self._save_cache()
        for member, size in entries:
            self.members.setdefault(os.path.basename(member), []).append(
                (zip_path, member, int(size))
            )
        self._listed.append(zip_path)

    def load(self) -> "RemoteZipIndex":
        """List every zip."""
        for zip_path in self.zip_paths:
            self.list_zip(zip_path)
        return self

    def find(self, name: str, folder: Optional[str] = None) -> Optional[Tuple[str, str, int]]:
        """The (zip, member, size) of a video by base name among the zips listed so far.

        A name without a matching suffix is matched by stem (the JSON may say
        `x.mp4` where the archive holds `x.webm`). With `folder`, a member
        under that directory wins over same-named ones elsewhere.
        """
        candidates = self.members.get(os.path.basename(name), [])
        if not candidates:
            stem = os.path.splitext(os.path.basename(name))[0]
            for base, entries in self.members.items():
                if os.path.splitext(base)[0] == stem and base.lower().endswith(VIDEO_SUFFIXES):
                    candidates = entries
                    break
        if not candidates:
            return None
        if folder:
            folder = folder.strip("/")
            for entry in candidates:
                if os.path.dirname(entry[1]).endswith(folder):
                    return entry
        return candidates[0]

    def locate(
        self, names: Iterable[str], folder: Optional[str] = None
    ) -> Dict[str, Tuple[str, str, int]]:
        """
        Where each wanted name lives, listing further zips only while some are missing.

        Video-MME's 19 archives are in dataset order, so a run over the first
        questions is served by the first one or two listings.
        """
        pending = set(names)
        found: Dict[str, Tuple[str, str, int]] = {}
        for zip_path in self.zip_paths:
            for name in list(pending):
                entry = self.find(name, folder)
                if entry is not None:
                    found[name] = entry
                    pending.discard(name)
            if not pending:
                break
            self.list_zip(zip_path)
        for name in list(pending):
            entry = self.find(name, folder)
            if entry is not None:
                found[name] = entry
                pending.discard(name)
        return found

    def fetch(self, wanted: Dict[str, Tuple[str, str]], video_dir: str) -> int:
        """
        Stream the wanted members into `video_dir`, one archive at a time.

        `wanted` maps the file name to write (relative to `video_dir`) to
        (zip path, member). Members are grouped per zip so each central
        directory is read once. If the ranged read of an archive fails, that
        archive is downloaded whole through the Hugging Face cache and the
        members are extracted from the local copy.
        """
        by_zip: Dict[str, List[Tuple[str, str]]] = {}
        for target_name, (zip_path, member) in wanted.items():
            by_zip.setdefault(zip_path, []).append((member, target_name))
        os.makedirs(video_dir, exist_ok=True)
        fetched = 0
        for zip_path, items in sorted(by_zip.items()):
            size = sum(
                entry[2]
                for member, _ in items
                for entry in self.members.get(os.path.basename(member), [])
                if entry[0] == zip_path and entry[1] == member
            )
            print(f"  fetching {len(items)} video(s), {size / 2**20:.0f} MiB, out of {zip_path}...")
            try:
                with self._open_remote(zip_path) as handle:
                    with zipfile.ZipFile(handle) as archive:
                        fetched += self._extract(archive, items, video_dir)
            except Exception as error:
                print(
                    f"  ranged read of {zip_path} failed ({type(error).__name__}: "
                    f"{error}); downloading the whole archive instead"
                )
                from huggingface_hub import hf_hub_download

                local = hf_hub_download(self.repo, filename=zip_path, repo_type="dataset")
                with zipfile.ZipFile(local) as archive:
                    fetched += self._extract(archive, items, video_dir)
        return fetched

    @staticmethod
    def _extract(archive: zipfile.ZipFile, items: List[Tuple[str, str]], video_dir: str) -> int:
        count = 0
        for member, target_name in items:
            target = os.path.join(video_dir, target_name)
            if os.path.exists(target):
                count += 1
                continue
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            with archive.open(member) as source, open(target + ".part", "wb") as sink:
                while True:
                    chunk = source.read(1 << 20)
                    if not chunk:
                        break
                    sink.write(chunk)
            os.replace(target + ".part", target)
            count += 1
        return count
