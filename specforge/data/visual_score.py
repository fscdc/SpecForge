"""Per-token visual-dependency scores for the MMFlash objective.

The score of a generated token is how much the target's next-token distribution
moves when the image is taken away:

    kl_t = KL( p_target(. | x_<t, image) || p_target(. | x_<t, no image) )

It is produced offline by ``scripts/score_visual_kl.py`` and stored in a
*sidecar* keyed by the training record's ``id``, one JSON line per image row::

    {"id": "wikipedia_2m#110-0", "n_tokens": 812, "n_loss": 143,
     "fp": "3f9c1a0b7e2d4551", "kl": [0.01, 3.2, ...], "entropy": [0.4, 1.1, ...]}

``kl``/``entropy`` are compact: one value per loss-mask position, in ascending
position order, so ``len(kl) == sum(loss_mask)``. ``n_tokens`` is the length of
the with-image ``input_ids`` the scorer saw and ``fp`` a digest of those exact
token ids, so the producer can refuse a sidecar built with a different
tokenizer/template/resolution -- or one built from a DIFFERENT regen of the
same corpus, where the ids match but the responses do not.

This module turns raw KL (nats, unbounded, heavy-tailed) into ``g in [0, 1]``
(:func:`transform_scores`), joins it onto records (:func:`VisualScoreTable`),
and expands the compact vector back onto the full token sequence in the wire
format the capture pipeline carries (:func:`expand_visual_score`).

Wire format
-----------
The capture transport stores client passthrough tensors as ``int64`` (see
``_spec_capture_payload``), so ``g`` travels quantised: ``round(g * SCALE)``.
Text-only rows carry :data:`TEXT_ONLY_SENTINEL` at every position; that single
channel therefore tells the trainer both *which rows have an image* and *how
visual each token is*. :func:`dequantize_visual_score` is the consumer-side
inverse and keeps the sentinel negative so the model can read the row flag.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

#: Quantisation scale of ``g`` on the wire; 1e-4 resolution is far below any
#: difference the weighting could react to.
SCALE = 10_000
#: Every position of a text-only row carries this value (negative, so it can
#: never collide with a quantised score).
TEXT_ONLY_SENTINEL = -1

TRANSFORMS = ("quantile", "saturate", "binary", "identity")

#: Resolution of the empirical CDF used by the quantile transform.
_QUANTILE_POINTS = 1000


def fingerprint_input_ids(input_ids: Sequence[int]) -> str:
    """Digest of a token sequence; the scorer stores it, the producer checks it.

    Must stay in step with ``scripts/score_visual_kl.py``.
    """
    return hashlib.blake2b(array("i", input_ids).tobytes(), digest_size=8).hexdigest()


def _sidecar_files(path: str) -> List[str]:
    """The sidecar's files: a single JSONL, or every ``*.jsonl`` in a directory."""
    if os.path.isdir(path):
        files = sorted(
            f
            for f in glob.glob(os.path.join(path, "*.jsonl"))
            if not os.path.basename(f).startswith(".")
        )
        if not files:
            raise FileNotFoundError(f"visual score directory {path!r} holds no *.jsonl")
        return files
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError(f"visual score sidecar {path!r} does not exist")


def iter_sidecar(path: str) -> Iterator[Dict[str, Any]]:
    """Yield the sidecar's records in file order, skipping blank lines."""
    for file in _sidecar_files(path):
        with open(file, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{file}:{line_number}: invalid JSON in visual score sidecar"
                    ) from exc
                yield record


def _quantile_cache_path(path: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, "quantiles.json")
    return path + ".quantiles.json"


def compute_quantile_edges(path: str, *, points: int = _QUANTILE_POINTS) -> np.ndarray:
    """Edges of the empirical KL CDF over every scored token of the corpus.

    Cached next to the sidecar (``quantiles.json``) because it only depends on
    the sidecar's content; delete the cache after re-scoring.
    """
    cache = _quantile_cache_path(path)
    if os.path.isfile(cache):
        with open(cache, encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("points") == points and payload.get("files") == _sidecar_files(path):
            return np.asarray(payload["edges"], dtype=np.float64)
    chunks: List[np.ndarray] = []
    for record in iter_sidecar(path):
        chunks.append(np.asarray(record["kl"], dtype=np.float32))
    if not chunks:
        raise ValueError(f"visual score sidecar {path!r} is empty")
    values = np.concatenate(chunks)
    edges = np.quantile(values, np.linspace(0.0, 1.0, points + 1))
    try:
        with open(cache, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "points": points,
                    "files": _sidecar_files(path),
                    "num_tokens": int(values.size),
                    "edges": edges.tolist(),
                },
                handle,
            )
    except OSError:
        # a read-only sidecar location just recomputes next time
        pass
    return edges


def transform_scores(
    kl: np.ndarray,
    transform: str,
    *,
    edges: Optional[np.ndarray] = None,
    saturate_c: Optional[float] = None,
    binary_threshold: float = 0.75,
) -> np.ndarray:
    """Map raw KL values to ``g`` in [0, 1] (float32).

    quantile
        Corpus CDF rank: ``g = P(KL <= kl)``. Distribution-free; the top 25% of
        tokens get ``g >= 0.75``.
    saturate
        ``g = kl / (kl + c)`` with ``c`` the 75th-percentile KL, so the old
        top-quartile threshold maps to ``g = 0.5`` and the tail stays ordered.
    binary
        ``1[quantile rank >= binary_threshold]``.
    identity
        ``clip(kl, 0, 1)`` -- for a sidecar that already stores ``g``.
    """
    if transform not in TRANSFORMS:
        raise ValueError(f"unknown visual score transform {transform!r}; expected {TRANSFORMS}")
    kl = np.asarray(kl, dtype=np.float64)
    if transform == "identity":
        return np.clip(kl, 0.0, 1.0).astype(np.float32)
    if edges is None:
        raise ValueError(f"transform {transform!r} needs the corpus quantile edges")
    points = len(edges) - 1
    # rank in (0, 1]: the share of corpus tokens whose KL is <= this one
    rank = np.searchsorted(edges, kl, side="right") / float(points)
    rank = np.clip(rank, 0.0, 1.0)
    if transform == "quantile":
        return rank.astype(np.float32)
    if transform == "binary":
        return (rank >= binary_threshold).astype(np.float32)
    # saturate
    c = float(edges[int(round(0.75 * points))]) if saturate_c is None else float(saturate_c)
    c = max(c, np.finfo(np.float32).tiny)
    return (kl / (kl + c)).astype(np.float32)


@dataclass
class VisualScoreStats:
    files: List[str] = field(default_factory=list)
    rows: int = 0
    tokens: int = 0
    transform: str = ""
    kl_quantiles: Dict[str, float] = field(default_factory=dict)
    g_mean: float = 0.0
    g_share_ge_075: float = 0.0

    def describe(self) -> str:
        q = ", ".join(f"p{k}={v:.3f}" for k, v in self.kl_quantiles.items())
        return (
            f"{self.rows} rows / {self.tokens} scored tokens from {len(self.files)} file(s); "
            f"transform={self.transform}; raw KL {q}; "
            f"g mean={self.g_mean:.3f}, share(g>=0.75)={self.g_share_ge_075:.3f}"
        )


class VisualScoreTable:
    """``id -> (n_tokens, fingerprint, g[float32 over loss positions])``."""

    def __init__(self, entries: Dict[str, Tuple[int, str, np.ndarray]], stats: VisualScoreStats):
        self._entries = entries
        self.stats = stats

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, record_id: Any) -> bool:
        return str(record_id) in self._entries

    def get(self, record_id: Any) -> Optional[Tuple[int, str, np.ndarray]]:
        return self._entries.get(str(record_id))

    @classmethod
    def load(
        cls,
        path: str,
        *,
        transform: str = "quantile",
        binary_threshold: float = 0.75,
    ) -> "VisualScoreTable":
        edges = None
        if transform != "identity":
            edges = compute_quantile_edges(path)
        entries: Dict[str, Tuple[int, str, np.ndarray]] = {}
        stats = VisualScoreStats(files=_sidecar_files(path), transform=transform)
        g_sum = 0.0
        g_hi = 0
        for record in iter_sidecar(path):
            record_id = str(record["id"])
            kl = np.asarray(record["kl"], dtype=np.float32)
            n_loss = int(record.get("n_loss", kl.size))
            if kl.size != n_loss:
                raise ValueError(
                    f"visual score row {record_id!r}: n_loss={n_loss} but {kl.size} kl values"
                )
            if record_id in entries:
                raise ValueError(f"visual score sidecar lists {record_id!r} twice")
            g = transform_scores(
                kl, transform, edges=edges, binary_threshold=binary_threshold
            )
            entries[record_id] = (int(record["n_tokens"]), str(record.get("fp", "")), g)
            stats.rows += 1
            stats.tokens += int(kl.size)
            g_sum += float(g.sum())
            g_hi += int((g >= 0.75).sum())
        if not entries:
            raise ValueError(f"visual score sidecar {path!r} is empty")
        if edges is not None:
            points = len(edges) - 1
            stats.kl_quantiles = {
                str(p): float(edges[int(round(p / 100 * points))]) for p in (25, 50, 75, 90, 99)
            }
        stats.g_mean = g_sum / max(stats.tokens, 1)
        stats.g_share_ge_075 = g_hi / max(stats.tokens, 1)
        return cls(entries, stats)


def quantize(g: np.ndarray) -> np.ndarray:
    """``g`` in [0, 1] -> wire integers in [0, SCALE]."""
    return np.rint(np.clip(g, 0.0, 1.0) * SCALE).astype(np.int64)


def text_only_visual_score(length: int) -> array:
    """The channel for a row without an image: the sentinel at every position."""
    return array("i", [TEXT_ONLY_SENTINEL]) * length


def expand_visual_score(
    loss_mask: Sequence[int],
    scores: Optional[Tuple[int, str, np.ndarray]],
    *,
    input_ids: Sequence[int],
) -> Tuple[array, str]:
    """Spread a compact score vector onto the full token sequence (quantised).

    Returns ``(channel, status)`` where status is ``"scored"`` when the sidecar
    entry aligned, ``"missing"`` when there was no entry, or ``"misaligned"``
    when the entry was built from a different token sequence; the last two fall
    back to all-zero scores (plain verification-aware weights, still an image
    row) rather than dropping the row, so the training set stays identical to a
    run without a sidecar and only the weighting differs.

    Alignment is decided by the fingerprint of ``input_ids`` when the sidecar
    carries one, and by the token counts alone otherwise. The counts are not
    enough on their own: two regens of the same corpus share their record ids
    and differ only in the response, which for a short reply can leave both
    counts equal.
    """
    num_tokens = len(input_ids)
    channel = array("i", [0]) * num_tokens
    if scores is None:
        return channel, "missing"
    n_tokens, fingerprint, g = scores
    positions = [index for index, flag in enumerate(loss_mask) if flag]
    if n_tokens != num_tokens or len(positions) != len(g):
        return channel, "misaligned"
    if fingerprint and fingerprint != fingerprint_input_ids(input_ids):
        return channel, "misaligned"
    for position, value in zip(positions, quantize(g).tolist()):
        channel[position] = int(value)
    return channel, "scored"


def dequantize_visual_score(channel: "Any") -> "Any":
    """Consumer-side inverse of :func:`expand_visual_score` (torch tensors).

    Scores come back as float32 in [0, 1]; the text-only sentinel stays
    negative (``-1.0``) so ``(visual_score >= 0).any(dim=1)`` is the row's
    has-image flag. Padding positions must be filled with the sentinel by the
    collator, never with zero, or a text-only row padded next to a longer one
    would read as multimodal.
    """
    import torch

    values = channel.to(torch.float32)
    return torch.where(values < 0, torch.full_like(values, -1.0), values / float(SCALE))


def join_visual_scores(
    records: Sequence[Mapping[str, Any]],
    table: Optional[VisualScoreTable],
    *,
    key: str = "_visual_score",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Attach the sidecar entry to every image record (``None`` when absent).

    Done once in the main process, so a pool of encode workers never loads the
    sidecar 32 times over. Text-only rows get no entry (they carry no image to
    score) and are counted separately.
    """
    counts = {"image_scored": 0, "image_missing": 0, "text_only": 0}
    joined: List[Dict[str, Any]] = []
    for record in records:
        entry = None
        if record.get("image") is None:
            counts["text_only"] += 1
        elif table is not None and record.get("id") in table:
            entry = table.get(record["id"])
            counts["image_scored"] += 1
        else:
            counts["image_missing"] += 1
        joined.append({**record, key: entry})
    return joined, counts


__all__ = [
    "SCALE",
    "TEXT_ONLY_SENTINEL",
    "TRANSFORMS",
    "VisualScoreStats",
    "VisualScoreTable",
    "fingerprint_input_ids",
    "compute_quantile_edges",
    "dequantize_visual_score",
    "expand_visual_score",
    "iter_sidecar",
    "join_visual_scores",
    "quantize",
    "text_only_visual_score",
    "transform_scores",
]
