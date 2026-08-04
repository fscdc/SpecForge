"""Rewrite the image root of a conversation JSONL so it resolves on this host.

Records carry absolute image paths, so a dataset produced on one machine points
at a directory that does not exist on the next one. This rewrites only the root
prefix and keeps the per-source suffix (``coco/train2017/...``, ``sam/images/...``)
untouched, so one flag moves a whole corpus between hosts.

The old root can be inferred: with ``--infer-old-root`` the longest common
directory prefix over the sampled rows is used, which avoids having to spell out
a path that came from another machine.

By default the rewritten dataset takes over the input's name and the original
file is kept alongside it with a ``_deep100`` suffix (``data.jsonl`` ->
``data_deep100.jsonl``). ``--in-place`` overwrites the input without keeping
that copy; ``--output`` writes somewhere else and leaves the input untouched.

Examples:
    python scripts/rebase_image_paths.py /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/rengen_data/ \
        --old-root /Project_Storage/CFP-04/CFP04-CF-054/fengsicheng/specforge/data \
        --new-root /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="source conversation JSONL")
    parser.add_argument(
        "--new-root",
        required=True,
        help="directory the rewritten image paths should live under",
    )
    parser.add_argument(
        "--old-root",
        default=None,
        help="prefix to strip; omit together with --infer-old-root",
    )
    parser.add_argument(
        "--infer-old-root",
        action="store_true",
        help="derive the old root from the longest common prefix of the rows",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="destination JSONL (default: take over the input's name and keep "
        "the original as '<name>_deep100<ext>')",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="overwrite the input file without keeping the _deep100 copy",
    )
    parser.add_argument(
        "--image-field",
        default="image",
        help="record field holding the image path (default: image)",
    )
    parser.add_argument(
        "--check-exists",
        action="store_true",
        help="stat every rewritten path and report the ones that are missing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing anything",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if bool(args.old_root) == bool(args.infer_old_root):
        raise SystemExit("pass exactly one of --old-root or --infer-old-root")
    if args.in_place and args.output is not None:
        raise SystemExit("--in-place and --output are mutually exclusive")
    if args.in_place and args.dry_run:
        raise SystemExit("--in-place cannot be combined with --dry-run")
    return args


def iter_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, record)``, skipping blank lines."""
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def infer_old_root(path: Path, image_field: str) -> str:
    """Longest common directory prefix over every image path in the file.

    Sources sit in sibling directories (``coco/``, ``sam/``, ...), so the common
    prefix is the shared data root rather than any one source. This reads the
    whole file on purpose: corpora are usually grouped by source, so a prefix
    taken from the first rows would collapse onto whichever source leads the
    file and strip a directory level that later rows still need.
    """
    paths: list[str] = []
    for _line_number, record in iter_records(path):
        value = record.get(image_field)
        if isinstance(value, str) and value:
            paths.append(value)
    if not paths:
        raise SystemExit(f"no {image_field!r} values found in {path}")

    common = os.path.commonpath(paths) if len(paths) > 1 else os.path.dirname(paths[0])
    if not common or common == os.sep:
        raise SystemExit(
            "could not infer a shared root (paths have nothing in common); "
            "pass --old-root explicitly"
        )
    return common


def rebase(value: str, old_root: str, new_root: str) -> str | None:
    """Swap ``old_root`` for ``new_root``; None when the path is not under it."""
    old = old_root.rstrip("/")
    if value == old:
        return new_root.rstrip("/")
    if not value.startswith(old + "/"):
        return None
    return os.path.join(new_root.rstrip("/"), value[len(old) + 1 :])


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.input.is_file():
        raise SystemExit(f"input not found: {args.input}")

    old_root = args.old_root or infer_old_root(args.input, args.image_field)
    new_root = args.new_root
    print(f"old root : {old_root}")
    print(f"new root : {new_root}")

    backup = None
    if args.in_place:
        destination = args.input
    elif args.output is not None:
        destination = args.output
    else:
        destination = args.input
        backup = args.input.with_name(
            args.input.stem + "_deep100" + args.input.suffix
        )
        if backup.exists() and not args.dry_run:
            raise SystemExit(f"refusing to overwrite existing backup: {backup}")

    total = rewritten = skipped = missing_field = 0
    unmatched: list[str] = []
    missing_files: list[str] = []

    # Write to a temporary file in the destination directory and rename at the
    # end, so an interrupted run can never leave a half-written dataset behind:
    # the input keeps its old content (and its name) until the new file is
    # complete, and only then is it renamed to the _deep100 backup.
    handle = None
    temporary = None
    if not args.dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary = tempfile.mkstemp(
            dir=destination.parent, prefix=destination.name, suffix=".tmp"
        )
        handle = os.fdopen(file_descriptor, "w", encoding="utf-8")

    try:
        for _line_number, record in iter_records(args.input):
            total += 1
            value = record.get(args.image_field)
            if not isinstance(value, str) or not value:
                missing_field += 1
            else:
                moved = rebase(value, old_root, new_root)
                if moved is None:
                    skipped += 1
                    if len(unmatched) < 5:
                        unmatched.append(value)
                else:
                    record[args.image_field] = moved
                    rewritten += 1
                    if args.check_exists and not os.path.exists(moved):
                        if len(missing_files) < 5:
                            missing_files.append(moved)
                        elif missing_files:
                            missing_files.append("")  # count-only past the sample
            if handle is not None:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if handle is not None:
            handle.close()

    if temporary is not None:
        if backup is not None:
            os.replace(args.input, backup)
        os.replace(temporary, destination)

    print(f"\nrecords      : {total}")
    print(f"  rewritten  : {rewritten}")
    print(f"  not under old root : {skipped}")
    print(f"  no {args.image_field!r} field  : {missing_field}")
    if unmatched:
        print("  examples left untouched:")
        for value in unmatched:
            print(f"    {value}")
    if args.check_exists:
        found = sum(1 for value in missing_files if value)
        print(f"  missing on disk    : {'>=' if found >= 5 else ''}{found}")
        for value in missing_files[:5]:
            if value:
                print(f"    {value}")
    if args.dry_run:
        print("\ndry run: nothing written")
    else:
        print(f"\nwrote {destination}")
        if backup is not None:
            print(f"original kept as {backup}")

    if skipped and not rewritten:
        sys.exit("no path matched the old root — check --old-root")


if __name__ == "__main__":
    main()
