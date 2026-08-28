# # # # from datasets import load_dataset

# # # # ds = load_dataset("Lin-Chen/ShareGPT4V", "ShareGPT4V", split="train")
# # # # print("total rows:", len(ds))

# # # # prefixes = {}
# # # # for row in ds:
# # # #     image = row.get("image")
# # # #     if not image:
# # # #         continue
# # # #     prefix = image.rsplit("/", 1)[0] if "/" in image else "(no directory)"
# # # #     prefixes[prefix] = prefixes.get(prefix, 0) + 1

# # # # print()
# # # # print(f"{len(prefixes)} unique prefixes:")
# # # # for p in sorted(prefixes):
# # # #     print(f"  {p}  (count={prefixes[p]})")







# # # #################################




# """Randomly sample rows from a regenerated JSONL and print them for inspection.

# Usage:
#     python inspect_regen.py <path/to/regen.jsonl> [-n 10] [--seed 42]
# """

# import argparse
# import json
# import random
# from pathlib import Path


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Randomly sample and pretty-print rows from a regenerated JSONL."
#     )
#     parser.add_argument("jsonl_path", type=Path, help="Path to the regenerated JSONL file.")
#     parser.add_argument(
#         "-n", "--num-samples", type=int, default=10, help="How many rows to show (default: 10)."
#     )
#     parser.add_argument(
#         "--seed", type=int, default=None, help="Random seed for reproducible sampling."
#     )
#     return parser.parse_args()


# def main():
#     args = parse_args()
#     if args.seed is not None:
#         random.seed(args.seed)

#     with args.jsonl_path.open("r", encoding="utf-8") as handle:
#         rows = [json.loads(line) for line in handle if line.strip()]

#     total = len(rows)
#     if total == 0:
#         print(f"{args.jsonl_path} is empty.")
#         return

#     k = min(args.num_samples, total)
#     sampled = random.sample(rows, k)
#     print(f"Sampled {k} of {total} rows from {args.jsonl_path}\n")

#     for i, row in enumerate(sampled, start=1):
#         print("=" * 80)
#         print(f"[{i}/{k}]  id: {row.get('id', '(no id)')}  status: {row.get('status', '(no status)')}")

#         image = row.get("image")
#         if image is not None:
#             # Print as a file:// URL so most terminals make it clickable.
#             print(f"image: {image}")
#             print(f"       file://{Path(image).resolve()}")

#         for msg in row.get("conversations", []):
#             role = msg.get("role", "?")
#             content = msg.get("content", "")
#             print(f"\n  --- {role} ---")
#             print(f"  {content}")

#             reasoning = msg.get("reasoning_content")
#             if reasoning:
#                 print(f"\n  --- {role} (reasoning) ---")
#                 print(f"  {reasoning}")
#         print()


# if __name__ == "__main__":
#     main()

# # # #


# # """Count the number of lines in a JSONL file.

# # Usage:
# #     python test.py <path/to/file.jsonl>
# # """

# # import sys
# # from pathlib import Path


# # def main():
# #     if len(sys.argv) != 2:
# #         print("Usage: python test.py <path/to/file.jsonl>")
# #         sys.exit(1)

# #     jsonl_path = Path(sys.argv[1])
# #     with jsonl_path.open("r", encoding="utf-8") as f:
# #         count = sum(1 for line in f if line.strip())

# #     print(f"{jsonl_path}: {count} lines")


# # if __name__ == "__main__":
# #     main()


"""Randomly sample rows from a JSONL file.

The samples are written next to the source file as <name>_<count>.jsonl,
e.g. data.jsonl -n 100 -> data_100.jsonl.

Usage:
    python test.py <path/to/file.jsonl> -n 100 [--seed 42]
"""

import argparse
import json
import random
from pathlib import Path


def sample_jsonl(jsonl_path, num_samples, seed=None):
    """Return `num_samples` randomly picked rows from a JSONL file.

    Uses reservoir sampling, so the file is streamed once and only the kept
    rows live in memory -- safe for JSONL files too large to load at once.
    If the file holds fewer than `num_samples` non-empty lines, all rows are
    returned. The returned rows are parsed dicts, in random order.
    """
    if num_samples <= 0:
        return []

    rng = random.Random(seed)
    reservoir = []
    seen = 0

    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            seen += 1
            if len(reservoir) < num_samples:
                reservoir.append(line)
            else:
                j = rng.randrange(seen)
                if j < num_samples:
                    reservoir[j] = line

    return [json.loads(line) for line in reservoir]


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample rows from a JSONL file."
    )
    parser.add_argument("jsonl_path", type=Path, help="Path to the JSONL file.")
    parser.add_argument(
        "-n", "--num-samples", type=int, default=10, help="How many rows to sample (default: 10)."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible sampling."
    )
    args = parser.parse_args()

    rows = sample_jsonl(args.jsonl_path, args.num_samples, seed=args.seed)
    if not rows:
        print(f"No rows sampled from {args.jsonl_path}.")
        return

    # Sits next to the source file, named <stem>_<count><suffix>.
    src = args.jsonl_path
    out_path = src.with_name(f"{src.stem}_{len(rows)}{src.suffix}")
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} sampled rows to {out_path}")


if __name__ == "__main__":
    main()
