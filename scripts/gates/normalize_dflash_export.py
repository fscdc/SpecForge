"""Normalize a SpecForge DFlash-family HF export for SGLang DFLASH loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _check_domino_serving_fields(method_config: Dict[str, Any], block_size) -> None:
    """Fail on a Domino config SGLang's Domino DFLASH parser would refuse."""
    if not isinstance(method_config.get("shift_label"), bool):
        raise ValueError("Domino export needs a bool dflash_config.shift_label")
    prefix_len = method_config.get("pure_draft_prefix_len")
    if isinstance(prefix_len, bool) or prefix_len != 1:
        raise ValueError(
            "SGLang serves Domino only with dflash_config.pure_draft_prefix_len=1, "
            f"got {prefix_len!r}"
        )
    for key in ("gru_hidden_dim", "emb_dim"):
        value = method_config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Domino export needs a positive dflash_config.{key}")
    if block_size <= 1:
        raise ValueError(f"Domino export needs block_size > 1, got {block_size}")


def normalize_export(config_path: str, expected_block_size: int) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    block_size = config.get("block_size")
    if block_size != expected_block_size:
        raise ValueError(
            f"exported block_size={block_size!r}, expected {expected_block_size}"
        )
    method_config = config.get("dflash_config") or {}
    projector_type = method_config.get("projector_type", "dflash")
    if projector_type not in {"dflash", "domino", "mmflash"}:
        raise ValueError(
            "export is not DFlash-family: "
            f"dflash_config.projector_type={projector_type!r}"
        )
    if projector_type == "domino":
        _check_domino_serving_fields(method_config, block_size)

    config["architectures"] = ["DFlashDraftModel"]
    config.pop("auto_map", None)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--block-size", type=int, required=True)
    args = parser.parse_args()
    normalize_export(args.config, args.block_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
