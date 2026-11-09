# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Export a DataFlow training checkpoint to an SGLang spec-decoder draft directory.

Weight-name compatibility is the silent-failure risk here: a key the serving
loader does not expect is skipped (or zero-filled) without an error. The
per-architecture trainer-key -> serving-key map therefore lives in
``WEIGHT_MAPS`` below, and this exporter validates the produced state against it.

For ``LlamaForCausalLMEagle3`` the map is the identity: SpecForge's draft module
names (``midlayer.*`` / ``fc`` / ``norm`` / ``lm_head`` + the ``t2d``/``d2t``
buffers) are exactly what SGLang's EAGLE3 draft loader reads.

Domino and MMDomino checkpoints (``DominoDraftModel``: the DFlash backbone plus
Domino's GRU logits correction) export as SGLang DFLASH drafts, served with
``--speculative-algorithm DFLASH`` by an SGLang that carries the Domino port.
Their module names (``layers.*`` / ``fc`` / ``hidden_norm`` / ``norm`` /
``prefix_gru`` / ``embed_proj``) are already the serving names; the config is
what changes. SGLang resolves the draft class by name and has no
``DominoDraftModel``: its ``DFlashDraftModel`` builds the Domino head from
``dflash_config`` (``projector_type="domino"``, ``shift_label``,
``pure_draft_prefix_len``, ``gru_hidden_dim``, ``emb_dim``), the layout of the
Domino authors' checkpoints. An SGLang without the port ignores those weights
and serves the backbone as plain DFLASH, so serve these exports only with it.
DFlash and MMFlash keep their ``--to hf`` export, normalized for SGLang by
``scripts/gates/normalize_dflash_export.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

from specforge.export.checkpoint_io import materialize_draft, resolve_training_state

#: per-architecture trainer-key -> serving-key renames ({} = identity).
WEIGHT_MAPS: Dict[str, Dict[str, str]] = {
    "LlamaForCausalLMEagle3": {},
}

#: keys the sglang EAGLE3 spec-decoder loader requires in a draft checkpoint.
_REQUIRED_SERVING_KEYS = ("fc.weight", "norm.weight", "lm_head.weight", "t2d", "d2t")

#: training strategies whose checkpoints export as SGLang DFLASH Domino drafts.
DOMINO_STRATEGIES = ("domino", "mmdomino")

#: keys SGLang's Domino DFLASH loader requires; it refuses a draft without the
#: GRU correction head.
_REQUIRED_DOMINO_SERVING_KEYS = (
    "fc.weight",
    "hidden_norm.weight",
    "norm.weight",
    "prefix_gru.weight_ih_l0",
    "prefix_gru.weight_hh_l0",
    "embed_proj.0.weight",
    "embed_proj.2.weight",
)

#: draft semantics a Domino-family run persists in its checkpoint as
#: ``{strategy}_{field}`` (see the providers' ``resume_contract``).
_DOMINO_CONTRACT_FIELDS = (
    "shift_label",
    "pure_draft_prefix_len",
    "mask_token_id",
    "target_layer_ids",
    "block_size",
)


def _serving_state(state_dict, weight_map: Dict[str, str]):
    out = {weight_map.get(k, k): v for k, v in state_dict.items()}
    bad_prefix = [k for k in out if k.startswith("draft_model.")]
    if bad_prefix:
        raise ValueError(
            f"serving state still carries trainer prefixes: {sorted(bad_prefix)}"
        )
    missing = [k for k in _REQUIRED_SERVING_KEYS if weight_map.get(k, k) not in out]
    if missing:
        raise ValueError(
            f"serving state is missing required keys {missing}; the sglang loader "
            f"would silently produce a broken draft. Present: {sorted(out)[:8]}..."
        )
    return out


def export_to_sglang(
    checkpoint_path: str,
    draft_config_path: str,
    output_dir: str,
    *,
    vocab_mapping_path: Optional[str] = None,
    weight_map: Optional[Dict[str, str]] = None,
) -> str:
    """Write an sglang-loadable draft directory; returns ``output_dir``.

    ``weight_map`` overrides the per-architecture entry in :data:`WEIGHT_MAPS`.
    ``vocab_mapping_path`` refreshes the ``t2d``/``d2t`` buffers when the
    checkpoint predates them. Neither applies to Domino and MMDomino
    checkpoints, which export as SGLang DFLASH drafts (see the module doc).
    """
    state = resolve_training_state(checkpoint_path)
    if state.get("strategy") in DOMINO_STRATEGIES:
        if vocab_mapping_path is not None or weight_map is not None:
            raise ValueError(
                "Domino drafts serve under their own module names and carry no "
                "vocab mapping; drop vocab_mapping_path and weight_map"
            )
        return _export_domino(state, draft_config_path, output_dir)
    if state.get("strategy") != "eagle3":
        raise ValueError(
            "the specialized SGLang exporter currently supports EAGLE3, Domino "
            f"and MMDomino checkpoints only, got strategy={state.get('strategy')!r}; "
            "use --to hf for DFlash, MMFlash and P-EAGLE draft model directories"
        )
    model = materialize_draft(
        state, draft_config_path, vocab_mapping_path=vocab_mapping_path
    )
    if weight_map is None:
        weight_map = WEIGHT_MAPS.get(type(model).__name__, {})
    # the model's state dict includes any refreshed t2d/d2t buffers; drop the
    # embeddings exactly as the trainer-side checkpoint filter does.
    full = {k: v for k, v in model.state_dict().items() if "embed" not in k.lower()}
    model.save_pretrained(output_dir, state_dict=_serving_state(full, weight_map))
    return output_dir


def _domino_dflash_config(model, state: Dict[str, Any]) -> Dict[str, Any]:
    """The ``dflash_config`` SGLang's Domino path parses, checked against the run.

    Values come from the materialized draft, i.e. what the weights were built
    for, and must agree with the semantics the run persisted in its resume
    contract (``{strategy}_shift_label`` and friends): a ``--draft-config``
    that disagrees with the run loads cleanly and serves a near-zero
    acceptance draft. The mask token id the run resolved fills in for a draft
    config without one, as training wrote it into the live config.
    """
    strategy = state["strategy"]
    method = dict(getattr(model.config, "dflash_config", None) or {})
    mask_token_id = method.get("mask_token_id")
    if mask_token_id is None:
        mask_token_id = state.get(f"{strategy}_mask_token_id")
    if mask_token_id is None:
        raise ValueError(
            "neither the draft config nor the checkpoint names the mask token; "
            "set dflash_config.mask_token_id"
        )
    served = {
        "projector_type": "domino",
        "shift_label": bool(model.shift_label),
        "pure_draft_prefix_len": int(model.pure_draft_prefix_len),
        "gru_hidden_dim": int(model.gru_hidden_dim),
        "emb_dim": int(model.emb_dim),
        "mask_token_id": int(mask_token_id),
        "target_layer_ids": [int(layer_id) for layer_id in model.target_layer_ids],
    }
    block_size = int(model.block_size)
    exported = dict(served, block_size=block_size)
    mismatched = {}
    for field in _DOMINO_CONTRACT_FIELDS:
        recorded = state.get(f"{strategy}_{field}")
        if recorded is None:  # checkpoints from before the resume contract
            continue
        if isinstance(recorded, tuple):
            recorded = list(recorded)
        if recorded != exported[field]:
            mismatched[field] = {
                "draft_config": exported[field],
                "checkpoint": recorded,
            }
    if mismatched:
        raise ValueError(
            f"--draft-config disagrees with the {strategy} run that wrote the "
            f"checkpoint: {mismatched}"
        )
    # SGLang's Domino rollout is written for one pure-draft slot only.
    if served["pure_draft_prefix_len"] != 1:
        raise ValueError(
            "SGLang serves Domino drafts only with pure_draft_prefix_len=1, got "
            f"{served['pure_draft_prefix_len']}"
        )
    if block_size <= 1:
        raise ValueError(f"Domino drafts need block_size > 1, got {block_size}")
    return {**method, **served}


def _export_domino(
    state: Dict[str, Any], draft_config_path: str, output_dir: str
) -> str:
    from specforge.modeling.draft.domino import DominoDraftModel

    model = materialize_draft(state, draft_config_path)
    if not isinstance(model, DominoDraftModel):
        raise ValueError(
            f"{state['strategy']} checkpoints export as Domino drafts, but the "
            f"draft config builds {type(model).__name__}"
        )
    method_config = _domino_dflash_config(model, state)
    serving_state = model.state_dict()
    missing = [key for key in _REQUIRED_DOMINO_SERVING_KEYS if key not in serving_state]
    if missing:
        raise ValueError(
            f"Domino serving state is missing required keys {missing}; SGLang "
            "would refuse the draft"
        )
    model.save_pretrained(output_dir, state_dict=serving_state)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    config["architectures"] = ["DFlashDraftModel"]
    config.pop("auto_map", None)  # no domino.py ships with the export
    config["dflash_config"] = method_config
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=export_to_sglang.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--draft-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-mapping", default=None)
    args = parser.parse_args(argv)
    out = export_to_sglang(
        args.checkpoint,
        args.draft_config,
        args.output_dir,
        vocab_mapping_path=args.vocab_mapping,
    )
    print(f"exported sglang draft to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
