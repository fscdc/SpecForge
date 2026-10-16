# coding=utf-8
"""One resolved contract shared by server-capture launch and production."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from specforge.algorithms.registry import AlgorithmRegistration
from specforge.config import Config


@dataclass(frozen=True)
class ServerCaptureContract:
    method: str
    aux_layer_ids: tuple[int, ...]
    target_hidden_size: int
    target_vocab_size: int
    draft_vocab_size: int
    #: The target config's ``architectures``: the SGLang model class the capture
    #: server instantiates, which decides its model-specific launch rules.
    target_architectures: tuple[str, ...] = ()


def resolve_server_capture_contract(
    cfg: Config,
    *,
    algorithm: AlgorithmRegistration,
) -> ServerCaptureContract:
    """Resolve engine flags and feature dimensions from canonical model config."""
    from specforge.modeling.target.target_utils import (
        load_target_config,
        target_text_config,
        target_vocab_size,
    )
    from specforge.training.model_loading import draft_config_dict

    streaming = algorithm.providers.server_streaming_for(cfg.model.input_modality)

    target_cfg = load_target_config(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    architectures = tuple(getattr(target_cfg, "architectures", None) or ())
    target_cfg = target_text_config(target_cfg)
    model_provider = algorithm.providers.model
    draft_cfg = draft_config_dict(cfg, provider=model_provider.draft_config)
    layers = model_provider.resolve_capture_layers(cfg, draft_cfg, target_cfg)
    if not layers:
        raise ValueError("draft config does not define target capture layer ids")
    if any(
        isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
        for layer in layers
    ):
        raise ValueError(
            "resolved server capture layer ids must be non-negative integers, "
            f"got {layers!r}"
        )
    if len(set(layers)) != len(layers):
        raise ValueError(
            f"resolved server capture layer ids must be unique, got {layers!r}"
        )
    _check_layers_against_target(cfg, layers, target_cfg)

    return ServerCaptureContract(
        method=streaming.capture_method,
        aux_layer_ids=tuple(layers),
        target_hidden_size=int(target_cfg.hidden_size),
        target_vocab_size=target_vocab_size(target_cfg),
        draft_vocab_size=int(
            draft_cfg.get("draft_vocab_size") or draft_cfg["vocab_size"]
        ),
        target_architectures=architectures,
    )


def _check_layers_against_target(cfg: Config, layers, target_cfg) -> None:
    """Fail on layer ids past the target's depth; warn on its last layer.

    Capture id ``k`` is the output of decoder layer ``k``, so an id must be
    below ``num_hidden_layers``; otherwise the run would only fail later, per
    row, on the captured feature width. The last layer is captured before the
    final norm by SGLang but is HF's post-norm ``hidden_states[-1]``.
    """
    num_layers = getattr(target_cfg, "num_hidden_layers", None)
    if isinstance(num_layers, bool) or not isinstance(num_layers, int):
        return
    out_of_range = sorted(layer for layer in layers if layer >= num_layers)
    if out_of_range:
        raise ValueError(
            f"capture layer ids {out_of_range!r} are out of range for "
            f"{cfg.model.target_model_path!r}, whose {num_layers} decoder layers "
            f"are numbered 0..{num_layers - 1}"
        )
    if num_layers - 1 in layers:
        warnings.warn(
            f"capture layer id {num_layers - 1} is the last decoder layer of "
            f"{cfg.model.target_model_path!r}: SGLang captures it before the "
            "final norm, while HF hidden_states[-1] is after it, so HF-based "
            "tools see a different feature for it",
            stacklevel=2,
        )


__all__ = ["ServerCaptureContract", "resolve_server_capture_contract"]
