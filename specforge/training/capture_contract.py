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
    _check_layers_against_target(cfg, layers, target_cfg, architectures)

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


def _check_layers_against_target(
    cfg: Config, layers, target_cfg, architectures: tuple[str, ...] = ()
) -> None:
    """Fail on layer ids past the target's depth or, outside Gemma 4, on its last layer.

    Capture id ``k`` is the output of decoder layer ``k``, so an id must be
    below ``num_hidden_layers``; otherwise the run would only fail later, per
    row, on the captured feature width. SGLang taps id ``k`` at the input of
    layer ``k + 1``, which the last layer does not have: Qwen3.5/Qwen3-VL
    targets then fail at capture-server start (``self.layers[k + 1]``) and the
    dense Llama/Qwen ones never capture it. Only the Gemma 4 models read it
    after the layer loop, before the final norm -- where HF's
    ``hidden_states[-1]`` is after it, hence the warning there.
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
        model_type = str(getattr(target_cfg, "model_type", "") or "")
        if not (
            model_type.startswith("gemma4")
            or any(str(name).startswith("Gemma4") for name in architectures)
        ):
            raise ValueError(
                f"capture layer id {num_layers - 1} is the last decoder layer of "
                f"{cfg.model.target_model_path!r}; SGLang captures id k at the input "
                "of layer k + 1, which only its Gemma 4 models provide for the last "
                f"layer. Use ids below {num_layers - 1}"
            )
        warnings.warn(
            f"capture layer id {num_layers - 1} is the last decoder layer of "
            f"{cfg.model.target_model_path!r}: SGLang captures it before the "
            "final norm, while HF hidden_states[-1] is after it, so HF-based "
            "tools see a different feature for it",
            stacklevel=2,
        )


__all__ = ["ServerCaptureContract", "resolve_server_capture_contract"]
