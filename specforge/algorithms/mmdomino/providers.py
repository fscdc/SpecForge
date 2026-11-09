"""Built-in MMDomino registration and executable providers.

MMDomino trains Domino's drafter (``DominoDraftModel``: the DFlash backbone
plus the GRU logits correction) with the MMFlash objective on MMFlash's data
path. Everything model-side is Domino's -- the draft architecture and config,
capture layers, ``lambda_base`` schedule and resume semantics -- and
everything data-side is MMFlash's: the image capture through the multimodal
input adapter, the per-token ``visual_score`` passthrough the objective weights
by, and the collator that dequantises it. See
``specforge.algorithms.common.mmdomino_model`` for the objective.
"""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.defaults import no_missing_checkpoint_keys
from specforge.algorithms.common.dflash_family_data import (
    NORMALIZER_ID,
    build_mmflash_collator,
    build_offline_normalizer,
    build_offline_reader,
)
from specforge.algorithms.common.mm_server_input import build_image_input_adapter
from specforge.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ModelProvider,
    OfflineCaptureLayout,
    OfflineDataProvider,
    ServerCaptureLayout,
    ServerStreamingProvider,
    StepProvider,
    make_registration,
)
from specforge.algorithms.contracts import (
    AlgorithmCapabilities,
    AlgorithmSpec,
    DraftRequirement,
    FeatureContract,
    FeatureMode,
    OfflineStorageContract,
)

ALGORITHM_NAME = "mmdomino"
DRAFT_ARCHITECTURE = "DominoDraftModel"


def build_step(
    wrapped_model,
    *,
    target_head=None,
    lambda_start=1.0,
    decay_ratio=0.5,
):
    del target_head
    from specforge.training.strategies.base import MMDominoTrainStrategy

    return MMDominoTrainStrategy(
        wrapped_model,
        lambda_start=lambda_start,
        decay_ratio=decay_ratio,
    )


def step_options(config):
    from specforge.algorithms.model_providers import domino_strategy_kwargs

    return domino_strategy_kwargs(config)


def resume_contract(config, draft_model, training_model):
    """Persist Domino's model, sampling and schedule semantics plus the MMFlash loss."""

    return {
        "mmdomino_draft_num_hidden_layers": int(draft_model.config.num_hidden_layers),
        "mmdomino_target_layer_ids": tuple(
            int(layer_id) for layer_id in draft_model.target_layer_ids
        ),
        "mmdomino_block_size": int(training_model.block_size),
        "mmdomino_mask_token_id": int(training_model.mask_token_id),
        "mmdomino_attention_backend": str(training_model.attention_backend),
        "mmdomino_num_anchors": int(training_model.num_anchors),
        "mmdomino_loss_decay_gamma": training_model.loss_decay_gamma,
        "mmdomino_shift_label": bool(training_model.shift_label),
        "mmdomino_pure_draft_prefix_len": int(draft_model.pure_draft_prefix_len),
        "mmdomino_lambda_base_start": float(config.training.lambda_base_start),
        "mmdomino_lambda_base_decay_ratio": float(
            config.training.lambda_base_decay_ratio
        ),
        "mmdomino_loss_type": str(training_model.loss_type),
        "mmdomino_visual_alpha": float(training_model.visual_alpha),
        "mmdomino_smoothing": float(training_model.mmflash_smoothing),
    }


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_registered_draft

    return build_registered_draft(config, draft_config)


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.model_providers import build_mmdomino_model

    return build_mmdomino_model(
        config,
        draft_model,
        draft_config,
        target_config,
        tokenizer,
    )


def resolve_capture_layers(config, draft_config, target_config):
    from specforge.algorithms.model_providers import resolve_dflash_capture_layers

    return resolve_dflash_capture_layers(config, draft_config, target_config)


def minimum_loss_tokens(config, draft_config):
    from specforge.algorithms.model_providers import dflash_min_loss_tokens

    return dflash_min_loss_tokens(config, draft_config)


def needs_input_tools(config, draft_model):
    from specforge.algorithms.model_providers import dflash_needs_input_tools

    return dflash_needs_input_tools(config, draft_model)


def algorithm_spec() -> AlgorithmSpec:
    ready = {"input_ids", "loss_mask", "hidden_states"}
    return AlgorithmSpec(
        name=ALGORITHM_NAME,
        draft=DraftRequirement(
            compatible_architectures={DRAFT_ARCHITECTURE},
            default_architecture=DRAFT_ARCHITECTURE,
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.OFFLINE,
                modality="text",
                required_tensors=ready,
                storage=OfflineStorageContract(
                    format="specforge_hidden_states_v1",
                    required_tensors=ready,
                    normalizer=NORMALIZER_ID,
                ),
            ),
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="text",
                required_tensors=ready,
            ),
            # as MMFlash: image capture also carries the visual-dependency
            # channel, required so a capture without it fails as a wiring error
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="image",
                required_tensors=ready | {"visual_score"},
            ),
        ),
        capabilities=AlgorithmCapabilities(
            attention_backends={"eager", "sdpa", "flex_attention"},
        ),
    )


def algorithm_providers() -> AlgorithmProviders:
    collator = build_mmflash_collator
    return AlgorithmProviders(
        algorithm_name=ALGORITHM_NAME,
        step=StepProvider(
            build=build_step,
            options=step_options,
            resume_contract=resume_contract,
            allowed_missing_checkpoint_keys=no_missing_checkpoint_keys,
            uses_external_target_head=False,
        ),
        model=ModelProvider(
            draft_config=DraftConfigProvider(
                architecture=DRAFT_ARCHITECTURE,
                expected_auto_map_model="domino.DominoDraftModel",
            ),
            build_draft=build_draft,
            build_training_model=build_training_model,
            resolve_capture_layers=resolve_capture_layers,
            minimum_loss_tokens=minimum_loss_tokens,
            needs_input_tools=needs_input_tools,
            default_dataloader_num_workers=8,
        ),
        offline=(
            OfflineDataProvider(
                modality="text",
                normalizer_id=NORMALIZER_ID,
                capture_layout=OfflineCaptureLayout(
                    capture_method="dflash",
                    aux_feature="hidden_states",
                    last_hidden_feature=None,
                    passthrough=(
                        ("input_ids", "input_ids"),
                        ("loss_mask", "loss_mask"),
                    ),
                ),
                build_reader=partial(build_offline_reader, ALGORITHM_NAME),
                build_normalizer=build_offline_normalizer,
                build_collator=collator,
            ),
        ),
        server_streaming=(
            ServerStreamingProvider(
                modality="text",
                capture_method="dflash",
                target_representation=None,
                layout=ServerCaptureLayout(
                    aux_feature="hidden_states",
                    last_hidden_feature=None,
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                    ),
                ),
                build_collator=collator,
            ),
            ServerStreamingProvider(
                modality="image",
                capture_method="dflash",
                target_representation=None,
                layout=ServerCaptureLayout(
                    aux_feature="hidden_states",
                    last_hidden_feature=None,
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                        # quantised int64 on the wire, see specforge.data.visual_score
                        ("visual_score", "visual_score", ()),
                    ),
                ),
                build_collator=collator,
                build_input_adapter=build_image_input_adapter,
            ),
        ),
    )


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())


__all__ = ["algorithm_providers", "algorithm_spec", "create_registration"]
