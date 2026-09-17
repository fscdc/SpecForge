# coding=utf-8
"""MMFlash was forked from DFlash; this test pins the parts that still match.

MMFlash has since diverged on purpose -- its objective weights image rows by a
per-token visual-dependency channel (``visual_score``), so the training model,
the strategy's ``forward_loss``, the image capture layout/contract, the
collator and the resume contract are deliberately different. The assertions
that covered those were removed as documented departures. What remains must
still match DFlash exactly, apart from the name: the draft architecture and its
source, the draft-config defaults, the text/offline capture layouts, the
capture protocol and the strategy's class shape.

Tokens that are format rather than identity are exempt from the rename and are
expected to appear verbatim in both: the ``dflash_config`` draft-config key,
the ``"dflash"`` loss-type name, the ``capture_method="dflash"`` wire protocol,
and the shared ``dflash_kernels`` / ``dflash_family_data`` modules.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from specforge.algorithms.builtin import builtin_algorithm_registry

REPO_ROOT = Path(__file__).resolve().parents[2]

# Every way the name is spelled, longest first so "DFlashDraftModel" is rewritten
# before the bare "DFlash" inside it would be.
RENAMES = (
    ('"dflash.DFlashDraftModel"', '"mmflash.MMFlashDraftModel"'),
    ("DFlashDraftModel", "MMFlashDraftModel"),
    ("OnlineDFlashModel", "OnlineMMFlashModel"),
    ("DFlashTrainStrategy", "MMFlashTrainStrategy"),
    ("Qwen3DFlashDecoderLayer", "Qwen3MMFlashDecoderLayer"),
    ("Qwen3DFlashAttention", "Qwen3MMFlashAttention"),
    ("create_dflash_sdpa_mask", "create_mmflash_sdpa_mask"),
    ("create_dflash_block_mask", "create_mmflash_block_mask"),
    ("_dflash_objective_chunk_terms", "_mmflash_objective_chunk_terms"),
    ("dflash_mask_mod", "mmflash_mask_mod"),
    ("dflash_attn_mask", "mmflash_attn_mask"),
    ("resolve_dflash_kernels", "resolve_mmflash_kernels"),
    ("resolve_dflash_capture_layers", "resolve_mmflash_capture_layers"),
    ("populate_dflash_generated_config", "populate_mmflash_generated_config"),
    ("apply_dflash_overrides", "apply_mmflash_overrides"),
    ("dflash_needs_input_tools", "mmflash_needs_input_tools"),
    ("dflash_min_loss_tokens", "mmflash_min_loss_tokens"),
    ("build_dflash_draft", "build_mmflash_draft"),
    ("build_dflash_model", "build_mmflash_model"),
    ("specforge.modeling.draft.dflash import", "specforge.modeling.draft.mmflash import"),
    (
        "common.dflash_family_model import OnlineDFlashModel",
        "common.mmflash_model import OnlineMMFlashModel",
    ),
    ('strategy != "dflash"', 'strategy != "mmflash"'),
    ('training.strategy=dflash"', 'training.strategy=mmflash"'),
    ('ALGORITHM_NAME = "dflash"', 'ALGORITHM_NAME = "mmflash"'),
    ('name = "dflash"', 'name = "mmflash"'),
    ("dflash_kernels=", "mmflash_kernels="),
    ("dflash_kernels or", "mmflash_kernels or"),
    ("dflash_kernels:", "mmflash_kernels:"),
    ("dflash_model", "mmflash_model"),
    ('"dflash_', '"mmflash_'),  # resume-contract keys
)


def _normalise(text: str) -> str:
    """DFlash source with every identity token rewritten to its MMFlash spelling.

    Docstrings and comments are dropped: the clones deliberately explain that
    they are clones, and the wording of that explanation is not what parity is
    about. Code is.
    """
    for old, new in RENAMES:
        text = text.replace(old, new)
    text = re.sub(r'"""[\s\S]*?"""', '', text)
    text = "\n".join(line.split("  #")[0].rstrip() for line in text.splitlines())
    text = re.sub(r"^\s*#.*$", "", text, flags=re.M)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def _code_of(path: str, *, start: str | None = None, stop: str | None = None) -> str:
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    if start is not None:
        text = text[text.index(start):]
    if stop is not None:
        text = text[: text.index(stop)]
    return text


class MMFlashParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry = builtin_algorithm_registry()
        cls.dflash = registry.resolve("dflash")
        cls.mmflash = registry.resolve("mmflash")

    # -- registration --------------------------------------------------------
    def test_spec_matches_dflash_except_name_and_architecture(self):
        d, m = self.dflash.spec, self.mmflash.spec
        self.assertEqual("mmflash", m.name)
        self.assertEqual("MMFlashDraftModel", m.draft.default_architecture)
        self.assertEqual({"MMFlashDraftModel"}, set(m.draft.compatible_architectures))
        self.assertEqual(d.draft.supported_overrides, m.draft.supported_overrides)
        self.assertEqual(d.draft.fixed_override_values, m.draft.fixed_override_values)
        self.assertEqual(d.capabilities, m.capabilities)
        # text and offline contracts are shared; the image contract additionally
        # requires the visual_score channel (the documented MMFlash departure)
        by_key = lambda spec: {(c.mode, c.modality): c for c in spec.feature_contracts}  # noqa: E731
        dc, mc = by_key(d), by_key(m)
        self.assertEqual(set(dc), set(mc))
        for key, contract in dc.items():
            if key[1] == "image":
                self.assertEqual(
                    contract.required_tensors | {"visual_score"},
                    mc[key].required_tensors,
                )
            else:
                self.assertEqual(contract, mc[key])

    def test_providers_match_dflash_except_identity(self):
        d, m = self.dflash.providers, self.mmflash.providers
        self.assertEqual("mmflash", m.algorithm_name)
        self.assertEqual("MMFlashDraftModel", m.model.draft_config.architecture)
        self.assertEqual(
            "mmflash.MMFlashDraftModel", m.model.draft_config.expected_auto_map_model
        )
        dd = d.model.draft_config.target_defaults
        md = m.model.draft_config.target_defaults
        self.assertEqual(
            (dd.model_type, dd.num_hidden_layers, dd.draft_vocab_size),
            (md.model_type, md.num_hidden_layers, md.draft_vocab_size),
        )
        self.assertEqual(
            d.model.default_dataloader_num_workers,
            m.model.default_dataloader_num_workers,
        )
        self.assertEqual(
            d.model.allow_missing_warm_start_embedding,
            m.model.allow_missing_warm_start_embedding,
        )
        self.assertEqual(d.step.uses_external_target_head, m.step.uses_external_target_head)
        self.assertEqual(d.vocab_mapping_modes, m.vocab_mapping_modes)
        # the capture protocol is shared, exactly as Domino and DSpark share it;
        # the text layout is identical, the image layout adds visual_score
        for modality in ("text", "image"):
            ds, ms = d.server_streaming_for(modality), m.server_streaming_for(modality)
            self.assertEqual("dflash", ms.capture_method)
            self.assertEqual(ds.target_representation, ms.target_representation)
            self.assertEqual(ds.layout.aux_feature, ms.layout.aux_feature)
            self.assertEqual(ds.layout.last_hidden_feature, ms.layout.last_hidden_feature)
            extra = (("visual_score", "visual_score", ()),) if modality == "image" else ()
            self.assertEqual(ds.layout.passthrough + extra, ms.layout.passthrough)
        self.assertEqual(d.server_streaming_for("text").layout, m.server_streaming_for("text").layout)
        do, mo = d.offline_for("text"), m.offline_for("text")
        self.assertEqual(do.normalizer_id, mo.normalizer_id)
        self.assertEqual(do.capture_layout, mo.capture_layout)

    def test_resume_contract_extends_dflash_contract_with_visual_alpha(self):
        from types import SimpleNamespace

        draft = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=3), target_layer_ids=[1, 5, 9])
        model = SimpleNamespace(block_size=16, mask_token_id=7, attention_backend="sdpa", num_anchors=32,
                                loss_decay_gamma=None, loss_type="dflash", dpace_alpha=0.5, visual_alpha=1.0,
                                mmflash_smoothing=0.5)
        d = self.dflash.providers.step.resume_contract(None, draft, model)
        m = self.mmflash.providers.step.resume_contract(None, draft, model)
        renamed = {k.replace("dflash_", "mmflash_", 1): v for k, v in d.items()}
        self.assertEqual({**renamed, "mmflash_visual_alpha": 1.0, "mmflash_smoothing": 0.5}, m)
        self.assertTrue(all(k.startswith("mmflash_") for k in m))

    # -- step strategy ------------------------------------------------------
    def test_strategy_class_is_a_renamed_dflash_strategy(self):
        from specforge.training.strategies.base import DFlashTrainStrategy, MMFlashTrainStrategy

        self.assertEqual("mmflash", MMFlashTrainStrategy.name)
        # visual_score is optional at the batch level (offline / text capture /
        # evaluation carry none), so the required set is still DFlash's
        self.assertEqual(DFlashTrainStrategy.required_features, MMFlashTrainStrategy.required_features)
        self.assertEqual(DFlashTrainStrategy.__mro__[1:], MMFlashTrainStrategy.__mro__[1:])

    # -- source text -----------------------------------------------------------
    def test_draft_model_source_is_dflash_modulo_rename(self):
        self.assertEqual(_normalise(_code_of("specforge/modeling/draft/dflash.py")),
                         _normalise(_code_of("specforge/modeling/draft/mmflash.py")))

    def test_block_forward_source_is_dflash_modulo_rename(self):
        """Everything up to the objective is still DFlash: masks, anchors, block forward."""
        dflash = _code_of("specforge/algorithms/common/dflash_family_model.py",
                          start="def compute_accept_len", stop="    def _dflash_objective_chunk_terms")
        mmflash = _code_of("specforge/algorithms/common/mmflash_model.py",
                           start="def compute_accept_len", stop="    def _base_weights")
        # the constructor gained visual_alpha and mmflash_smoothing; strip
        # those lines before comparing
        mmflash = "\n".join(
            line for line in mmflash.splitlines()
            if "visual_alpha" not in line
            and "mmflash_smoothing" not in line
            and "_objective_printed" not in line
        )
        self.assertEqual(_normalise(dflash), _normalise(mmflash))


if __name__ == "__main__":
    unittest.main()
