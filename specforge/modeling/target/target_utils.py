import gc
import glob
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from transformers import AutoConfig


class _RawConfigShim:
    """Attribute view for released checkpoints with an unregistered model type."""

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name):
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(name) from None
        return _RawConfigShim(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict:
        return dict(self._data)


def load_target_config(
    model_path: str,
    *,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """Load a target config, falling back to its public raw ``config.json``."""

    try:
        return AutoConfig.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
    except (ValueError, KeyError, OSError) as auto_error:
        if os.path.isdir(model_path):
            config_path = os.path.join(model_path, "config.json")
        elif os.path.isfile(model_path):
            config_path = model_path
        else:
            try:
                config_path = hf_hub_download(
                    repo_id=model_path,
                    filename="config.json",
                    cache_dir=cache_dir,
                )
            except Exception:
                raise auto_error
        try:
            with open(config_path, encoding="utf-8") as config_file:
                return _RawConfigShim(json.load(config_file))
        except (OSError, ValueError):
            raise auto_error


def target_text_config(config):
    return getattr(config, "text_config", config)


def target_vocab_size(config) -> int:
    text_config = target_text_config(config)
    return int(
        getattr(text_config, "padded_vocab_size", None) or text_config.vocab_size
    )


# Targets whose input-embedding module multiplies every lookup by
# sqrt(hidden_size). SGLang's DFLASH worker embeds draft blocks through that
# module (target_model.get_input_embeddings()), so the trainer must too. Gemma
# and Gemma 2 apply their normalizer in the model forward instead, so their
# embedding modules, and the draft inputs SGLang builds from them, are unscaled.
_SCALED_EMBEDDING_MODEL_TYPES = frozenset(
    {"gemma3", "gemma3_text", "gemma3n", "gemma3n_text", "gemma4", "gemma4_text"}
)


def target_embedding_scale(config) -> Optional[float]:
    """The factor the target's own embedding module applies, or None.

    This is SGLang's Python float ``hidden_size ** 0.5`` (50.596 for hidden
    2560), not HF's copy of it cast to the weight dtype (50.5 in bf16):
    SGLang's embedding is what serves DFlash-family drafts.
    """
    text_config = target_text_config(config)
    model_types = {
        getattr(config, "model_type", None),
        getattr(text_config, "model_type", None),
    }
    if model_types.isdisjoint(_SCALED_EMBEDDING_MODEL_TYPES):
        return None
    return float(text_config.hidden_size) ** 0.5


def target_logit_softcap(config) -> Optional[float]:
    """The target's ``final_logit_softcapping``, or None when it has none.

    The served logits are ``cap * tanh(logits / cap)``. A missing, zero or
    negative cap is off, as in SGLang's logits processor.
    """
    cap = getattr(target_text_config(config), "final_logit_softcapping", None)
    if cap is None:
        cap = getattr(config, "final_logit_softcapping", None)
    if cap is None or float(cap) <= 0:
        return None
    return float(cap)


class _ScaledEmbedding(nn.Embedding):
    """An embedding whose lookups are multiplied by a Python-float scale.

    Mirrors SGLang's ``Gemma3TextScaledWordEmbedding`` (also used by Gemma 3n
    and Gemma 4). The scale is a plain attribute, not a buffer, so moving the
    module to bf16 cannot round it, and the weight itself stays raw for a tied
    LM head that shares it.
    """

    def __init__(self, *args, embed_scale: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_scale = float(embed_scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, embed_scale={self.embed_scale}"


class TargetEmbeddingsAndHead(nn.Module):
    """
    Efficiently loads only the embedding layer and lm_head from a pretrained model.
    Handles safetensors slicing and Weight Tying correctly.

    ``embed_scale`` multiplies embedding lookups (never the weight, which a tied
    LM head shares): None takes the target's own scale (sqrt(hidden_size) for
    Gemma 3/3n/4, otherwise none), and 1.0 turns scaling off.
    """

    def __init__(self, config, embed_scale: Optional[float] = None):
        super().__init__()
        self.config = config
        text_config = target_text_config(config)
        vocab_size = target_vocab_size(text_config)
        hidden_size = int(text_config.hidden_size)
        if embed_scale is None:
            embed_scale = target_embedding_scale(config)
        padding_idx = getattr(text_config, "pad_token_id", None)
        if embed_scale is None or float(embed_scale) == 1.0:
            self.embed_tokens = nn.Embedding(
                vocab_size,
                hidden_size,
                padding_idx=padding_idx,
            )
        else:
            self.embed_tokens = _ScaledEmbedding(
                vocab_size,
                hidden_size,
                padding_idx=padding_idx,
                embed_scale=embed_scale,
            )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        embed_key: Optional[str] = None,
        lm_head_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
        embed_scale: Optional[float] = None,
    ) -> "TargetEmbeddingsAndHead":

        # 1. Load Config
        config = load_target_config(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        instance = cls(config, embed_scale=embed_scale)
        if isinstance(instance.embed_tokens, _ScaledEmbedding):
            print(
                "Scaling target embedding lookups by "
                f"{instance.embed_tokens.embed_scale!r}; the LM head stays unscaled."
            )

        if embed_key is None:
            embed_key = "model.embed_tokens.weight"
        if lm_head_key is None:
            lm_head_key = "lm_head.weight"

        # 2. Resolve Model Path
        local_model_path = model_path
        if not os.path.exists(local_model_path):
            try:
                local_model_path = snapshot_download(
                    repo_id=model_path,
                    cache_dir=cache_dir,
                    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model"],
                )
            except Exception as e:
                print(f"Warning: Snapshot download failed or path check failed: {e}")

        # 3. Handle Weight Tying
        tie_weights = getattr(config, "tie_word_embeddings", False)

        # 4. Load Weights
        instance._load_weights(local_model_path, embed_key, lm_head_key, tie_weights)

        text_config = target_text_config(config)
        mup_multiplier = getattr(
            text_config,
            "logits_mup_width_multiplier",
            getattr(config, "logits_mup_width_multiplier", None),
        )
        if mup_multiplier:
            if tie_weights:
                raise RuntimeError(
                    "cannot fold logits_mup_width_multiplier into a tied "
                    "embedding/LM head"
                )
            instance.lm_head.weight.data.div_(float(mup_multiplier))
            instance.lm_head_mup_folded = float(mup_multiplier)

        # 5. Move to Device & Freeze
        instance.to(device=device, dtype=dtype)
        instance.eval()
        instance.requires_grad_(False)

        return instance

    def _load_weights(
        self, model_path: str, embed_key: str, lm_head_key: str, tie_weights: bool
    ) -> set[str]:
        index_files = glob.glob(os.path.join(model_path, "*.index.json"))
        weight_map = {}
        files_to_load = {}
        required_keys = [embed_key]
        if not tie_weights:
            required_keys.append(lm_head_key)

        if index_files:
            with open(index_files[0], "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})

            missing_from_index = sorted(set(required_keys) - weight_map.keys())
            if missing_from_index:
                raise ValueError(
                    "Required target weight keys are missing from the checkpoint "
                    f"index: {missing_from_index}"
                )
            for key in required_keys:
                files_to_load[key] = weight_map[key]
        else:
            safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
            bins = glob.glob(os.path.join(model_path, "*.bin"))
            target_file = safetensors[0] if safetensors else (bins[0] if bins else None)

            if not target_file:
                raise FileNotFoundError("No checkpoint found.")

            filename = os.path.basename(target_file)
            files_to_load.update({key: filename for key in required_keys})

        loaded_keys = set()

        file_to_keys_map = {}
        for key, filename in files_to_load.items():
            full_path = os.path.join(model_path, filename)
            if full_path not in file_to_keys_map:
                file_to_keys_map[full_path] = []
            file_to_keys_map[full_path].append(key)

        for file_path, keys in file_to_keys_map.items():
            loaded_keys.update(
                self._load_file_content(file_path, keys, embed_key, lm_head_key)
            )

        missing_keys = sorted(set(required_keys) - loaded_keys)
        if missing_keys:
            raise RuntimeError(
                "Required target weight tensors were not loaded from the "
                f"checkpoint: {missing_keys}"
            )

        if tie_weights:
            print(
                "Weight tying detected: Sharing weights between Embeddings and LM Head."
            )
            self.lm_head.weight = self.embed_tokens.weight

        return loaded_keys

    def _load_file_content(
        self,
        file_path: str,
        keys_to_extract: list,
        target_embed_key: str,
        target_head_key: str,
    ) -> set[str]:
        """Helper to load specific keys from a file"""
        print(f"Loading {keys_to_extract} from {os.path.basename(file_path)}...")

        state_dict_part = {}

        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                for k in keys_to_extract:
                    if k in f.keys():
                        state_dict_part[k] = f.get_tensor(k)
        else:
            print(
                f"Warning: Loading .bin file {os.path.basename(file_path)} into RAM. Convert to safetensors for efficiency."
            )
            full_state = torch.load(file_path, map_location="cpu")
            for k in keys_to_extract:
                if k in full_state:
                    state_dict_part[k] = full_state[k]
            del full_state
            gc.collect()

        loaded_keys = set()
        for k, tensor in state_dict_part.items():
            if k == target_embed_key:
                if tensor.shape != self.embed_tokens.weight.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {k}. Expected "
                        f"{self.embed_tokens.weight.shape}, got {tensor.shape}"
                    )
                with torch.no_grad():
                    self.embed_tokens.weight.copy_(tensor)
                print(" -> Loaded Embeddings")
            elif k == target_head_key:
                if tensor.shape != self.lm_head.weight.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {k}. Expected {self.lm_head.weight.shape}, got {tensor.shape}"
                    )
                with torch.no_grad():
                    self.lm_head.weight.copy_(tensor)
                print(" -> Loaded LM Head")
            else:
                continue
            loaded_keys.add(k)

        return loaded_keys


__all__ = [
    "TargetEmbeddingsAndHead",
    "load_target_config",
    "target_embedding_scale",
    "target_logit_softcap",
    "target_text_config",
    "target_vocab_size",
]
