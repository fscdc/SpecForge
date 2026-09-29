import glob
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

from specforge.modeling.target.target_utils import target_text_config
from specforge.utils import get_local_device, padding


class TargetHead(nn.Module):
    def __init__(
        self,
        model_path,
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        # A VL checkpoint nests the text stack's sizes under text_config; the
        # head projects the text hidden size onto the text vocabulary.
        text_config = target_text_config(self.config)
        self.hidden_size = text_config.hidden_size
        self.vocab_size = text_config.vocab_size

        self.fc = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        lm_head_key: str = "lm_head.weight",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        embedding_key: Optional[str] = None,
    ) -> "TargetHead":
        target_head = cls(
            model_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        target_head.load_weights(
            model_path=model_path,
            lm_head_key=lm_head_key,
            cache_dir=cache_dir,
            embedding_key=embedding_key,
        )
        target_head.freeze_weights()
        target_head = target_head.eval().to(
            device=get_local_device(), dtype=torch.bfloat16
        )
        return target_head

    def _ties_word_embeddings(self) -> bool:
        for candidate in (self.config, target_text_config(self.config)):
            tied = getattr(candidate, "tie_word_embeddings", None)
            if tied is not None:
                return bool(tied)
        return False

    @torch.no_grad()
    def load_weights(
        self,
        model_path,
        lm_head_key: str = "lm_head.weight",
        cache_dir: Optional[str] = None,
        embedding_key: Optional[str] = None,
    ):
        """Copy the target's output projection into ``fc``.

        A checkpoint that ties word embeddings ships no ``lm_head_key`` at all;
        its head IS the input embedding, so that tensor is loaded instead:
        ``embedding_key`` when the caller knows it (the run config's
        ``model.embedding_key``), otherwise the checkpoint's single
        ``*embed_tokens.weight`` entry.
        """
        if os.path.exists(model_path):
            self.model_path = model_path
        else:
            self.model_path = snapshot_download(repo_id=model_path, cache_dir=cache_dir)

        # model_path is a local directory
        # check if there is file ending with index.json
        glob_path = os.path.join(self.model_path, "*.index.json")
        index_json_path = glob.glob(glob_path)

        if len(index_json_path) == 0:
            raise FileNotFoundError(f"No index.json file found in {self.model_path}")
        if len(index_json_path) > 1:
            raise FileNotFoundError(
                f"Multiple index.json files found in {self.model_path}"
            )
        index_json_path = index_json_path[0]

        with open(index_json_path, "r") as f:
            index_json = json.load(f)
        weight_map = index_json["weight_map"]
        if lm_head_key not in weight_map:
            if not self._ties_word_embeddings():
                raise KeyError(
                    f"{lm_head_key!r} is not in {index_json_path} and the model "
                    "does not tie word embeddings; set model.lm_head_key to the "
                    "checkpoint's output projection"
                )
            candidates = (
                [embedding_key]
                if embedding_key
                else [key for key in weight_map if key.endswith("embed_tokens.weight")]
            )
            if len(candidates) != 1 or candidates[0] not in weight_map:
                raise KeyError(
                    f"{lm_head_key!r} is not in {index_json_path}; the model ties "
                    "word embeddings but the embedding tensor could not be "
                    f"identified (candidates: {candidates!r}) -- set "
                    "model.embedding_key"
                )
            lm_head_key = candidates[0]
        ckpt_file = weight_map[lm_head_key]

        if ckpt_file.endswith(".safetensors"):
            with safe_open(
                os.path.join(self.model_path, ckpt_file), framework="pt"
            ) as f:
                lm_head = f.get_tensor(lm_head_key)
        else:
            state_dict = torch.load(os.path.join(self.model_path, ckpt_file))
            lm_head = state_dict[lm_head_key]
        self.fc.weight.copy_(lm_head)

    def freeze_weights(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def forward(self, hidden_states):
        return self.fc(hidden_states)

    def preprocess(self, input_ids, target, loss_mask):
        # apply pading
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None]
        return input_ids, target, loss_mask
