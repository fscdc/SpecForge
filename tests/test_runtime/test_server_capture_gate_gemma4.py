# coding=utf-8
"""Server-capture gate for a Gemma 4 target: a LIVE patched SGLang server
captures a tiny random ``Gemma4ForConditionalGeneration`` into a LIVE Mooncake
store, and every feature is checked against an independent HF forward.

What is pinned here (no fake index file; the checkpoint is a single
``model.safetensors`` like the real one):

- capture semantics: the aux taps (1, 2 and 4; layer 4 sits in the KV-shared
  layers) equal HF ``hidden_states[k+1]`` position by position and tap by tap,
  to a relative L2 error of 0.05 and a cosine of 0.999. The x40 embedding
  below puts the residual stream at RMS ~7.5, where bf16 rounding alone breaks
  an elementwise tolerance; two correct bf16 implementations stay under 0.02.
  Stock SGLang hands back each tap already overwritten by the next layer's
  fused add-RMSNorm, 0.11-0.16 off; the gemma4-spec patch copies the tap;
- the frozen ``TargetHead`` (single-file load, Gemma 4 embedding key)
  reproduces HF's softcapped logits from the captured last hidden state. The
  tied embedding is scaled x40 so raw logits pass the softcap of 30, which the
  test asserts, otherwise it could not tell a capped head from a raw one;
- the DFlash-family noise embedding (``TargetEmbeddingsAndHead.embed_tokens``)
  is bitwise SGLang's Gemma 4 embedding module: a bf16 lookup times the Python
  float ``sqrt(hidden_size)``. hidden_size is 80, whose square root is not a
  bf16 value, so HF's bf16-rounded scale gives different bits (also asserted);
- the multimodal request path: an image row sent as processor-expanded
  ``input_ids`` plus ``image_data`` matches HF exactly before the image, and
  after it under one of the image attention masks SGLang may apply (causal,
  per-image bidirectional, or bidirectional within the triton kernel's BLOCK_M
  query tiles); the matching one is printed;
- one EAGLE3 and one MMFlash train step on the captured features give a
  finite loss.

The checkpoint has the real vocabulary (262144) and special-token ids and
carries the real snapshot's tokenizer, processor config and chat template:
SGLang loads the processor of a multimodal target even with
``--skip-tokenizer-init``. Those files must be in the local HF cache
(``google/gemma-4-E4B-it``); the weights are not needed. Local runs need a GPU,
sglang with ``patches/sglang/v0.5.14/spec-capture.patch`` and
``gemma4-spec.patch`` (scripts/apply_sglang_spec_capture_patch.sh,
scripts/apply_sglang_gemma4_patch.sh), the ``mooncake`` package and a
``mooncake_master`` binary; opt in with ``SPECFORGE_RUN_SERVER_CAPTURE_TESTS=1``.
"""

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import torch

from tests.utils import terminate_process_trees

CUDA = torch.cuda.is_available()
ENABLED = os.environ.get("SPECFORGE_RUN_SERVER_CAPTURE_TESTS") == "1"
PORT = 30993
MOONCAKE_RPC_PORT, MOONCAKE_HTTP_PORT, MOONCAKE_METRICS_PORT = 50161, 8191, 9191
GEMMA4_REPO = "google/gemma-4-E4B-it"
PROCESSOR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
)
AUX_LAYER_IDS = [1, 2, 4]
EMBEDDING_KEY = "model.language_model.embed_tokens.weight"
MASK_TOKEN_ID = 4  # Gemma 4's own <mask>
H, TOL = 80, 2e-2  # fixture hidden size; bf16 logit tolerance
# hidden states, per position and tap: relative L2 error and cosine to HF
MAX_REL_ERR, MIN_COS = 0.05, 0.999
EMBED_GAIN = 40.0  # raw logits beyond the softcap
SOFTCAP = 30.0


def _capture_schema(algorithm: str, modality: str):
    from specforge.algorithms.builtin import builtin_algorithm_registry
    from specforge.inference.adapters.server_capture import ServerCaptureSchema

    registration = builtin_algorithm_registry().resolve(algorithm)
    layout = registration.providers.server_streaming_for(modality).layout
    return ServerCaptureSchema(
        aux_feature=layout.aux_feature,
        last_hidden_feature=layout.last_hidden_feature,
        passthrough=layout.passthrough,
        attention_mask_feature=layout.attention_mask_feature,
    )


def _patched_sglang() -> bool:
    return importlib.util.find_spec("sglang.srt.spec_capture_sink") is not None


def _gemma4_spec_patched() -> bool:
    """The gemma4-spec sentinels, read without importing sglang."""
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.origin:
        return False
    root = os.path.dirname(os.path.dirname(spec.origin))
    for relative, sentinel in (
        (
            "sglang/srt/models/gemma4_causal.py",
            "aux_hidden_states.append(hidden_states.clone())",
        ),
        ("sglang/srt/models/gemma4_mm.py", "and vision_bidirectional"),
        ("sglang/srt/multimodal/processors/gemma4.py", "gpu_image_decode = False"),
    ):
        try:
            with open(os.path.join(root, relative), encoding="utf-8") as handle:
                if sentinel not in handle.read():
                    return False
        except OSError:
            return False
    return True


def _mooncake_available() -> bool:
    return importlib.util.find_spec("mooncake.store") is not None


def _cached_snapshot():
    """The local snapshot holding the Gemma 4 tokenizer/processor files, or None."""
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            GEMMA4_REPO,
            allow_patterns=["config.json", *PROCESSOR_FILES],
            local_files_only=True,
        )
    except Exception:
        return None
    if all(os.path.exists(os.path.join(path, name)) for name in PROCESSOR_FILES):
        return path
    return None


def _tcp_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _image_masks(length, image_runs, variant, window, block_m, dtype, device):
    """Additive HF masks per layer type for the image-row comparison.

    ``bidir``: causal plus each image run attending to itself both ways, on
    full and sliding layers (what SGLang's Gemma 4 asks the triton backend
    for). ``tiled``: that mask as the triton extend kernel applies it, whose
    loop over key tiles stops at the end of the query's BLOCK_M tile.
    """
    positions = torch.arange(length, device=device)
    query, key = positions[:, None], positions[None, :]
    causal = key <= query
    group = torch.full((length,), -1, dtype=torch.long, device=device)
    for index, (start, end) in enumerate(image_runs):
        group[start : end + 1] = index
    same = (group[:, None] == group[None, :]) & (group[:, None] >= 0)
    masks = {}
    for layer_type, tile in (
        ("full_attention", block_m["full"]),
        ("sliding_attention", block_m["sliding"]),
    ):
        if variant == "bidir":
            allowed = causal | same
        else:
            allowed = causal | (same & (key < (query // tile + 1) * tile))
        if layer_type == "sliding_attention":
            allowed = allowed & ((query - key) < window)
        mask = torch.zeros((length, length), dtype=dtype, device=device)
        masks[layer_type] = mask.masked_fill(~allowed, torch.finfo(dtype).min)[
            None, None
        ]
    return masks


@unittest.skipUnless(
    CUDA and ENABLED,
    "Gemma 4 server-capture gate: set SPECFORGE_RUN_SERVER_CAPTURE_TESTS=1 (GPU)",
)
class TestGemma4ServerCaptureGate(unittest.TestCase):
    server = None
    master = None
    workdir = None
    target_dir = None

    @classmethod
    def setUpClass(cls):
        if not _patched_sglang():
            raise unittest.SkipTest(
                "installed sglang lacks spec_capture_sink — apply "
                "patches/sglang/v0.5.14/spec-capture.patch "
                "(scripts/apply_sglang_spec_capture_patch.sh)"
            )
        if not _gemma4_spec_patched():
            raise unittest.SkipTest(
                "installed sglang lacks the gemma4-spec patch — apply "
                "patches/sglang/v0.5.14/gemma4-spec.patch "
                "(scripts/apply_sglang_gemma4_patch.sh)"
            )
        if not _mooncake_available():
            raise unittest.SkipTest("mooncake package not installed")
        binary = shutil.which("mooncake_master")
        if binary is None:
            raise unittest.SkipTest("no mooncake_master binary on PATH")
        snapshot = _cached_snapshot()
        if snapshot is None:
            raise unittest.SkipTest(
                f"{GEMMA4_REPO} tokenizer/processor files are not in the local HF cache"
            )

        cls.workdir = tempfile.mkdtemp(prefix="gemma4_capture_gate_")
        cls.addClassCleanup(shutil.rmtree, cls.workdir, True)
        # Class cleanups still run when setUpClass raises. Register process
        # cleanup last so it runs before the temporary log directory is removed.
        cls.addClassCleanup(cls._cleanup_processes)
        cls.target_dir = os.path.join(cls.workdir, "target")
        cls._write_tiny_target(snapshot, cls.target_dir)
        cls._build_rows()

        for port in (PORT, MOONCAKE_RPC_PORT, MOONCAKE_HTTP_PORT, MOONCAKE_METRICS_PORT):
            if _tcp_open(port):
                raise unittest.SkipTest(f"port {port} is already in use")
        cls.mooncake_env = {
            "MOONCAKE_MASTER_SERVER_ADDR": f"127.0.0.1:{MOONCAKE_RPC_PORT}",
            "MOONCAKE_METADATA_SERVER": f"http://127.0.0.1:{MOONCAKE_HTTP_PORT}/metadata",
            "MOONCAKE_LOCAL_HOSTNAME": "127.0.0.1",
            "MOONCAKE_PROTOCOL": "tcp",
        }
        cls.master = subprocess.Popen(
            [
                binary,
                "--enable_http_metadata_server=true",
                "--http_metadata_server_host=127.0.0.1",
                f"--rpc_port={MOONCAKE_RPC_PORT}",
                f"--http_metadata_server_port={MOONCAKE_HTTP_PORT}",
                f"--metrics_port={MOONCAKE_METRICS_PORT}",
            ],
            stdout=open(os.path.join(cls.workdir, "mooncake_master.log"), "w"),
            stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            start_new_session=True,
        )
        deadline = time.time() + 60
        while not _tcp_open(MOONCAKE_RPC_PORT):
            if cls.master.poll() is not None or time.time() > deadline:
                raise unittest.SkipTest(
                    f"mooncake_master did not start; see {cls.workdir}/mooncake_master.log"
                )
            time.sleep(1)

        env = dict(
            os.environ,
            FLASHINFER_DISABLE_VERSION_CHECK="1",
            MOONCAKE_GLOBAL_SEGMENT_SIZE=str(1 << 30),
            **cls.mooncake_env,
        )
        cls.server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                cls.target_dir,
                "--dtype",
                "bfloat16",
                "--skip-tokenizer-init",
                "--attention-backend",
                "triton",
                "--mm-attention-backend",
                "sdpa",
                "--mem-fraction-static",
                "0.3",
                "--chunked-prefill-size",
                "-1",
                "--context-length",
                "1024",
                "--enable-spec-capture",
                "--spec-capture-method",
                "dflash",
                "--spec-capture-aux-layer-ids",
                *[str(i) for i in AUX_LAYER_IDS],
                "--port",
                str(PORT),
            ],
            stdout=open(os.path.join(cls.workdir, "server.log"), "w"),
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        import requests

        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                if requests.get(f"http://localhost:{PORT}/health", timeout=5).ok:
                    return
            except requests.RequestException:
                pass
            if cls.server.poll() is not None:
                break
            time.sleep(5)
        raise RuntimeError(
            f"sglang server did not become healthy; see {cls.workdir}/server.log"
        )

    @classmethod
    def _write_tiny_target(cls, snapshot, target_dir):
        """A random Gemma 4 with the real vocabulary, special ids and processor."""
        from transformers import Gemma4Config, Gemma4ForConditionalGeneration
        from transformers.models.gemma4.configuration_gemma4 import (
            Gemma4TextConfig,
            Gemma4VisionConfig,
        )

        with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as handle:
            real = json.load(handle)
        text = Gemma4TextConfig(
            vocab_size=262144,
            vocab_size_per_layer_input=262144,
            hidden_size=H,
            hidden_size_per_layer_input=16,
            intermediate_size=128,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            global_head_dim=64,
            sliding_window=32,
            final_logit_softcapping=SOFTCAP,
            # each KV-shared layer (4, 5) needs an earlier layer of its type
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            num_kv_shared_layers=2,
            tie_word_embeddings=True,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=1,
            max_position_embeddings=4096,
            hidden_activation="gelu_pytorch_tanh",
            rms_norm_eps=1e-6,
            rope_parameters={
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1e6,
                    "rope_type": "proportional",
                },
                "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
            },
        )
        vision = Gemma4VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=16,
            patch_size=16,
            pooling_kernel_size=3,
            position_embedding_size=10240,
            default_output_length=280,
        )
        special = {
            name: real[name]
            for name in (
                "image_token_id",
                "boi_token_id",
                "eoi_token_id",
                "video_token_id",
                "audio_token_id",
                "boa_token_id",
                "eoa_token_id",
            )
        }
        config = Gemma4Config(
            text_config=text.to_dict(),
            vision_config=vision.to_dict(),
            audio_config=None,
            tie_word_embeddings=True,
            **special,
        )
        torch.manual_seed(1234)
        model = Gemma4ForConditionalGeneration(config)
        with torch.no_grad():
            model.model.language_model.embed_tokens.weight.mul_(EMBED_GAIN)
        model.to(torch.bfloat16).save_pretrained(target_dir)
        for name in PROCESSOR_FILES:
            shutil.copy(os.path.join(snapshot, name), os.path.join(target_dir, name))
        cls.image_token_id = int(real["image_token_id"])

    @classmethod
    def _build_rows(cls):
        """Text rows and one image row, encoded by the trainer's own path."""
        import numpy as np
        from PIL import Image
        from transformers import AutoProcessor

        from specforge.data.mm_preprocessing import (
            _assistant_header_ids,
            _end_token_ids,
            encode_mm_record,
        )

        cls.processor = AutoProcessor.from_pretrained(cls.target_dir)
        header, end = _assistant_header_ids(cls.processor), _end_token_ids(cls.processor)

        def encode(record):
            payload = encode_mm_record(
                record,
                cls.processor,
                image_root="",
                max_length=1024,
                train_only_last_turn=False,
                header_ids=header,
                end_ids=end,
            )
            assert payload is not None, record
            return payload

        answer = (
            "The sliding window of this model is thirty two tokens, so this reply "
            "is long enough to cross it more than once before it ends."
        )
        text = {
            "conversations": [
                {"role": "user", "content": "Tell me about the attention window."},
                {"role": "assistant", "content": answer},
            ]
        }
        payload = encode(text)
        # the same prompt twice: the second would hit the first one's radix
        # entry if the adapter did not isolate attempts
        cls.text_rows = [payload, dict(payload)]
        image_path = os.path.join(cls.workdir, "image.png")
        noise = np.random.RandomState(0).randint(0, 256, (64, 96, 3), dtype=np.uint8)
        Image.fromarray(noise).save(image_path)
        cls.image_record = {
            "image": image_path,
            "conversations": [
                {"role": "user", "content": "<image>\nDescribe the picture."},
                {"role": "assistant", "content": answer},
            ],
        }
        cls.image_row = encode(cls.image_record)
        ids = cls.image_row["input_ids"]
        positions = [i for i, token in enumerate(ids) if token == cls.image_token_id]
        assert positions and positions == list(range(positions[0], positions[-1] + 1))
        cls.image_span = (positions[0], positions[-1])

    @classmethod
    def _cleanup_processes(cls):
        terminate_process_trees(cls.server, cls.master, grace_s=30)

    # -- helpers ---------------------------------------------------------------
    def _store(self, store_id):
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

        return MooncakeFeatureStore(
            store_id=store_id,
            setup_kwargs={
                "local_hostname": self.mooncake_env["MOONCAKE_LOCAL_HOSTNAME"],
                "metadata_server": self.mooncake_env["MOONCAKE_METADATA_SERVER"],
                "global_segment_size": 1 << 28,
                "local_buffer_size": 1 << 28,
                "protocol": self.mooncake_env["MOONCAKE_PROTOCOL"],
                "rdma_devices": "",
                "master_server_addr": self.mooncake_env["MOONCAKE_MASTER_SERVER_ADDR"],
            },
        )

    def _tasks(self, payloads, run_id):
        from specforge.runtime.contracts import PromptTask

        return [
            PromptTask(
                task_id=f"t{i}",
                run_id=run_id,
                source_id="gemma4-gate",
                payload={
                    "input_ids": list(p["input_ids"]),
                    "loss_mask": list(p["loss_mask"]),
                    "image": p["image"],
                },
                max_length=len(p["input_ids"]),
            )
            for i, p in enumerate(payloads)
        ]

    def _capture(self, payloads, *, algorithm, modality, run_id, required, image=False):
        """Capture rows one request at a time; returns (store, [(out, handle)])."""
        from specforge.algorithms.common.mm_server_input import ImageServerInputAdapter
        from specforge.inference.adapters.server_capture import (
            SGLangServerCaptureAdapter,
        )
        from specforge.inference.capture import CaptureConfig
        from specforge.runtime.contracts import SampleRef

        store = self._store(f"gemma4-gate-{run_id}")
        adapter = SGLangServerCaptureAdapter(
            f"http://localhost:{PORT}",
            store,
            run_id=run_id,
            algorithm=algorithm,
            schema=_capture_schema(algorithm, modality),
            request_input_adapter=ImageServerInputAdapter(None) if image else None,
        )
        contract = CaptureConfig.from_strategy(
            required_features=required,
            aux_hidden_state_layer_ids=tuple(AUX_LAYER_IDS),
            target_repr="hidden_state",
            target_hidden_size=H,
        )
        fetched = []
        for task in self._tasks(payloads, run_id):
            (ref,) = adapter.produce_refs([task], capture=contract)
            self.assertIsInstance(ref, SampleRef, f"expected a ref, got failure: {ref}")
            fetched.append(store.get(ref))
        return store, fetched

    def _hf_model(self):
        from transformers import Gemma4ForConditionalGeneration

        return (
            Gemma4ForConditionalGeneration.from_pretrained(
                self.target_dir, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .cuda()
            .eval()
        )

    def _hf_reference(self, model, inputs, attention_mask=None):
        """Aux taps (``hidden_states[k+1]`` = output of layer k), last hidden, logits."""
        kwargs = {k: v.cuda() for k, v in inputs.items() if k != "attention_mask"}
        kwargs["attention_mask"] = (
            inputs["attention_mask"].cuda() if attention_mask is None else attention_mask
        )
        with torch.no_grad():
            out = model(**kwargs, output_hidden_states=True, use_cache=False)
        aux = torch.cat([out.hidden_states[i + 1] for i in AUX_LAYER_IDS], dim=-1)
        return aux[0].float().cpu(), out.hidden_states[-1][0].float().cpu(), out.logits[0].float().cpu()

    def _hf_text_reference(self, payloads):
        model = self._hf_model()
        refs = []
        for payload in payloads:
            ids = torch.tensor([payload["input_ids"]])
            refs.append(
                self._hf_reference(
                    model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
                )
            )
        del model
        torch.cuda.empty_cache()
        return refs

    def _assert_hidden_close(self, got, ref, what):
        """Captured vs HF hidden states, one ``H``-wide row per position and tap.

        Scale-aware on purpose: an elementwise atol/rtol of 2e-2 rejects a
        correct capture at this fixture's RMS of ~7.5.
        """
        self.assertEqual(tuple(got.shape), tuple(ref.shape), what)
        got, ref = got.float().reshape(-1, H), ref.float().reshape(-1, H)
        rel = (got - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)
        cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
        detail = (
            f"{what}: max relative L2 error {float(rel.max()):.4f}, "
            f"min cosine {float(cos.min()):.5f}"
        )
        self.assertLessEqual(float(rel.max()), MAX_REL_ERR, detail)
        self.assertGreaterEqual(float(cos.min()), MIN_COS, detail)

    def _assert_logits_close(self, recomputed, reference, last_hidden):
        """TargetHead vs HF logits, on the logit scale, with the softcap engaged."""
        scale = float(reference.abs().max())
        torch.testing.assert_close(recomputed, reference, rtol=TOL, atol=TOL * scale)
        from safetensors import safe_open

        with safe_open(
            os.path.join(self.target_dir, "model.safetensors"), framework="pt"
        ) as handle:
            weight = handle.get_tensor(EMBEDDING_KEY).float()
        raw = last_hidden.float() @ weight.T
        self.assertGreater(float(raw.abs().max()), SOFTCAP, "fixture: the cap never engages")
        self.assertGreater(
            float((raw - reference).abs().max()),
            20 * TOL * scale,
            "fixture: uncapped logits would pass this tolerance",
        )

    # -- tests -------------------------------------------------------------------
    def test_eagle3_capture_matches_hf_and_trains(self):
        store, fetched = self._capture(
            self.text_rows,
            algorithm="eagle3",
            modality="text",
            run_id="eagle3",
            required={"input_ids", "attention_mask", "loss_mask", "hidden_state", "target"},
        )
        refs = self._hf_text_reference(self.text_rows)
        from specforge.modeling.target.target_head import TargetHead

        # no index file next to the single-shard checkpoint, as on the real one
        self.assertFalse(
            any(name.endswith(".index.json") for name in os.listdir(self.target_dir))
        )
        head = TargetHead.from_pretrained(self.target_dir, embedding_key=EMBEDDING_KEY)
        outs = []
        for payload, (out, handle), (aux_ref, last_ref, logits_ref) in zip(
            self.text_rows, fetched, refs
        ):
            length = len(payload["input_ids"])
            self.assertEqual(out["hidden_state"].shape, (1, length, len(AUX_LAYER_IDS) * H))
            self.assertEqual(out["target"].shape, (1, length, H))
            self.assertEqual(out["input_ids"].tolist(), [payload["input_ids"]])
            self._assert_hidden_close(out["hidden_state"][0], aux_ref, "aux taps")
            self._assert_hidden_close(out["target"][0], last_ref, "post-norm last hidden")
            with torch.no_grad():
                recomputed = head(out["target"].cuda()).float().cpu()[0]
            self._assert_logits_close(recomputed, logits_ref, out["target"][0])
            outs.append(out)
            store.release(handle)
        self._eagle3_train_step(outs, head)

    def _eagle3_train_step(self, fetched, head):
        from specforge.algorithms.eagle3.model import OnlineEagle3Model
        from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.contracts import TrainBatch
        from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
        from specforge.training.controller import TrainerCore
        from specforge.training.strategies.base import Eagle3TrainStrategy
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29578")
        torch.manual_seed(0)
        config_path = os.path.join(self.workdir, "eagle3_draft.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    **fx.TINY_DRAFT_CONFIG,
                    "hidden_size": H,
                    "bos_token_id": 2,
                    "eos_token_id": 106,
                    "vocab_size": 262144,
                    "draft_vocab_size": 64,
                    "eagle_config": {
                        "use_aux_hidden_state": True,
                        "eagle_aux_hidden_state_layer_ids": AUX_LAYER_IDS,
                    },
                },
                handle,
            )
        dm = AutoDraftModel.from_config(
            AutoDraftModelConfig.from_file(config_path),
            attention_backend="sdpa",
            torch_dtype=torch.bfloat16,
        ).cuda()
        dm.load_vocab_mapping(
            fx.write_vocab_mapping(
                os.path.join(self.workdir, "vm.pt"), vocab=262144, draft_vocab=64
            )
        )
        dm.freeze_embedding()
        model = OnlineEagle3Model(dm, length=3, attention_backend="sdpa").cuda()
        backend = FSDPTrainingBackend(ParallelConfig.from_distributed())
        backend.prepare_model(model, wrap=False)
        backend.set_optimizer(
            BF16Optimizer(dm, lr=1e-3, max_grad_norm=0.5, warmup_ratio=0.0, total_steps=10)
        )
        core = TrainerCore(Eagle3TrainStrategy(model, target_head=head), backend)
        for i, out in enumerate(fetched):
            batch = TrainBatch(
                sample_ids=[f"e{i}"],
                strategy="eagle3",
                tensors={
                    "input_ids": out["input_ids"].cuda(),
                    "attention_mask": out["attention_mask"].cuda(),
                    "loss_mask": out["loss_mask"].cuda(),
                    "target": out["target"].cuda(),
                    "hidden_state": out["hidden_state"].cuda(),
                },
                metadata={"target_repr": "hidden_state", "ttt_length": 3},
            )
            result = core.train_step(batch)
            self.assertTrue(torch.isfinite(torch.tensor(result.loss)))

    def test_noise_embedding_is_sglang_embedding_module(self):
        import torch.nn.functional as F
        from safetensors import safe_open
        from sglang.srt.models.gemma4_causal import Gemma4TextScaledWordEmbedding

        from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

        with safe_open(
            os.path.join(self.target_dir, "model.safetensors"), framework="pt"
        ) as handle:
            weight = handle.get_tensor(EMBEDDING_KEY)
        served = Gemma4TextScaledWordEmbedding(
            weight.shape[0], H, 0, embed_scale=H**0.5
        )
        with torch.no_grad():
            served.weight.copy_(weight)
        served = served.to(device="cuda", dtype=torch.bfloat16)
        parts = TargetEmbeddingsAndHead.from_pretrained(
            self.target_dir,
            embed_key=EMBEDDING_KEY,
            device="cuda",
            dtype=torch.bfloat16,
        )
        ids = torch.tensor([self.text_rows[0]["input_ids"]], device="cuda")
        with torch.no_grad():
            ours = parts.embed_tokens(ids)
            theirs = served(ids)
            hf_rounded = F.embedding(ids, served.weight) * torch.tensor(
                H**0.5, dtype=torch.bfloat16, device="cuda"
            )
        self.assertTrue(torch.equal(ours, theirs), "noise embedding != SGLang embed module")
        self.assertFalse(
            torch.equal(theirs, hf_rounded),
            "fixture: the float and the bf16-rounded scale give the same bits",
        )
        # the head stays the raw table (SGLang drafts by argmax of raw logits)
        self.assertEqual(
            parts.lm_head.weight.data_ptr(), parts.embed_tokens.weight.data_ptr()
        )

    def test_mmflash_capture_matches_hf_and_trains(self):
        store, fetched = self._capture(
            self.text_rows[:1],
            algorithm="mmflash",
            modality="text",
            run_id="mmflash",
            required={"input_ids", "hidden_states", "loss_mask"},
        )
        (out, handle), = fetched
        self.assertEqual(sorted(out), ["hidden_states", "input_ids", "loss_mask"])
        aux_ref, _, _ = self._hf_text_reference(self.text_rows[:1])[0]
        self._assert_hidden_close(out["hidden_states"][0], aux_ref, "MMFlash aux taps")
        store.release(handle)
        self._mmflash_train_step(out)

    def _mmflash_train_step(self, out):
        from specforge.algorithms.common.mmflash_model import OnlineMMFlashModel
        from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
        from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.contracts import TrainBatch
        from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
        from specforge.training.controller import TrainerCore
        from specforge.training.strategies.base import MMFlashTrainStrategy
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29578")
        torch.manual_seed(0)
        config_path = os.path.join(self.workdir, "mmflash_draft.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "architectures": ["MMFlashDraftModel"],
                    "auto_map": {"AutoModel": "mmflash.MMFlashDraftModel"},
                    "block_size": 4,
                    "bos_token_id": 2,
                    "eos_token_id": 106,
                    "dflash_config": {
                        "mask_token_id": MASK_TOKEN_ID,
                        "target_layer_ids": AUX_LAYER_IDS,
                    },
                    "head_dim": 32,
                    "hidden_act": "silu",
                    "hidden_size": H,
                    "initializer_range": 0.02,
                    "intermediate_size": 128,
                    "layer_types": ["full_attention"],
                    "max_position_embeddings": 1024,
                    "max_window_layers": 1,
                    "model_type": "qwen3",
                    "num_attention_heads": 4,
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 2,
                    "num_target_layers": 6,
                    "pad_token_id": 0,
                    "rms_norm_eps": 1e-6,
                    "rope_scaling": None,
                    "rope_theta": 1000000,
                    "sliding_window": None,
                    "tie_word_embeddings": True,
                    "use_sliding_window": False,
                    "vocab_size": 262144,
                },
                handle,
            )
        draft_config = AutoDraftModelConfig.from_file(config_path)
        draft_config._attn_implementation = "sdpa"
        dm = AutoDraftModel.from_config(draft_config, torch_dtype=torch.bfloat16).cuda()
        parts = TargetEmbeddingsAndHead.from_pretrained(
            self.target_dir, embed_key=EMBEDDING_KEY, device="cuda", dtype=torch.bfloat16
        )
        model = OnlineMMFlashModel(
            draft_model=dm,
            target_lm_head=parts.lm_head,
            target_embed_tokens=parts.embed_tokens,
            mask_token_id=MASK_TOKEN_ID,
            block_size=4,
            attention_backend="sdpa",
            num_anchors=8,
            loss_type="dflash",
        ).to(device="cuda", dtype=torch.bfloat16)
        backend = FSDPTrainingBackend(ParallelConfig.from_distributed())
        backend.prepare_model(model, wrap=False)
        backend.set_optimizer(
            BF16Optimizer(dm, lr=1e-3, max_grad_norm=1.0, warmup_ratio=0.0, total_steps=10)
        )
        core = TrainerCore(MMFlashTrainStrategy(model), backend)
        batch = TrainBatch(
            sample_ids=["m0"],
            strategy="mmflash",
            tensors={
                "input_ids": out["input_ids"].cuda(),
                "hidden_states": out["hidden_states"].cuda(),
                "loss_mask": out["loss_mask"].cuda(),
            },
        )
        result = core.train_step(batch)
        self.assertTrue(torch.isfinite(torch.tensor(result.loss)))

    def test_image_row_capture_matches_hf(self):
        from PIL import Image
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            _get_block_sizes_for_extend_attention,
        )

        from specforge.data.mm_preprocessing import to_chat_messages

        store, fetched = self._capture(
            [self.image_row],
            algorithm="eagle3",
            modality="image",
            run_id="image",
            required={"input_ids", "attention_mask", "loss_mask", "hidden_state", "target"},
            image=True,
        )
        (out, handle), = fetched
        aux = out["hidden_state"][0].float()
        store.release(handle)
        length = len(self.image_row["input_ids"])
        self.assertEqual(aux.shape, (length, len(AUX_LAYER_IDS) * H))

        text = self.processor.apply_chat_template(
            to_chat_messages(self.image_record["conversations"]),
            tokenize=False,
            add_generation_prompt=False,
        )
        with Image.open(self.image_record["image"]) as handle:
            image = handle.convert("RGB")
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        self.assertEqual(inputs["input_ids"][0].tolist(), self.image_row["input_ids"])
        model = self._hf_model()
        text_config = model.config.get_text_config()
        block_m = {
            "sliding": _get_block_sizes_for_extend_attention(
                text_config.head_dim, text_config.head_dim
            )[3],
            "full": _get_block_sizes_for_extend_attention(
                text_config.global_head_dim, text_config.global_head_dim
            )[3],
        }
        refs = {"causal": self._hf_reference(model, inputs)[0]}
        for variant in ("bidir", "tiled"):
            masks = _image_masks(
                length,
                [self.image_span],
                variant,
                int(text_config.sliding_window),
                block_m,
                model.dtype,
                torch.device("cuda"),
            )
            refs[variant] = self._hf_reference(model, inputs, attention_mask=masks)[0]
        del model
        torch.cuda.empty_cache()

        first, last = self.image_span
        # the prompt before the image does not depend on the image mask
        self._assert_hidden_close(aux[:first], refs["causal"][:first], "before the image")
        later = slice(first, length)
        taps = aux[later].view(-1, len(AUX_LAYER_IDS), H)
        cosines = {}
        for variant, ref in refs.items():
            ref_taps = ref[later].view(-1, len(AUX_LAYER_IDS), H)
            per_tap = torch.nn.functional.cosine_similarity(taps, ref_taps, dim=-1).mean(dim=0)
            cosines[variant] = [round(float(value), 5) for value in per_tap]
        best = max(cosines, key=lambda name: min(cosines[name]))
        print(
            f"\n[gemma4 gate] image row (image tokens {first}..{last}, BLOCK_M {block_m}): "
            f"per-tap cosine from the image on, by HF mask: {cosines}; SGLang matches {best!r}",
            file=sys.stderr,
        )
        self.assertGreaterEqual(
            min(cosines[best]),
            0.99,
            f"no HF image mask reproduces the captured image row: {cosines}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
