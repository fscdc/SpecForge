# SGLang patch inventory and supported version

SpecForge pins `sglang==0.5.14`. The online patch is also kept compatible with
SGLang's public `inkling-support` layout. There are two deliberately separate
SGLang integration surfaces.

## Online: external spec-capture server

Online training uses
[`patches/sglang/v0.5.14/spec-capture.patch`](../../patches/sglang/v0.5.14/spec-capture.patch).
The patch adds `--enable-spec-capture` and a server-side sink that:

1. captures requested auxiliary and final hidden states during prefill;
2. writes tensors directly into Mooncake using
   `MooncakeFeatureStore`'s key layout; and
3. returns only key, shape, and dtype metadata in
   `meta_info["spec_capture"]`.

The client boundary is
[`adapters/server_capture.py`](adapters/server_capture.py). Algorithm-owned
providers map generic server artifacts (`aux`, `last_hidden`, passthrough
inputs) to training feature names. No trainer or producer process imports
SGLang model-runner internals or loads a target model.

The same patch is dry-run validated against the v0.5.14 tag and the public
`inkling-support` branch. Capture requests carry a unique `extra_key`, so every
training sample executes a full prefill even when radix cache support is
present. Capture launch configs therefore leave radix cache enabled, including
for hybrid targets that require the unified radix tree.

Apply the patch with `scripts/apply_sglang_spec_capture_patch.sh`. The
server-capture unit and GPU gates must pass before updating the SGLang pin.

## Offline: dedicated local capture

[`../offline_capture`](../offline_capture) is used exclusively by
`scripts/prepare_hidden_states.py`. Its `sglang_backend` owns the local,
version-pinned APIs required for offline EAGLE3 preprocessing:

| Dependency | Upgrade risk |
|---|---|
| `CaptureHiddenMode.FULL` and logits-processor replacement | hidden-state output fields or pruning behavior may change |
| `set_eagle3_layers_to_capture` / `set_dflash_layers_to_capture` | strategy-specific layer-selection APIs may move |
| `ScheduleBatch`, `ForwardBatch`, and `ModelRunner` construction | constructor and memory-pool setup may change |
| splitting captured states by request input length | token packing conventions may change |
| DP-attention/model-parallel initialization patches | distributed group signatures may change |

This package computes no logits and supports text EAGLE3 and DFlash-family
state capture needed by the preprocessing script. It does not provide
HF/custom backends, VLM capture, online rollout, or a general target-engine
factory.

`tests/test_runtime/test_sglang_0514_compat.py` guards the patched 0.5.14 API
seams, and
`tests/test_offline_capture/test_sglang_backend.py`
provides the GPU smoke coverage for dense and MoE offline capture.

## Gemma 4 targets

A `google/gemma-4-*` target (SGLang's `Gemma4ForConditionalGeneration`) also
needs
[`patches/sglang/v0.5.14/gemma4-spec.patch`](../../patches/sglang/v0.5.14/gemma4-spec.patch).
It makes four changes:

1. `srt/models/gemma4_causal.py`, `Gemma4TextModel.forward`: the layer loop
   captures `hidden_states.clone()`. On CUDA each decoder layer's fused
   add-RMSNorm adds the attention branch to the layer's input in place. A
   captured reference was therefore read back as the next layer's mid-block
   residual, not as the layer output that HF `hidden_states` and SGLang's own
   out-of-place native path (torch.compile) return. The capture of the last
   layer, taken after the loop, was already correct.
2. The same function: when the caller passes `input_embeds` it runs the stack
   on a copy. The multimodal model always passes them, and it keeps the same
   tensor as `forward_batch.mm_input_embeds`, which the EAGLE3 draft reads back
   as its prompt embeddings. Layer 0 used to overwrite that tensor. Qwen3.5
   serving aliases the same way; this patch does not change that.
3. `srt/models/gemma4_mm.py`, `Gemma4ForConditionalGeneration.forward`: the
   per-image bidirectional attention mask is installed only when the text
   config sets `use_bidirectional_attention` to `"vision"`, as in HF's
   `Gemma4Model.forward`. Stock SGLang installs it for every Gemma 4 target on
   the triton backend, but E2B and E4B (`None`) are causal over the whole
   prompt in HF, so their image tokens attended to later tokens of the same
   image, which they never did in training. The mask also reached later keys
   only inside a triton query tile, and only for an image that lay whole in
   one extend window (else causal), so an image prompt's output depended on
   the radix-cache state: 2 of 10 chartqa and 2 of 10 mmstar answers were
   identical between radix on and off, against 10 of 10 text answers.
4. `srt/multimodal/processors/gemma4.py`: `gpu_image_decode = False`. nvJPEG
   decodes a grayscale JPEG to one channel, and the HF Gemma4 image processor
   converts only PIL images to RGB. The vision projection then failed
   (`2520x256 and 768x768`) and took the server down: 86 crashes during the
   Gemma regeneration. A capture-server crash ends a managed run. The number
   of image tokens depends only on the image size, so prompts are unchanged.

On text the target's own outputs do not change. Image prompts of E2B/E4B now
get HF's causal attention, and JPEGs are decoded by PIL instead of nvJPEG. The
Gemma 4 E4B regeneration (regen_data/gemma4-E4B-it_llava-ov15-1M-prompted_*)
ran before change 3, so its image-row answers were generated under the
bidirectional mask.

**Capture and serving hosts must carry the same patch.** Gemma 4 drafts are
trained on the patched features, the true layer outputs. Served by an unpatched
target, a draft sees different features and silently accepts fewer tokens.
Features captured before the patch cannot be mixed with patched ones.

- Training and capture: `bash scripts/apply_sglang_gemma4_patch.sh`. It
  records the applied copy as `sglang/.gemma4_spec_patch.applied`. `--check`
  exits 0 only when the tree carries this exact patch, and `--reverse`
  restores the stock files.
- Benchmarks and serving: `scripts/benchmark_helper.sh` applies it with the
  other serving patches, as `gemma4-spec`.
- Both scripts read the state from one sentinel per file:
  `aux_hidden_states.append(hidden_states.clone())` in `gemma4_causal.py`,
  `and vision_bidirectional` in `gemma4_mm.py` and
  `gpu_image_decode = False` in `processors/gemma4.py`. A tree carrying the
  earlier two-file revision is "partial" by these sentinels;
  `apply_sglang_gemma4_patch.sh` reverses its recorded copy and applies this
  one, and `benchmark_helper.sh` refuses it and points there. The managed-launch
  preflight (`_managed_preflight` in `specforge/launch_plan.py`) checks the
  same strings and refuses a Gemma 4 target on an unpatched SGLang unless
  `SPECFORGE_ALLOW_UNPATCHED_GEMMA4=1` is set.

Serving a Gemma 4 target also needs
`SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION=1` in the server's environment.
Without it, `Gemma4ForConditionalGeneration.forward` does `positions += 1` in
place on the verify positions. The DFlash worker, which also serves MMFlash
drafts, later uses those positions to apply RoPE to the committed context K in
the draft KV cache, which then sits one position too far. CUDA-graph replay
copies the positions first, so only eager verifies are affected: batches above
the CUDA-graph maximum, or `--disable-cuda-graph`. The variable is harmless for
capture, for EAGLE3 and for the target's outputs.

One thing is deliberately left stock: the unsharded embedding that
`Gemma4ForConditionalGeneration.get_embed_and_head` returns. It matters only
for EAGLE3 serving with TP>1, and every config uses `tp_size` 1.
