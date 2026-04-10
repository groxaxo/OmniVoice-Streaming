# OmniVoice inference performance audit

This note records the current inference-path bottlenecks that were reviewed with an emphasis on **CUDA Ampere GPUs**, plus the fixes landed in this repository.

## Pass 1 — Fixed bottlenecks

| Area | Bottleneck | Change |
| --- | --- | --- |
| Iterative decoding attention | `omnivoice/models/omnivoice.py` built a dense `(2B, 1, L, L)` boolean attention mask for every inference batch. That is quadratic in sequence length, wastes VRAM, and prevents the inference path from taking advantage of the flex-attention fast path. | Replaced the dense mask with a **BlockMask** built from per-sequence document ids. Padding tokens now keep the old diagonal-only behavior without materializing a dense mask. CUDA inference now defaults to `attn_implementation="flex_attention"` unless the caller overrides it. |
| Logit scoring | The decoder converted the **entire** batched logits tensor to `float32` before slicing out the per-sample target spans. This added avoidable memory traffic and temporary allocations on every decoding step. | Kept batched logits in model dtype and only promote the smaller per-sample slices inside `_predict_tokens_with_scoring()` where the higher-precision log-softmax is actually needed. |
| CUDA runtime selection | The single-item CLI, batch CLI, demo, and server each hard-coded their own device logic, and the GPU-facing entry points always loaded the model in `float16`. That missed a safe Ampere optimization path and made runtime behavior inconsistent across tools. | Centralized runtime helpers in `omnivoice.utils.common`: device detection, auto dtype resolution, and TF32 enablement. Inference now auto-selects **bf16 on Ampere-or-newer CUDA GPUs** when available, otherwise fp16 on CUDA and fp32 elsewhere. |

## Pass 2 — Additional fixed bottlenecks

| Area | Bottleneck | Change |
| --- | --- | --- |
| Decode-loop token update | `_generate_iterative` scattered top-k predicted tokens back via `sample_tokens.flatten()` → `flat_tokens[topk_idx] = …` → `copy_()` → `tokens[i:i+1] = sample_tokens`.  Because `sample_tokens` was a non-contiguous slice of `tokens`, `flatten()` always created a temporary copy, and the final `tokens` assignment was a redundant CUDA write. Three CUDA ops fired per sample per decoding step. | Eliminated the temporary tensor entirely.  `topk_flat` indices are unravelled to `(c_idx, t_idx)` and predicted tokens are written directly into `tokens[i, c_idx, t_idx]`.  The two `batch_input_ids` propagation lines are merged into the same direct index writes, reducing round-trips from 3 to 1. |
| Max + argmax double pass | `_predict_tokens_with_scoring` called both `log_probs.argmax(dim=-1)` and `log_probs.max(dim=-1)[0]` in the greedy path — two sequential reductions over the full `(1, C, T, V)` vocabulary tensor. | Replaced with a single `confidence_scores, pred_tokens = log_probs.max(dim=-1)` call in the greedy path, halving vocabulary-dimension reduction work. |
| Python loop in doc-id builder | `_build_block_mask_document_ids` iterated over `2×B` rows in Python, issuing one `torch.arange` call and one tensor fill per row. For B=8 that was 16 kernel launches with a Python loop at the top. | Replaced with fully vectorised broadcast: a single `torch.arange` + `unsqueeze` + `where` for the whole batch. Zero Python-loop overhead regardless of batch size. |
| Missing cuDNN auto-tuning | `configure_cuda_inference` in `omnivoice/utils/common.py` enabled TF32 matmul but did not set `torch.backends.cudnn.benchmark = True`. The HiggsAudioV2 audio tokenizer contains fixed-size convolutional layers; without `benchmark = True`, cuDNN picks a generic algorithm instead of auto-tuning to the fastest one for this hardware/shape. | Added `torch.backends.cudnn.benchmark = True` to `configure_cuda_inference`. The first encode/decode call with a new input shape incurs a one-time tuning cost; all subsequent calls for the same shape run the optimal cuDNN algorithm. |
| Unbounded voice-prompt cache | `OmniVoiceService._voice_prompt_cache` was a plain `dict` with no eviction policy.  Each `VoiceClonePrompt` entry holds GPU tensors; serving many distinct voices over a long-running server would accumulate tensors indefinitely until OOM. | Replaced with `_VoicePromptLRUCache`, an `OrderedDict`-backed LRU with a configurable `maxsize` (default 128).  Entries are promoted on access; the least-recently-used entry is evicted when the cap is reached. |

## Remaining bottlenecks worth tracking

These are still present after the changes above and should be treated as follow-up work rather than regressions:

1. **Reference prompt preprocessing is still repeated outside the server cache path.** The OpenAI-compatible server caches `voice_clone_prompt` objects, but the batch CLI still rebuilds prompts if the same reference audio is used across multiple jobs.
2. **Batch-duration clustering reloads reference audio on CPU.** `omnivoice/cli/infer_batch.py` loads each reference waveform once for duration estimation and again later during generation. A metadata-only duration path would reduce startup cost for large JSONL jobs.
3. **Chunked long-form generation remains sequential within an utterance.** `omnivoice/models/omnivoice.py::_generate_chunked()` batches across items, but chunk dependencies still serialize work within each long sample.
4. **Audio decode/post-process work is still per-item.** The final tokenizer decode, silence trimming, cross-fade, and fade/pad steps remain outside the main generation loop and are not batched.

## Ampere-specific guidance

For RTX 30-series and other Ampere GPUs, the best current path in this repo is:

- use the default CUDA device selection
- let `OmniVoice.from_pretrained(..., device_map="cuda")` choose the dtype automatically
- keep `nj_per_gpu=1` unless you have enough VRAM to hold multiple full model copies per card

That combination now gives the inference path:

- BF16 weights/activations when the GPU supports them
- TF32 matmul fast paths for remaining FP32 kernels
- flex-attention block masks instead of dense quadratic masks
- single-pass vocabulary max/argmax in the greedy decode loop
- cuDNN auto-tuned convolution algorithms for the audio tokenizer
- LRU-bounded voice prompt cache (no GPU tensor leak under long-running service load)
