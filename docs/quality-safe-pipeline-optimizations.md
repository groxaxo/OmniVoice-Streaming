# Quality-safe pipeline optimizations still available

This note is a follow-up to `docs/inference-performance-audit.md`.

The goal here is narrower: identify optimizations that are still available in
the current inference and serving pipeline **without changing output audio
quality**. In practice that means:

1. no changes to diffusion steps, guidance scale, temperatures, or token update logic
2. no quantization, compile-time graph rewrites, or alternate attention math
3. no changes to silence thresholds, chunking policy, sample rate, or codec bitrate
4. only caching, batching, metadata probing, or I/O-path improvements that keep
   the model inputs and post-processing math equivalent

## Highest-value missed optimizations

| Priority | Area | Current code path | Safe fix | Why audio quality stays unchanged |
| --- | --- | --- | --- | --- |
| P0 | Reuse voice-clone prompts outside the server cache path | `omnivoice/cli/infer_batch.py` still passes `ref_audio` + `ref_text` into `worker_model.generate()` for every batch item, which causes `OmniVoice.generate()` to rebuild `VoiceClonePrompt` objects via `create_voice_clone_prompt()` on every run. `omnivoice/cli/demo.py` also rebuilds prompts on every request. | Add a per-worker/per-process LRU cache keyed by `(ref_audio_path, ref_text, preprocess_prompt, sampling_rate)` and pass `voice_clone_prompt=` directly once cached. Reuse the same cache idea already used in `omnivoice/openai_tts_server.py`. | The reused object already contains the exact `ref_audio_tokens`, `ref_text`, and `ref_rms` the model would compute again. This removes repeated preprocessing, but the model sees the same prompt tensors. |
| P0 | Avoid full audio decode during batch-duration estimation | `estimate_sample_total_duration()` in `omnivoice/cli/infer_batch.py` calls `load_audio()` just to compute reference duration, and `load_audio()` performs full decode/resample work. The same file is loaded again later when the prompt is built for generation. | Replace the duration probe with a metadata-only path: first use `torchaudio.info()`, then fall back to `ffprobe` or `AudioSegment` metadata if needed. Memoize duration by normalized file path for repeated references in the same job. | This changes only how batches are scheduled. The actual generation path still reads the original reference audio and produces the same prompt/audio output. |
| P1 | Cache fixed token prefixes for long-form chunked generation | `_generate_chunked()` repeatedly calls `_generate_iterative()`, and `_prepare_inference_inputs()` re-tokenizes the same style wrapper (`<|denoise|>`, language, instruct) and the same reference-text prefix for every chunk. | Precompute per-item prefix token tensors once for the fixed parts of the request, then tokenize only the chunk-specific text each round. Concatenate the cached prefix with the new chunk tokens before padding. | The token IDs remain identical to the current path. The change is only to reuse previously computed token tensors instead of regenerating them. |
| P1 | Decode chunk batches together and merge in linear time | `_decode_and_post_process()` decodes each chunk separately when given a list, and `cross_fade_chunks()` repeatedly grows the merged tensor with `torch.cat()`, which turns long outputs into repeated copy work. | Decode all chunk token tensors in one `audio_tokenizer.decode()` batch when memory allows, then apply the same fade logic to per-chunk clones and concatenate once (or write into a preallocated output buffer). | The decoder weights, input tokens, fade weights, and silence gap duration all stay the same. The only change is execution order and memory allocation strategy. |
| P1 | Remove the temp-WAV round-trip from server response encoding | `_waveform_to_bytes()` in `omnivoice/openai_tts_server.py` writes a temporary WAV file, then launches `ffmpeg` to read that file back and encode the requested format. | Keep the same codecs and bitrate, but move the transport to memory: write WAV to `BytesIO` for `wav`, and pipe WAV/PCM bytes to `ffmpeg` over stdin/stdout for `mp3`, `flac`, `ogg`, and `opus`. | The synthesized waveform is unchanged. For compressed formats, the encoder settings stay identical; only the filesystem hop is removed. |
| P2 | Move output-file writes off the batch worker critical path | `run_inference_batch()` saves each generated waveform to disk before the worker returns, so GPU-owning worker processes stay occupied by synchronous CPU/disk work after synthesis is finished. | Hand completed tensors to a small thread pool or a dedicated saver process, and let the worker immediately start the next GPU batch. Keep filenames and `torchaudio.save()` parameters unchanged. | This changes only when files are written, not the tensors being written. Output waveforms and file formats remain the same. |

## Exact code locations behind the remaining work

### 1. Prompt reuse is still missing in the CLIs

- `omnivoice/cli/infer_batch.py`
  - `estimate_sample_total_duration()` loads the reference audio only to measure duration
  - `run_inference_batch()` passes raw `ref_audio` and `ref_text` into `worker_model.generate()`
- `omnivoice/cli/demo.py`
  - `_gen_core()` always calls `model.create_voice_clone_prompt(...)` on clone requests
- `omnivoice/models/omnivoice.py`
  - `generate()` rebuilds `voice_clone_prompt` objects whenever `ref_audio` is provided

### 2. Chunk decode/post-process still has avoidable serial work

- `omnivoice/models/omnivoice.py`
  - `_decode_and_post_process()` decodes list inputs one chunk at a time
  - `_generate_chunked()` already batches chunk generation across items, so decode/merge is now the next obvious long-form hot path
- `omnivoice/utils/audio.py`
  - `cross_fade_chunks()` repeatedly concatenates the growing output tensor

### 3. Server packaging still pays unnecessary disk I/O

- `omnivoice/openai_tts_server.py`
  - `_waveform_to_bytes()` always writes `output.wav` to a temporary directory before re-reading it through `ffmpeg`

## Recommended implementation order

1. **Batch CLI prompt cache**
   - biggest repeated CPU/GPU savings for large JSONL jobs
   - simplest change because the server already demonstrates the caching pattern
2. **Metadata-only duration probe**
   - removes duplicated audio decode before generation even starts
3. **Chunk decode + merge rewrite**
   - highest payoff for long-form generation where chunk count grows
4. **Prefix-token caching**
   - especially useful once long-form chunking is common
5. **In-memory server encoding**
   - worthwhile for request latency and SSD churn, but lower priority than prompt/decode work
6. **Async output saves**
   - nice final cleanup for throughput-heavy offline jobs

## Validation rules to keep output quality unchanged

Every implementation should prove one of these invariants:

| Optimization | Required validation |
| --- | --- |
| Prompt cache | Cached `VoiceClonePrompt` fields match a freshly computed prompt for the same `(ref_audio, ref_text, preprocess_prompt)` input. |
| Duration metadata probe | Batch assignment changes are allowed, but generated audio for a given sample must be unchanged when run alone vs in batch. |
| Prefix-token cache | `input_ids` and `audio_mask` produced by the cached path exactly match the current `_prepare_inference_inputs()` output. |
| Chunk decode/merge rewrite | Merged waveform matches the current implementation within floating-point equivalence; fade boundaries and silence gaps must be identical. |
| In-memory encoding | WAV responses must be byte-equivalent or PCM-equivalent; compressed formats must keep the same codec and bitrate settings. |
| Async file saves | Saved waveforms loaded back from disk must match the original tensors exactly within the current `torchaudio.save()` format behavior. |

## Optimizations intentionally excluded from this list

These may be useful later, but they are **not** in this document because they
can change output quality, generation behavior, or numerical results:

- reducing `num_step`
- changing CFG / temperatures / `t_shift`
- changing silence-removal thresholds
- changing chunk-size heuristics
- quantization beyond the current tested path
- speculative `torch.compile()` or graph-capture changes on the autoregressive loop
- swapping encoder/decoder codecs or lowering output bitrate

## Short conclusion

The biggest quality-safe wins still left are not in the core token sampler
anymore. They are in **reusing identical prompt work**, **stopping duplicated
reference-audio reads**, **making long-form decode/post-process linear-time**,
and **removing avoidable disk I/O from the serving path**.
