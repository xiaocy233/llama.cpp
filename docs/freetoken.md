# FreeToken paper execution path

This experimental implementation follows *FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution*, arXiv:2608.16157v1 (17 August 2026). It is a GGUF adaptation inside llama.cpp, not the FreeToken engine or its FTW format. It is disabled by default.

The reference repository was inspected at `FlashML-org/FreeToken@60059b41fbf71e545593d9a66cec697c97d4525b`, particularly `python/freetoken/moe/pool_policy.py`. No implementation code was copied. The paper determines scope: newer per-layer/per-layout cache partitioning, additional predictors, model-specific optimizations and application/UI features from that repository are not imported.

## Paper mapping

| Paper mechanism | Implementation |
| --- | --- |
| Complete host expert pool, GPU non-expert layers | Existing GGUF loader and `--cpu-moe -ngl 999`; the populated host banks are registered for mapped CUDA access. |
| One global complete-expert LRU | `src/llama-moe-cache.cpp`, `ggml/src/ggml-cuda/freetoken.cu`; one cache spans all routed layers. Gate/up/down matrices move as one expert. |
| Bandwidth-adaptive misses | Device routing computes `q = clamp(round(m * B_P / B_H), 1, m)` for `m > 0`, otherwise zero. Hits are protected during victim selection. Available slots can further cap the fill count. Remaining misses execute on the CPU. |
| Device planning and fixed graph replay | Deduplication, hit lookup and multi-victim selection run on the GPU. Work lists and CPU job buffers have stable addresses for each layer/batch shape. No per-token route list is read by the application thread. |
| CPU/GPU overlap | A persistent CPU pool uses native GGML dot/quantization kernels. CUDA host callbacks submit and join CPU work while GPU fills and expert kernels run. The selected expert outputs are merged with the original routing weights. |
| Prefill double buffering | Two complete layer banks use space from the same expert cache. The next layer transfer overlaps current-layer compute. No separate pair of layer-sized buffers is allocated. |
| Semantic recurrent checkpoints | Server checkpoints before/after recognized special tokens, evicts checkpoints by access time, and uses the existing deepest surviving prefix restore path. The expert LRU is separate from the server's prompt/KV state. |
| Elastic memory budget | `llama_freetoken_resize_cache()` drains execution, releases the old cache, rebuilds for the new budget and reserves fresh graphs. It retains the loaded host model and recurrent/KV state. The current rebuild re-registers host pages, so resize latency still includes host registration; it is not claimed to match the paper engine's resize latency. |
| Load final host layout before pinning | GGUF's deployed native quantized rows are used directly. This avoids a second weight format, conversion, and a forward warmup. |

`B_P` measures the deployed mapped-host gather into the actual cache banks. `B_H` measures expert matrix-vector kernels with the configured CPU worker count after host registration, avoiding cold file faults in the normal pinned path. Both are decimal GB/s. Startup calibration executes isolated kernels, not a model forward pass. If `B_P >= B_H`, the formula legitimately chooses all-GPU fills; CPU support does not imply CPU work on every request.

## Build and use

Build with the normal NVIDIA CUDA backend:

```sh
cmake -S . -B build-freetoken -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-freetoken -j --target llama-server llama-cli test-freetoken test-freetoken-model
```

Example for one GPU:

```sh
CUDA_VISIBLE_DEVICES=7 build-freetoken/bin/llama-server \
    -m /path/to/model.gguf --cpu-moe -ngl 999 \
    --freetoken --moe-cache-mib 12288 -fit off \
    -c 1024 -b 128 -ub 128 -t 3 -tb 3 \
    --host 127.0.0.1 --port 8097
```

`--moe-cache-mib` limits the expert-bank allocation, including padding and zero-expert slots. It is **not a total VRAM limit**: non-expert weights, KV/recurrent state, graph storage, CUDA context and job buffers also consume memory. Measure total NVML usage and leave sufficient headroom. Use `-fit off` when a fixed experimental allocation is required.

Omitting `--freetoken-bandwidth` profiles both bandwidths at startup. `--freetoken-bandwidth 6,24` supplies externally measured values instead. It must not be used to report invented measured bandwidth or a tuned performance result without disclosure.

Without `--freetoken`, the existing llama.cpp cache and execution path remain selected. Do not combine this path with expert LoRA, Hadamard transforms, biased/scaled expert matrices or a non-SwiGLU FFN.

## Cache and checkpoint details

Expert geometry must match across routed layers, but native quantization types may differ. Raw cache banks use a padded expert stride and typed layer views, so mixed GGUF types share one global LRU. This padding can reduce the number of resident experts. Each CPU route has a distinct zero-weight GPU expert ID, because native grouped GEMM assumes distinct IDs within a token. One final padding slot also protects native CUDA MMQ overreads.

For this adaptation, batches of up to eight tokens use the hybrid decode path. Larger batches use full-layer double buffering if the expert cache holds two layers. If it does not, or host registration is unavailable, larger batches use llama.cpp's existing on-demand path. The threshold is an implementation choice, not a paper-prescribed constant. A host-registration failure is reported; small batches then use CPU experts instead of silently reading pageable memory from a device kernel.

Semantic anchors include recognized single control tokens for thinking, tool calls/responses and chat turn boundaries. Ordinary text that merely resembles a marker is not treated as an anchor. Checkpoints are limited by the existing `--ctx-checkpoints` budget and use the existing server prefix cache. Prompt checkpoints support recurrent/hybrid models; generation checkpoints currently require non-speculative, text-only completion. Other model families and multimodal input have not been validated by this implementation. Draft contexts use the native speculative path and do not inherit the target FreeToken cache or bandwidth settings. With DFlash2 and a maximum of seven draft tokens, the target verifies up to eight tokens through the hybrid decode path. Speculative rejection restores the native sequence state; the expert cache holds immutable weights and is independent of that state.

The resize API is declared in `src/llama-ext.h`:

```cpp
bool llama_freetoken_resize_cache(llama_context * ctx, size_t bytes);
```

The caller must hold exclusive access to the context at a scheduler safe point. Zero releases the expert cache and uses native llama.cpp placement (CPU expert decode with the default offload threshold); a later nonzero budget rebuilds it. This API does not reload the model and does not automatically decide an operating-system memory budget. A failed resize attempts to recover the previous budget and returns false. Do not call it concurrently with generation or context mutation. No server HTTP resize endpoint is added.

## Validation

`test-freetoken` constructs independent native CPU and GPU FFNs. It checks changing routes, shared hits, eviction, mixed quantization, repeated graph execution, prefill fallback and double buffering. It compares both complete output and the independent CPU/GPU partial-output sum. Native CPU and CUDA quantized arithmetic is not bitwise identical; results report the native baseline error separately from the hybrid merge error. The model-shaped case includes 2560/640 dimensions with Q4_K, Q5_K, Q5_1 and Q8_0.

`test-freetoken-model` accepts the usual model arguments and `--freetoken`. It loads one GGUF model, saves a prompt state, then shrinks, releases and regrows the cache. After each change it restores the same state and compares next-token logits. For zero cache it also constructs an independent native context from the same loaded model and compares against that context, since native CPU placement can differ numerically from CUDA expert execution. The difference from the initial CUDA path is printed separately. Model validation is an explicit executable rather than an automatically downloaded CTest fixture.

These checks establish implementation behavior, not dataset accuracy, FreeToken's published speedups, or a comparison with QwenEdge. Performance and answer quality need their own controlled experiments.

## DFlash2 integration

Add `--spec-type draft-dflash --spec-draft-model /path/to/draft-BF16.gguf --spec-draft-n-max 7 --spec-draft-n-min 0 --spec-draft-p-min 0 --spec-draft-ngl all` to the target launch. Rebudget `--moe-cache-mib` after adding the draft: draft weights, draft KV and target rollback state also count toward total VRAM. The draft is BF16 only when the supplied GGUF has that precision.

`test-arg-parser` checks that the draft does not inherit FreeToken settings. `test-freetoken` includes eight-token batches with changing routes and repeated graph execution. Model-level validation must also compare greedy output against a separate non-speculative control; this server version does not support per-request speculative parameter overrides.
