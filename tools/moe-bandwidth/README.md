# Offline MoE CPU latency and H2D profile

Build `llama-moe-bandwidth` and `llama-server` with CUDA enabled. Calibrate on the inference host with the same CPU/NUMA placement, model quantization, kernels and expert pinning options. Use one idle GPU:

```sh
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-moe-bandwidth latency.txt \
    -m MODEL.gguf -t 8 --n-cache-layers 4 --n-cache-slots 64 --n-cache-pin -1
```

The tool fixes the CPU pool at 8 participating threads and measures 1, 5 and 10 distinct experts, each activated by the same single token. Other thread counts are rejected. It loads weights but generates no tokens. For each distinct expert layout, it warms up 8 jobs and collects 3 groups of 300 samples per expert count. The median of the three group medians is the fitted point.

Each sample measures CPU submission through result-ready publication, including worker dispatch, input preparation, expert compute and internal synchronization. Graph allocation and weight-cache flushing are outside the timer. Before every sample, the selected weights are flushed from CPU caches. A separate thread continuously transfers other experts to GPU, so H2D does not warm the CPU test weights. Calibration requires x86 SSE2 and more than 10 experts. The first 10 experts supply CPU work; the remaining experts supply H2D work. This measures a cold-cache cost, not the benefit of repeatedly using a warm expert.

The three measured points fit `T_CPU(x) = a*x + b` by least squares, in microseconds. `T_CPU(0) = 0`. A negative fitted intercept is constrained to zero and the slope is refitted through the origin. Nonpositive slopes or invalid measurements fail calibration. The log reports the points, coefficients and maximum fit error. The profile stores both coefficients and all three measured points. H2D remains an effective decimal GB/s rate, measured concurrently over these runs; it includes submission, stream synchronization and pageable-to-pinned staging when needed.

Run inference with the saved profile:

```sh
export GGML_MOE_HYBRID=1
export GGML_MOE_HYBRID_POLICY=balanced
export GGML_MOE_HYBRID_THREADS=8
export GGML_MOE_HYBRID_BW_PROFILE=/absolute/path/latency.txt
./build/bin/llama-server <existing model arguments>
```

Inference only reads the profile. The reader checks version, GPU PCI bus ID/description, CPU thread count, model top-k, expert types/size/input width and pinned status. It also checks that the stored coefficients agree with the measured points. Old v1 bandwidth profiles are rejected; regenerate them with this tool. Recalibrate after changing the host, CPU/NUMA placement, kernels or hardware. Profiles do not encode or detect every such change.

Cache hits stay on GPU. The selector sorts misses by ascending token assignment count, then oldest use, and enumerates all possible CPU prefix lengths. For each split it compares:

```
T_CPU = x > 0 ? a*x + b : 0
T_H2D = g * expert_bytes / (h2d_gbs * 1000)
score = max(T_CPU, T_H2D)
```

Here `g` is the number of unique misses copied to GPU, and `x` is the number of token-expert assignments computed by CPU. Choose the lowest score subject to available GPU slots; equal scores favor fewer GPU copies. All misses may go to CPU. The balanced policy has no CPU assignment cap, and `GGML_MOE_HYBRID_MAX_ASSIGNMENTS` is no longer read. Backend batch and tensor capacity limits still apply. The GPU compute and merge terms are unchanged and are not added to this cost model.

For single-token decode, `x` is the number of CPU experts. Repeated tokens on the same expert count separately, as the current CPU path performs separate matrix-vector work. Applying this single-token fit to repeated-token work or to more than 10 assignments is an approximation; these cases are not separately calibrated.

The internal `GGML_MOE_HYBRID_BW_MEASURE` switch is set only inside the profiler process. Leave it unset for inference. The output is published only after initialization succeeds and at least one entry exists. The versioned text format has one record per layout:

```
ggml-moe-latency-v2
"gpu_pci_id" "gpu_description" threads top_k pinned expert_bytes input_width gate_type up_type down_type cpu_a_us cpu_b_us h2d_gbs cpu_1_us cpu_5_us cpu_10_us
```

The second line describes fields, not a usable measurement. Coefficients must come from a new offline calibration; no historical benchmark values are built into the runtime.

## CPU expert cap for parameter sweeps

Use the legacy policy to bound the number of distinct CPU experts in each layer batch:

```sh
export GGML_MOE_HYBRID=1
export GGML_MOE_HYBRID_POLICY=legacy
export GGML_MOE_HYBRID_MAX_EXPERTS=8
```

The default is `-1`, which leaves the legacy selection unchanged. A value of `0` keeps all expert work on GPU; a positive value caps distinct CPU experts, not token-expert assignments or CPU threads. Candidates must be uncached, below `GGML_MOE_HYBRID_MIN_TOKENS`, and outside `GGML_MOE_HYBRID_REUSE_WINDOW`. The selector prefers fewer token assignments, and prefetching excludes at most the selected CPU candidates. `GGML_MOE_HYBRID_MAX_TOKENS` separately limits the batch size eligible for hybrid execution.

An enabled cap is incompatible with `GGML_MOE_HYBRID_POLICY=balanced`. GPU slots must also hold the experts assigned to GPU: the runtime asserts if slot overflow would force more experts onto CPU than the cap permits. For the experiment's maximum of eight verification positions with top-8 routing, 64 slots per layer avoid this capacity conflict. This is a constraint of the current implementation.

The `MoE pilot` shutdown records report per-layer synchronous and prefetched weight bytes, CPU/GPU token-expert assignments, and `cpu_peak`, the largest number of distinct CPU experts in a batch. These are context-lifetime counters, including prefill and all requests; transfer bytes are not exposed transfer time.

Server response `timings` also includes `decode_rounds`, `draft_verif_steps`, and `decode_output_tokens`. Decode rounds include ordinary decode iterations and speculative verification iterations; the first output token comes from prefill and is excluded. `decode_output_tokens` is `max(predicted_n - 1, 0)`. For a nonzero decode token count, compute TPOT as `predicted_ms / decode_output_tokens`, and average committed tokens per round as `decode_output_tokens / decode_rounds`. This adds measurements without changing the existing generation timer.
