# DFlash tuning and online observation

Use `scripts/moe_adaptive.py` for both standalone tuning and online observation. **Standalone `tune` collects data and returns recommended parameters in one invocation. Online `serve` collects every 20 output tokens, searches once after each completed response, and applies the result to the next request.** The search backend is invoked by the tool; no manual handoff is required for standalone tuning.

The initial supported execution path is batch-one DFlash with seven draft tokens, greedy target sampling, pinned CUDA expert slots, LFU replacement, and legacy CPU diversion. GPU cache capacity and resident layers are fixed inputs. The experimental collector is enabled only by `LLAMA_MOE_AUTOTUNE=offline|online`.

## Build

On the CUDA machine, build `llama-server` and `llama-moe-autotune`. The numerical search tool itself uses the CPU and requires no model files. The collector reuses the existing routing/copy/CPU paths; it does not change quantization or target routing decisions.

## Standalone tuning: collect, search, and report

```sh
python3 scripts/moe_adaptive.py tune \
    --server build/bin/llama-server \
    --model /path/to/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    --draft-model /path/to/Qwen3.6-35B-A3B-DFlash-bf16.gguf \
    --prompt /path/to/coding-prompt.txt --gpu GPU-UUID \
    --slots 64 --cache-layers 4 --ctx 8192 --tokens 256 \
    --cost-rounds 16 --output /path/to/new-tuning
```

This starts one server, loads the models once, and sends one completion request. After eight initial full verification rounds, the default budget samples verification sizes 1, 2, 4, and 8, first with CPU cap 0 and then cap 4. Each service cell has one warmup round and one measured round: 16 short rounds total, embedded in the same generation. Warmup excludes first-use graph/buffer/thread setup from fitted costs. Every round still produces ordinary output tokens. There is no separate bandwidth or CPU microbenchmark process and no configuration sweep.

After collection, the tool closes its server and automatically searches the saved measurements using the CPU. It writes `search.json`, includes the recommendation in `summary.json` and `session.json`, and reports collection time, search time, and total time. The default backend is `llama-moe-autotune` beside the supplied server executable; `--tool` overrides its path. The executable is checked before loading models. `calibrate` remains an alias for the same complete workflow.

`--cost-rounds 0` disables directed cost samples. A compatible `--profile` also skips them. Missing measurements remain missing; the collector does not fabricate service costs. The profile identity covers model/executable/shared-library file identities, device, CPU affinity and threads, context size, prompt-length bucket, expert shapes, and slots. Use one writer per profile.

The command prints the observed confidence vectors, accepted/retained lengths, copies, CPU assignments, GPU and CPU/GPU service cells, sample counts, wall times, and recommendation. EOS is respected unless `--ignore-eos` is explicitly used for a fixed-length experiment. If the response supplies too few observations for search, tuning fails with a data-coverage error and retains the collected files.

## Collect during requests, tune between requests

The native `llama-server` collector samples every 20 committed output tokens. After a complete response has been queued for delivery, it searches once and retains the chosen configuration for the next request. The current request never changes configuration. Python is optional; it only supplies native environment settings.

```sh
python3 scripts/moe_adaptive.py serve \
    --server build/bin/llama-server \
    --model /models/target.gguf --draft-model /models/dflash.gguf \
    --gpu GPU-UUID --output /tmp/moe-online \
    --profile /tmp/moe-calibration/costs.json \
    --npred 0 --threshold 0 --cpu-cap 0 \
    --sample-tokens 20
```

The first drafted Decode is a bootstrap sample. The collector then samples the first drafted iteration beginning at or beyond 20, 40, 60, ... committed Decode output tokens. It counts `accepted + 1`, excludes prefill and discarded draft tokens, and never splits an iteration to hit a boundary. If one iteration advances the counter from 18 to 24, the next iteration is sampled. Ordinary Decode steps at the output tail advance the counter but have no invented draft record. `--sample-tokens 1` samples every drafted iteration. Standalone `tune` still samples every iteration.

A sampled iteration records confidences, acceptance, layer routes, cache state, copies, and GPU/CPU service times. Other iterations skip these observations. No search or configuration application happens during generation. At request end the controller searches the rolling window, by default the last 32 collected samples. It considers at most 64 combinations of prediction width, truncation threshold, and CPU cap. GPU cache capacity stays fixed. The selected values become active together at the first Decode of the next request. A queued request may wait for the preceding request's search; search time is reported separately from Decode TPOT.

The search requires at least eight usable samples and sufficient measured service costs. Both full and threshold-truncated observations are retained; final batches shortened only by the output budget are excluded from search. A candidate is supported only if its entire requested prefix was verified in every sample in the window. Missing acceptance labels or routes are never invented, and threshold-selected full samples are not used alone. Less aggressive truncation may therefore be unsupported after truncated observations. Without enough coverage, the existing configuration is retained and the reason is logged. Canceled requests do not trigger search.

`npred`, `threshold`, and `cpu-cap` are starting values. `--observe-only`, or `LLAMA_MOE_AUTOTUNE_ADAPT=0`, disables request-end tuning while retaining collection. A compatible standalone profile can supply CPU and short-batch service costs that the current online configuration does not execute. Online mode never inserts calibration probes or forces longer verification. Cost profiles are checked against model/runtime/hardware identity, context, and CPU affinity.

Each new request resets the token counter and workload window. Service costs may be reused. Requests with nonzero temperature are rejected; the server remains available. Reports and profiles are saved every `--window` collected samples and at request completion.

`telemetry.jsonl` contains at most one `recommendation` per completed online request, with `previous_config`, `next_config`, `applies_to=next_request`, and search time. `request` records show the configuration used by each request. `request_end` reports include `output_tokens`, `sampled_rounds`, `sample_tokens`, `search_ms`, and `controller_ms`. `output_tokens_before` and `next_sample_tokens` make the collection schedule auditable. Expert frequencies count sampled steps only. `sampled_round_tpot_ms` describes sampled steps, not end-to-end TPOT; use the server's total Decode time divided by committed Decode output tokens for that.

## Reuse saved observations without collecting again

Standalone `tune` already performs search. To explicitly tune from data previously collected online, use the same tool's `search` entry:

```sh
python3 scripts/moe_adaptive.py search \
    --tool build/bin/llama-moe-autotune \
    --trace /path/to/new-observation/telemetry.jsonl \
    --output /path/to/new-search.json
```

This reuses the same search backend without loading models or controlling an inference server. `--request N` selects one request; the default is the last request. Search needs at least eight untruncated rounds with per-layer records plus a completed cost snapshot. It uses up to the most recent observation window, by default 32 samples. This offline trace reader still requires full-draft references. Request-end online search can also use partial records, but only evaluates prefixes with observed routes and acceptance in every window sample.

Search evaluates `{0,8,16,24} x {0,0.4,0.6,0.8} x {0,2,4,8}` for prediction width, truncation threshold, and CPU cap. It ranks only candidates supported by the recorded data, prints the recommendation and search time, and writes the full ranking to the result JSON. The recommendation is not applied to any process. An explicit inference launch is required to use it.

For candidate prefix K, the estimated committed tokens are `1 + min(observed_full_acceptance, K)`. Each layer simulation uses observed cache owners, LFU scores, prefetch eligibility, predicted votes, and actual IDs to estimate CPU assignments and deduplicated copies. Recent observations receive exponentially decaying weight. Predicted times combine measured GPU/joint service, a verification residual, effective copy service, and observed draft time. Missing interior batch sizes can be interpolated between measured endpoints; unobserved endpoints are not extrapolated. CPU assignment cells without sufficient measurements use measured per-assignment cost and a measured protocol residual, and estimated lookups are reported.

This is a fast one-round counterfactual from the observed cache, not a separate long-term cache trajectory for every candidate. Overlap is approximated from preceding-layer service and the verification residual. Changed round boundaries can change future routes and acceptance. Therefore a recommendation is an estimate to validate, not a guarantee of the optimal configuration.

The runner builds a common width-24 predictor graph. `npred` selects the per-position prefix used by the cache manager; zero disables predicted copies but retains predictor computation. CPU cap zero similarly retains the hybrid graph. These settings are not an all-optimizations-disabled baseline.

## Files and timing units

- `telemetry.jsonl`: observed values, sampled layer snapshots, cost snapshots, and expert frequencies. Online adaptation also writes a request-end recommendation.
- `costs.json`: reusable measured service profile with sample counts, excluding workload acceptance/frequency history.
- `session.json`: standalone startup/load, request, prefill, decode including collection, shutdown, collection total, search, and overall elapsed seconds, plus the recommended parameters. Search wall time includes backend startup and file loading; search compute is also reported separately.
- `manifest.json`, `response.json`, `summary.json`: launch configuration, generated output, and collection summary.
- `search.json` (or the explicit search output path): candidate estimates, recommendation, search time, and file-loading-plus-search time.

`cost_probe_ms` is the execution time of directed calibration rounds, not their incremental slowdown. In online mode it is always zero. `controller_ms` accounts for observation and previous reporting work; current report serialization appears in the next report. Server wall time is the end-to-end authority. Draft/verify observations are per round; obtain ms/token by dividing total time by total committed tokens, not by averaging per-round ratios.

```sh
python3 scripts/moe_adaptive.py report /path/to/new-observation --summary-only
```

Native controls use the `LLAMA_MOE_AUTOTUNE_` prefix: required `OUTPUT`; optional `PROFILE`, `PROFILE_KEY`, `WINDOW`, `SAMPLE_TOKENS`, `ADAPT`, `COST_ROUNDS`, `MAX_PRED`, `DETAIL`, and the initial online configuration `NPRED`, `THRESHOLD`, `CPU_CAP`. The runner forces online `COST_ROUNDS=0`, and the native collector independently disables online probes. Do not combine this initial path with an acceptance predictor, expert substitution, a different CPU/cache policy, or checkpoint replay. Recurrent models require at least seven rollback snapshots.
