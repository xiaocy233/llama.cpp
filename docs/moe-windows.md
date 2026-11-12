# MoE inference on Windows

`build-cuda.bat` builds the server, CLI, benchmark and native autotuning tool.
It requires Visual Studio C++ build tools, CMake, Ninja and a CUDA toolkit.
`CUDA_ARCHS` defaults to `native`; set it explicitly for a different target GPU.
Use `CUDAToolkit_ROOT` to select the toolkit and `JOBS` to limit build parallelism.

```powershell
$env:CUDAToolkit_ROOT = 'D:\tools\cuda-13.0.2'
$env:CUDA_ARCHS = '120'
./build-cuda.bat
./start-server.ps1 --dry-run
./start-server.ps1
```

The launcher defaults to the tested Qwen3.6-35B-A3B-MTP-UD-IQ3_S configuration:

| Setting | Value |
| --- | --- |
| Context / batch / ubatch | 10240 / 2048 / 2048 |
| MTP maximum draft tokens / MTP ubatch | 3 / 128 |
| Prediction width / prefetch copy budget | 8 / 4 |
| CPU expert cap / draft threshold | 0 / 0 |
| CPU affinity | Logical CPUs 0-3 |
| Main / expert pool thread budget | 1 / 3; expert pool disabled at cap 0 |
| Resident expert layers / default cache slots | 0 / 86 |
| Host pin budget | 16000 MiB |
| Sampling / thinking | Greedy / on |

The 40-layer cache allocation in the launcher is specific to this model and the
8 GiB GPU setup. For another model, adjust the cache settings and the per-layer
slot list to its layer count and available memory. Hardware power and fan settings
are managed separately; the launcher does not change them.

Set `MODEL`, `SERVER`, `CPUSET`, `CPU_THREADS`, `EXPERT_THREADS`, `NPRED`,
`PREFETCH_BUDGET`, `CPU_CAP` or `DRAFT_THRESHOLD` to override the corresponding
launcher settings. `MOE_LAYER_SLOTS` overrides the per-layer list. Extra server
arguments can be passed after `--`.

Prediction width controls candidate routing IDs. The prefetch budget limits how
many missing predicted experts are copied; these are independent settings.
For on-demand-only inference, set `NPRED=0`, `PREFETCH_BUDGET=0`, `CPU_CAP=0`
and `DRAFT_THRESHOLD=0`.

## Implementation and fallbacks

- Pinned expert weights use batch CUDA copies when the backend supports them.
  Location maps are published after the associated weight copies.
- Long sparse MoE prefill uses compact expert tile maps on Blackwell CC 12.0.
  Other architectures and short inputs keep the existing MMQ path.
- GDN prefill reuses registers across four state columns for the validated
  non-KDA, 128-wide, long-input path on Blackwell CC 12.0. Other shapes retain
  the original recurrence path. State snapshots remain available for MTP rollback.
- Compact greedy sampling is used only when grammar, penalties, probability
  reporting and other filters do not need the full candidate array.
- CPU expert execution retains a host fallback when CUDA graph execution or
  detailed autotune timing requires it.

Compatibility switches remain available: `GGML_CUDA_MOE_FLAT_GRID=0`,
`GGML_CUDA_GDN_PREFILL_COLUMNS=1`, `GGML_MOE_BATCH_PREFILL_COPY=0` and
`GGML_MOE_CPU_GRAPH=0`. These select existing fallback implementations.

## Autotuning

```powershell
$env:PROMPT_FILE = 'D:\prompts\representative.txt'
./run-autotune.ps1 --dry-run
./run-autotune.ps1
```

The autotuner reads `start-server.ps1 --export-config`; it has no separate inference
defaults. This includes the batch sizes, MTP ubatch, cache slots, decay policy,
pin budget, CPU affinity and optimized CUDA paths. `NPRED` stays fixed. Search
varies the missing-expert prefetch copy budget (0 through 8, bounded by `NPRED`),
draft threshold (0, 0.4, 0.6, 0.8) and CPU expert cap (0, 2, 4, 8). A zero copy
budget still keeps the fixed prediction graph; it is not the same as `NPRED=0`.

Native instrumentation and the hybrid pool remain enabled while screening, so
those timings are not final production throughput. The finalists, the incumbent
and the zero-control configuration are checked with ordinary servers. The best
model proposal with CPU diversion also gets checked even if instrumentation ranked
it poorly. These servers have no tuning
instrumentation, and no hybrid pool at CPU cap zero. Each configuration gets two
measurements in forward/reverse order, each in a fresh server with an excluded
128-token warmup. The final selection minimizes aggregate decode ms/token and
retains the incumbent when it is within 3 percent of the fastest measured candidate.
Decode uses `decode_output_tokens`, excluding the token produced during prefill.
This is a local selection for the supplied prompt, not a global optimum.

`selected-config.json` is exported only after ordinary-server validation finishes.
Apply its `selected.prefetch_budget`, `selected.threshold` and `selected.cpu_cap`
as `PREFETCH_BUDGET`, `DRAFT_THRESHOLD` and `CPU_CAP`; keep its `prediction_width`
as `NPRED`. The full resolved deployment is in `manifest.json`. The new trace
format uses `prefetch_budget`; old prefix-width traces and cost profiles must be
recollected. `production-validation` contains each complete response, timings,
command and server log. Rankings distinguish predictions, native screening,
native confirmation and ordinary production measurements.

The Python entry point is `scripts/moe_adaptive.py`; `--check-gpu-idle` optionally
checks for other GPU workloads. It now uses `--launcher` and production environment
overrides instead of a second set of model/cache/batch command-line options.
Online `serve` still collects instrumented observations; only standalone `tune`
performs the ordinary-server final check.

Local prompts, generated logs, profiler captures and benchmark outputs are ignored
by Git. Numerical regression coverage is maintained in `tests/test-backend-ops.cpp`,
with sampler and autotune checks in their respective test executables.
