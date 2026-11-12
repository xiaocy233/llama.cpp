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

The autotuner exports separate predicted, screened and confirmed rankings.
Only a configuration that passes native validation is selected. Model predictions
are not measured throughput. The Python entry point is `scripts/moe_adaptive.py`;
`--check-gpu-idle` optionally checks for other GPU workloads before starting.

Local prompts, generated logs, profiler captures and benchmark outputs are ignored
by Git. Numerical regression coverage is maintained in `tests/test-backend-ops.cpp`,
with sampler and autotune checks in their respective test executables.
