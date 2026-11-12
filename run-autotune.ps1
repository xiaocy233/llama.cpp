$topN = 5

$ErrorActionPreference = 'Stop'

function Get-Setting($name, $default) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrEmpty($value)) { return $default }
    return $value
}

function Stop-Launcher($message, $code = 1) {
    [Console]::Error.WriteLine($message)
    exit $code
}

if ($args.Count -gt 0 -and $args[0] -cin '-h', '--help') {
    @'
Usage: .\run-autotune.ps1 [--dry-run]
Loads a temporary server, calibrates costs, compares candidates, then stops it.

Set environment variables in PowerShell before starting:
  $env:PROMPT_FILE = 'D:\prompts\chat-1k.txt'
  $env:GPU = '0'
  $env:CPUSET = '0-3'
  .\run-autotune.ps1 --dry-run

CPU_THREADS defaults to 4, including EXPERT_THREADS=3.
CPUSET defaults to 0-3. Set it to none to disable affinity.
PROMPT_FILE sets the input text file (PROMPT is reserved by cmd.exe).
Inference settings are exported from start-server.ps1, including cache layout and MTP settings.
Search varies PREFETCH_BUDGET, DRAFT_THRESHOLD and CPU_CAP; NPRED stays fixed.
The final selection is rechecked in ordinary servers without tuning instrumentation.
TRIAL_BUDGET limits candidate measurements (10..24).
RAW_PROMPT=0 uses the chat template with THINKING=on. THINKING also accepts off.
IGNORE_EOS=1 measures fixed output lengths. Set 0 for natural stopping.
Model paths and token budgets can be overridden through environment variables.
OUTPUT_DIR must be a new directory. Its default is autotune-results/TIMESTAMP-PID.
selected-config.json is exported only after native candidate validation succeeds.
Edit $topN to show the top N configurations per ranking (0 shows all).
ranked-configs.json and ranked-configs.csv always contain all scored candidates, grouped by stage.
'@
    exit 0
}
if ($args.Count -gt 1 -or ($args.Count -eq 1 -and $args[0] -cne '--dry-run')) {
    Stop-Launcher 'Invalid arguments. Use --help.' 2
}
$dryRun = $args.Count -eq 1

$python = 'python'
$promptFile = "$PSScriptRoot\prompts\chat-4k.txt"
$gpu = 0
$cpuSet = '0-3'
$cpuThreads = 4
$expertThreads = 3
$rawPrompt = 0
$thinking = 'on'
$ignoreEos = 0
$outputDir = "$PSScriptRoot\autotune-results\$(Get-Date -Format 'yyyyMMdd-HHmmss')-$PID"

$python = Get-Setting PYTHON $python
$promptFile = Get-Setting PROMPT_FILE $promptFile
$gpu = Get-Setting GPU $gpu
$cpuSet = Get-Setting CPUSET $cpuSet
$cpuThreads = Get-Setting CPU_THREADS $cpuThreads
$expertThreads = Get-Setting EXPERT_THREADS $expertThreads
$rawPrompt = Get-Setting RAW_PROMPT $rawPrompt
$thinking = Get-Setting THINKING $thinking
$ignoreEos = Get-Setting IGNORE_EOS $ignoreEos
$outputDir = Get-Setting OUTPUT_DIR $outputDir

$productionText = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\start-server.ps1" --export-config
if ($LASTEXITCODE -ne 0) { Stop-Launcher 'Cannot export production configuration.' }
$production = ($productionText -join "`n") | ConvertFrom-Json
$server = $production.command[0]
$model = $production.command[[Array]::IndexOf($production.command, '-m') + 1]
$cpuThreads = [int]$production.main_threads + [int]$production.expert_threads
$expertThreads = [int]$production.expert_threads
$cpuSet = $production.cpu_set
$gpu = $production.environment.CUDA_VISIBLE_DEVICES
$thinking = $production.command[[Array]::IndexOf($production.command, '--reasoning') + 1]

$cpuBudget = 0
$expertBudget = 0

if ($cpuThreads -notmatch '^[1-9][0-9]*$' -or ![int]::TryParse("$cpuThreads", [ref]$cpuBudget)) {
    Stop-Launcher 'CPU_THREADS must be a positive integer.' 2
}
if ($expertThreads -notmatch '^[1-9][0-9]*$' -or ![int]::TryParse("$expertThreads", [ref]$expertBudget)) {
    Stop-Launcher 'EXPERT_THREADS must be a positive integer.' 2
}
if ($expertBudget -ge $cpuBudget -or $expertBudget -gt 512) {
    Stop-Launcher 'EXPERT_THREADS must be smaller than CPU_THREADS and at most 512.' 2
}
if ($rawPrompt -notin '0', '1' -or $ignoreEos -notin '0', '1') { Stop-Launcher 'RAW_PROMPT and IGNORE_EOS must be 0 or 1.' 2 }
if ($thinking -cnotin 'on', 'off') { Stop-Launcher 'THINKING must be on or off.' 2 }

$lock = $null
$process = [Diagnostics.Process]::GetCurrentProcess()
$originalAffinity = $process.ProcessorAffinity
$exitCode = 1
Push-Location -LiteralPath $PSScriptRoot
try {
    if (!(Get-Command $python -CommandType Application -ErrorAction SilentlyContinue)) { throw "Python not found: $python" }
    if (!(Test-Path -LiteralPath $server -PathType Leaf)) { throw "Missing executable: $server. Run build-cuda.bat first." }
    if (!(Test-Path -LiteralPath $model -PathType Leaf)) { throw "Missing model: $model" }
    if (!(Test-Path -LiteralPath $promptFile -PathType Leaf)) { throw "Missing prompt: $promptFile. Set PROMPT_FILE to your prompt text file." }
    if (Test-Path -LiteralPath $outputDir) { throw 'OUTPUT_DIR already exists; choose a new directory.' }

    $mainThreads = $cpuBudget - $expertBudget
    $tuneArgs = @(
        '-u', "$PSScriptRoot\scripts\moe_adaptive.py", 'tune'
        '--launcher', "$PSScriptRoot\start-server.ps1"
        '--prompt', $promptFile, '--thinking', $thinking, '--gpu', $gpu, '--output', $outputDir
        '--tokens', (Get-Setting TOKENS 512), '--confirm-tokens', (Get-Setting CONFIRM_TOKENS 1024)
        '--warmup-tokens', (Get-Setting WARMUP_TOKENS 0), '--trial-budget', (Get-Setting TRIAL_BUDGET 24)
        '--cost-rounds', (Get-Setting COST_ROUNDS 48), '--window', (Get-Setting WINDOW 32)
        '--timeout', (Get-Setting TIMEOUT 1800)
        '--top', $topN
    )
    if ($rawPrompt -eq '1') { $tuneArgs += '--raw-prompt' }
    if ($ignoreEos -eq '1') { $tuneArgs += '--ignore-eos' }
    "GPU=$gpu CPUSET=$cpuSet main_threads=$mainThreads expert_threads=$expertBudget"
    "RAW_PROMPT=$rawPrompt THINKING=$thinking IGNORE_EOS=$ignoreEos"
    "Output: $outputDir"
    $display = (@($python) + $tuneArgs) | ForEach-Object { if ("$_" -match '\s') { '"{0}"' -f $_ } else { $_ } }
    "Command: $($display -join ' ')"
    if ($dryRun) { exit 0 }

    if ($cpuSet -ne 'none') {
        $available = $originalAffinity.ToInt64()
        $mask = 0L
        $count = 0
        foreach ($part in ($cpuSet -split ',')) {
            if ($part -notmatch '^(\d+)(?:-(\d+))?$') { throw "Invalid CPUSET: $cpuSet" }
            $first = [int]$Matches[1]
            $last = if ($Matches[2]) { [int]$Matches[2] } else { $first }
            if ($last -lt $first -or $last -ge [IntPtr]::Size * 8) { throw "Invalid CPU range: $part" }
            for ($cpu = $first; $cpu -le $last; $cpu++) {
                $bit = 1L -shl $cpu
                if (!($available -band $bit)) { throw "CPU $cpu unavailable. Set CPUSET to an available range or none." }
                if (!($mask -band $bit)) { $count++ }
                $mask = $mask -bor $bit
            }
        }
        if ($count -lt $cpuBudget) { throw 'CPUSET contains fewer CPUs than CPU_THREADS.' }
        $process.ProcessorAffinity = [IntPtr]$mask
    }

    try {
        $lock = [IO.File]::Open("$PSScriptRoot\.autotune.lock", [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    } catch {
        throw "Cannot acquire .autotune.lock; another run may hold it. $($_.Exception.Message)"
    }
    $env:SERVER = "$server"
    $env:MODEL = "$model"
    $env:GPU = "$gpu"
    $env:CPU_THREADS = "$cpuThreads"
    $env:EXPERT_THREADS = "$expertThreads"
    $env:THINKING = "$thinking"
    $env:OMP_NUM_THREADS = "$mainThreads"
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:PYTHONUTF8 = '1'
    & $python @tuneArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $sessionPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath((Join-Path $outputDir 'session.json'))
    $session = [IO.File]::ReadAllText($sessionPath) | ConvertFrom-Json
    if ($session.status -ne 'complete' -or $session.selection_status -ne 'validated' -or !$session.recommended) {
        throw 'Tuning did not finish candidate validation. Inspect session.json and server.log; no selected config was exported.'
    }
    $result = [ordered]@{
        status = 'validated'
        selected = $session.recommended
        prediction_width = $session.prediction_width
        production_validation = $session.production_validation
        manifest = [IO.Path]::Combine([IO.Path]::GetDirectoryName($sessionPath), 'manifest.json')
        session = $sessionPath
        scope = 'Best retained candidate under this prompt and fixed deployment; not a global optimum.'
    }
    $destination = Join-Path ([IO.Path]::GetDirectoryName($sessionPath)) 'selected-config.json'
    [IO.File]::WriteAllText($destination, ($result | ConvertTo-Json -Depth 20) + [Environment]::NewLine)
    "Validated selection: $($session.recommended | ConvertTo-Json -Compress)"
    "Saved: $destination"
    $exitCode = 0
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
} finally {
    if ($null -ne $lock) { $lock.Dispose() }
    $process.ProcessorAffinity = $originalAffinity
    Pop-Location
}
exit $exitCode
