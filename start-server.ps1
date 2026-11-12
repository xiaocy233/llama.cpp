$nclayers = 0
$ncslots = 86
$specType = 'draft-mtp'
$draft_max = 3
$mtp_ubatch = 128
$npred = 8
$prefetch = 4
$threshold = 0
$cpu_cap = 0
$server = "$PSScriptRoot\build-cuda\bin\llama-server.exe"
$model = "D:\models\Qwen3.6-35B-A3B-MTP-UD-IQ3_S.gguf"
$draftModel = "D:\models\Qwen3.6-35B-A3B-DFlash2-Q4_K_M.gguf"
$gpu = 0
$bindHost = '127.0.0.1'
$port = 8080
$cpuSet = [Environment]::GetEnvironmentVariable('CPUSET')
if ($null -eq $cpuSet) { $cpuSet = '0-3' }
if ($cpuSet -eq '') { $cpuSet = 'none' }
$cpuThreads = 4
$expertThreads = 3

$reasoning = 'on'
$logFile = "$PSScriptRoot\server.log"

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

$model = Get-Setting MODEL $model
$draftModel = Get-Setting DRAFT_MODEL $draftModel
$server = Get-Setting SERVER $server
$gpu = Get-Setting GPU $gpu
$specType = Get-Setting SPEC_TYPE $specType
$cpuThreads = Get-Setting CPU_THREADS $cpuThreads
$expertThreads = Get-Setting EXPERT_THREADS $expertThreads
$npred = Get-Setting NPRED $npred
$prefetch = Get-Setting PREFETCH_BUDGET $prefetch
$threshold = Get-Setting DRAFT_THRESHOLD $threshold
$cpu_cap = Get-Setting CPU_CAP $cpu_cap

$dryRun = $false
$extra = @()
$passThrough = $false
foreach ($arg in $args) {
    if ($passThrough) { $extra += $arg; continue }
    switch -CaseSensitive ($arg) {
        { $_ -in '-h', '--help' } {
            @'
Usage: .\start-server.ps1 [--dry-run] [-- additional llama-server arguments]
The server runs in the foreground. Ctrl-C stops it.
Set environment variables in PowerShell before starting, for example:
  $env:GPU = '0'
  $env:CPUSET = '0-3'
  $env:MODEL = 'D:\Models\main.gguf'
  $env:DRAFT_MODEL = 'D:\Models\draft.gguf'
CPUSET defaults to 0-3. Set it to none to disable affinity.
CPU_THREADS is the total budget. EXPERT_THREADS=0 disables the hybrid pool.
CUDA_VISIBLE_DEVICES takes precedence over GPU. SPEC_TYPE defaults to draft-mtp.
Edit $nclayers and $ncslots at the top of start-server.ps1 to set resident MoE layers and cache slots.
Edit $draft_max there to set the maximum number of draft tokens per round; default 3.
Edit $mtp_ubatch there to set the MTP prompt processing micro-batch size.
Edit $npred, $prefetch, $threshold and $cpu_cap there to apply the selected tuning configuration.
Prediction width and prefetch copy budget are independent.
'@
            exit 0
        }
        '--dry-run' { $dryRun = $true }
        '--'        { $passThrough = $true }
        default     { Stop-Launcher "Unknown argument: $arg. Use --help." 2 }
    }
}

if ($cpuThreads -notmatch '^[1-9][0-9]*$') { Stop-Launcher 'CPU_THREADS must be positive.' 2 }
if ($expertThreads -notmatch '^[0-9]+$') { Stop-Launcher 'EXPERT_THREADS must be nonnegative.' 2 }
if ([int]$expertThreads -ge [int]$cpuThreads) { Stop-Launcher 'EXPERT_THREADS must be less than CPU_THREADS.' 2 }
if ($specType -cnotin 'none', 'draft-dflash', 'draft-mtp') { Stop-Launcher 'SPEC_TYPE: use none, draft-dflash or draft-mtp.' 2 }
if ($reasoning -cnotin 'on', 'off', 'auto') { Stop-Launcher 'REASONING: use on, off or auto.' 2 }
if (!(Test-Path -LiteralPath $server -PathType Leaf)) { Stop-Launcher "Missing executable: $server. Run build-cuda.bat first." }
if (!(Test-Path -LiteralPath $model -PathType Leaf)) { Stop-Launcher "Missing model: $model" }
if ($specType -eq 'draft-dflash' -and !(Test-Path -LiteralPath $draftModel -PathType Leaf)) { Stop-Launcher "Missing draft model: $draftModel" }

$mainThreads = [int]$cpuThreads - [int]$expertThreads
if ($null -eq $env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES = "$gpu" }
$env:GGML_MOE_HYBRID_MAX_EXPERTS = "$cpu_cap"
$env:GGML_MOE_HYBRID = '0'
if ([int]$expertThreads -gt 0 -and [int]$cpu_cap -ne 0) {
    $env:GGML_MOE_HYBRID = '1'
    $env:GGML_MOE_HYBRID_THREADS = "$expertThreads"
}
$env:LLAMA_COMPACT_GREEDY = '1'
$env:GGML_MOE_BATCH_COPY = '1'
$env:GGML_MOE_BATCH_PREFETCH = '1'
$env:GGML_MOE_PREFETCH_BUDGET = "$prefetch"
$env:GGML_MOE_CACHE_POLICY = 'decay'
$env:GGML_MOE_CACHE_DECAY = '8'
$env:GGML_CUDA_SMALL_CONCAT = '1'
$env:GGML_CUDA_SCHEDULE = 'auto'
$env:LLAMA_ARG_BACKEND_SAMPLING = '0'
$env:GGML_MOE_LAYER_SLOTS = Get-Setting MOE_LAYER_SLOTS '142,160,128,142,128,116,96,92,78,98,98,104,90,72,60,66,64,64,60,60,60,70,72,72,74,72,64,64,72,78,66,72,84,84,80,78,90,90,104,122'
$env:OMP_NUM_THREADS = "$mainThreads"
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

$serverArgs = @(
    '-m', $model, '--alias', 'qwenedge', '--host', $bindHost, '--port', $port
    '--split-mode', 'none', '--main-gpu', '0', '-ngl', 'all', '--fit', 'off'
    '-t', $mainThreads, '-tb', $mainThreads, '--poll', '0', '--poll-batch', '0'
    '-c', (Get-Setting CTX_SIZE 10240), '-b', (Get-Setting BATCH_SIZE 2048), '-ub', (Get-Setting UBATCH_SIZE 2048), '-fa', 'on'
    '--parallel', '1', '--threads-http', '2', '--cache-ram', '0', '--ctx-checkpoints', '0'
    '--no-context-shift', '--no-cache-prompt', '--load-mode', 'none'
    '--n-cache-layers', $nclayers, '--n-cache-slots', $ncslots
    '--n-cache-predict', $npred, '--n-cache-pin', (Get-Setting CACHE_PIN 16000)
    '--spec-type', $specType, '--spec-draft-n-max', $draft_max
    '--spec-draft-n-min', '0', '--spec-draft-p-min', $threshold
    '--spec-draft-threads', $mainThreads, '--spec-draft-threads-batch', $mainThreads
    '--spec-draft-sampling', 'greedy', '--temp', '0', '--samplers', 'temperature'
    '--repeat-penalty', '1', '--seed', '42', '--n-predict', '4096'
    '--jinja', '--reasoning', $reasoning, '--reasoning-format', 'deepseek'
    '--perf', '--no-webui', '--log-file', $logFile
)
if ($specType -eq 'draft-dflash') { $serverArgs += '--spec-draft-model', $draftModel, '--spec-draft-ngl', 'all' }
if ($specType -eq 'draft-mtp') { $serverArgs += '--spec-mtp-ubatch', $mtp_ubatch }
$serverArgs += $extra
"GPU=$env:CUDA_VISIBLE_DEVICES CPUSET=$cpuSet main_threads=$mainThreads expert_threads=$expertThreads hybrid=$env:GGML_MOE_HYBRID cpu_cap=$cpu_cap npred=$npred prefetch=$prefetch"
$display = (@($server) + $serverArgs) | ForEach-Object { if ("$_" -match '\s') { '"{0}"' -f $_ } else { $_ } }
"Command: $($display -join ' ')"
if ($dryRun) { exit 0 }

if ($cpuSet -ne 'none') {
    $process = [Diagnostics.Process]::GetCurrentProcess()
    $available = $process.ProcessorAffinity.ToInt64()
    $mask = 0L
    $count = 0
    foreach ($part in ($cpuSet -split ',')) {
        if ($part -notmatch '^(\d+)(?:-(\d+))?$') { Stop-Launcher "Invalid CPUSET: $cpuSet" }
        $first = [int]$Matches[1]
        $last = if ($Matches[2]) { [int]$Matches[2] } else { $first }
        if ($last -lt $first -or $last -ge [IntPtr]::Size * 8) { Stop-Launcher "Invalid CPU range: $part" }
        for ($cpu = $first; $cpu -le $last; $cpu++) {
            $bit = 1L -shl $cpu
            if (!($available -band $bit)) { Stop-Launcher "CPU $cpu unavailable. Set CPUSET to an available range or none." }
            if (!($mask -band $bit)) { $count++ }
            $mask = $mask -bor $bit
        }
    }
    if ($count -lt [int]$cpuThreads) { Stop-Launcher 'CPUSET contains fewer CPUs than CPU_THREADS.' }
    # The server inherits the mask from this launcher process.
    $process.ProcessorAffinity = [IntPtr]$mask
}

try {
    $address = [Net.Dns]::GetHostAddresses($bindHost)[0]
    $listener = [Net.Sockets.TcpListener]::new($address, [int]$port)
    $listener.ExclusiveAddressUse = $true
    $listener.Start()
    $listener.Stop()
} catch {
    Stop-Launcher "Cannot bind ${bindHost}:${port}: $($_.Exception.Message)"
}

$logPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($logFile)
[void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($logPath))
& (Get-Item -LiteralPath $server).FullName @serverArgs
exit $LASTEXITCODE
