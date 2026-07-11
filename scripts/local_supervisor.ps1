# Patient training supervisor for a shared consumer GPU (Windows).
#
# The desktop (browser, Discord, games) competes for VRAM, so instead of failing,
# this waits until enough VRAM is free, then runs train -> eval, retrying with
# checkpoint resume if the GPU is yanked away mid-run. Progress: outputs/supervisor.log
#
#   powershell -File scripts/local_supervisor.ps1

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".venv-train\Scripts\python.exe"
$log = "outputs\supervisor.log"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_OFFLINE = "1"
New-Item -ItemType Directory -Force outputs | Out-Null

function Log($msg) { "[$(Get-Date -Format 'MM-dd HH:mm:ss')] $msg" | Tee-Object -Append $log }
function FreeMiB { [int](nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) }
function WaitVram($needMiB) {
    while ((FreeMiB) -lt $needMiB) {
        Log "waiting for VRAM: $(FreeMiB) MiB free, need $needMiB"
        Start-Sleep 300
    }
    Log "VRAM available: $(FreeMiB) MiB free"
}

# ---- stage 1: train (sentinel: training_log.json written on success) ----
$trainDone = "models\tickettriage-lora\training_log.json"
$attempt = 0
while (-not (Test-Path $trainDone) -and $attempt -lt 20) {
    $attempt++
    WaitVram 7000
    Log "training attempt $attempt (plain LoRA; DoRA is impractically slow on bnb-4bit)"
    & $py -u scripts\train.py --no-dora --resume *>> outputs\train.log
    if (Test-Path $trainDone) { Log "TRAINING COMPLETE" } else { Log "training exited without finishing (rc=$LASTEXITCODE); will retry"; Start-Sleep 120 }
}
if (-not (Test-Path $trainDone)) { Log "SUPERVISOR GIVING UP on training after $attempt attempts"; exit 1 }

# ---- stage 2: eval (sentinel: results/eval_results.json) ----
$evalDone = "results\eval_results.json"
$attempt = 0
while (-not (Test-Path $evalDone) -and $attempt -lt 20) {
    $attempt++
    WaitVram 5200   # generation tolerates WDDM paging; training did not
    Log "eval attempt $attempt (base vs fine-tuned, free metrics only)"
    & $py -u scripts\run_eval.py `
        --base "hf:unsloth/Qwen3-8B-bnb-4bit" `
        --finetuned "hf:unsloth/Qwen3-8B-bnb-4bit@models/tickettriage-lora" `
        --skip-judge *>> outputs\eval.log
    if (Test-Path $evalDone) { Log "EVAL COMPLETE" } else { Log "eval exited without finishing (rc=$LASTEXITCODE); will retry"; Start-Sleep 120 }
}
if (Test-Path $evalDone) { Log "SUPERVISOR DONE - training + eval finished" } else { Log "SUPERVISOR GIVING UP on eval"; exit 1 }
