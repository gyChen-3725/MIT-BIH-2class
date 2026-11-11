$ErrorActionPreference = 'Stop'

$Repository = 'E:\envs\Projects\Sleep-apnea-detection\MIT-BIH-2class'
$RunRoot = Join-Path $Repository 'runs\daeac_raw_e3_epoch5fold_2class'
$Python = 'C:\Users\cgy_3\miniconda3\envs\pytorch-env\python.exe'
$StatusPath = Join-Path $RunRoot 'aggregate_status.txt'
$LogPath = Join-Path $RunRoot 'aggregate.log'

'Waiting for training process 29396.' | Set-Content -Encoding utf8 $StatusPath
Wait-Process -Id 29396 -ErrorAction SilentlyContinue

$Missing = @(0..4 | Where-Object {
    -not (Test-Path (Join-Path $RunRoot "fold_$_\metrics.json")) -or
    -not (Test-Path (Join-Path $RunRoot "fold_$_\predictions.csv"))
})
if ($Missing.Count -gt 0) {
    "Training stopped before completion; missing folds: $($Missing -join ', ')." |
        Set-Content -Encoding utf8 $StatusPath
    exit 1
}

Push-Location $Repository
try {
    & $Python -B scripts\aggregate_oof.py --config configs\daeac_e3_epoch5fold.yaml *>&1 |
        Out-File -FilePath $LogPath -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        "Aggregation failed with exit code $LASTEXITCODE; see aggregate.log." |
            Set-Content -Encoding utf8 $StatusPath
        exit $LASTEXITCODE
    }
    'Five folds completed and OOF aggregation passed.' |
        Set-Content -Encoding utf8 $StatusPath
} finally {
    Pop-Location
}
