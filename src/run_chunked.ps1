# Chunked runner for official train baselines (stable under slow dots API)
# Usage:
#   .\run_chunked.ps1 -Task a -Total 50 -Chunk 5
#   .\run_chunked.ps1 -Task ab -Total 30 -Chunk 5
param(
  [ValidateSet('a', 'b', 'ab')]
  [string]$Task = 'a',
  [int]$Total = 30,
  [int]$Chunk = 5,
  [string]$Systems = 'closed_vocab,open_vocab'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'dots_config.ps1')
Set-DotsOntoBridgEnv
if (-not $env:ONTOBRIDG_TIMEOUT) { $env:ONTOBRIDG_TIMEOUT = '300' }
if (-not $env:ONTOBRIDG_MAX_RETRIES) { $env:ONTOBRIDG_MAX_RETRIES = '3' }

$py = $env:MIMO_PYTHON
if (-not $py) { $py = 'python' }

$tasks = @()
if ($Task -eq 'ab') { $tasks = @('a', 'b') } else { $tasks = @($Task) }

$script = Join-Path $PSScriptRoot 'run_official_train_baseline.py'
foreach ($t in $tasks) {
  $offset = 0
  while ($offset -lt $Total) {
    $n = [Math]::Min($Chunk, $Total - $offset)
    Write-Host ""
    Write-Host "=== TASK=$t OFFSET=$offset LIMIT=$n SYSTEMS=$Systems ===" -ForegroundColor Cyan
    & $py $script --backend openai --task $t --offset $offset --limit $n `
      --systems $Systems --verbose --model $env:ONTOBRIDG_MODEL
    $code = $LASTEXITCODE
    Write-Host "chunk exit=$code offset=$offset" -ForegroundColor $(if ($code -eq 0) { 'Green' } else { 'Yellow' })
    if ($code -ne 0) {
      Write-Warning "Chunk failed; retrying same offset after 5s..."
      Start-Sleep -Seconds 5
      & $py $script --backend openai --task $t --offset $offset --limit $n `
        --systems $Systems --verbose --model $env:ONTOBRIDG_MODEL
    }
    $offset += $n
  }
}
Write-Host "All requested chunks attempted for Task=$Task Total=$Total" -ForegroundColor Green
