# Chunked official train baseline runner (stable under slow dots API)
# Usage: & D:\NLP\src\run_chunked_baseline.ps1 -Task a -Total 30 -Chunk 5 -Systems open_vocab,closed_vocab
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

foreach ($t in $tasks) {
  $offset = 0
  while ($offset -lt $Total) {
    $n = [Math]::Min($Chunk, $Total - $offset)
    Write-Host ""
    Write-Host "=== TASK=$t OFFSET=$offset LIMIT=$n SYSTEMS=$Systems ===" -ForegroundColor Cyan
    & $py (Join-Path $PSScriptRoot 'run_official_train_baseline.py') `
      --backend openai --task $t --offset $offset --limit $n `
      --model $env:ONTOBRIDG_MODEL --systems $Systems --verbose
    $code = $LASTEXITCODE
    Write-Host "chunk exit=$code" -ForegroundColor $(if ($code -eq 0) { 'Green' } else { 'Yellow' })
    if ($code -ne 0) {
      Write-Warning "Chunk failed at offset=$offset; continuing after brief pause."
      Start-Sleep -Seconds 5
    }
    $offset += $n
    Start-Sleep -Seconds 1
  }
}
Write-Host "All chunks attempted for Task=$Task Total=$Total" -ForegroundColor Cyan
