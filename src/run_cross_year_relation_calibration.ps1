# Cross-year LLMs4OL 2025 relation calibration.
param(
  [int]$Seed = 20260923,
  [ValidateRange(1, 4)]
  [int]$Workers = 2,
  [ValidateRange(4096, 65536)]
  [int]$MaxTokens = 16384,
  [ValidateRange(0, 10000)]
  [int]$PriorFailedConnectionAttempts = 0,
  [switch]$AuditOnly,
  [switch]$RefreshCache,
  [switch]$VerboseRun
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'dots_config.ps1')
Set-DotsOntoBridgEnv

if ([string]::IsNullOrWhiteSpace($env:ONTOBRIDG_TIMEOUT)) {
  $env:ONTOBRIDG_TIMEOUT = '240'
}
if ([string]::IsNullOrWhiteSpace($env:ONTOBRIDG_MAX_RETRIES)) {
  $env:ONTOBRIDG_MAX_RETRIES = '4'
}
$env:ONTOBRIDG_MAX_TOKENS = [string]$MaxTokens

$py = $env:MIMO_PYTHON
if ([string]::IsNullOrWhiteSpace($py)) { $py = 'python' }

$extra = @()
if ($AuditOnly) { $extra += '--audit-only' }
if ($RefreshCache) { $extra += '--refresh-cache' }
if ($VerboseRun) { $extra += '--verbose' }

& $py (Join-Path $PSScriptRoot 'run_cross_year_relation_calibration.py') `
  --seed $Seed `
  --prior-failed-connection-attempts $PriorFailedConnectionAttempts `
  --workers $Workers `
  --backend openai `
  --model $env:ONTOBRIDG_MODEL `
  @extra
exit $LASTEXITCODE
