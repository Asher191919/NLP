# Resumable leakage-controlled LLMs4OL 2026 train evaluation.
# Examples:
#   .\run_blind_eval.ps1 -Task ab -Limit 1 -Workers 1 -VerboseRun
#   .\run_blind_eval.ps1 -Task ab -Limit 30 -Workers 2 -VerboseRun
param(
  [ValidateSet('a', 'b', 'ab')]
  [string]$Task = 'ab',
  [int]$Limit = 30,
  [int]$Offset = 0,
  [ValidateRange(1, 4)]
  [int]$Workers = 1,
  [string]$Systems = 'closed_vocab,open_vocab',
  [ValidateRange(0.0, 1.0)]
  [double]$ConfidenceThreshold = 0.85,
  [ValidateRange(0.0, 1.0)]
  [double]$ImplicitConfidenceThreshold = 0.95,
  [ValidateRange(0.0, 1.0)]
  [double]$CalibratedConfidenceThreshold = 0.5,
  [ValidateRange(0, 80)]
  [int]$MaxNonTax = 8,
  [ValidateRange(0, 10000)]
  [int]$MaxImplicitGapChars = 600,
  [string]$CalibrationFile = '',
  [string]$AllowedNonTaxPredicates = '',
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
if ([string]::IsNullOrWhiteSpace($env:ONTOBRIDG_MAX_TOKENS)) {
  # Some official samples exhaust 8192 tokens on reasoning before emitting final JSON.
  $env:ONTOBRIDG_MAX_TOKENS = '16384'
}

$py = $env:MIMO_PYTHON
if ([string]::IsNullOrWhiteSpace($py)) { $py = 'python' }

$extra = @()
if ($AuditOnly) { $extra += '--audit-only' }
if ($RefreshCache) { $extra += '--refresh-cache' }
if ($VerboseRun) { $extra += '--verbose' }
if (-not [string]::IsNullOrWhiteSpace($CalibrationFile)) {
  $extra += '--calibration-file'
  $extra += $CalibrationFile
}
if (-not [string]::IsNullOrWhiteSpace($AllowedNonTaxPredicates)) {
  $extra += '--allowed-nontax-predicates'
  $extra += $AllowedNonTaxPredicates
}

& $py (Join-Path $PSScriptRoot 'run_official_blind_eval.py') `
  --task $Task `
  --limit $Limit `
  --offset $Offset `
  --workers $Workers `
  --systems $Systems `
  --confidence-threshold $ConfidenceThreshold `
  --implicit-confidence-threshold $ImplicitConfidenceThreshold `
  --calibrated-confidence-threshold $CalibratedConfidenceThreshold `
  --max-nontax $MaxNonTax `
  --max-implicit-gap-chars $MaxImplicitGapChars `
  --backend openai `
  --model $env:ONTOBRIDG_MODEL `
  @extra
exit $LASTEXITCODE
