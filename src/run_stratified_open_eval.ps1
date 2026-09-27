# Leakage-controlled open-vocabulary comparison on non-tax-density strata.
param(
  [ValidateSet('a', 'b', 'ab')]
  [string]$Task = 'ab',
  [ValidateRange(1, 20)]
  [int]$SamplesPerStratum = 5,
  [int]$Seed = 20260922,
  [ValidateRange(1, 4)]
  [int]$Workers = 2,
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
  [string]$Output = '',
  [string]$Systems = 'open_vocab,open_vocab_guarded',
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
if (-not [string]::IsNullOrWhiteSpace($Output)) {
  $extra += '--output'
  $extra += $Output
}

& $py (Join-Path $PSScriptRoot 'run_stratified_open_eval.py') `
  --task $Task `
  --samples-per-stratum $SamplesPerStratum `
  --seed $Seed `
  --workers $Workers `
  --confidence-threshold $ConfidenceThreshold `
  --implicit-confidence-threshold $ImplicitConfidenceThreshold `
  --calibrated-confidence-threshold $CalibratedConfidenceThreshold `
  --max-nontax $MaxNonTax `
  --max-implicit-gap-chars $MaxImplicitGapChars `
  --systems $Systems `
  --backend openai `
  --model $env:ONTOBRIDG_MODEL `
  @extra
exit $LASTEXITCODE
