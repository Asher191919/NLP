# Guarded-v2 generation for the blind human annotation packet.
param(
  [ValidateRange(1, 4)]
  [int]$Workers = 2,
  [ValidateRange(4096, 65536)]
  [int]$MaxTokens = 16384,
  [ValidateRange(1, 8)]
  [int]$ParseRetries = 2,
  [ValidateRange(0, 10000)]
  [int]$PriorFailedUnitAttempts = 0,
  [ValidateRange(0, 10000)]
  [int]$PriorFailedLowLevelAttempts = 0,
  [string]$Manifest = "",
  [string]$FrozenAnnotations = "",
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
if ([string]::IsNullOrWhiteSpace($Manifest)) {
  $Manifest = Join-Path (Split-Path $PSScriptRoot -Parent) `
    'experiments\annotations\relation_text_annotation_v1\manifest.json'
}

$extra = @()
if ($AuditOnly) { $extra += '--audit-only' }
if ($RefreshCache) { $extra += '--refresh-cache' }
if ($VerboseRun) { $extra += '--verbose' }
if (-not [string]::IsNullOrWhiteSpace($FrozenAnnotations)) {
  $extra += '--frozen-annotations'
  $extra += $FrozenAnnotations
}

& $py (Join-Path $PSScriptRoot 'run_annotated_relation_calibration.py') `
  --manifest $Manifest `
  --workers $Workers `
  --parse-retries $ParseRetries `
  --prior-failed-unit-attempts $PriorFailedUnitAttempts `
  --prior-failed-low-level-attempts $PriorFailedLowLevelAttempts `
  --backend openai `
  --model $env:ONTOBRIDG_MODEL `
  @extra
exit $LASTEXITCODE
