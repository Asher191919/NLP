# Independent guarded-v2 relation calibration split.
param(
  [ValidateSet('a', 'b', 'ab')]
  [string]$Task = 'ab',
  [ValidateRange(1, 100)]
  [int]$SamplesPerTask = 12,
  [string]$SampleIndices = '',
  [ValidateSet('open_vocab_guarded_v2', 'open_vocab_guarded_v3_relations')]
  [string]$System = 'open_vocab_guarded_v2',
  [int]$Seed = 20260923,
  [ValidateRange(0, 10000)]
  [int]$ExcludePrefix = 30,
  [ValidateRange(0, 10000)]
  [int]$PriorFailedConnectionAttempts = 0,
  [string]$ExcludeEvalResult = 'D:\NLP\experiments\results\official_stratified_open_openai_ab_s5_seed20260922.json',
  [ValidateRange(1, 4)]
  [int]$Workers = 2,
  [ValidateRange(4096, 65536)]
  [int]$MaxTokens = 16384,
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
if (-not [string]::IsNullOrWhiteSpace($SampleIndices)) {
  $extra += '--sample-indices'
  $extra += $SampleIndices
}

& $py (Join-Path $PSScriptRoot 'run_relation_calibration.py') `
  --task $Task `
  --samples-per-task $SamplesPerTask `
  --system $System `
  --seed $Seed `
  --exclude-prefix $ExcludePrefix `
  --prior-failed-connection-attempts $PriorFailedConnectionAttempts `
  --exclude-eval-result $ExcludeEvalResult `
  --workers $Workers `
  --backend openai `
  --model $env:ONTOBRIDG_MODEL `
  @extra
exit $LASTEXITCODE
