# 一键：官方 train baseline（真实 dots LLM）
# 用法:
#   .\run_dots_baseline.ps1
#   .\run_dots_baseline.ps1 -Limit 5 -VerboseRun
#   .\run_dots_baseline.ps1 -Task ab -Limit 10 -Systems 'open_vocab'
param(
  [ValidateSet('a','b','ab')]
  [string]$Task = 'a',
  [int]$Limit = 5,
  [int]$Offset = 0,
  [string]$Systems = 'closed_vocab,open_vocab',
  [switch]$VerboseRun
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'dots_config.ps1')
Set-DotsOntoBridgEnv

$py = $env:MIMO_PYTHON
if (-not $py) { $py = 'python' }

$extra = @()
if ($VerboseRun) { $extra += '--verbose' }

& $py (Join-Path $PSScriptRoot 'run_official_train_baseline.py') `
  --backend openai --task $Task --limit $Limit --offset $Offset `
  --model $env:ONTOBRIDG_MODEL --systems $Systems @extra
exit $LASTEXITCODE

