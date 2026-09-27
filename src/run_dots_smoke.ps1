# 一键：冒烟测试（配置已预填，只需 dots_config.ps1 里的 API Key）
# 用法: .\run_dots_smoke.ps1
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'dots_config.ps1')
Set-DotsOntoBridgEnv

$py = $env:MIMO_PYTHON
if (-not $py) { $py = 'python' }

& $py (Join-Path $PSScriptRoot 'test_api_smoke.py')
exit $LASTEXITCODE
