# =============================================================================
# dots studio configuration (base URL and model are prefilled)
# Read the API key from ONTOBRIDG_API_KEY / OPENAI_API_KEY first.
# This file is safe to track. Never write a real key here.
# Source: https://dots.ai/platform/docs
# =============================================================================

$script:DOTS_BASE_URL = 'https://note3-prev-api.askdiandian.com/v1'
$script:DOTS_MODEL    = 'dots3-note-prev'

# Backward-compatible placeholder; keep this empty.
$script:DOTS_API_KEY  = ''

function Get-DotsApiKey {
  if (-not [string]::IsNullOrWhiteSpace($env:ONTOBRIDG_API_KEY)) { return $env:ONTOBRIDG_API_KEY }
  if (-not [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) { return $env:OPENAI_API_KEY }
  if (-not [string]::IsNullOrWhiteSpace($script:DOTS_API_KEY)) {
    return $script:DOTS_API_KEY
  }
  $jsonPath = Join-Path $PSScriptRoot 'dots_config.json'
  if (Test-Path $jsonPath) {
    try {
      $cfg = Get-Content -Raw -Path $jsonPath -Encoding UTF8 | ConvertFrom-Json
      $k = [string]$cfg.api_key
      if (-not [string]::IsNullOrWhiteSpace($k)) { return $k }
    } catch {}
  }
  return ''
}

function Set-DotsOntoBridgEnv {
  $key = Get-DotsApiKey
  if ([string]::IsNullOrWhiteSpace($key)) {
    throw "Missing API key. Set ONTOBRIDG_API_KEY or OPENAI_API_KEY."
  }
  $env:ONTOBRIDG_BASE_URL = $script:DOTS_BASE_URL
  $env:ONTOBRIDG_MODEL    = $script:DOTS_MODEL
  $env:ONTOBRIDG_API_KEY  = $key
  $env:OPENAI_API_KEY     = $key
  # Official prompts are long, so use a conservative default timeout.
  if ([string]::IsNullOrWhiteSpace($env:ONTOBRIDG_TIMEOUT)) { $env:ONTOBRIDG_TIMEOUT = '180' }
  if ([string]::IsNullOrWhiteSpace($env:ONTOBRIDG_MAX_RETRIES)) { $env:ONTOBRIDG_MAX_RETRIES = '3' }
  Write-Host "Configured:"
  Write-Host "  BASE_URL = $($env:ONTOBRIDG_BASE_URL)"
  Write-Host "  MODEL    = $($env:ONTOBRIDG_MODEL)"
  Write-Host "  API_KEY  = set ($($key.Length) chars)"
  Write-Host "  TIMEOUT  = $($env:ONTOBRIDG_TIMEOUT)s"
}
