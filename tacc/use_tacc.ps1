param(
    [string]$BaseUrl = 'http://127.0.0.1:8001/v1',
    [string]$Model = 'qwen3.8-27b',
    [string]$ApiKeyEnv = 'TACC_API_KEY',
    [switch]$SkipPreflight
)
$ErrorActionPreference = 'Stop'
# Process-only settings. Does not overwrite .env or the Ollama baseline.
$key = [Environment]::GetEnvironmentVariable($ApiKeyEnv,'Process')
if (-not $key) { $key='local' }
if (-not $SkipPreflight) {
    $oldKey=$env:MODEL_API_KEY
    $env:MODEL_API_KEY=$key
    python "$PSScriptRoot\smoke.py" --base-url $BaseUrl --model $Model
    $code=$LASTEXITCODE
    $env:MODEL_API_KEY=$oldKey
    if ($code -ne 0) { throw "TACC smoke failed ($code); backend settings not switched." }
}
$env:MODEL_BASE_URL = $BaseUrl
$env:MODEL_NAME = $Model
$env:MODEL_PROVIDER = 'openai'
$env:MODEL_API_KEY = $key
$env:MODEL_REASONING_EFFORT = ''
$env:MODEL_TIMEOUT = '240'
Write-Host "Process environment now targets vLLM: $BaseUrl / $Model"
Write-Host 'Start Uvicorn from this SAME shell. For Ollama, open a clean terminal or restore env settings.'
