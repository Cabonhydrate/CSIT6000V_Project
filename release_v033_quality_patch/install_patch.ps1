param([string]$RepoRoot = '.')
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $RepoRoot).Path
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'patch_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json

# Validate every destination before touching the repo (atomic preflight).
foreach ($entry in $manifest.files) {
    $relative = $entry.path.Replace('/', [IO.Path]::DirectorySeparatorChar)
    $destination = Join-Path $root $relative
    $source = Join-Path $PSScriptRoot $relative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Patch source is missing: $relative"
    }
    $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sourceHash -ne $entry.new_sha256) {
        throw "Patch source SHA256 mismatch: $relative"
    }
    if ($entry.expected_sha256) {
        if (-not (Test-Path -LiteralPath $destination -PathType Leaf)) {
            throw "Existing source file is missing: $relative"
        }
        $current = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($current -ne $entry.expected_sha256) {
            throw "Baseline SHA256 mismatch for $relative. Do not overwrite local modifications."
        }
    } elseif (Test-Path -LiteralPath $destination) {
        throw "New test file already exists: $relative"
    }
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backup = Join-Path $root "patch_backups\release_v033_$stamp"
foreach ($entry in $manifest.files) {
    $relative = $entry.path.Replace('/', [IO.Path]::DirectorySeparatorChar)
    $destination = Join-Path $root $relative
    $source = Join-Path $PSScriptRoot $relative
    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    if ($entry.expected_sha256) {
        $backupPath = Join-Path $backup $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $backupPath) -Force | Out-Null
        Copy-Item -LiteralPath $destination -Destination $backupPath
    }
    Copy-Item -LiteralPath $source -Destination $destination -Force
}
Write-Host "Installed release v0.3.3. Backups: $backup"
Write-Host 'Verify: cd backend; python -m pytest -q tests\test_release_quality_v033.py'
