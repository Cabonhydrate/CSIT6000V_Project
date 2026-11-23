param(
    [string]$RepoRoot = "."
)
$ErrorActionPreference = "Stop"

$repo = (Resolve-Path $RepoRoot).Path
$package = $PSScriptRoot
$manifestPath = Join-Path $package "patch_manifest.json"
$manifest = Get-Content $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json

if (!(Test-Path (Join-Path $repo "backend\app\main.py")) -or
    !(Test-Path (Join-Path $repo "frontend\app\page.tsx"))) {
    throw "RepoRoot must be the inner adaptive-agentic-rag folder containing backend and frontend."
}

# Preflight everything: never partially install over unexpected local changes.
foreach ($entry in $manifest.files) {
    $relative = $entry.path.Replace('/', [IO.Path]::DirectorySeparatorChar)
    $destination = Join-Path $repo $relative
    $source = Join-Path $package $relative
    if (!(Test-Path $source)) { throw "Missing patch source: $source" }
    $actualNew = (Get-FileHash -Path $source -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualNew -ne $entry.new_sha256) { throw "Patch archive integrity mismatch: $relative" }
    if ($entry.expected_sha256) {
        if (!(Test-Path $destination)) { throw "Missing required baseline file: $relative" }
        $current = (Get-FileHash -Path $destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($current -ne $entry.expected_sha256) {
            throw "Baseline SHA256 mismatch: $relative. Do not overwrite local changes."
        }
    } elseif (Test-Path $destination) {
        throw "New patch file already exists: $relative. Inspect it before replacing."
    }
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backup = Join-Path $repo ("patch_backups\general_learning_v032_" + $stamp)
foreach ($entry in $manifest.files) {
    $relative = $entry.path.Replace('/', [IO.Path]::DirectorySeparatorChar)
    $destination = Join-Path $repo $relative
    $source = Join-Path $package $relative
    if (Test-Path $destination) {
        $backed = Join-Path $backup $relative
        New-Item -ItemType Directory -Force -Path (Split-Path $backed -Parent) | Out-Null
        Copy-Item -LiteralPath $destination -Destination $backed -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
    $installed = (Get-FileHash -Path $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($installed -ne $entry.new_sha256) { throw "Installation verification failed: $relative" }
    Write-Host ("Installed " + $relative)
}
Write-Host "Patch installed. Backups: $backup"
Write-Host "Run backend pytest and frontend tsc/build as described in README."
