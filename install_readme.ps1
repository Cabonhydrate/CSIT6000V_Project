param(
    [string]$RepoRoot = '.'
)
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path -LiteralPath $RepoRoot).Path
$source = $PSScriptRoot
$rootReadme = Join-Path $repo 'README.md'
$taccDir = Join-Path $repo 'tacc'
$guide = Join-Path $source 'tacc\README_TACC.md'
$quick = Join-Path $source 'tacc\TACC_QUICKSTART.md'
$section = Join-Path $source 'root_tacc_section.md'

if (!(Test-Path -LiteralPath (Join-Path $repo 'backend\app\main.py')) -or
    !(Test-Path -LiteralPath (Join-Path $repo 'frontend\package.json'))) {
    throw 'RepoRoot must be the INNER project folder containing backend and frontend.'
}
if (!(Test-Path -LiteralPath $rootReadme)) {
    throw 'README.md is missing. Review the supplied complete README.md and create one before merging.'
}

$manifest = Get-Content -LiteralPath (Join-Path $source 'docs_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($entry in $manifest.files) {
    $file = Join-Path $source $entry.path.Replace('/', [IO.Path]::DirectorySeparatorChar)
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing document source: $($entry.path)"
    }
    $hash = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $entry.sha256) {
        throw "Document source has changed: $($entry.path)"
    }
}

$start = '<!-- TACC-COURSE-SDK-START -->'
$end = '<!-- TACC-COURSE-SDK-END -->'
$block = Get-Content -LiteralPath $section -Raw -Encoding UTF8
$content = Get-Content -LiteralPath $rootReadme -Raw -Encoding UTF8

# Preserve local README customizations: replace only our managed section, else insert it.
$startPos = $content.IndexOf($start)
$endPos = $content.IndexOf($end)
if (($startPos -ge 0) -xor ($endPos -ge 0)) {
    throw 'README contains only one managed TACC marker. Resolve the broken section manually.'
}
if ($startPos -ge 0) {
    if ($endPos -le $startPos) { throw 'README TACC markers are out of order.' }
    $updated = $content.Substring(0,$startPos) + $block.TrimEnd() + $content.Substring($endPos + $end.Length)
    Write-Host 'Replacing managed TACC README section.'
} else {
    $anchor = '## 3. Test without downloading real QASPER'
    $position = $content.IndexOf($anchor)
    if ($position -lt 0) { $position = $content.IndexOf('## References') }
    if ($position -ge 0) {
        $updated = $content.Substring(0,$position) + $block.TrimEnd() + "`r`n`r`n" + $content.Substring($position)
    } else {
        $updated = $content.TrimEnd() + "`r`n`r`n" + $block.TrimEnd() + "`r`n"
    }
    Write-Host 'Adding TACC section while retaining all existing README sections.'
}

# Correct exact known stale phrasing from the original Day1 README if present.
$updated = $updated.Replace('### Local model (no TACC)', '### Local model (Ollama; see TACC deployment below)')
$updated = $updated.Replace('9. **TACC (intentionally omitted):** future stage adds vLLM GPU deployment/configuration and verified model endpoint routing, without modifying the Agent workflow.',
 '9. **TACC course SDK documentation:** See the TACC section above and `tacc/README_TACC.md`. Actual GPU and network verification is still pending.')
$updated = $updated.Replace('> **Scope of this release:** Local application implementation only. No TACC configuration, cluster scheduler, TACC SSH tunnel, or remote GPU deployment.',
 '> **Historical Day1 scope:** The original package omitted TACC. Course TACC setup is now documented below; GPU deployment has not yet been verified.')

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backupDir = Join-Path $repo "patch_backups\tacc_docs_$stamp"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Copy-Item -LiteralPath $rootReadme -Destination (Join-Path $backupDir 'README.md') -Force
if (Test-Path -LiteralPath (Join-Path $taccDir 'README_TACC.md')) {
    New-Item -ItemType Directory -Path (Join-Path $backupDir 'tacc') -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $taccDir 'README_TACC.md') -Destination (Join-Path $backupDir 'tacc\README_TACC.md') -Force
}
if (Test-Path -LiteralPath (Join-Path $taccDir 'TACC_QUICKSTART.md')) {
    New-Item -ItemType Directory -Path (Join-Path $backupDir 'tacc') -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $taccDir 'TACC_QUICKSTART.md') -Destination (Join-Path $backupDir 'tacc\TACC_QUICKSTART.md') -Force
}

New-Item -ItemType Directory -Path $taccDir -Force | Out-Null
Set-Content -LiteralPath $rootReadme -Value $updated -Encoding UTF8 -NoNewline
Copy-Item -LiteralPath $guide -Destination (Join-Path $taccDir 'README_TACC.md') -Force
Copy-Item -LiteralPath $quick -Destination (Join-Path $taccDir 'TACC_QUICKSTART.md') -Force

Write-Host "Updated README.md and TACC docs. Backup: $backupDir"
Write-Host 'No program code, private keys, Slurm scripts or vLLM files were modified.'
