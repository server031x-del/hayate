param(
    [string]$Distribution = "Ubuntu",
    [string]$VenvPath = "",
    [string]$ModelDir = "",
    [switch]$DownloadModel
)

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
$scriptPath = Join-Path $workspaceRoot "scripts\setup_fasth3_wsl.sh"
$removeNul = { param($value) ([string]$value -replace [char]0, "").Trim() }

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$Value)
    $raw = $Value.Trim()
    if ($raw -match '^(?<drive>[A-Za-z]):[\\/](?<rest>.*)$') {
        return "/mnt/$($Matches.drive.ToLower())/$($Matches.rest -replace '\\', '/')"
    }
    if ($raw -match '^\\\\wsl(?:\.localhost|\$)\\(?<distro>[^\\]+)\\(?<rest>.*)$') {
        return "/$($Matches.rest -replace '\\', '/')"
    }
    if ($raw.StartsWith('/')) {
        return $raw.Replace('\', '/')
    }
    $resolved = Resolve-Path -LiteralPath $raw -ErrorAction SilentlyContinue
    if ($resolved) {
        return Convert-ToWslPath ([string]$resolved)
    }
    return $raw.Replace('\', '/')
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "wsl.exe was not found. Install a Linux/Windows WSL environment supported by FastVideo."
}

$distributions = @(wsl.exe --list --quiet 2>$null | ForEach-Object { & $removeNul $_ } | Where-Object { $_ })
if ($distributions -notcontains $Distribution) {
    throw "WSL distribution '$Distribution' was not found. Available: $($distributions -join ', ')"
}

$linuxScript = Convert-ToWslPath ([string]$scriptPath)
if (-not $linuxScript) {
    throw "Could not resolve the setup script path inside WSL."
}

$arguments = @($linuxScript)
if ($VenvPath) { $arguments += @("--venv", $VenvPath) }
if ($ModelDir) {
    $linuxModelDir = Convert-ToWslPath $ModelDir
    if (-not $linuxModelDir) {
        throw "Could not convert the model directory to a WSL path: $ModelDir"
    }
    $arguments += @("--model-dir", $linuxModelDir)
}
if ($DownloadModel) { $arguments += "--download-model" }

Write-Host "FastH3 WSL setup: $Distribution" -ForegroundColor Cyan
& wsl.exe -d $Distribution -- bash @arguments
if ($LASTEXITCODE -ne 0) {
    throw "FastH3 WSL setup failed (exit code $LASTEXITCODE)"
}

$effectiveVenv = if ($VenvPath) {
    $VenvPath.TrimStart("/")
} else {
    "home/$((& $removeNul (wsl.exe -d $Distribution -- printenv USER)))/hayate-fasth3-venv"
}
Write-Host ""
Write-Host "HAYATE Settings -> FastVideo Python:" -ForegroundColor Green
Write-Host "wsl://$Distribution/$effectiveVenv/bin/python"
if ($ModelDir) {
    Write-Host "HAYATE Settings -> FastVideo model directory:" -ForegroundColor Green
    $resolvedModelDir = Resolve-Path -LiteralPath $ModelDir -ErrorAction SilentlyContinue
    if ($resolvedModelDir) {
        Write-Host $resolvedModelDir.Path
    } else {
        Write-Host $ModelDir
    }
}
