param(
    [switch]$SkipSync,
    [switch]$SkipUpstream
)

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
Set-Location $workspaceRoot

Write-Host "HAYATE setup" -ForegroundColor Cyan
Write-Host "Workspace: $workspaceRoot"

if (-not $SkipSync -and -not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv が見つかりません。https://docs.astral.sh/uv/ からインストールして再実行してください。"
}

# Model weights are intentionally never downloaded by this bootstrap script.
# They are large, separately licensed artifacts and can be selected from the
# Studio Settings screen after the runtime has been installed.
$standardDirectories = @(
    "models",
    "models\minimax-h3-snapshot",
    "models\text_encoders",
    "models\vae",
    "models\lora",
    "outputs",
    "outputs\prompt_cache",
    "data\webui"
)
foreach ($relativePath in $standardDirectories) {
    $directory = Join-Path $workspaceRoot $relativePath
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

if (-not $SkipUpstream) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Error "git が見つかりません。監査済み maybleMyers/h3 checkout の取得にはGitが必要です。"
    }
    $lockPath = Join-Path $workspaceRoot "upstream\h3.lock.json"
    $upstreamLock = Get-Content -LiteralPath $lockPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $checkoutPath = Join-Path $workspaceRoot "upstream\h3"
    if (Test-Path -LiteralPath $checkoutPath) {
        if (-not (Test-Path -LiteralPath (Join-Path $checkoutPath ".git"))) {
            Write-Error "upstream\h3 は存在しますがGit checkoutではありません。既存内容は変更しません。"
        }
        $currentCommit = (& git -C $checkoutPath rev-parse HEAD 2>$null).Trim()
        if ($LASTEXITCODE -ne 0 -or $currentCommit -ne [string]$upstreamLock.commit) {
            Write-Error "upstream\h3 は監査済みcommitと一致しません。既存checkoutは変更しません。期待: $($upstreamLock.commit) / 現在: $currentCommit"
        }
        Write-Host "監査済みmaybleMyers/h3 checkoutを確認しました。"
    } else {
        Write-Host "監査済みmaybleMyers/h3 checkoutを取得します..." -ForegroundColor Yellow
        & git clone --no-checkout ([string]$upstreamLock.repository) $checkoutPath
        if ($LASTEXITCODE -ne 0) {
            throw "maybleMyers/h3 のcloneに失敗しました (exit code $LASTEXITCODE)"
        }
        & git -C $checkoutPath fetch --depth 1 origin ([string]$upstreamLock.commit)
        if ($LASTEXITCODE -ne 0) {
            throw "監査済みh3 commitの取得に失敗しました (exit code $LASTEXITCODE)"
        }
        & git -C $checkoutPath checkout --detach ([string]$upstreamLock.commit)
        if ($LASTEXITCODE -ne 0) {
            throw "監査済みh3 commitへのcheckoutに失敗しました (exit code $LASTEXITCODE)"
        }
    }
} else {
    Write-Host "maybleMyers/h3 checkoutの準備はスキップしました。"
}

if (-not $SkipSync) {
    Write-Host "Installing the generation and WebUI dependencies..." -ForegroundColor Yellow
    & uv sync --extra generation --extra webui
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync に失敗しました (exit code $LASTEXITCODE)"
    }
} else {
    Write-Host "依存関係の同期はスキップしました。"
}

Write-Host ""
Write-Host "セットアップが完了しました。" -ForegroundColor Green
Write-Host "1. HAYATE Studio の設定画面で『標準フォルダを準備』とモデル状態の確認"
Write-Host "2. ライセンスを確認して必要なモデルだけ『ダウンロード』"
Write-Host "3. .\START_HAYATE_WEBUI.cmd で Studio を起動"
Write-Host ""
Write-Host "モデルの容量が大きいため、Gitには重みを含めていません。"
