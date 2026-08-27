# HAYATEをGitから構築する

モデルの重みはサイズとライセンスの都合でGitリポジトリに含めません。
Gitから取得した後、依存関係と標準フォルダだけを先に準備し、必要なモデルは
Studioの設定画面から個別に取得します。

## Windows（推奨）

1. [uv](https://docs.astral.sh/uv/) をインストールします。
2. HAYATEをcloneして、リポジトリのルートで `SETUP_HAYATE.cmd` を実行します。
3. `START_HAYATE_WEBUI.cmd` を実行してStudioを開きます。
4. 設定 → モデルセットアップで標準フォルダを確認し、ライセンスを確認してから必要なモデルをダウンロードします。

セットアップは`upstream/h3.lock.json`に固定された監査済み
[`maybleMyers/h3`](https://github.com/maybleMyers/h3) commitも
`upstream/h3/`へ取得します。既存checkoutがある場合は上書きやcheckout変更を行わず、
commitが一致しなければ停止して明示します。既存の外部checkoutを設定画面から使用する場合は
`-SkipUpstream`を指定できます。

PowerShellから実行する場合は次のコマンドです。

```powershell
Set-Location M:/Project/HAYATE
.\scripts\setup_hayate.ps1
```

依存関係がすでにインストール済みで標準フォルダだけ作りたい場合は、次を使用します。

```powershell
.\scripts\setup_hayate.ps1 -SkipSync
```

既存の外部h3 checkoutを使い、標準checkoutの取得も省略する場合:

```powershell
.\scripts\setup_hayate.ps1 -SkipUpstream
```

## Linux / WSL

```bash
uv sync --extra generation --extra webui
git clone https://github.com/maybleMyers/h3 upstream/h3
git -C upstream/h3 checkout --detach 94220c1fdf14d6d9d40be06fb99f55b27c0d9024
mkdir -p models/minimax-h3-snapshot models/text_encoders models/vae models/lora \
  outputs/prompt_cache data/webui
uv run hayate webui --host 0.0.0.0 --port 7860 --allow-network
```

## モデルについて

Studioのダウンロード機能は、HAYATEが検証済みとして登録した固定の公開元・revision・
SHA-256だけを使用します。途中で止めても安全に再実行でき、同名の既存ファイルが一致しない
場合は上書きしません。モデルの利用条件は各公開元の規約を確認してください。

標準構成は次のとおりです。

```text
models/
├─ minimax-h3-snapshot/   # upstreamのconfig・scheduler・tokenizer等
├─ text_encoders/         # Qwen3-VL NVFP4/AWQ
├─ vae/                   # Audio VAE / 追加VAE
└─ lora/                  # PDD Acc LoRA と AdaLN affine map
```

大容量モデルの取得は初回セットアップに含めず、空き容量・ライセンス・必要な
プロファイルを確認したうえで行います。
