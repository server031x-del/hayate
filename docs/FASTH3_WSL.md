# FastH3 / FastVideo（Windows WSL）

HAYATEのFastH3経路は、ComfyUI用のKijai単一
`minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors`
を読み込みません。実行に使うのは、FastVideo公式の自己完結型ディレクトリ
スナップショットだけです。

- 公開元: [FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)
- 固定revision: `5ea076f35b84da4c3c82217112fa733d8eea2ae1`
- 実行対象: T2VA（テキスト→映像＋音声）
- サンプリング: `[999, 749, 500, 250]` + 終端0（5 sigma points、4 DiT forwards）
- VSA: sparsity `0.9`、tile `64`

公式FastVideoの対応OSはLinuxまたはWindows WSL、Python 3.10–3.12、CUDA
12.6/13.0です。WindowsネイティブのHAYATEは、`wsl://`ランタイム記法で
WSL側の専用環境を呼び出します。通常のH3実行環境とは分離されるため、
FastVideoの依存関係で既存のWebUIを壊しません。

## 1. ランタイムだけを準備する

管理者権限は不要です。PowerShellでリポジトリのルートから実行します。

```powershell
.\scripts\setup_fasth3_wsl.ps1 -Distribution Ubuntu
```

この操作は専用の`~/hayate-fasth3-venv`を作成し、FastVideo 0.2.1、
FastVideo Kernel 0.3.5、PyTorch 2.12 CUDA 13.0を入れて、CUDA/APIの
プローブまで実行します。モデル重みは取得しません。

この標準セットはstrict（Triton VSA）用です。Blackwell最速プロファイルを
使う場合は、FastVideo公式の`[fasth3]`追加依存（FA4とsm100a対応kernel）を
同じ専用環境へ入れてください。Studioの診断が`flash_attn.cute`と
`vsa_sm100a`を確認できない間は、最速プロファイルを有効にしません。

完了時に表示された値を、Studioの設定へそのまま入力します。

```text
FastVideo Python:
wsl://Ubuntu/home/<WSLユーザー>/hayate-fasth3-venv/bin/python
```

## 2. モデルを取得する（明示的に実行した場合だけ）

公式スナップショットはおよそ148 GB（約138 GiB）です。Hugging Faceキャッシュと一時
領域を含め、保存先には少なくとも160 GiBの空きを用意してください。既定の
`M:`ドライブのように空きが足りない場所では、コマンドは開始前に停止します。
大容量モデルのため、SSDの空きが十分な`D:`などへ保存するのが安全です。

```powershell
.\scripts\setup_fasth3_wsl.ps1 `
  -Distribution Ubuntu `
  -ModelDir D:\HAYATE\models\fastvideo\FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree `
  -DownloadModel
```

公開元がライセンス同意済みのHugging Faceログインを要求する場合は、WSL側で
`hf auth login`を済ませるか、PowerShellから`HF_TOKEN`環境変数を一時設定して
実行してください。トークンはHAYATEの設定や生成プロセスへ保存・転送しません。

同じコマンドを再実行すると、同じrevisionの未取得ファイルをローカル
Hugging Faceキャッシュから再開します。別revisionのファイルを混在させない
ため、更新時は新しいディレクトリへ取得してから設定を切り替えてください。

取得後、Studioの **設定 → FastVideo model directory** にWindows側の保存先を
入力し、**利用条件を診断**を押します。診断はmanifest、各コンポーネントの
config、3種のweight index、全シャード、CUDA、VSAカーネルを読み込みなしで
確認します。

CLIでも同じ確認ができます。

```powershell
uv run hayate fasth3-check `
  --python wsl://Ubuntu/home/<WSLユーザー>/hayate-fasth3-venv/bin/python `
  --model-dir D:\HAYATE\models\fastvideo\FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree
```

## 現在の安全な実行範囲

- HAYATEの標準H3（maybleMyers/h3）経路とは別プロセス・別ランタイムです。
- 12 GiB級GPUではDiTをlayerwise CPU offloadし、Text Encoder/VAEもoffload
 します。これは **FastH3 VSA 4-Step** のstrict経路です。
- Blackwell（compute capability 10.0/10.3）でsm100a VSA拡張とFA4が入っている
  場合だけ、Studioの **FastH3 Blackwell最速** が有効になります。これは
  `profile=all`、regional compile、parallel VAE decodeを使うため、strict経路と
  数値が一致しません。診断が条件を満たさない場合はstrictを選びます。
- 現在のアダプターは1ジョブ1 GPU、T2VAのみです。I2V/FL2VAとFastH3内の
  複数GPU並列は、別の品質・メモリ検証が終わるまで有効化しません。
- WSLから見えるシステムRAMが少ない場合、診断は警告を表示します。FastVideo
  の35BモデルをCPU offloadするには十分なRAMと高速なSSDが必要です。

## 参照

- [FastVideo GPU installation](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/getting_started/installation/gpu.md)
- [FastVideo basic FastH3 example](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3.py)
- [FastH3 model card](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)
