# HAYATE shared ComfyUI runtime

KijaiのMiniMax H3 VSA単一ファイルは、現時点ではComfyUI側のVSA実装を使うのが
最も確実です。HAYATE本体とは別のM:ドライブ上のランタイムを使い、モデル重みは
`M:\Project\HAYATE\models`を追加モデルパスとして参照します。ComfyUI側へモデルを
コピーしないため、同じ重みを二重に保存しません。

## 配置

- ComfyUI: `M:\Project\HAYATE-ComfyUI`
- Python: `M:\Project\HAYATE-ComfyUI\.venv\Scripts\python.exe`
- 共有モデル: `M:\Project\HAYATE\models`
- 出力: `M:\Project\HAYATE\outputs`
- ComfyUIユーザーデータ: `M:\Project\HAYATE\data\comfyui-user`

`M:\Project\HAYATE-ComfyUI\extra_model_paths.yaml`で次を参照しています。

- `diffusion_models`: HAYATEのモデルルート
- `text_encoders`: `models\text_encoders`
- `vae`: HAYATEのモデルルートと`models\vae`
- `loras`: `models\lora`

## 起動

```powershell
.\scripts\start_comfyui_hayate.ps1
```

既定は`0.0.0.0:8189`です。認証を設定していないため、信頼できるVPN/LAN内で
のみ使用してください。ローカル端末だけで使う場合は`-Listen 127.0.0.1`を
明示します。

```powershell
.\scripts\start_comfyui_hayate.ps1 -Listen 0.0.0.0 -Port 8189
```

既定の起動はDynamic VRAM、async offload 2ストリーム、pinned memory無効です。
これは32GB RAM未満のマシンでも起動しやすくするための安全側設定です。速度を
優先する場合は、下記のTurbo構成を使います。

速度比較用に、Comfy-KitchenのINT8 attentionと品質変化が小さい高速化を有効に
できます。現在のRTX 3060環境では次の起動方法をベンチマーク対象にします。

```powershell
.\scripts\start_comfyui_hayate.ps1 -EnableComfyKitchenAttention -EnableFastOptimizations
```

### Turbo構成（現在の推奨）

RTX 3060 12GB、32GB RAM、同じVSAグラフ（608x352、124フレーム、4ステップ）で
実測した最速構成です。

```powershell
.\scripts\start_comfyui_hayate.ps1 -EnableTurbo
```

`-EnableTurbo`は上記のSageAttention、Triton、全`--fast`、pinned memory、
async offload 4ストリームをまとめて有効にするショートカットです。

測定値（モデルロード後のウォーム実行）は次のとおりです。

| 構成 | 実測時間 |
| --- | ---: |
| 既定（Pytorch attention、offload 2、pinned無効） | 120.7秒 |
| Comfy-Kitchen＋fp16/autotune、offload 4、pinned | 100.5秒 |
| Comfy-Kitchen＋Triton＋全`--fast`、offload 4、pinned | 88.5秒 |
| SageAttention＋Triton＋全`--fast`、offload 4、pinned | **80.5秒** |

表は同一マシンでの最速ウォーム値です。32GB RAMではバックグラウンド負荷と
ページングにより、Turboの実測は80.5〜96.4秒の範囲で揺れました。Turboは同一
シード比較で黒フレーム無し、608x352/24fps/124フレーム/音声付きの出力を確認
しています。SageAttentionは近似attentionのため、最終用途では品質を目視確認
してください。初回はTritonカーネル準備とモデルロードが加わります。
またpinned memoryが約13GB確保されるため、空きRAMが少ない場合は次の安全側構成に
戻してください。

Turboに必要な追加ランタイムは、ComfyUIの仮想環境（`M:\Project\HAYATE-ComfyUI\.venv`）
へ導入済みです。`triton-windows==3.6.0.post26`と公式のSageAttention 2.2.0
cu130/torch2.10+ホイールを使用し、RTX 3060上でCUDA smoke testを通しています。
モデルは引き続きHAYATEの`M:\Project\HAYATE\models`を参照し、重みを複製しません。
別マシンで同じプロファイルを構築する場合（ComfyUIのvenv作成後）は、次の2行を
実行します。キャッシュはM:側へ置きます。

```powershell
$py = "M:\Project\HAYATE-ComfyUI\.venv\Scripts\python.exe"
$env:UV_CACHE_DIR = "M:\Project\HAYATE-ComfyUI\.cache\uv"
uv pip install --python $py "triton-windows==3.6.0.post26"
uv pip install --python $py --no-deps `
  "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post5/sageattention-2.2.0%2Bcu130torch2.10.0andhigher.post5-cp310-abi3-win_amd64.whl"
```

```powershell
.\scripts\start_comfyui_hayate.ps1 `
  -EnableComfyKitchenAttention `
  -EnableFastOptimizations `
  -AsyncOffloadStreams 2
```

`-EnableFastDisk`はM: SSDで比較しましたが、ウォーム118.8秒となりTurboより遅かった
ため、通常は使用しません。`-EnableSageAttention`と`-EnableComfyKitchenAttention`、
`-EnableFastDisk`と`-EnablePinnedMemory`は同時指定できません。

`--fast`は実験的機能を含むため、出力の色・動き・音声を確認してから常用します。

## Kijai VSAモデル

Kijaiの`minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors`
は、HAYATEのモデルセットアップから`M:\Project\HAYATE\models`へ取得します。
ComfyUIの`models`フォルダへコピーしません。モデルカードはComfyUI 0.31.0以上と
`int8_convrot` VAEの同バージョンを要求しており、HAYATE専用ブランチはそのVSA
パッチを含む`Kijai/ComfyUI`の`vsa`ブランチに固定しています。

このチェックポイントはHAYATE標準のmayble H3ローダーやFastVideo公式ディレクトリ
ローダーへ渡さず、ComfyUI VSAワークフローだけで使用します。
