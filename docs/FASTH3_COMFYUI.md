# HAYATE FastH3 INT8 (ComfyUI)

HAYATEの生成画面で **FastH3 INT8** を選び、テキストから動画と音声を生成する経路です。
通常H3、公式FastVideoとは別の実行環境を使います。

## Colabで準備

1. GitHub版 `colab/HAYATE.ipynb` をColabで開き、最新版取得とWebUI起動セルを実行。
2. 対応CUDA GPU (SM80以上) で、追加の「FastH3 INT8の実行環境を準備」セルの
   `INSTALL_COMFY_FASTH3` を有効にして実行。CPU/T4/TPUではこのセルをスキップ。
3. WebUIの設定から **FastH3：高速4-Step** 構成を選び、規約確認後に未取得モデルを取得。
4. 再スキャン後、生成画面で **FastH3 INT8** を選ぶ。初回は512×288・約5秒から確認。

既存のDriveノートブックには追加セルが自動挿入されません。GitHub版を開き直すか、
最新版の取得・展開後に次のセルを追加します。

```python
import runpy
runpy.run_path('/content/HAYATE-webui/colab/setup_comfy_fasth3.py', run_name='__main__')
```

モデルは約39.5 GiBの4ファイル。TEXT・VIDEO VAE・AUDIO VAEは通常H3と共有し、
追加で必要なのは約21.3 GiBのFastH3 DiTです。通常H3のW4A8 DiT、PDD、
約148GBの公式FastVideoスナップショットはこの経路では不要です。

## 実行内容

- INT8 ConvRot FastH3 / Video VAE、NVFP4 AWQ text encoder、FP32 Audio VAE
- Euler、固定4ステップsigma列、video/audio shift 12/3
- SolAttn VSA: 4×4×4 video cubes、conditioning保持。既定10%、7.5% / 5%も選択可能
- Chunk FeedForward: chunks=2、threshold=4096
- Mozer Fast VAE: tile batch=2、メモリ節約用1も選択可能

HAYATEが専用ComfyUIプロセスをloopback上でジョブごとに起動します。
WebSocket進捗をHAYATEへ送り、API履歴のMP4を検証して生成一覧へ保存します。
ジョブ終了・失敗・キャンセルで専用プロセスを終了します。途中保存は未対応です。
プロセスを共有しないため、毎回モデル読込時間が発生します。

画質、実GPU上の速度、1生成1円以下の達成は未検証です。記事の測定値をHAYATEの
測定結果とは扱いません。VSAの動作ログが確認できない実行は成功扱いにしません。
ダウンロード所要時間は取得・検証を含む最後の操作時間です。以前のファイルで
時刻記録がない場合は「記録なし」と表示します。

## 出典と固定版

- [参考記事](https://note.com/sepiablue/n/n4157aa9f4f7d)
- [著者のAPIワークフロー](https://github.com/sepiablue-ai/minimax_h3_workflows/blob/main/fasth3_vsa5_chunk2_fastvae_fhd.json)
- Kijai/ComfyUI VSA: `10febb01d7be73d1491cf5e5347b5ab8b6c2c09e`
- Kijai/ComfyUI-KJNodes: `57105374f47d0fbb49c9c3926fb981702e0a4b5c`
- Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE: `b719329e0ecf35f0ae08d241c363ed1e56adbb95`
- comfy-kitchen: `0.2.33` (専用venv、Colabのtorch版を保持)
- [SolAttn v5](https://github.com/user-attachments/files/31576773/sol_attn_minimax_v5.py):
  SHA256 `97c9d56fdc7c9a102e59bff9ac8d79503299514d061892088a03d99dcf415b0c`

既存の汎用H3VSAノードは使いません。外部コードはインストーラーで固定版を取得し、
各配布元のライセンスに従います。HAYATEは生成数学やVAEを再実装せず、APIを接続します。

## 開始・終了画像

生成画面から開始画像、任意で終了画像を指定できます。専用入力フォルダにPNGとして保存し、上流のMiniMaxH3ImageToVideoのfirst_frame / last_frameへ接続します。終了画像のみは受け付けません。画像条件付き生成の実GPUでの画質・速度は未検証です。
