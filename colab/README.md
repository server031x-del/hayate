# HAYATE Google Colab

[Colabで開く](https://colab.research.google.com/github/server031x-del/hayate/blob/codex/colab-fast-h3/colab/HAYATE.ipynb)

**Experimental: 無料T4では環境診断・契約テストまで。有料GPUでの生成速度・出力品質は未検証です。**
既存HAYATEの未コミット変更に依存しない独立したColab用アダプターです。
推論はVM内のComfyUI/VSAを再利用し、ノートブックから操作します。
外部トンネルや公開WebUI、ローカルPCのモデル起動は行いません。

1. ノートブックを開き、無料T4で環境診断とCPUのみのテストを実行します。
2. 有料環境ではSM80以降、VRAM 20 GiB以上、RAM 30 GiB以上を事前条件とします。
   RAM 64 GiB以上を推奨。この閾値は保守的な導入方針で、動作保証・実測最小値ではありません。
3. モデル約39.5 GiBと作業領域8 GiB以上の空きを確保します。
4. MiniMaxモデルのライセンスを確認し、モデル取得セルを明示的に実行します。
5. 608×352、124フレーム（24fpsで約5.17秒）、4ステップで生成します。
   同じセッションでseedを変えて繰り返すと、モデル・同一プロンプトのキャッシュを再利用できます。
6. 目視・音声確認で採用した本数を費用セルに入力します。最後に出力をダウンロードして
   ランタイムを接続解除します。VM終了時にモデルと出力は失われます。

## 費用

`1採用動画の円 = 接続時間(秒) / 3600 × CU/時 × 円/CU / 採用本数`

Colab画面の消費CU/時と、実際の購入金額÷購入CUを入力してください。
値が不明なら「不明」のままとし、無料テストの費用を有料時の実績に流用しません。
接続時間にはインストール、モデルDL、初回ロード、コンパイル、失敗、待機も含めます。
画面に表示される実際のCU消費差分がある場合は、その差分×円/CUで精算する方が正確です。
契約料金・税・未消化CUも含める場合は、実支出÷採用本数で別途評価してください。

例示（料金表ではありません）: 100円/時なら1円以内には平均36秒/採用動画が必要。
300円/時なら12秒です。現時点で1円以下を達成したという実測はありません。
解像度・長さを下げた動画を同じ成果として比較しないでください。

## 最適化方針と限界

- REUSE: Kijai ComfyUI固定commit、既存HAYATEの量子化モデル情報とVSAグラフ。
- MODIFY: MIT VSAノードを同梱し、VSA非対応・カーネル失敗時のdense fallbackをエラー化。
- ADD: Colab環境判定、固定revisionとSHA256検証、VM内永続ワーカー、成功出力確認、費用集計。
- モデルはVMのSSDへ取得し、ComfyUIにはsymlinkで参照。Drive越しの重みストリーミングを避けます。
- ColabのTorch/torchvision/torchaudioの組み合わせは制約ファイルで保持します。
  依存解決できないランタイムは停止し、未検証のCUDA wheelへ自動置換しません。
- VSAのTritonソースはPyPI vsa 0.0.3のSHA256固定アーカイブから取得します。
  Hopper専用拡張はビルドしません。GPU世代ごとの実カーネル検証は今後必要です。
- pinned memoryは空きRAM 48 GiB以上だけで有効化。全`--fast`やtorch.compileは
  初回負担と品質を実測するまで既定にしません。
- 公式FastVideoの約148 GiBスナップショットはこの経路では取得しません。
- TPUはCUDA/Tritonの代わりに選択しても動きません。この実装では対象外です。
- T4のSM75・標準RAMではこの高速化経路を停止します。別モデルへの勝手な置換はしません。
- 出力の先頭フレームが復号できることは確認しますが、黒画面、全フレーム品質、音声同期の
  合格を保証しません。費用の分母は人が採用した動画数で最終評価してください。

## 第三者コード

`vendor/h3_vsa`はMITの
[barelymining/ComfyUI-MiniMax-H3-FastVideo](https://github.com/barelymining/ComfyUI-MiniMax-H3-FastVideo)
に由来する既存ローカルHAYATE用改変版（2026-09-08取得）です。
既存版には埋込gateと量子化Linear対応が含まれます。LICENSEを同梱しています。
元版との同一性や上流commitは主張しません。今回追加の変更はstrict VSAエラー化です。
ComfyUI本体は別checkoutで取得し、そのGPLライセンスを保持します。モデル重みはGitに含めません。

## 参照

- [Colab FAQ: 制限、GPU、RAM、無料枠のUI](https://research.google.com/colaboratory/faq.html)
- [Kijai ComfyUI固定版](https://github.com/Kijai/ComfyUI/tree/10febb01d7be73d1491cf5e5347b5ab8b6c2c09e)
- [MiniMax H3モデルライセンス](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
