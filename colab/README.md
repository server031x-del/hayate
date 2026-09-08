# HAYATE Studio on Colab

[Colabで開く](https://colab.research.google.com/github/server031x-del/hayate/blob/codex/colab-fast-h3/colab/HAYATE.ipynb)

1. GitHub取得セルを実行します。手動ZIPアップロードやGitHub認証は不要です。
2. 必要ならモデルのライセンスを確認し、取得フラグを有効にします（約30 GiB）。
3. WebUI起動セルを実行します。更新時もモデル・設定・出力を保持します。
4. 公開URLセルを実行し、HAYATE Studioを開きます。
5. 終了セルは作業終了時だけ実行してください。STOP_SERVICESを有効にすると停止します。

無料T4/CPUではUI確認まで。生成には対応GPU、依存ライブラリ、十分なVRAM/RAM、モデルが必要です。有料契約でもGPU互換性は保証されません。品質・時間・費用は実測して比較してください。

VM終了でローカルデータは失われます。必要な出力は保存してください。公開URLを知る人はWebUIにアクセスできます。

旧ComfyUI実験コードはruntime.pyに残していますが、このノートブックでは実行しません。
