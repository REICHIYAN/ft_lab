# llm_ft_tinyllama

TinyLlama-1.1B-Chat-v1.0 をベースに、

- Full Fine-tuning（全層更新）
- LoRA
- QLoRA（4bit量子化 + LoRA）
- Prefix Tuning

を一通り試し、さらに

- vLLM による OpenAI 互換 API サーバ
- LangChain + LlamaIndex による RAG & モデル比較

までを一気通貫で動かすための最小構成リポジトリです。

## 1. セットアップ

```bash
git clone <your-repo-url>.git
cd llm_ft_tinyllama

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

GPU / CUDA 環境（例: Google Colab Pro の T4/A100）が推奨です。

---

## 2. データ

### 2.1 教師データ

`data/toy_qa.jsonl` にデモ用の Q&A データがあります。

フォーマット:

```jsonl
{"question": "...", "answer": "..."}
```

自分のタスクに合わせてここを書き換えれば、そのまま再学習できます。

### 2.2 RAG用ドキュメント

`docs/` 配下の `.txt` ファイルが RAG の対象になります。  
必要に応じて自由に追加・差し替えて下さい。

---

## 3. Fine-tuning の実行

TinyLlama-1.1B-Chat-v1.0 をベースに、それぞれの手法を実行します。

### 3.1 Full Fine-tuning（全層更新）

```bash
python train_full.py
```

出力:

- `models/ft_full/` にフル微調整済みモデルが保存されます。

### 3.2 LoRA

```bash
python train_lora.py
```

出力:

- `models/ft_lora/` に LoRA アダプタが保存されます（PEFT 形式）。

### 3.3 QLoRA（4bit量子化 + LoRA）

```bash
python train_qlora.py
```

出力:

- `models/ft_qlora/` に QLoRA アダプタが保存されます。

### 3.4 Prefix Tuning（Prefix / Adapter 系）

```bash
python train_prefix.py
```

出力:

- `models/ft_prefix/` に Prefix アダプタが保存されます。

---

## 4. vLLM でのサービング

ここでは簡略化のため、まずは **Full FT TinyLlama** のみを vLLM でサーブする例を示します。

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./models/ft_full \
  --served-model-name ft_full_tinyllama \
  --port 8001 \
  --host 0.0.0.0 \
  --api-key token-full \
  --gpu-memory-utilization 0.7
```

LoRA / QLoRA / Prefix については、

- vLLM の LoRA/Adapter 機能を使う
- あるいは「ベース + アダプタ」をマージした checkpoint を別名で保存し、それを `--model` に指定する

といったパターンが現実的です。

---

## 5. LangChain + LlamaIndex + vLLM で RAG & モデル比較

vLLM サーバを起動した状態で、別ターミナル（もしくは別セル）から:

```bash
python app_rag_compare.py \
  --docs_dir docs \
  --question "LangChain, LlamaIndex, vLLM と TinyLlama の役割の違いを説明してください。"
```

- `docs/` のテキストが LlamaIndex でインデックス化され、
- LangChain 経由で vLLM の OpenAI 互換 API を叩き、
- モデルごとの回答を比較表示します。

`app_rag_compare.py` 内の LLM クライアント設定は、  
起動した vLLM サーバの `--served-model-name` / ポート / API キーに合わせて調整して下さい。

---

## 6. 典型的な実行フロー（Colab Pro を想定）

1. 依存インストール
   ```bash
   pip install -r requirements.txt
   ```

2. 4種類の Fine-tuning
   ```bash
   python train_full.py
   python train_lora.py
   python train_qlora.py
   python train_prefix.py
   ```

3. vLLM で Full FT TinyLlama をサーブ
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model ./models/ft_full \
     --served-model-name ft_full_tinyllama \
     --port 8001 \
     --host 0.0.0.0 \
     --api-key token-full
   ```

4. LangChain + LlamaIndex + vLLM で RAG & 応答確認
   ```bash
   python app_rag_compare.py \
     --docs_dir docs \
     --question "TinyLlama を使った RAG の構成を説明してください。"
   ```

---

## 7. 今後の拡張

- ベースモデルを Llama 3, Qwen2, DeepSeek 等に差し替える
- 教師データを実ドメインの Q&A に差し替える
- LoRA / QLoRA / Prefix のハイパーパラメータを調整する
- vLLM の LoRA 対応機能で「ベース + 複数アダプタ」を動的切り替えする

といった方向で、そのまま研究・PoC・面接用デモに拡張できる構成になっています。
