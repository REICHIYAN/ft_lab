# README (English → Japanese alternating)

## Overview  
This repository provides a minimal end-to-end toolkit for experimenting with TinyLlama-1.1B across multiple fine-tuning paradigms and serving pipelines.  
本リポジトリは TinyLlama-1.1B を複数の微調整手法とサービング基盤で一気通貫に扱える、最小構成の実験用ツールキットです。

## 1. Setup  
Clone the repository and prepare a virtual environment.  
リポジトリをクローンし、仮想環境を準備してください。

```bash
git clone <your-repo-url>.git
cd llm_ft_tinyllama
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU resources such as Google Colab Pro (T4/A100) are recommended.  
GPU を備えた Colab Pro（T4/A100 など）での実行を推奨します。

## 2. Dataset  
The demo SFT corpus is located at `data/toy_qa.jsonl`.  
教師デモデータは `data/toy_qa.jsonl` に保存されています。

Format:  
フォーマットは以下の通りです。

```
{"question": "...", "answer": "..."}
```

## 3. Fine-Tuning Methods  
TinyLlama-1.1B-Chat-v1.0 is fine-tuned using four representative approaches.  
TinyLlama-1.1B-Chat-v1.0 を代表的な 4 手法で微調整できます。

### 3.1 Full Fine-Tuning  
Run:  
実行：

```
python train_full.py
```

### 3.2 LoRA  
Run:  
実行：

```
python train_lora.py
```

### 3.3 QLoRA  
Run:  
実行：

```
python train_qlora.py
```

### 3.4 Prefix Tuning  
Run:  
実行：

```
python train_prefix.py
```

## 4. Serving via vLLM  
Here we demonstrate serving the fully fine-tuned model.  
ここでは、フル微調整モデルを vLLM で提供する方法を示します。

```
python -m vllm.entrypoints.openai.api_server --model ./models/ft_full
```

## 5. RAG Comparison  
Run:  
実行：

```
python app_rag_compare.py --docs_dir docs --question "Explain LoRA."
```

