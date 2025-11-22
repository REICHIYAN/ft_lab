
# TinyLlama Fine-Tuning Toolkit (Updated — No vLLM)

---

## 🌐 Overview

This repository provides a **compact, end-to-end fine-tuning and evaluation toolkit**
for **TinyLlama-1.1B-Chat-v1.0**, enabling reproducible experiments across:

- Full Fine-Tuning (FT)
- LoRA
- QLoRA
- RAG evaluation (HuggingFace embeddings via LlamaIndex)
- Unified model comparison utilities

vLLM is **not used in this project** and all related descriptions have been removed.

Prefix Tuning is intentionally excluded to keep the stack minimal.

---

## 🏗️ Architecture & Tech Stack

```
┌───────────────────────────────────────────┐
│                Training Layer             │
│  • Full FT (HF Trainer)                   │
│  • LoRA (PEFT)                            │
│  • QLoRA (4bit + LoRA)                    │
└───────────────┬───────────────────────────┘
                │
┌───────────────▼───────────────────────────┐
│               Model Outputs               │
│  models/ft_full/                          │
│  models/ft_lora/                          │
│  models/ft_qlora/                         │
└───────────────┬───────────────────────────┘
                │
┌───────────────▼───────────────────────────┐
│            Evaluation Utilities           │
│  • app_rag_compare.py (RAG pipeline)      │
│  • compare_adapters.py                    │
└────────────────────────────────────────────┘
```

---

## 📚 Fine-Tuning Methods

### **1. Full Fine-Tuning**

```bash
python train_full.py
```

---

### **2. LoRA**

```bash
python train_lora.py
```

---

### **3. QLoRA**

```bash
python train_qlora.py
```

---

## 🔍 RAG Evaluation (Updated)

This project includes a simple RAG demo using **LlamaIndex** and **HuggingFace embeddings**.

We use:

- `llama-index-embeddings-huggingface`
- `sentence-transformers`

Example (inside `app_rag_compare.py`):

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Run:

```bash
python app_rag_compare.py --docs_dir docs --question "Explain LoRA."
```

---

## 🧭 Model Comparison

```bash
python compare_adapters.py
```

Compares:

- Full FT
- LoRA
- QLoRA

---

## 📁 Repository Structure

```
llm_ft_tinyllama/
├── train_full.py
├── train_lora.py
├── train_qlora.py
├── compare_adapters.py
├── app_rag_compare.py
├── requirements.txt
│
├── models/
│   ├── ft_full/
│   ├── ft_lora/
│   └── ft_qlora/
│
├── docs/
│
└── data/
    └── toy_qa.jsonl
```

---

## 🛠 Requirements

The following are required:

```
torch>=2.1.0
transformers>=4.39.0
accelerate>=0.27.0
sentencepiece>=0.1.99
einops>=0.7.0

datasets>=2.18.0
peft>=0.10.0
bitsandbytes>=0.42.0

langchain>=0.2.0
langchain-openai>=0.1.0
llama-index>=0.10.0

python-dotenv>=1.0.0

llama-index-embeddings-huggingface
sentence-transformers
```

Install:

```bash
pip install -r requirements.txt
```

---

## 🙌 Final Notes

This repository is a **clean, extensible baseline** for TinyLlama fine‑tuning and LlamaIndex‑based RAG experiments.
