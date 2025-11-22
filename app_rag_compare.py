#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG + Model Comparison demo for TinyLlama FT / LoRA / QLoRA.

- Uses LlamaIndex for retrieval (local HuggingFace embedding)
- Uses vLLM (OpenAI-compatible server) for generation

No OpenAI API key is required.
Prefix Tuning is completely excluded.
"""

import argparse
from typing import Dict, List

from langchain_openai import ChatOpenAI

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ---- vLLM model endpoints (edit here to match your environment) ----
MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "ft_full": {
        "model": "ft_full_tinyllama",
        "base_url": "http://localhost:8001/v1",
    },
    "ft_lora": {
        "model": "ft_lora_tinyllama",
        "base_url": "http://localhost:8002/v1",
    },
    "ft_qlora": {
        "model": "ft_qlora_tinyllama",
        "base_url": "http://localhost:8003/v1",
    },
}


# ---- RAG: Retriever (LlamaIndex + local embedding) ----
def build_llamaindex_retriever(docs_dir: str, top_k: int = 3):
    """Build a LlamaIndex retriever using a local HuggingFace embedding model."""

    # ✅ Use local HuggingFace embedding instead of OpenAI
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Set as global default to avoid OpenAI usage
    Settings.embed_model = embed_model

    docs = SimpleDirectoryReader(docs_dir).load_data()
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever


def retrieve_context_chunks(retriever, question: str) -> List[str]:
    """Retrieve top-k context chunks for a question."""
    nodes = retriever.retrieve(question)
    return [n.text for n in nodes]


# ---- vLLM-backed LLM (ChatOpenAI wrapper) ----
def build_llm(model_key: str, temperature: float = 0.2) -> ChatOpenAI:
    """
    Build a ChatOpenAI client that talks to a vLLM OpenAI-compatible server.

    We use:
      - base_url from MODEL_CONFIGS[model_key]["base_url"]
      - model    from MODEL_CONFIGS[model_key]["model"]
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model key: {model_key}. "
            f"Available keys: {list(MODEL_CONFIGS.keys())}"
        )

    cfg = MODEL_CONFIGS[model_key]

    llm = ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key="dummy-key",  # vLLM側で認証を見ていないなら何でもよい
        temperature=temperature,
    )
    return llm


# ---- Prompt construction ----
def build_messages(question: str, context_chunks: List[str]) -> List[dict]:
    """Build OpenAI-style chat messages for RAG prompt."""
    context_str = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks))

    system_msg = (
        "You are a helpful assistant.\n"
        "Use ONLY the given context to answer the user's question.\n"
        "If the answer is not contained in the context, say that you don't know.\n"
    )

    user_msg = (
        "Question:\n"
        f"{question}\n\n"
        "Context:\n"
        f"{context_str}\n\n"
        "Please answer in a concise way."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ---- CLI args ----
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG + model comparison demo (TinyLlama FT / LoRA / QLoRA)"
    )

    parser.add_argument(
        "--docs_dir",
        type=str,
        default="docs",
        help="Directory containing source documents for RAG.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask the RAG system.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of context chunks to retrieve.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="ft_full",
        help=(
            "Comma-separated list of model keys to query. "
            "Available: " + ",".join(MODEL_CONFIGS.keys())
        ),
    )

    return parser.parse_args()


# ---- main ----
def main():
    args = parse_args()

    # Parse model keys
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    for key in model_keys:
        if key not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model key: {key}. "
                f"Available keys: {list(MODEL_CONFIGS.keys())}"
            )

    print("=== RAG + Model Comparison ===")
    print(f"Docs dir : {args.docs_dir}")
    print(f"Question : {args.question}")
    print(f"Top-k    : {args.top_k}")
    print(f"Models   : {model_keys}")
    print()

    # 1. Build retriever & get context
    retriever = build_llamaindex_retriever(args.docs_dir, top_k=args.top_k)
    context_chunks = retrieve_context_chunks(retriever, args.question)

    print("=== Retrieved Context (top-k) ===")
    for i, chunk in enumerate(context_chunks, start=1):
        preview = chunk.replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"[{i}] {preview}")
    print()

    # 2. Query each model with the same RAG prompt
    for key in model_keys:
        llm = build_llm(key)
        messages = build_messages(args.question, context_chunks)

        print("=" * 70)
        print(f"Model: {key} (id={MODEL_CONFIGS[key]['model']}, base_url={MODEL_CONFIGS[key]['base_url']})")
        print("-" * 70)

        resp = llm.invoke(messages)
        print(resp.content)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
