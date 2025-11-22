# app_rag_compare.py
\"\"\"LangChain + LlamaIndex + vLLM で RAG + モデル比較を行うデモ。

前提:
- vLLM の OpenAI互換サーバが、以下のように起動している想定:
    - Full FT : http://localhost:8001/v1  （モデルID: "ft_full_tinyllama"）

必要に応じて LoRA / QLoRA / Prefix 用のエンドポイントも追加してください。
\"\"\"import argparse
from typing import List

from langchain_openai import ChatOpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


def build_llamaindex_retriever(docs_dir: str, top_k: int = 3):
    reader = SimpleDirectoryReader(docs_dir)
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever


def build_llm_client() -> ChatOpenAI:
    """
    vLLM の OpenAI互換サーバを1つ想定したクライアント。
    --served-model-name ft_full_tinyllama
    --port 8001
    --api-key token-full
    で起動している前提です。
    """
    llm = ChatOpenAI(
        base_url="http://localhost:8001/v1",
        api_key="token-full",
        model="ft_full_tinyllama",
        temperature=0.2,
    )
    return llm


def build_rag_prompt(query: str, context_chunks: List[str]) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "You are a helpful assistant that answers based only on the given context.\n\n"
        f"Context:\n{context_text}\n\n"
        f"User question:\n{query}\n\n"
        "Answer in Japanese, be concise but clear."
    )
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs_dir",
        type=str,
        default="docs",
        help="Directory containing documents for RAG.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="User question to ask the model.",
    )
    args = parser.parse_args()

    retriever = build_llamaindex_retriever(args.docs_dir)
    nodes = retriever.retrieve(args.question)
    context_chunks = [n.get_content() for n in nodes]

    if not context_chunks:
        print("No context retrieved from docs/. Check documents.")
        return

    prompt = build_rag_prompt(args.question, context_chunks)
    llm = build_llm_client()

    print(f"=== Question ===\n{args.question}\n")
    print("=== Retrieved Context (top-3) ===")
    for i, chunk in enumerate(context_chunks, start=1):
        print(f"[{i}] {chunk[:200]}...\n")

    print("\n====== Model: ft_full_tinyllama ======")
    msg = [{"role": "user", "content": prompt}]
    resp = llm.invoke(msg)
    print(resp.content)


if __name__ == "__main__":
    main()
