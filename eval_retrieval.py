#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eval_retrieval.py

Evaluate retrieval quality (independent of model fine-tuning) on:
1. retrieval recall@k
2. context precision@k
3. RAGAS answer_relevance (optional)

Input JSONL schema is same as eval_models.py, but this script expects:
- 'gold_doc_ids' to be non-empty (for recall/precision)
- 'contexts' to have 'doc_id' for each retrieved chunk
- 'answers[model_key]' used as 'answer' column for RAGAS
"""

import argparse
import json
from typing import Dict, List, Any, Optional

# Optional ragas
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import answer_relevance as ragas_answer_relevance
    from datasets import Dataset as HFDataset
    _HAS_RAGAS = True
except Exception:
    _HAS_RAGAS = False


def load_items(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(obj)
    return items


def compute_recall_precision_at_k(
    items: List[Dict[str, Any]],
    k: int,
) -> Dict[str, Optional[float]]:
    """
    Compute retrieval recall@k and precision@k based on:
      - gold_doc_ids
      - contexts[*].doc_id
    """
    total = 0
    sum_recall = 0.0
    sum_precision = 0.0

    for obj in items:
        gold_doc_ids: List[str] = obj.get("gold_doc_ids", [])
        contexts: List[Dict[str, Any]] = obj.get("contexts", [])
        if not gold_doc_ids or not contexts:
            continue

        retrieved_ids = [c.get("doc_id") for c in contexts[:k] if "doc_id" in c]
        if not retrieved_ids:
            continue

        total += 1
        gold_set = set(gold_doc_ids)
        retrieved_set = set(retrieved_ids)

        true_positives = len(gold_set & retrieved_set)
        recall = true_positives / len(gold_set)
        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0

        sum_recall += recall
        sum_precision += precision

    if total == 0:
        return {"recall_at_k": None, "precision_at_k": None, "n": 0}

    return {
        "recall_at_k": sum_recall / total,
        "precision_at_k": sum_precision / total,
        "n": total,
    }


def compute_ragas_answer_relevance(
    items: List[Dict[str, Any]],
    model_key: str,
) -> Optional[float]:
    """
    Compute RAGAS answer_relevance for a given model_key using:
      - question
      - contexts[*].text
      - answers[model_key] as 'answer'
      - reference_answer as 'ground_truth' (if available)
    """
    if not _HAS_RAGAS:
        return None

    questions: List[str] = []
    answers: List[str] = []
    contexts_list: List[List[str]] = []
    ground_truths: List[str] = []

    for obj in items:
        q = obj.get("question")
        contexts = obj.get("contexts", [])
        answers_dict = obj.get("answers", {})
        ans = answers_dict.get(model_key)
        if q is None or ans is None or not contexts:
            continue

        questions.append(q)
        answers.append(ans)
        contexts_list.append([c.get("text", "") for c in contexts])

        ref = obj.get("reference_answer")
        if ref is None:
            # fallback: use answer itself as weak ground_truth
            ref = ans
        ground_truths.append(ref)

    if not questions:
        return None

    dataset = HFDataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )
    res = ragas_evaluate(dataset, metrics=[ragas_answer_relevance])
    return float(res["answer_relevance"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality (recall@k, precision@k, RAGAS answer_relevance)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to JSONL file with evaluation items.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Top-k for recall@k and precision@k (default: 3).",
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default="ft_full",
        help="Model key to use for answer_relevance (e.g., ft_full, ft_lora, ft_qlora).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="retrieval_metrics.json",
        help="Where to save aggregated retrieval metrics as JSON.",
    )
    args = parser.parse_args()

    items = load_items(args.data_path)
    print(f"Loaded {len(items)} items from {args.data_path}")

    rp = compute_recall_precision_at_k(items, k=args.k)
    print("\n=== Retrieval Metrics ===")
    print(f"  recall@{args.k}:   {rp['recall_at_k']}")
    print(f"  precision@{args.k}: {rp['precision_at_k']}")
    print(f"  n (used items):     {rp['n']}")

    ar = compute_ragas_answer_relevance(items, model_key=args.model_key)
    print(f"\nRAGAS answer_relevance (model={args.model_key}): {ar}")

    out = {
        "k": args.k,
        "recall_at_k": rp["recall_at_k"],
        "precision_at_k": rp["precision_at_k"],
        "n_items": rp["n"],
        "ragas_answer_relevance": ar,
        "model_key": args.model_key,
    }
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved retrieval metrics to {args.output_path}")


if __name__ == "__main__":
    main()
