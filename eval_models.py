#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eval_models.py

Evaluate three models (ft_full, ft_lora, ft_qlora) on:
1. LLM-as-a-Judge score (requires OpenAI-compatible LLM; optional)
2. BERTScore-F1 (requires bert-score; optional)
3. Task accuracy (exact / relaxed match; always available if reference exists)
4. RAGAS faithfulness (requires ragas + LLM config; optional)
5. Hallucination rate (requires judge LLM; optional)

Input: JSONL file, each line:
{
  "id": "q1",
  "question": "...",
  "contexts": [
    {"doc_id": "doc1", "text": "..."},
    {"doc_id": "doc2", "text": "..."}
  ],
  "answers": {
    "ft_full": "...",
    "ft_lora": "...",
    "ft_qlora": "..."
  },
  "reference_answer": "...",        # optional
  "gold_doc_ids": ["doc1", "..."]   # optional (not used here directly)
}
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

# Optional dependencies
try:
    from bert_score import score as bert_score
    _HAS_BERTSCORE = True
except Exception:
    _HAS_BERTSCORE = False

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness as ragas_faithfulness
    from datasets import Dataset as HFDataset
    _HAS_RAGAS = True
except Exception:
    _HAS_RAGAS = False

try:
    # OpenAI Python SDK v1.x
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


MODEL_KEYS = ["ft_full", "ft_lora", "ft_qlora"]


@dataclass
class Example:
    """One evaluation example loaded from JSONL."""
    qid: str
    question: str
    contexts: List[Dict[str, str]]
    answers: Dict[str, str]
    reference_answer: Optional[str]
    gold_doc_ids: List[str]


@dataclass
class ModelMetrics:
    """Metrics aggregated per model."""
    judge_scores: List[float]
    hallucination_flags: List[bool]
    bert_f1_scores: List[float]
    exact_matches: int
    relaxed_matches: int
    total_with_reference: int

    def __init__(self) -> None:
        self.judge_scores = []
        self.hallucination_flags = []
        self.bert_f1_scores = []
        self.exact_matches = 0
        self.relaxed_matches = 0
        self.total_with_reference = 0

    def to_summary(self) -> Dict[str, Any]:
        n_judge = len(self.judge_scores)
        n_bert = len(self.bert_f1_scores)
        n_hall = len(self.hallucination_flags)
        return {
            "judge_score_mean": (sum(self.judge_scores) / n_judge) if n_judge > 0 else None,
            "judge_score_count": n_judge,
            "hallucination_rate": (sum(self.hallucination_flags) / n_hall) if n_hall > 0 else None,
            "hallucination_count": n_hall,
            "bertscore_f1_mean": (sum(self.bert_f1_scores) / n_bert) if n_bert > 0 else None,
            "bertscore_f1_count": n_bert,
            "exact_accuracy": (
                self.exact_matches / self.total_with_reference
                if self.total_with_reference > 0
                else None
            ),
            "relaxed_accuracy": (
                self.relaxed_matches / self.total_with_reference
                if self.total_with_reference > 0
                else None
            ),
            "n_with_reference": self.total_with_reference,
        }


def load_examples(path: str) -> List[Example]:
    """Load evaluation examples from a JSONL file."""
    examples: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("id", "")
            question = obj["question"]
            contexts = obj.get("contexts", [])
            answers = obj["answers"]
            reference = obj.get("reference_answer")
            gold_doc_ids = obj.get("gold_doc_ids", [])
            examples.append(
                Example(
                    qid=qid,
                    question=question,
                    contexts=contexts,
                    answers=answers,
                    reference_answer=reference,
                    gold_doc_ids=gold_doc_ids,
                )
            )
    return examples


# ---------------- LLM-as-a-Judge utilities -----------------


def call_judge_llm_openai(
    question: str,
    context: str,
    answers: Dict[str, str],
    model: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Call OpenAI ChatCompletion as a judge.

    Returns a dict:
    {
      "scores": {"ft_full": 8.5, "ft_lora": 8.0, "ft_qlora": 7.2},
      "hallucinations": {"ft_full": false, "ft_lora": true, ...}
    }

    The prompt is simple and deterministic-oriented.
    """
    prompt = (
        "You are an impartial evaluator.\n"
        "Given a question, supporting context, and three model answers, "
        "rate each answer from 0 to 10 and mark whether it contains hallucinations "
        "(content not supported by the context or obviously false).\n\n"
        "Return a strict JSON object with keys 'scores' and 'hallucinations'.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
    )

    for key in MODEL_KEYS:
        ans = answers.get(key, "")
        prompt += f"Answer ({key}):\n{ans}\n\n"

    prompt += (
        "Now respond ONLY with JSON, no extra text. Example:\n"
        '{\"scores\": {\"ft_full\": 8.0, \"ft_lora\": 7.5, \"ft_qlora\": 7.0}, '
        "\"hallucinations\": {\"ft_full\": false, \"ft_lora\": true, \"ft_qlora\": false}}\n"
    )

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    content = resp["choices"][0]["message"]["content"]
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError:
        # Fallback: if parsing fails, return empty; caller should skip
        return {"scores": {}, "hallucinations": {}}


# ---------------- RAGAS utilities -----------------


def compute_ragas_faithfulness(
    examples: List[Example],
    per_model_answers: Dict[str, List[str]],
) -> Dict[str, Optional[float]]:
    """
    Compute RAGAS faithfulness per model.
    Requires ragas + datasets and LLM / embeddings to be configured in ragas.
    """
    if not _HAS_RAGAS:
        return {k: None for k in MODEL_KEYS}

    results: Dict[str, Optional[float]] = {}
    for model_key in MODEL_KEYS:
        # Build HF Dataset for this model
        questions: List[str] = []
        answers: List[str] = []
        contexts: List[List[str]] = []
        ground_truths: List[str] = []

        for ex, ans in zip(examples, per_model_answers[model_key]):
            if not ex.contexts:
                continue
            questions.append(ex.question)
            answers.append(ans)
            contexts.append([c["text"] for c in ex.contexts])
            # ground_truth: if reference is present, use that; else use answer itself as weak label
            gt = ex.reference_answer if ex.reference_answer is not None else ans
            ground_truths.append(gt)

        if not questions:
            results[model_key] = None
            continue

        dataset = HFDataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        ragas_res = ragas_evaluate(dataset, metrics=[ragas_faithfulness])
        results[model_key] = float(ragas_res["faithfulness"])
    return results


# ---------------- BERTScore utilities -----------------


def compute_bertscore(
    refs: List[str],
    cands: List[str],
    lang: str = "en",
) -> List[float]:
    if not _HAS_BERTSCORE:
        return []
    P, R, F1 = bert_score(cands, refs, lang=lang)
    return [float(f) for f in F1]


# ---------------- Accuracy utilities -----------------


def normalize_text(s: str) -> str:
    """Simple normalization: strip, lowercase."""
    return " ".join(s.strip().lower().split())


# ---------------- Main evaluation -----------------


def evaluate_models(
    examples: List[Example],
    judge_model_name: Optional[str] = None,
    bert_lang: str = "en",
    use_ragas: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Main evaluation routine.

    Returns:
      {model_key: {metric_name: value}}
    """
    metrics: Dict[str, ModelMetrics] = {k: ModelMetrics() for k in MODEL_KEYS}

    # Collect answers per model for RAGAS / BERTScore
    answers_per_model: Dict[str, List[str]] = {k: [] for k in MODEL_KEYS}
    refs_for_bertscore: Dict[str, List[str]] = {k: [] for k in MODEL_KEYS}

    # --- 1) Task accuracy + BERTScore (reference-based) ---
    for ex in examples:
        has_ref = ex.reference_answer is not None
        for key in MODEL_KEYS:
            ans = ex.answers.get(key, "")
            answers_per_model[key].append(ans)
            if has_ref:
                metrics[key].total_with_reference += 1
                ref = ex.reference_answer or ""
                # exact
                if normalize_text(ans) == normalize_text(ref):
                    metrics[key].exact_matches += 1
                # relaxed: ref が ans に含まれる or ans が ref に含まれる
                if normalize_text(ref) in normalize_text(ans) or normalize_text(ans) in normalize_text(ref):
                    metrics[key].relaxed_matches += 1
                refs_for_bertscore[key].append(ref)

    # BERTScore
    if _HAS_BERTSCORE:
        for key in MODEL_KEYS:
            refs = refs_for_bertscore[key]
            cands = answers_per_model[key][: len(refs)]
            if refs and cands:
                f1_list = compute_bertscore(refs, cands, lang=bert_lang)
                metrics[key].bert_f1_scores.extend(f1_list)

    # --- 2) LLM-as-a-Judge + hallucination rate ---
    if judge_model_name is not None and _HAS_OPENAI:
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai.api_key:
            print("[WARN] OPENAI_API_KEY is not set. Skipping LLM-as-a-Judge.")
        else:
            for ex in examples:
                context_text = "\n\n".join(c["text"] for c in ex.contexts)
                judge_res = call_judge_llm_openai(
                    question=ex.question,
                    context=context_text,
                    answers=ex.answers,
                    model=judge_model_name,
                )
                scores = judge_res.get("scores", {})
                hallucinations = judge_res.get("hallucinations", {})
                for key in MODEL_KEYS:
                    if key in scores:
                        metrics[key].judge_scores.append(float(scores[key]))
                    if key in hallucinations:
                        metrics[key].hallucination_flags.append(bool(hallucinations[key]))

    # --- 3) RAGAS faithfulness ---
    ragas_scores: Dict[str, Optional[float]] = {}
    if use_ragas:
        ragas_scores = compute_ragas_faithfulness(examples, answers_per_model)

    # --- aggregate ---
    summary: Dict[str, Dict[str, Any]] = {}
    for key in MODEL_KEYS:
        base = metrics[key].to_summary()
        if use_ragas:
            base["ragas_faithfulness"] = ragas_scores.get(key)
        summary[key] = base
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ft_full / ft_lora / ft_qlora on multiple metrics."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to JSONL file with evaluation examples.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="OpenAI chat model name for LLM-as-a-Judge (e.g., gpt-4o). "
             "If not set or openai not available, judge metrics are skipped.",
    )
    parser.add_argument(
        "--bert_lang",
        type=str,
        default="en",
        help="Language code for BERTScore (e.g., 'en', 'ja').",
    )
    parser.add_argument(
        "--use_ragas",
        action="store_true",
        help="If set, compute RAGAS faithfulness (requires ragas).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="model_metrics.json",
        help="Where to save aggregated metrics as JSON.",
    )
    args = parser.parse_args()

    exs = load_examples(args.data_path)
    print(f"Loaded {len(exs)} examples from {args.data_path}")

    summary = evaluate_models(
        exs,
        judge_model_name=args.judge_model,
        bert_lang=args.bert_lang,
        use_ragas=args.use_ragas,
    )

    print("\n=== Model Metrics Summary ===")
    for key, metrics in summary.items():
        print(f"\n[{key}]")
        for mname, val in metrics.items():
            print(f"  {mname}: {val}")

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics to {args.output_path}")


if __name__ == "__main__":
    main()
