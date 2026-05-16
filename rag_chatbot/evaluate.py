"""
evaluate.py  —  RAGAS evaluation for the RAG pipeline
-------------------------------------------------------
Usage:
    python evaluate.py --qa_path test_questions.json

test_questions.json format:
[
  {
    "question": "What is the main topic of chapter 2?",
    "ground_truth": "Chapter 2 discusses neural networks."
  },
  ...
]
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_recall,
    faithfulness,
)

from rag_chatbot.rag_chain import query as rag_query
from rag_chatbot.config import LLM_MODEL, TOP_K


def run_evaluation(qa_path: str | Path, model: str = LLM_MODEL, top_k: int = TOP_K):
    qa_path = Path(qa_path)
    if not qa_path.exists():
        print(f"[error] QA file not found: {qa_path}")
        return

    with open(qa_path) as f:
        qa_pairs = json.load(f)

    print(f"Running RAGAS evaluation on {len(qa_pairs)} questions …\n")

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, pair in enumerate(qa_pairs):
        q = pair["question"]
        gt = pair.get("ground_truth", "")
        print(f"  [{i+1}/{len(qa_pairs)}] {q[:60]}…")

        result = rag_query(q, model=model, top_k=top_k)
        questions.append(q)
        answers.append(result.answer)
        contexts.append([s["text"] for s in result.sources])
        ground_truths.append(gt)

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    print("\nComputing RAGAS metrics …")
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])

    print("\n── RAGAS Results ──────────────────────────────")
    df = scores.to_pandas()
    print(df[["faithfulness", "answer_relevancy", "context_recall"]].describe())

    out_path = Path("ragas_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")
    return df


# ── Sample QA generator ────────────────────────────────────────────────────

def create_sample_qa(output_path: str = "test_questions.json"):
    """Create a sample QA file template to fill in."""
    sample = [
        {
            "question": "What is the main topic of this document?",
            "ground_truth": "Fill in the expected answer here."
        },
        {
            "question": "What are the key findings or conclusions?",
            "ground_truth": "Fill in the expected answer here."
        },
    ]
    with open(output_path, "w") as f:
        json.dump(sample, f, indent=2)
    print(f"Sample QA file created: {output_path}")
    print("Edit it with real questions and ground truth answers, then run evaluate.py.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_path", default="test_questions.json")
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--create_sample", action="store_true")
    args = parser.parse_args()

    if args.create_sample:
        create_sample_qa()
    else:
        run_evaluation(args.qa_path, model=args.model, top_k=args.top_k)
