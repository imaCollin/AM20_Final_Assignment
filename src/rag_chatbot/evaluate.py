from __future__ import annotations

import json
from datetime import datetime, timezone

from src.rag_chatbot.chatbot import generate_answer
from src.rag_chatbot.config import settings


REFUSAL_MARKERS = [
    "cannot find",
    "not in the knowledge base",
    "insufficient",
    "don't have",
]


def load_golden_dataset() -> list[dict]:
    return json.loads(settings.golden_dataset_path.read_text(encoding="utf-8"))


def evaluate_case(case: dict) -> dict:
    result = generate_answer(case["question"])
    answer_lower = result["answer"].lower()

    if case["should_answer"]:
        expected_phrases = case.get("expected_answer_contains", [])
        answer_supported = all(phrase.lower() in answer_lower for phrase in expected_phrases)
    else:
        answer_supported = any(marker in answer_lower for marker in REFUSAL_MARKERS)

    actual_sources = sorted({chunk.source for chunk in result["citations"]})
    expected_sources = sorted(case.get("expected_sources", []))
    source_match = set(expected_sources).issubset(set(actual_sources))

    return {
        "id": case["id"],
        "question": case["question"],
        "should_answer": case["should_answer"],
        "passed_answer_check": answer_supported,
        "passed_source_check": source_match,
        "expected_sources": expected_sources,
        "actual_sources": actual_sources,
        "model_answer": result["answer"],
    }


def run_evaluation() -> dict:
    settings.evaluation_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_golden_dataset()
    cases = [evaluate_case(case) for case in dataset]
    answer_passes = sum(case["passed_answer_check"] for case in cases)
    source_passes = sum(case["passed_source_check"] for case in cases)
    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(cases),
        "answer_pass_rate": answer_passes / len(cases) if cases else 0.0,
        "source_pass_rate": source_passes / len(cases) if cases else 0.0,
        "cases": cases,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = settings.evaluation_dir / f"evaluation_{timestamp}.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": summary, "output_path": str(output_path)}


if __name__ == "__main__":
    result = run_evaluation()
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved evaluation to {result['output_path']}")
