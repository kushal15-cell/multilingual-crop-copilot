from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def build_report(path: Path) -> dict:
    if not path.exists():
        return {"events": 0, "message": "No event file found"}
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    advice = [event for event in events if event.get("event_type") == "advice_completed"]
    decisions = [event for event in events if event.get("event_type") == "review_decided"]
    feedback = [event for event in events if event.get("event_type") == "farmer_feedback"]
    confidences = [float(event["confidence"]) for event in advice if "confidence" in event]
    latencies = [float(event["latency_ms"]) for event in advice if "latency_ms" in event]
    statuses = Counter(event.get("status", "unknown") for event in advice)
    labels = Counter(event.get("disease", "unknown") for event in advice)
    return {
        "events": len(events),
        "advice_requests": len(advice),
        "status_distribution": statuses,
        "disease_distribution": labels,
        "mean_confidence": float(np.mean(confidences)) if confidences else None,
        "confidence_p10": float(np.quantile(confidences, 0.1)) if confidences else None,
        "latency_p50_ms": float(np.quantile(latencies, 0.5)) if latencies else None,
        "latency_p95_ms": float(np.quantile(latencies, 0.95)) if latencies else None,
        "review_rate": (
            statuses.get("pending_review", 0) / len(advice) if advice else None
        ),
        "review_decisions": Counter(event.get("decision", "unknown") for event in decisions),
        "feedback_count": len(feedback),
        "helpful_rate": (
            sum(bool(event.get("helpful")) for event in feedback) / len(feedback)
            if feedback
            else None
        ),
        "abstention_rate": (
            statuses.get("need_more_information", 0) / len(advice) if advice else None
        ),
        "note": "Distribution changes are monitoring signals, not automatic proof of drift.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=Path("artifacts/events.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring_report.json"))
    args = parser.parse_args()
    report = build_report(args.events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=list), encoding="utf-8")
    print(json.dumps(report, indent=2, default=list))


if __name__ == "__main__":
    main()
