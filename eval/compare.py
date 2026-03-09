from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _index_steps(report: Dict[str, Any]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    indexed: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for suite in report.get("suites", []):
        for scenario in suite.get("scenarios", []):
            for step in scenario.get("steps", []):
                key = (suite["suite_id"], scenario["scenario_id"], step["step_id"])
                indexed[key] = step
    return indexed


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two KiMate-AI evaluation reports.")
    parser.add_argument("baseline", help="Earlier or trusted report JSON")
    parser.add_argument("candidate", help="New report JSON to compare")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    baseline = _load_report(baseline_path)
    candidate = _load_report(candidate_path)

    baseline_steps = _index_steps(baseline)
    candidate_steps = _index_steps(candidate)
    common_keys = sorted(set(baseline_steps) & set(candidate_steps))

    improved = []
    regressed = []
    score_changed = []

    for key in common_keys:
        before = baseline_steps[key]
        after = candidate_steps[key]
        if before["pass"] and not after["pass"]:
            regressed.append((key, before, after))
        elif not before["pass"] and after["pass"]:
            improved.append((key, before, after))
        elif before["score"] != after["score"]:
            score_changed.append((key, before, after))

    print(f"baseline : {baseline_path}")
    print(f"candidate: {candidate_path}")
    print(f"common steps: {len(common_keys)}")
    print(f"improved pass/fail: {len(improved)}")
    print(f"regressed pass/fail: {len(regressed)}")
    print(f"score changed only : {len(score_changed)}")

    if regressed:
        print()
        print("Regressions:")
        for key, before, after in regressed:
            suite_id, scenario_id, step_id = key
            print(f"- {suite_id} / {scenario_id} / {step_id}")
            for failed in after.get("failed_checks", []):
                print(f"  expected={failed['expected']} actual={failed['actual']} ({failed['name']})")

    if improved:
        print()
        print("Improvements:")
        for key, _, _ in improved:
            suite_id, scenario_id, step_id = key
            print(f"- {suite_id} / {scenario_id} / {step_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
