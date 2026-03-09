from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi.testclient import TestClient

from .scoring import score_response

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


def build_client() -> TestClient:
    _load_dotenv()
    from src.main import app

    return TestClient(app)


def _default_case_files() -> List[Path]:
    cases_dir = ROOT / "eval" / "cases"
    return sorted(path for path in cases_dir.glob("*.json") if path.is_file())


def _sanitize_label(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized or "unknown"


def _analysis_from_response(endpoint: str, response: Dict[str, Any]) -> Dict[str, Any] | None:
    if endpoint == "analyze":
        return response
    if endpoint == "screen_detect":
        return response.get("ai_analysis")
    return None


def _request_payload(
    scenario: Dict[str, Any],
    step: Dict[str, Any],
    dialogue_history: List[Dict[str, Any]],
    last_btn: str | None,
) -> Tuple[str, Dict[str, Any]]:
    endpoint = step.get("endpoint") or scenario.get("endpoint", "analyze")
    data = deepcopy(step.get("input", {}))
    session_id = data.get("session_id") or scenario.get("session_id") or scenario["scenario_id"]

    if endpoint == "analyze":
        payload = {
            "session_id": session_id,
            "user_input": data.get("user_input"),
            "ocr_texts": data.get("ocr_texts", []),
            "dialogue_history": deepcopy(dialogue_history),
            "last_btn": data.get("last_btn", last_btn),
        }
        return endpoint, payload

    if endpoint == "screen_detect":
        payload = {
            "session_id": session_id,
            "previous_texts": data.get("previous_texts", []),
            "current_texts": data.get("current_texts", []),
            "dialogue_history": deepcopy(dialogue_history),
            "user_input": data.get("user_input"),
            "last_btn": data.get("last_btn", last_btn),
        }
        return endpoint, payload

    raise ValueError(f"Unsupported endpoint: {endpoint}")


def _post(client: TestClient, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    path = "/api/analyze" if endpoint == "analyze" else "/api/screen/detect"
    response = client.post(path, json=payload)
    response.raise_for_status()
    return response.json()


def _append_history(
    dialogue_history: List[Dict[str, Any]],
    request_payload: Dict[str, Any],
    analysis: Dict[str, Any] | None,
) -> None:
    user_input = request_payload.get("user_input")
    if user_input:
        dialogue_history.append({"role": "user", "utterance": user_input})

    if analysis:
        dialogue_history.append(
            {
                "role": "assistant",
                "utterance": analysis.get("response_message", ""),
                "action": analysis.get("action"),
            }
        )


def _summarize_step_results(step_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(step_results)
    passed = sum(1 for step in step_results if step["pass"])
    failed = total - passed
    critical_failures = sum(1 for step in step_results if step["severity"] == "critical" and not step["pass"])
    average_score = round(sum(step["score"] for step in step_results) / total, 1) if total else 0.0
    return {
        "total_steps": total,
        "passed_steps": passed,
        "failed_steps": failed,
        "critical_failures": critical_failures,
        "average_score": average_score,
    }


def _run_scenario(client: TestClient, suite_id: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
    dialogue_history = deepcopy(scenario.get("initial_dialogue_history", []))
    last_btn = scenario.get("initial_last_btn")
    step_results: List[Dict[str, Any]] = []

    for step_index, step in enumerate(scenario.get("steps", []), start=1):
        endpoint, payload = _request_payload(scenario, step, dialogue_history, last_btn)
        severity = step.get("severity", "normal")
        raw_response: Dict[str, Any]
        error: str | None = None

        try:
            raw_response = _post(client, endpoint, payload)
        except Exception as exc:
            raw_response = {"_error": str(exc)}
            error = str(exc)

        analysis = _analysis_from_response(endpoint, raw_response) if not error else None
        scoring = score_response(step.get("expected", {}), raw_response, analysis)

        if not error:
            _append_history(dialogue_history, payload, analysis)
            if analysis:
                action = analysis.get("action") or {}
                params = action.get("params") or {}
                if action.get("type") == "click_text":
                    last_btn = params.get("target_text", last_btn)

        step_results.append(
            {
                "suite_id": suite_id,
                "scenario_id": scenario["scenario_id"],
                "step_index": step_index,
                "step_id": step["step_id"],
                "name": step.get("name", step["step_id"]),
                "severity": severity,
                "endpoint": endpoint,
                "pass": scoring["pass"],
                "score": scoring["score"],
                "error": error,
                "request": payload,
                "response": raw_response,
                "failed_checks": scoring["failed_checks"],
                "checks": scoring["checks"],
                "matched_expectation": scoring.get("matched_expectation"),
                "matched_alternative": scoring.get("alternative_index"),
                "rationale": step.get("rationale", ""),
            }
        )

    summary = _summarize_step_results(step_results)
    return {
        "scenario_id": scenario["scenario_id"],
        "name": scenario.get("name", scenario["scenario_id"]),
        "session_id": scenario.get("session_id"),
        "summary": summary,
        "steps": step_results,
    }


def _run_suite(client: TestClient, case_file: Path) -> Dict[str, Any]:
    suite = json.loads(case_file.read_text(encoding="utf-8"))
    scenario_results = [
        _run_scenario(client, suite["suite_id"], scenario) for scenario in suite.get("scenarios", [])
    ]

    all_steps = [step for scenario in scenario_results for step in scenario["steps"]]
    summary = _summarize_step_results(all_steps)
    return {
        "suite_id": suite["suite_id"],
        "description": suite.get("description", ""),
        "case_file": str(case_file.relative_to(ROOT)),
        "summary": summary,
        "scenarios": scenario_results,
    }


def _combined_summary(suites: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_steps = []
    for suite in suites:
        for scenario in suite["scenarios"]:
            all_steps.extend(scenario["steps"])
    return _summarize_step_results(all_steps)


def _print_summary(report: Dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"model_label      : {report['model_label']}")
    print(f"openai_model     : {report['env'].get('OPENAI_MODEL')}")
    print(f"ai_server_mock   : {report['env'].get('AI_SERVER_MOCK')}")
    print(f"suites           : {len(report['suites'])}")
    print(f"total_steps      : {summary['total_steps']}")
    print(f"passed_steps     : {summary['passed_steps']}")
    print(f"failed_steps     : {summary['failed_steps']}")
    print(f"critical_failures: {summary['critical_failures']}")
    print(f"average_score    : {summary['average_score']}")

    for suite in report["suites"]:
        suite_summary = suite["summary"]
        print()
        print(f"[{suite['suite_id']}] {suite_summary['passed_steps']}/{suite_summary['total_steps']} passed")
        for scenario in suite["scenarios"]:
            scenario_summary = scenario["summary"]
            print(
                f"  - {scenario['scenario_id']}: "
                f"{scenario_summary['passed_steps']}/{scenario_summary['total_steps']} passed"
            )
            for step in scenario["steps"]:
                if step["pass"]:
                    continue
                print(f"    * FAIL {step['step_id']} ({step['severity']})")
                for failed in step["failed_checks"]:
                    print(
                        f"      - {failed['name']}: "
                        f"expected={failed['expected']} actual={failed['actual']}"
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run KiMate-AI evaluation suites.")
    parser.add_argument(
        "--case-file",
        action="append",
        dest="case_files",
        help="Path to a suite JSON file. If omitted, all files under eval/cases are used.",
    )
    parser.add_argument(
        "--model-label",
        help="Custom label stored in the report. Defaults to OPENAI_MODEL or 'mock'.",
    )
    parser.add_argument(
        "--output",
        help="Output report path. Defaults to eval/results/<timestamp>__<model>.json",
    )
    args = parser.parse_args()

    case_files = [Path(path).resolve() for path in args.case_files] if args.case_files else _default_case_files()
    if not case_files:
        raise SystemExit("No case files found.")

    client = build_client()

    model_label = args.model_label or os.getenv("OPENAI_MODEL") or "mock"
    if os.getenv("AI_SERVER_MOCK", "0") == "1" and not args.model_label:
        model_label = "mock"

    suites = [_run_suite(client, case_file) for case_file in case_files]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if args.output else ROOT / "eval" / "results" / (
        f"{timestamp}__{_sanitize_label(model_label)}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_label": model_label,
        "env": {
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", ""),
            "AI_SERVER_MOCK": os.getenv("AI_SERVER_MOCK", "0"),
        },
        "summary": _combined_summary(suites),
        "suites": suites,
    }

    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(report)
    print()
    print(f"report saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
