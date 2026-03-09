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

from openai import OpenAI

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


def _default_case_files() -> List[Path]:
    cases_dir = ROOT / "eval" / "cases"
    return sorted(path for path in cases_dir.glob("*.json") if path.is_file())


def _sanitize_label(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized or "unknown"


def _normalize(texts: List[str]) -> set[str]:
    return {text.strip().lower() for text in texts if text and text.strip()}


def _jaccard_similarity(previous: List[str], current: List[str]) -> float:
    prev_set = _normalize(previous)
    curr_set = _normalize(current)
    if not prev_set and not curr_set:
        return 1.0
    union = prev_set | curr_set
    if not union:
        return 0.0
    return len(prev_set & curr_set) / len(union)


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


def _load_prompt_profile(profile_name: str) -> Dict[str, Any]:
    profile_dir = ROOT / "eval" / "prompt_variants" / profile_name
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Missing prompt profile directory: {profile_dir}")

    profile_meta_path = profile_dir / "profile.json"
    profile_meta: Dict[str, Any] = {}
    if profile_meta_path.is_file():
        profile_meta = json.loads(profile_meta_path.read_text(encoding="utf-8"))

    base_profile = None
    extends = profile_meta.get("extends")
    if extends:
        base_profile = _load_prompt_profile(extends)

    system_path = profile_dir / "system.txt"
    system_append_path = profile_dir / "system_append.txt"
    few_shots_path = profile_dir / "few_shots.json"
    few_shots_append_path = profile_dir / "few_shots_append.json"

    if system_path.is_file():
        system_prompt = system_path.read_text(encoding="utf-8").strip()
    elif base_profile:
        system_prompt = base_profile["system_prompt"]
    else:
        raise FileNotFoundError(f"Missing system prompt file: {system_path}")

    if system_append_path.is_file():
        appended = system_append_path.read_text(encoding="utf-8").strip()
        if appended:
            system_prompt = f"{system_prompt}\n\n{appended}"

    if few_shots_path.is_file():
        few_shots = json.loads(few_shots_path.read_text(encoding="utf-8"))
    elif base_profile:
        few_shots = deepcopy(base_profile["few_shots"])
    else:
        raise FileNotFoundError(f"Missing few-shot file: {few_shots_path}")

    if few_shots_append_path.is_file():
        few_shots.extend(json.loads(few_shots_append_path.read_text(encoding="utf-8")))

    file_refs = []
    if base_profile:
        file_refs.extend(base_profile["file_refs"])
    for path in [profile_meta_path, system_path, system_append_path, few_shots_path, few_shots_append_path]:
        if path.is_file():
            file_refs.append(str(path.relative_to(ROOT)))

    return {
        "name": profile_name,
        "system_prompt": system_prompt,
        "few_shots": few_shots,
        "file_refs": file_refs,
    }


def _build_messages(profile: Dict[str, Any], payload: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": profile["system_prompt"]}]
    for example in profile["few_shots"]:
        messages.append({"role": "user", "content": json.dumps(example["user"], ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(example["assistant"], ensure_ascii=False)})

    user_payload = {
        "task": "analyze_kiosk",
        "session_id": payload["session_id"],
        "user_input": payload.get("user_input") or "",
        "available_texts": payload.get("ocr_texts", []),
        "dialogue_history": payload.get("dialogue_history", []),
        "last_btn": payload.get("last_btn"),
    }
    messages.append({"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)})
    return messages


def _call_analyze(
    client: OpenAI,
    profile: Dict[str, Any],
    payload: Dict[str, Any],
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    messages = _build_messages(profile, payload)
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=messages,
    )
    content = completion.choices[0].message.content or "{}"
    parsed = json.loads(content)
    return parsed


def _run_step(
    client: OpenAI,
    profile: Dict[str, Any],
    endpoint: str,
    payload: Dict[str, Any],
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    if endpoint == "analyze":
        return _call_analyze(client, profile, payload, model, temperature)

    if endpoint == "screen_detect":
        previous = payload.get("previous_texts", [])
        current = payload.get("current_texts", [])
        similarity = _jaccard_similarity(previous, current)
        is_changed = similarity < float(os.getenv("SCREEN_CHANGE_THRESHOLD", "0.6"))

        analysis = None
        if is_changed:
            analyze_payload = {
                "session_id": payload["session_id"],
                "user_input": payload.get("user_input"),
                "ocr_texts": current,
                "dialogue_history": payload.get("dialogue_history", []),
                "last_btn": payload.get("last_btn"),
            }
            analysis = _call_analyze(client, profile, analyze_payload, model, temperature)

        return {
            "is_changed": is_changed,
            "similarity_score": similarity,
            "ai_analysis": analysis,
        }

    raise ValueError(f"Unsupported endpoint: {endpoint}")


def _analysis_from_response(endpoint: str, response: Dict[str, Any]) -> Dict[str, Any] | None:
    if endpoint == "analyze":
        return response
    if endpoint == "screen_detect":
        return response.get("ai_analysis")
    return None


def _run_scenario(
    client: OpenAI,
    profile: Dict[str, Any],
    suite_id: str,
    scenario: Dict[str, Any],
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    dialogue_history = deepcopy(scenario.get("initial_dialogue_history", []))
    last_btn = scenario.get("initial_last_btn")
    step_results: List[Dict[str, Any]] = []

    for step_index, step in enumerate(scenario.get("steps", []), start=1):
        endpoint, payload = _request_payload(scenario, step, dialogue_history, last_btn)
        severity = step.get("severity", "normal")
        error: str | None = None
        raw_response: Dict[str, Any]

        try:
            raw_response = _run_step(client, profile, endpoint, payload, model, temperature)
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

    return {
        "scenario_id": scenario["scenario_id"],
        "name": scenario.get("name", scenario["scenario_id"]),
        "session_id": scenario.get("session_id"),
        "summary": _summarize_step_results(step_results),
        "steps": step_results,
    }


def _run_suite(
    client: OpenAI,
    profile: Dict[str, Any],
    case_file: Path,
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    suite = json.loads(case_file.read_text(encoding="utf-8"))
    scenario_results = [
        _run_scenario(client, profile, suite["suite_id"], scenario, model, temperature)
        for scenario in suite.get("scenarios", [])
    ]
    all_steps = [step for scenario in scenario_results for step in scenario["steps"]]
    return {
        "suite_id": suite["suite_id"],
        "description": suite.get("description", ""),
        "case_file": str(case_file.relative_to(ROOT)),
        "summary": _summarize_step_results(all_steps),
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
    print(f"runner_type      : {report['runner_type']}")
    print(f"prompt_profile   : {report['prompt_profile']}")
    print(f"model_label      : {report['model_label']}")
    print(f"openai_model     : {report['env'].get('OPENAI_MODEL')}")
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
    parser = argparse.ArgumentParser(description="Run KiMate-AI eval cases with a prompt profile directly against OpenAI.")
    parser.add_argument("--prompt-profile", required=True, help="Directory name under eval/prompt_variants/")
    parser.add_argument(
        "--case-file",
        action="append",
        dest="case_files",
        help="Path to a suite JSON file. If omitted, all files under eval/cases are used.",
    )
    parser.add_argument("--model", help="Model name. Defaults to OPENAI_MODEL or gpt-5.1.")
    parser.add_argument("--model-label", help="Custom label stored in the report.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature.")
    parser.add_argument("--output", help="Output report path.")
    args = parser.parse_args()

    _load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for eval.prompt_runner")

    case_files = [Path(path).resolve() for path in args.case_files] if args.case_files else _default_case_files()
    if not case_files:
        raise SystemExit("No case files found.")

    profile = _load_prompt_profile(args.prompt_profile)
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-5.1")
    model_label = args.model_label or f"{profile['name']}__{model}"
    client = OpenAI(api_key=api_key)

    suites = [_run_suite(client, profile, case_file, model, args.temperature) for case_file in case_files]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if args.output else ROOT / "eval" / "results" / (
        f"{timestamp}__{_sanitize_label(model_label)}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner_type": "prompt_runner",
        "prompt_profile": profile["name"],
        "prompt_files": profile["file_refs"],
        "model_label": model_label,
        "env": {
            "OPENAI_MODEL": model,
            "AI_SERVER_MOCK": "0",
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
