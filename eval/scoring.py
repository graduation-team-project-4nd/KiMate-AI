from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Sequence


def _actual_action(analysis: Dict[str, Any] | None) -> Dict[str, Any]:
    if not analysis:
        return {}
    return analysis.get("action") or {}


def _actual_params(analysis: Dict[str, Any] | None) -> Dict[str, Any]:
    action = _actual_action(analysis)
    return action.get("params") or {}


def _analysis_present(expected: Dict[str, Any]) -> bool:
    analysis_keys = {
        "status",
        "action_type",
        "target_text",
        "candidates_exact_unordered",
        "candidates_must_include",
        "message_should_contain",
        "message_should_contain_any",
        "message_should_not_contain",
        "confidence_min",
        "confidence_max",
        "response_message",
    }
    return any(key in expected for key in analysis_keys)


def _check(
    name: str,
    passed: bool,
    expected: Any = None,
    actual: Any = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "expected": expected,
        "actual": actual,
    }


def _match_scalar(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


def _evaluate_single_expectation(
    expected: Dict[str, Any],
    response: Dict[str, Any],
    analysis: Dict[str, Any] | None,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    if "screen_is_changed" in expected:
        checks.append(
            _check(
                "screen_is_changed",
                response.get("is_changed") == expected["screen_is_changed"],
                expected=expected["screen_is_changed"],
                actual=response.get("is_changed"),
            )
        )

    if "similarity_min" in expected:
        actual = response.get("similarity_score")
        checks.append(
            _check(
                "similarity_min",
                actual is not None and actual >= expected["similarity_min"],
                expected=expected["similarity_min"],
                actual=actual,
            )
        )

    if "similarity_max" in expected:
        actual = response.get("similarity_score")
        checks.append(
            _check(
                "similarity_max",
                actual is not None and actual <= expected["similarity_max"],
                expected=expected["similarity_max"],
                actual=actual,
            )
        )

    if _analysis_present(expected):
        checks.append(_check("analysis_present", analysis is not None, expected=True, actual=analysis is not None))
        if analysis is None:
            return {
                "pass": False,
                "score": 0,
                "checks": checks,
                "failed_checks": [check for check in checks if not check["passed"]],
            }

    if "status" in expected:
        actual = analysis.get("status") if analysis else None
        checks.append(_check("status", _match_scalar(actual, expected["status"]), expected["status"], actual))

    if "action_type" in expected:
        actual = _actual_action(analysis).get("type")
        checks.append(
            _check(
                "action_type",
                _match_scalar(actual, expected["action_type"]),
                expected["action_type"],
                actual,
            )
        )

    if "target_text" in expected:
        actual = _actual_params(analysis).get("target_text")
        checks.append(
            _check(
                "target_text",
                _match_scalar(actual, expected["target_text"]),
                expected["target_text"],
                actual,
            )
        )

    if "candidates_exact_unordered" in expected:
        actual = _actual_params(analysis).get("candidates") or []
        expected_candidates = expected["candidates_exact_unordered"]
        checks.append(
            _check(
                "candidates_exact_unordered",
                sorted(actual) == sorted(expected_candidates),
                expected_candidates,
                actual,
            )
        )

    if "candidates_must_include" in expected:
        actual = _actual_params(analysis).get("candidates") or []
        required = expected["candidates_must_include"]
        checks.append(
            _check(
                "candidates_must_include",
                all(candidate in actual for candidate in required),
                required,
                actual,
            )
        )

    if "response_message" in expected:
        actual = analysis.get("response_message") if analysis else None
        checks.append(
            _check(
                "response_message",
                actual == expected["response_message"],
                expected["response_message"],
                actual,
            )
        )

    if "message_should_contain" in expected:
        actual = analysis.get("response_message", "") if analysis else ""
        required_terms: Sequence[str] = expected["message_should_contain"]
        checks.append(
            _check(
                "message_should_contain",
                all(term in actual for term in required_terms),
                required_terms,
                actual,
            )
        )

    if "message_should_contain_any" in expected:
        actual = analysis.get("response_message", "") if analysis else ""
        candidate_terms: Sequence[str] = expected["message_should_contain_any"]
        checks.append(
            _check(
                "message_should_contain_any",
                any(term in actual for term in candidate_terms),
                candidate_terms,
                actual,
            )
        )

    if "message_should_not_contain" in expected:
        actual = analysis.get("response_message", "") if analysis else ""
        forbidden_terms: Sequence[str] = expected["message_should_not_contain"]
        checks.append(
            _check(
                "message_should_not_contain",
                all(term not in actual for term in forbidden_terms),
                forbidden_terms,
                actual,
            )
        )

    if "confidence_min" in expected:
        actual = analysis.get("confidence") if analysis else None
        checks.append(
            _check(
                "confidence_min",
                actual is not None and actual >= expected["confidence_min"],
                expected["confidence_min"],
                actual,
            )
        )

    if "confidence_max" in expected:
        actual = analysis.get("confidence") if analysis else None
        checks.append(
            _check(
                "confidence_max",
                actual is not None and actual <= expected["confidence_max"],
                expected["confidence_max"],
                actual,
            )
        )

    total = len(checks)
    passed_count = sum(1 for check in checks if check["passed"])
    score = round((passed_count / total) * 100) if total else 100
    failed_checks = [check for check in checks if not check["passed"]]
    return {
        "pass": not failed_checks,
        "score": score,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def score_response(
    expected: Dict[str, Any],
    response: Dict[str, Any],
    analysis: Dict[str, Any] | None,
) -> Dict[str, Any]:
    alternatives = expected.get("alternatives")
    if alternatives:
        candidate_results: List[Dict[str, Any]] = []
        for index, alternative in enumerate(alternatives, start=1):
            result = _evaluate_single_expectation(alternative, response, analysis)
            result["alternative_index"] = index
            result["matched_expectation"] = deepcopy(alternative)
            candidate_results.append(result)

        candidate_results.sort(
            key=lambda item: (
                item["pass"],
                item["score"],
                -len(item["failed_checks"]),
            ),
            reverse=True,
        )
        best = candidate_results[0]
        best["alternatives_evaluated"] = len(candidate_results)
        return best

    result = _evaluate_single_expectation(expected, response, analysis)
    result["matched_expectation"] = deepcopy(expected)
    return result
