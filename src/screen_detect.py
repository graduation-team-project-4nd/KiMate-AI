from __future__ import annotations

from typing import Iterable, List, Set

from .ai_service import AIService
from .models import AnalyzeRequest, ScreenDetectRequest, ScreenDetectResponse


def _normalize(texts: Iterable[str]) -> Set[str]:
    return {text.strip().lower() for text in texts if text and text.strip()}


def jaccard_similarity(previous: List[str], current: List[str]) -> float:
    prev_set = _normalize(previous)
    curr_set = _normalize(current)
    if not prev_set and not curr_set:
        return 1.0
    union = prev_set | curr_set
    if not union:
        return 0.0
    intersection = prev_set & curr_set
    return len(intersection) / len(union)


class ScreenDetector:
    def __init__(self, ai_service: AIService, threshold: float = 0.6):
        self.ai_service = ai_service
        self.threshold = threshold

    async def detect(self, payload: ScreenDetectRequest) -> ScreenDetectResponse:
        similarity = jaccard_similarity(payload.previous_texts, payload.current_texts)
        is_changed = similarity < self.threshold

        ai_analysis = None
        if is_changed:
            analyze_payload = AnalyzeRequest(
                session_id=payload.session_id,
                user_input=payload.user_input,
                ocr_texts=payload.current_texts,
                dialogue_history=payload.dialogue_history,
                last_btn=payload.last_btn,
            )
            ai_analysis = await self.ai_service.analyze(analyze_payload)

        return ScreenDetectResponse(
            is_changed=is_changed,
            similarity_score=similarity,
            ai_analysis=ai_analysis,
        )
