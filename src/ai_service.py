from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Sequence

from openai import AsyncOpenAI
from openai import OpenAIError

from .models import AnalyzeRequest, AnalyzeResponse, Action

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
당신은 시각장애인·외국인을 위한 키오스크 안내 AI입니다. 항상 한국어로 짧고 명확하게 말하고, 버튼을 눌러야 하면 진동 안내(가까워질수록 진동이 빨라짐)를 포함합니다.
- 출력은 반드시 JSON 하나이며, schema는 status(success|ambiguous|fail), confidence(0-1), response_message, action(type: click_text|speak_only|ask_clarification, params) 입니다.
- action.type이 click_text일 때 params.target_text는 반드시 available_texts의 문자열을 정확히 사용합니다.
- action.type이 ask_clarification이면 params.candidates 배열(available_texts에서 최대 3개)을 제공합니다.
- user_input이 비어 있어도 현재 화면 정보를 요약하고 필요한 조치를 안내합니다.
- 화면에 원하는 항목이 없으면 가장 관련된 상위 버튼(카테고리 등)을 안내하거나, 없음을 알려 speak_only로 응답합니다.
- last_btn을 활용해 직전 흐름을 이어가며, 동일 메뉴 반복 안내를 피합니다.
""".strip()


def _few_shot_messages() -> List[Dict[str, str]]:
    examples: List[Dict[str, Any]] = [
        {
            "user": {
                "task": "analyze_kiosk",
                "user_input": "소고기 들어간 걸로 줘",
                "available_texts": ["불고기 버거", "치즈 버거", "사이드", "음료"],
                "dialogue_history": [],
                "last_btn": None,
            },
            "assistant": {
                "status": "ambiguous",
                "confidence": 0.62,
                "response_message": "불고기 버거와 치즈 버거 중 어떤 것을 선택할까요?",
                "action": {
                    "type": "ask_clarification",
                    "params": {"candidates": ["불고기 버거", "치즈 버거"]},
                },
            },
        },
        {
            "user": {
                "task": "analyze_kiosk",
                "user_input": "맥너겟 4조각으로 줘",
                "available_texts": [
                    "후렌치 후라이 -미디엄",
                    "맥너겟 4조각",
                    "골든 모짜렐라 치즈스틱",
                ],
                "dialogue_history": [],
                "last_btn": "세트 선택",
            },
            "assistant": {
                "status": "success",
                "confidence": 0.91,
                "response_message": "맥너겟 4조각 버튼으로 안내하겠습니다. 손가락을 움직이면 목표에 가까워질수록 진동이 빨라집니다.",
                "action": {"type": "click_text", "params": {"target_text": "맥너겟 4조각"}},
            },
        },
        {
            "user": {
                "task": "analyze_kiosk",
                "user_input": "에스프레소 없어?",
                "available_texts": [
                    "코카콜라- 미디엄",
                    "스프라이트- 미디엄",
                    "환타 - 미디엄",
                    "코카콜라 제로 - 미디엄",
                    "아이스 아메리카노 - 미디엄",
                ],
                "dialogue_history": [],
                "last_btn": "디핑 소스 선택",
            },
            "assistant": {
                "status": "fail",
                "confidence": 0.44,
                "response_message": "에스프레소는 현재 화면에 없습니다. 보이는 음료 중에서 선택하시겠어요?",
                "action": {"type": "speak_only", "params": {}},
            },
        },
    ]

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in examples:
        messages.append({"role": "user", "content": json.dumps(example["user"], ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(example["assistant"], ensure_ascii=False)})
    return messages


class AIService:
    def __init__(self, model: str, mock: bool = False, client: AsyncOpenAI | None = None):
        api_key = os.getenv("OPENAI_API_KEY")
        self.mock = mock or not api_key
        self.model = model
        self.client = client if not self.mock else None
        if not self.mock and api_key:
            self.client = client or AsyncOpenAI(api_key=api_key)
        elif not api_key and not mock:
            logger.warning("OPENAI_API_KEY가 설정되지 않아 mock 모드로 동작합니다.")

    async def analyze(self, payload: AnalyzeRequest) -> AnalyzeResponse:
        if self.mock:
            return self._mock_response(payload)

        try:
            messages = self._build_messages(payload)
            completion = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=messages,
            )
            content = completion.choices[0].message.content or "{}"
            parsed = json.loads(content)
            return AnalyzeResponse.model_validate(parsed)
        except (OpenAIError, json.JSONDecodeError, ValueError) as exc:
            logger.exception("AI 분석 실패: %s", exc)
            return self._fallback_response(payload, error=str(exc))

    def _build_messages(self, payload: AnalyzeRequest) -> Sequence[Dict[str, Any]]:
        base_messages = _few_shot_messages()
        user_payload: Dict[str, Any] = {
            "task": "analyze_kiosk",
            "session_id": payload.session_id,
            "user_input": payload.user_input or "",
            "available_texts": payload.ocr_texts,
            "dialogue_history": [turn.model_dump() for turn in payload.dialogue_history],
            "last_btn": payload.last_btn,
        }
        base_messages.append({"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)})
        return base_messages

    def _mock_response(self, payload: AnalyzeRequest) -> AnalyzeResponse:
        target = payload.ocr_texts[0] if payload.ocr_texts else None
        if target:
            message = f"{target} 버튼으로 안내하겠습니다. 손가락을 움직이면 목표에 가까워질수록 진동이 빨라집니다."
            action = Action(type="click_text", params={"target_text": target})
            return AnalyzeResponse(
                status="success",
                confidence=0.5,
                response_message=message,
                action=action,
            )
        message = "화면에서 선택할 수 있는 텍스트가 없습니다. 화면을 다시 비춰주세요."
        return AnalyzeResponse(
            status="fail",
            confidence=0.3,
            response_message=message,
            action=Action(type="speak_only", params={}),
        )

    def _fallback_response(self, payload: AnalyzeRequest, error: str) -> AnalyzeResponse:
        logger.error("OpenAI 오류로 기본 응답 반환: %s", error)
        message = "잠시 오류가 발생했습니다. 화면을 다시 한번 확인해 주세요."
        return AnalyzeResponse(
            status="fail",
            confidence=0.2,
            response_message=message,
            action=Action(type="speak_only", params={}),
        )
