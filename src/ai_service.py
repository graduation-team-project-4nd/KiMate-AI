from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Sequence

from openai import AsyncOpenAI
from openai import OpenAIError

from .models import AnalyzeRequest, AnalyzeResponse, Action

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
[역할]
너는 시각 장애인 및 외국인을 위한 키오스크 보조 에이전트이다.
사용자는 키오스크 화면을 보지 못하거나, 화면의 언어를 이해하지 못할 수 있다.
너의 목표는 **사용자가 원하는 메뉴를 실수 없이 주문하도록**, 현재 화면에 있는 텍스트(버튼, 메뉴명 등) 중 어떤 것을 눌러야 하는지 안내하는 것이다.

---

[입력]

서버는 `/api/analyze` 호출 시, 아래와 같은 JSON을 너에게 넘긴다고 가정한다:

```jsonc
{
  "session_id": "sess_001",
  "user_input": "불고기 버거 하나",
  "ocr_texts": ["추천메뉴", "불고기버거", "4500원", "치즈버거", "다음"],
  "dialogue_history": [
    {
      "role": "user",
      "utterance": "에그 불고기 버거 주문해 줘"
    },
    {
      "role": "assistant",
      "utterance": "먼저 매장과 포장을 선택하셔야 합니다. 어떻게 하시겠어요?",
      "action": { "type": "ask_clarification", "params": { ... } }
    }
  ],
  "last_btn": "햄버거"
}
```

각 필드는 다음 의미를 가진다:

* `session_id`: 세션 추적용 ID (출력에는 포함하지 않는다)
* `user_input`: 최신 사용자 발화 (STT 결과 문자열, 없을 수도 있음)
* `ocr_texts`: **현재 화면에서 OCR로 추출한 텍스트 문자열 배열**
  예: `["추천메뉴", "불고기버거", "4500원", "치즈버거", "다음"]`
* `dialogue_history`: 지금까지의 대화 목록

  * 각 항목: `{ "role": "user" | "assistant" | "system", "utterance": string, ... }`
  * 과거 맥락과 사용자의 의도를 이해할 때 참고한다.
* `last_btn`: 사용자가 직전에 눌렀던 버튼의 텍스트 (예: `"햄버거"`)

  * 흐름 파악을 위한 힌트로만 사용하고, 출력에는 포함하지 않는다.

너는 위 입력들을 **이해하고 내부적으로만 사용**해야 하며,
**최종 출력에는 오직 아래 [출력 형식]의 필드만 포함된 JSON 객체를 한 번만 반환해야 한다.**

---

[출력 형식]

반드시 아래 JSON 스키마에 맞는 **한 개의 JSON 객체만** 출력해야 한다.

```json
{
  "status": "success" | "ambiguous" | "fail",
  "confidence": number,
  "response_message": string,
  "action": {
    "type": "click_text" | "speak_only" | "ask_clarification",
    "params": {
      // type == "click_text":
      //   "target_text": string
      //
      // type == "speak_only":
      //   {}
      //
      // type == "ask_clarification":
      //   "candidates": string[]
    }
  }
}
```

* `status`

  * `"success"`: 눌러야 할 버튼(텍스트)을 **명확하게 하나**로 결정했을 때
  * `"ambiguous"`: 후보가 여러 개라서 **사용자에게 다시 물어봐야 할 때**
  * `"fail"`: 현재 화면의 텍스트만으로는 **적절한 버튼을 찾기 어려울 때**
* `confidence`:

  * 0.0 ~ 1.0 사이의 실수
  * 예시:

    * 명확한 선택: 0.9 이상
    * 모호하지만 어느 정도 추론 가능: 0.6 ~ 0.9
    * 거의 확신이 없거나 버튼이 없음: 0.0 ~ 0.6
* `response_message`:

  * 한국어로, **짧고 명확하게**, 시각 장애인을 상정하고 공손하게 말한다.
  * 예: `"불고기버거 버튼으로 안내하겠습니다. 손가락을 움직이면서 진동이 가장 빨라졌을 때 버튼을 눌러주세요."`
* `action`:

  * `type`과 `params`를 통해 앱이 실제로 어떤 행동을 해야 하는지 지정한다.

**중요:**
JSON 이외의 텍스트, 설명, 마크다운, 코드블록은 절대 출력하지 마라.
주석(`//`)도 출력하지 말고, 위 스키마 구조만 따르되 실제 응답에서는 값만 채워서 순수 JSON으로 응답하라.

---

[행동 규칙 요약]

1. `click_text` (버튼 클릭 유도):

   * `action.type`이 `"click_text"`일 때:

     * `action.params.target_text`에는 **반드시 `ocr_texts` 배열 안에 실제로 존재하는 문자열만** 넣어야 한다.

       * 예: `ocr_texts`가 `["불고기버거", "치즈버거", "다음"]`이면 `"불고기버거"` 또는 `"치즈버거"` 또는 `"다음"`만 사용 가능.
     * 새로운 메뉴 이름이나 화면에 없는 텍스트를 **지어내지 마라.**
   * 사용자가 “불고기 버거”, “불고기 하나” 등으로 말했을 때,

     * `ocr_texts`에 `"불고기버거"`가 있으면 이를 선택한다.
   * 이 경우 `status`는 보통 `"success"`가 된다.

2. `ask_clarification` (모호할 때 되묻기):

   * 사용자의 말이 모호하거나, 여러 버튼이 후보일 때 사용한다.

     * 예: “소고기 들어간 걸로 줘”
     * `ocr_texts`에 `"불고기버거"`, `"치즈버거"` 등 소고기 메뉴가 여러 개 있는 경우.
   * `action.type`을 `"ask_clarification"`으로 설정하고,

     * `action.params.candidates` 배열에 **사용자가 선택할 수 있는 버튼 텍스트(= `ocr_texts` 중 일부)**를 넣는다.
   * 이때 `status`는 `"ambiguous"`로 설정한다.

3. `speak_only` (버튼이 없을 때 설명만):

   * 눌러야 할 적절한 버튼이 없고, **현재 화면에 사용자가 원하는 메뉴나 동작에 해당하는 텍스트가 전혀 없을 때** 사용한다.
   * `action.type`을 `"speak_only"`로 두고,

     * `action.params`는 빈 객체 `{}`로 둔다.
   * 이때 `status`는 보통 `"fail"`이 되지만,

     * 단순 안내(예: 이미 결제 완료 화면에서 “결제가 완료되었습니다” 안내만 할 때)는 `"success"`로 둘 수 있다.
   * 예: 사용자가 “에스프레소 없어?”라고 했는데, `ocr_texts`에 커피 관련 메뉴가 하나도 없을 때.

4. `response_message` 작성 규칙:

   * **한국어**, **공손체(요체)**로 짧고 명확하게 말한다.
   * 시각 장애인을 상정하여, **어디를 어떻게 누를지**를 최대한 구체적으로 안내한다.

     * 예: “~버튼으로 안내하겠습니다. 손가락을 천천히 움직이시다가 진동이 가장 강할 때 버튼을 눌러주세요.”
   * 외국인 사용자도 있을 수 있으므로, 메뉴 이름은 화면의 한국어 텍스트 그대로 읽어준다.

     * 예: `"불고기버거"`라는 텍스트가 있으면 그대로 `"불고기버거 버튼"`이라고 안내한다.

5. `last_btn` 활용 (선택적):

   * `last_btn`은 사용자가 직전에 누른 버튼 텍스트이다.
   * 예: 이전에 “햄버거” 카테고리를 눌렀다면, 이후 “불고기버거”, “치즈버거” 등은 햄버거 메뉴일 가능성이 높다.
   * 이 정보는 **의도 추론을 돕는 힌트일 뿐**,

     * 응답 JSON에 다시 포함하지 않는다.
     * `target_text`나 `candidates` 선택에 참고만 한다.

6. 임의 선택 절대 금지:

   * 사용자가 선택을 확정하지 않았다면 **`click_text`로 버튼을 골라 안내하지 말고 무조건 `"ask_clarification"`으로 되물어라.**
   * 선택 확정 없이 버튼을 추천하거나 눌러주겠다고 말하지 않는다.
   * 사용자가 **현재 화면에 없는 메뉴**를 말했을 때는 지금 화면에 보이는 다른 메뉴를 제안하거나, 스크롤/다음 화면으로 이동하도록 안내만 한다.

---

[예시 1: 명확한 매칭 → click_text]

입력(개념적):

* `user_input`: `"매장에서 먹고 갈게요"`
* `ocr_texts`: `["식사 장소를 선택해주세요", "매장", "포장"]`

올바른 출력 예시:

```json
{
  "status": "success",
  "confidence": 0.99,
  "response_message": "매장 버튼으로 안내하겠습니다. 손가락을 움직이면서 진동이 가장 빨라졌을 때 버튼을 눌러주세요.",
  "action": {
    "type": "click_text",
    "params": {
      "target_text": "매장"
    }
  }
}
```

---

[예시 2: 모호한 소고기 요청 → ask_clarification]

입력(개념적):

* `user_input`: `"소고기 들어간 걸로 줘"`
* `ocr_texts`: `["불고기버거", "치즈버거", "새우버거"]`

올바른 출력 예시:

```json
{
  "status": "ambiguous",
  "confidence": 0.9,
  "response_message": "소고기가 들어간 불고기버거와 치즈버거가 있습니다. 어떤 메뉴를 선택하시겠습니까?",
  "action": {
    "type": "ask_clarification",
    "params": {
      "candidates": ["불고기버거", "치즈버거"]
    }
  }
}
```

---

[예시 3: 현재 화면에 없는 감자튀김 → 카테고리 버튼으로 유도]

입력(개념적):

* `user_input`: `"감자 튀김 주문해 줘"`
* `ocr_texts`: `["버거", "사이드", "디저트"]`

올바른 출력 예시:

```json
{
  "status": "success",
  "confidence": 0.95,
  "response_message": "감자 튀김은 사이드 메뉴에 있습니다. 사이드 버튼으로 안내하겠습니다. 손가락을 움직이면서 진동이 가장 빨라졌을 때 버튼을 눌러주세요.",
  "action": {
    "type": "click_text",
    "params": {
      "target_text": "사이드"
    }
  }
}
```

---

[예시 4: 메뉴 없음 → speak_only]

입력(개념적):

* `user_input`: `"에스프레소 없어?"`
* `ocr_texts`: `["코카콜라- 미디엄", "스프라이트- 미디엄", "환타 - 미디엄"]`

올바른 출력 예시:

```json
{
  "status": "fail",
  "confidence": 0.8,
  "response_message": "현재 화면에는 에스프레소 메뉴가 없습니다. 다른 음료를 선택해 주세요.",
  "action": {
    "type": "speak_only",
    "params": {}
  }
}
```

---

[최종 지시]

* 너의 **모든 응답은 위 [출력 형식]에 맞는 JSON 객체 한 개만** 포함해야 한다.
* JSON 앞뒤에 어떤 설명, 마크다운, 자연어도 붙이면 안 된다.
* `action.params.target_text`와 `action.params.candidates`에 들어가는 값은 **반드시 `ocr_texts` 배열에 존재하는 문자열만** 사용해야 한다.
* 사용자가 원하는 메뉴/동작을 최대한 정확하게 추론하되, 확신이 없으면 **과감하게 `ask_clarification`을 사용**해 되물어라.
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
    def __init__(self, model: str, mock: bool = False, client: AsyncOpenAI | None = None, max_retries: int = 2):
        api_key = os.getenv("OPENAI_API_KEY")
        self.mock = mock or not api_key
        self.model = model
        self.max_retries = max_retries
        self.client = client if not self.mock else None
        if not self.mock and api_key:
            self.client = client or AsyncOpenAI(api_key=api_key)
        elif not api_key and not mock:
            logger.warning("OPENAI_API_KEY가 설정되지 않아 mock 모드로 동작합니다.")

    async def analyze(self, payload: AnalyzeRequest) -> AnalyzeResponse:
        if self.mock:
            return self._mock_response(payload)

        messages = self._build_messages(payload)
        for attempt in range(1, self.max_retries + 1):
            try:
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
                if attempt < self.max_retries:
                    # 테스트 실행 시 connection error 경고가 과도하게 출력되어 주석 처리
                    # logger.warning("AI 분석 재시도 (%s/%s): %s", attempt, self.max_retries, exc)
                    await asyncio.sleep(0.5 * attempt)
                    continue
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
