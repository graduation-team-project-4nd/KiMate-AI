from __future__ import annotations

import os
from typing import Dict, List, Optional

from pathlib import Path
import sys

from fastapi.testclient import TestClient

# Ensure project root is on sys.path so `src` imports work when running the script directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_client() -> TestClient:
    # Load .env first so explicit settings win.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    # Use mock by default ONLY if not explicitly set; respects .env/real env.
    os.environ.setdefault("AI_SERVER_MOCK", "1")
    os.environ.setdefault("OPENAI_MODEL", "gpt-5.1")
    # Import after setting env so AI service picks up the mock flag.
    from src.main import app

    return TestClient(app)


def call_analyze(
    client: TestClient,
    session_id: str,
    user_input: Optional[str],
    ocr_texts: List[str],
    dialogue_history: List[Dict],
    last_btn: Optional[str],
) -> Dict:
    payload = {
        "session_id": session_id,
        "user_input": user_input,
        "ocr_texts": ocr_texts,
        "dialogue_history": dialogue_history,
        "last_btn": last_btn,
    }
    res = client.post("/api/analyze", json=payload)
    res.raise_for_status()
    return res.json()


def call_screen_detect(
    client: TestClient,
    session_id: str,
    previous_texts: List[str],
    current_texts: List[str],
    dialogue_history: List[Dict],
    user_input: Optional[str],
    last_btn: Optional[str],
) -> Dict:
    payload = {
        "session_id": session_id,
        "previous_texts": previous_texts,
        "current_texts": current_texts,
        "dialogue_history": dialogue_history,
        "user_input": user_input,
        "last_btn": last_btn,
    }
    res = client.post("/api/screen/detect", json=payload)
    res.raise_for_status()
    return res.json()


def append_turn(history: List[Dict], role: str, utterance: str, action: Optional[Dict] = None) -> None:
    history.append({"role": role, "utterance": utterance, "action": action})


def print_resp(prefix: str, resp: Dict) -> None:
    print(f"\n[{prefix}] status={resp.get('status')}, confidence={resp.get('confidence')}")
    print(f"  message: {resp.get('response_message')}")
    print(f"  action : {resp.get('action')}")


def print_history(history: List[Dict]) -> None:
    if not history:
        print("  dialogue_history: (empty)")
        return
    print("  dialogue_history:")
    for turn in history:
        role = turn.get("role", "?")
        utterance = turn.get("utterance") or ""
        action = turn.get("action") or {}
        action_type = action.get("type")
        params = action.get("params") or {}
        action_desc = ""
        if action_type == "click_text":
            action_desc = f" [{action_type}:{params.get('target_text')}]"
        elif action_type:
            action_desc = f" [{action_type}]"
        print(f"    - {role}: {utterance}{action_desc}")


def run_mid_demo(client: TestClient) -> None:
    print("\n=== 중간 시연 시나리오 ===")
    session_id = "sess_mid"
    screen = ["버거", "사이드", "음료", "불고기 버거", "새우버거", "치즈버거", "치킨버거"]
    dialogue_history: List[Dict] = []
    last_btn: Optional[str] = None

    # 유스케이스 1: 소고기 들어간 메뉴
    user_input = "소고기 들어간 걸로 줘"
    resp1 = call_analyze(client, session_id, user_input, screen, dialogue_history, last_btn)
    append_turn(dialogue_history, "user", user_input)
    append_turn(dialogue_history, "assistant", resp1["response_message"], resp1["action"])
    if resp1["action"]["type"] == "click_text":
        last_btn = resp1["action"]["params"].get("target_text", last_btn)
    print_resp("유스케이스1-1", resp1)
    print_history(dialogue_history)

    # 사용자가 구체적으로 답한 상황
    user_input2 = "불고기 버거 줘"
    resp2 = call_analyze(client, session_id, user_input2, screen, dialogue_history, last_btn)
    append_turn(dialogue_history, "user", user_input2)
    append_turn(dialogue_history, "assistant", resp2["response_message"], resp2["action"])
    if resp2["action"]["type"] == "click_text":
        last_btn = resp2["action"]["params"].get("target_text", last_btn)
    print_resp("유스케이스1-2", resp2)
    print_history(dialogue_history)

    # 유스케이스 2: 화면에 없는 메뉴(감자 튀김)
    user_input3 = "감자 튀김 주문해 줘"
    resp3 = call_analyze(client, session_id, user_input3, screen, dialogue_history, last_btn)
    append_turn(dialogue_history, "user", user_input3)
    append_turn(dialogue_history, "assistant", resp3["response_message"], resp3["action"])
    print_resp("유스케이스2", resp3)
    print_history(dialogue_history)


def run_final_demo(client: TestClient) -> None:
    print("\n=== 최종 시연 시나리오 ===")
    session_id = "sess_final"
    dialogue_history: List[Dict] = []
    last_btn: Optional[str] = None

    screens = [
        {
            "name": "화면1-식사장소",
            "texts": ["식사 장소를 선택해주세요", "매장", "포장"],
            "user_inputs": ["에그 불고기 버거 주문해 줘"],
        },
        {
            "name": "화면1-식사장소",
            "texts": ["식사 장소를 선택해주세요", "매장", "포장"],
            "user_inputs": ["매장"],
        },
        {
            "name": "화면2-메인카테고리",
            "texts": [
                "홈",
                "메뉴 알아보기",
                "추천 메뉴",
                "추천메뉴",
                "버거&세트",
                "버거",
                "이달의",
                "커피&디저트",
                "해피스낵",
                "사이드",
                "커피",
                "상콤달콤함이 하늘을 찔러!",
                "디저트",
                "음료",
                "지금 가장 가까운 맥카페!",
                "해피빈",
            ],
            "user_inputs": [None],
        },
        {
            "name": "화면3-버거리스트1",
            "texts": [
                "홈",
                "전체",
                "치킨",
                "씨푸드",
                "불고기",
                "추천 메뉴",
                "버거",
                "더블 빅맥",
                "트리플 치즈 버거",
                "7000원, 675kcal",
                "5600원, 619kcal",
                "해피스낵",
                "사이드",
                "커피",
                "빅맥",
                "더블 불고기 버거",
                "4600원, 583kcal",
                "4400원, 636kcal",
                "디저트",
                "음료",
                "해피빈",
            ],
            "user_inputs": [None],
        },
        {
            "name": "화면4-버거리스트2",
            "texts": [
                "홈",
                "전체",
                "치킨",
                "씨푸드",
                "불고기",
                "추천 메뉴",
                "버거",
                "맥스파이시 상하이 버거",
                "에그 불고기 버거",
                "8000원, 897kcal",
                "4400원, 630kcal",
                "해피스낵",
                "사이드",
                "커피",
                "맥치킨",
                "맥치킨 모짜렐라",
                "5200원, 621kcal",
                "6600원, 789kcal",
                "디저트",
                "음료",
                "해피빈",
            ],
            "user_inputs": [None],
        },

        # 5번 화면: None → 실제 발화
        {
            "name": "화면5-세트여부",
            "texts": ["세트로 주문하시겠습니까?", "세트 선택", "단품 선택", "취소", "4400원, 630kcal"],
            "user_inputs": [None],
        },
        {
            "name": "화면5-세트여부",
            "texts": ["세트로 주문하시겠습니까?", "세트 선택", "단품 선택", "취소", "4400원, 630kcal"],
            "user_inputs": ["세트"],
        },

        # 6번 화면: None → 실제 발화
        {
            "name": "화면6-세트선택",
            "texts": [
                "주문 확인하기",
                "에그 불고기 버거 - 세트",
                "에그 불고기 버거 - 라지세트",
                "5400원, 890kcal",
                "6400원, 940kcal",
            ],
            "user_inputs": [None],
        },
        {
            "name": "화면6-세트선택",
            "texts": [
                "주문 확인하기",
                "에그 불고기 버거 - 세트",
                "에그 불고기 버거 - 라지세트",
                "5400원, 890kcal",
                "6400원, 940kcal",
            ],
            "user_inputs": ["그냥 세트"],
        },

        # 7번 화면: None → 실제 발화
        {
            "name": "화면7-사이드선택",
            "texts": [
                "세트메뉴 사이드를 선택하세요",
                "후렌치 후라이 -미디엄",
                "맥너겟 4조각",
                "골든 모짜렐라 치즈스틱",
                "332kcal",
                "171kcal",
                "159kcal",
            ],
            "user_inputs": [None],
        },
        {
            "name": "화면7-사이드선택",
            "texts": [
                "세트메뉴 사이드를 선택하세요",
                "후렌치 후라이 -미디엄",
                "맥너겟 4조각",
                "골든 모짜렐라 치즈스틱",
                "332kcal",
                "171kcal",
                "159kcal",
            ],
            "user_inputs": ["맥너겟 4조각으로 줘"],
        },

        # 8번 화면: None → 실제 발화
        {
            "name": "화면8-소스선택",
            "texts": ["디핑 소스를 선택하세요", "랜덤 소스", "오렌지 칠리 소스", "케이준 소스", "스위트 칠리 소스", "소스 없음"],
            "user_inputs": [None],
        },
        {
            "name": "화면8-소스선택",
            "texts": ["디핑 소스를 선택하세요", "랜덤 소스", "오렌지 칠리 소스", "케이준 소스", "스위트 칠리 소스", "소스 없음"],
            "user_inputs": ["오렌지 칠리 소스 줘"],
        },

        # 9번 화면: None → 실제 2단계 발화
        {
            "name": "화면9-음료선택",
            "texts": [
                "세트메뉴 음료를 선택하세요",
                "코카콜라- 미디엄",
                "스프라이트- 미디엄",
                "환타 - 미디엄",
                "코카콜라 제로 - 미디엄",
                "아이스 아메리카노 - 미디엄",
                "아이스 카페 라떼 - 미디엄",
            ],
            "user_inputs": [None],
        },
        {
            "name": "화면9-음료선택",
            "texts": [
                "세트메뉴 음료를 선택하세요",
                "코카콜라- 미디엄",
                "스프라이트- 미디엄",
                "환타 - 미디엄",
                "코카콜라 제로 - 미디엄",
                "아이스 아메리카노 - 미디엄",
                "아이스 카페 라떼 - 미디엄",
            ],
            "user_inputs": ["에스프레소 없어?"],
        },
        {
            "name": "화면9-음료선택",
            "texts": [
                "세트메뉴 음료를 선택하세요",
                "코카콜라- 미디엄",
                "스프라이트- 미디엄",
                "환타 - 미디엄",
                "코카콜라 제로 - 미디엄",
                "아이스 아메리카노 - 미디엄",
                "아이스 카페 라떼 - 미디엄",
            ],
            "user_inputs": ["그럼 코카콜라-미디엄으로 줘"],
        },

        # 10번 화면: None → 실제 발화
        {
            "name": "화면10-요약",
            "texts": [
                "에그 불고기 버거 - 세트",
                "5400원, 890kcal",
                "에그 불고기 버거",
                "맥너겟 4조각 (기간 한정 교환 중)",
                "오렌지 칠리 소스",
                "코카-콜라 - 미디엄",
                "-",
                "1",
                "+",
                "재료추가/변경",
                "수정",
                "취소",
                "장바구니 추가",
            ],
            "user_inputs": [None],
        },
        {
            "name": "화면10-요약",
            "texts": [
                "에그 불고기 버거 - 세트",
                "5400원, 890kcal",
                "에그 불고기 버거",
                "맥너겟 4조각 (기간 한정 교환 중)",
                "오렌지 칠리 소스",
                "코카-콜라 - 미디엄",
                "-",
                "1",
                "+",
                "재료추가/변경",
                "수정",
                "취소",
                "장바구니 추가",
            ],
            "user_inputs": ["없어"],
        },

        {
            "name": "화면11-카테고리로복귀",
            "texts": [
                "홈",
                "전체",
                "치킨",
                "씨푸드",
                "불고기",
                "추천 메뉴",
                "버거",
                "맥스파이시 상하이 버거",
                "에그 불고기 버거",
                "8000원, 897kcal",
                "4400원, 630kcal",
                "해피스낵",
                "사이드",
                "커피",
                "맥치킨",
                "맥치킨 모짜렐라",
                "5200원, 621kcal",
                "6600원, 789kcal",
                "디저트",
                "음료",
                "해피빈",
                "주문내역",
                "5400원",
            ],
            "user_inputs": [None],
        },
        {
            "name": "화면12-주문내역",
            "texts": ["취소", "에그 불고기 버거 - 세트", "-1+", "세부 정보 표시", "추가주문", "주문 완료"],
            "user_inputs": [None],
        },

        # 13번 화면: None → 실제 발화
        {
            "name": "화면13-결제선택",
            "texts": ["결제 방법을 선택해주세요", "카드결제", "모바일 상품권"],
            "user_inputs": [None],
        },
        {
            "name": "화면13-결제선택",
            "texts": ["결제 방법을 선택해주세요", "카드결제", "모바일 상품권"],
            "user_inputs": ["카드 결제로"],
        },

        {
            "name": "화면14-결제안내",
            "texts": [
                "결제를 진행해주세요",
                "IC 신용/체크카드 사용 시",
                "카드를 화살표 방향으로 투입구에 넣어주세요",
                "결제 오류 시, 카드를 긁어주세요",
            ],
            "user_inputs": [None],
        },
    ]

    for idx, step in enumerate(screens, 1):
        print(f"\n--- {step['name']} ({idx}/{len(screens)}) ---")
        user_inputs = step.get("user_inputs") or [None]
        primary_input = user_inputs[0]

        resp = call_analyze(
            client=client,
            session_id=session_id,
            user_input=primary_input,
            ocr_texts=step["texts"],
            dialogue_history=dialogue_history,
            last_btn=last_btn,
        )

        if primary_input:
            append_turn(dialogue_history, "user", primary_input)
        append_turn(dialogue_history, "assistant", resp["response_message"], resp["action"])
        if resp["action"]["type"] == "click_text":
            last_btn = resp["action"]["params"].get("target_text", last_btn)
        print_resp("analyze", resp)
        print_history(dialogue_history)

        # Additional user inputs on the same screen (e.g., 되묻기 이후 답변)
        extra_inputs = user_inputs[1:]
        for extra_idx, extra in enumerate(extra_inputs, start=1):
            resp = call_analyze(
                client,
                session_id=session_id,
                user_input=extra,
                ocr_texts=step["texts"],
                dialogue_history=dialogue_history,
                last_btn=last_btn,
            )
            if extra:
                append_turn(dialogue_history, "user", extra)
            append_turn(dialogue_history, "assistant", resp["response_message"], resp["action"])
            if resp["action"]["type"] == "click_text":
                last_btn = resp["action"]["params"].get("target_text", last_btn)
            print_resp(f"추가 발화 {extra_idx}", resp)
            print_history(dialogue_history)


if __name__ == "__main__":
    client = build_client()
    run_mid_demo(client)
    run_final_demo(client)
    print("\n시나리오 실행 완료 (AI_SERVER_MOCK=%s)" % os.getenv("AI_SERVER_MOCK", "0"))
