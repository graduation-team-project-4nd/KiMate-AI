AI Server for KiMate-AI
=======================

시각장애인·외국인용 키오스크 보조 애플리케이션의 AI 서버 뼈대입니다. 백엔드 서버가 호출하는 두 개의 엔드포인트(`/api/analyze`, `/api/screen/detect`)를 FastAPI로 제공합니다.

요구사항 요약
-------------
- 모델: OpenAI Chat API (`OPENAI_MODEL`, 기본값 `gpt-4.1`).
- `/api/analyze`: 사용자 발화 + 화면 OCR 텍스트를 기반으로 액션 추천.
- `/api/screen/detect`: 이전/현재 OCR 비교로 화면 전환 감지. 전환 시 내부적으로 AI 분석을 수행해 `ai_analysis` 포함.
- 액션 타입: `click_text`, `speak_only`, `ask_clarification`.

시작하기
--------
1. 의존성 설치
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. 환경 변수
   ```bash
   export OPENAI_API_KEY=sk-...
   export OPENAI_MODEL=gpt-4.1           # 선택 사항
   export SCREEN_CHANGE_THRESHOLD=0.6    # 선택 사항
   export AI_SERVER_MOCK=0               # 1이면 OpenAI 호출 없이 더미 응답
   ```

3. 서버 실행
   ```bash
   uvicorn src.main:app --reload --port 8000
   ```

4. 샘플 호출
   ```bash
   curl -X POST http://localhost:8000/api/analyze \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "sess_001",
       "user_input": "불고기 버거 하나",
       "ocr_texts": ["추천메뉴", "불고기버거", "4500원", "치즈버거", "다음"],
       "dialogue_history": [],
       "last_btn": "햄버거"
     }'
  ```

5. 시나리오 리플레이 (중간/최종 데모)
   ```bash
   export AI_SERVER_MOCK=1  # 기본적으로 모의 응답
   python scripts/run_scenarios.py
   ```
   - 실제 LLM으로 검증하려면 `AI_SERVER_MOCK=0`로 바꾸고 `OPENAI_API_KEY`를 설정하세요.

구조
----
- `src/main.py`: FastAPI 엔트리포인트, 라우터, DI.
- `src/models.py`: Pydantic 요청/응답 모델 정의.
- `src/ai_service.py`: 프롬프트 설계, OpenAI 호출, 기본/더미 응답.
- `src/screen_detect.py`: 화면 유사도 계산(Jaccard)과 감지 로직.

주요 동작
--------
- Analyze
  - 입력: `user_input`, `ocr_texts`, `dialogue_history`, `last_btn`
  - 출력: `status`(`success|ambiguous|fail`), `response_message`, `action`
  - 프롬프트는 시각 장애인/외국인 보조에 특화된 안내, 진동 안내 멘트 포함, 후보 되묻기, 화면에 없는 경우 상위 버튼/대안 안내를 포함합니다.
- Screen Detect
  - Jaccard 유사도가 `SCREEN_CHANGE_THRESHOLD` 미만이면 `is_changed=true`.
  - 화면이 변하면 내부적으로 `analyze`를 재사용해 `ai_analysis`를 넣어줍니다.

기타
----
- OpenAI 호출 실패 또는 `AI_SERVER_MOCK=1`이면 규격에 맞춘 더미 응답을 반환합니다.
- 대화 히스토리는 백엔드에서 관리하며, AI 서버는 stateless합니다.
