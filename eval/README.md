# Eval

`run_scenarios.py`는 시연용으로는 유용하지만, 모델 출력이 맞는지 틀린지는 판단해주지 않습니다.  
`eval/`은 기존 앱 파일을 바꾸지 않고 별도의 평가 흐름을 추가하기 위한 폴더입니다.

## 구성

- `cases/`: 입력과 허용 가능한 정답을 정의한 평가 케이스
- `runner.py`: 현재 FastAPI 앱을 그대로 호출해 케이스를 실행하는 러너
- `prompt_runner.py`: 프롬프트 프로파일로 직접 메시지를 만들어 OpenAI를 호출하는 러너
- `scoring.py`: 모델 출력과 기대 규칙을 비교하는 채점기
- `compare.py`: 저장된 두 평가 리포트를 비교하는 도구
- `prompt_variants/README.md`: 프롬프트 프로파일 구조와 현재 추천 버전 설명
- `PROMPT_CHANGELOG.md`: 프롬프트 변경 이력과 평가 해석 메모
- `results/`: 생성된 평가 리포트 저장 위치

## 동작 방식

각 케이스에는 다음 정보가 들어 있습니다.

- 요청 payload 형태
- 기대하는 `status`
- 기대하는 `action.type`
- 필요하면 `target_text`, 후보 목록, 메시지 키워드 같은 추가 조건
- 회귀가 눈에 띄도록 하는 `severity`

채점은 단순 문자열 완전 일치만 보지 않습니다.  
여러 출력이 모두 허용 가능하면 `alternatives`로 복수 정답을 정의할 수 있습니다.

## 권장 사용 흐름

1. 현재 모델로 같은 케이스 묶음을 실행합니다.
2. 리포트를 저장합니다.
3. 모델 버전이나 프롬프트를 바꾼 뒤 다시 실행합니다.
4. 두 리포트를 비교합니다.

## 명령어

`eval/cases/` 아래의 모든 스위트를 실행:

```bash
python -m eval.runner
```

최종 시연 스위트만 실행:

```bash
python -m eval.runner --case-file eval/cases/final_demo.json
```

핵심 회귀 케이스만 실행:

```bash
python -m eval.runner --case-file eval/cases/critical_cases.json
```

두 리포트 비교:

```bash
python -m eval.compare eval/results/REPORT_A.json eval/results/REPORT_B.json
```

프롬프트 프로파일을 직접 OpenAI에 실행:

```bash
python -m eval.prompt_runner --prompt-profile baseline --case-file eval/cases/critical_cases.json
```

## 환경 메모

- 실제 평가에는 `AI_SERVER_MOCK=0`을 권장합니다.
- `AI_SERVER_MOCK=1`이어도 러너는 동작하지만, mock 응답이 항상 첫 OCR 텍스트를 누르기 때문에 많은 케이스가 실패합니다.
- `runner.py`는 `TestClient`로 기존 앱을 불러와 실행합니다.
- `prompt_runner.py`는 로컬 FastAPI 앱을 거치지 않고, `eval/prompt_variants/`의 프롬프트 파일을 읽어 OpenAI를 직접 호출합니다.

## 리포트 해석

- `pass=true`: 해당 케이스에 정의된 모든 체크를 만족
- `score`: 가장 잘 맞는 기대 조건 기준 충족 비율
- `critical_failures`: `critical`로 표시된 step 중 실패한 개수

일반적인 회귀 판단 기준은 다음과 같습니다.

- `critical_failures`가 늘어나면 막아야 하는 회귀로 봄
- `click_text.target_text`를 잘못 누르는 오류는 메시지 문구 차이보다 더 심각하게 봄
- 스크롤 상황에서 `speak_only`와 `click_text`를 혼동하는 것은 회귀로 취급
