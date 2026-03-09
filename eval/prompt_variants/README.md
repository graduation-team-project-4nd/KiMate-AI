# 프롬프트 프로파일

`eval/prompt_variants/`는 `python -m eval.prompt_runner`에서 사용하는 프롬프트 프로파일 모음입니다.

`baseline/`만 단독으로 완성된 프롬프트이고, 나머지 프로파일은 `profile.json`의 `extends`를 통해 부모 프로파일을 상속하는 델타 형태입니다.

## 상속 구조

```text
baseline
└── optimized_v1
    ├── optimized_v2
    │   └── optimized_v3
    └── generalized_v1
        └── generalized_v2
```

## 프로파일 요약

| 프로파일 | 부모 | 핵심 변경점 | few-shot 변화 | 검증 결과 요약 | 비고 |
| --- | --- | --- | --- | --- | --- |
| `baseline` | 없음 | 프롬프트 실험 전 `src/ai_service.py`에 있던 원본 프롬프트 추출 | 기본 예시 3개 | `gpt-5.1`: `final_demo 20/23`, `gpt-5.4`: `final_demo 20/23` | 출발점 |
| `optimized_v1` | `baseline` | 빈 `user_input` 선택 화면 처리 강화, 목표 메뉴 미노출 시 스크롤 유도, `없어 -> 장바구니 추가` 규칙 추가 | `+3` | `gpt-5.1`: `final_demo 22/23`, `gpt-5.4`: 현재 evaluator 기준 `final_demo 22/23` | 초기 후보 중 일반화 성능이 비교적 양호 |
| `optimized_v2` | `optimized_v1` | 장바구니 추가 이후 `주문내역` 같은 주문 확인 진입 버튼 우선 규칙 추가 | `+1` | `gpt-5.1`: `final_demo 22/23`, `gpt-5.4`: `final_demo 22/23` | 균형은 좋지만 최종 채택 버전은 아님 |
| `optimized_v3` | `optimized_v2` | 장바구니 추가 이후 전환 화면 처리 강화 | `+1` | `gpt-5.1`: `23/23`, `gpt-5.4`: `23/23` on `final_demo` | 점수는 가장 높지만 eval에 더 맞춰진 편 |
| `generalized_v1` | `optimized_v1` | 추가 few-shot 없이, 장바구니 추가 이후 전환 화면을 일반화된 시스템 규칙으로 재정의 | `+0` | `gpt-5.4`: `final_demo 22/23`, `final_demo + critical_cases + post_add_to_cart_navigation = 28/29` | `screen11`은 해결했지만 옵션 선택 화면 1개 회귀 |
| `generalized_v2` | `generalized_v1` | 장바구니 이후 규칙이 진짜 add-to-cart 직후에만 동작하도록 가드 추가 | `+0` | `gpt-5.1`: `final_demo 23/23`, `gpt-5.4`: `final_demo 23/23`, 전체 묶음 `29/29` | 현재 추천 버전 |

## 현재 상태

- `2026-03-09` 기준으로 실제 서버의 `src/ai_service.py` 프롬프트는 `generalized_v2`를 반영합니다.
- 이 문서에서 참조하는 리포트는 `eval/results/2026-03-09/` 아래에 정리되어 있습니다.

## 메모

- `optimized_v1`의 `gpt-5.4` 결과는 원래 `20/23`으로 기록됐지만, 이후 `screen4`와 `screen10` 평가 기준이 완화되면서 현재는 `22/23`으로 해석합니다.
- 즉 이 경우는 모델 성능 저하라기보다, 보수적인 응답을 evaluator가 너무 엄격하게 오답 처리했던 문제에 가깝습니다.
- 자세한 변경 배경과 타임라인은 `eval/PROMPT_CHANGELOG.md`를 참고하면 됩니다.
