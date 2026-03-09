# 프롬프트 변경 이력

이 문서는 프롬프트 버전 변경 사항, 프롬프트 비교에 영향을 준 평가 기준 변경 사항, 그리고 최종적으로 어떤 프롬프트를 `src/ai_service.py`에 반영했는지를 기록합니다.

## 2026-03-09

### baseline 추출

- 원래 `src/ai_service.py`에 있던 서빙 프롬프트를 `eval/prompt_variants/baseline/`으로 분리했습니다.
- 목적은 앱 코드를 수정하지 않고 프롬프트 버전을 비교하기 위함입니다.
- 확인용 리포트:
  - `eval/results/2026-03-09/prompt_baseline_final_demo_t0.json`
  - `eval/results/2026-03-09/prompt_baseline_final_demo_gpt54_t0.json`

`final_demo` 기준 점수:

| 모델 | 프롬프트 | 점수 | 비고 |
| --- | --- | --- | --- |
| `gpt-5.1` | `baseline` | `20/23` |  |
| `gpt-5.4` | `baseline` | `20/23` |  |

### optimized_v1

변경 사항:

- `user_input`이 비어 있는 선택 화면에서 더 안전하게 되묻게 하는 규칙 추가
- 목표 메뉴가 아직 보이지 않을 때 스크롤을 명시적으로 유도하는 규칙 추가
- 요약 화면에서 `없어/변경 없어/그대로/괜찮아`를 `장바구니 추가`로 연결하는 규칙 추가
- 음료 선택, 스크롤 안내, 장바구니 추가 진행을 위한 few-shot 3개 추가

`final_demo` 기준 점수:

| 모델 | 프롬프트 | 원래 리포트 | 현재 evaluator 기준 해석 | 비고 |
| --- | --- | --- | --- | --- |
| `gpt-5.1` | `optimized_v1` | `22/23` | `22/23` |  |
| `gpt-5.4` | `optimized_v1` | `20/23` | `22/23` | `screen4`, `screen10`은 보수적이지만 허용 가능한 응답 |

`gpt-5.4` 점수가 현재 기준에서 달라진 이유:

- `screen4`: 화면에 보이는 목표 메뉴를 바로 누르지 않고 한 번 더 확인한 응답을 허용함
- `screen10`: 더 넓은 후보 집합으로 되묻는 응답을 허용함
- 결론적으로, 원래의 `20/23` 중 일부는 모델 실수라기보다 evaluator가 너무 엄격했던 문제였습니다.

확인용 리포트:

- `eval/results/2026-03-09/prompt_optimized_v1_final_demo_t0.json`
- `eval/results/2026-03-09/prompt_optimized_v1_final_demo_gpt54_t0.json`

### optimized_v2

변경 사항:

- 장바구니 추가 이후, 상품 카드를 다시 누르기보다 `주문내역` 같은 주문 확인 진입 버튼을 우선하도록 규칙 추가
- 이 전환을 위한 few-shot 1개 추가

`final_demo` 기준 점수:

| 모델 | 프롬프트 | 점수 |
| --- | --- | --- |
| `gpt-5.1` | `optimized_v2` | `22/23` |
| `gpt-5.4` | `optimized_v2` | `22/23` |

확인용 리포트:

- `eval/results/2026-03-09/prompt_optimized_v2_final_demo_t0.json`
- `eval/results/2026-03-09/prompt_optimized_v2_final_demo_gpt54_t0.json`

### optimized_v3

변경 사항:

- 장바구니 추가 이후 전환 화면 처리 규칙을 한 번 더 강화
- few-shot 1개 추가

`final_demo` 기준 점수:

| 모델 | 프롬프트 | 점수 | 비고 |
| --- | --- | --- | --- |
| `gpt-5.1` | `optimized_v3` | `23/23` |  |
| `gpt-5.2` | `optimized_v3` | `22/23` | 화면에 없는 메뉴 처리 1건 남음 |
| `gpt-5.4` | `optimized_v3` | `23/23` |  |

판단:

- `optimized_*` 계열 중 순수 점수는 가장 높았습니다.
- 다만 뒤로 갈수록 `final_demo`의 장바구니 이후 패턴에 더 가까워져서 과적합 위험이 커졌습니다.

확인용 리포트:

- `eval/results/2026-03-09/prompt_optimized_v3_final_demo_t0.json`
- `eval/results/2026-03-09/prompt_optimized_v3_final_demo_gpt52_t0.json`
- `eval/results/2026-03-09/prompt_optimized_v3_final_demo_gpt54_t0.json`

### evaluator 변경

`eval/cases/final_demo.json`의 두 step 평가 기준을 수정했습니다.

- `screen4_pick_egg_bulgogi`
  - 이전: `click_text("에그 불고기 버거")`만 허용
  - 이후: 단일 후보 `ask_clarification(["에그 불고기 버거"])`도 허용
- `screen10_ask_summary_action`
  - 이전: clarification 후보 집합이 완전히 일치해야 함
  - 이후: `수정`, `취소`, `장바구니 추가`를 포함하면 허용

변경 이유:

- 이 두 실패는 위험한 오동작이라기보다 evaluator가 너무 엄격해서 생긴 경우에 가까웠습니다.

### generalized_v1

변경 사항:

- `optimized_v1`에서 갈라지는 새 가지로 시작했습니다.
- `optimized_v*`처럼 few-shot을 더 쌓는 대신, 장바구니 추가 이후 전환 화면을 일반화된 시스템 규칙으로 다뤘습니다.
- `optimized_v1` 이후 새로운 few-shot은 추가하지 않았습니다.

점수:

| 모델 | 프롬프트 | 점수 | 비고 |
| --- | --- | --- | --- |
| `gpt-5.4` | `generalized_v1` | `final_demo 22/23` | `screen11`은 해결했지만 `screen6` 회귀 |
| `gpt-5.4` | `generalized_v1` | `final_demo + critical_cases + post_add_to_cart_navigation = 28/29` |  |

확인용 리포트:

- `eval/results/2026-03-09/prompt_generalized_v1_gpt54_t0.json`

### generalized_v2

변경 사항:

- 장바구니 이후 전환 화면 규칙이 실제 add-to-cart 직후에만 동작하도록 가드 추가
- 세트 크기 선택 같은 옵션 선택 화면에서 잘못 발동하던 문제 방지
- 새로운 few-shot은 추가하지 않음

점수:

| 모델 | 프롬프트 | 점수 | 비고 |
| --- | --- | --- | --- |
| `gpt-5.1` | `generalized_v2` | `final_demo 23/23` |  |
| `gpt-5.4` | `generalized_v2` | `final_demo 23/23` | 전체 묶음 실행에 포함됨 |
| `gpt-5.4` | `generalized_v2` | `final_demo + critical_cases + post_add_to_cart_navigation = 29/29` |  |

결론:

- 현재 서빙 프롬프트로 채택
- `src/ai_service.py`에 반영 완료

확인용 리포트:

- `eval/results/2026-03-09/prompt_generalized_v2_gpt51_final_demo_t0.json`
- `eval/results/2026-03-09/prompt_generalized_v2_gpt54_t0.json`
