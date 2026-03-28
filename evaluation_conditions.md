# OGBench 공식 Evaluation 조건 (ICLR 2025)

> 출처: `../ogbench/ogbench.pdf` (Seohong Park et al., ICLR 2025)
> 해당 조건들은 논문 Appendix E.4 및 Section 4, Table 2 캡션, Table 7, Table 9에 명시됨.

---

## 1. 에피소드 수 (Rollouts per Goal)

- **공식 기준: 태스크당 goal별 50 rollouts**
- 총 rollout 수 = 3 eval epochs × 5 goals × 50 rollouts = **750 rollouts per task**
- 저자 허용 완화 기준: 계산 부담을 줄이기 위해 **goal당 20 rollouts**까지 허용 (Appendix E.4)

---

## 2. Seeds 수 & 통계 보고 방식

- **State-based tasks: 8 seeds**
- **Pixel-based tasks: 4 seeds**
- 보고 형식: **mean ± standard deviation** (표준오차나 95% CI가 아님)
- Figure 2 집계 레벨에서만 **95% bootstrap confidence interval** 사용
- 비교 기준: 행(row)에서 최고값의 **95% 이상**인 값을 bold 처리

---

## 3. 평가할 Task 범위

- 각 환경당 **5개의 사전 정의된 state-goal 쌍** (task1 ~ task5)
- 벤치마크 공식 비교: **5개 task 전부** 평가 필수
- 환경별 "default 단일 task" (Table 9):
  - Maze 계열 (pointmaze, antmaze, humanoidmaze, visual-antmaze 등): **task1**
  - antsoccer: **task4**
  - cube/scene: **task2**
  - puzzle-3x3/4x4: **task4**, puzzle-4x5/4x6: **task2**
- **단일 goal 평가 금지**: 논문 Table 4에서 단일 goal 평가가 메소드 랭킹을 역전시킬 수 있음을 명시적으로 경고

---

## 4. Success Rate 정의

- **Binary success**: 성공(1) / 실패(0), 부분 점수 없음
- 환경별 기준:
  - **Locomotion (AntMaze, PointMaze 등)**: agent의 (x,y) 위치와 goal 위치 간 거리만 기준. 관절 각도 무관
  - **Manipulation (Cube 등)**: 물체 위치 기준. 로봇 팔 자세, 물체 방향 무관
  - **Powderworld**: 픽셀 매칭 (1픽셀 shift 허용 tolerance), 불일치 픽셀 수가 임계값 이하이면 성공

---

## 5. 에피소드 길이 (최대 스텝 수)

| 환경 | 최대 스텝 |
|---|---|
| pointmaze-* | 1000 |
| antmaze-medium/large/teleport | 1000 |
| antmaze-giant | 1000 (데이터셋 에피소드는 2000) |
| humanoidmaze-medium/large | 2000 |
| humanoidmaze-giant | 4000 |
| antsoccer-* | 1000 |
| visual-antmaze-* | 1000 |
| cube-single | 200 |
| cube-double | 500 |
| cube-triple/quadruple | 1000 |
| scene | 750 |
| puzzle-3x3/4x4 | 500 |
| puzzle-4x5/4x6 | 1000 |
| powderworld-* | 500 |

- **Goal 도달 시 에피소드 즉시 종료** (early termination on success)

---

## 6. Overall Success Rate 계산 방식

```
task_success = mean over 5 goals (각 goal의 50 rollout binary success 평균)
overall = mean over all tasks (task별 task_success 평균)
```

- **에피소드 수 기준이 아닌 task 수 기준으로 평균** (태스크마다 rollout 수가 동일하므로 동치이나 의미상 task 평균)
- 단일 "OGBench 전체 점수"는 없음 — 데이터셋(환경)별로 보고

---

## 7. 환경 Reset / 초기화 조건

- **Goal은 사전 정의된 5개** (Appendix F, Figures 4~10)
- 각 rollout마다 초기 상태와 goal 상태에 **약간의 랜덤 perturbation** 적용
- 사전 정의된 state-goal 쌍 근방에서 randomized reset

---

## 8. Action 설정 (Temperature / Noise)

- **Continuous-action 정책**: 평가 시 학습된 Gaussian 정책의 **deterministic mean** 사용 (noise 없음)
- **Powderworld (discrete)**: temperature = **0.3** (logit을 0.3으로 나눔) — stuckness 방지용
- 학습 데이터 수집 시: pointmaze σ=0.5, explore 데이터셋 σ=1.0, 나머지 σ=0.2

---

## 9. 최종 점수 계산 방식 (중요)

- 학습 중 **100K gradient step마다** 평가
- 최종 보고 점수 = **마지막 3개 evaluation epoch의 평균**
  - State-based: step 800K, 900K, 1M의 평균
  - Pixel-based: step 300K, 400K, 500K의 평균
- **epoch 최대값(max-over-epochs) 사용 금지** (Zheng et al. 2024b 방식과 명시적으로 구분, 논문 각주 7)

---

## 10. 기타 조건

- Discount factor γ: antmaze-giant, humanoidmaze 계열은 **0.995**, 나머지 **0.99**
- 하이퍼파라미터: value-learning 파라미터는 환경 전체에 동일 적용, policy-extraction 파라미터만 환경 카테고리별 조정
- 훈련 스텝: state-based **1M steps**, pixel-based **500K steps**

---

# eval_all.sh 기준 충족 여부 평가

| 공식 조건 | eval_all.sh 현황 | 충족 여부 |
|---|---|---|
| **goal당 50 rollouts** | `NUM_SEEDS=3` (seed당 1 에피소드 = 총 3 rollouts/task) | ❌ 부족 (50 필요, 3만 실행) |
| **8 seeds (state-based)** | `NUM_SEEDS=3` | ❌ 부족 |
| **5개 task 전부 평가** | `NUM_TASKS`=yaml에서 읽음(5), `START_TASK_IDX=1` | ✅ |
| **Binary success rate** | df_planning.py의 `success_rate` 메트릭 | ✅ (구조상 동일) |
| **Mean ± std 보고** | `collect_eval_results.py`는 mean만 출력, std 없음 | ❌ std 없음 |
| **Goal당 rollout 분리 집계** | seed를 단순 반복으로 처리, goal별 분리 없음 | ❌ goal 5개 분리 없음 |
| **마지막 3 epoch 평균** | 단일 체크포인트 평가 | ❌ 단일 ckpt만 사용 |
| **max-over-epochs 금지** | 해당 없음 (단일 ckpt) | △ 해당 없음 |
| **deterministic mean policy** | 알고리즘 내부 설정에 따름 | △ 확인 필요 |
| **에피소드 길이 1000 (antmaze-giant)** | `val_max_loops=16` × `open_loop_horizon` 설정에 따름 | △ 설정 의존 |

## 요약

**eval_all.sh는 OGBench 공식 evaluation 기준을 충족하지 않습니다.**

가장 큰 미충족 항목:
1. **Rollout 수 절대 부족**: 공식 50 rollouts/goal vs 현재 3 seeds (약 6% 수준)
2. **Goal 분리 없음**: OGBench는 5개 사전 정의된 goal 각각에 대해 50 rollouts를 돌리는 구조인데, MCTD는 task_id로 goal을 선택하고 seed를 반복하는 방식 → goal 5개 × 50 rollouts 구조가 아님
3. **Std 미보고**: 논문 기준은 mean ± std (8 seeds)
4. **마지막 3 epoch 평균 미적용**: 단일 체크포인트만 평가

**실용적 권장사항**: 공식 비교를 위해서는 최소 `NUM_SEEDS=50` (각 seed가 단일 에피소드라면), goal별 분리 로직 추가, std 보고 기능이 필요합니다. 단, MCTD는 알고리즘 특성상 에피소드당 계산 비용이 매우 크므로 저자 허용 완화 기준인 **goal당 20 rollouts × 5 goals × 5 tasks = 500 jobs**부터 시작하는 것이 현실적입니다.
