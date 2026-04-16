# OGBench 공식 Evaluation 조건 정리

> 출처: `../ogbench/ogbench.pdf` (Seohong Park et al., ICLR 2025)
>
> 이 문서는 다음을 분리해서 정리한다.
> 1. 논문/공식 구현이 직접 말하는 평가 조건
> 2. 현재 저장소의 `eval.sh`가 그 조건을 어디까지 만족하는지
> 3. OGBench가 제안하는 프로토콜에 더 가깝게 맞추기 위해 `eval.sh`와 하위 평가 파이프라인에서 무엇을 바꿔야 하는지

---

## 0. 용어 정리

이 문서에서 가장 중요한 것은 아래 네 개를 섞지 않는 것이다.

- **evaluation goal**: OGBench가 각 환경마다 제공하는 사전 정의된 5개의 test-time goal 중 하나
- **task_id**: 이 저장소에서 OGBench evaluation goal에 대응하는 식별자 (`task1` ~ `task5`)
- **rollout**: 하나의 고정된 `task_id` 아래에서 실행되는 **한 번의 evaluation episode**
- **evaluation epoch**: 학습 중 특정 step에서 수행되는 주기적 평가 1회
- **training seed**: 서로 독립적으로 학습된 모델/run 1개

즉:

- `task_id`는 **goal 축**
- rollout 수는 **episode 반복 축**
- evaluation epoch는 **checkpoint 축**
- training seed는 **독립 학습 run 축**

이 네 축은 서로 대체되지 않는다.

---

## 1. 논문에 명시된 공식 평가 조건

### 1.1 Multi-goal evaluation

- OGBench의 각 task/environment는 **5개의 evaluation goal**을 가진다.
- 논문은 **single-goal evaluation을 쓰지 말라**고 명시적으로 경고한다.
- Table 4의 핵심 메시지는 다음과 같다.
  - single-goal evaluation은 offline GCRL method 간 순위를 왜곡할 수 있다.
  - OGBench는 이를 피하기 위해 **multi-goal evaluation**을 사용한다.

참고:

- `Table 9`의 default task는 **singletask variant**용이다.
- 즉 `task1`, `task2` 같은 default task 표는 "goal-conditioned main benchmark에서 5개 goal 평가를 대체하는 규칙"이 아니다.

근거:

- Table 4, p.13
- Table 9, p.31
- Appendix F, p.34

---

### 1.2 Rollouts per goal

- 공식 benchmark에서는 **각 test-time goal당 50 rollouts**를 수행한다.
- 따라서 한 checkpoint/model에 대해:
  - `5 goals × 50 rollouts = 250 rollouts / evaluation epoch`
- 마지막 3개 evaluation epoch 평균까지 포함하면:
  - `3 epochs × 5 goals × 50 rollouts = 750 rollouts / task`

논문은 계산량 완화를 위해 아래와 같은 축소 버전도 허용한다.

- **각 goal당 20 episodes**로 줄이는 것 가능
- 다만 이 경우는 **reduced protocol**로 명시하는 편이 정확하다.

근거:

- Appendix E.4, p.27

---

### 1.3 Seeds와 보고 방식

- **State-based tasks**: 8 training seeds
- **Pixel-based tasks**: 4 training seeds
- 보고 형식: **mean ± standard deviation**
- Figure 2 같은 category-level aggregate plot에서만 **95% bootstrap confidence interval** 사용
- Table 2에서는 row-best의 **95% 이상 값**을 bold 처리

중요:

- 여기서의 `8 seeds / 4 seeds`는 **평가 rollout seed가 아니라 독립 학습 run 수**다.
- 따라서 rollout을 50번 반복하는 것으로 training seed 조건을 만족했다고 볼 수 없다.

근거:

- Table 2, p.11
- Figure 2 caption, p.12

---

### 1.4 최종 점수 산정

- 학습 중 **100K gradient steps마다 평가**
- 최종 보고 점수는 **마지막 3개 evaluation epoch 평균**
- State-based:
  - 800K, 900K, 1M step 평균
- Pixel-based:
  - 300K, 400K, 500K step 평균
- **max-over-epochs** 방식은 쓰지 않는다.

근거:

- Appendix E.4, p.27
- 각주 7 관련 문맥, p.12-13

---

### 1.5 Success 정의

- 성공 여부는 **binary success**다.
- 부분 점수 기반 metric이 아니다.

환경별 성공 기준:

- **Locomotion**
  - agent 또는 ball과 goal 위치 간 거리만 기준
  - joint pose는 성공 판정에 포함되지 않음
- **Manipulation**
  - object configuration 기준
  - arm pose는 성공 판정에 포함되지 않음
  - cube는 위치만 보고 orientation은 무시
- **Powderworld**
  - goal image와 현재 image의 pixel 매칭
  - 1-pixel shift tolerance 허용
  - 불일치 pixel 수가 threshold 이하이면 성공

근거:

- p.25-26

---

### 1.6 Reset / randomized evaluation

- 각 predefined state-goal pair에 대해 **multiple rollouts**
- 각 rollout은 **slightly randomized initial state와 goal state**를 사용
- 즉 benchmark는 같은 `task_id`라도 rollout마다 미세하게 다른 reset이 들어간다.

근거:

- p.6
- Appendix F, p.34

---

### 1.7 Episode termination과 최대 길이

- **Goal에 도달하면 episode는 즉시 종료**된다.
- 환경별 최대 episode length는 Table 7 기준을 따른다.

| 환경 | 최대 episode length |
|---|---|
| `pointmaze-*` | 1000 |
| `antmaze-*` | 1000 |
| `humanoidmaze-medium/large` | 2000 |
| `humanoidmaze-giant` | 4000 |
| `antsoccer-*` | 1000 |
| `visual-antmaze-*` | 1000 |
| `visual-humanoidmaze-medium/large` | 2000 |
| `visual-humanoidmaze-giant` | 4000 |
| `cube-single` | 200 |
| `cube-double` | 500 |
| `cube-triple/quadruple` | 1000 |
| `scene` | 750 |
| `puzzle-3x3/4x4` | 500 |
| `puzzle-4x5/4x6` | 1000 |
| `powderworld-*` | 500 |

주의:

- `Table 8`의 dataset episode length와 `Table 7`의 environment maximum episode length는 다를 수 있다.
- 예를 들어 `antmaze-giant-stitch-v0`의 dataset episode length는 200이지만, 환경의 최대 episode length는 1000이다.

근거:

- Table 7, p.29
- Table 8, p.30

---

### 1.8 평가 시 action / policy convention

- 논문의 reference implementation에서는 continuous-action policy를 평가할 때 **learned Gaussian policy의 deterministic mean**을 사용한다.
- Powderworld는 discrete action space이므로 **temperature = 0.3**의 stochastic policy를 사용한다.

주의:

- 이 조항은 논문이 제공하는 JAX baselines에 대한 설명이다.
- MCTD처럼 다른 policy/executor 구조를 쓰는 방법에서는 완전히 동일하게 대응되지 않을 수 있다.
- 다만 benchmark 비교를 하려면, **evaluation-time stochasticity를 어떻게 다뤘는지 반드시 명시**하는 것이 맞다.

근거:

- Appendix E.4, p.27

---

### 1.9 기타 benchmark 문맥

- Discount factor:
  - `0.995`: `{antmaze, pointmaze}-giant`, `humanoidmaze`
  - `0.99`: others
- State-based는 보통 1M gradient steps
- Pixel-based는 보통 500K gradient steps

이 항목들은 주로 **훈련 설정**에 관한 것이며, `eval.sh`의 직접 책임은 아니다.

근거:

- Table 10, p.32

---

## 2. 현재 저장소에서 `eval.sh`가 의미하는 것

현재 구현을 OGBench 용어로 매핑하면 다음과 같다.

- `algorithm.task_id`
  - OGBench의 **evaluation goal 1개**에 해당
- `NUM_TASKS`
  - 평가할 goal 수
- `NUM_SEEDS`
  - 이 저장소에서는 사실상 **같은 checkpoint에 대한 반복 evaluation job 수**
  - 논문이 말하는 **training seed 수**와는 다르다
- `experiment.tasks=["validation"]`
  - validation loader를 통해 evaluation 진입
- `experiment.validation.batch_size=1`
  - 현재 의도상 **한 job = 한 rollout**으로 쓰려는 구조

중요한 해석:

- 현재 `eval.sh`는 **single checkpoint 평가 런처**다.
- 따라서 구조적으로 다음 두 가지는 애초에 만족할 수 없다.
  - 마지막 3개 evaluation epoch 평균
  - 8개 independent training seeds 평균 ± std

---

## 3. `eval.sh` 기준 현재 충족 여부

여기서는 **현재의 `eval.sh`만 기준**으로 본다.  
`eval_all.sh` 기준 평가는 이 문서에서 다루지 않는다.

| 공식 조건 | 현재 `eval.sh` 상태 | 판단 | 설명 |
|---|---|---|---|
| 5개 evaluation goal 사용 | `NUM_TASKS=5` | △ | 현재 `START_TASK_IDX=2`이지만 generator가 modulo wrap-around를 하므로 결과적으로 `task1~task5`를 한 번씩 돈다. 다만 의미가 불분명하므로 `1`로 두는 편이 맞다. |
| multi-goal evaluation | goal 5개 평가 의도 | ✅ | `task_id`가 goal 축으로 작동한다. |
| goal당 50 rollouts | `NUM_SEEDS=1` | ❌ | 현재는 goal당 1 rollout 수준이다. |
| goal당 20 rollouts 완화 기준 | 미충족 | ❌ | 공식 완화 기준조차 만족하지 못한다. |
| state-based 8 training seeds | 단일 checkpoint 선택 | ❌ | rollout 반복과 training seed는 다른 축이다. |
| last 3 evaluation epochs 평균 | 단일 checkpoint 평가 | ❌ | 현재 구조상 한 checkpoint만 평가한다. |
| mean ± std over training seeds | 없음 | ❌ | 현재는 training-seed 축 aggregate가 없다. |
| randomized reset | OGBench env가 담당 | △ | 환경은 지원하지만, 현재 pipeline이 rollout/episode 경계를 정확히 지켜야 의미가 있다. |
| goal 도달 시 즉시 종료 | OGBench env가 담당 | △ | env는 그렇게 동작하지만, 상위 loop가 `done`을 어떻게 처리하는지가 중요하다. |
| 환경 최대 episode length 준수 | 불명확 | △ | env 내부 horizon은 있으나 현재 executor loop가 `done` 이후를 어떻게 처리하는지 확인이 필요하다. |
| binary success | `success_rate` 사용 | ✅ | 구조상 binary success 집계다. |
| default task 규칙 사용 | 해당 없음 | N/A | main multi-goal benchmark에는 필수 규칙이 아니다. |

---

## 4. `eval.sh`만 봤을 때 꼭 추가 충족되어야 하는 것

아래 항목이 `eval.sh` 기준의 핵심 부족분이다.

### 4.1 Goal당 rollout 수

현재:

- `NUM_SEEDS=1`

공식 benchmark에 맞추려면:

- **공식 프로토콜**: `50 rollouts / goal`
- **완화 프로토콜**: `20 rollouts / goal`

즉 `eval.sh`는 최소한 다음 둘 중 하나를 명시적으로 지원해야 한다.

- `official`: 50
- `reduced`: 20

그리고 현재의 `NUM_SEEDS`라는 이름은 오해를 만든다.  
이 값은 training seed가 아니라 rollout 반복 수이므로 이름을 바꾸는 것이 낫다.

권장:

- `NUM_SEEDS` → `NUM_ROLLOUTS_PER_GOAL`

---

### 4.2 마지막 3개 evaluation epoch 평균

현재:

- 사용자가 고른 **단일 checkpoint**만 평가

공식 benchmark에 맞추려면:

- state-based는 `800K`, `900K`, `1M`
- pixel-based는 `300K`, `400K`, `500K`

즉 `eval.sh`는 단일 checkpoint 런처가 아니라, 아래 중 하나가 되어야 한다.

- 동일 training run의 **마지막 3개 benchmark checkpoint를 자동 선택**
- 또는 사용자가 **3개 checkpoint를 명시적으로 고르게 함**

그 뒤 결과를 아래 순서로 평균해야 한다.

```text
rollout 평균
-> goal(task_id) 평균
-> 3 checkpoint 평균
```

---

### 4.3 8개 independent training seeds 평균 ± std

현재:

- `interaction_seed`나 반복 job 수는 있어도
- **독립적으로 학습된 8개 model/run을 aggregate**하는 구조는 없다.

공식 benchmark에 맞추려면:

- state-based: 8개의 독립 training run
- pixel-based: 4개의 독립 training run

즉 `eval.sh`는 장기적으로 아래 구조를 지원해야 한다.

- `MODEL_IDS=(seed1 seed2 ... seed8)` 또는
- 동일 experiment group에서 seed별 model을 자동 탐색

그리고 최종 보고는:

```text
각 training seed의 final score
= (마지막 3 checkpoints 평균)

최종 benchmark score
= training seeds에 대한 mean ± std
```

---

### 4.4 `task_id` 시작값의 명시성

현재:

- `START_TASK_IDX=2`

실제로는 generator의 modulo wrap-around 때문에 `NUM_TASKS=5`일 때 `task1~task5`를 모두 돌 수 있다.  
하지만 benchmark 문서 관점에서는 이 설정이 불필요하게 헷갈린다.

권장:

- `START_TASK_IDX=1`

이건 correctness보다는 **명시성 문제**다.

---

## 5. `eval.sh`만 바꿔서는 안 되고, 하위 평가 파이프라인도 같이 수정해야 하는 부분

아래는 엄밀히 말해 `eval.sh` 파일 내부만의 문제는 아니다.  
하지만 `eval.sh`가 benchmark 프로토콜을 따르려면 결국 함께 수정되어야 한다.

### 5.1 rollout seed가 실제 OGBench env reset에 연결되어야 함

현재 파이프라인에서는 `interaction_seed`가 job config로 들어가지만, OGBench branch에서 그 seed가 environment reset randomness에 직접 연결되지 않는 경로가 있다.

의미:

- `NUM_ROLLOUTS_PER_GOAL=50`으로 늘려도
- 실제로는 재현 가능한 50개의 distinct rollout이 되지 않을 수 있다.

필요한 수정 방향:

- OGBench env 생성/초기화 시 `interaction_seed`를 명시적으로 적용
- rollout index마다 고유 seed를 넣고, 그 seed가 reset randomization에 실제 반영되게 보장

---

### 5.2 한 job이 정확히 한 episode/rollout이어야 함

OGBench 정의에서 rollout은 **한 episode**다.  
그런데 현재 파이프라인은 이 episode 경계를 흐릴 위험이 있다.

핵심 이유:

- `DummyVecEnv`는 `done=True`일 때 **자동 reset**한다.
- 따라서 상위 loop가 `done`을 episode boundary로 강하게 종료하지 않으면
- 하나의 evaluation run 안에 **여러 episode가 이어붙는 구조**가 될 수 있다.

이 경우 아래 조건이 함께 깨진다.

- rollout 수 정의
- episode length 정의
- randomized reset 의미

필요한 수정 방향:

- `done` 또는 `truncated`가 발생한 순간 **그 rollout 전체를 종료**
- auto-reset 이후의 다음 episode를 같은 run의 일부로 쓰지 않기
- 필요하면 `terminal_observation`를 명시적으로 사용

이 항목은 우선순위가 높다.

---

### 5.3 결과 수집기가 benchmark 축을 올바르게 구분해야 함

현재 collector는 대체로 `task_id -> success list` 수준의 평균만 다룬다.  
공식 benchmark에 맞추려면 적어도 아래 축을 분리해야 한다.

- rollout index
- goal / `task_id`
- checkpoint epoch
- training seed

최종 집계 순서 권장:

```text
1. rollout 평균 -> goal success
2. 5 goals 평균 -> checkpoint score
3. 마지막 3 checkpoint 평균 -> one training-seed final score
4. 8 training-seed mean ± std -> benchmark report
```

---

### 5.4 benchmark deviation은 명시적으로 기록해야 함

MCTD는 OGBench reference baseline과 다른 evaluation executor를 쓴다.  
따라서 아래 항목 중 하나라도 다르면 결과 표기에서 분리해야 한다.

- 50이 아닌 20 rollouts 사용
- 단일 checkpoint만 사용
- 8 training seeds 대신 1개 model만 사용
- evaluation-time stochasticity 처리 방식이 다름

권장 표기:

- `OGBench-official`
- `OGBench-reduced-20rollouts`
- `single-ckpt exploratory eval`

처럼 protocol 이름을 명확히 나누기

---

## 6. `eval.sh`에 대해 권장하는 구체 수정안

이 절은 실제 구현 작업을 염두에 둔 제안이다.

### 6.1 `eval.sh`에서 직접 바꿀 것

1. `START_TASK_IDX=1`로 변경
2. `NUM_SEEDS`를 `NUM_ROLLOUTS_PER_GOAL`로 변경
3. rollout 모드를 명시적으로 선택 가능하게 추가
   - `official=50`
   - `reduced=20`
4. 단일 checkpoint 선택 대신 `LAST_3_CKPTS` 선택 로직 추가
5. 장기적으로 `MODEL_IDS` 배열 또는 run-group 기반 multi-seed 평가 진입점 추가

예시 개념:

```bash
ROLLOUT_MODE=official        # official | reduced
NUM_ROLLOUTS_PER_GOAL=50     # reduced면 20
START_TASK_IDX=1
CHECKPOINT_MODE=last3        # single | last3
TRAINING_SEED_MODE=single    # single | multi
```

---

### 6.2 `eval.sh`가 호출하는 하위 코드에서 같이 바꿔야 할 것

1. OGBench env reset에 rollout seed를 실제로 연결
2. `done/truncated` 즉시 rollout 종료
3. collector가 `goal -> checkpoint -> training seed` 축으로 aggregate
4. 결과 출력 형식을 `mean ± std`까지 확장
5. reduced protocol 사용 시 결과 라벨에 명시

---

## 7. 최종 요약

현재 `eval.sh`는 다음 목적에는 적합하다.

- 특정 checkpoint를 빠르게 sanity-check
- 5개 OGBench goal을 한 번씩 돌아보는 exploratory eval

하지만 다음 이유로 OGBench 논문 표와 직접 비교 가능한 공식 평가 런처라고 보기는 어렵다.

- goal당 rollout 수가 부족함
- 마지막 3 checkpoints 평균이 없음
- 8 independent training seeds 평균 ± std가 없음
- rollout seed가 실제 env reset randomness에 확실히 연결되어 있는지 불명확함
- 한 run이 정확히 한 episode인지 보장하는 정리가 필요함

따라서 `eval.sh`를 benchmark 지향적으로 고치려면, 최소한 아래 네 가지를 만족시키는 방향으로 바꿔야 한다.

1. **5 goals × 50 rollouts** 또는 최소한 **5 goals × 20 rollouts**
2. **마지막 3 evaluation checkpoints 평균**
3. **8 independent training seeds 평균 ± std**
4. **한 rollout = 한 episode 보장**

이 네 가지가 갖춰져야 OGBench가 제안하는 평가 프로토콜에 실질적으로 가까워진다.
