# Diff 분석: HEAD vs 커밋 `1f6b602`

> **목적**: `1f6b602bf26466578cd19bfc5ff35a6fcfa82d76` 커밋에서 궤적 생성이 더 잘 됐던 이유를 설명하는 모든 코드/설정 차이를 파악한다.

---

## 요약 — 가장 유력한 원인 (우선순위 순)

| 순위 | 파일 | 변경 내용 | 영향 |
|------|------|-----------|------|
| 1 | `df_planning.yaml` | `mctd_guidance_scales: [6.0] → [0.0]` | ❌ Goal guidance 완전 비활성화 |
| 2 | `df_planning.yaml` | `anchor_guidance_scale: 10.0 → 0.4` | ❌ Anchor guidance 25× 약화 |
| 3 | `df_planning.py` | scale=None 시 `guidance_fn = lambda x: 0` | ❌ Value 추정 시 guidance 없음 |
| 4 | `df_planning.yaml` | `scheduling_matrix: smooth → causal` | ⚠️ 다른 denoising 순서, 장거리 품질 저하 |
| 5 | `df_planning.py` | `calculate_values`에 `_obs_parent` 제로 토큰 추가 | ⚠️ Value 추정 시 잘못된 conditioning |
| 6 | `df_planning.yaml` | `padding_mode: "same" → "zero"` | ⚠️ Padding 전략 변경 |
| 7 | `guidance.py` | `weighted_loss`: `mean` → `sum/active_count` | ⚠️ Gradient 크기 변화 |
| 8 | `diffusion.py` | `frozen_mask + masked_pred_x_start` 추가 | ✅ 유익한 수정 (HEAD가 더 좋음) |
| 9 | `df_planning.py` | `_child.obs_pos`: `qpos[:2]` → 29D 연결 | ℹ️ 영향 낮음 (use_rollout=True 경로만) |
| 10 | 여러 파일 | `[:2]` → `[:pos_dim]` 일반화 | ✅ 안전, 로직 변화 없음 |

---

## 섹션 1: `configurations/algorithm/df_planning.yaml`

```diff
-mctd_guidance_scales: [6.0]
+mctd_guidance_scales: [0.0]

-padding_mode: "same"
+padding_mode: "zero"

-horizon_scale: 0.15
+horizon_scale: 0.3

-scheduling_matrix: smooth
+scheduling_matrix: "causal"

-anchor_guidance_scale: 10.0
+anchor_guidance_scale: 0.4

-network_size: 128
-attn_heads: 4
-dim_feedforward: 512
+network_size: 256
+attn_heads: 8
+dim_feedforward: 1024
```

### 분석

#### 1-A. `mctd_guidance_scales: [6.0] → [0.0]` ❌ **치명적**

`mctd_guidance_scales`는 `parallel_plan()`에 `guidance_scale` 텐서로 전달되는 파티클별 guidance scale 배열이다. `[6.0]`이면 `guidance.py`의 `goal_guidance()`가 각 생성된 플랜을 목표 방향으로 밀어주는 강한 gradient 신호를 적용한다. `[0.0]`이면 scale 텐서가 전부 0 → `dist_per_batch = 0 * loss ≡ 0` → **목표 방향으로의 gradient 신호가 전혀 없다**.

명시적인 goal guidance 없이도 diffusion 모델은 어느 정도의 궤적을 생성할 수 있어야 한다. 그러나 anchor guidance까지 없는 상황(다음 항목)과 결합되면 denoising 중에 gradient 보정이 문자 그대로 0이 된다. 모델 prior(즉, 전체 학습 궤적들의 평균적 행동)가 완전히 지배하게 된다.

#### 1-B. `anchor_guidance_scale: 10.0 → 0.4` ❌ **치명적**

`anchor_dist_guidance`는 시간적 연속성을 강제한다: 각 세그먼트의 첫 프레임을 이전 세그먼트의 마지막 프레임 쪽으로 끌어당겨 불연속적인 플랜을 방지한다. Scale 10.0은 매 denoising step마다 의미 있는 gradient를 만들어낸다. Scale 0.4는 25× 약하다.

High noise 레벨(t=90~100)에서 모델 prior는 매우 강하고 학습 데이터 평균 쪽으로 수렴한다. Scale 0.4로는 gradient 신호가 prior를 이기기에 너무 약하다 → 플랜 첫 프레임이 obs_parent에서 멀리 벗어남 → **depth=0에서 30유닛 warp 발생**.

#### 1-C. `scheduling_matrix: smooth → causal` ⚠️ **중간**

`smooth` 모드는 모든 플랜 토큰을 동시에 denoising하며 매 step마다 모든 토큰을 업데이트한다. 생성 과정에서 모든 위치가 상호작용하기 때문에 전역적으로 일관된 플랜이 만들어진다.

`causal` 모드는 토큰을 왼쪽에서 오른쪽으로 denoising한다: 앞쪽 토큰이 noise_level=0에 도달한 뒤에야 뒤쪽 토큰이 업데이트된다. 플랜이 국소적으로는 일관될 수 있지만 장거리 일관성을 잃는다. 긴 horizon(`horizon_scale: 0.3` → 300 스텝)에서는 이 문제가 특히 심하다.

#### 1-D. `horizon_scale: 0.15 → 0.3` ⚠️ **중간**

`horizon = episode_len * horizon_scale`. 0.15에서 0.3으로 바뀌면 계획 horizon이 두 배로 늘어난다. 더 긴 horizon은:
- 더 많은 플랜 토큰을 denoising해야 함 (수렴이 더 어려움)
- Guidance 신호가 더 긴 시퀀스에 걸쳐 희석됨
- 약한 guidance(anchor scale 0.4)와 결합되면 모델이 더 쉽게 벗어남

#### 1-E. `padding_mode: "same" → "zero"` ⚠️ **중간**

`"same"` 패딩은 conditioning/패딩 영역에서 마지막 토큰을 반복하여 diffusion 모델에 연속성 단서를 제공한다. `"zero"`는 침묵 프레임을 삽입한다. `"same"`에서는 모델이 prefix에서 obs_parent 토큰이 반복되는 것을 보게 되어 → 플랜이 어디서 시작해야 하는지에 대한 더 강한 conditioning 신호를 받는다.

#### 1-F. `network_size: 128→256, attn_heads: 4→8, dim_feedforward: 512→1024` ⚠️ **체크포인트 불일치**

아키텍처가 9.7M 파라미터에서 훨씬 큰 모델로 바뀐다. 만약 불러오려는 체크포인트가 `network_size=128`로 학습된 것이라면, 이를 `network_size=256` 모델에 불러오면 조용히 실패하거나 쓰레기 출력이 나온다.

**필수 확인**: 불러오는 체크포인트가 yaml에 정의된 아키텍처와 일치하는지 확인할 것. 128-network 체크포인트라면 `network_size: 128`로 되돌려야 한다.

---

## 섹션 2: `algorithms/diffusion_forcing/df_planning.py`

### 2-A. `guidance_scale is None`일 때 `guidance_fn = lambda x: 0` ❌ **치명적**

**Good commit (1f6b602)**:
```python
if guidance_scale is None:
    guidance_scale = self.guidance_scale   # self.guidance_scale (기본값 2.0)로 폴백

guidance_fn = lambda x: guidance.combined_guidance(self, x, goal, horizon, guidance_scale)
```

**현재 HEAD**:
```python
if guidance_scale is None:
    guidance_fn = lambda x: 0              # ← 0을 반환, guidance가 전혀 없음
else:
    guidance_fn = lambda x: guidance.combined_guidance(self, x, goal, horizon, guidance_scale)
```

`parallel_plan()`은 `calculate_values()`에서 `guidance_scale` 인자 **없이** 호출된다. Good commit에서는 `self.guidance_scale`(2.0)으로 폴백하여 전체 `combined_guidance`를 적용했다. 현재 코드에서는 이 분기가 `lambda x: 0`을 반환 → **value 추정이 guidance 없이 플랜을 생성한다**. 모델 prior가 지배한다.

즉, 모든 value 추정값이 unguided 궤적에서 나오는데, 이는 본질적으로 데이터 분포에서의 무작위 샘플이다. 모든 플랜이 무작위이고 warp 검사가 실패하기 때문에 MCTS 트리가 의미 있는 노드를 찾을 수 없다.

### 2-B. `calculate_values`에 `_obs_parent` 제로 토큰 추가 ⚠️ **중간**

**Good commit**:
```python
value_estimation_plans.append(
    torch.cat([_plan_rearranged, _sim_pad], dim=0)
)  # (plan_tokens + sim_pad, 1, fs*c)
```

**현재 HEAD**:
```python
_obs_parent = torch.zeros((1, 1, _plan_rearranged.shape[-1]), device=self.device)
value_estimation_plans.append(
    torch.cat([_obs_parent, _plan_rearranged, _sim_pad], dim=0)
)  # (1 + plan_tokens + sim_pad, 1, fs*c)
```

conditioning obs_parent 토큰으로 **제로 텐서**가 앞에 추가된다. `parallel_plan()` 내부에서 이 제로 토큰이 denoising의 앵커로 사용된다 — diffusion 모델에게 "정규화 좌표계에서 (0,0) 위치에서 시작하라"고 알리는 것인데, 이는 비정규화 공간에서 학습 데이터 평균 위치에 해당한다. 즉, 모든 value 추정 플랜이 실제 `obs_parent` 위치가 아닌 평균 위치에 앵커링된다.

또한 총 텐서 길이가 `n_tokens = plan_tokens + sim_pad`에서 `n_tokens = 1 + plan_tokens + sim_pad`로 늘어나지만 `self.n_tokens`는 고정되어 있다. 즉, `sim_pad_tokens`가 이미 `n_tokens - 1`을 채우도록 계산되었을 텐데, 토큰을 1개 더 추가하면 시퀀스 길이가 `n_tokens + 1`이 되어 **diffusion 모델의 positional encoding에서 범위를 벗어날 가능성**이 있다.

### 2-C. `_child.obs_pos` 할당 (use_rollout=True 경로) ℹ️ **낮음**

**Good commit**:
```python
_child.obs_pos = _new_sim_state["qpos"][:2]   # 2D 위치 (x, y)
```

**현재 HEAD**:
```python
_child.obs_pos = np.concatenate([_new_sim_state["qpos"], _new_sim_state["qvel"]])[:self.observation_dim]
# → antmaze에서 29D 연결
```

`obs_pos`는 warp 검사(`||plan[0] - starts|| > warp_threshold`)에 사용된다. 29D obs_pos를 사용하면 L2 거리에 속도 차원이 포함되는데, 정지 시에는 0이지만 이동 중에는 0이 아니다. 이로 인해 warp 거리가 부풀려지고 false warp 감지가 더 많이 발생할 수 있다. 단, 현재 설정은 `use_rollout=false`이므로 `use_rollout=True` 분기는 실행되지 않아 **현재는 비활성 상태**다.

### 2-D. `[:2]` → `[:self.pos_dim]` / `[:self.observation_dim]` 일반화 ✅ **안전**

많은 `[:2]` 슬라이스가 `[:self.pos_dim]` 또는 `[:self.observation_dim]`으로 대체됐다. `pos_dim=2`인 antmaze에서는 기능적으로 동일하다. 다른 환경을 위한 정확성 일반화다.

---

## 섹션 3: `algorithms/diffusion_forcing/guidance.py`

### 3-A. `weighted_loss`: `mean(dim)` → `sum / active_count` ⚠️ **중간**

**Good commit**:
```python
return (dist * weight).mean(dim=dim)
```

**현재 HEAD**:
```python
weighted_sum = (dist * weight).sum(dim=dim)
active_count = (weight > 0).float().sum(dim=dim).clamp(min=1)
return weighted_sum / active_count
```

**의도**: Guidance 신호 희석 방지. `mean`을 사용하면 T=100 토큰 중 active가 3개뿐인 경우(앵커 타겟), 유효 gradient가 33× 희석된다. 이 수정은 `total=100` 대신 `active_count=3`으로 나누어 active 위치당 gradient를 33× 강하게 만든다.

**실효 guidance 크기에 미치는 영향**: `anchor_guidance_scale: 10.0 → 0.4`와 `weighted_loss` 변경(`mean/100` → `sum/3`)을 함께 고려하면:
- Good commit 유효 gradient: `10.0 * (dist / 100)` = `0.10 * dist`
- 현재 유효 gradient: `0.4 * (dist / 3)` = `0.133 * dist`
- 결론: 현재가 active 프레임당 약 **1.3× 강하다** (단, goal guidance = 0이므로 별 의미 없음)

`mctd_guidance_scales: [0.0]`이고 `rdf_guidance_scale: 0.0`이라 `anchor_guidance_scale=0.4`만 활성화되어 있다. `weighted_loss` 변경이 25× 스케일 감소를 부분적으로 보상하지만 완전하지는 않다.

### 3-B. `dist_target = dist_hilp + dist_mse` → `dist_rmse` ⚠️ **중간**

**Good commit**:
```python
dist_target = dist_hilp + dist_mse   # HILP value function + MSE 거리
```

**현재 HEAD**:
```python
dist_rmse = torch.sqrt(dist_mse)
dist_target = dist_rmse              # RMSE만 (HILP 주석 처리)
```

Good commit은 HILP value function 거리(스킬 표현 기반으로 학습된 장거리 거리 메트릭)와 유클리드 MSE를 결합했다. HILP는 미로 토폴로지를 존중하는 **비유클리드 거리**를 제공한다 — 벽을 돌아가는 것을 올바르게 처리한다. RMSE는 벽을 무시하는 직선 거리만 측정한다.

단, `mctd_guidance_scales: [0.0]`이므로 이 함수는 호출되지 않는다(goal_guidance가 0을 반환). 이 차이는 **현재 비활성** 상태이지만 goal guidance를 다시 활성화하면 중요해진다.

### 3-C. `ignore_latest = 3*frame_stack → 6*frame_stack` (`segment_rdf_guidance`) ℹ️ **비활성**

RDF 반발 무시 윈도우가 두 배로 늘었다. `rdf_guidance_scale: 0.0`이므로 비활성 상태다.

---

## 섹션 4: `algorithms/diffusion_forcing/models/diffusion.py`

```diff
+                frozen_mask = self.add_shape_channels(curr_noise_level == next_noise_level)
+                masked_pred_x_start = torch.where(frozen_mask, orig_x, model_pred.pred_x_start)
+                guidance_results = guidance_fn(masked_pred_x_start)
-                guidance_results = guidance_fn(model_pred.pred_x_start)
```

**이 부분은 HEAD가 더 좋다.** Good commit은 원시 `model_pred.pred_x_start`를 guidance 함수에 전달했다. 고정된(frozen) 토큰(obs_parent, noise_level=0)의 경우, 모델이 오염된 `pred_x_start`를 생성할 수 있다(노이즈가 있는 컨텍스트를 기반으로 예측하기 때문). 현재 코드는 고정 토큰 예측을 실제 `orig_x`로 교체하여:
1. 앵커 타겟이 정확히 실제 obs_parent 프레임임을 보장
2. 고정 토큰에 대한 gradient가 0임을 보장 (frozen = 이미 완전히 denoised)

추가로, `allow_unused=True` + null guard로 guidance가 gradient를 생성하지 않을 때(예: scale=0) 크래시를 방지한다.

---

## 섹션 5: 요약 표

| # | 위치 | Good Commit (`1f6b602`) | 현재 HEAD | 영향 | 비고 |
|---|------|------------------------|-----------|------|------|
| 1 | yaml | `mctd_guidance_scales: [6.0]` | `[0.0]` | ❌ 치명적 | Goal guidance OFF |
| 2 | yaml | `anchor_guidance_scale: 10.0` | `0.4` | ❌ 치명적 | 25× 약화 |
| 3 | df_planning.py | None 시 `guidance_scale = self.guidance_scale` | None 시 `lambda x: 0` | ❌ 치명적 | Value 추정 unguided |
| 4 | yaml | `scheduling_matrix: smooth` | `causal` | ⚠️ 중간 | 다른 생성 전략 |
| 5 | yaml | `padding_mode: "same"` | `"zero"` | ⚠️ 중간 | 약한 conditioning |
| 6 | yaml | `horizon_scale: 0.15` | `0.3` | ⚠️ 중간 | 길수록 어렵다 |
| 7 | df_planning.py | `_obs_parent` 제로 토큰 없음 | 제로 토큰 추가 | ⚠️ 중간 | Value 추정 앵커 오류 |
| 8 | guidance.py | `mean(dim)` | `sum/active_count` | ⚠️ 중간 | 프레임당 33× 강화 |
| 9 | guidance.py | `dist_hilp + dist_mse` | `dist_rmse`만 | ⚠️ 중간 | 현재 비활성 |
| 10 | yaml | `network_size: 128` | `256` | ⚠️ 체크포인트 | 불러온 ckpt와 일치해야 함 |
| 11 | diffusion.py | frozen_mask 없음 | frozen_mask + masked | ✅ 더 좋음 | HEAD가 정확함 |
| 12 | df_planning.py | obs_pos에 `qpos[:2]` | 29D 연결 | ℹ️ 낮음 | use_rollout=True만 |
| 13 | 여러 파일 | `[:2]` | `[:pos_dim]` | ✅ 안전 | 일반화 |

---

## 권장 수정 사항 (good commit 동작 복원용)

관련 없는 개선 사항은 되돌리지 않고 궤적 품질을 복원하려면:

### 수정 1 (필수): yaml에서 guidance scale 복원
```yaml
mctd_guidance_scales: [6.0]
anchor_guidance_scale: 10.0
```

### 수정 2 (필수): `df_planning.py`에서 guidance_fn 폴백 복원
```python
# Good commit 기준 약 795번째 줄 / HEAD 기준 약 800번째 줄
if guidance_scale is None:
    guidance_scale = self.guidance_scale   # ← 이 부분 복원
# 삭제: guidance_fn = lambda x: 0
guidance_fn = lambda x: guidance.combined_guidance(self, x, goal, horizon, guidance_scale)
```

### 수정 3 (중요): `calculate_values`에서 불필요한 `_obs_parent` 제로 토큰 제거
```python
value_estimation_plans.append(
    torch.cat([_plan_rearranged, _sim_pad], dim=0)   # ← 원본, _obs_parent 없음
)
```

### 수정 4 (중요): scheduling_matrix와 padding_mode 복원
```yaml
scheduling_matrix: smooth
padding_mode: "same"
```

### 수정 5 (확인 필요): 아키텍처가 체크포인트와 일치해야 함
`network_size=128`로 학습된 체크포인트로 검증하는 경우:
```yaml
architecture:
  network_size: 128
  attn_heads: 4
  dim_feedforward: 512
```

---

*`git diff 1f6b602bf26466578cd19bfc5ff35a6fcfa82d76 HEAD`로 생성됨*
