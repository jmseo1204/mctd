# Uncertainty-Guided Tree Expansion: 이론적 배경 및 구현 명세

## 1. 문제 배경

### 1.1 전체 시스템 개요

본 시스템은 **tree-structured diffusion planner**입니다. Start state에서 goal state까지의 경로를 점진적으로 확장하는 트리 탐색 구조를 사용하며, 각 노드에서 여러 자식으로 분기할 수 있습니다.

- 각 트리 노드는 하나의 state $s$에 대응
- Root 노드는 start state, expansion의 목표는 goal state $s_g$에 도달
- 각 노드에서 diffusion model로 길이 $L$의 sub-plan을 생성하여 자식 노드로 확장
- 자식 노드는 sub-plan의 tail state (마지막 state)

### 1.2 Sub-Plan 생성 메커니즘

Sub-plan 생성은 다음과 같은 SDE 기반 diffusion sampling입니다:

$$d\mathbf{X}_t = \bigl[\mathbf{s}_\theta(\mathbf{X}_t, t) + w\cdot\nabla_{\mathbf{X}} V(\mathbf{X}; s_g)\bigr]\,dt + \sigma(t)\,d\mathbf{W}$$

- $\mathbf{s}_\theta$: 학습된 score function (training trajectory 분포로부터)
- $V(s; s_g) = -\|\phi(s) - \phi(s_g)\|$: goal guidance potential
- 전체는 길이 $L$의 state 시퀀스를 한 번에 생성 (joint distribution의 score)

### 1.3 Temporal Distance 보존 임베딩

별도로 학습된 encoder $\phi: \mathcal{S} \to \mathbb{R}^k$가 있으며, 이는 다음 성질을 보장:

$$\|\phi(s_1) - \phi(s_2)\| \approx T(s_1, s_2)$$

여기서 $T(s_1, s_2)$는 두 state 사이의 temporal distance (goal까지 도달하는 데 필요한 최소 스텝 수와 유사한 개념). 즉 $\phi$-공간에서의 유클리드 거리가 곧 "도달 시간 거리"입니다.

### 1.4 해결하고자 하는 문제

각 트리 노드에서 **몇 개의 자식으로 분기할지**를 결정해야 합니다. 불확정성이 높은 노드에서는 더 많이 분기하여 다양한 경로를 탐색하고, 불확정성이 낮은 노드에서는 적게 분기하여 계산 자원을 절약합니다.

따라서 **현재 노드의 uncertainty를 정량화하는 지표**가 필요합니다. 이 지표는:

- 현재 노드에서 sub-plan을 $N$번 샘플링한 결과의 tail state들로부터 계산
- Goal까지의 잔여 temporal distance를 반영
- 가우시안 가정 등 강한 분포 가정에 의존하지 않음
- 임베딩 차원이 커도 수치적으로 안정적 ($\det$ 발산 회피)

---

## 2. 이론적 유도

### 2.1 전체 구조: 엔트로피 체인 룰

현재 노드 $s_{\text{curr}}$에서 goal $s_g$까지의 expansion을 마르코프 체인으로 모델링:

$$X_0 = s_{\text{curr}} \;\to\; X_1 \;\to\; X_2 \;\to\; \cdots \;\to\; X_{M_{\text{rem}}} \approx s_g$$

각 $X_m$은 $m$번째 expansion 후의 state, $M_{\text{rem}}$은 goal 도달까지의 총 expansion 횟수.

엔트로피 체인 룰에 의해:

$$H(X_{M_{\text{rem}}} \mid X_0) = \sum_{m=0}^{M_{\text{rem}}-1} H(X_{m+1} \mid X_m)$$

이것은 마르코프 성질 하에서 정확한 등식입니다.

### 2.2 Per-Step 엔트로피 $\ln K$의 정의

각 $H(X_{m+1} \mid X_m)$을 현재 노드의 $N$개 sub-plan tail state의 분포로부터 추정합니다.

**가우시안 근사 없이 상한을 얻는 방법 — 최대 엔트로피 원리:**

고정된 공분산 $\boldsymbol{\Sigma}$를 가진 모든 분포 중에서 가우시안이 미분 엔트로피를 최대화합니다. 즉 임의 분포 $p$에 대해:

$$h(p) \leq \frac{1}{2}\ln\det(2\pi e\,\boldsymbol{\Sigma})$$

이는 가우시안 가정이 아니라 **보수적 상한**으로 작동합니다.

**수치적 안정성 문제와 방향 분해:**

$\phi$-공간의 차원 $k$가 크면, $N$개 샘플의 공분산이 rank-deficient이 되어 $\det \to 0$, $\ln\det \to -\infty$로 발산합니다. 이를 회피하기 위해 **goal 방향과 수직 방향의 2차원 분해**를 사용합니다.

우리 문제에서 의미 있는 두 가지 불확정성은:

- **도착 시점 불확정성** ($\sigma_\parallel$): goal 방향으로 언제 도착하는가
- **경로 불확정성** ($\sigma_\perp$): 서로 다른 경로를 탐색하는가

이 2개 성분만 사용하면 rank 문제가 원천적으로 발생하지 않습니다.

### 2.3 잔여 경로의 효과: $M_{\text{rem}}$ 계수

현재 노드에서 한 번의 expansion의 불확정성이 $\ln K$일 때, goal 도달까지 $M_{\text{rem}}$번의 expansion이 남았다면, 정상성 가정 하에서 총 불확정성은 체인 룰에 의해 $(1 + M_{\text{rem}}) \cdot \ln K$가 됩니다.

$M_{\text{rem}}$은 "goal까지 남은 expansion 횟수"로, $\phi$-공간의 temporal distance 보존 성질을 활용하여 다음과 같이 추정합니다:

$$M_{\text{rem}} = \frac{T_{\text{tail}}}{\bar{\Delta}}$$

- $T_{\text{tail}}$: sub-plan tail에서 goal까지의 평균 잔여 temporal distance
- $\bar{\Delta}$: 한 번의 expansion으로 줄어드는 평균 temporal distance

### 2.4 최종 지표

$$U(s_{\text{curr}}) = \ln K \cdot \left(1 + \frac{T_{\text{tail}}}{\bar{\Delta}}\right)$$

**각 요소의 역할:**

- $\ln K$: 현재 expansion의 local uncertainty (tail state들이 얼마나 퍼져 있는가)
- $1 + T_{\text{tail}}/\bar{\Delta}$: goal까지의 잔여 경로 길이에 따른 누적 효과
- 두 요소의 곱: "goal 근처의 불일관 plan"과 "멀리 있는 일관 plan" 모두 높은 uncertainty로 평가

---

## 3. 수학적 기호와 실제 구현 변수의 대응

### 3.1 입력

| 기호              | 의미                                               | 타입                   |
| ----------------- | -------------------------------------------------- | ---------------------- |
| $s_{\text{curr}}$ | 현재 트리 노드의 state                             | state                  |
| $s_g$             | Goal state                                         | state                  |
| $\{s_i\}_{i=1}^N$ | 현재 노드에서 생성된 $N$개 sub-plan의 tail state들 | state list of size $N$ |
| $\phi(\cdot)$     | Temporal distance 보존 encoder                     | `nn.Module`            |
| $k$               | 임베딩 차원                                        | int                    |

### 3.2 중간 계산값

**Step 1 — 임베딩 계산:**

$$\mathbf{z}_{\text{curr}} = \phi(s_{\text{curr}}) \in \mathbb{R}^k$$
$$\mathbf{z}_g = \phi(s_g) \in \mathbb{R}^k$$
$$\mathbf{z}_i = \phi(s_i) \in \mathbb{R}^k \quad \text{for } i = 1, \dots, N$$

**Step 2 — Sub-plan tail의 평균과 공분산:**

$$\bar{\mathbf{z}} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{z}_i$$

$$\hat{\boldsymbol{\Sigma}}_\phi = \frac{1}{N}\sum_{i=1}^{N}(\mathbf{z}_i - \bar{\mathbf{z}})(\mathbf{z}_i - \bar{\mathbf{z}})^T \in \mathbb{R}^{k \times k}$$

(분모로 $N$ 또는 $N-1$ 모두 가능. $N$이 작으면 $N-1$을 사용하는 것이 unbiased.)

**Step 3 — Goal 방향 단위 벡터:**

$$\hat{\mathbf{g}} = \frac{\mathbf{z}_g - \mathbf{z}_{\text{curr}}}{\|\mathbf{z}_g - \mathbf{z}_{\text{curr}}\|} \in \mathbb{R}^k$$

Goal 방향은 **현재 노드에서 goal로 향하는 방향**으로 정의합니다 (`z_g - z_curr`, 역방향 아님).

**Step 4 — 방향별 분산:**

$$\sigma_\parallel^2 = \hat{\mathbf{g}}^T \hat{\boldsymbol{\Sigma}}_\phi\,\hat{\mathbf{g}} \quad \text{(스칼라)}$$

$$\sigma_\perp^2 = \text{tr}(\hat{\boldsymbol{\Sigma}}_\phi) - \sigma_\parallel^2 \quad \text{(스칼라)}$$

- $\sigma_\parallel^2$: tail state들의 goal 방향 분산 (도착 시점 불확정성)
- $\sigma_\perp^2$: 나머지 모든 방향의 분산 합 (경로 선택 불확정성)
- 수치 안정성을 위해 $\sigma_\parallel^2, \sigma_\perp^2$에 작은 $\epsilon$ (예: $10^{-8}$) 추가

**Step 5 — Per-Step 엔트로피:**

$$\ln K = \frac{1}{2}\ln(2\pi e\,\sigma_\parallel^2) + \frac{1}{2}\ln(2\pi e\,\sigma_\perp^2)$$

이것을 전개하면:

$$\ln K = \ln(2\pi e) + \ln\sigma_\parallel + \ln\sigma_\perp$$

즉 $2\pi e$의 상수와 두 표준편차의 로그합. 구현 시 이 전개된 형태가 더 수치적으로 안정적입니다.

**Step 6 — Temporal Distance 관련 양:**

현재 노드에서 goal까지의 temporal distance:

$$T_{\text{curr}} = \|\mathbf{z}_g - \mathbf{z}_{\text{curr}}\|$$

각 sub-plan tail에서 goal까지의 temporal distance:

$$T_i = \|\mathbf{z}_g - \mathbf{z}_i\| \quad \text{for } i = 1, \dots, N$$

Tail의 평균 잔여 temporal distance:

$$T_{\text{tail}} = \frac{1}{N}\sum_{i=1}^{N} T_i$$

한 번의 expansion으로 줄어든 평균 temporal distance:

$$\bar{\Delta} = T_{\text{curr}} - T_{\text{tail}}$$

- $\bar{\Delta}$는 "sub-plan 하나가 평균적으로 goal에 얼마나 가까워지게 하는가"를 나타냅니다.
- 수치 안정성 주의: $\bar{\Delta} \leq 0$인 경우 (sub-plan이 오히려 goal에서 멀어진 경우) 처리가 필요합니다. 이 경우는 diffusion model의 guidance가 제대로 작동하지 않았다는 신호이므로, 매우 높은 uncertainty로 처리합니다 (아래 구현 노트 참조).

**Step 7 — 최종 Uncertainty:**

$$U = \ln K \cdot \left(1 + \frac{T_{\text{tail}}}{\bar{\Delta}}\right)$$

이 값을 기반으로 트리 노드에서 분기할 자식 수를 결정합니다.

---

## 4. 구현 명세

### 4.1 함수 시그니처

```python
def compute_uncertainty(
    s_curr,              # 현재 노드의 state
    s_goal,              # goal state
    tail_states,         # sub-plan tail states list, 길이 N
    phi_encoder,         # temporal distance 보존 encoder (nn.Module)
    eps=1e-8,            # 수치 안정성을 위한 작은 값
) -> dict:
    """
    현재 노드의 uncertainty를 계산한다.

    Returns:
        {
            'U': float,              # 최종 uncertainty 값
            'ln_K': float,           # per-step entropy
            'sigma_parallel': float, # goal 방향 표준편차
            'sigma_perp': float,     # 수직 방향 표준편차
            'T_curr': float,         # 현재 노드의 goal까지 거리
            'T_tail': float,         # tail states의 goal까지 평균 거리
            'Delta_bar': float,      # 평균 진행량
            'M_rem': float,          # 잔여 expansion 횟수 (= T_tail / Delta_bar)
        }
    """
```

### 4.2 알고리즘 상세 단계

1. `phi_encoder`를 통해 `s_curr`, `s_goal`, 모든 `tail_states`를 임베딩 → `z_curr`, `z_goal`, `Z` (shape `[N, k]`)
2. `z_bar = Z.mean(dim=0)` 계산
3. 공분산 행렬 `Sigma_phi = (Z - z_bar).T @ (Z - z_bar) / N` 또는 `/ (N-1)` 계산 (shape `[k, k]`)
4. Goal 방향 단위 벡터 `g_hat = (z_goal - z_curr) / ||z_goal - z_curr||`
5. 방향별 분산:
   - `sigma_parallel_sq = g_hat.T @ Sigma_phi @ g_hat` (스칼라)
   - `sigma_perp_sq = trace(Sigma_phi) - sigma_parallel_sq` (스칼라)
   - 둘 다 `max(value, eps)` 적용
6. `ln_K = log(2 * pi * e) + 0.5 * log(sigma_parallel_sq) + 0.5 * log(sigma_perp_sq)`
7. Temporal distances:
   - `T_curr = ||z_goal - z_curr||`
   - `T_i = ||z_goal - Z[i]||` for each `i`
   - `T_tail = mean(T_i)`
   - `Delta_bar = T_curr - T_tail`
8. `Delta_bar`에 대한 방어적 처리 (아래 노트 참조)
9. `M_rem = T_tail / Delta_bar`
10. `U = ln_K * (1 + M_rem)`
11. 결과 dict 반환

### 4.3 중요한 구현 노트

**수치 안정성:**

- `sigma_parallel_sq`와 `sigma_perp_sq`는 이론적으로 양수여야 하지만, 부동소수점 오차로 매우 작은 음수가 될 수 있음. `max(value, eps)` 클립 필수.
- `log(0)` 방지를 위해 모든 로그 인수에 `eps` 추가.

**$\bar{\Delta} \leq 0$ 처리:**

Sub-plan이 goal 방향으로 진행하지 못한 경우입니다. 다음 옵션들이 있습니다:

1. **클리핑**: `Delta_bar = max(Delta_bar, eps_progress)` where `eps_progress`는 작은 양수 (예: `T_curr * 0.01`). 이 경우 `M_rem`이 매우 커져 uncertainty가 자연스럽게 매우 높아집니다.
2. **상수 대체**: 이런 상황을 명시적으로 감지하고 `U = U_max` (미리 정의된 최대값)로 설정.
3. **플래그 반환**: 결과 dict에 `'degenerate': True`를 추가하고 호출 측에서 처리.

기본으로는 옵션 1을 권장하며, `eps_progress`를 하이퍼파라미터로 노출.

**배치 처리:**

여러 노드의 uncertainty를 한 번에 계산해야 하는 경우, 위 연산을 배치 차원으로 벡터화. PyTorch의 `torch.einsum`, `torch.linalg` 함수들을 사용.

**`g_hat` 계산의 degenerate 케이스:**

`z_curr ≈ z_goal`인 경우 (이미 goal에 도달), 단위 벡터가 정의되지 않음. 이 경우 `T_curr < eps`를 체크하여 `U = 0` (또는 매우 작은 값)으로 조기 반환.

### 4.4 단위 테스트 시나리오

구현 검증을 위한 권장 테스트 케이스:

1. **모든 tail이 동일한 위치**: `Sigma_phi ≈ 0`, `ln_K → -∞` (클리핑 후 작은 값), `U`가 최소값
2. **Tail이 goal 바로 앞**: `T_tail ≈ 0`, `M_rem ≈ 0`, `U ≈ ln_K`
3. **Tail이 전혀 진행하지 못함**: `Delta_bar ≈ 0`, degenerate 케이스 처리 확인
4. **Tail이 goal 방향으로만 퍼짐**: `sigma_parallel >> sigma_perp`, `ln_K`에서 두 성분의 기여도 확인
5. **Tail이 수직 방향으로만 퍼짐**: `sigma_perp >> sigma_parallel`, 갈림길 시나리오 재현

---

## 5. 요약

- **$\ln K$**: 현재 노드에서 한 번의 expansion의 불확정성. Goal 방향과 수직 방향의 2차원 분해로 rank 문제 회피. 최대 엔트로피 원리로 분포 독립적 상한 제공.
- **$1 + T_{\text{tail}}/\bar{\Delta}$**: 잔여 expansion 횟수를 반영. $\phi$-공간의 temporal distance 보존 성질 활용.
- **$U = \ln K \cdot (1 + T_{\text{tail}}/\bar{\Delta})$**: 두 요소의 곱으로, 엔트로피 체인 룰과 정상성 가정에 기반. Goal 근처 불일관 plan과 멀리 있는 일관 plan 모두 적절히 평가.

이 지표는 tree-structured diffusion planner에서 각 노드의 분기 수를 결정하는 데 사용됩니다.
