# Variance 기반 Uncertainty 정식화

## Temporal-Distance Variance Decomposition

---

## 1. 문제 배경과 새 접근의 동기

기존 **Radial–Angular Entropy** 방식(entropy 모드)은 tail embedding들의 분포를 goal 중심 polar coordinates로 분해한 뒤, 최대 엔트로피 원리를 통해 uncertainty를 정의한다. 이 방식은 정보이론적으로 잘 정당화되지만, 다음과 같은 실용적 한계가 있다.

1. **해석의 간접성**: Radial entropy $H_R$와 angular entropy $H_A$는 각각 goal-distance spread와 route diversity를 표현하지만, 이 두 양이 최종 score $U = T_\mathrm{curr} \cdot H_\mathrm{local}$에 기여하는 방식은 log-scale로 calibration된 것이어서 각 항의 magnitude가 intuitive하지 않다.

2. **단일 reference point 의존**: 기존 방식은 goal($z_g$)을 유일한 reference로 삼아 radial/angular spread를 정의한다. 그러나 현재 노드($z_\mathrm{curr}$)로부터의 spread는 직접 반영되지 않는다. 현재 노드를 기준으로 sub-plan들이 "얼마나 다양한 step-size로 전진하는가"도 planner에게 중요한 정보다.

3. **Embedding space geometry 의존**: 고차원 embedding space에서의 pairwise distance는 실제 temporal distance와 비선형 관계를 가진다. 기존 방식은 최종 multiplier $T_\mathrm{curr}$에만 temporal distance 변환을 적용하고, local spread 계산에는 embedding space L2 distance를 그대로 쓴다.

이에 본 문서는 모든 spread 계산을 처음부터 **temporal distance space**에서 수행하는 새로운 variance 기반 uncertainty 정식화를 제안한다. 이 방식은 세 가지 분산 항의 가중 합으로 uncertainty를 구성하며, 각 항은 서로 다른 관점에서 sub-plan 분포의 불확실성을 측정한다.

---

## 2. 새 정식화의 핵심 발상

핵심 아이디어는 다음과 같다.

> 불확실성을 embedding space에서의 entropy 대신,  
> **temporal distance space에서의 variance**로 정의한다.

구체적으로, 세 관점에서 분산을 측정한다.

1. **Internal Variance** (`internal_var`):  
   생성된 N개의 sub-plan tail들이 서로 temporal하게 얼마나 다양한가?  
   → 모든 tail pair 사이의 pairwise temporal distance들의 분산

2. **External Target Variance** (`external_target_var`):  
   각 sub-plan tail이 target(goal) state로부터 temporal하게 얼마나 다양한 거리에 있는가?  
   → target state로부터 각 tail까지의 temporal distance들의 분산

3. **External Current Variance** (`external_curr_var`):  
   각 sub-plan tail이 현재 상태로부터 temporal하게 얼마나 다양한 거리에 있는가?  
   → current state로부터 각 tail까지의 temporal distance들의 분산

세 항을 linear combination으로 합산하여 최종 uncertainty score를 구성한다.

$$
U_\mathrm{var} = V_\mathrm{int} + \lambda \cdot V_\mathrm{ext,g} + \eta \cdot V_\mathrm{ext,c}
$$

---

## 3. 이 문서에서 무엇이 exact이고 무엇이 설계 선택인가

각 수식의 지위를 먼저 명확히 한다.

### 3.1 Exact definition / identity

- Embedding-to-temporal distance 변환 $\tau(\cdot)$의 수식 (HILP convergence로부터 유도)
- 각 variance 항의 표본 분산 공식 (unbiased estimator)
- König–Huygens 항등식을 이용한 mean-free 표현

### 3.2 Design choice (이론적으로 유일하지 않으나 논리적으로 정당화됨)

- Temporal distance space에서 variance를 계산하는 선택 (embedding space L2 대신)
- 세 항의 linear combination 형태
- 가중치 $\lambda$, $\eta$를 하이퍼파라미터로 두는 선택

### 3.3 Empirical estimator

- 유한 샘플 $N$개로부터 계산하는 모든 표본 분산 값들

---

## 4. 기본 설정과 기호

현재 노드의 state $s_\mathrm{curr}$, target(goal) state $s_g$, 그리고 현재 노드에서 생성된 $N$개의 sub-plan tail state
$$
\{s_i\}_{i=1}^N
$$
를 가정한다.

Temporal-distance-preserving encoder
$$
\phi : \mathcal{S} \to \mathbb{R}^D
$$
를 통해 각 state를 embedding space로 보낸다.

각 embedding을 다음과 같이 둔다.

$$
z_\mathrm{curr} = \phi(s_\mathrm{curr}), \qquad z_g = \phi(s_g), \qquad z_i = \phi(s_i)
$$

---

## 5. Temporal Distance 변환 $\tau$

HILP (Hilbert-space Inverse Reinforcement Learning via Potential) value function이 수렴하면

$$
\|\phi(s) - \phi(g)\| \approx \frac{1 - \gamma^{d^*(s,g)}}{1 - \gamma}
$$

가 성립한다. 여기서 $d^*(s,g)$는 두 state 사이의 optimal temporal distance(step 수)이고, $\gamma$는 discount factor이다. 이 식을 $d^*$에 대해 역산하면

$$
\tau(e) := \frac{\log\!\left(1 - e\cdot(1-\gamma)\right)}{\log \gamma}
$$

를 얻는다. $\tau : [0, \frac{1}{1-\gamma}) \to [0, \infty)$는 strictly monotone increasing 함수이다.

**주의**: $\tau$는 embedding L2 distance $e = \|u - v\|$를 입력으로 받는다. 따라서 본 문서에서 두 state $u, v$의 temporal distance는

$$
d^T(u, v) := \tau(\|z_u - z_v\|)
$$

로 정의된다. 본 문서의 모든 분산 계산은 이 $d^T$ 값들을 원소로 삼는다.

---

## 6. 세 분산 항의 수학적 정의

### 6.1 External Target Variance $V_\mathrm{ext,g}$

각 tail $s_i$에서 target state $s_g$까지의 temporal distance를

$$
t_i^g := d^T(s_i, s_g) = \tau\!\left(\|z_i - z_g\|\right), \qquad i = 1, \ldots, N
$$

으로 정의한다. $V_\mathrm{ext,g}$는 이 $N$개의 scalar temporal distance들의 **unbiased 표본 분산**이다.

**표준 공식:**

$$
V_\mathrm{ext,g} = \frac{1}{N-1}\sum_{i=1}^N \left(t_i^g - \bar{t}^g\right)^2, \qquad \bar{t}^g = \frac{1}{N}\sum_{i=1}^N t_i^g
$$

**König–Huygens 항등식을 통한 mean-free 표현:**

$$
\boxed{
V_\mathrm{ext,g} = \frac{1}{N(N-1)}\sum_{i < j}\!\left(t_i^g - t_j^g\right)^2
}
$$

두 식은 대수적으로 동치이다. mean-free 형태는 mean $\bar{t}^g$를 명시적으로 계산하지 않고도 분산을 얻는 방법이다.

### 6.2 External Current Variance $V_\mathrm{ext,c}$

각 tail $s_i$에서 current state $s_\mathrm{curr}$까지의 temporal distance를

$$
t_i^c := d^T(s_i, s_\mathrm{curr}) = \tau\!\left(\|z_i - z_\mathrm{curr}\|\right), \qquad i = 1, \ldots, N
$$

으로 정의한다. $V_\mathrm{ext,c}$는 이 $N$개 값들의 unbiased 표본 분산이다.

$$
\boxed{
V_\mathrm{ext,c} = \frac{1}{N(N-1)}\sum_{i < j}\!\left(t_i^c - t_j^c\right)^2
}
$$

### 6.3 Internal Pairwise Variance $V_\mathrm{int}$

모든 tail pair $(i, j)$, $i < j$에 대해 pairwise temporal distance를

$$
d_{ij}^T := d^T(s_i, s_j) = \tau\!\left(\|z_i - z_j\|\right)
$$

로 정의한다. 총 $M = \binom{N}{2} = \frac{N(N-1)}{2}$개의 값이 존재한다.

$V_\mathrm{int}$는 이 $M$개의 pairwise temporal distance들의 **unbiased 표본 분산**이다.

**표준 공식:**

$$
V_\mathrm{int} = \frac{1}{M-1}\sum_{(i,j): i < j}\!\left(d_{ij}^T - \bar{d}^T\right)^2, \qquad \bar{d}^T = \frac{1}{M}\sum_{(i,j): i < j} d_{ij}^T
$$

**mean-free 형태 (König–Huygens):**

$$
\boxed{
V_\mathrm{int} = \frac{1}{M(M-1)}\sum_{\substack{(i,j),(k,l) \\ i<j,\; k<l \\ (i,j)\neq(k,l)}}\!\frac{1}{2}\left(d_{ij}^T - d_{kl}^T\right)^2
}
$$

**$N$이 작을 때 주의**: $M = 1$ ($N=2$)인 경우 분산이 정의되지 않는다(분모 $M-1=0$). $N \geq 3$ (즉 $M \geq 3$)이어야 $V_\mathrm{int}$가 유효하다.

---

## 7. 최종 Uncertainty Score

$$
\boxed{
U_\mathrm{var} = V_\mathrm{int} + \lambda \cdot V_\mathrm{ext,g} + \eta \cdot V_\mathrm{ext,c}
}
$$

여기서 $\lambda \geq 0$, $\eta \geq 0$은 하이퍼파라미터이다.

**부호**: 세 항 모두 분산이므로 항상 $\geq 0$이다. $U_\mathrm{var} = 0$은 모든 sub-plan tail이 temporal distance 관점에서 완전히 동일한 위치에 있음을 의미한다.

---

## 8. 각 항의 의미와 정당성

### 8.1 $V_\mathrm{ext,g}$: "얼마나 남았는지"의 불확실성

$t_i^g$는 sub-plan tail $s_i$가 target으로부터 temporal하게 얼마나 떨어져 있는지를 나타낸다. 이 값이 크면 tail이 target에서 멀리 있고, 작으면 가깝다.

$V_\mathrm{ext,g}$가 크다는 것은: 어떤 sub-plan은 target에 거의 도달하지만, 어떤 sub-plan은 아직 target에서 멀리 있다는 뜻이다. 즉 **앞으로 몇 번의 확장이 더 필요한지**에 대한 불확실성이 높다.

이는 기존 entropy 방식의 radial term $H_R$이 포착하던 것과 개념적으로 대응되지만, log-scale entropy 대신 직접 temporal distance variance를 사용한다.

### 8.2 $V_\mathrm{ext,c}$: "이번 step이 얼마나 나아갔는지"의 불확실성

$t_i^c$는 sub-plan tail $s_i$가 current state로부터 temporal하게 얼마나 떨어져 있는지를 나타낸다. 즉, 이번 diffusion sampling이 생성한 sub-plan의 "step size"에 해당한다.

$V_\mathrm{ext,c}$가 크다는 것은: 어떤 sub-plan은 큰 진전을 이루지만, 어떤 sub-plan은 거의 현재 위치에 머무른다는 뜻이다. 이는 **현재 노드에서의 sub-plan 생성 자체가 불안정**함을 나타낸다.

이 term은 기존 entropy 방식에 직접 대응하는 항이 없다. 기존 방식은 goal을 유일한 reference로 삼기 때문에, current state로부터의 spread는 implicit하게 geometry correction $\hat{C}_\mathrm{geom}$에 흡수될 뿐 독립적으로 측정되지 않는다.

### 8.3 $V_\mathrm{int}$: Sub-plan들 사이의 내재적 다양성

$d_{ij}^T$는 두 tail $s_i$, $s_j$가 서로 temporal하게 얼마나 떨어져 있는지를 나타낸다.

$V_\mathrm{int}$가 크다는 것은: 어떤 pair의 tail들은 서로 temporally 매우 먼 반면, 어떤 pair들은 temporally 가깝다는 뜻이다. 즉 **생성된 sub-plan들이 얼마나 불균일하게 퍼져 있는지**를 측정한다.

기존 entropy 방식의 angular term $H_A$가 goal로부터의 방향 다양성을 측정한다면, $V_\mathrm{int}$는 어떤 reference도 없이 sub-plan들 자체 사이의 temporal 다양성을 측정한다. Goal이나 current state로부터 모두 비슷한 거리에 있더라도, 서로 다른 방향으로 퍼진 tail들은 높은 $V_\mathrm{int}$를 가진다.

---

## 9. 기존 Entropy 방식과의 비교

| 특성 | Entropy 방식 (entropy 모드) | Variance 방식 (variance 모드) |
|---|---|---|
| 최종 score 형태 | $U = T_\mathrm{curr} \cdot H_\mathrm{local}$ | $U_\mathrm{var} = V_\mathrm{int} + \lambda V_\mathrm{ext,g} + \eta V_\mathrm{ext,c}$ |
| Reference point | Goal만 사용 | Goal + Current 모두 사용 |
| Temporal distance 사용 | 최종 multiplier ($T_\mathrm{curr}$)에만 | 모든 spread 계산에 직접 사용 |
| Angular/Radial 분리 | 명시적 (polar decomposition) | 없음 (distance variance만 사용) |
| Geometry correction | 필요 ($C_\mathrm{geom}$) | 불필요 |
| Log-scale | 사용 (entropy) | 미사용 (직접 variance) |
| $N=2$ 허용 | 허용 | $V_\mathrm{int}$ 정의 불가 ($N \geq 3$ 필요) |
| 해석 직관성 | 간접적 (entropy upper bound) | 직접적 (temporal step variance) |

---

## 10. 가중치 $\lambda$, $\eta$의 역할

$\lambda$와 $\eta$는 각각 $V_\mathrm{ext,g}$와 $V_\mathrm{ext,c}$가 전체 uncertainty에 기여하는 상대적 비중을 조절한다.

- $\lambda = 0$: target 기준 spread를 무시, internal + current spread만 사용
- $\eta = 0$: current 기준 spread를 무시, internal + target spread만 사용
- $\lambda = \eta = 0$: 오직 $V_\mathrm{int}$만 사용 (sub-plan들 사이의 내재적 다양성만 측정)

세 항의 단위(temporal step 수의 제곱)는 동일하므로, 가중치는 무차원 상수로서 세 관점 중 어느 것에 더 민감하게 반응할지를 결정하는 설계 파라미터다.

---

## 11. 수치적 고려 사항

### 11.1 분산의 하한

세 항 모두 $\geq 0$이다. 모든 sub-plan이 temporally 동일한 위치에 있으면 분산이 0이 되어 $U_\mathrm{var} = 0$이다. 이는 "완전히 확실한 노드"를 의미하며, uncertainty 기반 value function에서 해당 노드의 value를 최소화(가장 낮은 uncertainty = 가장 좋음)하는 방향으로 사용된다.

### 11.2 Outlier 민감성

분산은 outlier에 민감하다. 하나의 sub-plan tail이 특이하게 먼 temporal distance를 가지면 분산이 크게 증가할 수 있다. Entropy 방식이 log-scale을 통해 이를 자연스럽게 억제하는 반면, variance 방식은 이 효과가 없다. 실험적으로 문제가 될 경우 $\tau$ 값에 clipping을 적용하거나, 분산 대신 median absolute deviation을 고려할 수 있다.

---

## 12. 요약

Variance 기반 uncertainty는 모든 spread 계산을 temporal distance space에서 수행한다는 점에서 기존 entropy 방식과 근본적으로 다르다. 세 항의 의미는 다음과 같다.

$$
\underbrace{V_\mathrm{int}}_{\text{sub-plan들 사이의 내재적 temporal 다양성}}
+ \lambda \cdot \underbrace{V_\mathrm{ext,g}}_{\text{target까지 남은 거리의 불확실성}}
+ \eta \cdot \underbrace{V_\mathrm{ext,c}}_{\text{이번 step 크기의 불확실성}}
$$

이 세 항은 서로 다른 reference(없음 / goal / current)와 서로 다른 의미(내재적 다양성 / 도착 시점 불확실성 / 진전 크기 불확실성)를 가지므로, 상호 보완적으로 sub-plan 분포의 불확실성을 측정한다.
