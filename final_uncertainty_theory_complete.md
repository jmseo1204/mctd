# Uncertainty-Guided Tree Expansion (이론 보강 최종판)
## Radial–Angular Entropy 기반 Uncertainty 정식화

---

## 1. 문제 배경

본 시스템은 **tree-structured diffusion planner**이다. 시작 상태에서 goal state까지의 경로를 트리 형태로 점진적으로 확장하며, 각 노드에서 diffusion sampling을 통해 여러 개의 sub-plan을 생성한다. 각 sub-plan의 tail state는 다음 확장의 후보 자식 노드가 된다.

이때 중요한 질문은 다음과 같다.

> **현재 노드는 goal에 대해 얼마나 불확실한가?**

이 불확실성은 단순한 분산이나 단순한 goal-direction spread로는 충분히 표현되지 않는다. 실제로 planner 입장에서 알고 싶은 것은 다음 두 종류의 불확실성이다.

1. **goal까지 남은 거리의 불확실성**  
   서로 다른 sub-plan tail들이 goal까지 얼마나 멀리 남아 있는지가 서로 다르면, planner는 앞으로 몇 번의 확장이 더 필요한지에 대해 불확실하다.

2. **같은 거리 수준에서의 경로 다양성**  
   어떤 경우에는 tail state들이 goal까지의 거리 자체는 비슷하지만, 서로 전혀 다른 위치에 퍼져 있을 수 있다. 이는 "언제 도착하느냐"의 문제라기보다 "어떤 route로 갈 수 있느냐"의 문제다.

기존 방식은 embedding covariance를 goal 방향과 그 수직 방향으로 나누고, 가우시안 최대 엔트로피 상한을 이용해 local uncertainty를 구성하였다. 그러나 이 방식에는 다음 한계가 있다.

- 목표로 하는 quantity가 실제로는 **goal까지의 거리 분산**과 **route diversity**인데, goal 방향 축으로 covariance를 자르는 방식은 이 두 의미를 직접 측정하지 못한다.
- 동일한 goal-distance를 갖는 샘플들이 goal-centered sphere 위에 넓게 퍼져 있는 경우, 기존 방식의 goal-parallel / goal-perpendicular decomposition은 원하는 의미를 안정적으로 반영하지 못한다.
- 고차원 covariance의 \(\log \det\)는 rank deficiency 문제에 취약하다.

따라서 본 문서는 covariance decomposition 대신, **goal-relative radial–angular decomposition**을 통해 uncertainty를 다시 정의한다.

---

## 2. 새 정식화의 핵심 발상

핵심 아이디어는 다음과 같다.

> 불확실성은 embedding 좌표축의 분산이 아니라,  
> **goal까지 거리의 변동(radial spread)** 과  
> **같은 거리 수준에서의 경로 다양성(angular spread)**  
> 로 나누어 정의되어야 한다.

이를 위해 우리는 각 tail embedding의 위치 자체보다, **goal과의 거리** 및 **tail들끼리의 pairwise distance**를 직접 사용한다. 이 방식은 다음 장점을 갖는다.

- goal-relative semantics가 직접 반영된다.
- 좌표축 선택에 의존하지 않는다.
- covariance determinant를 계산하지 않으므로 수치적으로 안정적이다.
- entropy decomposition과 개념적으로 대응되는 구조를 만들 수 있다.

---

## 3. 이 문서에서 무엇이 exact이고 무엇이 근사인가

이론적 주장 수준을 먼저 명확히 하겠다. 본 문서의 각 수식은 아래 네 부류로 나뉜다.

### 3.1 Exact identity

다음은 정의상 또는 좌표변환상 **정확한 식**이다.

- goal-centered polar decomposition에 대한 entropy chain rule
- pairwise spread의 radial / off-radial 분해
- geometry correction의 Jacobian 항
- normalized angular residual과 unit-direction pairwise spread의 동치식

### 3.2 Rigorous upper bound

다음은 최대 엔트로피 원리 또는 정보이론적 기본 부등식에 의해 **엄밀한 upper bound**이다.

- 1차원 radial variable \(R\)에 대한
  $$
  h(R)\le \frac12 \log(2\pi e\,\mathrm{Var}(R))
  $$
- conditional entropy에 대한
  $$
  h(A\mid R)\le h(A)
  $$

### 3.3 Surrogate / approximation

다음은 exact entropy가 아니라, **second-order surrogate**이다.

- \(h(A)\)를 직접 density estimation 하지 않고, pairwise second moment를 이용한 upper-bound surrogate로 근사
- 이에 따라 angular term을
  $$
  \frac{m-1}{2}\log(\cdot)
  $$
  꼴로 calibration

### 3.4 Empirical estimator

다음은 샘플 \(N\)개로부터 계산하는 실제 estimator이다.

- \(S_R\), \(S_{\mathrm{tot}}\), \(S_U\), \(S_A\)
- \(\widehat C_{\mathrm{geom}}\)
- \(H_R\), \(H_A\), \(H_{\mathrm{local}}\)
- 최종 score \(U=T_{\mathrm{curr}}H_{\mathrm{local}}\)

즉, 이 문서의 최종 estimator는 **exact entropy 자체**가 아니라,

> **exact radial–angular decomposition에서 출발하여,  
> radial part는 rigorous upper bound로,  
> angular part는 pairwise entropy upper bound에 기반한 surrogate로 구성한 uncertainty measure**

이다.

---

## 4. 기본 설정과 기호

현재 노드 \(s_{\mathrm{curr}}\), goal \(s_g\), 그리고 현재 노드에서 생성된 \(N\)개의 tail state
$$
\{s_i\}_{i=1}^N
$$
를 가정한다.

Temporal-distance-preserving encoder
$$
\phi:\mathcal S\to \mathbb R^m
$$
를 사용하여 각 state를 embedding space로 보낸다. 여기서 \(m\)은 embedding 차원이다.

각 embedding을 다음과 같이 둔다.

$$
z_{\mathrm{curr}}=\phi(s_{\mathrm{curr}}), \qquad z_g=\phi(s_g), \qquad z_i=\phi(s_i)
$$

본 문서에서는 **geodesic distance를 사용하지 않고**, embedding space의 유클리드 거리를 사용한다. 따라서 local uncertainty의 radius / angle spread를 분해하는 모든 기하는 다음 거리를 기반으로 한다.

$$
\|u-v\| \quad \text{for } u,v\in \mathbb R^m
$$

다만 이는 temporal distance를 개념적으로 완벽히 표현하는 양은 아니다. 구현에서는 radial / angular variance를 안정적으로 정의하기 위해 embedding distance를 local geometry의 surrogate로 사용하고, **최종 score의 global multiplier인 \(T_{\mathrm{curr}}\)** 에 대해서만 embedding distance를 `emb_dist_to_temporal_dist()`로 변환한 temporal distance를 사용한다.

---

## 5. Goal-relative radial–angular variables

각 tail sample에 대해 goal까지의 거리를

$$
r_i := \|z_i-z_g\|
$$

로 정의한다. 이는 현재 sub-plan tail이 goal로부터 얼마나 떨어져 있는지를 나타내는 scalar quantity이다.

또한 두 tail embedding 사이의 pairwise distance를

$$
d_{ij}:=\|z_i-z_j\|
$$

로 둔다.

이제 planner 관점에서 원하는 두 불확실성은 다음으로 대응된다.

- \(r_i\)들의 spread: **goal-distance uncertainty**
- goal까지의 거리 차이를 제거한 뒤에도 남는 pairwise separation: **route diversity**

추가로, 각 tail sample의 **normalized direction**을

$$
u_i := \frac{z_i-z_g}{r_i}
\qquad (r_i>0)
$$

로 정의한다. \(u_i\)는 goal을 중심으로 보았을 때 tail sample이 어느 방향에 위치하는지를 나타내는 unit vector이다. 후술하듯, angular entropy를 정당하게 구성하려면 결국 이 \(u_i\)들의 분포를 다루어야 한다.

---

## 6. 전체 pairwise spread와 exact decomposition

먼저 tail samples 전체의 pairwise spread를 정의한다.

$$
S_{\mathrm{tot}}
:=
\frac{1}{2N(N-1)}\sum_{i\ne j} d_{ij}^2
$$

이 quantity는 샘플 cloud 전체가 embedding space에서 얼마나 넓게 퍼져 있는지를 나타낸다.

이제 radial part를 다음과 같이 정의한다.

$$
S_R
:=
\frac{1}{2N(N-1)}\sum_{i\ne j}(r_i-r_j)^2
$$

이는 goal까지 거리의 차이만을 모아서 계산한 spread이다. 따라서 의미는 명확하다.

> \(S_R\)는 tail들이 goal까지 얼마만큼 다른 distance-to-go를 가지는지를 측정한다.

다음으로 off-radial residual을

$$
S_U
:=
\frac{1}{2N(N-1)}\sum_{i\ne j}\Big(d_{ij}^2-(r_i-r_j)^2\Big)
$$

로 정의한다.

그러면 정의에 의해

$$
S_{\mathrm{tot}}=S_R+S_U
$$

가 정확히 성립한다.

### 6.1 왜 \(S_U\)가 의미 있는가

\(d_{ij}^2\)는 두 tail sample 사이의 전체 거리이고, \((r_i-r_j)^2\)는 그 중에서 "goal까지 남은 거리 차이"만으로 설명되는 부분이다. 따라서

$$
d_{ij}^2-(r_i-r_j)^2
$$

는 다음 의미를 갖는다.

> **goal-distance 차이만으로는 설명되지 않는 pairwise separation**

즉, 두 샘플이 goal까지는 비슷하게 남아 있어도 서로 다른 위치로 퍼져 있다면 이 값이 커진다. 반대로 서로 다른 goal-distance만 가질 뿐 같은 ray 상에 놓여 있다면 이 값은 작다.

이 quantity는 planner가 원하는 두 번째 불확실성, 즉 **route diversity**의 원형(raw) quantity로 해석할 수 있다.

### 6.2 왜 \(S_U\)만으로는 충분하지 않은가

그러나 \(S_U\)는 그대로 angular entropy term으로 쓰기에는 한 가지 문제가 있다.  
서로 다른 pair \((i,j)\)의 반지름 \(r_i,r_j\)가 다를 때, 같은 angular discrepancy라도 \(r_i r_j\)가 큰 pair는 더 큰 residual을 만든다. 즉 \(S_U\)는 pure angular uncertainty와 radial scale이 섞여 있다.

따라서 \(h(A\mid R)\) 혹은 \(h(A)\)에 대응하는 angular quantity를 만들려면, **반지름 효과를 제거한 normalized angular spread**가 필요하다.

---

## 7. 왜 entropy decomposition을 도입하는가

지금까지의 \(S_R\), \(S_U\)는 geometric spread quantity이다. 그러나 우리는 uncertainty를 단순히 spread로만 쓰고 싶은 것이 아니라, 기존 방식처럼 **정보 엔트로피적 의미**를 가진 quantity로 calibration하고 싶다.

이를 위해 goal-centered polar decomposition을 고려한다.

---

## 8. Goal-centered polar coordinates와 entropy decomposition

임의의 연속 random variable \(X\in\mathbb R^m\)를 생각하자. 여기서 \(X\)는 tail embedding distribution을 나타내는 random variable로 생각할 수 있다. goal embedding \(z_g\)를 기준점으로 둘 때,

$$
R := \|X-z_g\|
$$

를 radial variable이라 하고, 방향 variable \(A\)를 다음과 같이 둔다.

$$
A := \frac{X-z_g}{\|X-z_g\|}
$$

그러면 거의 모든 점에서

$$
X = z_g + R A
$$

로 쓸 수 있다. 즉 \(X\)를 goal-centered polar coordinates \((R,A)\)로 바꾸어 표현한 것이다.

### 8.1 Jacobian과 volume element

유클리드 공간 \(\mathbb R^m\)에서 polar coordinates의 volume element는 잘 알려진 대로

$$
dx = r^{m-1}\,dr\,d\omega(a)
$$

이다. 여기서 \(d\omega(a)\)는 unit sphere \(S^{m-1}\) 위의 표준 면적 measure이다.  
이 식이 말하는 바는 다음과 같다.

- 같은 \(dr\)만큼 radial coordinate가 증가해도,
- 차원이 \(m\)일 때 가능한 angular 방향의 volume은 \(r^{m-1}\)에 비례하여 커진다.

이 \(r^{m-1}\)가 바로 geometry correction의 근원이다.

보다 일반적으로 쓰면,
$$
dx = A_g(r,a)\,dr\,d\omega(a)
$$
인데, 본 문서에서는 유클리드 polar coordinates를 쓰므로
$$
A_g(r,a)=r^{m-1}
$$
이다.

### 8.2 Density 변환

\(X\)의 density를 \(p_X(x)\), polar variable \((R,A)\)의 density를 \(p_{R,A}(r,a)\)라 하면, change of variables에 의해

$$
p_X(x)\,dx = p_{R,A}(r,a)\,dr\,d\omega(a)
$$

이고 따라서

$$
p_X(x)=\frac{p_{R,A}(r,a)}{r^{m-1}}
$$

가 된다.

### 8.3 Entropy 유도

Differential entropy의 정의에 의해

$$
h(X) = -\mathbb E[\log p_X(X)]
$$

이므로 위 density 식을 대입하면

$$
h(X)
=
-\mathbb E\left[\log p_{R,A}(R,A)- (m-1)\log R\right]
$$

즉,

$$
h(X)=h(R,A)+(m-1)\mathbb E[\log R]
$$

를 얻는다.

이제 chain rule
$$
h(R,A)=h(R)+h(A\mid R)
$$
를 적용하면 최종적으로

$$
\boxed{
h(X)=h(R)+h(A\mid R)+(m-1)\mathbb E[\log R]
}
$$

가 된다.

---

## 9. 위 식의 의미

이 식은 이 문서 전체의 핵심적인 이론적 출발점이다.

- \(h(R)\): goal까지 distance-to-go의 uncertainty
- \(h(A\mid R)\): 같은 거리 shell 안에서 direction / route가 얼마나 다양하게 퍼져 있는가
- \((m-1)\mathbb E[\log R]\): polar coordinates로 바꾸었을 때 생기는 volume-growth correction

즉, planner가 원하는 두 종류의 uncertainty가 entropy chain rule 안에서 자연스럽게 분리된다.

---

## 10. Geometry correction이 왜 \((m-1)\mathbb E[\log R]\)인가

앞 절에서 보았듯, geometry correction은 임의로 넣은 항이 아니다.  
이 항은 오직 다음 사실에서 나온다.

> 유클리드 \(m\)차원 공간에서 goal-centered polar coordinates의 Jacobian determinant가 \(r^{m-1}\)이다.

이 때문에 density가 변환될 때 \(r^{m-1}\)이 denominator로 들어가고, entropy에서는 그 로그의 기댓값이 추가된다.

따라서 geometry correction은 정확히

$$
(m-1)\mathbb E[\log R]
$$

이다.

### 10.1 샘플 기반 estimator

실제 구현에서는 \(R\)의 true distribution을 모르므로 empirical average로 대체한다.

$$
\widehat C_{\mathrm{geom}}
=
(m-1)\frac1N\sum_{i=1}^N \log(r_i+\varepsilon)
$$

여기서 \(\varepsilon>0\)는 \(\log 0\) 방지를 위한 안정화 상수이다.

### 10.2 왜 이 항을 빼면 안 되는가

이 항을 빼버리면, \(h(X)\)와 \(h(R)+h(A\mid R)\) 사이의 정확한 change-of-variables relation이 끊어진다.  
즉 geometry correction은 단지 "미세한 보정"이 아니라,

> **entropy decomposition을 정확히 성립시키는 Jacobian term**

이다.

따라서 문서에서 이 항을 생략하거나 축약하면, 전체 이론 전개의 정당성이 약해진다.

---

## 11. Radial term: \(h(R)\)에 대한 rigorous upper bound

이제 decomposition의 첫 번째 항 \(h(R)\)를 다룬다.

\(R\)는 1차원 scalar random variable이다. 이때 최대 엔트로피 정리에 의해, 고정된 분산을 가지는 모든 실수-valued random variable 중 Gaussian이 differential entropy를 최대화한다. 따라서

$$
h(R)\le \frac12\log\big(2\pi e\,\mathrm{Var}(R)\big)
$$

가 항상 성립한다.

이는 exact upper bound다.

### 11.1 샘플 기반 radial spread

\(\mathrm{Var}(R)\)의 empirical surrogate로 pairwise form을 사용하면

$$
S_R=
\frac{1}{2N(N-1)}\sum_{i\ne j}(r_i-r_j)^2
$$

가 된다. 이는 sample variance와 같은 scale의 quantity이며, goal-distance uncertainty를 직접 반영한다.

따라서 radial entropy upper bound estimator를

$$
\boxed{
H_R
:=
\frac12\log\big(2\pi e(S_R+\varepsilon)\big)
}
$$

로 둔다.

이 값은 다음 의미를 갖는다.

> tail samples의 distance-to-go distribution이 가지는 1차원 entropy를,  
> sample spread를 이용하여 upper bound한 값

---

## 12. Angular term: 왜 기존 \(S_U\) 기반 설명만으로는 부족한가

문제는 두 번째 항 \(h(A\mid R)\)이다.

### 12.1 직접 계산이 어려운 이유

\(A\mid R=r\)는 반지름 \(r\)인 sphere 위의 조건부 분포다. 이를 직접 계산하려면 각 \(r\)에 대해 shell 위 density를 알아야 한다. 그러나 실제 planner에서는 tail sample \(N\)개만 존재할 뿐, 조건부 density \(p(A\mid R)\)를 직접 알 수 없다.

즉 \(h(A\mid R)\)는 이론적으로는 잘 정의되지만, 실질적으로는 직접 계산하기 어렵다.

### 12.2 단순 \(S_U\) 연결의 약점

이전 버전에서는 \(S_U\)를 "same-distance shell 위 spread"의 surrogate로 보았다. 그러나 이 연결은 충분히 강하지 않다. 이유는 다음과 같다.

1. \(S_U\)는 서로 다른 반지름 \(r_i,r_j\)를 가진 pair를 모두 섞어 계산한 **unconditional quantity**이다.
2. 그런데 \(h(A\mid R)\)는 말 그대로 **conditional entropy**이다.
3. 또한 \(S_U\)는 반지름이 큰 pair에 더 큰 weight를 부여하므로 pure angular spread가 아니라 radius scale과 섞인 quantity다.

따라서 \(h(A\mid R)\)와 이론적으로 더 설득력 있게 연결하려면, 먼저 반지름 효과를 제거한 angular statistic을 구성해야 한다.

---

## 13. Normalized angular spread \(S_A\)

### 13.1 핵심 항등식

각 sample을
$$
z_i = z_g + r_i u_i
$$
로 쓰자. 여기서
$$
u_i = \frac{z_i-z_g}{r_i}
$$
는 unit direction이다.

그러면 pairwise distance는

$$
d_{ij}^2
=
\|r_i u_i-r_j u_j\|^2
=
r_i^2+r_j^2-2r_i r_j\,u_i^\top u_j
$$

이고,

$$
(r_i-r_j)^2 = r_i^2+r_j^2-2r_i r_j
$$

이므로 둘을 빼면

$$
d_{ij}^2-(r_i-r_j)^2
=
2r_i r_j(1-u_i^\top u_j)
$$

를 얻는다.

한편 unit vector에 대해

$$
\|u_i-u_j\|^2 = 2-2u_i^\top u_j = 2(1-u_i^\top u_j)
$$

이므로 최종적으로

$$
\boxed{
d_{ij}^2-(r_i-r_j)^2
=
r_i r_j\,\|u_i-u_j\|^2
}
$$

가 **정확히 성립**한다.

### 13.2 normalized angular statistic

위 식을 반영하여, 본 문서는 angular spread를 raw residual \(S_U\)로 두지 않고 다음 normalized statistic으로 재정의한다.

$$
\boxed{
S_A
:=
\frac{1}{2N(N-1)}
\sum_{i\ne j}
\frac{d_{ij}^2-(r_i-r_j)^2}{r_i r_j+\varepsilon}
}
$$

위 exact identity를 이용하면,

$$
S_A
\approx
\frac{1}{2N(N-1)}\sum_{i\ne j}\|u_i-u_j\|^2
$$

로 해석할 수 있다. 즉 \(S_A\)는 **goal-centered unit-direction space에서의 pairwise spread**이다.

### 13.3 왜 \(S_A\)가 더 적절한가

이제 각 pair에서 반지름 효과 \(r_i r_j\)를 제거했기 때문에, \(S_A\)는 더 이상 "큰 radius pair가 더 큰 값을 만드는" 문제를 갖지 않는다.  
즉 \(S_A\)는 radial scale과 분리된 **pure angular uncertainty**를 측정한다.

이것이 바로 \(h(A\mid R)\) 혹은 최소한 \(h(A)\)에 연결할 수 있는 올바른 statistic이다.

---

## 14. \(S_A\)와 \(h(A\mid R)\)의 정당한 연결

### 14.1 직접 \(h(A\mid R)\)를 추정하지 않는 전략

여기서 핵심 아이디어는 조건부 entropy를 직접 추정하려 하지 않는 것이다. 대신 정보이론의 기본 부등식

$$
\boxed{
h(A\mid R)\le h(A)
}
$$

를 사용한다.

즉 \(h(A\mid R)\)를 upper bound하려면 먼저 \(h(A)\)를 upper bound하면 된다.  
이 전략의 장점은 shell binning이나 explicit conditional density estimation 없이도 이론 전개가 가능하다는 점이다.

### 14.2 pairwise entropy bound

유클리드 random vector \(Y\in\mathbb R^d\)에 대해, trace-variance 형태의 pairwise entropy bound를 쓸 수 있다.

$$
h(Y)
\le
\frac{d}{2}
\log\left(
\frac{\pi e}{d}\,\mathbb E\|Y-Y'\|^2
\right)
+ C_d
$$

여기서 \(Y'\)는 \(Y\)의 independent copy이고, \(C_d\)는 차원 \(d\)에만 의존하는 상수다.  
실제 uncertainty score에서는 additive constant는 ranking에 영향을 주지 않으므로 흡수해도 된다. 따라서 scale 측면에서

$$
h(Y)
\lesssim
\frac{d}{2}
\log\left(
\mathbb E\|Y-Y'\|^2
\right)
$$

라고 볼 수 있다.

### 14.3 이를 angular variable에 적용

이제 \(Y=A\)로 두고, angular variable의 effective dimension을 \(d=m-1\)로 둔다. 그러면

$$
h(A)
\lesssim
\frac{m-1}{2}
\log\left(
\frac{\pi e}{m-1}\,\mathbb E\|A-A'\|^2
\right)
$$

를 얻는다.

그런데 \(A\)의 empirical pairwise spread가 바로 \(S_A\)이므로,

$$
\mathbb E\|A-A'\|^2 \approx S_A
$$

라고 둘 수 있다. 따라서

$$
\boxed{
h(A\mid R)\le h(A)
\lesssim
\frac{m-1}{2}
\log\left(
\frac{\pi e}{m-1}(S_A+\varepsilon)
\right)
}
$$

라는 upper-bound surrogate chain을 얻는다.

### 14.4 이 결과의 의미

이제 angular term은 더 이상 "퍼짐이 비슷하니까 대응된다" 수준의 약한 설명이 아니다.  
우리는 다음을 주장할 수 있다.

- \(S_A\)는 unit-direction random variable \(A\)의 pairwise second moment를 empirical하게 측정한다.
- pairwise entropy bound에 의해, 이 second moment는 \(h(A)\)의 upper-bound scale을 결정한다.
- 그리고 \(h(A\mid R)\le h(A)\)이므로, \(S_A\)는 \(h(A\mid R)\)에 대한 **정당한 upper-bound surrogate**를 제공한다.

즉 본 문서의 angular term은 단순 heuristic이 아니라,

> **conditional entropy를 직접 추정할 수 없을 때,  
> unconditional directional entropy의 pairwise upper bound를 통해 우회적으로 구성한 surrogate**

이다.

---

## 15. Angular entropy surrogate의 최종 정의

위 논의를 바탕으로 angular term을 다음과 같이 정의한다.

$$
\boxed{
H_A
:=
\frac{m-1}{2}
\log\left(
\frac{\pi e}{m-1}(S_A+\varepsilon)
\right)
}
$$

이 값은 exact entropy는 아니지만, \(h(A\mid R)\)를 반영하는 **pairwise entropy upper-bound surrogate**이다.

---

## 16. 최종 local uncertainty

지금까지의 세 항을 합치면 local uncertainty를 다음과 같이 정의할 수 있다.

$$
\boxed{
H_{\mathrm{local}}
=
\frac12\log\big(2\pi e(S_R+\varepsilon)\big)
+
\lambda H_A
+
\eta \widehat C_{\mathrm{geom}}
}
$$

즉 구체적으로 쓰면

$$
\boxed{
H_{\mathrm{local}}
=
\frac12\log\big(2\pi e(S_R+\varepsilon)\big)
+
\lambda\frac{m-1}{2}
\log\left(
\frac{\pi e}{m-1}(S_A+\varepsilon)
\right)
+
\eta \widehat C_{\mathrm{geom}}
}
$$

여기서:

- 첫 번째 항은 **radial entropy upper bound**
- 두 번째 항은 **angular conditional entropy에 대한 pairwise upper-bound surrogate**
- 세 번째 항은 **exact change-of-variables relation에서 유도된 geometry correction의 empirical estimator**

이다.

\(\lambda\)는 route diversity의 중요도를 조절하는 하이퍼파라미터이고, \(\eta\)는 geometry correction의 반영 강도를 조절한다.  
이론적으로는 \(\eta=1\)이 가장 자연스럽지만, 실제 구현에서는 calibration을 위해 조절할 수 있다.

---

## 17. 기존 방식과의 차이

기존 문서에서는 local uncertainty를 \(\ln K\)로 정의하고, 전체 score를

$$
U = \ln K \cdot (1+M_{\mathrm{rem}})
$$

와 같이 두었다.

본 문서에서는 local uncertainty를 더 직접적으로 entropy-calibrated quantity로 정의하고, global score는 남은 temporal distance와의 곱으로 단순화한다.

먼저 현재 노드에서 goal까지의 distance를

$$
T_{\mathrm{curr}} := \|z_{\mathrm{curr}}-z_g\|
$$

로 두고, 최종 uncertainty를

$$
\boxed{
U := T_{\mathrm{curr}} \cdot H_{\mathrm{local}}
}
$$

로 정의한다.

이 정의의 의미는 다음과 같다.

- \(H_{\mathrm{local}}\): 지금 이 순간 local branching uncertainty가 얼마나 큰가
- \(T_{\mathrm{curr}}\): goal까지 아직 얼마나 멀리 남아 있는가

따라서

> **goal에서 멀리 떨어져 있으면서 local uncertainty도 크면, 전체 planning uncertainty가 크다.**

이는 기존 \((1+M_{\mathrm{rem}})\) 구조보다 단순하며, 동시에 "remaining horizon \(\times\) local entropy"라는 직관을 직접 반영한다.

구현에서는 이 \(T_{\mathrm{curr}}\)를 embedding L2 자체로 두지 않고, 현재 노드의 embedding goal-distance를 `emb_dist_to_temporal_dist()`로 변환한 temporal distance로 사용한다.

---

## 18. 구현 명세

### 18.1 입력

- 현재 노드 state: \(s_{\mathrm{curr}}\)
- goal state: \(s_g\)
- tail states list: \(\{s_i\}_{i=1}^N\)
- encoder \(\phi\)
- numerical stability constant \(\varepsilon\)

### 18.2 계산 단계

1. 임베딩 계산:
   $$
   z_{\mathrm{curr}}=\phi(s_{\mathrm{curr}}), \quad z_g=\phi(s_g), \quad z_i=\phi(s_i)
   $$

2. goal distance:
   $$
   r_i=\|z_i-z_g\|
   $$

3. pairwise distance:
   $$
   d_{ij}=\|z_i-z_j\|
   $$

4. radial spread:
   $$
   S_R=\frac{1}{2N(N-1)}\sum_{i\ne j}(r_i-r_j)^2
   $$

5. raw off-radial residual (optional diagnostic):
   $$
   S_U=\frac{1}{2N(N-1)}\sum_{i\ne j}\Big(d_{ij}^2-(r_i-r_j)^2\Big)
   $$

6. normalized angular spread:
   $$
   S_A=
   \frac{1}{2N(N-1)}
   \sum_{i\ne j}
   \frac{d_{ij}^2-(r_i-r_j)^2}{r_i r_j+\varepsilon}
   $$

7. geometry correction:
   $$
   \widehat C_{\mathrm{geom}}
   =(m-1)\frac1N\sum_i \log(r_i+\varepsilon)
   $$

8. radial entropy term:
   $$
   H_R=\frac12\log\big(2\pi e(S_R+\varepsilon)\big)
   $$

9. angular entropy surrogate:
   $$
   H_A=
   \frac{m-1}{2}
   \log\left(
   \frac{\pi e}{m-1}(S_A+\varepsilon)
   \right)
   $$

10. local uncertainty:
   $$
   H_{\mathrm{local}}
   =
   H_R+\lambda H_A+\eta\widehat C_{\mathrm{geom}}
   $$

11. current distance:
   $$
   T_{\mathrm{curr}}=\mathrm{emb\_dist\_to\_temporal\_dist}\!\left(\|z_{\mathrm{curr}}-z_g\|\right)
   $$

12. final uncertainty:
   $$
   U=T_{\mathrm{curr}}H_{\mathrm{local}}
   $$

### 18.3 수치 안정성 노트

- \(r_i\)가 0에 매우 가까운 경우 \(u_i=(z_i-z_g)/r_i\)는 불안정해질 수 있다. 이를 방지하기 위해 \(r_i r_j+\varepsilon\)를 denominator에 사용한다.
- goal 근처의 샘플이 매우 많으면 angular term은 본질적으로 의미가 약해질 수 있다. 이는 실제로도 "goal 바로 근처에서는 방향보다 거리 자체가 중요하다"는 해석과 일치한다.
- \(S_U\)는 최종 estimator에는 직접 쓰지 않지만, raw off-radial residual을 모니터링하는 diagnostic quantity로 보존할 수 있다.

---

## 19. 최종 요약

이 문서의 핵심 논리는 다음과 같다.

1. planner가 원하는 uncertainty는 **goal-distance uncertainty**와 **route diversity**로 나뉜다.
2. 이 두 quantity는 pairwise spread의 exact decomposition \(S_{\mathrm{tot}}=S_R+S_U\)로 출발할 수 있다.
3. 한편 goal-centered polar coordinates에서 entropy는
   $$
   h(X)=h(R)+h(A\mid R)+(m-1)\mathbb E[\log R]
   $$
   로 exact decomposition 된다.
4. radial part \(h(R)\)는 1차원 Gaussian maximum entropy bound로 rigorous하게 upper bound할 수 있다.
5. angular part는 raw residual \(S_U\)를 그대로 쓰지 않고, normalized angular spread
   $$
   S_A=
   \frac{1}{2N(N-1)}
   \sum_{i\ne j}
   \frac{d_{ij}^2-(r_i-r_j)^2}{r_i r_j+\varepsilon}
   $$
   를 통해 pure directional spread로 정규화한다.
6. \(S_A\)는 unit-direction variable \(A\)의 pairwise second moment를 empirical하게 측정하며,
   $$
   h(A\mid R)\le h(A)
   $$
   와 pairwise entropy bound를 결합하면 \(h(A\mid R)\)의 upper-bound surrogate를 구성할 수 있다.
7. geometry correction은 Jacobian \(r^{m-1}\)에서 정확히 유도되며, empirical average \(\widehat C_{\mathrm{geom}}\)로 구현된다.
8. 따라서 최종적으로
   $$
   H_{\mathrm{local}}
   =
   \frac12\log(2\pi e(S_R+\varepsilon))
   +
   \lambda\frac{m-1}{2}
   \log\left(
   \frac{\pi e}{m-1}(S_A+\varepsilon)
   \right)
   +
   \eta \widehat C_{\mathrm{geom}}
   $$
   는 **entropy-consistent radial–angular uncertainty measure**가 된다.
9. 여기에 남은 distance-to-go \(T_{\mathrm{curr}}\)를 곱해
   $$
   U=T_{\mathrm{curr}}H_{\mathrm{local}}
   $$
   를 최종 branching score로 사용한다.

즉, 본 문서의 제안은 다음 한 문장으로 요약할 수 있다.

> **우리는 embedding covariance의 축분해 대신, goal-relative pairwise spread decomposition을 사용하여 local uncertainty를 radial entropy upper bound와 normalized angular pairwise entropy surrogate의 합으로 정의하고, 이를 remaining distance로 가중한 최종 uncertainty score를 제안한다.**
