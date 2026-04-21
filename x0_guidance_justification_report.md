# `x0` Direct Guidance Justification Report

## Executive Summary

이 저장소의 현재 구현은 `configurations/algorithm/df_planning.yaml`에서 `use_directly_inject_guidance_to_x0: true`로 설정되어 있으며, diffusion 모델 자체도 `pred_x0` objective로 학습/추론된다 (`configurations/algorithm/ckpt_df_planning.yaml`).

핵심은 다음이다.

1. 현재 코드의 비교는 "`x0` 기반 objective" 대 "`xt` 기반 objective"의 비교가 아니다. 두 경우 모두 guidance loss는 `pred_x_start = x̂0` 위에서 계산된다.
2. 진짜 차이는 **guidance를 어디에 주입하느냐**이다.
   - DPS-style branch: `∂V/∂x_t`를 이용해 `pred_noise`를 수정한다.
   - direct-`x0` branch: `∂V/∂x̂0`를 직접 사용해 `x̂0`를 수정한 뒤, 그에 맞는 `pred_noise`를 다시 계산한다.
3. 이 저장소에서는 guidance가 trajectory 전체에 dense하게 걸리지 않고, **segment tail/head 같은 소수의 시점**과 **`x, y` 좌표 같은 소수의 차원**에만 걸린다.
4. 따라서 `xt` branch에서는 clean-space의 sparse/local guidance가 denoiser Jacobian을 통해 **시간축과 비-guided 차원으로 leakage**되기 쉽고, 이 leakage가 plan feasibility를 해칠 가능성이 높다.
5. Maze 환경은 wall/bottleneck 근처에서 guidance 방향이 급격히 꺾인다. 이런 고곡률 환경에서는 Jacobian을 한 번 더 통과한 `xt` guidance보다, clean trajectory estimate인 `x̂0`를 직접 수정하는 방식이 더 faithful할 가능성이 크다.

내 판단으로, 이 환경에서 `x0` direct guidance가 더 잘 나오는 **가장 설득력 있는 1차 원인**은 다음의 조합이다.

- `pred_x0` parameterization
- sparse, segment-boundary-centered guidance
- partial-coordinate guidance (`x, y`만 직접 지도)
- maze의 비선형/고곡률 value landscape

즉, "`maze라서`"만도 아니고, "`tail만 guidance해서`"만도 아니다. **local/partial guidance를 `pred_x0` 모델에 넣는 구조에서, `xt` 주입이 Jacobian-mediated distortion을 만들기 때문**이라는 설명이 가장 강하다. Maze 구조와 tail-only 설계는 그 차이를 더 크게 드러내는 증폭 요인으로 보는 것이 적절하다.

---

## 1. What The Current Code Actually Does

### 1.1 Direct `x0` guidance is explicitly enabled

- `configurations/algorithm/df_planning.yaml:58-60`
  - `use_directly_inject_guidance_to_x0: true`
  - `direct_x0_guidance_scale: 0.2`
  - `direct_x0_eta_cap: 1.0`

### 1.2 The diffusion model is a `pred_x0` model

- `configurations/algorithm/ckpt_df_planning.yaml:34-42`
  - `diffusion.objective: pred_x0`

이는 매우 중요하다. 모델이 기본적으로 예측하는 주된 대상 자체가 `x̂0`이므로, auxiliary guidance를 `x̂0` 공간에서 해석하는 것이 구조적으로 자연스럽다.

### 1.3 Both branches compute the guidance objective on `x̂0`

`algorithms/diffusion_forcing/models/diffusion.py:508-547`를 보면:

- 먼저 `model_pred.pred_x_start`를 얻고
- `guidance_fn(model_pred.pred_x_start)`를 호출한다.

즉, 현재 코드에서 guidance loss `V`는 원래부터 `x̂0` 위에서 정의된다. 그 다음 분기만 달라진다.

- `algorithms/diffusion_forcing/models/diffusion.py:549-551`
  - `xt` branch: `pred_noise`를 `∂V/∂x_t`로 수정
  - direct-`x0` branch: `x_start`를 `∂V/∂x̂0`로 수정

이 점 때문에, 논문에서는 "`우리 방식은 clean-sample objective를 쓴다`"가 아니라,

> "우리는 same clean-sample objective를 유지한 채, 그 corrective step을 latent/noisy state인 `x_t`에 주입하는 대신 denoised estimate인 `x̂0`에 직접 주입한다"

라고 쓰는 것이 정확하다.

### 1.4 Current guidance is sparse in time and sparse in coordinates

#### Goal guidance is applied only at segment tails

- `algorithms/diffusion_forcing/guidance.py:165-179`
  - `get_segment_tail_positions(...)`
- `algorithms/diffusion_forcing/guidance.py:322-326`
  - `tail_pos = get_segment_tail_positions(...)`
- `algorithms/diffusion_forcing/guidance.py:404-458`
  - HILP/RMSE guidance is assembled only at tail positions

#### Anchor/RDF guidance is also local

- `algorithms/diffusion_forcing/guidance.py:484-544`
  - `anchor_dist_guidance`: segment head를 직전 tail 쪽으로 당김
- `algorithms/diffusion_forcing/guidance.py:546-649`
  - `segment_rdf_guidance`: segment tail을 same-segment head에서 밀어냄

#### Guidance is applied mainly on position dimensions

- `algorithms/diffusion_forcing/guidance.py:384-385`
- `algorithms/diffusion_forcing/guidance.py:456-458`
- `algorithms/diffusion_forcing/guidance.py:530-531`
- `algorithms/diffusion_forcing/guidance.py:624-625`

모두 `planner.pos_dim_indices`를 통해 `x, y` 같은 spatial dimensions에 guidance를 건다.

즉, 이 guidance는 trajectory token 전체를 densely supervise하는 것이 아니라,

- 시간적으로는: tail/head boundary 위주
- 차원적으로는: `x, y` 위주

인 **sparse/local/partial guidance**다.

### 1.5 Feasibility is sensitive to frame-to-frame discontinuity

- `algorithms/diffusion_forcing/df_planning.py:1617-1641`

여기서 feasibility는 연속 프레임 간 distance가 `plan_feasibility_delta`보다 작은지, 그리고 stagnant하지 않은지를 본다. 따라서 guidance가 시간축 전체로 퍼져서 local discontinuity를 만들면, success보다 먼저 feasibility가 깨질 수 있다.

---

## 2. Mathematical Difference Between DPS-Style `xt` Injection and Direct `x0` Injection

코드가 쓰는 표기와 대응되는 직관적 수식만 적으면 다음과 같다.

모델이 `pred_x0`이므로

\[
\hat x_0 = \frac{1}{\sqrt{\bar\alpha_t}} x_t - \sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}} \hat\epsilon_t
\]

### 2.1 DPS-style `xt` injection

코드:

- `algorithms/diffusion_forcing/models/diffusion.py:618-623`

즉,

\[
\hat\epsilon_t' = \hat\epsilon_t - g_t,
\quad
g_t := \nabla_{x_t} V(\hat x_0)
\]

그러면 implied clean update는

\[
\hat x_0'
= \hat x_0 + \sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}\, g_t
\]

그런데 chain rule에 의해

\[
g_t
= \nabla_{x_t}V(\hat x_0)
= J_\theta(x_t,t)^\top \nabla_{\hat x_0}V(\hat x_0)
\]

이므로 실제로는

\[
\hat x_0'
= \hat x_0
+ \sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}
+ J_\theta^\top \nabla_{\hat x_0}V(\hat x_0)
\]

가 된다.

즉, clean-space에서 의도한 방향이 **denoiser Jacobian transpose에 의해 회전/확대/축소/누출**된다.

### 2.2 Direct `x0` injection

코드:

- `algorithms/diffusion_forcing/models/diffusion.py:591-616`

즉,

\[
\hat x_0' = \hat x_0 + \eta_t \nabla_{\hat x_0}V(\hat x_0),
\quad
\eta_t = \kappa \sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}
\]

그 다음

\[
\hat\epsilon_t' = \text{reconstruct\_noise}(x_t, \hat x_0')
\]

를 통해 DDIM-consistent noise prediction을 다시 맞춘다.

### 2.3 The key difference

정리하면, 두 방식의 차이는 거의 한 줄이다.

- `xt` injection:
  \[
  \Delta \hat x_0 \propto J_\theta^\top \nabla_{\hat x_0}V
  \]
- direct `x0` injection:
  \[
  \Delta \hat x_0 \propto \nabla_{\hat x_0}V
  \]

따라서 현재 실험의 질문은 사실상

> "`J_\theta^\top`를 한 번 더 거친 guidance가 더 좋은가, 아니면 clean-space gradient를 그대로 쓰는 게 더 좋은가?"

의 질문이다.

이 저장소의 설계에서는 후자가 더 유리할 가능성이 크다.

---

## 3. Related Work That Can Support This Design

정직하게 말하면, **MCTS-style maze segment planner에서 exact same update rule을 직접적으로 평가한 대표 논문은 찾지 못했다.**  
하지만 아래 문헌들은 현재 구현을 충분히 정당화할 수 있는 강한 근거를 제공한다.

### 3.1 Closest precedent: clean/denoised-space guidance

#### (A) Universal Guidance for Diffusion Models

- Arpit Bansal et al., 2023
- Link: <https://arxiv.org/abs/2302.07121>

핵심 포인트:

- guidance model은 noisy latent가 아니라 **denoised image**에서 평가하는 것이 낫다고 주장한다.
- 논문은 아예 clean-data space에서 guided change를 구한 뒤, 그 변화를 noisy space로 선형적으로 옮기는 **backward universal guidance**를 제안한다.

왜 중요한가:

- 현재 코드의 direct-`x0` branch와 가장 가까운 precedent다. 정확히 같은 update rule은 아니지만, "clean-space에서 guidance를 계산/적용하고, 필요하면 noisy state로 다시 옮긴다"는 발상 자체가 매우 유사하다.
- 특히 논문 본문은 "guidance models are trained on clean images"이므로 noisy state에서 직접 guidance하면 domain gap이 생긴다고 지적한다.
- 이 저장소의 HILP/RMSE/anchor/RDF guidance도 본질적으로 clean trajectory semantics에 대한 signal이다. noisy latent보다 clean trajectory estimate에서 해석하는 편이 더 자연스럽다.

논문에서 바로 가져올 수 있는 메시지:

> off-the-shelf guidance should be evaluated and applied in the denoised domain when the external guidance is semantically defined on clean samples rather than noisy states.

#### (B) Towards Accurate Guided Diffusion Sampling through Symplectic Adjoint Method (SAG)

- Jiachun Pan et al., 2023
- Link: <https://arxiv.org/abs/2312.12030>

핵심 포인트:

- 기존 training-free guidance는 clean image의 one-step estimate를 쓰는데, early timestep에서는 그 estimate가 부정확해 guidance가 부정확해질 수 있다고 지적한다.
- 더 accurate한 clean estimate를 바탕으로 guidance를 계산하는 것이 좋다고 주장한다.

왜 중요한가:

- direct-`x0`가 무조건 좋다는 논문은 아니지만, 적어도 **guidance 품질은 clean estimate의 품질과 alignment에 크게 좌우된다**는 근거를 준다.
- 당신의 case에서는 guidance 자체가 sparse subgoal/segment semantics 위에 정의되므로, `x̂0`를 직접 조정하는 것이 특히 더 의미가 있다.

### 3.2 Mainstream baseline you are deviating from

#### (C) Diffusion Posterior Sampling for General Noisy Inverse Problems (DPS)

- Hyungjin Chung et al., 2022/2024
- Link: <https://arxiv.org/abs/2209.14687>

핵심 포인트:

- noisy inverse problems에서 posterior sampling을 위한 mainstream guidance 계열.
- likelihood/objective는 clean estimate와 관련되지만, corrective step은 latent/noisy process 쪽에 주입된다.

왜 중요한가:

- 논문에서 "우리는 DPS-style latent guidance가 아니라 clean-estimate guidance를 사용한다"는 비교축을 세울 수 있다.
- 다만 reviewer에게는 이 점도 함께 말해야 한다.

중요한 nuance:

> DPS는 inverse problem posterior correction을 위해 설계된 방법이다.  
> 당신의 guidance는 full observation likelihood가 아니라, sparse task energy / segment connection / goal-reaching signal이다.

즉, **inverse-problem posterior sampler의 mainstream을 planning energy guidance에 그대로 따르는 것이 항상 최선은 아니다.**

### 3.3 Guidance misalignment at high noise is a known problem

#### (D) Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models

- Tuomas Kynkäänniemi et al., 2024
- Link: <https://arxiv.org/abs/2404.07724>

핵심 포인트:

- guidance는 sampling chain 초반 high-noise stage에서 해롭고, 중간 구간에서만 유효하다고 보고한다.

왜 중요한가:

- 현재 코드의 `direct_x0_eta_cap`은 high-noise에서 direct guidance가 과해지는 것을 막는 안정화 장치다.
- 논문적으로는 "`x0` direct guidance가 안정적인 이유 중 일부는, 우리가 high-noise에서의 guidance explosion을 명시적으로 제어했기 때문"이라고 설명할 수 있다.

중요한 주의점:

- 이 논문은 `x0` direct를 주장하는 것은 아니다.
- 하지만 **high-noise guidance 자체가 위험하다**는 점을 뒷받침한다.

#### (E) Enhancing Diffusion Posterior Sampling for Inverse Problems by Integrating Crafted Measurements (DPS-CM)

- Shijie Zhou et al., 2024/2025
- Link: <https://arxiv.org/abs/2411.09850>

핵심 포인트:

- naive posterior guidance는 early stages에서 prior와 misalignment를 일으켜 누적 오차를 키운다고 지적한다.
- 개선된 conditioning/guidance construction이 sampling quality를 높인다고 주장한다.

왜 중요한가:

- direct-`x0` branch가 더 잘 나오는 현상을 "`guidance information should be injected in a representation that is aligned with the current denoising stage and prior geometry`"라는 broader literature와 연결할 수 있다.

### 3.4 Support from planning literature: representation matters for guidance

#### (F) GPD: Guided Polynomial Diffusion for Motion Planning

- Ajit Srikanth et al., 2025
- Link: <https://arxiv.org/abs/2501.18229>

핵심 포인트:

- motion planning에서 diffusion representation을 Bernstein coefficient space로 바꾸면 **cost guidance effectiveness**와 inference speed가 크게 좋아진다고 주장한다.

왜 중요한가:

- planning literature에서도 guidance의 성능은 "guidance 자체"만이 아니라 **어떤 representation/parameterization 위에서 guidance를 거느냐**에 강하게 의존한다는 근거다.
- 당신의 결과를 "representation choice matters; in our planner, clean trajectory estimate `x̂0` is a better guidance space than noisy latent `x_t`"로 연결할 수 있다.

#### (G) EDMP: Ensemble-of-costs-guided Diffusion for Motion Planning

- Kallol Saha et al., 2023/2024
- Link: <https://arxiv.org/abs/2309.11414>

핵심 포인트:

- motion planning에서 inference-time external cost guidance 자체가 충분히 정당한 접근임을 보여준다.

왜 중요한가:

- 당신의 HILP/RMSE/anchor/RDF guidance가 "planning-time external energy guidance"라는 broader family 안에 있다는 점을 뒷받침한다.

### 3.5 RL literature support: intermediate guidance is hard

#### (H) Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning

- Cheng Lu et al., 2023
- Link: <https://arxiv.org/abs/2304.12824>

핵심 포인트:

- energy-guided diffusion에서 intermediate guidance는 본질적으로 hard-to-estimate라고 지적한다.
- naive heuristic guidance보다 exact intermediate guidance를 학습하는 것이 낫다고 주장한다.

왜 중요한가:

- planning / RL 문맥에서도 "`중간 sampling state에서 guidance를 어떻게 정의하느냐`"가 핵심 문제라는 근거다.
- 당신의 결과는 이 문제에 대한 한 practical answer로 볼 수 있다: **적어도 현재 planner에서는 `x_t` intermediate guidance보다 `x̂0` direct correction이 더 잘 작동한다.**

---

## 4. My Main Claim: Why `x0` Guidance Is Better In This Codebase

여기서는 설명을 "가능성 나열"이 아니라, **우선순위가 있는 주장**으로 정리한다.

### Claim 1. The strongest reason is `pred_x0` alignment

이 모델은 `pred_x0` objective로 학습되며, inference에서도 먼저 `x̂0`를 예측한다.  
그런데 `xt` guidance는 그 clean prediction에 대해 정의된 gradient를 다시 `x_t` 쪽으로 backpropagate해서 사용한다.

이것은 곧,

- 모델이 직접 예측한 semantic object: `x̂0`
- external guidance가 의미를 가지는 object: clean trajectory / goal geometry / boundary geometry

둘이 모두 `x̂0`에 정렬되어 있는데, 굳이 Jacobian을 거쳐 `x_t`로 옮긴 뒤 다시 `x̂0`를 재구성하는 셈이다.

논문식으로 더 간단히 말하면:

> When the denoiser is parameterized to predict `x0` and the auxiliary objective is naturally defined on the denoised trajectory, directly updating `x̂0` avoids an unnecessary Jacobian-mediated mismatch.

### Claim 2. Sparse/local/partial guidance makes `xt` leakage especially harmful

이 저장소에서 guidance는 trajectory 전체나 모든 bundle dimensions에 걸리지 않는다.

- 시간적으로는 segment tail/head에만 걸린다.
- 차원적으로는 대부분 `x, y`에만 걸린다.

그런데 `xt` branch는

\[
\nabla_{x_t}V = J_\theta^\top \nabla_{\hat x_0}V
\]

를 쓰므로, 원래 tail의 `x,y`에만 있던 clean gradient가

- 다른 시점
- 다른 obs dims
- 심지어 action/reward bundle

으로 퍼질 수 있다.

이 leakage는 당신의 환경에서 특히 나쁘다. 이유는 feasibility가 frame-to-frame smoothness에 민감하기 때문이다.

보다 직접적으로 말하면:

- direct-`x0` branch는 "tail의 `x,y`를 살짝 옮기고, 그에 맞는 noise를 다시 계산"한다.
- `xt` branch는 "tail의 `x,y`에서 나온 요구사항을 latent state 전체에 퍼뜨리는" 쪽에 가깝다.

segment stitching 문제에서는 전자가 더 자연스럽다.

### Claim 3. Tail-only guidance is probably part of the reason, but not the whole reason

사용자 가설 중 "segment 생성에서 tail 부분만 guidance를 주기 때문인가?"는 **맞는 방향**이다. 다만 더 정확히는 다음처럼 써야 한다.

> Tail-only guidance 자체가 핵심이라기보다는, tail-only guidance처럼 local objective를 쓰는 상황에서 `xt` guidance의 non-local leakage가 더 치명적으로 드러난다.

즉,

- if dense all-frame objective: `xt` leakage가 덜 문제될 수 있음
- if sparse boundary objective: `xt` leakage가 훨씬 큰 문제

이므로 tail-only 설계는 direct-`x0`의 이점을 증폭하는 요인이다.

### Claim 4. Maze geometry likely amplifies the gap

Maze 환경의 HILP/RMSE guidance field는 open space에서는 비교적 smooth할 수 있지만,

- wall 근처
- bottleneck 진입 전후
- shortest path direction이 급히 바뀌는 corner

에서는 방향 변화가 급격하다.

이런 고곡률 영역에서는 clean-space gradient의 의미가 매우 국소적이다. 그런데 `xt` branch는 그 국소적 gradient를 denoiser Jacobian을 통해 다른 시점/차원으로 섞는다. 그러면 원래 의도했던 "이 tail을 여기 방향으로 보내라"는 명령이 trajectory 전체의 이상한 흔들림으로 바뀔 수 있다.

따라서 내 판단은 다음이다.

- "`maze gradient가 급격히 바뀌어서` direct-`x0`가 더 좋다"는 설명은 **부분적으로 맞다.**
- 하지만 더 근본적인 설명은 "**sparse clean-space objective를 `xt`로 옮기면서 생기는 distortion**"이다.
- maze geometry는 그 distortion의 피해를 크게 만드는 환경적 조건이다.

### Claim 5. Robustness to noisy external guidance is also consistent with `x0` direct update

당신의 관찰대로 noisy guidance에도 `x0` direct가 robust했다면, 이는 다음과 같이 해석할 수 있다.

1. noisy guidance는 원래 tail/head `x,y` 같은 semantically meaningful coordinates에서 생긴다.
2. direct-`x0`는 그 coordinates에만 직접 correction을 건다.
3. 이후 sampler는 그 수정된 clean estimate에 맞는 noise를 재구성하고, 다음 denoising step에서 prior consistency를 다시 회복한다.

반면 `xt` branch에서는 noisy guidance가 Jacobian을 타고 latent 전체에 섞일 수 있다. 그러면 noise in guidance가 곧 noise in latent coupling으로 번질 수 있다.

즉, robustness 측면에서도 direct-`x0`가 더 그럴듯하다.

### Claim 6. But some of the gain may come from the stabilizers, not only the space choice

이건 reviewer가 가장 먼저 물을 수 있는 지점이다.

현재 config를 보면:

- 두 branch 모두 `diffusion.max_guidance_ratio` soft clipping은 받는다.
- 그러나 direct-`x0` branch에는 추가로 `direct_x0_eta_cap`이 있다.

즉, 현재 결과를 그대로 논문에 쓰면 reviewer가

> "좋아진 게 `x0` space 때문인지, 아니면 `eta_cap`이라는 extra stabilizer 때문인지 분리되어 있지 않다"

고 지적할 수 있다.

이 지적은 타당하다. 따라서 논문에서는 반드시 다음과 같이 써야 한다.

> We attribute the improvement primarily to clean-space guidance, but note that high-noise stabilization (`eta` capping) is an important implementation detail and should be controlled for in ablations.

---

## 5. The Most Defensible Paper Narrative

현재 상태에서 가장 방어력 높은 narrative는 아래다.

### Recommended main narrative

> Our planner differs from DPS-style latent guidance by injecting guidance directly into the denoised trajectory estimate `x̂0`, rather than into the noisy state `x_t`. This choice is particularly well-motivated in our setting because the diffusion model is parameterized as `pred_x0`, while the auxiliary guidance is defined on sparse, semantically meaningful clean-trajectory coordinates (segment tails/heads and 2D positions). Under these conditions, `x_t`-space guidance introduces a Jacobian-mediated distortion that can leak corrections across time and unguided dimensions, whereas `x̂0`-space guidance preserves locality. Empirically, this yields substantially higher feasibility and stronger robustness to noisy external guidance in maze planning.

### Recommended secondary narrative

> The advantage of `x̂0` guidance is amplified in maze environments, where the value/guidance field changes abruptly near walls and bottlenecks. In such high-curvature regions, preserving the intended clean-space direction of the guidance is more important than in open-space settings.

### What not to say too strongly

다음 문장은 과도하다.

> "`x0` guidance is universally better than DPS-style `xt` guidance."

더 정확한 문장은 이렇다.

> "`x0` guidance is better in our planner, especially under sparse segment-boundary guidance and `pred_x0` parameterization."

---

## 6. Suggested Additional Analyses For The Paper

이 섹션은 꼭 중요하다. 지금 결과만으로도 서술은 가능하지만, 아래 분석이 있으면 논문 방어력이 훨씬 올라간다.

### 6.1 Matched ablation: separate "space effect" from "stabilizer effect"

필수.

실험:

1. `xt` guidance, current default
2. `x0` guidance, current default
3. `x0` guidance, `direct_x0_eta_cap = null`
4. `xt` guidance + matched high-noise attenuation schedule

보고할 metric:

- feasible plan ratio
- final success rate
- replanning count
- guidance-noise robustness curve

이 ablation이 있어야 "`x0`라서 좋아진 것"과 "`cap 때문에 좋아진 것`"을 분리할 수 있다.

### 6.2 Jacobian distortion analysis

핵심 진단.

각 sampling step에서 다음을 계산:

\[
\Delta \hat x_0^{(xt)}
= \sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}} \nabla_{x_t}V
\]

그리고

\[
g_{clean} = \nabla_{\hat x_0}V
\]

비교 항목:

- cosine similarity between `Δx̂0^(xt)` and `g_clean`
- norm ratio `||Δx̂0^(xt)|| / ||g_clean||`
- noise-step별 variance

만약 cosine이 낮거나 variance가 크면, "`xt` branch가 clean-space direction을 충실히 보존하지 못한다"는 직접 증거가 된다.

### 6.3 Guidance leakage analysis

아주 중요하고, 현재 코드 구조와 잘 맞는다.

정의:

- guided coordinates:
  - active tail/head positions
  - `pos_dim_indices`
- non-guided coordinates:
  - 나머지 시간축
  - 나머지 obs/action/reward dims

측정:

\[
r_{\text{leak}}
= \frac{\|\Delta_{\text{non-guided}}\|}{\|\Delta_{\text{all}}\|}
\]

예상:

- direct-`x0`: leak ratio가 작아야 함
- `xt`: leak ratio가 커야 함

이 분석이 잘 나오면, 논문의 핵심 주장을 거의 직접적으로 증명한다.

### 6.4 Tail-only hypothesis test

실험:

1. all tails
2. last tail only
3. dense all-frame guidance

를 각각 `x0`/`xt`에서 비교.

예상:

- sparse할수록 direct-`x0` advantage가 커질 가능성이 높다.

이 결과가 나오면 "`tail-only guidance라서 x0가 더 유리하다`"를 정제된 방식으로 주장할 수 있다.

### 6.5 Maze geometry dependence

각 planning instance를 다음으로 stratify:

- bottleneck proximity
- local wall curvature
- shortest-path turning angle

그리고 `x0 - xt` 성능 gap을 각 구간에서 비교.

예상:

- turn이 급하고 bottleneck이 심한 구간일수록 `x0` advantage가 커질 수 있다.

이 분석이 들어가면 "`maze gradient direction changes abruptly`" 가설을 단순 speculation이 아니라 데이터 기반 주장으로 바꿀 수 있다.

### 6.6 Robustness-to-guidance-noise experiment

외부 guidance에 artificial noise를 추가:

- gradient direction noise
- goal perturbation
- HILP value perturbation

보고:

- feasibility
- success
- plan smoothness

이 실험은 이미 관찰과 맞아 떨어질 가능성이 크고, reviewer 설득력이 높다.

### 6.7 Smoothness / stitching diagnostics

현재 planner와 특히 잘 맞는 metric:

- max per-frame jump
- mean per-frame jump
- segment head-anchor distance
- segment tail-to-goal distance
- RDF collision/repulsion violation

예상:

- `x0` direct는 success뿐 아니라 boundary stitching quality도 개선해야 한다.

---

## 7. Recommended Wording For The Paper

### Short version for the main paper

> We inject guidance directly into the denoised trajectory estimate `x̂0` instead of the noisy latent `x_t`. This design is motivated by two properties of our planner: the denoiser is parameterized as a `pred_x0` model, and the auxiliary guidance is defined only on sparse, semantically meaningful trajectory coordinates such as segment tails/heads and 2D positions. In this regime, `x_t`-space guidance introduces a Jacobian-mediated distortion that spreads local corrections across time and unguided dimensions, which harms plan feasibility. Direct `x̂0` guidance preserves locality and empirically yields substantially better feasibility and stronger robustness to noisy guidance.

### Slightly more careful version

> Our method departs from DPS-style latent guidance by applying the correction in clean trajectory space. This is not claimed to be universally superior; rather, it is particularly effective in our setting because the denoiser predicts `x0` directly and the external guidance acts on sparse boundary states in maze planning. Under these conditions, clean-space guidance is better aligned with the model parameterization and less prone to leaking updates into unguided coordinates.

### One-sentence reviewer-proof note

> We additionally cap the high-noise clean-space step size, following the broader observation that overly strong early-stage guidance is harmful in diffusion sampling.

---

## 8. Final Position

내 최종 입장은 다음과 같다.

1. **선행연구 근거는 충분히 있다.**
   - exact same planner는 아니지만,
   - `Universal Guidance`가 가장 가까운 clean-space precedent이고,
   - `SAG`, `Limited Guidance Interval`, `DPS-CM`이 "guidance는 clean estimate alignment와 noise-stage alignment가 중요하다"는 점을 뒷받침하며,
   - `GPD`, `EDMP`, `CEP`가 planning/RL 문맥에서 guidance representation과 intermediate guidance quality가 중요하다는 점을 보강한다.

2. **왜 `x0`가 더 좋은지에 대한 가장 강한 설명은 Jacobian distortion/leakage다.**
   - 단순히 tail-only라서가 아니라,
   - tail-only + partial-coordinate guidance + `pred_x0` model이기 때문에 `xt` leakage가 치명적이다.

3. **maze geometry explanation은 보조 가설로 쓰는 것이 좋다.**
   - 환경의 sharp turn/bottleneck이 gap을 키웠을 가능성은 높다.
   - 하지만 primary mechanism은 clean-space alignment와 locality preservation이다.

4. **논문에서는 `eta_cap` confound를 반드시 통제해야 한다.**
   - 그렇지 않으면 reviewer가 "`x0` space effect" 해석을 약화시킬 수 있다.

한 문장으로 요약하면:

> 이 코드베이스에서 direct-`x0` guidance가 더 잘 되는 가장 설득력 있는 이유는, guidance가 clean trajectory의 sparse boundary coordinates에 정의되어 있는데 `xt` injection이 그 local signal을 denoiser Jacobian을 통해 비국소적으로 왜곡시키기 때문이다. Maze의 sharp geometry와 tail-centered guidance 구조는 이 차이를 더 크게 만든다.

---

## References

1. Hyungjin Chung, Jeongsol Kim, Michael T. McCann, Marc L. Klasky, Jong Chul Ye. "Diffusion Posterior Sampling for General Noisy Inverse Problems." ICLR 2023. <https://arxiv.org/abs/2209.14687>
2. Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, Tom Goldstein. "Universal Guidance for Diffusion Models." ICML 2023. <https://arxiv.org/abs/2302.07121>
3. Jiachun Pan, Hanshu Yan, Jun Hao Liew, Jiashi Feng, Vincent Y. F. Tan. "Towards Accurate Guided Diffusion Sampling through Symplectic Adjoint Method." 2023. <https://arxiv.org/abs/2312.12030>
4. Tuomas Kynkäänniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, Jaakko Lehtinen. "Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models." NeurIPS 2024. <https://arxiv.org/abs/2404.07724>
5. Shijie Zhou, Huaisheng Zhu, Rohan Sharma, Jiayi Chen, Ruiyi Zhang, Kaiyi Ji, Changyou Chen. "Enhancing Diffusion Posterior Sampling for Inverse Problems by Integrating Crafted Measurements." 2024. <https://arxiv.org/abs/2411.09850>
6. Cheng Lu, Huayu Chen, Jianfei Chen, Hang Su, Chongxuan Li, Jun Zhu. "Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning." ICML 2023. <https://arxiv.org/abs/2304.12824>
7. Kallol Saha, Vishal Mandadi, Jayaram Reddy, Ajit Srikanth, Aditya Agarwal, Bipasha Sen, Arun Singh, Madhava Krishna. "EDMP: Ensemble-of-costs-guided Diffusion for Motion Planning." ICRA 2024. <https://arxiv.org/abs/2309.11414>
8. Ajit Srikanth, Parth Mahanjan, Kallol Saha, Vishal Mandadi, Pranjal Paul, Pawan Wadhwani, Brojeshwar Bhowmick, Arun Singh, Madhava Krishna. "GPD: Guided Polynomial Diffusion for Motion Planning." 2025. <https://arxiv.org/abs/2501.18229>
