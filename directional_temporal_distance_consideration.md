[PROMPT]
antmaze-teleport env의 경우 teleport 입구와 출구가 존재한다. 이때 agent는 입구에 다가가면 출구로 teleport되지만, 반대로 출구에 간다고 입구로 teleport 되진 않는다. 즉, agent의 이동 경로 planning에 directional 한 특성이 있다.

현재 diffusion model planner는 ./train.sh 를 통해 학습되며, agent가 env에서 이동했던 offline rollout episode를 학습 데이터로 쓰면서 경로를 생성한다. 즉, diffusion 모델이 생성하는 경로는 agent가 이동했던 시퀀스의 순서와 동일한 순서를 가진다.

start에서 시작하여 goal에 도달하는 순방향 planning을 할 때는 이 diffusion 모델 output의 순서가 크게 상관없지만, 문제는 우리의 planning 알고리즘은 start를 기점으로 plan을 만들고, 동시에 goal을 기점으로 plan을 만들어서 중간에서 둘을 stitch하는 방식이라는 것이다. 이는 plan의 양방향성(즉, 주어진  plan을 head에서 tail로 가는 것이나 tail에서 head로 가는 것이나 둘 다 가능한 성질)을 가정하고 만든 설계지만, teleport라는 특이한 환경에서는 이 양방향성이 성립하질 않는다. 

이를 해결하기 위해 goal conditioned directional temporal distance estimator를 학습 완료하였다. goal conditioned temporal distance란  현재 state s로부터 목표 g 까지 최단 action이 얼마나 소모되는지를 value function 형태로 구한 지표로, 일반적인 경우엔 V(s, g) = V(g, s)이기에 기존 코드에서는 V(s, g) := |phi(s) - phi(g)| 형태로 value function을 설계하여 encoder phi 를 학습하는 구조였지만, teleport 같은 directional nature을 가진 env에서는 s, g의 방향성이 존재하기에 V(s, g) = |ReLU(phi(s) - phi(g))|  와 같은 형태로 temporal dist를 학습하여 V(s, g) ≠ V(g, s) 를 estimator 설계에 반영하였고, 학습된 모델 결과를 시각화해보니 teleport 출구에 g 가 위치할 때 입구에서 s가 존재할 때는 temporal dist가 낮게 나왔지만, 반대로 입구에 g가 있고 출구에 s가 있으면 temporal dist가 높게 나오는 것을 확인했다. (cf.  ../HILP/hilp_gcrl/run_train.sh 를 통해 어떻게 quasimetric에서 모델 학습하는지 확인 가능)

이제 다음 구현 단계는 잘 학습된 directional temporal dist estimator(이하 hilp)를 기존 코드에 적용하는 일이야. 현재 hilp는 알고리즘 상 정말 다양한 곳에 쓰이고 있다. 생성된 시퀀스의 feasibility를 검사할 때 쓰이기도 하고, 각 노드의 value를 계산할 때도 temporal dist 기반으로 value가 산정되고 있고, diffusion 모델의 guidnace로도 temporal dist가 사용되는 등등.. 그 외에도 너는 ‘hilp가 사용되는 모든 관련 코드를 찾아내는 것’이 제일 먼저 해야할 일이야. 

hilp가 사용되는 코드 범위를 리스팅했다면, 다음으로 해야할 일은 ‘방향성’을 확인하는 것이다. 대전제는 다음과 같아: agent는 반드시 S에서 G로 향한다는 것. 때문에 현재 online hemiltonian planner (혹은 fixed temporal dist planner일 수도 있으나 둘 중 뭘 사용해도 아래의 내용은 일반성을 잃지 안는다)은 S에서 시작하여 G로 끝나도록 {S, G, waypoint1, waypoint2, …} 의 순서를 바꿔가면서 최적의 헤밀턴 경로 [S, W_i, … , W_j, G] 를 반환하고 있고,  {S, G, waypoint1, waypoint2, …} 각각에 해당하는 tree는 edge의 순서 상관없이 자신과 인접한 다른 tree와 접합이 되게 하고있어. (자세한 로직은 ./multi-tree_hemiltonian_path_implementation_plan.md 참고) 예를 들어 online hemiltonian planner 가 이번 라운드에 설정한 헤밀턴 경로가 [S, W2, W1, W3, G] 라고 해보자. 이러면 W2 기준으로 연결해야하는 tree는 S와 W1이 되고, edge 기준으로 보면 (S→W2) 와 (W2→W1) 이 된다. 즉, 단순히 이웃한 tree를 연결하는 것을 넘어서, 자신이 속한 edge 를 추출하여 자신(W2)이 역방향(S→W2)에서 시퀀스를 생성하는 입장인지, 순방향(W2→W1)에서 시퀀스를 생성하는 입장인지 파악할 수 있다는 점이야. 이렇게 W2 에 속한 node가 expand할 때 자신이 target으로 하는 노드가 속한 tree를 기준으로 자신이 순방향/역방향 시퀀스를 생성하는지 여부를 판단할 수 있다면, hilp를 사용하는 모든 코드에 대해서 어느 방향성으로 value function에 s와 g를 넣어야하는지 알 수 있어. 

현재 떠오르는 한 가지 예외사항이 있는데, configurations/algorithm/df_planning.yaml의 reliable_TD_threshold 값에 따라서 online hemiltonian planner를 사용하여 얻은 헤밀턴 경로와 무관하게 임의의 tree 에 속한 node를 target으로 지정해버리면 방향성을 파악할 수 없다는 점이야. (./multi-tree_hemiltonian_path_implementation_plan.md 참고) 아마 reliable_TD_threshold  범위 바깥이라면 단순히 가까운 상대를 기준으로 target_node를 지정할텐데, ‘지정된 target_node가 속한 tree와 확장하는 노드 자신이 속한 tree가 모두 waypoints 에 해당할 경우에만 순방향으로 고정’할게. 비록 reliable_TD_threshold  범위 바깥이라 임의의 tree에 속한 node를 target node로 지정한다고 해도, 확장하는 노드가 S 혹은 G에 속해있거나, 아니면 target으로  지정된 노드가 S 혹은 G에 속해있다면 방향성을 추론할 수 있으니까. (반드시 S에서 agent가 출발해서 G로 돌아와야하기에)

---

## 2026-04-23 감사 결과 / 구현 계획 보강

이 절은 아래 기존 계획 중

- `execution_sequence`
- `unordered_equivalence`
- `source_is_left propagation`
- `_check_plan_batch_feasibility`

관련 항목을 **supersede**한다. 이유는 현재 코드가 `route edge` 방향 주입은 일부 반영했지만,
`plan/frame sequence` 계층과 `source_is_left 유실 경로`에는 아직 prompt의 일반 multi-waypoint
가정이 완전히 반영되지 않았기 때문이다.

### 1. 감사 결론

현재 구현은 아래 두 부분으로 나뉜다.

- **이미 방향성이 잘 반영된 축**
  - ordered anchor-pair repository
  - directional root-cost / bridge-cost matrix
  - target selection / dynamic goal selection의 `source_is_left`
  - guidance의 backward mask
  - meeting acceptance에서 left/right node 재정렬
  - fixed temporal route용 ordered temporal matrix

- **아직 prompt를 완전히 반영하지 못한 축**
  - `check feasibility`와 그 caller들
  - `_compute_node_uncertainty`
  - `plan_history` / segment slice를 읽는 frame-order-sensitive helper들
  - active anchor runtime 내부의 `source_is_left` silent fallback
  - sibling dedup / tail clustering의 `unordered_equivalence` 사용

핵심 원인은 다음이다.

- 현재 multi-anchor runtime은 tree storage를 모두 `anchor-root -> outward`의 **local canonical order**로 유지한다.
- 이 자체는 맞다. prompt도 "별도 backward pipeline"이 아니라 "기존 pipeline + 방향 정보 주입"을 요구한다.
- 하지만 몇몇 HILP call site는 여전히
  - 저장된 tensor/frame 순서를 곧바로 TD 방향으로 간주하거나,
  - `source_is_left`가 비어 있으면 `_tree_uses_forward_prefix_semantics(...)`로 대체하거나,
  - 방향이 정의되지 않는 pairwise equivalence를 임시로 symmetric reduction(`max(TD(a→b), TD(b→a))`)으로 처리한다.

즉, 현재 구현은 `route-level direction injection`은 들어갔지만, `frame-level sequence direction`
과 `fallback-less propagation contract`는 아직 완성되지 않았다.

### 2. 수정이 필요한 코드 범위

아래는 현재 코드 감사 기준의 **전체 HILP/TD 관련 범위**와 상태다.

#### 2-1. 이미 prompt와 합치되는 지점

- `df_planning.py::_init_multi_tree_planner_state`
  - ordered `(i, j)` root cost를 모두 계산한다.
- `df_planning.py::_build_pairwise_anchor_selection_state`
  - 2-anchor case도 ordered cost를 유지한다.
- `df_planning.py::_select_multi_tree_target_node`
  - target tree가 정해진 뒤 `source_anchor_idx`, `target_anchor_idx` 기준으로 방향을 계산한다.
- `df_planning.py::_select_dynamic_goal`
  - `source_is_left`를 받아 `V(source, target)` / `V(target, source)`를 구분한다.
- `df_planning.py::_update_multi_tree_direct_bridges_from_expansions`
  - 양 방향 ordered bridge를 따로 업데이트한다.
- `df_planning.py::_repair_missing_pair_connection`
  - ordered pair 기준으로 repair한다.
- `df_planning.py::_select_multi_tree_round_meetings`
  - meeting candidate를 left/right node로 재정렬한 뒤 route 방향으로 TD를 계산한다.
- `guidance.py`
  - backward guidance mask와 `compute_grads_wrt_second_arg`가 들어가 있다.
- `plan_viz.py`
  - heatmap / grad-field가 `source_is_left`를 받는다.
- `env_executor.py`
  - 이미 assembled S→G plan을 기준으로 nearest-frame / sub-goal matching을 한다.
- `utils/route_metric_utils.py::compute_pairwise_temporal_distance_matrix`
  - ordered `src -> dst` temporal matrix를 만든다.

#### 2-2. prompt 기준으로 추가 수정이 필요한 지점

- `df_planning.py::_check_plan_batch_feasibility`
  - 현재는 저장된 frame order를 그대로 `state_t -> state_{t+1}`로 본다.
  - reverse-edge candidate의 sequence direction contract가 없다.
- `df_planning.py::_compute_node_uncertainty`
  - 아직 `source_is_left`를 받지 않고 `TD(curr_obs -> target_obs)`를 하드코딩한다.
- `df_planning.py::_compute_uncertainty_and_clusters`
  - `_check_plan_batch_feasibility(...)` 호출 시 sequence direction을 넘기지 않는다.
  - `_compute_node_uncertainty(...)` 호출 시도 direction-less다.
- `df_planning.py::_run_global_uncertainty_expansion_round`
  - expansion/replan feasibility 호출에 direction flag를 넘기지 않는다.
  - uncertainty sampling / uncertainty compute 경로는 `source_is_left`를 가지고 있으나 일부 fallback이 남아 있다.
- `df_planning.py::_run_multi_tree_online_hamiltonian_planner`
  - mixed-parent expansion path에서도 feasibility / uncertainty call에 direction 주입이 빠져 있다.
- `df_planning.py::_ensure_uncertainty_roots_initialized_multi`
  - 현재는 `target_info["source_is_left"]`를 받지만, 이후 uncertainty 하위 helper까지 mandatory contract로 강제되지 않는다.
- `plan_postproc.py::_extract_new_segment_obs`
  - 여전히 "모든 tree가 uniform forward prefix semantics"라는 가정이 주석/contract에 남아 있다.
  - storage는 그대로 두되, frame-order-sensitive caller가 execution-view를 요청할 수 있어야 한다.
- `plan_postproc.py::_extract_node_obs_slice`
  - `reverse`는 있지만 route/sequence contract 이름이 아니다.
  - left/right route semantics를 더 명시적으로 받는 wrapper가 필요하다.
- `plan_postproc.py::_deduplicate_by_endpoint`
  - 현재 duplicate 판정은 `mode="unordered_equivalence"`를 쓴다.
  - 이는 earlier user decision인 "그쪽도 directional TD 사용"과 충돌한다.
- `df_planning.py::_compute_node_uncertainty`
  - tail clustering에서 `_compute_unordered_equivalence_temporal_dist_np`를 쓴다.
  - 이것도 같은 이유로 재검토 대상이다.

### 3. 이번 보강에서 확정하는 설계 불변식

#### 3-1. Storage invariant

- `TreeNode.plan_history`와 `parallel_plan()`의 출력 tensor는 계속 **anchor-root local canonical order**로 저장한다.
- 즉, multi-waypoint directional 지원을 위해 tree 자체를 backward tree로 바꾸지 않는다.
- prompt의 요구대로 기존 pipeline을 유지하고, 방향 정보는 **읽는 쪽 helper**에서 주입한다.

#### 3-2. Route-direction invariant

- route-edge HILP 호출은 모두 `source_tree`, `target_tree`의 anchor 인덱스와 현재 tentative route를 통해
  direction을 계산한다.
- `reliable_TD_threshold` 때문에 non-neighbor target을 잡은 경우의 fallback rule은 아래 helper 하나에 모은다.

```text
resolve_route_direction(source_anchor, target_anchor):
  1. tentative route 상 adjacent면 route order를 따른다.
  2. source == S 이면 forward
  3. target == G 이면 forward
  4. source == G 이면 backward
  5. target == S 이면 backward
  6. 둘 다 waypoint면 forward 고정
```

즉, prompt에서 말한 예외사항은 `_is_source_left_of_target(...)`의 계약으로 **명시화**하고,
"왜 이 방향이 선택되었는지"도 debug log에 남긴다.

#### 3-3. Sequence-direction invariant

- frame-to-frame / prefix-to-tail / plan-history slice 같은 **sequence-sensitive** TD 호출은
  더 이상 저장된 tensor 순서만으로 방향을 정하지 않는다.
- 각 call site는 아래 둘 중 하나를 명시해야 한다.
  - `route_execution`
    - 현재 candidate가 global route에서 left side인지 right side인지에 따라 TD argument order를 정함
  - `local_generation`
    - diffusion output의 local forward order 그대로 TD argument order를 정함

이 둘은 현재 코드에서 분리돼 있지 않다. 이 부분은 아래 Open Questions에 적은 확인이 필요하다.

#### 3-4. No-silent-fallback invariant

- active anchor planner runtime에서는 target이 정해진 이후 `source_is_left`가 비어 있으면 **bug**다.
- `_tree_uses_forward_prefix_semantics(...)`는 legacy two-tree / non-anchor compatibility용으로만 남기고,
  anchor runtime 내부의 route-edge HILP call site에서는 default fallback으로 쓰지 않는다.

### 4. 기존 계획 대비 수정되는 핵심 구현 항목

#### 단계 A: route direction helper를 selection-reason contract로 고정

**대상**

- `df_planning.py::_is_source_left_of_target`
- `df_planning.py::_select_multi_tree_target_node`
- `df_planning.py::_select_multi_tree_expansion_parents`
- `df_planning.py::_ensure_uncertainty_roots_initialized_multi`

**수정 내용**

- `_is_source_left_of_target(...)`를 단순 route-order helper가 아니라
  prompt의 fallback rule을 문서화한 **single source of truth**로 고정한다.
- `_select_multi_tree_target_node(...)`는 반환 dict에 아래 provenance를 추가한다.
  - `source_is_left`
  - `direction_reason`
    - `tentative_neighbor`
    - `reliable_threshold_far_fallback`
    - `start_anchor_fallback`
    - `goal_anchor_fallback`
    - `waypoint_waypoint_forward_fallback`
- 이후 parent selection / root uncertainty init / expansion candidate construction은
  이 provenance를 그대로 넘기고, 같은 pair에 대해 다시 tree-local default를 쓰지 않는다.

#### 단계 B: `source_is_left` propagation을 mandatory로 변경

**대상**

- `df_planning.py::_run_global_uncertainty_expansion_round`
- `df_planning.py::_run_multi_tree_online_hamiltonian_planner` 내부 mixed expansion path
- `df_planning.py::_run_fast_uncertainty_sampling`
- `df_planning.py::_compute_uncertainty_and_clusters`
- `df_planning.py::_init_root_node_uncertainty`
- `tree_node.py::TreeNode`

**수정 내용**

- active anchor runtime에서 candidate / child / root uncertainty vinfo는 모두 `source_is_left`를 반드시 가진다.
- 아래 형태의 코드는 anchor runtime에서 제거한다.

```python
info.get("source_is_left", self._tree_uses_forward_prefix_semantics(...))
```

- 대신
  - selection 단계에서 값을 채운다
  - child node에도 저장한다
  - uncertainty / feasibility / visualization call까지 그대로 전달한다
- 즉, active anchor runtime에서는 `source_is_left is None`을 허용하지 않는다.

#### 단계 C: sequence-sensitive TD helper를 분리

**대상**

- `df_planning.py::_check_plan_batch_feasibility`
- `plan_postproc.py::_extract_new_segment_obs`
- `plan_postproc.py::_extract_node_obs_slice`
- future frame-order-sensitive helper 전부

**새 helper**

- `_resolve_sequence_direction(source_is_left: bool, sequence_mode: str) -> bool`
  - `True`면 stored frame order를 TD order로 사용
  - `False`면 stored frame order의 TD argument order를 반대로 사용
- 또는 동등한 의미의 helper로
  - `_compute_directed_sequence_distance_batch(from_frames, to_frames, source_is_left, sequence_mode)`
  - `_get_execution_view_of_plan(plan_frames, source_is_left, sequence_mode)`

중요한 점은 `plan_history` storage를 바꾸는 것이 아니라, TD를 계산할 때만 이 helper를 통하게 하는 것이다.

#### 단계 D: feasibility를 direction-aware sequence contract로 변경

**대상**

- `df_planning.py::_check_plan_batch_feasibility`
- 그 모든 caller

**필수 변경점**

- `_check_plan_batch_feasibility(...)`에
  - `source_is_left_flags`
  - `sequence_mode`
를 추가한다.
- caller는 아래에서 모두 넘긴다.
  - expansion round feasibility
  - mixed expansion round feasibility
  - `_compute_uncertainty_and_clusters()` 내부 uncertainty sample feasibility

**현재 누락**

- expansion/replan feasibility 호출
- uncertainty sample feasibility 호출
- root uncertainty init에서 내려가는 uncertainty feasibility 호출

**검토 대상 규칙**

- continuity
- progress
- prior-tail anti-loop

이 세 규칙이 `route_execution`을 따를지 `local_generation`을 따를지는 Open Question 1에 명시한다.

#### 단계 E: uncertainty의 `T_curr` 방향을 route-aware로 변경

**대상**

- `df_planning.py::_compute_node_uncertainty`
- `df_planning.py::_compute_uncertainty_and_clusters`

**수정 내용**

- `_compute_node_uncertainty(...)`에 `source_is_left: bool`를 추가한다.
- directional model이면
  - `source_is_left=True`  → `TD(curr_obs -> target_obs)`
  - `source_is_left=False` → `TD(target_obs -> curr_obs)`
- `_compute_uncertainty_and_clusters(...)`는 이미 `source_is_left_flags`를 갖고 있으므로
  이를 `_compute_node_uncertainty(...)`까지 그대로 전달한다.

이 단계가 없으면 `expected_root_node_dist`와 `temporal_dist` value는 general multi-waypoint case에서
여전히 한쪽 방향으로 bias된다.

#### 단계 F: segment / plan-history helper의 contract를 실행 문맥 기준으로 명시

**대상**

- `plan_postproc.py::_extract_new_segment_obs`
- `plan_postproc.py::_extract_node_obs_slice`
- `plan_postproc.py::_compute_plan_gap_to_target`

**수정 내용**

- `_extract_new_segment_obs(...)`의 현재 주석
  - "모든 tree를 uniform forward prefix semantics로 본다"
  를 제거한다.
- 새 wrapper를 추가한다.
  - `_extract_route_ordered_segment_obs(node, plan_tokens, source_is_left, role)`
  - 여기서 `role`은 `source_segment`, `target_segment`, `meeting_left`, `meeting_right` 등
    frame order가 왜 필요한지를 설명하는 용도다.
- `_compute_plan_gap_to_target(...)`는 이미 left/right를 caller가 정렬해서 넘기면 맞게 동작하므로,
  helper contract도 `best_node/target_node`가 아니라 `left_node/right_node` 문맥으로 문서화한다.

#### 단계 G: `unordered_equivalence` 사용처 재검토

아래 두 곳은 earlier user decision과 충돌 가능성이 있다.

- `plan_postproc.py::_deduplicate_by_endpoint`
- `df_planning.py::_compute_node_uncertainty` 내부 tail clustering

현재 둘 다 사실상

```text
max(TD(a→b), TD(b→a))
```

형태의 symmetric reduction에 기대고 있다.

하지만 user decision은 "그쪽도 directional TD 사용"이었다. 따라서 기존 계획의
`unordered_equivalence` 항목은 아래 중 하나로 바뀌어야 한다.

- 옵션 1. pairwise predicate 자체를 route-direction-aware로 정의한다.
  - 예: sibling dedup이면 같은 parent/target을 공유하므로 `source_is_left` 기준 한 방향 TD만 사용
- 옵션 2. duplicate/equivalence는 여전히 symmetric predicate가 필요하므로
  - `max`, `min`, `both<thres` 중 어떤 rule을 쓸지 별도 확정한다

이 항목은 Open Question 2에 남긴다.

### 5. 함수 단위 수정 대상 요약

아래 함수들은 문서상 **수정 필수**다.

- `algorithms/diffusion_forcing/df_planning.py`
  - `_is_source_left_of_target`
  - `_select_multi_tree_target_node`
  - `_ensure_uncertainty_roots_initialized_multi`
  - `_check_plan_batch_feasibility`
  - `_compute_node_uncertainty`
  - `_run_fast_uncertainty_sampling`
  - `_compute_uncertainty_and_clusters`
  - `_run_global_uncertainty_expansion_round`
  - `_run_multi_tree_online_hamiltonian_planner`

- `algorithms/diffusion_forcing/plan_postproc.py`
  - `_extract_node_obs_slice`
  - `_extract_new_segment_obs`
  - `_compute_plan_gap_to_target`
  - `_deduplicate_by_endpoint`

- `algorithms/diffusion_forcing/tree_node.py`
  - `TreeNode` propagation contract 점검

아래 함수들은 현재 기준 **방향성 설계가 이미 맞는 편**이므로 문서상 유지로 둔다.

- `algorithms/diffusion_forcing/guidance.py`
- `algorithms/diffusion_forcing/env_executor.py`
- `algorithms/diffusion_forcing/plan_viz.py`
- `utils/route_metric_utils.py`

### 6. Open Questions

#### Open Question 1. reverse-edge candidate의 feasibility / frame sequence direction

현재 가장 중요한 판단 포인트다.

- stored plan order는 계속 `anchor-root -> outward` local order로 둔다.
- 그런데 reverse-edge candidate에서 frame-order-sensitive TD를 계산할 때,
  아래 둘 중 무엇을 기준으로 할지 확정이 필요하다.

옵션 A. `route_execution`
- global S→G route에서 실제로 agent가 그 tree segment를 통과하는 방향을 따른다.
- 즉, reverse-edge candidate는 stored frame pair `(x_t, x_{t+1})`에 대해
  TD direction을 반대로 본다.
- 내 권장안이다. prompt의 "agent는 반드시 S에서 G로 향한다"와 가장 직접적으로 맞는다.

옵션 B. `local_generation`
- diffusion이 실제로 생성한 local sequence order를 따른다.
- 즉, feasibility는 stored frame pair의 순서를 그대로 TD direction으로 쓴다.
- route direction은 target selection / value / guidance / assembly에서만 사용한다.

#### Open Question 2. sibling dedup / tail clustering의 directional predicate

earlier user decision상 이쪽도 directional TD를 써야 한다. 다만 duplicate/equivalence는
본질적으로 pairwise predicate라 아래 중 어떤 규칙을 쓸지 선택이 필요하다.

- 옵션 A. 같은 `source_is_left` 방향의 단일 TD만 사용
- 옵션 B. `TD(a→b) < thres` and `TD(b→a) < thres`를 모두 만족해야 duplicate
- 옵션 C. 기존 `max(TD(a→b), TD(b→a)) < thres` 유지

내 권장안은 **sibling dedup은 옵션 A**, **tail clustering은 옵션 B**다.
이유는 sibling dedup은 이미 동일 parent/target route context가 있고,
clustering은 truly pairwise equivalence라 한 방향만으로 collapse시키기 위험하기 때문이다.


## 구현 계획

### 설계 철학 / 호환성 불변식

본 수정은 directional HILP만을 위한 **별도 planner/runtime pipeline을 추가하지 않는다.**
기존 anchor planner, uncertainty sampling, guidance, feasibility check, postprocess,
env execution의 **함수 구조와 round loop는 유지**하고, 각 TD/HILP call site에
`direction context`를 주입하는 방식으로만 수정한다.

핵심 불변식은 아래와 같다.

- **Single-pipeline invariant**
  - directional / non-directional HILP 모두 동일한 `_run_multi_tree_online_hamiltonian_planner`
    루프, 동일한 planner_state, 동일한 `parallel_plan -> postprocess -> bridge update -> meeting accept -> assemble`
    순서를 사용한다.
  - 방향성 때문에 planner, guidance, executor의 별도 함수 세트를 만들지 않는다.

- **Direction-contract invariant**
  - 모든 TD/HILP 호출은 아래 셋 중 하나로 분류되어야 한다.
    1. `route_edge`: tentative Hamiltonian route의 edge A→B 문맥
    2. `execution_sequence`: 실제 plan frame 또는 rollout에서 앞 frame → 뒤 frame 문맥
    3. `unordered_equivalence`: 단일 시간방향이 없는 동일성/중복성 판단 문맥
  - 더 이상 `_compute_distance(a, b)`를 "그냥 거리"로 호출하지 않는다. 각 call site는
    어떤 문맥의 방향을 쓰는지 명시해야 한다.

- **Symmetric-compatibility invariant**
  - 방향성을 고려하지 않는 HILP라면 `V(a, b) = V(b, a)` 이다.
  - 이 경우 direction-aware helper를 거쳐도 각 scalar 값은 기존 코드와 동일해야 한다.
    - ordered root-cost matrix는 수치적으로 기존 symmetric matrix와 동일
    - `cum_from_root == cum_to_root`
    - directed total cost는 기존 `_compute_source_target_total_cost`와 동일
    - `unordered_equivalence`에서 양방향 TD를 모두 계산하더라도 reduction 결과는 기존 단일 TD와 동일
    - guidance는 backward mask가 있더라도 기존 `compute_grads(obs, target)` / `V(obs, target)`
      경로로 collapse되어야 한다

- **Legacy-process invariant for symmetric models**
  - non-directional HILP에서는 planner의 reachability / forced-adjacency semantics가
    수정 전 기존 코드와 동일해야 한다.
  - 따라서 내부 repository는 ordered key를 유지하더라도, **solver/accepted-adjacency layer는
    symmetric model일 때 기존 unordered canonical view를 노출**해야 한다.
  - 이는 separate pipeline이 아니라, 같은 ordered repository 위에 올려진 compatibility view다.

- **Directed-provenance invariant**
  - directional model(`quasimetric`)에서는 accepted edge와 connection repository가
    edge 방향을 provenance로 보존한다.
  - 이 모드에서는 `A→B`로 accepted 된 edge를 이후 `B→A`로 재사용하는 것을 금지한다.

- **Teleport checkpoint assumption**
  - `teleport`가 붙은 모든 HILP checkpoint는 `quasimetric`으로 간주한다.
  - 즉, teleport 계열에서는 directional mode가 기본이며 reverse reuse 금지 규칙이 적용된다.
  - non-teleport 계열은 explicit override가 없으면 기존 symmetric aggregator(`neg_l2`) 경로를 기본으로 사용한다.

---

### 공통 DirectionContext

방향 정보는 ad-hoc boolean 하나로 흩뿌리지 않고, 아래 의미를 공유하는 공통 문맥으로 전달한다.

- `model_is_directional`
  - resolved HILP aggregator가 `quasimetric`인지 여부
- `mode`
  - `route_edge` | `execution_sequence` | `unordered_equivalence`
- `source_anchor_idx`, `target_anchor_idx`
  - anchor-level 문맥이 있을 때의 source/target
- `route_left_anchor_idx`, `route_right_anchor_idx`
  - route상 실제 실행 방향의 left/right anchor
- `source_is_left`
  - 현재 source node/tree가 route edge의 left side인지 여부
- `from_obs`, `to_obs`
  - 실제 TD를 계산해야 하는 방향의 observation endpoint

DirectionContext의 source는 아래와 같다.

- **route_edge**
  - `_is_source_left_of_target(...)`와 tentative route에서 계산
  - `reliable_TD_threshold` 예외는 prompt에 적힌 fallback 규칙을 그대로 사용
- **execution_sequence**
  - materialized plan frame order, parent→child tree growth order, env rollout의 현재 step order에서 계산
- **unordered_equivalence**
  - 단일 실행 방향이 없으므로 `a→b`, `b→a`를 둘 다 계산하고 명시적 reduction rule을 쓴다
  - 기본 closeness predicate는 `max(TD(a→b), TD(b→a))`
  - symmetric model에서는 두 값이 같으므로 기존 단일 거리와 정확히 같은 결과가 된다

---

### 방향 정보 주입 범위 (active call-site inventory)

아래는 active runtime에서 HILP/TD가 쓰이는 지점과, 각 지점의 방향 문맥이다.

- **`route_edge` 문맥**
  - `_init_multi_tree_planner_state`
  - `_build_pairwise_anchor_selection_state`
  - `_select_multi_tree_target_node`
  - `_select_multi_tree_expansion_parents`
  - `_compute_selection_value_for_target`
  - `_update_multi_tree_direct_bridges_from_expansions`
  - `_select_multi_tree_round_meetings`
  - `_repair_missing_pair_connection`
  - `_compute_node_uncertainty`의 `T_curr = TD(curr → target)`
  - guidance에서 edge target을 향한 tail guidance 계산

- **`execution_sequence` 문맥**
  - `_update_expanded_children_state`: `parent → child`, `child → parent`
  - `_check_plan_batch_feasibility`
    - continuity: `plan[t] → plan[t+1]`
    - progress: `current_obs → final_obs`
    - prior-tail anti-loop: `prior_frame → new_tail`
  - `plan_postproc._compute_plan_gap_to_target`
    - route-left segment frame → route-right segment frame
  - `plan_postproc._reorder_plan_by_proximity`
    - current frame → candidate future frame
  - `env_executor`의 nearest-frame / sub-goal matching
    - current agent obs → candidate plan frame

- **`unordered_equivalence` 문맥**
  - `plan_postproc._deduplicate_by_endpoint`
    - endpoint pair 간 duplicate 판단은 `max(TD(i→j), TD(j→i))`
  - `cluster_tail_by_temporal_dist`
    - tail sample pair 간 cluster closeness도 동일한 reduction 사용

- **자동 전파되는 지점**
  - `compute_guidance_grad_np`를 호출하는 visualization / grad-field helper는
    guidance 본체와 동일한 DirectionContext를 그대로 따른다
  - `compute_pairwise_temporal_distance_matrix`는 ordered matrix를 직접 반환하므로
    fixed temporal route solver도 directional matrix를 그대로 사용한다

---

### 전제: 방향 정의와 비용 공식

Hamiltonian route `[S=0, W1, ..., Wk, G=n-1]` 에서 edge A→B는 "agent가 A-anchor에서 B-anchor로 이동"을 의미한다.

모든 tree는 anchor-rooted forward-prefix semantics (anchor에서 바깥 방향으로 성장)를 사용한다. 따라서:

- **forward tree (left side of edge A→B = A-tree)**:
  - agent가 A_root → ... → bp_A 순방향으로 이동
  - 관련 누적 비용: `A_node.cum_from_root` = sum of V(parent→child) from A_root to A_node

- **backward tree (right side of edge A→B = B-tree)**:
  - tree는 B_root → ... → bp_B 방향으로 성장하지만, final assembly에서 뒤집혀 agent는 bp_B → ... → B_root 순서로 실행됨
  - 관련 누적 비용: `B_node.cum_to_root` = sum of V(child→parent) from B_node back to B_root

따라서 edge A→B에서 bridge (bp_A, bp_B) 쌍의 실제 통행 비용:

```
bridge_cost(A→B) = A_node.cum_from_root
                 + V(A_node.obs, B_node.obs)   ← T_curr, 항상 left→right
                 + B_node.cum_to_root
```

T_curr은 항상 "route상 왼쪽 tree node → 오른쪽 tree node" 방향이다.

---

### 단계별 구현 목록

#### 단계 1: `_anchor_pair_key` → ordered

**파일**: `df_planning.py`, line 906

```python
# Before
def _anchor_pair_key(self, a: int, b: int) -> Tuple[int, int]:
    return (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))

# After
def _anchor_pair_key(self, a: int, b: int) -> Tuple[int, int]:
    return (int(a), int(b))  # ordered: (source, target)
```

이에 따라 **모든 call site**에서 symmetric fill 제거:

- `_update_multi_tree_direct_bridges_from_expansions` (line 5712, 5714): `[pair_key[::-1]] = bridge_cost` 두 줄 제거. 대신 reverse 방향은 별도 계산 후 `[(target_anchor, source_anchor)]`에 저장 (단계 7에서 상술).
- `_repair_missing_pair_connection` (line 6008, 6010): 동일하게 symmetric fill 제거.
- `_assemble_multi_tree_plan_bundle` (line 6080 등): ordered key를 사용하므로 `pair_key = (src_idx, dst_idx)`가 route 방향 그대로. 기존 flip 감지 로직 (line 6138-6146) 단순화 가능: ordered key로 꺼낸 conn은 `source_anchor_idx == src_idx`가 항상 보장되므로 flip 경로 불필요.
- `_select_multi_tree_round_meetings`의 `forced_pairs` 집합 (line 5732-5736): ordered key 사용 시 집합 원소가 방향성을 가짐. meeting 검출은 양방향 tree 모두에서 발생 가능하므로 아래 단계 10에서 별도 처리.

ordered key는 **repository 내부 표준형**이고, solver/accepted-adjacency가 어떤 view를 읽는지는
아래 단계 1-b에서 별도로 정의한다.

---

#### 단계 1-b: ordered repository 위의 compatibility view

**파일**: `df_planning.py`, `utils/route_metric_utils.py`

모든 pair repository는 ordered key를 저장하되, read path는 helper를 통해서만 접근한다.

- `best_direct_bridge_raw`, `best_direct_bridge_info`, `accepted_pair_edges`는
  내부적으로 `(source_anchor, target_anchor)` ordered key를 사용
- 새 helper:
  - `_hilp_is_directional()` / `_hilp_is_symmetric()`
  - `_get_pair_repository_key(...)`
  - `_get_solver_forced_pairs_view(planner_state)`
  - `_get_connection_info_for_pair(planner_state, pair_key, allow_symmetric_compat_view=...)`

모드별 규칙은 아래와 같다.

- **directional model (`quasimetric`)**
  - ordered edge를 그대로 solver/assembly에 전달
  - `A→B` accepted edge의 reverse reuse 금지
  - `forced adjacency`도 ordered pair로 유지

- **symmetric model**
  - 내부 ordered repo는 유지
  - 단, solver/accepted-adjacency layer는 **기존 코드와 동일한 unordered canonical view**를 사용
  - 즉, legacy path와 같은 reachable adjacency semantics를 유지한다
  - 이는 새 planner를 만드는 것이 아니라, same repository 위에 symmetric compatibility view를 얹는 것이다

이 단계가 있어야만 "direction-aware injection을 사용해도 non-directional model에서는 기존 process와 동일"이라는
요구를 만족할 수 있다.

---

#### 단계 2: 비대칭 비용 행렬 초기화

**파일**: `df_planning.py`
**대상**: `_init_multi_tree_planner_state` (line 5490-5506), `_build_pairwise_anchor_selection_state` (line 5528-5542)

현재는 `root_cost[i,j] = root_cost[j,i] = same_value`. 변경 후:

```python
for i in range(n_anchors):
    for j in range(n_anchors):
        if i == j:
            continue
        td = V(trees[i].root_node.obs → trees[j].root_node.obs)  # V(i→j)
        # root nodes have cum_from_root = cum_to_root = 0
        root_cost[i, j] = self._transform_temporal_dist_for_total_cost(td)
```

loop를 `j in range(i+1, n_anchors)` 대신 `j in range(n_anchors)`로 변경하고, symmetric fill 제거. HILP call 횟수는 `n*(n-1)`로 증가하나 n이 작으므로 (보통 2~5) 문제없음.

---

#### 단계 3: `TreeNode`에 `cum_temporal_dist_to_root` 추가

**파일**: `tree_node.py`, `__init__` (line 18)

```python
def __init__(self, ...,
             cum_temporal_dist_from_root: float = 0.0,
             cum_temporal_dist_to_root: float = 0.0):
    ...
    self.cum_temporal_dist_from_root: float = float(cum_temporal_dist_from_root)
    self.cum_temporal_dist_to_root: float = float(cum_temporal_dist_to_root)
```

- `cum_temporal_dist_from_root`: root → node 방향 누적 (기존, sum of V(parent→child))
- `cum_temporal_dist_to_root`: node → root 방향 누적 (신규, sum of V(child→parent))
- root node: 두 값 모두 `0.0`

---

#### 단계 4: `_update_expanded_children_state`에서 양방향 누적

**파일**: `df_planning.py`, line 8689-8729

현재 `V(parent.obs, child.obs)` 하나만 계산. 변경 후 양방향 모두 계산:

```python
_td_fwd = float(V(parent_node.obs → child.obs))  # parent→child
_td_rev = float(V(child.obs → parent_node.obs))  # child→parent

_child.cum_temporal_dist_from_root = (
    float(getattr(parent_node, "cum_temporal_dist_from_root", 0.0)) + _td_fwd
)
_child.cum_temporal_dist_to_root = (
    float(getattr(parent_node, "cum_temporal_dist_to_root", 0.0)) + _td_rev
)
```

두 곳 (use_rollout=True 경로 line 8688, use_rollout=False 경로 line 8720) 모두 동일하게 적용.

---

#### 단계 5: 방향 판단 헬퍼 `_is_source_left_of_target` 추가

**파일**: `df_planning.py`

tentative route를 기준으로 source_anchor가 target_anchor보다 route상 왼쪽(선행)인지 반환. `True`이면 agent가 source→target 방향으로 이동하는 edge이므로 "forward" direction.

```python
def _is_source_left_of_target(
    self,
    planner_state: dict,
    source_anchor_idx: int,
    target_anchor_idx: int,
) -> bool:
    order = list(planner_state.get("tentative_solver_result", {}).get("anchor_order", []))
    s, t = int(source_anchor_idx), int(target_anchor_idx)
    if s in order and t in order:
        s_pos = order.index(s)
        t_pos = order.index(t)
        if abs(s_pos - t_pos) == 1:
            return s_pos < t_pos  # 인접한 경우: route 순서 그대로
    # 비인접 fallback (reliable_TD_threshold 범위 초과 등):
    n = len(planner_state["trees"])
    if s == 0:          return True   # S는 항상 출발점
    if t == n - 1:      return True   # G는 항상 도착점
    if s == n - 1:      return False  # G가 source가 되는 경우
    if t == 0:          return False  # S가 target이 되는 경우
    return True  # 둘 다 waypoint → 순방향 고정
```

---

#### 단계 6: `_compute_directed_source_target_total_cost` 추가

**파일**: `df_planning.py`

기존 `_compute_source_target_total_cost`를 대체하는 방향 인식 버전:

```python
def _compute_directed_source_target_total_cost(
    self,
    source_node: "TreeNode",
    target_node: "TreeNode",
    t_curr: float,
    source_is_left: bool,
) -> float:
    if source_is_left:
        # source가 forward(left) tree: cum_from_root + T_curr + target.cum_to_root
        src_cum = float(getattr(source_node, "cum_temporal_dist_from_root", 0.0))
        tgt_cum = float(getattr(target_node, "cum_temporal_dist_to_root", 0.0))
    else:
        # source가 backward(right) tree: cum_to_root + T_curr + target.cum_from_root
        src_cum = float(getattr(source_node, "cum_temporal_dist_to_root", 0.0))
        tgt_cum = float(getattr(target_node, "cum_temporal_dist_from_root", 0.0))
    return src_cum + self._transform_temporal_dist_for_total_cost(t_curr) + tgt_cum
```

기존 `_compute_source_target_total_cost`는 root-root init (cum=0이라 어느 쪽이든 동일)에서만 남기거나, 모든 호출처를 일괄 교체. 교체가 필요한 호출처:
- `_update_multi_tree_direct_bridges_from_expansions` (단계 7에서 처리)
- `_select_multi_tree_round_meetings` (단계 10에서 처리)
- `_repair_missing_pair_connection` (단계 11에서 처리)
- `_compute_selection_value_for_target` → 이 함수에도 `source_is_left: bool` 파라미터 추가 필요 (단계 9에서 처리)

---

#### 단계 7: `_update_multi_tree_direct_bridges_from_expansions` 양방향 업데이트

**파일**: `df_planning.py`, line 5672-5723

각 (source_node, target_tree) 쌍에 대해 두 방향을 모두 계산 및 저장:

```python
target_obs = np.stack([n.obs for n in target_nodes])  # (N, D)
src_rep = np.repeat(source_node.obs[None], N, axis=0)  # (N, D)

# Forward direction: source_anchor → target_anchor
tds_fwd = V(src_rep, target_obs)          # (N,)
best_fwd_idx = argmin(tds_fwd)
bridge_cost_fwd = _compute_directed_source_target_total_cost(
    source_node, target_nodes[best_fwd_idx], tds_fwd[best_fwd_idx], source_is_left=True
)
fwd_key = (source_anchor_idx, target_anchor_idx)
if bridge_cost_fwd < best_direct_bridge_raw[fwd_key]:
    # best_direct_bridge_raw, effective_pair_cost, best_direct_bridge_info 업데이트

# Reverse direction: target_anchor → source_anchor
tds_rev = V(target_obs, src_rep)          # (N,) — args 순서 교환
best_rev_idx = argmin(tds_rev)
bridge_cost_rev = _compute_directed_source_target_total_cost(
    target_nodes[best_rev_idx], source_node, tds_rev[best_rev_idx], source_is_left=True
)
rev_key = (target_anchor_idx, source_anchor_idx)
if bridge_cost_rev < best_direct_bridge_raw[rev_key]:
    # best_direct_bridge_raw, effective_pair_cost, best_direct_bridge_info 업데이트
```

`best_direct_bridge_info[fwd_key]`와 `best_direct_bridge_info[rev_key]`는 각각 독립적으로 최선 쌍을 저장. 동일 pair에서 forward best node ≠ reverse best node일 수 있음.

HILP call 횟수: 기존 per-pair 1 call → 2 calls (fwd + rev). 배치 크기는 같으므로 처리 시간 약 2배.

---

#### 단계 8: `_select_multi_tree_target_node` 방향 인식

**파일**: `df_planning.py`, line 1522-1643

현재 `temporal_dists = V(query_obs, target_obs)` 고정. 변경 후:

```python
is_fwd = self._is_source_left_of_target(planner_state, source_anchor_idx, target_anchor_idx)
if is_fwd:
    temporal_dists = V(query_rep, tgt_rep)          # V(source→target)
else:
    temporal_dists = V(tgt_rep, query_rep)          # V(target→source), args 교환
```

`use_segment_metric=True` 케이스 (line 1594-1595)도 동일 방향 로직 적용:
- `is_fwd=True`: `node_scores = min(temporal_dists, axis=0)` — 기존과 동일
- `is_fwd=False`: `temporal_dists`의 args가 이미 교환되어 있으므로 min 방향은 유지

---

#### 단계 9: `_select_multi_tree_expansion_parents` + `_compute_selection_value_for_target` 방향 인식

**파일**: `df_planning.py`, line 5601-5670, 4987-5003

`_compute_selection_value_for_target`에 `source_is_left: bool` 파라미터 추가:

```python
def _compute_selection_value_for_target(
    self, source_node, target_node, t_curr, source_is_left: bool,
) -> float:
    if self.uncertainty_mode == "expected_root_node_dist":
        return -self._compute_directed_source_target_total_cost(
            source_node, target_node, t_curr, source_is_left
        )
    if self.uncertainty_mode == "temporal_dist":
        return -float(t_curr)
    return float("nan")
```

`_select_multi_tree_expansion_parents` 내부 (line 5636-5653):

```python
is_fwd = self._is_source_left_of_target(planner_state, tree.anchor_idx, target_tree.anchor_idx)
if is_fwd:
    target_td = V(node.obs → target_node.obs)
else:
    target_td = V(target_node.obs → node.obs)
value = self._compute_selection_value_for_target(node, target_node, target_td, source_is_left=is_fwd)
```

---

#### 단계 10: `_select_multi_tree_round_meetings` 방향 인식 및 pair_key 정규화

**파일**: `df_planning.py`, line 5725-5809

**pair_key 생성 방식 변경**: 현재 `pair_key = _anchor_pair_key(source_anchor_idx, target_anchor_idx)`로 "어느 tree가 확장했는가" 기준으로 key를 만들고 있다. ordered key 도입 후에는 W1-tree child가 S에 접근해도 pair_key가 route edge 방향 `(0,1)`이 되도록 **route 방향으로 정규화**한다.

근거: W1-tree가 S 방향으로 target을 설정한 것 자체가 `_select_multi_tree_target_node`가 tentative route `S→W1`을 기반으로 결정한 것이므로, 어느 쪽에서 meeting이 감지되든 해당 edge는 항상 `(S=0, W1=1)` = `(0,1)`이다.

```python
# pair_key를 route edge 방향으로 정규화
is_fwd = self._is_source_left_of_target(planner_state, source_anchor_idx, target_anchor_idx)
if is_fwd:
    route_left, route_right = source_anchor_idx, target_anchor_idx
else:
    route_left, route_right = target_anchor_idx, source_anchor_idx
pair_key = (route_left, route_right)  # 항상 route 방향으로 정규화

# forced_pairs도 동일하게 route 방향 ordered key
forced_pairs = {
    (int(order[i]), int(order[i + 1]))
    for i in range(len(order) - 1)
}
if pair_key not in forced_pairs:
    continue  # 자연스럽게 동작

# 비용 계산: route_left node (forward side), route_right node (backward side)
left_node  = node if is_fwd else meeting_target_node
right_node = meeting_target_node if is_fwd else node
t_curr = V(left_node.obs, right_node.obs)   # 항상 left→right
bridge_cost = self._compute_directed_source_target_total_cost(
    left_node, right_node, t_curr, source_is_left=True
)
```

`accepted_pair_edges` 저장도 동일하게 `pair_key = (route_left, route_right)` 사용.

단, **solver가 이 accepted edge를 어떻게 읽는지**는 단계 1-b / 단계 14의 compatibility rule을 따른다.

- directional model: ordered pair 그대로 forced adjacency에 들어간다
- symmetric model: solver에서는 기존 unordered adjacency view로 canonicalize된다

즉, accepted provenance 자체는 ordered로 남기되, symmetric model에서만 legacy process parity를 위해
unordered solver view를 제공한다.

---

#### 단계 11: `_repair_missing_pair_connection` 방향 인식

**파일**: `df_planning.py`, line 5930-6011

이 함수는 `source_anchor_idx → target_anchor_idx` 방향이 명시적으로 전달됨 (assembly에서 route 순서대로 호출). 따라서 항상 forward (source가 left):

```python
tds = V(source.obs, target_obs)   # V(source→target)
bridge_cost = _compute_directed_source_target_total_cost(
    source_node, target_node, tds[best], source_is_left=True
)
```

`best_direct_bridge_raw[pair_key]`, `effective_pair_cost[pair_key]` 업데이트에서 symmetric fill 제거 (단계 1 반영).

---

#### 단계 12: `_compute_node_uncertainty` — temporal dist 계산 encapsulation

**파일**: `df_planning.py`, line 4892-4985

방향: `_compute_node_uncertainty`는 현재 expanding 중인 node (curr_obs)에서 target_node (goal_obs)까지의 temporal dist를 구함. 방향은 항상 **curr → goal** (agent가 curr에서 goal로 이동하려는 문맥).

기존 embedding norm 방식 교체:

```python
# Before
_emb_dist_curr = float(np.linalg.norm(z_goal - z_curr))
_t_curr = float(np.asarray(_temporal_dist_fn(np.asarray(_emb_dist_curr))).item())

# After: value function 직접 사용 (embedding norm 대신 actual V 값)
_v_curr = self._compute_hilp_values(
    curr_obs[None].astype(np.float32),
    goal_obs[None].astype(np.float32),
)  # (1,) tensor, negative
_t_curr = float(self.emb_dist_to_temporal_dist(
    (-_v_curr).cpu().numpy(), gamma=gamma
)[0])
```

`_temporal_dist_fn` lambda도 동일하게 교체:

```python
# Before
_temporal_dist_fn = lambda emb_d: self.emb_dist_to_temporal_dist(emb_d, gamma)

# After: no longer a generic fn over embedding dist;
# T_curr는 이미 위에서 직접 계산. _temporal_dist_fn은
# cluster_tail_by_temporal_dist 등 embedding-space 연산에만 남김.
# 단, 이 함수들은 z_tail (embedding) 기반으로 동작하므로 내부는 변경 최소화.
```

`is_degenerate` 분기, `variance`, `entropy` 모드에서도 `T_curr` 계산 경로를 통일하여 value function 기반으로 변경. `z_curr`, `z_goal`는 여전히 cluster/uncertainty 계산용으로 `get_phi`에서 얻음. T_curr 계산만 value function으로 전환.

---

#### 단계 13: Guidance — backward tree 방향 지원

**파일**: `hilp_loader.py`, `guidance.py`, `df_planning.py`

**13-a. `hilp_loader.py`에 `compute_grads_wrt_second_arg` 추가**

기존 `compute_grads(obs, goal)` = ∂V(obs, goal)/∂obs[:2].

신규 `compute_grads_wrt_second_arg(obs, goal)` = ∂V(obs, goal)/∂goal[:2].

JAX 기반 구현: `jax.grad(lambda g: V(obs, g))(goal)`. `HILPMemoizedWrapper`에도 위임 메서드 추가.

**13-b. `compute_guidance_grad_np` / `goal_guidance`에 `is_backward` 파라미터 추가**

```python
def compute_guidance_grad_np(planner, obs_np, target_np, hilp_fn, is_backward: bool = False):
    if not planner._hilp_is_directional():
        # symmetric model compatibility: 기존 코드와 동일한 경로 유지
        hilp_grad_np = hilp_fn.compute_grads(obs_np, goal_rep_np)
        hilp_values_np = planner._compute_hilp_values(obs_np, goal_rep_np).cpu().numpy()
    elif is_backward:
        hilp_grad_np = hilp_fn.compute_grads_wrt_second_arg(obs_np, goal_rep_np)
        # directional backward edge일 때만 second-arg gradient 사용
    else:
        hilp_grad_np = hilp_fn.compute_grads(obs_np, goal_rep_np)  # 기존
```

즉, guidance도 **새로운 backward 전용 pipeline을 만들지 않고**, 같은 `compute_guidance_grad_np`
안에서 direction metadata를 해석한다. symmetric model이면 `is_backward=True`가 들어와도
기존 path로 collapse되므로 수정 전 코드와 동일한 작동 방식을 유지한다.

**13-c. per-batch `is_backward_mask` threading**

`_run_global_uncertainty_expansion_round` 내에서 각 expansion candidate의 방향 정보를 결정하고, batch 구성 시 `is_backward_mask: Tensor(B,) bool`을 생성.

`p_mctd_plan` → `guidance_fn` lambda에 mask 추가:

```python
guidance_fn = lambda x: guidance.combined_guidance(
    self, x, goal, horizon, guidance_scale,
    is_backward_mask=backward_mask,
    ...
)
```

`combined_guidance` → `goal_guidance`로 mask 전달.

---

#### 단계 14: directed forced adjacency + fixed temporal solver contract

**파일**: `utils/route_metric_utils.py`, `df_planning.py`

`compute_pairwise_temporal_distance_matrix` 자체는 이미 `M[i, j] = TD(anchor_i → anchor_j)` ordered matrix를 만든다.
이 부분은 directional / symmetric 모두 일반성을 잃지 않는다.

하지만 `solve_fixed_endpoint_hamiltonian_path_with_forced_adjacency(...)`는 현재 forced pair를
`(min(a, b), max(a, b))`로 canonicalize하고 있어 directional provenance를 버린다.
이 부분은 수정이 필요하다.

수정 후 contract는 아래와 같다.

- **directional model**
  - forced adjacency는 ordered pair `(a, b)`를 그대로 유지
  - solver는 route 안에 subsequence `a -> b`가 실제로 존재하는지 검사
  - accepted `A→B` edge를 `B→A`로 만족시킨 것으로 간주하지 않는다

- **symmetric model**
  - solver forced adjacency view는 기존 코드와 동일하게 unordered canonical pair를 사용
  - 즉, legacy unordered adjacency semantics를 그대로 유지

이 단계가 빠지면 planner 상위 레이어에서 ordered provenance를 계산해도 solver가 다시 방향을 지워버리므로,
방향성 버그가 route 단계로 재유입된다.

---

### 구현 순서 (권장)

1. **단계 0** — resolved aggregator + memo cache contract 확정
2. **단계 1 / 1-b** — ordered repository + symmetric compatibility view helper
3. **단계 3** — `TreeNode.cum_temporal_dist_to_root` 추가
4. **단계 2** — 비대칭 root-cost matrix 초기화
5. **단계 4** — `_update_expanded_children_state` 양방향 누적
6. **단계 5 / 6** — 공통 DirectionContext + directed total-cost helper
7. **단계 7 ~ 11** — route_edge 문맥 call site들에 방향 정보 주입
8. **`_compute_distance` call-site audit** — execution_sequence / unordered_equivalence 문맥 분리
9. **단계 12 + cluster/dedup 보완** — uncertainty / equivalence 계열 direction rule 반영
10. **단계 13** — Guidance 방향 지원 (단, symmetric compatibility collapse 포함)
11. **단계 14** — directed forced adjacency solver contract 마무리

---

### 미결 사항 / 추후 확인 필요

- **단계 13-c의 `is_backward_mask` threading**: `_run_global_uncertainty_expansion_round`에서 candidate batch를 구성하는 시점에 각 candidate의 tree anchor와 target anchor를 알고 있으므로 mask 생성 자체는 가능. 단, 이 정보가 guidance lambda closure로 전달되는 경로(expansion round → p_mctd_plan → guidance_fn)가 현재 설계상 없으므로 인터페이스 변경 필요.
- **`_ensure_uncertainty_roots_initialized_multi`**: root node의 `_compute_node_uncertainty` 호출 시 direction 전달 (단계 12와 연동).
- **`_select_dynamic_goal`**: deprecated path (use_uncertainty_as_value=False일 때만 사용). 현재 `V(current, candidates)` 방향이 forward에는 맞으나 backward tree에는 맞지 않음. deprecated이므로 이번 구현 범위에서 제외 가능.
- **`_update_multi_tree_meeting_targets_from_expansions`**: `_select_multi_tree_target_node`를 `use_segment_metric=True`로 호출. 단계 8의 방향 인식이 여기에도 자동 적용됨.
- **`_compute_distance` call-site audit 강제**: helper 자체를 남기더라도 모든 call site가 `route_edge` / `execution_sequence` / `unordered_equivalence`
  중 어느 문맥인지 명시해야 한다. naked `_compute_distance(a, b)` 호출은 더 이상 허용하지 않는다.

---

### 단계 0: HILPJax aggregator 하드코딩 수정 + grid caching 파이프라인 수정

이 단계는 **단계 1~14보다 먼저** 수행해야 한다. 현재 코드는 directional HILP 체크포인트를 로드해도 symmetric V를 사용한다.

#### 문제 진단

**`td_models/hilp/hilp.py`**, line 257:
```python
aggregator = "neg_l2"  # ← 하드코딩. 실제 학습된 모델은 "quasimetric"
```

사용자의 모델은 `share_encoder=True` + `aggregator='quasimetric'`으로 학습됨:
- `V(s,g) = -||ReLU(phi(s) - phi(g))||`  — 비대칭 ✓

현재 코드(`neg_l2` + `share_encoder=True` 경우):
- `psi_s = phi(s)`, `phi_g = phi(g)` (동일 encoder)
- `V(s,g) = -||phi(s) - phi(g)||` = `V(g,s)` — **대칭, 즉 directional 정보 손실**

따라서 directional HILP 로드 자체가 아직 올바르지 않다.

grid caching에서도 동일 문제가 연쇄됨:
- `_build_memo_grids`: `aggregator = hilp_model._aggregator` = `"neg_l2"` (잘못됨)
- `HILPMemoizedWrapper._aggregate`: `"neg_l2"` / `"inner_prod"` 케이스만 존재, `"quasimetric"` 없음
- `_save_memo_cache`: `aggregator = np.array("neg_l2")` 하드코딩
- `_load_memo_from_cache`: `aggregator = "neg_l2"` 하드코딩

`share_encoder=True`이므로 `psi_grids[0] == phi_g_grids[0]` (동일 내용). aggregator가 `"quasimetric"`이어야 비대칭 V가 나온다.

#### 수정 목록

**1. `td_models/hilp/hilp.py` — `HILPJax.__init__`**

`aggregator`를 pkl에서 auto-detect하는 방법은 없으므로 resolved aggregator를 caller가 넘긴다.
단, 이번 설계에서는 아래 규칙을 기본으로 둔다.

- explicit override가 있으면 override 우선
- checkpoint basename 또는 dataset token에 `teleport`가 들어가면 `quasimetric`
- 그렇지 않으면 기존 symmetric default인 `neg_l2`

즉, `teleport` 계열은 모두 `quasimetric`이라는 이번 작업의 가정을 loader 정책에도 반영한다.

```python
def resolve_hilp_aggregator(checkpoint_path: str, cfg_value: Optional[str]) -> str:
    if cfg_value is not None:
        return str(cfg_value)
    ckpt_name = os.path.basename(checkpoint_path).lower()
    if "teleport" in ckpt_name:
        return "quasimetric"
    return "neg_l2"
```

그 위에서 `HILPJax`는 resolved aggregator를 명시적으로 받는다:

```python
class HILPJax:
    def __init__(self, pkl_path: str, device, hilp_gcrl_path: str = None,
                 aggregator: str = "neg_l2"):  # 명시적 지정 가능
        ...
        # 단계 3 주석 교체
        aggregator = aggregator  # caller가 df_planning.yaml에서 지정
        ...
        self._aggregator = aggregator
```

**2. `td_models/hilp/hilp.py` — `HILPJax.value()`, `compute_values_np()`, `_compute_grads_jax()`**

각각에 `"quasimetric"` 분기 추가:

```python
# value() 및 compute_values_np()
if self._aggregator == "neg_l2":
    dist_sq = ((psi_s - phi_g) ** 2).sum(axis=-1)
    v_np = -np.sqrt(np.maximum(dist_sq, 1e-6))
elif self._aggregator == "quasimetric":
    relu_diff = np.maximum(psi_s - phi_g, 0)
    dist_sq = (relu_diff ** 2).sum(axis=-1)
    v_np = -np.sqrt(np.maximum(dist_sq, 1e-6))
else:  # inner_prod
    v_np = (psi_s * phi_g).sum(axis=-1)
```

`_compute_grads_jax()` 내 `value_fn`:
```python
if aggregator == "neg_l2":
    dist_sq = jnp.sum((psi - phi_g_single) ** 2)
    return -jnp.sqrt(jnp.maximum(dist_sq, 1e-6))
elif aggregator == "quasimetric":
    relu_diff = jax.nn.relu(psi - phi_g_single)
    dist_sq = jnp.sum(relu_diff ** 2)
    return -jnp.sqrt(jnp.maximum(dist_sq, 1e-6))
else:  # inner_prod
    return jnp.sum(psi * phi_g_single)
```

**3. `algorithms/diffusion_forcing/hilp_loader.py` — `HILPMemoizedWrapper._aggregate()`**

```python
def _aggregate(self, psi: np.ndarray, phi_g: np.ndarray) -> np.ndarray:
    if self._aggregator == "neg_l2":
        dist_sq = ((psi - phi_g) ** 2).sum(axis=-1)
        return -np.sqrt(np.maximum(dist_sq, 1e-6)).astype(np.float32)
    elif self._aggregator == "quasimetric":
        relu_diff = np.maximum(psi - phi_g, 0)
        dist_sq = (relu_diff ** 2).sum(axis=-1)
        return -np.sqrt(np.maximum(dist_sq, 1e-6)).astype(np.float32)
    else:  # inner_prod
        return (psi * phi_g).sum(axis=-1).astype(np.float32)
```

**4. `hilp_loader.py` — `_save_memo_cache()`**

```python
# Before
"aggregator": np.array("neg_l2"),

# After
"aggregator": np.array(wrapper._aggregator),
```

**5. `hilp_loader.py` — `_load_memo_from_cache()`**

```python
# Before
aggregator="neg_l2",

# After
aggregator=str(data["aggregator"]) if "aggregator" in data else "neg_l2",
```

**6. `hilp_loader.py` — `load_raw_hilp_model()` / `get_hilp_fn()`에 resolved aggregator threading**

```python
def load_raw_hilp_model(..., hilp_aggregator: str = "neg_l2"):
    ...
    if checkpoint_path.endswith(".pkl"):
        from td_models.hilp import HILPJax
        model = HILPJax(checkpoint_path, device, aggregator=hilp_aggregator)
    ...

def get_hilp_fn(..., hilp_aggregator: str = "neg_l2"):
    if not use_memoization:
        return load_raw_hilp_model(..., hilp_aggregator=hilp_aggregator)
    ...
    hilp_model = load_raw_hilp_model(..., hilp_aggregator=hilp_aggregator)
    wrapper = _build_memo_grids(hilp_model, ...)
    ...
```

**7. `configurations/algorithm/df_planning.yaml`에 optional `hilp_aggregator` override**

```yaml
hilp_aggregator: null  # null이면 checkpoint/dataset 이름으로 resolve
```

`_get_hilp_value_fn()` (또는 `get_hilp_fn` 호출 지점)에서 resolved aggregator를 계산해 전달한다.
teleport 계열은 explicit override가 없어도 자동으로 `quasimetric`을 선택한다.

#### 기존 cache 파일 무효화

aggregator 변경 시 기존 `*_memo_G*.npz` 파일을 삭제해야 한다. `get_hilp_fn`에서 cache path를 구성할 때 aggregator 이름을 포함시키면 자동 무효화 가능:

```python
cache_path = os.path.join(ckpt_dir, f"{ckpt_stem}_memo_G{grid_size}_{hilp_aggregator}.npz")
```

---

### 단계 12 보완: `cluster_tail_by_temporal_dist` encapsulation

**파일**: `algorithms/diffusion_forcing/uncertainty_estimator.py`

`cluster_tail_by_temporal_dist`는 `unordered_equivalence` 문맥이다.
즉, 단일 방향을 고를 수 없으므로 tail sample pair마다 directional TD를 양방향 모두 계산하고,
명시적 reduction rule로 closeness를 정의해야 한다.

**방향 결정**: tail sample들은 같은 노드에서 생성된 여러 가능한 미래 경로 끝단이다.
여기서는 "동일 cluster로 묶어도 되는가"를 보수적으로 판단해야 하므로 closeness를 아래처럼 둔다.

```python
td_ij = raw_td(obs_i -> obs_j)
td_ji = raw_td(obs_j -> obs_i)
cluster_closeness = max(td_ij, td_ji)
```

`max(td_ij, td_ji)`를 쓰는 이유는 directional model에서 한쪽 방향만 가까운 pseudo-match를 막기 위해서다.
symmetric model에서는 `td_ij == td_ji`이므로 기존 단일 TD와 정확히 같다.

단, `z_tail`은 현재 embedding (latent) 공간의 벡터이므로 value function에 직접 넣을 수 없다.
따라서 `cluster_tail_by_temporal_dist`는 `obs` 공간 좌표를 추가로 받아야 한다.

**함수 시그니처 변경**:

```python
# Before
def cluster_tail_by_temporal_dist(z_tail: np.ndarray, n_clusters: int, ...) -> np.ndarray:

# After
def cluster_tail_by_temporal_dist(
    z_tail: np.ndarray,          # (N, D) — embedding (kept for compatibility)
    obs_tail: np.ndarray,        # (N, obs_dim) — observation coordinates
    n_clusters: int,
    hilp_value_fn: Callable,     # lambda obs_a, obs_b → float scalar
    ...
) -> np.ndarray:
```

`hilp_value_fn`은 `df_planning.py` 호출 지점에서 다음과 같이 directional TD helper 형태로 전달:

```python
hilp_raw_td = lambda a, b: float(
    self._compute_raw_state_temporal_dist_np(a[None], b[None])[0]
)
hilp_equiv_td = lambda a, b: max(hilp_raw_td(a, b), hilp_raw_td(b, a))
```

---

### 단계 13-a 보완: `compute_grads_wrt_second_arg` 구현 세부사항

#### `HILPJax.compute_grads_wrt_second_arg`

```python
def compute_grads_wrt_second_arg(self, obs_np: np.ndarray, goal_np: np.ndarray) -> np.ndarray:
    """∂V(obs, goal)/∂goal[:2] via JAX grad."""
    try:
        return self._compute_grads_wrt_goal_jax(obs_np, goal_np)
    except Exception:
        return self._compute_grads_wrt_goal_fd(obs_np, goal_np)

def _compute_grads_wrt_goal_jax(self, obs_np: np.ndarray, goal_np: np.ndarray) -> np.ndarray:
    import jax, jax.numpy as jnp

    if not hasattr(self, '_grad_fn_goal'):
        aggregator = self._aggregator
        agent = self._agent

        def value_fn_wrt_goal(goal_single, psi_s_single):  # (obs_dim,), (skill_dim,) → scalar
            phi_g = agent.get_phi_goal(goal_single[None])[0]
            if aggregator == "neg_l2":
                dist_sq = jnp.sum((psi_s_single - phi_g) ** 2)
                return -jnp.sqrt(jnp.maximum(dist_sq, 1e-6))
            elif aggregator == "quasimetric":
                relu_diff = jax.nn.relu(psi_s_single - phi_g)
                dist_sq = jnp.sum(relu_diff ** 2)
                return -jnp.sqrt(jnp.maximum(dist_sq, 1e-6))
            else:
                return jnp.sum(psi_s_single * phi_g)

        self._grad_fn_goal = jax.jit(jax.vmap(jax.grad(value_fn_wrt_goal, argnums=0)))

    N = obs_np.shape[0]
    psi_s = np.array(self._agent.get_psi(obs_np))  # (N, D)
    grads_jnp = self._grad_fn_goal(jnp.array(goal_np), jnp.array(psi_s))  # (N, obs_dim)
    return np.array(grads_jnp)[:, :2]  # (N, 2)
```

#### `HILPMemoizedWrapper.compute_grads_wrt_second_arg`

goal embedding (`phi_g_grid`)에 대해 finite difference:

```python
def compute_grads_wrt_second_arg(
    self,
    obs_np: np.ndarray,
    goal_np: np.ndarray,
    eps: float = 0.5,
) -> np.ndarray:
    """∂V(obs, goal)/∂goal[:2] — finite diff over phi_g_grid."""
    N = obs_np.shape[0]
    obs_np   = obs_np.astype(np.float32)
    goal_rep = np.broadcast_to(goal_np[:1], (N, goal_np.shape[-1])).copy().astype(np.float32)

    # psi(obs)는 goal과 무관하므로 한 번만 계산
    xi_s, yi_s = self._xy_to_fidx(obs_np)
    psi = self._bilinear(self._psi_grids[0], xi_s, yi_s)  # (N, D)

    grads = np.zeros((N, 2), dtype=np.float32)
    for dim in range(2):
        g_p = goal_rep.copy(); g_p[:, dim] += eps
        g_m = goal_rep.copy(); g_m[:, dim] -= eps
        xi_p, yi_p = self._xy_to_fidx(g_p)
        xi_m, yi_m = self._xy_to_fidx(g_m)
        phi_g_p = self._bilinear(self._phi_g_grids[0], xi_p, yi_p)
        phi_g_m = self._bilinear(self._phi_g_grids[0], xi_m, yi_m)
        grads[:, dim] = (
            self._aggregate(psi, phi_g_p) - self._aggregate(psi, phi_g_m)
        ) / (2 * eps)
    return grads
```

`share_encoder=True`이면 `psi_grids[0] == phi_g_grids[0]`이므로, `quasimetric` 하에서 `∂V/∂goal`을 올바르게 계산하려면 `_aggregate`에 `"quasimetric"` 케이스 구현이 선행되어야 한다 (단계 0에서 처리).
