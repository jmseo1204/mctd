# Multi-Tree Hemiltonian Path Implementation Plan

## 1. 목적

이 문서는 anchor refactoring 완료 이후, 현재 코드 기준의 multi-tree Hamiltonian path planner 설계와
남은 후속 작업을 정리하는 living document다.

중요한 전제는 아래와 같다.

- public planner entry는 더 이상 `multi_tree_hemiltonian` 플래그를 쓰지 않는다.
- public planner entry는 `use_anchor_planner` 기준으로 동작한다.
- 2-anchor와 multi-anchor는 같은 anchor planner semantics를 공유한다.
- compat runtime / shadow compare / legacy pairwise selector는 active planning path에서 제거되었다.

즉, 이 문서는 "legacy path에서 multi-tree로 어떻게 갈 것인가"를 설명하는 문서가 아니라,
"현재 anchor planner 안에서 multi-tree Hamiltonian runtime이 어떻게 동작하고 무엇이 남았는가"를 설명하는 문서다.

## 2. 현재 코드 기준 상태 요약

현재 기준 public surface와 runtime 축은 아래다.

- config entry
  - `configurations/algorithm/df_planning.yaml`
  - `use_anchor_planner: true`
  - `multi_tree_route_mode: online | fixed_temporal`
  - `uncertainty_mode: expected_root_node_dist`
  - `reliable_TD_threshold`
- launcher entry
  - `eval.sh`
  - `eval_all.sh`
  - `eval_hemiltonian.sh`
  - `scripts/generate_jobs_generalized.py --use_anchor_planner`
- runtime entry
  - `algorithms/diffusion_forcing/df_planning.py::_run_anchor_planner`
  - `algorithms/diffusion_forcing/df_planning.py::_run_multi_tree_online_hamiltonian_planner`
  - `algorithms/diffusion_forcing/df_planning.py::_run_anchor_policy_loop`

현재 active helper 축은 아래다.

- tentative route
  - `_compute_tentative_hamiltonian_solution`
- target selection
  - `_select_multi_tree_target_node`
- root uncertainty init
  - `_ensure_uncertainty_roots_initialized_multi`
- parent selection
  - `_select_multi_tree_expansion_parents`
- direct bridge update
  - `_update_multi_tree_direct_bridges_from_expansions`
- meeting target recomputation
  - `_update_multi_tree_meeting_targets_from_expansions`
- meeting acceptance
  - `_select_multi_tree_round_meetings`
  - `_accept_multi_tree_round_meetings`
- final assembly
  - `_assemble_multi_tree_plan_bundle`

반대로 아래는 앞으로 새 작업 기준으로 문서에 다시 등장시키지 않는다.

- `multi_tree_hemiltonian`
- `anchor_pairwise_policy`
- `anchor_shadow_compare`
- `_select_global_expansion_parents`
- `_select_round_plan_candidate`
- compat/native dual runtime 설명

## 3. 이번 문서에서 확정된 설계 결정

아래는 현재 코드와 anchor refactoring 결과를 반영한 확정 사항이다.

- benchmark parity가 1차 목표다. exact trace parity는 목표가 아니다.
- tie-break는 anchor semantics에 맞게 더 낮은 `bridge_cost`를 우선한다.
- `use_uncertainty_as_value=True` 경로만 현재 지원한다.
- `fixed_temporal`도 같은 anchor planner 범위 안에서 지원한다.
- `use_rollout=false` 기준 semantics가 현재 기준선이다.
- `use_rollout=true` 경로 cleanup은 별도 후속 작업이다.
- `uncertainty_mode=expected_root_node_dist`를 기본으로 사용한다.
- `temporal_dist`는 비교/실험용으로 유지할 수 있지만 main path의 기준은 아니다.
- fast uncertainty sampling, cluster-subplan reuse, KDE representative selection은 현재 config를 유지한다.
- target semantics는 `value_target_node`, `meeting_target_node`로 분리한다.
- cluster cache는 target provenance만으로 invalidate하지 않는다.
- accepted edge는 execution provenance로서 고정한다.
- 동일 pair의 더 짧은 raw bridge가 나와도 accepted edge 자체는 교체하지 않는다.
- accepted pair reject 규칙은 target selection / meeting acceptance에만 적용한다.
- raw `best_direct_bridge` / `effective_pair_cost`는 accepted pair에 대해서도 계속 갱신할 수 있다.
- tentative Hamiltonian solver 입력은 `D_direct` 계열 cost만 사용한다.
- walk closure는 solver 입력이 아니라 final assembly fallback 계층에서만 사용한다.
- closure fallback은 future anchor를 hidden intermediate로 허용하지 않는다.
- final assembly는 edge-local concat만 수행한다.
- edge concat 이후 `_reorder_plan_by_proximity`는 적용하지 않는다.
- meeting acceptance threshold는 열린 질문이 아니라 현재 native anchor 기준으로 고정한다.
  - `gap = _compute_plan_gap_to_target(...)`
  - accepted condition은 `gap < meeting_delta`

## 4. 핵심 개념

### 4.1 Tentative route source

`multi_tree_route_mode`는 tentative route의 source만 바꾼다.

- `online`
  - 현재 `effective_pair_cost`와 forced adjacency를 바탕으로
    `solve_fixed_endpoint_hamiltonian_path_with_forced_adjacency(...)`를 매 라운드 다시 푼다.
- `fixed_temporal`
  - `_compute_fixed_temporal_route_solution(...)`에서 만든 고정 route를
    `fixed_tentative_solver_result`로 저장하고 매 라운드 그대로 사용한다.

중요한 점은 아래다.

- 두 모드 모두 parent selection, meeting acceptance, final assembly는 같은 anchor runtime을 공유한다.
- 차이는 tentative route를 어디서 가져오느냐뿐이다.

### 4.2 Tree-level 거리 개념

현재 문서에서는 아래 세 층을 구분한다.

- `best_direct_bridge_raw`
  - tree pair 사이에서 관측된 최선 direct bridge total cost
- `effective_pair_cost`
  - tentative solver에 실제로 넣는 pair cost matrix
  - accepted 이후에도 raw direct 개선을 따라 계속 내려갈 수 있다
- restricted walk / repair fallback
  - direct connection이 없을 때 final assembly에서만 쓰는 보조 계층

핵심 원칙은 아래다.

- Hamiltonian solver는 closure cost를 직접 adjacency cost로 쓰지 않는다.
- solver는 direct adjacency 의미를 보존해야 한다.
- hidden intermediate anchor를 쓰는 closure는 solver 단계가 아니라 assembly 단계에서만 제한적으로 허용한다.

### 4.3 Node-level 거리 개념

`TreeNode`에는 `cum_temporal_dist_from_root`가 있다.

- root는 `0.0`
- child는 `parent.cum_temporal_dist_from_root + TD(parent, child)`

이 값은 아래 용도로 쓰인다.

- `expected_root_node_dist` selection value 계산
- source-target total cost 계산
- direct bridge repository 갱신

### 4.4 Target semantics 분리

현재 semantics는 아래처럼 분리된다.

- `value_target_node`
  - pre-expansion parent ranking / uncertainty conditioning용 target
  - `_select_multi_tree_target_node(..., use_segment_metric=False)`에서 선택
- `meeting_target_node`
  - child expansion 후, 새로 생성된 segment를 기준으로 다시 고른 target
  - `_select_multi_tree_target_node(..., use_segment_metric=True)` 계열 semantics로 갱신

중요한 점은 아래다.

- 두 target은 같을 수도 다를 수도 있다.
- parent ranking 때 쓴 provisional target을 meeting acceptance에 그대로 재사용하지 않는다.
- meeting acceptance는 항상 post-expansion fresh target 기준이다.

## 5. 현재 planner state 구조

`_init_multi_tree_planner_state(...)` 기준으로 planner state는 아래 필드를 중심으로 움직인다.

- `trees`
- `anchor_specs`
- `accepted_pair_edges`
- `accepted_neighbors`
- `best_direct_bridge_raw`
- `best_direct_bridge_info`
- `effective_pair_cost`
- `fixed_tentative_solver_result`
- `tentative_solver_result`

anchor initialization은 아래 규칙을 따른다.

- anchor order는 항상 `S`, `W1 ... Wk`, `G`
- waypoint root는 현재 goal root 초기화와 같은 방식으로 만든다
  - `initial_sim_state` 복사
  - `qpos[:2] = waypoint_xy`
  - `qvel = 0`
- multi-tree mode에서는 `S`, `W*`, `G` 모두 anchor-rooted forward-prefix tree semantics를 사용한다
  - 현재 `_init_multi_tree_planner_state(...)`에서도 모든 anchor가 `is_tree1=True`로 초기화된다

## 6. 현재 라운드 실행 흐름

현재 runtime은 `_run_multi_tree_online_hamiltonian_planner(...)` 안에서 아래 순서로 동작한다.

### Step 0. planner state 초기화

- start / waypoint / goal tree를 만든다
- root-to-root cost로 `best_direct_bridge_raw`와 `effective_pair_cost`를 초기화한다
- route mode가 `fixed_temporal`이면 fixed tentative solver result도 같이 넣는다
- 첫 tentative route를 계산한다

### Step 1. root uncertainty 초기화

`use_cluster_subplan_as_expansion=True`이면 `_ensure_uncertainty_roots_initialized_multi(...)`가 동작한다.

이 단계의 원칙은 아래다.

- root uncertainty sampling도 current target selection semantics를 따른다
- fast sampling / cluster reuse는 그대로 유지한다
- scalar value만 현재 `uncertainty_mode` 의미에 맞게 계산한다

### Step 2. tentative route 계산

매 라운드 시작 시 planner는 현재 `tentative_solver_result`를 기준으로 움직인다.

- `online`
  - `effective_pair_cost`와 forced adjacency에서 tentative route 계산
- `fixed_temporal`
  - 저장된 fixed route를 그대로 재사용

이 tentative route는 아래 두 곳의 hard reference다.

- target tree 적격성 판정
- accepted meeting eligibility 판정

### Step 3. parent selection

`_select_multi_tree_expansion_parents(...)`가 아래 규칙으로 parent 후보를 모은다.

- anchor가 이미 satisfied된 tree는 제외
- obs가 없거나 terminal인 node는 제외
- expandable하지 않은 node는 제외
- 각 node마다 `_select_multi_tree_target_node(..., use_segment_metric=False)`로 provisional target을 잡는다
- ranking value는 `_compute_selection_value_for_target(...)` 기준으로 계산한다

정렬 규칙은 현재 아래다.

- value desc
- depth desc
- `tree.anchor_idx`
- `node.name`

### Step 4. pre-expansion target selection

`_select_multi_tree_target_node(..., use_segment_metric=False)`의 현재 규칙은 아래다.

- source tree 제외
- accepted neighbor tree 제외
- already satisfied tree 제외
- 모든 candidate node를 temporal distance 기준으로 정렬
- `TD < reliable_TD_threshold`이면 tentative neighbor tree만 허용
- `TD >= reliable_TD_threshold`인 첫 후보는 non-neighbor tree여도 허용
- 모든 near candidate가 reject되면 best overall candidate로 fallback

추가 원칙:

- middle tree가 tentative path 상 양옆 neighbor를 모두 가질 때는 더 가까운 쪽이 먼저 선택된다
- 이 reject 규칙은 meeting creation을 제한하기 위한 것이지 raw direct bridge observation을 막기 위한 규칙이 아니다

### Step 5. expansion + child state update

`_run_global_uncertainty_expansion_round(...)`와 `_postprocess_tree_local_expansions(...)`를 거치며 child가 materialize된다.

현재 중요한 후처리는 아래다.

- child `obs` / `sim_state` 업데이트
- child `cum_temporal_dist_from_root` 업데이트
- visualization / logging

`cum_temporal_dist_from_root`는 현재 `_update_expanded_children_state(...)` 경로에서 갱신된다.

### Step 6. direct bridge update

`_update_multi_tree_direct_bridges_from_expansions(...)`가 이번 라운드에 생성된 child들을 기준으로
모든 tree pair direct bridge를 다시 스캔한다.

현재 규칙은 아래다.

- source child와 target tree 전체 node 사이의 minimum TD candidate를 찾는다
- `bridge_cost = cum_td_i + h(T_curr) + cum_td_j`
- 이 값이 더 작으면 `best_direct_bridge_raw` 갱신
- 같은 값으로 `effective_pair_cost`도 갱신
- orientation-aware `best_direct_bridge_info`도 저장

중요:

- accepted pair여도 raw direct observation은 계속 갱신된다
- accepted edge 고정과 raw direct cost update는 별개다

### Step 7. meeting target recomputation

`_update_multi_tree_meeting_targets_from_expansions(...)`는 expanded child마다
meeting target을 fresh하게 다시 잡는다.

현재 semantics는 아래다.

- child가 이번 라운드에 실제로 생성한 새 segment를 기준으로 본다
- metric은 segment-to-node minimum temporal distance다
- accepted neighbor tree는 제외한다
- satisfied anchor는 제외한다
- tentative path의 neighbor tree만 meeting candidate로 허용한다

즉, meeting target은 "현재 parent가 보고 있던 target"이 아니라
"새 segment가 실제로 가장 잘 붙을 수 있는 allowed target"이다.

### Step 8. meeting selection / acceptance

`_select_multi_tree_round_meetings(...)`와 `_accept_multi_tree_round_meetings(...)`가 accepted edge를 처리한다.

현재 acceptance 조건은 아래다.

- pair가 현재 tentative route의 forced adjacent pair여야 한다
- source/target anchor가 아직 satisfied 상태가 아니어야 한다
- target이 accepted neighbor가 아니어야 한다
- `gap = _compute_plan_gap_to_target(...)`가 유효해야 한다
- `gap < meeting_delta`를 만족해야 한다

same-pair tie-break는 현재 아래 순서를 따른다.

- 낮은 `bridge_cost`
- 낮은 `meeting_target_td`
- 낮은 `gap`
- `source_node.name`

accepted되면 아래 상태가 갱신된다.

- `accepted_pair_edges`
- `accepted_neighbors`

### Step 9. tentative route 재계산

meeting acceptance와 raw direct update가 끝나면 `_compute_tentative_hamiltonian_solution(...)`를 다시 호출한다.

즉, 현재 runtime은 아래의 online loop다.

1. 현재 pair cost로 tentative route 계산
2. 그 route가 허용하는 방향으로 expansion / meeting 수행
3. 새 direct bridge와 accepted edge를 반영해 route를 다시 계산

## 7. Final assembly와 fallback

현재 최종 출력 조립은 `_assemble_multi_tree_plan_bundle(...)` 하나로 통일된다.

이 함수는 pairwise legacy postprocess나 best-node rescue를 메인 경로로 사용하지 않는다.
현재 active assembly는 아래 순서로 동작한다.

### 7.1 route source

- 우선 현재 `tentative_solver_result["anchor_order"]`를 사용한다
- solver가 infeasible이면 `None`

### 7.2 pair materialization 우선순위

각 adjacent pair `A -> B`에 대해 아래 순서로 materialize한다.

1. accepted edge
2. `best_direct_bridge_info`
3. restricted anchor walk
4. edge-repair fallback

### 7.3 restricted walk semantics

direct pair가 없으면 `_find_restricted_anchor_walk(...)`를 사용한다.

현재 허용 규칙은 아래다.

- tentative order prefix 안의 anchor만 intermediate로 허용
- source / target은 항상 허용
- accepted edge와 discovered direct bridge를 모두 edge 후보로 사용할 수 있다

즉, closure는 아무 anchor나 밟는 shortest walk가 아니라
"이미 route prefix에 들어온 anchor만 intermediate로 허용하는 restricted walk"다.

### 7.4 edge-repair fallback

restricted walk도 없으면 `_repair_missing_pair_connection(...)`가 마지막 rescue layer로 동작한다.

현재 동작은 아래다.

- source tree의 materialized node들과 target tree node들을 전수 비교
- 최선 pairwise repair connection을 찾으면 `best_direct_bridge_info`에 주입
- 그 즉시 assembly를 계속 진행

이 fallback의 성격은 아래와 같다.

- 별도 planner 전환이 아니다
- main search 종료 후 final assembly 단계의 pair materialization 완화 계층이다
- route semantics를 유지한 채 missing edge를 메우는 마지막 수단이다

### 7.5 orientation handling

edge builder는 stored connection orientation과 desired route orientation이 다르면 flip을 허용한다.

중요한 원칙:

- tree 성장 semantics는 anchor-rooted forward-prefix로 통일한다
- 하지만 final route를 `S -> ... -> G` 방향으로 반환하기 위한 edge-level orientation 보정은 계속 필요하다
- 따라서 assembly 시 flip이 존재한다는 사실만으로 legacy bidirectional residue로 취급하지 않는다

## 8. 현재 구현 포인트 맵

앞으로 planner를 수정할 때 우선 봐야 할 지점은 아래다.

- config / public entry
  - `configurations/algorithm/df_planning.yaml`
  - `eval_hemiltonian.sh`
  - `scripts/generate_jobs_generalized.py`
- planner state init
  - `df_planning.py::_init_multi_tree_planner_state`
- tentative solver
  - `df_planning.py::_compute_tentative_hamiltonian_solution`
  - `utils/route_metric_utils.py::solve_fixed_endpoint_hamiltonian_path_with_forced_adjacency`
- target selection
  - `df_planning.py::_select_multi_tree_target_node`
- uncertainty value path
  - `df_planning.py::_compute_node_uncertainty`
  - `df_planning.py::_compute_uncertainty_and_clusters`
- parent selection
  - `df_planning.py::_select_multi_tree_expansion_parents`
- round postprocess
  - `df_planning.py::_update_multi_tree_direct_bridges_from_expansions`
  - `df_planning.py::_update_multi_tree_meeting_targets_from_expansions`
  - `df_planning.py::_select_multi_tree_round_meetings`
  - `df_planning.py::_accept_multi_tree_round_meetings`
- final assembly
  - `df_planning.py::_assemble_multi_tree_plan_bundle`
  - `df_planning.py::_materialize_connection_segment`
  - `df_planning.py::_find_restricted_anchor_walk`
  - `df_planning.py::_repair_missing_pair_connection`
- edge postprocess helper
  - `plan_postproc.py`

## 9. 남은 후속 작업

현재 이 문서 기준의 남은 큰 작업은 아래다.

### 9.1 `use_rollout=true` 경로 cleanup

anchor refactoring 문서와 동일하게, 현재 가장 큰 남은 기능 작업은 이쪽이다.

정리 목표:

- backward rollout residue를 planner semantics와 분리
- multi-tree anchor-rooted prefix semantics와 rollout state update를 완전히 맞춤
- `online` / `fixed_temporal` 모두 같은 방향성 규칙을 따르게 함

### 9.2 planner 외부 terminology cleanup

기능적 blocker는 아니지만 아래 영역은 terminology cleanup 여지가 남아 있다.

- 일부 env / executor naming
- planner 외부 주석과 legacy 문구

이 작업은 runtime semantics 자체보다 유지보수성 개선 성격이 강하다.

### 9.3 문서 유지 규칙

이 문서는 앞으로 아래 규칙으로 유지한다.

- retired flag나 compat helper를 다시 active path처럼 서술하지 않는다
- 미결 질문이 이미 코드에서 닫혔으면 문서에도 즉시 확정 사항으로 옮긴다
- helper 이름은 현재 코드 기준으로 적는다
- 2-anchor와 multi-anchor를 다른 planner처럼 설명하지 않는다

## 10. 검증 기준

현재 권장 검증 순서는 아래다.

1. `eval.sh`
   - 2-anchor `online` 회귀 없음 확인
2. `eval.sh`
   - 2-anchor `fixed_temporal` 회귀 없음 확인
3. `eval_hemiltonian.sh`
   - waypoint 1개 smoke check
4. `eval_hemiltonian.sh`
   - waypoint 2개 이상에서 tentative route / accepted edge / final route가 일관적인지 확인
5. partial assembly case
   - direct edge 부재 시 restricted walk와 edge-repair fallback이 기대대로 동작하는지 확인

기본 원칙은 아래다.

- 자동 metric gate보다 사용자의 정성 평가가 우선이다
- 각 단계에서 route text, accepted pair, final assembled route의 정합성을 함께 본다
- 문제가 생기면 legacy path를 되살리는 대신 현재 anchor runtime 내부에서 원인을 좁힌다
