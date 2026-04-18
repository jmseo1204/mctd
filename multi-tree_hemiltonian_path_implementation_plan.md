# Multi-Tree Hemiltonian Path Implementation Plan

## 1. 이번 턴에서 확정된 설계 결정

사용자 지시에 따라 아래는 이미 확정된 사항으로 본다.

- `eval_hemiltonian.sh`는 `eval_all.sh` 스타일의 task 부여 방식을 따른다.
- 1차 실행 분량은 task별 1회, 총 5 jobs로 제한한다.
- `use_uncertainty_as_value=True` 경로만 지원한다.
- `uncertainty_mode`에 `expected_root_node_dist`를 추가한다.
- 1차 구현의 기본 selection value는 `-(cum_td_i + T_curr + cum_td_j)`를 사용한다.
  - `T_curr`: 현재 expanding node와 선택된 target node 사이의 temporal distance
- 첫 accepted connection은 고정한다.
  - 나중에 더 짧은 동일 pair connection이 나와도 accepted edge 자체는 교체하지 않는다.
- intermediate edge는 각 edge별 local postprocess만 적용한다.
- edge concat 이후 `_reorder_plan_by_proximity`는 적용하지 않는다.
- full connection 실패 시 best partial route를 fallback으로 실행한다.
- tentative Hamiltonian solver는 `D_direct`만 사용한다.
- `D_walk_global`은 fallback에서만 사용한다.
- fast uncertainty sampling / cluster-subplan reuse는 현재 config 그대로 유지한다.
- fast uncertainty sampling에서의 target selection도 main target selection과 같은
  `reliable_TD_threshold + tentative-neighbor rule`을 따른다.
- target node semantics는 `meeting_target_node`, `value_target_node`의 2종으로 분리한다.
  - `meeting_target_node`: child가 실제로 expand된 뒤, 새로 생성된 segment를 기준으로 fresh하게 재선정
  - `value_target_node`: pre-expansion parent ranking / uncertainty conditioning 단계에서 선정된 저장 target을 유지
- cluster cache는 별도 target-provenance key로 invalidate하지 않는다.
  - `cluster_subplans`는 parent에서 발견된 continuation mode 집합으로 해석한다.
  - 따라서 target tree / target node가 바뀌어도 기존 cluster를 바로 폐기하지 않는다.
- closure-based fallback에서는 future anchor를 hidden intermediate로 허용하지 않는다.
- middle tree가 tentative path 상 두 neighbor를 모두 가질 때는 더 가까운 쪽을 우선 target한다.
- 다만 과거 라운드에서 이미 접합된 neighbor tree 소속 후보는 target 적격성 심사에서 reject한다.
- accepted pair reject 규칙은 target selection / meeting acceptance에만 적용한다.
- `best_direct_bridge` / `D_direct` 관측 업데이트는 accepted pair에 대해서도 계속 수행할 수 있다.
- accepted된 pair의 solver 입력 cost는 raw direct cost를 계속 갱신해도 된다.
- 이유: `D_direct + hard adjacency constraint` 조합에서는 accepted pair cost는 feasible path들 사이에서 상수항이므로,
  tentative path argmin 자체에는 보통 영향을 주지 않는다.
- waypoint root 초기화는 현재 goal root 초기화 방식과 동일하게 간다.
  - `initial_sim_state` 복사
  - `qpos[:2] = waypoint_xy`
  - `qvel = 0`

## 2. 설계의 핵심 변화

이전 문서의 중심 아이디어는 아래였다.

- tree별 connection budget
- accepted edge를 greedily 누적
- budget 소진 시 탐색 종료

지금은 이 구조를 버리고 아래 구조로 바뀐다.

- 매 라운드마다 tree-level 최단거리 정보를 업데이트한다.
- 그 시점의 tree-level 거리표를 이용해 `start ... goal` constrained Hamiltonian path를 다시 푼다.
- 이 path를 `tentative optimal path`로 사용한다.
- target selection과 meeting acceptance는 이 tentative path를 기준으로 유도된다.
- accepted connection은 “현재 tentative path가 요구하는 인접 tree pair”일 때만 고정한다.

즉, 탐색을 connection counter가 이끄는 것이 아니라, online Hamiltonian planner가 이끈다.

## 3. 꼭 구분해야 하는 세 종류의 tree-level 거리 개념

여기서 중요한 설계 판단이 하나 있다.

사용자가 제안한 `i-hop update`, `2-tree-hop update`, `Floyd-Warshall`류의 closure는 의미가 있다. 다만 이 값을 Hamiltonian solver에 넣을 때는 무엇을 의미하는 거리인지 구분해야 한다.

예:

- `d(S, W2)`가 사실은 `S -> W1 -> W2`를 통해 얻어진 multi-hop 값일 수 있다.
- 그런데 Hamiltonian solver가 이것을 “S와 W2를 직접 이웃시키는 비용”으로 해석하면, `W1`를 이미 내부적으로 재사용한 비용을 또 adjacency cost로 쓰게 된다.
- 그러면 “한 번만 방문해야 하는 waypoint” 제약과 비용 해석이 충돌한다.
- 더 나쁜 경우, `d(A, B)`의 shortest walk가 `A -> T -> B`처럼 아직 나중에 방문해야 할 future anchor를 내부 경유할 수 있다.
- 그러면 tentative order가 `S -> A -> B -> T`로 나오더라도, 실제 `A-B` pair를 expand하면 `T`를 먼저 지나가게 되어 order semantics 자체가 깨진다.

따라서 거리 개념을 아래처럼 분리하는 것이 안전하다.

### 3.1 `D_direct`

의미:

- “tree_i와 tree_j를 직접 인접시키는 후보 edge”의 현재 최선 비용

초기화:

- 각 tree root observation끼리의 temporal distance

업데이트:

- 확장된 node를 매개로 계산한 direct bridge 후보로만 감소시킨다.
- multi-hop closure로 갱신하지 않는다.

용도:

- tentative Hamiltonian path solver의 실제 입력 cost matrix
- final fallback route의 unaccepted edge 후보 선택

### 3.2 `D_walk_global`

의미:

- 현재까지 발견된 direct bridge들을 그래프로 보고, tree-level all-pairs shortest walk를 취한 closure
- predecessor 정보까지 저장해서 `(i, j)` pair를 실제 tree sequence로 복원할 수 있어야 한다.

업데이트:

- `D_direct`가 바뀐 뒤 Floyd-Warshall 또는 그 동등한 방식으로 재계산

용도:

- best partial fallback에서 “직접 bridge가 없는 pair를 이미 발견된 bridge chain으로 메우는” 용도
- 진단용
- optional heuristic

중요:

- `D_walk_global`를 Hamiltonian solver의 adjacency cost로 직접 쓰면 hidden intermediate anchor 문제가 생길 수 있다.
- 특히 아직 나중에 방문해야 할 future anchor를 내부 경유하는 shortest walk가 허용되면 tentative order 의미가 붕괴한다.

현재 기준 권장안은:

- Hamiltonian solver는 `D_direct`만 사용
- `D_walk_global`는 fallback walk 복원용으로 유지

### 3.3 `D_walk_restricted(prefix/segment aware)` (선택적 개념)

의미:

- closure를 쓰되, intermediate anchor를 아무나 허용하지 않고
  - 이미 accepted된 segment 내부
  - 혹은 이미 방문한 prefix
  로 제한한 order-dependent walk distance

평가:

- 사용자 예시의 `A-B`를 `A-S + S-B`로 메우는 발상은, `S`가 이미 start로 먼저 방문되므로 이런 restricted walk 개념으로는 자연스럽다.
- 하지만 이 거리는 order-dependent라서 단일 정적 matrix로 표현하기 어렵다.
- 1차 구현에서 여기를 바로 구현하는 것은 복잡도가 크다.

따라서 현재 권장안은:

- tentative Hamiltonian solver 입력은 `D_direct`
- `A-S-B` 같은 closure concatenation은 우선 fallback execution에서만 사용
- solver 단계에서 closure를 쓰고 싶다면, “future anchor를 hidden intermediate로 허용할지”를 먼저 명확히 결정해야 한다.

## 4. 새 런타임 데이터 구조

### 4.1 Tree metadata

`MCTSTreeState`에 아래 metadata를 추가한다.

- `anchor_idx: int`
- `anchor_kind: Literal["start", "waypoint", "goal"]`
- `anchor_label: str`

anchor 순서는 항상 아래로 둔다.

1. `start`
2. `waypoint_1 ... waypoint_k`
3. `goal`

이 순서는 입력 anchor numbering일 뿐이며, 실제 방문 순서는 tentative Hamiltonian path가 결정한다.

### 4.2 Node metadata

`TreeNode`에 아래 필드를 추가한다.

- `cum_temporal_dist_from_root: float`

의미:

- 자기 tree의 root node에서 현재 node까지의 누적 temporal distance

업데이트 규칙:

- root node는 `0.0`
- child node는 아래 재귀식
  - `child.cum_temporal_dist_from_root = parent.cum_temporal_dist_from_root + T(parent, child)`

중요:

- `T(root, child)`를 root-child 직통 embedding distance로 재지 않는다.
- 반드시 `parent -> child` local TD를 더하는 누적 방식으로 계산한다.

### 4.3 Tree-pair direct bridge repository

새 repository를 둔다.

- `best_direct_bridge[(i, j)]`

여기에는 아래를 저장한다.

- `tree_i_idx`
- `tree_j_idx`
- `node_i`
- `node_j`
- `bridge_td`
- `cum_td_i`
- `cum_td_j`
- `total_td = cum_td_i + bridge_td + cum_td_j`
- `source_round`
- `is_accepted_edge: bool`

이 repository는 unordered pair 기준으로 관리한다.

같은 라운드에 동일 `{i, j}` pair가 여러 번 갱신되면:

- `total_td`가 더 작은 것을 채택한다.

중요:

- 이 repository는 “관측된 최단 direct bridge”를 저장하는 raw repository다.
- target selection에서 accepted pair를 reject하더라도, round postprocess에서 새 child node를 기준으로
  모든 other tree와의 최소 direct bridge를 스캔하면 accepted pair의 raw direct cost도 계속 낮아질 수 있다.
- 즉, `accepted pair reject`와 `raw direct bridge 관측`은 별개다.

### 4.4 Accepted edge repository

accepted connection은 별도로 관리한다.

- `accepted_pair_edges[(i, j)]`

여기에는 아래를 저장한다.

- `accepted node_i`
- `accepted node_j`
- `accepted total_td`
- `accepted round`

중요:

- accepted pair는 `best_direct_bridge`와 별개로 “실행에 사용할 고정 edge”다.
- 나중에 같은 pair의 더 짧은 bridge가 발견되어도 accepted edge는 교체하지 않는다.
- 따라서 pair별로 개념적으로는 아래 두 값을 분리해서 보는 것이 좋다.
  - `best_direct_bridge_raw[(i, j)]`
  - `effective_pair_cost[(i, j)]`
- accepted 전에는 둘이 같다.
- accepted 후에는:
  - `best_direct_bridge_raw[(i, j)]`는 계속 내려갈 수 있다.
  - `effective_pair_cost[(i, j)]`도 raw direct cost를 따라 계속 갱신해도 된다.
  - 다만 final execution은 여전히 `accepted_pair_edges[(i, j)]`를 사용한다.

### 4.5 Accepted segment state

accepted edge가 누적되면 anchor-level path segment가 생긴다.

이를 관리하기 위한 상태를 둔다.

- disjoint accepted path segments
- 각 tree의 accepted degree

여기서 강조할 점:

- 이 degree/segment state는 더 이상 “탐색을 budget으로 유도하는 장치”가 아니다.
- accepted adjacency constraint가 일관적인지 확인하고, constrained Hamiltonian solver에 block structure를 넘기기 위한 bookkeeping이다.

즉, 이전 문서의 `connection budget planner`와는 역할이 다르다.

## 5. Online Hamiltonian planner

### 5.1 왜 기존 `solve_fixed_endpoint_hamiltonian_path`를 그대로 쓰면 부족한가

현재 `utils/route_metric_utils.py::solve_fixed_endpoint_hamiltonian_path`는 아래만 처리한다.

- start 고정
- goal 고정
- waypoint 순열 brute-force

하지만 새 요구사항에서는 accepted edge가 hard constraint가 된다.

예:

- 이미 `{W1, W3}`가 accepted면
- 앞으로 tentative path는 반드시 `W1`과 `W3`를 이웃하게 유지해야 한다.

따라서 solver는 아래를 지원해야 한다.

- fixed endpoint
- hard adjacency constraints
- accepted edges가 여러 개면 그들이 이루는 segment/block을 유지

### 5.2 권장 구현 방식: segment/block brute force

tree 개수 `M`이 작기 때문에 가장 단순하고 안전한 구현은 generic graph algorithm보다 아래다.

1. accepted edge들로부터 disjoint path segments를 만든다.
2. segment 내부 순서는 고정하되, segment 전체 방향은 필요 시 flip 가능하게 둔다.
3. accepted되지 않은 singleton tree는 길이 1 segment로 본다.
4. 이 segment들을 super-node처럼 보고:
   - permutation
   - segment orientation
   을 brute-force해서 최적 경로를 찾는다.

장점:

- hard adjacency constraint를 정확히 반영할 수 있다.
- `start`, `goal`이 segment 내부 어디에 올 수 있는지도 명확히 제어 가능하다.
- 최종 plan assembly와 구조가 잘 맞는다.

### 5.3 tentative path solver 입력

입력:

- active trees / accepted segments
- `D_direct`
- accepted edge constraints

출력:

- `tentative_anchor_order`
- `tentative_adjacent_pairs`
- `tentative_total_cost`

이 tentative path는 매 라운드 시작 전에 한 번 계산한다.

## 6. 라운드 단위 실행 흐름

권장 흐름은 아래다.

### Round Step 0. tentative path 계산

현재 시점의:

- `D_direct`
- accepted segments

를 바탕으로 constrained Hamiltonian path를 푼다.

산출물:

- `tentative_anchor_order`
- `tentative_adjacent_pairs`

이 결과는 이번 라운드에서:

- target tree 적격성 판정
- meeting acceptance eligibility

의 기준으로 사용한다.

### Round Step 1. parent node selection

기본 원칙은 기존과 동일하다.

- tree 구분 없이 전체 expandable parent node를 모은다.
- value가 높은 순으로 정렬한다.
- `parallel_search_node`개 parent를 뽑는다.

tie-break는 기존처럼:

- 높은 value 우선
- 깊은 depth 우선
- name lexical

### Round Step 2. pre-expansion `value_target_node` selection

확장하려는 node `u in tree_i`에 대해:

1. `tree_i`를 제외한 모든 tree의 모든 node를 모은다.
2. 각 후보 node `v in tree_j`와의 temporal distance를 계산한다.
3. 가까운 순으로 정렬한다.
4. 순차적으로 적격성을 검사한다.

검사 규칙:

- 후보 node가 이미 accepted된 neighbor tree 소속이면 reject한다.
- 만약 `TD(u, v) < reliable_TD_threshold`이면:
  - `tree_j`가 현재 tentative path에서 `tree_i`의 인접 tree일 때만 target으로 채택
  - 아니면 reject 후 다음 후보 검사
- 처음으로 `TD(u, v) >= reliable_TD_threshold`인 후보를 만나면:
  - `tree_j`가 tentative 인접 tree인지와 무관하게 채택

결과:

- `target_tree = tree_j`
- `value_target_node = v`
- parent selection 시점의 ranking value는 `u`와 `v`를 기준으로 계산한다.

중요:

- 이 target selection 규칙은 “가까운 잘못된 tree로의 greedy attraction”을 억제하기 위한 장치다.
- threshold 밖의 먼 후보는 TD 자체의 신뢰도가 낮다고 보고 planner adjacency 강제를 풀어준다.
- tentative path 상 middle tree의 양 neighbor가 모두 후보를 제공하면, 더 가까운 쪽을 선택한다.
- 이 reject 규칙은 “그 tree를 향한 새로운 meeting을 만들지 않도록” 하기 위한 것이다.
- direct bridge raw observation 자체를 막기 위한 규칙은 아니다.
- 이 단계의 `value_target_node`는 parent ranking / uncertainty conditioning용 provisional target이다.
- 실제 child의 접합 판정에는 이 node를 그대로 재사용하지 않는다.

### Round Step 3. post-expansion meeting target recomputation

child `c`가 실제로 생성된 뒤에는 target semantics를 아래처럼 분리한다.

- `meeting_target_node`
  - child가 이번 라운드에 실제로 생성한 새 segment만을 기준으로 fresh하게 선택한다.
  - 후보는:
    - source tree 자신 제외
    - 이미 accepted된 neighbor tree 제외
    - 이미 satisfied된 tree 제외
    - pre-round tentative path 상 허용된 adjacent tree만 허용
  - segment와 node의 거리는 사용자가 지정한 정의를 따른다.
    - `dist(segment(c), node_j) = min_i temporal_dist(state_i, node_j)`
    - 여기서 `state_i`는 child가 이번 라운드에 생성한 새 segment 위 상태
  - 이 값을 최소화하는 candidate node를 `meeting_target_node`로 둔다.

- `value_target_node`
  - pre-expansion에서 parent ranking / uncertainty conditioning용으로 정해진 provisional target을
    child에 저장한 값을 그대로 유지한다.
  - post-expansion에는 이 stored value target을 덮어쓰지 않는다.
  - 다음 라운드에 해당 child가 다시 parent 후보가 될 때 fresh target/value는 그 시점에 transient하게
    다시 계산한다.

핵심 원칙:

- `meeting_target_node`와 `value_target_node`는 같은 node일 수도 있고 다를 수도 있다.
- cluster reuse는 exploration branching 수단으로 해석한다.
- 따라서 “cluster가 과거 어떤 target을 보고 생성되었는가”를 이유로
  `cluster_subplans`를 invalidate하지 않는다.

## 7. `temporal_dist` 모드와 기존 uncertainty path의 관계

이 부분은 구현상 매우 중요하다.

현재 `use_uncertainty_as_value=True` 경로는 사실상 아래를 포함한다.

- fast uncertainty sampling
- cluster_subplans
- uncertainty-derived value

하지만 `uncertainty_mode=expected_root_node_dist`에서는 value 정의가 완전히 다르다.

사용자 지시에 따라 1차 구현은 현재 `configurations/algorithm/df_planning.yaml` 세팅을 그대로 유지한다고 본다.

즉, `uncertainty_mode`만 `expected_root_node_dist`로 바뀌고 아래는 유지된다.

- `use_uncertainty_as_value: true`
- `use_cluster_subplan_as_expansion: true`
- `fast_sampling_multiple`
- `fast_sampling_steps`
- `use_kde_maximin_for_selecting_subplan_in_cluster`

재검토 결과, 이들을 반드시 꺼야 하는 치명적 이유는 없다.

왜 처음에 비활성화를 권장했는가:

- 현재 코드에서 root initialization, cluster_subplans 생성, node value 부여가 하나의 uncertainty path에 강하게 결합되어 있기 때문이다.
- 끄면 리팩토링이 단순해지기 때문에 보수적으로 제안했던 것이다.

하지만 실제 코드상:

- `cluster_labels`는 `uncertainty_mode`와 무관하게 `cluster_tail_by_temporal_dist(...)`로 항상 계산된다.
- 따라서 `expected_root_node_dist` 모드에서도 fast-sampling + clustering + cluster_subplan reuse를 유지할 수 있다.

현 시점 권장안은 아래다.

- global mixed expansion skeleton은 재사용한다.
- fast uncertainty sampling도 그대로 유지한다.
- cluster_subplans도 그대로 유지한다.
- 단, scalar value semantics만 바꾼다.

구체적으로:

- `uncertainty_mode=expected_root_node_dist`일 때 `_compute_node_uncertainty(...)`는
  - `cluster_labels`, `n_clusters`, `T_curr`는 계속 계산하고
  - selection에 쓰일 scalar value는
    `-(cum_td_i + T_curr + cum_td_j)`로 해석한다.
- `_compute_uncertainty_and_clusters(...)`는
  - cluster_subplans 생성 로직은 기존대로 유지하되
  - `values.append(float(unc_result["selection_value"]))`를 사용한다.

즉:

- “exploration branching은 현재 uncertainty/cluster config를 그대로 사용”
- “parent/root selection에 쓰이는 scalar value만 expected-root-node-distance 기반으로 바꾼다”

이 해석이 현재 config를 최대한 보존하면서도 사용자 요구와 가장 잘 맞는다.

## 8. `cum_temporal_dist_from_root` 계산 위치

현재 child node의 `obs/sim_state`는 `_update_expanded_children_state(...)`에서 채워진다.

따라서 최소 수정 위치는 여기다.

구체적으로:

1. child `obs`가 결정된 직후
2. `TD(parent.obs, child.obs)`를 계산하고
3. `parent.cum_temporal_dist_from_root + TD(parent, child)`를 child에 저장

이렇게 하면:

- rollout 사용 여부와 무관하게 한 곳에서 처리 가능
- direct bridge update 시 언제나 최신 누적 TD를 참조 가능

## 8.5 selection value로 무엇을 쓸 것인가

현재 확정값:

- `uncertainty_mode=expected_root_node_dist`일 때 selection value는
  `-(cum_td_i + T_curr + cum_td_j)`를 사용한다.

이 값을 채택한 이유:

- local TD만 보는 것이 아니라, 실제 root-to-root direct bridge total cost를 바로 ranking에 반영한다.
- online Hamiltonian planner가 계속 추적하는 tree-level direct bridge와 node-level selection objective를 일치시킨다.

비판적으로 보면 아래 리스크는 남아 있다.

1. shallow-node bias
   - `cum_td_i` 때문에 깊은 node가 불리해질 수 있다.
2. opposite-tree maturity bias
   - `cum_td_j` 때문에 상대 tree가 얼마나 성장했는지에 value가 좌우된다.
3. noise accumulation
   - 세 추정치를 합치므로 scalar variance가 커질 수 있다.

하지만 1차 구현에서는 이 리스크를 감수하고도 아래 장점이 더 크다고 본다.

- raw direct bridge table update와 같은 objective를 selection도 공유하게 된다.
- 사용자가 원하는 “나중 라운드에 더 낮은 root-to-root direct cost가 다시 발견될 확률 감소” 방향과 더 잘 맞는다.

실험 확장 여지는 남긴다.

- `temporal_dist`: `-T_curr`
- `expected_root_node_dist`: `-(cum_td_i + T_curr + cum_td_j)`

이 두 모드를 모두 지원하면 이후 비교가 가능하다.

## 9. Tree-level direct bridge update

### 9.1 direct bridge 후보 생성

이번 라운드에 새로 확장된 node `u in tree_i`에 대해, 모든 `tree_j != tree_i`마다:

1. `tree_j`의 모든 node와 `TD(u, v)`를 계산한다.
2. temporal distance가 최소인 `v* in tree_j`를 찾는다.
3. 아래 비용을 계산한다.

`candidate_total_td = cum_td_from_root(u) + TD(u, v*) + cum_td_from_root(v*)`

4. 이 값이 기존 `best_direct_bridge[(i, j)]`보다 짧으면 갱신한다.

중요:

- 이 단계는 target selection과 무관하다.
- meeting 판정과도 무관하다.
- 순수하게 tree-level direct adjacency cost estimate를 갱신하는 단계다.
- 따라서 accepted pair를 target으로 reject하더라도, raw direct bridge cost는 계속 더 낮아질 수 있다.

### 9.2 same-round duplicate pair 처리

동일 라운드에 동일 `{i, j}` pair에 대해 여러 candidate가 생기면:

- `candidate_total_td`가 가장 작은 것만 남긴다.

### 9.3 walk-closure update에 대한 평가

사용자가 제안한:

- 1-hop direct update
- 2-hop propagation
- i-tree-hop 반복

은 tree graph closure를 구하는 관점에서는 타당하다.

다만 구현 관점에서는 `M`이 작기 때문에 아래가 더 낫다.

- `D_direct` 갱신
- 그 뒤 `D_walk_global = floyd_warshall_with_predecessor(D_direct)`

이 방식이 낫다고 보는 이유:

- 코드가 단순하다.
- correctness가 명확하다.
- incremental multi-hop propagation 버그를 피할 수 있다.
- `M`이 작으면 성능 차이가 사실상 없다.

따라서 권장안은:

- direct bridge repository는 incremental update
- closure는 필요 시 라운드 끝마다 Floyd-Warshall로 통째로 재계산

추가 원칙:

- `best_direct_bridge_raw[(i, j)]`는 계속 갱신 가능
- accepted된 pair의 solver 입력 cost(`effective_pair_cost`)도 raw direct cost를 계속 반영할 수 있다.

## 10. Meeting acceptance

meeting 판정은 기존 2-tree 코드의 “preselected target만 본다”에서 벗어나,
이번 라운드에 생성된 child segment를 기준으로 `meeting_target_node`를 fresh하게
선정한 뒤 그 node를 대상으로 판정한다.

정리하면:

- parent selection 단계에서 사용한 provisional `value_target_node`는 meeting 판정에 직접 쓰지 않는다.
- child가 실제로 만든 새 segment를 기준으로 eligible node pool을 다시 훑는다.
- `dist(segment(c), node_j) = min_i temporal_dist(state_i, node_j)`를 사용해
  가장 가까운 eligible node를 `meeting_target_node`로 선택한다.
- meeting acceptance는 이 `meeting_target_node`를 기준으로 수행한다.

다만 accepted edge로 고정하는 조건은 바뀐다.

### 10.1 accepted eligibility

meeting candidate `{tree_i, tree_j}`를 accepted edge로 저장하려면:

- 이번 라운드에서 fresh하게 선택된 `meeting_target_node in tree_j`
- `{tree_i, tree_j}`가 현재 tentative path의 인접 pair여야 한다.

그렇지 않으면:

- meeting이 발생해도 accepted edge로는 저장하지 않는다.
- 단, best direct bridge repository 업데이트에는 반영할 수 있다.

### 10.2 same-pair multiple meetings in one round

동일 라운드에 동일 `{i, j}` pair meeting이 여러 개 생기면:

- `bridge_cost`가 더 짧은 경로 하나만 accepted 후보로 남긴다.
- tie-break가 필요하면 `dist(segment(c), meeting_target_node)`가 더 작은 쪽을 우선한다.

### 10.3 accepted edge 확정 후 처리

accepted edge `{i, j}`가 확정되면:

- `accepted_pair_edges[(i, j)]` 저장
- accepted segment state 갱신
- 이후 tentative path solver는 항상 이 pair를 인접 제약으로 유지

중요:

- accepted edge는 execution provenance이므로 고정한다.
- direct bridge repository는 별도로 더 짧아질 수 있어도 accepted edge는 바꾸지 않는다.

## 11. 탐색 종료와 fallback

탐색 종료 조건은 아래 둘 중 하나다.

- 모든 tree가 accepted edges에 의해 하나의 valid start-goal Hamiltonian chain으로 묶임
- search budget / termination 조건 도달

### 11.1 full success

full success면:

- accepted segment state로부터 최종 anchor order를 얻는다.
- 각 adjacent accepted pair에 대해 accepted edge record를 꺼내어 edge plan을 materialize한다.

### 11.2 partial fallback

full success가 아니면:

- 즉, search budget / `val_max_loops` / 기타 terminate 조건으로 main search loop가 끝났는데
  accepted edges만으로는 하나의 valid start-goal Hamiltonian chain이 완성되지 않은 경우
  fallback assembly로 들어간다.
- 현재 시점의 constrained tentative path를 사용한다.
- 각 adjacent pair마다:
  - accepted edge가 있으면 그것을 사용
  - 없으면 우선 `best_direct_bridge[(i, j)]`를 사용
  - direct bridge가 없고 `D_walk_global` 경로가 있으면, 그 predecessor chain을 따라
    “이미 earlier in tentative order 이거나 accepted된 anchor만 intermediate로 허용하는”
    multi-edge concat으로 메운다

즉, fallback은 “현재 online planner가 생각하는 best full order”를 따라가되:

- accepted edge 우선
- 그 다음 direct bridge
- 마지막으로 discovered walk closure

순으로 pairwise gap을 메운다.

중요:

- 이 closure-based fallback은 intermediate anchor 재방문을 일으킬 수 있다.
- 사용자의 `A-B`를 `[A-S, S-B]`로 메우는 예시는 이 범주에 속한다.
- 따라서 이것은 “strict direct adjacency”가 아니라 “현재까지 발견된 bridge graph 위의 executable walk”를 뜻한다.
- fallback은 “별도 planner로 전환”하는 것이 아니라, main search가 불완전하게 끝난 뒤
  final assembly 단계에서 pair materialization 정책을 완화하는 후처리 단계다.

## 12. Final plan assembly

최종 조립은 explicit node-pair edge builder를 중심으로 간다.

### 12.1 새 helper

`plan_postproc.py`에 아래 helper를 추가하는 것이 좋다.

- `_extract_output_plan_between_nodes(src_node, dst_node, plan_tokens, append_goal_pad=False, goal_normalized=None)`
- `_build_postprocessed_edge_from_nodes(src_node, dst_node, plan_tokens, append_goal_pad=False, goal_normalized=None)`

의미:

- `src tree root -> src_node`
- `dst tree root -> dst_node`

를 조합해 두 root를 잇는 하나의 edge plan을 만든다.

### 12.2 기존 helper 처리

기존 `_build_postprocessed_plan_from_node(...)`는 legacy wrapper로 남긴다.

- 2-tree mode에서는 그대로 사용
- multi-tree mode에서는 explicit node-pair helper 사용

### 12.3 concat 규칙

각 edge는 local postprocess까지만 적용한다.

그 후:

- edge sequence를 단순 concat
- anchor 중복 프레임이 있으면 접합 경계에서 최소한의 dedup만 수행 가능
- 전체 concat 후 `_reorder_plan_by_proximity`는 하지 않는다

## 13. 구현 순서와 함수 단위 수정 포인트

### Step 1. Config / launcher plumbing

#### `configurations/algorithm/df_planning.yaml`

추가/수정:

- `multi_tree_hemiltonian: false`
- `reliable_TD_threshold: <default>`
- `uncertainty_mode` 주석에 `temporal_dist`, `expected_root_node_dist` 추가
- 기존 `task_override_path`, `task_override_waypoint_group_idx`는 그대로 사용

의도:

- legacy path는 기본 off
- `eval_hemiltonian.sh`만 flag를 on

#### `scripts/generate_jobs_generalized.py`

수정 포인트:

- `ArgumentParser`에 아래 인자 추가
  - `--task-override-path`
  - `--task-override-waypoint-group-idx`
  - `--multi-tree-hemiltonian`
- `basic_job_config` 생성 시 아래 override 주입
  - `algorithm.task_override_path`
  - `algorithm.task_override_waypoint_group_idx`
  - `algorithm.multi_tree_hemiltonian`

#### `eval_hemiltonian.sh`

신규 파일:

- 구조는 `eval_all.sh`를 따른다.
- 단 repeats는 1, tasks는 5로 제한한다.
- 사용자 입력:
  - checkpoint
  - task override path
  - waypoint group idx
- 실행 인자:
  - `--num_tasks 5`
  - `--num_seeds 1`
  - `--task-override-path ...`
  - `--task-override-waypoint-group-idx ...`
  - `--multi-tree-hemiltonian`

### Step 2. Tree / node metadata 확장

#### `algorithms/diffusion_forcing/tree_node.py`

`TreeNode.__init__`에 추가:

- `cum_temporal_dist_from_root: float = 0.0`

`get_expandable_candidate()`가 반환하는 candidate dict에도 추가:

- `cum_temporal_dist_from_root`: child 생성 전에는 `None`

의도:

- expand 후 `_update_expanded_children_state(...)`에서 실제 값 채움

#### `algorithms/diffusion_forcing/df_planning.py`

`MCTSTreeState`에 추가:

- `anchor_idx`
- `anchor_kind`
- `anchor_label`

`DiffusionForcingPlanning.__init__`에 추가:

- `self.multi_tree_hemiltonian`
- `self.reliable_TD_threshold`

### Step 3. Episode-local multi-tree state 초기화

#### `algorithms/diffusion_forcing/df_planning.py::interact`

현재 2-tree 초기화 블록:

- `bidir_tree1 = self._init_mcts_tree(...)`
- `bidir_tree2 = self._init_mcts_tree(...)`

를 분기한다.

- legacy:
  - 기존 2-tree 그대로
- multi-tree:
  - 새 helper 호출

새 helper 제안:

- `_build_anchor_specs_from_current_task(start_obs, goal_obs, initial_sim_state) -> list[dict]`
- `_make_anchor_root_sim_state(initial_sim_state, anchor_xy) -> dict`
- `_init_multi_tree_states(anchor_specs, horizon) -> list[MCTSTreeState]`
- `_initialize_tree_distance_state(trees) -> dict`

역할:

- active waypoint group에서 anchor list 생성
- start/waypoint/goal root tree 생성
- `best_direct_bridge`, `D_direct`, `D_walk_global`, accepted segment state 초기화

### Step 4. Constrained Hamiltonian solver 추가

#### `utils/route_metric_utils.py`

기존 함수는 유지:

- `solve_fixed_endpoint_hamiltonian_path`

새 함수 제안:

- `build_accepted_segments(num_anchors, accepted_pairs, start_idx, goal_idx) -> list[dict]`
- `solve_constrained_fixed_endpoint_hamiltonian_path(distance_matrix, accepted_pairs, start_idx, goal_idx) -> dict`

반환 정보:

- `anchor_order`
- `adjacent_pairs`
- `segment_order`
- `total_cost`
- `feasible`

핵심 구현:

- accepted pairs로 disjoint segment 생성
- segment orientation 포함 brute-force
- `start`, `goal` endpoint 유지

### Step 5. `expected_root_node_dist` scalar value 경로 추가

#### `algorithms/diffusion_forcing/df_planning.py::_compute_node_uncertainty`

새 branch 추가:

- `self.uncertainty_mode == "temporal_dist"`
- `self.uncertainty_mode == "expected_root_node_dist"`

동작:

- 기존처럼
  - `cluster_labels`
  - `n_clusters`
  - `T_curr`
  는 계산
- 추가로 필요한 경우
  - `cum_td_i`
  - `cum_td_j`
  를 받아 selection value 계산

권장 반환 형식:

- `result["T_curr"]`
- `result["selection_value"]`

#### `algorithms/diffusion_forcing/df_planning.py::_compute_uncertainty_and_clusters`

수정:

- `temporal_dist` / `expected_root_node_dist` 모드일 때
  - `values.append(float(unc_result["selection_value"]))`
- cluster_subplans 생성은 기존 그대로 유지

#### `algorithms/diffusion_forcing/df_planning.py::_ensure_uncertainty_roots_initialized`

수정:

- 2-tree 전용 helper를 유지하되 multi-tree branch 추가

새 helper 제안:

- `_ensure_uncertainty_roots_initialized_multi(trees, horizon, conditions) -> dict[str, dict]`

역할:

- 각 root에 대해 fast uncertainty sampling + temporal_dist value 초기화

### Step 6. Target selection helper 일반화

#### `algorithms/diffusion_forcing/df_planning.py`

기존 helper:

- `_select_dynamic_goal(current_leaf_obs, opposite_tree_all_nodes)`

새 helper 제안:

- `_rank_target_candidates_from_tree_pool(current_leaf_obs, trees, excluded_tree_tag) -> list[dict]`
- `_select_online_hamiltonian_target_node(current_leaf_obs, source_tree, trees, tentative_adjacent_pairs, reliable_td_threshold, accepted_pairs) -> tuple[target_tree, target_node, td_value]`
- `_select_meeting_target_from_segment(segment_states, source_tree, trees, tentative_adjacent_pairs, accepted_pairs) -> Optional[dict]`

규칙:

- source tree 제외
- 이미 accepted된 neighbor tree면 reject
- `TD < reliable_TD_threshold`이면 tentative neighbor tree만 허용
- middle tree 양옆이 모두 가능하면 더 가까운 쪽
- 처음 만나는 `TD >= reliable_TD_threshold` 후보는 tree identity 무관하게 채택
- `meeting_target`은 child boundary obs가 아니라 이번 라운드 새 segment 전체를 사용한다.

적용 지점:

- main expansion target selection
- fast uncertainty sampling target selection
- round postprocess의 meeting target recomputation

### Step 7. Global parent selection / mixed expansion multi-tree화

#### `algorithms/diffusion_forcing/df_planning.py::_collect_global_expansion_candidates`

기존:

- `(tree, opposite_tree)` 전제

새 helper 제안:

- `_collect_global_expansion_candidates_multi(trees) -> list[dict]`

반환:

- `node`
- `tree`
- `value`

#### `algorithms/diffusion_forcing/df_planning.py::_select_global_expansion_parents`

새 helper 제안:

- `_select_global_expansion_parents_multi(trees) -> list[dict]`

정렬:

- value desc
- depth desc
- name asc

#### `algorithms/diffusion_forcing/df_planning.py::_run_global_uncertainty_expansion_round`

수정 방향:

- 함수 전체를 버리기보다 multi-tree metadata를 인자로 주입해 확장

필요한 추가 인자:

- `all_trees`
- `tentative_adjacent_pairs`
- `accepted_pairs`

이 함수 안에서 candidate별로:

- `selected_tree`
- `target_tree`
- provisional `value_target_node`

를 설정하게 한다.

### Step 8. `cum_temporal_dist_from_root` 및 direct bridge update

#### `algorithms/diffusion_forcing/df_planning.py::_update_expanded_children_state`

수정:

- child `obs`가 채워진 직후
  - `TD(parent.obs, child.obs)` 계산
  - `child.cum_temporal_dist_from_root` 저장

새 helper 제안:

- `_compute_temporal_distance(obs_a, obs_b) -> float`

#### `algorithms/diffusion_forcing/df_planning.py::_postprocess_tree_local_expansions`

현재 이 함수는:

- backprop
- child obs/sim_state update
- viz

까지만 한다.

multi-tree branch에서 추가할 새 helper:

- `_update_direct_bridge_repository_from_round(tree_batches, all_trees, distance_state)`
- `_select_meeting_target_node_from_new_segment(child_node, source_tree, planner_state) -> Optional[dict]`

역할:

- 새로 생성된 child node들로부터 모든 tree pair direct bridge 후보 계산
- `best_direct_bridge_raw` 갱신
- `effective_pair_cost` 갱신
- `D_walk_global` 재계산
- 각 child에 대해 fresh `meeting_target_node` 계산

주의:

- accepted pair도 raw observation은 계속 갱신할 수 있다.
- accepted pair cost도 raw direct cost를 계속 갱신할 수 있다.
- `cluster_subplans`는 target 변경만으로 invalidate하지 않는다.

### Step 9. Accepted meeting 처리

#### `algorithms/diffusion_forcing/df_planning.py`

현재 `_select_round_plan_candidate(...)`는 2-tree 즉시 종료용이다.

multi-tree용 새 helper 제안:

- `_collect_round_meeting_candidates(expanded_node_infos, tentative_adjacent_pairs) -> list[dict]`
- `_accept_round_meetings(meeting_candidates, accepted_state, distance_state) -> dict`

정책:

- pre-round tentative adjacent pair만 accepted eligibility 가짐
- 각 expanded child는 fresh `meeting_target_node`를 먼저 계산한 뒤에만 meeting 후보가 된다.
- `meeting_target_node` 선정에는 child의 이번 라운드 새 segment만 사용한다.
- `meeting_target_node` 선정 metric은
  `min_i temporal_dist(state_i, node_j)` 이다.
- 동일 `{i, j}` pair meeting 다수면 shortest만 채택
- 여러 pair는 같은 pre-round tentative path 기준으로 동시에 처리

### Step 10. Final assembly helpers

#### `algorithms/diffusion_forcing/plan_postproc.py`

새 helper:

- `_extract_output_plan_between_nodes(src_node, dst_node, plan_tokens, append_goal_pad=False, goal_normalized=None)`
- `_build_postprocessed_edge_from_nodes(src_node, dst_node, plan_tokens, append_goal_pad=False, goal_normalized=None)`

기존:

- `_build_postprocessed_plan_from_node(...)`

는 legacy wrapper로 유지

#### `algorithms/diffusion_forcing/df_planning.py`

새 helper 제안:

- `_assemble_success_plan_from_accepted_chain(anchor_order, accepted_pair_edges, plan_tokens) -> dict`
- `_assemble_partial_fallback_plan(anchor_order, accepted_pair_edges, best_direct_bridge, walk_state, plan_tokens) -> dict`
- `_materialize_fallback_pair(i, j, prefix_anchor_set, ...) -> list[edge_bundle]`

핵심:

- full success:
  - accepted edges만 사용
- partial fallback:
  - accepted edge
  - direct bridge
  - visited/accepted intermediate만 허용하는 walk closure
  순으로 사용

## 14. 검증 계획

최소 검증 케이스:

1. legacy `eval.sh`
   - 기존 2-tree 회귀 없음
2. `multi_tree_hemiltonian=true`, waypoint 0개
   - 일반화 코드가 2-tree case를 정상 처리
3. waypoint 1개
   - tentative path와 accepted edge가 일치하는지 확인
4. waypoint 2개 이상
   - accepted segment constraint가 solver에 유지되는지 확인
5. budget 내 full success 실패 case
   - partial fallback assembly가 동작하는지 확인

## 15. 현재 시점에서 추가로 명확히 받아야 할 질문

이번 턴 기준으로 아래는 확정된 것으로 본다.

- accepted edge adjacency 판정은 pre-round tentative path 기준
- accepted된 pair의 solver 입력 cost도 raw direct cost를 계속 갱신할 수 있다.
- `meeting_target_node` / `value_target_node` 2-target 구조를 사용한다.

남은 질문은 1개다.

- `meeting_target_node`를 `min_i temporal_dist(state_i, node_j)`로 선정한 뒤,
  최종 meeting acceptance threshold는 무엇으로 둘지 확정이 필요하다.
  - 안 A: 기존처럼 `_compute_plan_gap(child_node, meeting_target_node) < meeting_delta`
  - 안 B: 새로 정의한 `dist(segment(c), node_j) < meeting_delta`

이 한 점만 확인되면 구현에 바로 들어갈 수 있다.

## 16. 최종 권장안 요약

이번 설계에서 가장 중요한 판단은 아래 4개다.

1. `connection budget planner`를 버리고 online Hamiltonian planner로 바꾼다.
2. planner 입력 cost는 `D_direct`, closure는 보조 정보로만 둔다.
3. accepted edge는 hard constraint이자 execution provenance로서 고정한다.
4. partial fallback은 main search 종료 후 final assembly 단계에서만 발동하며, tentative full order 전체를 실행 대상으로 한다.

이 방향이면 기존 `df_planning.py`를 완전히 뒤집지 않고도:

- multi-tree root initialization
- temporal-distance-driven global expansion
- online constrained Hamiltonian planning
- accepted-edge-first fallback execution

을 한 흐름으로 묶을 수 있다.

## 17. Multi-Tree Directionality Cleanup Plan

현재 구현은 multi-tree search를 도입했지만, `G` anchor만 여전히 legacy bidirectional
backward-tree semantics를 일부 유지하고 있다. 특히 아래 항목들이 남아 있다.

- tree init에서 `G`만 `is_tree1=False`
- local segment extraction에서 tree 방향 분기
- rollout에서 `is_backward=(not active_tree.is_tree1)` 사용
- uncertainty/final-plan visualization에서 `"from_goal"` / suffix-valid-frame 가정
- 일부 pairwise postprocess helper의 forward/backward concatenation 가정

이번 턴에서는 bug fix 범위를 최소화하기 위해 `_extract_new_segment_obs()`만 먼저
수정했고, multi-tree mode에서는 새 segment를 항상 각 tree의 prefix
`[start_len:end_len)` 기준으로 슬라이싱하도록 맞췄다.

### 17.1 다음 리팩토링 순서

1. multi-tree mode에서 anchor initialization semantics 통일
   - `S`, `W*`, `G` 모두 동일한 "anchor-rooted forward-prefix tree"로 취급
   - legacy bidirectional mode만 `is_tree1` / backward semantics 유지

2. multi-tree rollout semantics 분리
   - `_update_expanded_children_state()`에서 multi-tree branch는
     `is_backward` 분기를 사용하지 않도록 정리
   - boundary obs / sim state update를 tree-prefix semantics에 맞게 통일

3. multi-tree visualization semantics 분리
   - `_get_plan_viz_valid_frame_bounds()`의 suffix valid-frame 가정을 multi-tree에는 적용하지 않음
   - `_node_path_label()`과 `from_goal` naming을 anchor-label 기반 표기로 교체

4. multi-tree postprocess / gap helper 점검
   - `_compute_plan_gap_to_target()`의 target-side flip이 multi-tree assembly/gap 의미와 맞는지 재검토
   - final edge builder에서 "forward tree vs backward tree" 전제 제거

### 17.2 확인이 필요한 구현 질문

1. multi-tree mode에서 `G` anchor도 다른 waypoint와 완전히 같은 semantics로 두고,
   rollout도 항상 anchor-rooted forward expansion으로 통일할지.
   - 내 권장: `yes`

2. multi-tree visualization에서 현재의 `forward_part_backward_part` 식 node path label을 버리고,
   단순히 `source_anchor_target_anchor_nodepath` 식으로 바꿀지.
   - 내 권장: `yes`

3. multi-tree mode에서 `_compute_plan_gap_to_target()`의 target-side `flip`도 제거할지.
   - 내 권장: `yes`, 다만 final edge postprocess와 함께 검토 필요
