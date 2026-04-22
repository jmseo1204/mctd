# Anchor Refactoring Implementation Plan

## 1. 목표

최종 목표는 아래 두 가지다.

- `eval.sh`, `eval_all.sh`, waypoint benchmark를 포함한 모든 public planner entry가 anchor 기반 pipeline으로 동작한다.
- 내부 pairwise policy까지 anchor 기반으로 통합하여, legacy bidirectional core path를 제거한다.

현재는 1차 전환이 끝난 상태다.

- public flag는 `multi_tree_hemiltonian`에서 `use_anchor_planner`로 전환되었다.
- public entry는 anchor planner를 사용한다.
- runtime planner는 native anchor policy만 사용한다.
- compat planner와 shadow compare는 제거되었다.

즉, 현재 상태는 아래로 요약된다.

- 외부 orchestration: anchor로 통일됨
- 내부 2-anchor decision policy: native anchor로 통일됨
- 남은 후속 작업: `use_rollout=true` 경로의 별도 cleanup

## 2. 이번 리팩토링에서 이미 확정된 사항

사용자와 합의된 사항은 아래와 같다.

- 1차 목표는 exact trace parity가 아니라 benchmark parity다.
- tie-break는 anchor 철학에 맞게 더 낮은 비용을 우선한다.
- `fixed_temporal` mode도 이번 리팩토링 범위에 포함한다.
- `use_uncertainty_as_value=True` 경로만 우선 지원한다.
- `use_rollout=false` 기준으로 먼저 통일한다.
- 단계별 구현 후 사용자가 `eval.sh`를 직접 돌려 정성적으로 동작/성능을 판단한다.
- shadow compare는 콘솔 요약을 출력하되, 사후 분석을 위해 파일 로그도 남긴다.

## 3. 현재 코드 상태 요약

현재 분기 지점은 대략 아래와 같다.

- 공통 loop helper
  - `algorithms/diffusion_forcing/df_planning.py::_run_anchor_policy_loop`
- native anchor backend
  - `algorithms/diffusion_forcing/df_planning.py::_run_multi_tree_online_hamiltonian_planner`
- native parent selection
  - `algorithms/diffusion_forcing/df_planning.py::_select_multi_tree_expansion_parents`
  - `algorithms/diffusion_forcing/df_planning.py::_select_multi_tree_target_node`
- native meeting acceptance / edge assembly
  - `algorithms/diffusion_forcing/df_planning.py::_select_multi_tree_round_meetings`
  - `algorithms/diffusion_forcing/df_planning.py::_assemble_multi_tree_plan_bundle`
- 이미 정리된 legacy residue
  - compat runtime / selector / round helper 제거 완료
  - `_node_path_label` anchor presentation으로 통일 완료

핵심은 “public path와 internal runtime 모두 native anchor semantics로 통일되었다”는 점이다.

## 4. 남은 리팩토링 단계

### Stage 2. Shadow Compare 인프라 추가

목표:

- 2-anchor에서 compat와 native를 같은 task에서 동시에 비교할 수 있게 한다.
- 실제 planner output은 아직 compat를 사용한다.
- native 결과는 콘솔 summary + 파일 로그로만 남긴다.

구현 방향:

- 새 debug config를 추가한다.
  - 예시: `anchor_shadow_compare: false`
- 2-anchor 경로에서 compat 실행 후, 조건이 맞으면 native path를 별도로 한 번 더 돌려 비교한다.
- 로그 디렉토리는 `logs/anchor_policy_shadow_compare/`로 둔다.
- 파일 형식은 JSONL로 두고, run/task별 1 row를 기본으로 한다.

로그 항목 후보:

- `task_idx`
- `seed`
- `route_mode`
- `compat_success`, `compat_first_reach`, `compat_reward`
- `compat_anchor_order`, `native_anchor_order`
- `compat_route_text`, `native_route_text`
- `compat_selected_pair`, `native_selected_pair`
- `compat_bridge_cost`, `native_bridge_cost`
- `diff_summary`

주의:

- shadow compare는 planner-level compare로만 구현한다.
- native shadow plan을 실제 env에 rollout하면 step counter / env internal state 부작용이 생길 수 있으므로, Stage 2에서는 실행 metric을 직접 비교하지 않는다.

이번 단계의 의도:

- 아직 policy는 바꾸지 않고, 이후 단계에서 native화할 때 빠르게 이상 징후를 추적할 수 있게 한다.

### Stage 3. 2-anchor Parent Selection을 Native Anchor 방식으로 전환

목표:

- 2-anchor에서도 parent ranking이 native anchor의 target-aware selection을 사용하게 만든다.
- `node.value` 중심 global ranking 의존성을 줄인다.

진행 상태:

- 구현 적용됨. `anchor_pairwise_compat`의 uncertainty branch가 native selector가 이해하는 최소 pairwise planner state를 만들고, `_select_multi_tree_expansion_parents`를 사용하도록 바뀌었다.
- 아직 meeting acceptance와 final assembly는 compat semantics를 유지한다.

구현 방향:

- compat 2-anchor 후보 수집을 별도 legacy collector로 두지 않고, anchor-style candidate selection hook으로 옮긴다.
- `S` tree와 `G` tree 각각에서 “현재 tentative route가 요구하는 neighbor”를 기준으로 target node를 잡는다.
- 2-anchor에서는 neighbor가 서로 하나뿐이므로, native selection은 자연스럽게 `S <-> G` pair로 축소된다.
- `fixed_temporal`도 동일한 selection path를 사용하되, route source만 fixed solver를 따르게 한다.

이 단계가 끝나면:

- 확장 parent를 고르는 방식은 2-anchor와 waypoint case가 거의 같은 구조를 공유하게 된다.
- 다만 meeting acceptance와 final bundle materialization은 아직 compat가 남아 있을 수 있다.

### Stage 4. 2-anchor Meeting Acceptance를 Edge-Centric Anchor 방식으로 전환

목표:

- 2-anchor meeting 판단을 node winner/fallback 중심에서 accepted edge 중심으로 바꾼다.
- 2-anchor도 anchor graph의 special case로 취급한다.

진행 상태:

- 구현 적용됨. `anchor_pairwise_compat`의 uncertainty branch가 expansion 후 native multi-tree와 같은 순서로
  - direct bridge 갱신
  - `meeting_target_node` 갱신
  - accepted edge 선택
  을 수행한다.
- accepted edge가 생기면 compat의 node-based meeting winner 대신 edge-local pairwise bundle을 바로 materialize해서 종료한다.
- accepted edge가 없을 때만 기존 compat round fallback을 사용한다.

구현 방향:

- `meeting_target_node` 기반 pair acceptance를 2-anchor에서도 사용한다.
- acceptance 조건은 native anchor와 동일하게 유지한다.
  - forced pair 여부
  - satisfied anchor reject
  - `gap < meeting_delta`
- tie-break는 bridge cost 우선으로 둔다.
- 필요하면 compat-style fallback plan은 transition 동안만 유지한다.

주의점:

- 시각화에서 가까워 보여도 acceptance는 `meeting_target_node`와의 `gap` 계산을 통과해야 한다.
- 2-anchor native화에서도 이 acceptance 기준은 waypoint case와 동일하게 유지하는 것이 목표다.

### Stage 5. 2-anchor Final Plan Materialization을 Anchor Assembly로 통일

목표:

- 2-anchor 최종 출력도 node-based compat postprocess가 아니라 edge assembly를 통해 생성한다.

진행 상태:

- 구현 적용됨. `anchor_pairwise_compat`도 search loop 종료 후 `planner_state`를 기준으로 `_assemble_multi_tree_plan_bundle()`를 먼저 시도한다.
- 따라서 2-anchor 최종 출력은 accepted edge / direct bridge / restricted walk / repair fallback을 거친 anchor assembly 결과를 우선 사용한다.
- assembly가 `None`일 때만 기존 `best_node` 기반 fallback이 남아 있다.

구현 방향:

- `S -> G`를 anchor order `[0, 1]`의 특수 케이스로 해석한다.
- final output은 connection bundle / edge assembly 경로를 사용한다.
- orientation mismatch는 현재 multi-tree 경로와 마찬가지로 assembly 시점에서 보정한다.
- 이 단계에서는 “backward tree”를 전제로 한 output reverse 의존성을 제거한다.

이 단계가 끝나면:

- 최종 출력 생성 규칙 자체가 2-anchor와 multi-anchor에서 동일해진다.

### Stage 6. 공통 Pairwise Edge-Repair Fallback 추가

목표:

- 2-anchor와 multi-anchor 모두에서 사용할 수 있는 공통 edge-repair fallback을 추가한다.
- assembly가 full route를 만들지 못하더라도, route semantics를 유지한 채 start-anchored repair를 시도한다.

구현 방향:

- fallback은 node 하나를 뽑아 전체 route를 대체하는 방식이 아니라, solver route의 missing pair edge마다 개별 repair를 수행하는 방식으로 설계한다.
- repair 대상은 현재 solver route에서 assembly가 concrete connection을 찾지 못한 pair다.
- 각 missing pair `A -> B`에 대해 pairwise edge-local repair candidate를 찾고, 성공하면 그 edge bundle을 assembly graph에 주입한다.
- repair는 start-anchored route semantics를 깨지 않아야 하므로, 전체 route와 무관한 single best node rescue는 사용하지 않는다.
- 2-anchor는 missing pair가 `S -> G` 하나뿐인 special case로 처리된다.

주의점:

- 이 fallback은 현재 native anchor pipeline이 이미 가진
  - accepted edge
  - `best_direct_bridge`
  - restricted walk
  이후에만 동작하는 추가 repair 계층이다.
- 즉, 기존 fallback을 대체하는 것이 아니라 마지막 rescue layer로 추가한다.

이 단계가 끝나면:

- 2-anchor와 multi-anchor 모두에서 “assembly가 못 잇는 missing pair”를 공통 방식으로 수리할 수 있다.

진행 상태:

- 구현 적용됨. `_assemble_multi_tree_plan_bundle()`가 missing direct edge 또는 walk sub-edge를 만났을 때, 두 anchor tree의 실제 materialized node들을 전수 비교해서 pairwise repair connection을 찾는다.
- repair 성공 시 `best_direct_bridge_info`에 주입되고 assembly가 그대로 이어진다.
- repair는 2-anchor와 multi-anchor 모두 동일한 코드 경로를 사용한다.

### Stage 7. `auto` 기본값을 Native 2-anchor로 전환

목표:

- `anchor_pairwise_policy=auto`일 때 waypoint가 없어도 native anchor policy를 기본으로 사용하게 한다.

진행 상태:

- 구현 적용됨. `anchor_pairwise_policy=auto`는 이제 waypoint가 없어도 `native_anchor`를 반환한다.
- `legacy_compat`는 manual override일 때만 진입한다.

구현 방향:

- 전환 직후에도 `legacy_compat`는 hidden fallback으로 남긴다.
- 사용자가 문제가 있다고 판단하면 일시적으로 compat로 되돌릴 수 있게 한다.
- shadow compare는 전환 직후에도 유지해 diff를 계속 축적한다.

전환 기준:

- 사용자가 단계별 `eval.sh` 결과를 보고 “기존과 동등하게 작동한다”고 판단하면 다음 단계로 진행한다.

### Stage 8. Compat를 Hidden Fallback으로 축소

목표:

- public/default path에서는 compat가 더 이상 사용되지 않게 한다.
- 내부 emergency switch로만 잠시 유지한다.

진행 상태:

- 구현 적용됨. public/default 2-anchor path는 native anchor를 사용하고, compat는 `anchor_pairwise_policy=legacy_compat`를 수동 지정했을 때만 사용된다.
- `anchor_shadow_compare`는 main planner가 compat이든 native이든 반대 policy를 shadow로 실행해 비교하도록 바뀌었다.

구현 방향:

- `auto`와 public scripts는 전부 native 2-anchor를 사용한다.
- compat는 debug/manual override로만 남긴다.
- shadow compare가 충분히 안정화되면 compat 호출 빈도를 줄인다.

### Stage 9. Compat 및 Legacy Residue 제거

진행 상태:

- 구현 적용됨. compat runtime, shadow compare, pairwise legacy helper, legacy visualization presentation을 제거했다.

목표:

- legacy bidirectional residue를 제거한다.

제거 대상:

- `_run_pairwise_anchor_compat_planner`
- 2-anchor에서의 `_select_global_expansion_parents` 사용 경로
- 2-anchor에서의 `_select_round_plan_candidate` 의존 경로
- backward-tree presentation 위주의 visualization label residue

Stage 9 결과:

- `_run_pairwise_anchor_compat_planner` 삭제
- `_run_anchor_planner`의 compat 분기 삭제
- `_select_global_expansion_parents` 삭제
- `_select_round_plan_candidate` 삭제
- `anchor_pairwise_policy`, `anchor_shadow_compare` 설정 제거
- `_node_path_label` anchor presentation으로 통일
- benchmark/hydra runtime은 native anchor planner만 사용

반대로 아래는 이번 Stage 9의 범위 밖으로 유지한다.

- `noise_schedule.py`의 bidirectional naming
- `env_manager.py`의 bidirectional naming
- `plan_postproc.py` 내부 legacy/bidirectional 주석 문구

이들은 기능적 런타임 의존성보다는 terminology cleanup 성격이 강하다.

이번 단계에서는 제외:

- `use_rollout=true` 경로의 backward rollout semantics cleanup

이건 다음 별도 작업으로 미룬다.

## 5. 단계별 검증 방식

이번 리팩토링은 자동 metric gate가 아니라 사용자 정성 평가 중심으로 진행한다.

기본 원칙:

- 각 단계 구현 완료 후 사용자가 `eval.sh`를 실행한다.
- 사용자는 기존 대비 “동작/성능이 체감상 같은지”를 판단한다.
- 이상이 없으면 다음 단계로 넘어간다.
- 이상이 있으면 그 단계에서 로그를 기반으로 원인을 좁힌다.

검증 우선순위:

1. 2-anchor `online`
2. 2-anchor `fixed_temporal`
3. waypoint가 있는 native anchor path에 회귀가 없는지 smoke check

## 6. 로그/디버그 운영 계획

shadow compare 도입 후 로그는 아래처럼 운영한다.

- 콘솔:
  - task 단위 1-line summary
  - compat/native의 success, reward, route text, 핵심 차이만 출력
- 파일:
  - `logs/anchor_policy_shadow_compare/*.jsonl`
  - 분석하기 쉬운 flat schema 유지

기본 원칙:

- 기본 로그는 task/run 단위로만 남긴다.
- per-round verbose dump는 기본 비활성화한다.
- 정말 필요한 경우에만 추가 verbose toggle을 넣는다.

## 7. 구현 순서 요약

권장 순서는 아래다.

1. Stage 2: shadow compare + 파일 로그
2. Stage 3: 2-anchor parent selection native화
3. Stage 4: 2-anchor meeting acceptance native화
4. Stage 5: 2-anchor final assembly native화
5. Stage 6: 공통 pairwise edge-repair fallback 추가
6. Stage 7: `auto`의 2-anchor default를 native로 전환
7. Stage 8: compat를 hidden fallback으로 축소
8. Stage 9: compat 및 legacy residue 제거

## 8. 후속 작업

현재 남은 큰 후속 작업은 아래다.

1. `use_rollout=true` 경로의 backward rollout semantics cleanup
2. planner 외부 유틸(`noise_schedule.py`, `env_manager.py`)의 terminology cleanup
