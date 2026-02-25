# 양방향 MCTS Rollout 연속성 분석 - 최종 보고서

## 📊 Executive Summary

4개 루프의 rollout 연속성 분석 결과, **심각한 구조적 문제 2가지**가 발견됨:

1. **Loop 2 (from_goal, depth 1): 조기 종료 (Early Termination)**
   - Trajectory length = 1 프레임 (expected ~200)
   - 최악의 end discontinuity: 43.54 units

2. **Loop 4 (from_goal, depth 2): 심각한 misalignment**
   - Start discontinuity = 31.98 units (CRITICAL)
   - End discontinuity = 32.73 units (CRITICAL)
   - 디노이징된 플랜이 실제 시뮬레이션과 완전히 맞지 않음

---

## 📈 Loop별 상세 비교 (df_planning.py:3539 vs 3596)

### LOOP 1 - from_start, Depth 1 (기준점)

**[3539 START]**
- Parent: [52.0, 0.0]
- Plan 1st: [50.855, 1.210]
- Start Δ: 1.666 ✓ PASS

**[3596 END]**
- Plan Last: [40.552, 15.738]
- Final State: [51.969, -0.246]
- End Δ: 19.643 ✗ ERROR
- Trajectory: 200 frames (정상)

**분석:** 시작은 양호하지만, 롤아웃 과정에서 점진적 divergence 발생. 200프레임 실행되었으므로 execution은 성공.

---

### LOOP 2 - from_goal, Depth 1 ⚠️ CRITICAL

**[3539 START]**
- Parent: [0.0, 36.0] (goal for backward tree)
- Plan 1st: [0.151, 34.804]
- Start Δ: 1.206 ✓ PASS (양호)

**[3596 END]** - **⚠️ 이상 발생**
- Plan Last: [15.926, 24.372] (100프레임 계획)
- Final State: [52.0, 0.0] ← **고정된 goal 위치!**
- End Δ: 43.536 ✗✗ CRITICAL
- **Trajectory: 1 프레임** (expected 200) ← **ANOMALY**

**핵심 발견:**
```
Parent → Traj(1st): 63.25 units
Traj Total Distance: 0.0 (움직이지 않음!)
Plan Total Distance: 57.31 units (큰 모션 계획했는데)
```

**근본 원인: Early Termination in _execute_plan_in_env**

df_planning.py:3319-3378의 실행 루프:
```python
while loop_cnt < self.open_loop_horizon:
    obs_numpy, reward, done, _ = envs.step(action_np)
    trajectory.append(bundle)
    loop_cnt += 1
    
    if done.any():
        break  # ← Loop 2에서 1번째 스텝 후 break!
```

**왜 done=True가 되는가:**
- Parent state 복원: agent가 goal [0.0, 36.0]에 배치
- 첫 번째 env.step() 실행
- 환경이 episode 종료 신호 반환 (goal 도달, reset flag, 또는 다른 termination condition)
- 루프가 1프레임만 수집 후 즉시 종료

**다른 루프와의 차이:**
- Loop 1 (from_start, depth 1): OK (200 frames)
- Loop 4 (from_goal, depth 2): OK (200 frames, 비록 심각한 discontinuity)
- **Loop 2만 문제** → from_goal + depth 1의 특정 조합 문제

---

### LOOP 3 - from_start, Depth 2

**[3539 START]**
- Parent: [51.969, -0.246]
- Plan 1st: [51.550, 3.656]
- Start Δ: 3.925 ⚠ WARNING

**[3596 END]**
- Plan Last: [27.572, 5.732]
- Final State: [52.092, -0.272]
- End Δ: 25.244 ✗✗ CRITICAL
- Trajectory: 200 frames (정상)

**분석:** 
- 시작부터 경고 수준의 discontinuity
- 심한 frame-by-frame divergence (mean=13.56, max=25.24)
- 하지만 execution은 완료됨 (200 frames)
- Loop 1보다 더 나쁜 결과

---

### LOOP 4 - from_goal, Depth 2 🚨 CRITICAL

**[3539 START]** - **이미 문제 시작**
- Parent: [52.0, 0.0]
- Plan 1st: [21.028, 7.959]
- Start Δ: **31.978 ✗✗ CRITICAL** ← 시작부터 큰 점프!

**[3596 END]**
- Plan Last: [28.637, 23.071]
- Final State: [52.098, 0.251]
- End Δ: **32.728 ✗✗ CRITICAL**
- Trajectory: 200 frames (실행됨)

**핵심 특성:**
```
Constant high divergence: mean=30.23, max=37.31, std=3.68
→ Trajectory는 항상 plan에서 ~30-37 units 떨어져 있음
→ 디노이징된 plan이 실제 simulation과 완전히 맞지 않음
```

**원인 분석:**
- Plan tokens [100-200]이 사용됨 (start_idx=100, end_idx=200)
- Plan first frame이 parent position에서 32 units 떨어짐
- 이는 frame_stack=10, segment 단위 디노이징에서 발생하는 문제로 추정

---

## 📊 종합 비교표

| Loop | Tree | Depth | Start Δ | End Δ | Traj Len | Plan Len | Div Mean | Div Max | Status |
|------|------|-------|---------|-------|----------|----------|----------|---------|--------|
| 1 | from_start | 1 | 1.67 | 19.64 | 200 | 100 | 21.53 | 34.49 | WARNING |
| 2 | from_goal | 1 | 1.21 | **43.54** | **1** | 100 | **62.45** | **62.45** | **ANOMALY** |
| 3 | from_start | 2 | 3.93 | 25.24 | 200 | 100 | 13.56 | 25.24 | ERROR |
| 4 | from_goal | 2 | **31.98** | **32.73** | 200 | 100 | **30.23** | **37.31** | **CRITICAL** |

---

## 🔍 패턴 분석

### FROM_START vs FROM_GOAL
```
FROM_START (Loops 1, 3):
- End Δ: 19.64, 25.24 (moderate)
- Traj Length: 200, 200 (normal)
- Pattern: 점진적 divergence, execution completes

FROM_GOAL (Loops 2, 4):
- End Δ: 43.54, 32.73 (catastrophic)
- Traj Length: 1, 200 (Loop 2 anomaly!)
- Pattern: Critical discontinuities, Loop 2 early termination
```

### DEPTH 1 vs DEPTH 2
```
DEPTH 1 (Loops 1, 2):
- Loop 1: OK but diverging (end=19.64)
- Loop 2: BROKEN with early termination (end=43.54)

DEPTH 2 (Loops 3, 4):
- Loop 3: Large divergence (end=25.24)
- Loop 4: Consistent misalignment (end=32.73, start=31.98)
```

### 연속성 심각도 분류
```
✓ PASS (< 2.0):    None
⚠ WARNING (2-5):   Loop 3 start (3.93)
✗ ERROR (5-20):    Loop 1 end (19.64)
✗✗ CRITICAL (>20): 
  - Loop 2 end (43.54) - Early termination
  - Loop 3 end (25.24) - Divergence
  - Loop 4 start (31.98) - Alignment error
  - Loop 4 end (32.73) - Persistent misalignment
```

---

## 💡 근본 원인 분석

### 원인 1: Loop 2 Early Termination

**Location:** df_planning.py:3377-3378
```python
if done.any():
    break
```

**발생 메커니즘:**
1. from_goal tree에서 parent는 goal position
2. `_set_sim_state(envs, parent_sim_state)` 호출 - agent를 goal에 배치
3. 첫 번째 env.step() 실행
4. 환경이 done=True 반환 (아마도 episode reset logic?)
5. 루프 즉시 종료 → 1 프레임만 수집

### 원인 2: Loop 4 Start Discontinuity (31.98 units)

**Location:** df_planning.py:3534-3546
```python
plan_slice_first_qpos = plan_slice[0, 0, :2]
first_frame_diff = np.linalg.norm(plan_slice_first_qpos - current_qpos)
```

**발생 메커니즘:**
1. Plan tokens [100-200] 범위 사용 (depth 2, second segment)
2. Plan slice first frame [21.028, 7.959]
3. Parent position [52.0, 0.0]
4. 거리: 31.98 units (엄청 큼!)

**왜 이런 일이?**
- Denoising이 plan slice [100-200]을 생성할 때, 
- 이전 slice [0-100]과의 연속성을 보장하지 못함
- Frame stack=10, segment-based denoising에서 경계 부분(frame 100)의 정합성 문제

### 원인 3: 공통적인 Frame-by-Frame Divergence

모든 루프에서 높은 frame-by-frame divergence:
- Loop 1: mean=21.53
- Loop 2: mean=62.45 (1프레임뿐)
- Loop 3: mean=13.56
- Loop 4: mean=30.23

**원인:**
- 디노이징된 plan이 예측하는 path와
- 실제 환경에서 action을 실행했을 때의 결과가 다름
- Model mismatch 또는 action computation 오류

---

## 🎯 핵심 발견 3가지

### 1️⃣ Loop 2의 조기 종료 (Early Termination)
- **증상:** trajectory_length = 1
- **원인:** done 플래그가 첫 스텝 후 True
- **영향:** 최악의 discontinuity (43.54)
- **심각도:** 🔴 CRITICAL

### 2️⃣ Loop 4의 심각한 시작점 오정렬
- **증상:** Start discontinuity = 31.98 units
- **원인:** Plan slice [100-200]이 parent state와 연속되지 않음
- **영향:** 전체 rollout이 plan에서 30+ units 떨어져 있음
- **심각도:** 🔴 CRITICAL

### 3️⃣ 모든 루프의 높은 Frame-by-Frame Divergence
- **증상:** Plan position vs Trajectory position의 큰 차이
- **원인:** Denoising plan과 실제 action execution의 mismatch
- **영향:** 예측된 motion을 따르지 못함
- **심각도:** 🟠 MAJOR

---

## 📋 권장 조치사항

### 즉시 조치 (High Priority)

1. **Loop 2 Early Termination 해결**
   - `_execute_plan_in_env`에서 done 플래그 처리 개선
   - from_goal 특수 케이스 처리
   - 또는 parent state restoration 메커니즘 검토

2. **Loop 4 Start Discontinuity 해결**
   - Plan slice 연결점(frame 100)에서의 연속성 보장
   - Segment 경계에서 강제 stitching 추가
   - 또는 frame_stack 크기 조정

### 중기 조치 (Medium Priority)

3. **Frame-by-Frame Divergence 감소**
   - Action computation 재검토 (_compute_action_from_plan)
   - PID controller 게인 조정
   - 또는 sub_goal interval 최적화

4. **Denoising Quality 개선**
   - 모델 재학습
   - Temperature 조정
   - Guidance strength 개선

### 장기 조치 (Low Priority)

5. **전체 양방향 MCTS 아키텍처 재검토**
   - from_start와 from_goal의 비대칭성 분석
   - Tree depth별 plan segment 전략 재평가

---

## 🔧 구체적 코드 개선안

### Fix 1: Loop 2 Early Termination 해결

```python
# df_planning.py:3280 근처에 추가
if "antmaze" in self.env_id:
    # from_goal rollout에서는 초기 done 신호를 무시
    # parent state 복원 후 한 번의 step 이후 done을 리셋
    self._set_sim_state(envs, parent_sim_state)
    obs_numpy, _, _, _ = envs.step(np.zeros(action_dim))  # dummy step
    done = np.zeros(batch_size, dtype=bool)  # Reset done flag
```

### Fix 2: Loop 4 Plan Stitching 개선

```python
# df_planning.py:3530 근처에 추가
if start_idx > 0:
    # Previous segment의 마지막 프레임과 현재 segment의 첫 프레임을 강제로 연결
    prev_segment_last = parent_node.plan_positions[-1]  # Get from parent
    offset = prev_segment_last - plan_slice[0, 0, :2]
    plan_slice[:, :, :2] = plan_slice[:, :, :2] + offset  # Stitch
```

### Fix 3: Frame-by-Frame Divergence 모니터링

```python
# df_planning.py:3596의 logging 강화
tracer.log(
    tag="rollout.divergence_analysis",
    data={
        "divergence_per_frame": divergence_analysis,
        "max_divergence_frame": max_divergence_idx,
        "divergence_trend": "increasing/decreasing/stable",
    },
    depth=1,
)
```

---

## 📌 결론

**상황 요약:**
- Loop 1: 안정적인 실행, 중간 정도의 divergence
- Loop 2: 구조적 버그 (early termination)
- Loop 3: 높은 divergence이지만 실행은 완료
- Loop 4: 극심한 misalignment (시작부터 문제)

**영향:**
- 트리 확장이 잘못된 value estimate를 기반으로 진행
- 최종 선택된 노드가 실제로는 좋지 않을 수 있음
- Planning quality 심각히 저하

**우선순위:**
1. 🔴 Loop 2 early termination 해결 (bug fix)
2. 🔴 Loop 4 plan stitching 개선 (architecture fix)
3. 🟠 전체 denoising-execution mismatch 개선 (quality enhancement)

---

## 📎 참고 정보

**분석 스크립트:** `/loop_comparison_analysis.py`
**로그 파일:** `/logs_memory_debug/run_20260225_030235.jsonl`

**주요 코드 위치:**
- `df_planning.py:3539` - START 로깅 (plan_slice_continuity)
- `df_planning.py:3596` - END 로깅 (final_state_continuity)
- `df_planning.py:3319-3378` - Execution loop (_execute_plan_in_env)
- `df_planning.py:3248-3388` - Plan execution function
- `df_planning.py:3390-3426` - Action computation function
