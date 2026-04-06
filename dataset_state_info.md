# OGBench Dataset State Structure Reference

소스 코드 기준:
- pointmaze/antmaze: `ogbench/locomaze/point.py`, `ogbench/locomaze/ant.py`, `ogbench/locomaze/maze.py`
- cube: `ogbench/manipspace/envs/cube_env.py`, `ogbench/manipspace/envs/manipspace_env.py`
- 데이터 로딩: `ogbench/utils.py`, `ogbench/relabel_utils.py`

---

## 0. 데이터셋 객체 공통 구조

### NPZ 파일 원시 필드

`.npz` 파일을 `np.load()`로 열면 다음 필드가 들어 있음 (`utils.py:30-48`):

| 필드 | 타입 | 형태 | 설명 |
|------|------|------|------|
| `observations` | float32 | `(T, obs_dim)` | 각 타임스텝의 observation |
| `actions` | float32 | `(T, act_dim)` | 각 타임스텝의 action |
| `terminals` | bool | `(T,)` | 에피소드 마지막 스텝 여부 |
| `qpos` | float32 | `(T, qpos_dim)` | raw MuJoCo qpos (항상 포함) |
| `qvel` | float32 | `(T, qvel_dim)` | raw MuJoCo qvel (항상 포함) |

> `qpos`, `qvel`은 `load_dataset(add_info=True)` 또는 singletask 변형에서만 반환됨.
> 기본 `load_dataset(add_info=False)` 시 파일에는 존재하나 반환 dict에서 제거됨.

### 에피소드 메모리 레이아웃

파일 내 데이터는 에피소드들이 평탄하게 이어붙여진 구조임:

```
                |<---  에피소드 1  --->|  |<---  에피소드 2  --->|  ...
observations: [ s0,  s1,  s2,  s3,  s4,  s0,  s1,  s2,  s3,  s4,  ...]
actions:      [ a0,  a1,  a2,  a3,  a4,  a0,  a1,  a2,  a3,  a4,  ...]
terminals:    [  F,   F,   F,   F,   T,   F,   F,   F,   F,   T,  ...]
```

- 에피소드 길이 = L+1 스텝 저장 (s₀~sₗ). 마지막 sₗ이 `terminal=True`.
- `(sₗ, aₗ)` 쌍은 유효한 transition이 아님 — sₗ₊₁이 없기 때문.
- 실제 유효 transition 수 = `T - num_episodes`.

### `load_dataset` 반환 형태

**`compact_dataset=False` (기본)**:
마지막 스텝(terminal=True)을 제거하고 `next_observations`를 생성:

```
observations:      [s0, s1, s2, s3, s0, s1, s2, s3, ...]  (각 에피소드 마지막 제거)
actions:           [a0, a1, a2, a3, a0, a1, a2, a3, ...]
next_observations: [s1, s2, s3, s4, s1, s2, s3, s4, ...]
terminals:         [ 0,  0,  0,  1,  0,  0,  0,  1, ...]  (terminal=1이 마지막 valid transition)
```

**`compact_dataset=True`**:
`next_observations` 없이 `valids` 마스크 사용:

```
observations: [s0, s1, s2, s3, s4, s0, s1, s2, s3, s4, ...]  (원본 유지)
valids:       [ 1,  1,  1,  1,  0,  1,  1,  1,  1,  0, ...]  (0이면 next=obs[t+1]이 무효)
terminals:    [ 0,  0,  0,  1,  1,  0,  0,  0,  1,  1, ...]  (마지막 2스텝이 1)
```

> compact 모드에서 `next_observations[t] = observations[t+1]`로 직접 indexing 가능.
> `valids[t]=0`인 스텝은 샘플링에서 제외해야 함.

### Reward 제공 방식

**GCRL (goal-conditioned) 변형** — `observations`에 reward 없음:
- 기본 데이터셋 (`navigate`, `stitch`, `play`, `noisy` 등)은 reward 필드가 없음.
- env.step()에서 goal 도달 시 reward=1, 미도달 시 0 반환 (온라인 평가 시).
- 오프라인 학습에서는 알고리즘이 `info['goal']`과 현재 state를 비교해 스스로 relabeling.

**Singletask 변형** — `load_dataset`이 `rewards`, `masks` 필드를 추가:
- 내부적으로 `relabel_utils.relabel_dataset()` 호출 (`utils.py:216-217`).
- 계산 방법 (`relabel_utils.py:30, 84`):
  - **maze 계열**: `rewards = (dist_to_goal <= tol).astype(float) - 1.0` → 성공=0, 실패=-1
  - **cube 계열**: `rewards = num_succeeded_cubes - num_total_cubes` → 모두 성공=0, 1개 미달=-1, ...
  - `masks = 1.0 - success` (성공 시 0 = episode 종료 마스크)

### 데이터셋 규모 요약

| 데이터셋 | 에피소드 수 | 스텝/에피소드 | 총 transitions |
|---------|-----------|------------|----------------|
| antmaze-giant-navigate | 500 | 2001 | ~1,000,500 |
| antmaze-giant-stitch | 5000 | 201 | ~1,005,000 |
| antmaze-medium-explore | 10000 | 501 | ~5,010,000 |
| cube-single-play | 1000 | 1001 | ~1,001,000 |
| cube-double-play | 1000 | 1001 | ~1,001,000 |
| cube-triple-play | 3000 | 1001 | ~3,003,000 |

Train/val 분리: 파일명에 `-val` suffix가 있는 파일이 validation set (약 10% 규모).

---

---

## 1. PointMaze

### 해당 데이터셋
| 환경 크기 | 데이터셋 타입 | 데이터셋 이름 예시 |
|-----------|-------------|-----------------|
| medium, large, giant, teleport | navigate | `pointmaze-medium-navigate-v0` |
| medium, large, giant, teleport | stitch | `pointmaze-large-stitch-v0` |

### State 구조: **2차원**

`get_ob()` → `self.data.qpos.flat.copy()` (`point.py:98`)

| 인덱스 | 의미 |
|--------|------|
| 0 | agent x 좌표 (maze world 단위) |
| 1 | agent y 좌표 (maze world 단위) |

### 좌표계 및 스케일
- **정규화 없음.** 값이 raw world 좌표 그대로 저장됨.
- Maze unit = 4.0, offset = (-4, -4). 예를 들어 medium maze의 (row=1, col=1) 셀 중심은 xy ≈ (0, 0).
- 셀 (i, j)의 world 좌표: `x = j * 4 - 4`, `y = i * 4 - 4` (`maze.py`의 `ij_to_xy`)

### Goal
- `info['goal']` = `np.array([goal_x, goal_y])` (2차원, 같은 좌표계)

### qpos/qvel 구조 (add_info=True 시)

| 필드 | 형태 | 내용 |
|------|------|------|
| `qpos` | `(T, 2)` | [x, y] — observations[0:2]와 동일 |
| `qvel` | `(T, 2)` | [vx, vy] — 매 스텝 0으로 리셋되므로 항상 ≈ 0 |

### 실제 데이터 헤드 예시

> pointmaze 데이터셋은 별도 다운로드되지 않아 antmaze 예시로 대신하며,
> state 구조는 동일하게 [x, y] 2차원임.

### Action 구조: **2차원**, `action_space = Box([-1, -1], [1, 1])`

| 인덱스 | 의미 |
|--------|------|
| 0 | x 방향 이동 명령 |
| 1 | y 방향 이동 명령 |

- 내부적으로 `action * 0.2`를 현재 qpos에 더하여 위치를 직접 갱신 (`point.py:68-70`)
- 즉 실제 위치 변화량 = `action * 0.2` → 한 스텝당 최대 ±0.2 이동
- 속도(qvel)는 매 스텝마다 0으로 리셋됨 — 관성 없음, 완전한 위치 제어
- 물리 시뮬레이션(`mj_step`)은 이후에 실행되지만 벽 충돌 처리 용도

---

## 2. AntMaze

### 해당 데이터셋
| 환경 크기 | 데이터셋 타입 | 비고 |
|-----------|-------------|------|
| medium, large, giant, teleport | navigate | - |
| medium, large, giant, teleport | stitch | - |
| medium, large, teleport | explore | giant-explore 없음 |

### State 구조: **29차원**

`get_ob()` → `np.concatenate([qpos, qvel])` (`ant.py:98-101`)

#### qpos (인덱스 0–14, 15차원)

| 인덱스 | 의미 |
|--------|------|
| 0 | torso **x** 좌표 |
| 1 | torso **y** 좌표 |
| 2 | torso **z** (높이, 정상 보행 시 ≈ 0.55) |
| 3 | torso 방향 quaternion **w** |
| 4 | torso 방향 quaternion **x** |
| 5 | torso 방향 quaternion **y** |
| 6 | torso 방향 quaternion **z** |
| 7 | hip_1 관절 각도 (앞왼쪽 다리, rad) |
| 8 | ankle_1 관절 각도 (앞왼쪽 다리, rad) |
| 9 | hip_2 관절 각도 (앞오른쪽 다리, rad) |
| 10 | ankle_2 관절 각도 (앞오른쪽 다리, rad) |
| 11 | hip_3 관절 각도 (뒷왼쪽 다리, rad) |
| 12 | ankle_3 관절 각도 (뒷왼쪽 다리, rad) |
| 13 | hip_4 관절 각도 (뒷오른쪽 다리, rad) |
| 14 | ankle_4 관절 각도 (뒷오른쪽 다리, rad) |

#### qvel (인덱스 15–28, 14차원)

| 인덱스 | 의미 |
|--------|------|
| 15 | torso x 방향 선속도 |
| 16 | torso y 방향 선속도 |
| 17 | torso z 방향 선속도 |
| 18 | torso 각속도 x |
| 19 | torso 각속도 y |
| 20 | torso 각속도 z |
| 21 | hip_1 각속도 |
| 22 | ankle_1 각속도 |
| 23 | hip_2 각속도 |
| 24 | ankle_2 각속도 |
| 25 | hip_3 각속도 |
| 26 | ankle_3 각속도 |
| 27 | hip_4 각속도 |
| 28 | ankle_4 각속도 |

### 좌표계 및 스케일
- **정규화 없음.** Raw MuJoCo 값 그대로.
- Agent x/y 위치: `state[:2]`
- Agent x/y = `get_xy()` = `qpos[:2]` (`ant.py:116`)
- 좌표계는 pointmaze와 동일: `x = j * 4 - 4`, `y = i * 4 - 4`

### Goal
- `info['goal']` = `np.array([goal_x, goal_y])` (2차원, xy 좌표만)

### Action 구조: **8차원**, `action_space = Box([-1]*8, [1]*8)`

| 인덱스 | 의미 | 대응 관절 |
|--------|------|----------|
| 0 | hip_4 (뒷오른쪽 hip) 토크 | gear=30 |
| 1 | ankle_4 (뒷오른쪽 ankle) 토크 | gear=30 |
| 2 | hip_1 (앞왼쪽 hip) 토크 | gear=30 |
| 3 | ankle_1 (앞왼쪽 ankle) 토크 | gear=30 |
| 4 | hip_2 (앞오른쪽 hip) 토크 | gear=30 |
| 5 | ankle_2 (앞오른쪽 ankle) 토크 | gear=30 |
| 6 | hip_3 (뒷왼쪽 hip) 토크 | gear=30 |
| 7 | ankle_3 (뒷왼쪽 ankle) 토크 | gear=30 |

- action 값은 [-1, 1] 범위의 normalized torque (`ctrlrange="-1.0 1.0"`, `ant.xml:87-94`)
- `do_simulation(action, frame_skip=5)` 으로 MuJoCo ctrl에 직접 전달 — 별도 스케일링 없음
- `gear=30`이므로 실제 적용 토크 = `action * 30`
- **액추에이터 순서 주의**: XML의 actuator 정의 순서가 qpos의 관절 순서(hip_1, ankle_1, ...)와 다름
  - qpos/qvel 인덱스 7 = hip_1이지만, action 인덱스 2 = hip_1

### qpos/qvel 구조 (add_info=True 시)

| 필드 | 형태 | 내용 |
|------|------|------|
| `qpos` | `(T, 15)` | [x, y, z, qw, qx, qy, qz, hip1, ank1, hip2, ank2, hip3, ank3, hip4, ank4] |
| `qvel` | `(T, 14)` | [vx, vy, vz, wx, wy, wz, dhip1, dank1, dhip2, dank2, dhip3, dank3, dhip4, dank4] |

`qpos == observations[0:15]`, `qvel == observations[15:29]` (값 동일)

### 실제 데이터 헤드 예시 (`antmaze-giant-stitch-v0.npz`)

```
파일 필드:
  observations : (1005000, 29),  float32
  actions      : (1005000,  8),  float32
  terminals    : (1005000,),     bool
  qpos         : (1005000, 15),  float32
  qvel         : (1005000, 14),  float32

총 5000 에피소드 × 201 스텝/에피소드

observations[0]:
  [0]    x     = 44.0000   # 에피소드 시작 위치
  [1]    y     = 36.0000
  [2]    z     = 0.7202    # 정상 보행 높이 ≈ 0.55~0.75
  [3:7]  quat  = [1.0783, 0.0749, 0.0161, -0.0923]  # wxyz
  [7:15] joints= [-0.0044, 0.0653, 0.0082, 0.0571, -0.0712, 0.0104, 0.0321, -0.0512]
  [15:18] vel  = [0.0865, 0.1399, -0.0007]   # 선속도 xyz
  [18:21] angv = [0.0348, -0.0103, -0.1408]  # 각속도 xyz
  [21:29] jvel = [-0.0712, -0.0058, 0.0084, 0.1825, -0.0587, 0.0799, 0.1309, 0.1545]

actions[0]:
  [-0.5465, -0.5273, -0.9979, -0.5226, 0.8825, 0.4325, 0.4843, -1.0000]
  # 8개 모터 torque, 범위 [-1, 1]

terminals[0:10]:
  [F, F, F, F, F, F, F, F, F, F]  # 201번째 인덱스에서 True
```

### 주의사항
- **qpos[3:7]이 quaternion**: 단위 quaternion (norm=1) 보장. 방향 계산 시 `[3]`이 w 성분.
- Hip 관절 범위: [-30°, 30°] → [-0.524, 0.524] rad
- Ankle 관절 범위: [30°, 70°] 또는 [-70°, -30°] (다리마다 상이)

---

## 3. Cube (Manipulation)

### 해당 데이터셋

| 환경 | 데이터셋 타입 | 데이터셋 이름 |
|------|-------------|-------------|
| single | play | `cube-single-play-v0` |
| single | noisy | `cube-single-noisy-v0` |
| double | play | `cube-double-play-v0` |
| double | noisy | `cube-double-noisy-v0` |
| triple | play | `cube-triple-play-v0` |
| triple | noisy | `cube-triple-noisy-v0` |
| quadruple | play | `cube-quadruple-play-v0` |
| quadruple | noisy | `cube-quadruple-noisy-v0` |

### State 구조

`compute_observation()` (`cube_env.py:766-794`)

총 차원 수 = **19 + 9 × N** (N = 큐브 수)
- cube-single: **28차원**
- cube-double: **37차원**
- cube-triple: **46차원**
- cube-quadruple: **55차원**

#### 로봇 팔 공통 부분 (인덱스 0–18, 19차원)

| 인덱스 | 차원 | 의미 | 스케일/정규화 |
|--------|------|------|-------------|
| 0–5 | 6 | UR5e 관절 각도 (joint_pos, rad) | raw |
| 6–11 | 6 | UR5e 관절 속도 (joint_vel, rad/s) | raw |
| 12–14 | 3 | end-effector xyz 위치 | `(xyz - [0.425, 0.0, 0.0]) * 10` |
| 15 | 1 | cos(end-effector yaw) | [-1, 1] |
| 16 | 1 | sin(end-effector yaw) | [-1, 1] |
| 17 | 1 | gripper 열림 정도 | `opening * 3`, 원래 범위 [0, 1] → 저장값 [0, 3] |
| 18 | 1 | gripper 접촉 여부 | [0, 1] (clip된 contact force) |

#### 큐브 i번째 (인덱스 19+9i ~ 27+9i, 9차원씩)

| 오프셋 | 차원 | 의미 | 스케일/정규화 |
|--------|------|------|-------------|
| +0–2 | 3 | 큐브 xyz 위치 | `(xyz - [0.425, 0.0, 0.0]) * 10` |
| +3–6 | 4 | 큐브 방향 quaternion (wxyz) | raw (단위 quaternion) |
| +7 | 1 | cos(큐브 yaw) | [-1, 1] |
| +8 | 1 | sin(큐브 yaw) | [-1, 1] |

큐브 순서: 0 = 첫 번째(빨강), 1 = 두 번째(파랑), 2 = 세 번째(주황), 3 = 네 번째(초록)

### 정규화 역변환

```python
# end-effector 실제 위치 복원
xyz_center = np.array([0.425, 0.0, 0.0])
effector_xyz = state[12:15] / 10.0 + xyz_center

# cube i의 실제 위치 복원 (i=0, 1, 2, 3)
cube_xyz = state[19 + 9*i : 22 + 9*i] / 10.0 + xyz_center

# cube i의 quaternion (wxyz, 정규화 불필요)
cube_quat_wxyz = state[22 + 9*i : 26 + 9*i]

# cube i의 yaw 복원
cube_yaw = np.arctan2(state[27 + 9*i], state[26 + 9*i])  # arctan2(sin, cos)

# end-effector yaw 복원
effector_yaw = np.arctan2(state[16], state[15])  # arctan2(sin, cos)

# gripper 열림 정도 복원 (0=닫힘, 1=최대)
gripper_opening = state[17] / 3.0
```

### 작업 공간 (workspace)
End-effector와 큐브의 실제 좌표 범위:
- x ≈ [0.30, 0.55] (중심 0.425)
- y ≈ [-0.25, 0.25] (중심 0.0)
- z: 큐브는 테이블 위 ≈ 0.02, end-effector는 ≈ 0.02 ~ 0.4

### Goal
- `info['goal']` = 목표 상태의 동일한 구조의 state 벡터 (차원 동일)

### Action 구조: **5차원**, `action_space = Box([-1]*5, [1]*5)`

환경은 [-1, 1]로 정규화된 action을 받아 내부에서 unnormalize (`manipspace_env.py:158-160`)

| 인덱스 | 의미 | 실제 범위 (unnormalized) |
|--------|------|------------------------|
| 0 | end-effector **Δx** (상대 이동) | [-0.05, 0.05] m |
| 1 | end-effector **Δy** (상대 이동) | [-0.05, 0.05] m |
| 2 | end-effector **Δz** (상대 이동) | [-0.05, 0.05] m |
| 3 | end-effector **Δyaw** (상대 회전) | [-0.3, 0.3] rad |
| 4 | gripper **Δopening** (상대 변화) | [-1.0, 1.0] |

- action은 delta (상대값): 현재 end-effector 위치/방향에 더해지는 값
- 역정규화: `unnormalized = 0.5 * (action + 1) * (high - low) + low`
  - `action_range = [0.05, 0.05, 0.05, 0.3, 1.0]`, `low = -action_range`, `high = +action_range`
- workspace 범위로 clip됨: x∈[0.25, 0.60], y∈[-0.35, 0.35], z∈[0.02, 0.35] (`manipspace_env.py:67`)
- gripper opening은 [0, 1]로 clip: 0=완전히 닫힘, 1=완전히 열림
- **IK 기반 제어**: delta action → target effector pose → IK → joint position target → PD 제어 (low-level)

### qpos/qvel 구조 (add_info=True 시)

cube-single 기준 (큐브 N개이면 큐브 파트가 7N 차원으로 확장):

| 필드 | 형태 | 내용 |
|------|------|------|
| `qpos` | `(T, 14+7N)` | arm(6) + gripper(8) + cube_i_xyz(3)+cube_i_quat_wxyz(4) × N |
| `qvel` | `(T, 14+6N)` | arm_vel(6) + gripper_vel(8) + cube_i_linvel(3)+cube_i_angvel(3) × N |

- `qpos[:,14+7i : 14+7i+3]` = 큐브 i의 실제 xyz 위치 (정규화 없음, 단위 m)
- `qpos[:,14+7i+3 : 14+7i+7]` = 큐브 i의 quaternion wxyz
- `relabel_utils.py`에서 reward 계산 시 `qpos_obj_start_idx=14`를 기준으로 큐브 위치를 직접 참조

| 환경 | qpos 차원 | qvel 차원 |
|------|----------|----------|
| cube-single | 14+7×1 = **21** | 14+6×1 = **20** |
| cube-double | 14+7×2 = **28** | 14+6×2 = **26** |
| cube-triple | 14+7×3 = **35** | 14+6×3 = **32** |
| cube-quadruple | 14+7×4 = **42** | 14+6×4 = **38** |

### 실제 데이터 헤드 예시 (`cube-single-play-v0.npz`)

```
파일 필드:
  observations : (1001000, 28),  float32
  actions      : (1001000,  5),  float32
  terminals    : (1001000,),     bool
  qpos         : (1001000, 21),  float32
  qvel         : (1001000, 20),  float32

총 1000 에피소드 × 1001 스텝/에피소드

observations[0]:  (xyz_center=[0.425, 0.0, 0.0], scaler=10)
  [0:6]   joint_pos  = [-1.722, -1.726,  1.789, -1.634, -1.571, -2.219]  (UR5e 관절각, rad)
  [6:12]  joint_vel  = [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000]  (리셋 직후 = 0)
  [12:15] eff_pos_n  = [ 0.160,  0.683,  3.023]  →  실제 xyz = [0.441, 0.068, 0.302] m
  [15]    cos(yaw)   = -0.476
  [16]    sin(yaw)   =  0.879  →  yaw = 2.067 rad
  [17]    gripper_n  =  0.000  →  실제 열림 = 0.000 (완전히 닫힘)
  [18]    contact    =  0.000
  [19:22] cube0_pos_n= [ 1.145,  2.077,  0.200]  →  실제 xyz = [0.540, 0.208, 0.020] m
  [22:26] cube0_quat = [-0.997,  0.000,  0.000,  0.080]  (wxyz)
  [26]    cos(yaw)   =  0.987
  [27]    sin(yaw)   = -0.159  →  yaw = -0.159 rad

qpos[0]:
  [0:6]   arm joints = [-1.722, -1.726,  1.789, -1.634, -1.571, -2.219]
  [6:14]  gripper    = [ 0.000,  0.000, ...,  0.000]  (8개, 닫힌 상태)
  [14:17] cube0 xyz  = [ 0.540,  0.208,  0.020]  ← 정규화 없는 실제 m 단위
  [17:21] cube0 quat = [-0.997,  0.000,  0.000,  0.080]

actions[0]:
  [ 0.125,  0.004, -0.122, -0.075, -0.053]  (정규화된 [-1,1])
  unnormalized: [Δx=0.006m, Δy=0.000m, Δz=-0.006m, Δyaw=-0.023rad, Δgrip=-0.053]
```

### 주의사항
- **큐브 quaternion이 wxyz 순서**: MuJoCo 기본 순서가 wxyz. scipy 등의 라이브러리는 xyzw를 사용하므로 주의.
- Cube yaw만 표현하는 이유: 큐브가 테이블 위에서 z축 회전만 자유롭게 움직이기 때문에 yaw가 핵심 방향 정보.
- `play` 데이터셋: oracle policy로 한 큐브씩 순서대로 목표 위치로 이동.
- `noisy` 데이터셋: play와 동일하되 10% 확률로 random action 삽입.
