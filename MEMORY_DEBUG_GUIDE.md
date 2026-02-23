# MCTD 메모리 디버깅 및 최적화 가이드

## 개요

WSL에서 VRAM/RAM이 MAX를 치고 튕기는 문제의 원인을 찾고 해결하기 위한 디버깅 및 최적화 전략입니다.

## 주요 메모리 문제 원인 분석

### 1. **parallel_search_num에 따른 환경 중복 생성**
- **위치**: `df_planning.py:1048-1069` (interact 함수)
- **문제**: 
  ```python
  envs = DummyVecEnv([lambda: ogbench.make_maze_env(...)] * batch_size)
  ```
  이 코드는 `batch_size` (보통 1)만큼 환경을 생성합니다.
  그런데 이 `interact`가 **여러 번 호출**될 수 있고, 각 호출마다 독립적인 환경이 생성됩니다.

- **영향**: 
  - 각 환경은 MuJoCo physics 시뮬레이터를 로드하며 이는 매우 무겁습니다
  - OGBench 환경의 메시 데이터도 메모리에 로드됩니다

### 2. **parallel_plan에서 plan_hist 무제한 누적**
- **위치**: `df_planning.py:974-1003`
- **문제**:
  ```python
  # 모든 디노이징 스텝의 plan_hist가 메모리에 쌓임
  plan_hist = []
  for m in range(noise_level.shape[1] - 1):
      plan_hist.append(...)  # 계속 누적
  plan_hist = torch.stack(plan_hist)  # 모두 스택
  ```
  - `parallel_search_num=6` × `mctd_num_denoising_steps=100` = 600개 계획 히스토리
  - 각각 `(batch, horizon*frame_stack, obs_dim)` 크기

- **영향**: GPU 메모리 폭증

### 3. **MCTS 트리 노드 메모리 누적**
- **위치**: `df_planning.py:1701-1714` (MCTSTreeState 생성)
- **문제**:
  - 각 노드는 완전한 상태를 저장: `plan_history`, `current_levels`, `sim_state`
  - `mctd_max_search_num=70` × `parallel_search_num=6` = 420개 노드
  - 메모리 정리 없음

- **영향**: CPU RAM 누적

### 4. **DQL Agent 로드 (AntMaze의 경우)**
- **위치**: `df_planning.py:1070-1118`
- **문제**: 
  ```python
  agent = Agent(state_dim=state_dim * 2, ...)
  agent.load_model(...)  # 큰 신경망 로드
  ```
  - AntMaze 데이터셋에서만 DQL 에이전트 로드
  - 추가 GPU 메모리 소비

- **영향**: GPU VRAM 추가 점유

## 현재 적용된 디버깅 로그

### 설정 (configurations/algorithm/df_planning.yaml)

```yaml
DEBUG: True
debug_log_level: 2              # 0=off, 1=basic, 2=detailed, 3=verbose
debug_log_interval: 10          # Log every 10 MCTS iterations
debug_memory_profile: True      # Enable memory profiling
max_plan_hist_keep: 1           # Keep only last 1 plan history (최적화)
```

### 로깅 레벨 설명

- **Level 0**: 메모리 프로파일링 비활성화
- **Level 1**: 기본 메모리 할당/해제 로그만
- **Level 2**: 상세 메모리 추적 + 주기적 상태 출력
- **Level 3**: 모든 스텝에서 메모리 로깅 (매우 느림)

### 출력 예시

```
[MEM] interact_start: [GPU: alloc=2500.0MB, reserved=3000.0MB, cached=500.0MB] [CPU: RSS=8000.0MB, VMS=12000.0MB] [Tensors: 1500] ()
[DEBUG] Created 1 environment instances
[MEM] mcts_search_start_bidir_mcts_from_start: [GPU: alloc=2600.0MB, ...]
[DEBUG] MCTS search bidir_mcts_from_start: iteration 10/70
[MEM] mcts_search_iter_10_bidir_mcts_from_start: [GPU: alloc=2650.0MB, ...]
```

## 사용 방법

### 1. 메모리 문제 디버깅 시작

```bash
# 설정 파일에서 디버깅 활성화
cd /mnt/c/Users/USER/Desktop/test_ogbench/mctd_repo

# gen_fixed_jobs.sh 실행 (작은 job 하나만)
./gen_fixed_jobs.sh

# 메모리 추적과 함께 실행
python3 run_jobs.py 2>&1 | tee debug_memory_run.log
```

### 2. 메모리 로그 분석

```bash
# 메모리 할당량 추이 추출
grep "\[MEM\]" debug_memory_run.log | awk '{print $3,$5}' > memory_trend.txt

# 최대 메모리 지점 찾기
grep "\[MEM\]" debug_memory_run.log | sort -t= -k2 -rn | head -5

# 에러 및 크래시 로그
grep -E "Error|CUDA|OOM|memory" debug_memory_run.log
```

### 3. 메모리 최적화 튜닝

configuration을 수정하여 메모리 사용 줄이기:

```yaml
# configurations/algorithm/df_planning.yaml
parallel_search_num: 2          # 6 → 2 (병렬 인스턴스 감소)
mctd_max_search_num: 30         # 70 → 30 (탐색 깊이 감소)
mctd_num_denoising_steps: 50    # 100 → 50 (디노이징 스텝 감소)
max_plan_hist_keep: 1           # 1로 유지 (최적화)
debug_log_interval: 5           # 빠른 로깅
```

## 최적화 전략

### Phase 1: 즉시 적용 가능한 최적화

1. **plan_hist 메모리 제한** (이미 적용됨)
   - `max_plan_hist_keep=1`: 마지막 1개 히스토리만 유지
   - GPU 메모리 감소: ~30-50%

2. **디버깅 로그 추가** (이미 적용됨)
   - 메모리 누수 지점 식별
   - 성능 오버헤드: ~2-5%

### Phase 2: 알고리즘 조정

1. **parallel_search_num 감소** (가장 효과적)
   ```yaml
   parallel_search_num: 6 → 2 또는 3
   ```
   - 메모리 감소: 선형 (2배 감소 = 2배 메모리 절약)
   - 성능 영향: 검색 품질 약간 감소

2. **mctd_max_search_num 감소**
   ```yaml
   mctd_max_search_num: 70 → 30
   ```
   - 메모리 감소: 트리 노드 메모리 ~57% 감소
   - 성능 영향: 계획 품질 감소

3. **mctd_num_denoising_steps 감소**
   ```yaml
   mctd_num_denoising_steps: 100 → 50
   ```
   - 메모리 감소: 디노이징 히스토리 ~50% 감소
   - 성능 영향: 계획 정도 감소, 속도 2배 향상

### Phase 3: 구조적 개선

1. **환경 재사용** (코드 수정 필요)
   - 현재: 매 `interact()` 호출마다 새 환경 생성
   - 개선: 환경을 재사용하거나 싱글톤으로 관리
   - 예상 효과: CPU RAM 20-30% 감소

2. **DQL Agent 캐싱** (AntMaze만 해당)
   - 현재: 매번 디스크에서 로드
   - 개선: 클래스 변수로 한 번만 로드
   - 예상 효과: GPU 메모리 5-10% 절약, 속도 향상

3. **Tree Node 메모리 정리**
   - 사용하지 않는 노드 주기적 정리
   - `plan_history` 크기 제한
   - 예상 효과: CPU RAM 10-20% 감소

## 실행 순서 (권장)

```bash
# 1단계: 디버깅 로그 수집
python3 run_jobs.py 2>&1 | tee step1_baseline.log
# 메모리 최고 지점 확인

# 2단계: max_plan_hist_keep 최적화 (이미 적용)
# 결과 확인

# 3단계: parallel_search_num 조정
# configurations/algorithm/df_planning.yaml 수정:
# parallel_search_num: 6 → 2
python3 run_jobs.py 2>&1 | tee step2_ps_num_2.log

# 4단계: mctd_max_search_num 조정
# mctd_max_search_num: 70 → 30
python3 run_jobs.py 2>&1 | tee step3_max_search_30.log

# 5단계: 메모리 로그 비교
grep "\[MEM\].*interact_start\|interact_envs_created" step*.log
```

## 메모리 프로파일러 API

### 코드에서 직접 사용

```python
from utils.memory_profiler import get_profiler

profiler = get_profiler()
if profiler:
    # 메모리 스냅샷 저장
    profiler.snapshot("my_checkpoint", phase="processing")
    
    # 두 시점 간 메모리 변화 계산
    delta = profiler.delta("before", "after")
    print(f"GPU Memory Increase: {delta['gpu_alloc_delta_mb']} MB")
    
    # 리포트 출력
    print(profiler.report())
```

## 문제 해결 팁

### 증상: "CUDA out of memory"
1. `parallel_search_num` 감소 (6 → 2)
2. `mctd_num_denoising_steps` 감소 (100 → 50)
3. `mctd_max_search_num` 감소 (70 → 30)

### 증상: "WSL crashed / kernel panic"
1. CPU RAM 부족 - 환경 수 감소
2. GPU VRAM 부족 - 배치 크기 감소
3. 스왑 부족 - WSL 메모리 설정 확인: `~/.wslconfig`
   ```ini
   [wsl2]
   memory=12GB    # WSL에 할당할 메모리
   swap=8GB
   ```

### 증상: 느린 메모리 누수 (몇 시간 후 크래시)
1. 디버깅 로그에서 "tensor_delta" 증가 추이 확인
2. 노드 메모리 정리 코드 추가 필요
3. plan_history 주기적 정리 구현

## 참고: 디버깅 로그 형식

```
[MEM] tag: [GPU: alloc=X.XMB, reserved=X.XMB, cached=X.XMB] [CPU: RSS=X.XMB, VMS=X.XMB] [Tensors: N] (phase)
```

- **alloc**: GPU에 할당된 메모리
- **reserved**: CUDA가 예약한 총 메모리
- **cached**: 할당되지 않은 예약된 메모리 (재사용 가능)
- **RSS**: 프로세스의 실제 물리 메모리 사용
- **VMS**: 프로세스의 가상 메모리 사용
- **Tensors**: 현재 활성 PyTorch 텐서 개수

## 다음 단계

1. 현재 로그 수집 (`debug_log_level: 2` 상태에서 run_jobs.py 실행)
2. 메모리 트렌드 분석
3. 위의 최적화 단계 차례로 적용
4. 각 단계 후 메모리 개선 정량화

---

**작성 일시**: 2026-02-24
**대상 파일**: 
- `configurations/algorithm/df_planning.yaml`
- `algorithms/diffusion_forcing/df_planning.py`
- `utils/memory_profiler.py`
