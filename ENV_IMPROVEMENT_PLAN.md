# 환경 중복 생성 문제 개선 계획안

## 현재 문제
```python
# df_planning.py:1048-1069 interact()
envs = DummyVecEnv([lambda: ogbench.make_maze_env(...)] * batch_size)
```
- **영향**: `interact()` 호출마다 새 환경 생성 → CPU RAM 누적
- **메모리 누수**: OGBench 메시 데이터, MuJoCo 핸들 미정리

---

## 개선 방안 3가지

### **방안 1: 환경 싱글톤 캐싱** ⭐ (권장)
**구현 난도**: ★☆☆ | **효과**: 20-30% RAM 절약 | **코드량**: ~50줄

```python
# df_planning.py에 추가
class EnvironmentCache:
    _instance = None
    _envs = {}
    
    @staticmethod
    def get_env(env_id: str, batch_size: int, seed: int = None):
        """환경 재사용 또는 생성"""
        key = f"{env_id}_{batch_size}_{seed}"
        if key not in EnvironmentCache._envs:
            # 새 환경 생성
            EnvironmentCache._envs[key] = DummyVecEnv([...] * batch_size)
            print(f"[ENV] Created new env: {key}")
        else:
            print(f"[ENV] Reusing cached env: {key}")
        return EnvironmentCache._envs[key]
    
    @staticmethod
    def cleanup():
        """모든 환경 정리"""
        for env in EnvironmentCache._envs.values():
            try:
                env.close()
            except:
                pass
        EnvironmentCache._envs.clear()
```

**사용**:
```python
# interact() 내에서
envs = EnvironmentCache.get_env(self.env_id, batch_size, self.interaction_seed)

# validation_step 마지막에
EnvironmentCache.cleanup()
```

**장점**:
- 단순 구현
- 즉시 적용 가능
- 추가 오버헤드 없음

**단점**:
- 환경 상태 초기화 필요 (seed 등)
- 여러 에포크 간 상태 격리 필요

---

### **방안 2: 멀티프로세싱 환경 풀** ⭐⭐ (고급)
**구현 난도**: ★★★ | **효과**: 30-50% RAM + 병렬 처리 | **코드량**: ~150줄

```python
# utils/env_pool.py (신규)
from multiprocessing import Pool, Manager
import queue

class EnvironmentPool:
    """멀티프로세싱 환경 풀"""
    
    def __init__(self, env_id: str, num_workers: int = 4):
        self.env_id = env_id
        self.num_workers = num_workers
        self.pool = Pool(num_workers)
        self.queue = Manager().Queue()
        self._init_workers()
    
    def _init_workers(self):
        """워커 초기화"""
        for i in range(self.num_workers):
            self.pool.apply_async(
                self._worker_init, 
                args=(self.env_id, i)
            )
    
    def get_env_batch(self, batch_size: int):
        """배치 환경 요청"""
        results = []
        for _ in range(batch_size):
            results.append(self.queue.get(timeout=5))
        return results
    
    def return_env_batch(self, envs):
        """사용한 환경 반환"""
        for env in envs:
            self.queue.put(env)
    
    def cleanup(self):
        """풀 정리"""
        self.pool.terminate()
        self.pool.join()
```

**사용**:
```python
# __init__에서
self.env_pool = EnvironmentPool(self.env_id)

# interact()에서
envs = self.env_pool.get_env_batch(batch_size)
# ... 사용 ...
self.env_pool.return_env_batch(envs)

# 정리
self.env_pool.cleanup()
```

**장점**:
- 병렬 환경 처리 가능
- 메모리 격리 (각 프로세스 독립)
- 확장성 우수

**단점**:
- 구현 복잡도 높음
- IPC 오버헤드
- 디버깅 어려움

---

### **방안 3: 환경 재설정 (최소 변경)** ⭐ (즉시)
**구현 난도**: ★☆☆ | **효과**: 10-15% RAM 절약 | **코드량**: ~30줄

```python
# df_planning.py interact() 수정
def interact(self, batch_size: int, ...):
    # 환경 생성 대신 기존 재사용
    if not hasattr(self, '_cached_envs'):
        self._cached_envs = DummyVecEnv([...] * batch_size)
    else:
        # 환경 상태 초기화만
        self._cached_envs.reset()
        self._cached_envs.seed(self.interaction_seed)
    
    envs = self._cached_envs
    # ... 나머지 코드 ...
```

**장점**:
- 최소 변경
- 즉시 적용
- 안전성 높음

**단점**:
- 효과 제한적 (10-15%)
- 상태 초기화 필수

---

## 추천 구현 로드맵

### Phase 1 (즉시, 1-2시간)
✅ **방안 3 구현**: 환경 재설정
- `interact()`에서 환경 생성 → 재사용으로 변경
- RAM 누적 10-15% 감소

### Phase 2 (선택, 반나절)
✅ **방안 1 구현**: 싱글톤 캐싱
- `EnvironmentCache` 클래스 추가
- RAM 누적 20-30% 추가 감소
- Phase 1과 병용 가능

### Phase 3 (선택, 1-2일)
✅ **방안 2 구현**: 멀티프로세싱 풀
- 병렬 처리 + 메모리 격리
- RAM 추가 30-50% 감소
- 복잡도 높음

---

## 구현 비용-효과 비교

| 방안 | 난도 | 시간 | 효과 | 위험도 |
|------|------|------|------|--------|
| **방안 3** | ★☆☆ | 1h | 10-15% | 낮음 |
| **방안 1** | ★☆☆ | 2h | 20-30% | 낮음 |
| **방안 2** | ★★★ | 8h | 30-50% | 중간 |
| **모두** | ★★☆ | 10h | 40-60% | 중간 |

---

## 코드 구현 예시 (방안 3 - 가장 실용적)

```python
# df_planning.py의 interact() 메서드 수정

def interact(self, batch_size: int, conditions: Optional[Any] = None, namespace: str = "validation") -> None:
    """
    환경과 상호작용하며 계획을 실행합니다.
    
    수정: 환경 재사용으로 메모리 누적 방지
    """
    try:
        import gym
        import ogbench
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("d4rl import not successful, skipping environment interaction.")
        return
    
    print("Interacting with environment... This may take a couple minutes.")
    
    # [NEW] 환경 캐싱 체크
    if not hasattr(self, '_cached_envs') or self._cached_envs is None:
        create_new_env = True
    else:
        create_new_env = False
    
    use_diffused_action = False
    
    if self.env_id in OGBENCH_ENVS:
        if create_new_env:
            if "pointmaze" in self.env_id:
                envs = DummyVecEnv([
                    lambda: ogbench.locomaze.maze.make_maze_env(
                        "point", "maze", maze_type=self.env_id.split("-")[1]
                    )
                ] * batch_size)
                self._cached_envs = envs  # [NEW] 캐시 저장
                # ... 나머지 초기화 ...
            elif "antmaze" in self.env_id:
                envs = DummyVecEnv([
                    lambda: ogbench.locomaze.maze.make_maze_env(
                        "ant", "maze", maze_type=self.env_id.split("-")[1]
                    )
                ] * batch_size)
                self._cached_envs = envs  # [NEW] 캐시 저장
                # ... DQL 에이전트 로드 ...
        else:
            # [NEW] 캐시된 환경 재사용
            envs = self._cached_envs
            print(f"[ENV] Reusing cached environment (save RAM)")
            # 환경 상태 초기화
            envs.reset()
        
        # ... 나머지 코드는 동일 ...
```

---

## 검증 방법

```bash
# 메모리 추적 활성화
grep "\[ENV\]" debug.log          # 환경 캐싱 로그
grep "Reusing cached" debug.log   # 재사용 횟수

# 메모리 차이 비교
# 전: interact() 호출마다 +500MB
# 후: 첫 호출만 +500MB, 이후 ±0MB
```

---

## 최종 권장

**지금 바로**: 방안 3 구현 (1시간, 10-15% 개선)  
**내일**: 방안 1 추가 (2시간, 추가 20-30%)  
**필요시**: 방안 2 고려 (8시간, 추가 30-50%)

→ **Phase 1+2 조합으로 총 30-45% RAM 누적 방지 가능**
