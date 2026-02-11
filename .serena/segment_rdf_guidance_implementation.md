# Segment RDF Guidance Implementation

## 개요
`df_planning.py`에 새로운 guidance function `segment_rdf_guidance`를 구현했습니다.

## 위치
- 파일: `/mnt/c/Users/USER/Desktop/test_ogbench/mctd_repo/algorithms/diffusion_forcing/df_planning.py`
- 라인: ~593-697 (particle_guidance 다음, guidance_fn 할당 이전)

## 기능 설명

### 입력
- `x`: shape `[t b (fs c)]`의 텐서

### 동작 방식
1. **Segment 분할**: `segment_size = horizon // self.sequence_dividing_factor`
2. **State 순회**: 각 segment i (i > 0)의 state j에 대해:
   - **Segment 판별**: `segment_i = (j - self.frame_stack) // segment_size`
   - **Candidate 탐색**: segment (i-1)에서 `|k - j| >= 3`인 모든 k를 후보로 선정
   - **j_pair 선택**: RDF kernel 값이 최대인 k를 j_pair로 선택
   - **Loss 계산**: `rdf_value = exp(-dist_sq / h)` (h=1.0)
3. **Loss 집계**: 모든 state의 RDF 값을 합산 후 평균
4. **출력**: `-mean_loss` (gradient descent가 repulsion을 최소화)

### 특징
- Segment 0의 states는 loss에 기여하지 않음 (이전 segment 없음)
- 절대 인덱스 기준으로 3 이상 차이나는 states만 고려
- RDF kernel 값이 가장 큰 pairing을 선택 (가장 유사한 state)
- Batch별로 독립적으로 처리

## Type Hints
함수 signature에 Python type hints 추가:
```python
def segment_rdf_guidance(x: torch.Tensor) -> torch.Tensor:
```

## 사용 방법
이 함수는 `goal_guidance` 또는 `particle_guidance`와 동일한 포맷으로 구현되어 있어,
`parallel_plan` 함수 내에서 `guidance_fn = segment_rdf_guidance`로 설정하여 사용 가능합니다.

## 디버그 출력
함수 실행 시 다음 정보를 출력:
- 입력 shape
- horizon, segment_size, sequence_dividing_factor
- 총 pair 수, 평균 loss

## 호환성
- 기존 코드와 완전히 독립적으로 구현
- 기존 주석 및 코드 수정 없음
- `goal_guidance`, `particle_guidance`와 동일한 인터페이스
