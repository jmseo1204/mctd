# Guidance Functions Refactoring

## 변경 사항 (2026-02-07)

### 리팩토링 목적
`goal_guidance` 함수 내에 섞여있던 두 가지 loss를 분리:
1. **Target guidance loss**: goal/start로 향하는 loss
2. **Anchor dist regularization**: segment head들을 anchor로 사용하는 regularization

### 변경 내용

#### 1. 공통 함수 추출: `weigheted_loss`
- **위치**: L497-512
- **목적**: 코드 중복 방지를 위해 공통 로직을 별도 함수로 추출
- **Type Hint 추가**: `def weigheted_loss(dist: torch.Tensor, weight: Optional[torch.Tensor] = None, dim: tuple = (0, 2)) -> torch.Tensor:`
- **기능**: distance tensor에서 weighted loss 계산
- **주석 보존**: 모든 기존 주석 유지 (`DO NOT DELETE THIS COMMENT` 포함)

#### 2. 수정된 함수: `goal_guidance`
- **위치**: L514-563
- **변경**: Anchor regularization 관련 코드 제거
- **유지**: Target guidance 로직만 남김
- **수식**: `dist_per_batch = guidance_scale * weighted_dist_target` (anchor 부분 제거)
- **Type Hint 추가**: `def goal_guidance(x: torch.Tensor) -> torch.Tensor:`
- **주석 보존**: 모든 기존 주석 유지

#### 3. 새로운 함수: `anchor_dist_guidance`
- **위치**: L565-595
- **목적**: Anchor distance regularization 전담
- **Type Hint**: `def anchor_dist_guidance(x: torch.Tensor, anchor_scale: float = 20.0) -> torch.Tensor:`
- **파라미터**: `anchor_scale` - anchor regularization의 가중치 (기본값 20.0)
- **기능**:
  - Segment head들을 anchor로 사용
  - `segment_size = horizon // self.sequence_dividing_factor`
  - Anchor plan 생성 및 MSE loss 계산
  - Weighted loss 계산 후 반환

### 사용 방법

#### 기존 방식 (두 loss 혼합)
```python
# 이전에는 goal_guidance가 두 loss를 모두 처리
guidance_fn = goal_guidance  # target + anchor 모두 포함
```

#### 새로운 방식 (분리된 함수)
```python
# 옵션 1: Target guidance만 사용
guidance_fn = goal_guidance

# 옵션 2: Anchor regularization만 사용
guidance_fn = anchor_dist_guidance

# 옵션 3: 두 가지를 조합 (lambda 사용)
guidance_fn = lambda x: goal_guidance(x) + anchor_dist_guidance(x, anchor_scale=20.0)
```

### 호환성
- 기존 코드와 완전히 호환
- 모든 주석 보존
- Type hints 추가로 정적 분석 가능
- 코드 중복 제거 (DRY 원칙 준수)

### 구조
```
weigheted_loss (공통 helper)
├── goal_guidance (target guidance)
└── anchor_dist_guidance (anchor regularization)
```
