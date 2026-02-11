# Padding Consistency Fix

## 문제점
`is_unknown_final_token` 기능 구현 중 발견된 중요한 버그:

### 원래 동작
- **Bidirectional 모드**: `pad_tokens = n_tokens - plan_tokens - 2`
  - Structure: `[init(1), chunk(plan_tokens), final(1), pad(48)]`
  - Total: 100 tokens

- **Unidirectional 모드 (OLD)**: `pad_tokens = n_tokens - plan_tokens - 1`
  - Structure: `[init(1), chunk(plan_tokens), pad(49)]`
  - Total: 100 tokens

### 발생 문제
`is_unknown_final_token=True`로 설정하면:
- `bidirectional=False`로 `_construct_sequence` 호출
- Padding이 **1개 더 많아짐** (48 → 49)
- MCTD tree operations에서 **shape mismatch** 가능

## 해결책

### `_construct_sequence`에 `reserve_final_token_space` 파라미터 추가

```python
def _construct_sequence(
    self, 
    start, goal, chunk, plan_tokens, batch_size, from_start, 
    bidirectional,
    reserve_final_token_space: bool = None  # 새 파라미터
) -> tuple:
    if reserve_final_token_space is None:
        reserve_final_token_space = bidirectional
    
    # ... bidirectional mode code ...
    
    else:
        # Unidirectional mode
        if reserve_final_token_space:
            # Reserve space for final_token even though not using it
            pad_tokens = max(0, self.n_tokens - plan_tokens - 2)
        else:
            # Original unidirectional calculation
            pad_tokens = 0 if self.causal else self.n_tokens - plan_tokens - 1
```

### 호출 시 전달
```python
use_bidirectional_sequence = self.bidirectional_search and not is_unknown_final_token
plan, pad_tokens = self._construct_sequence(
    start, goal, chunk, plan_tokens, batch_size, from_start, 
    use_bidirectional_sequence,
    reserve_final_token_space=self.bidirectional_search  # 항상 일관된 padding
)
```

## 결과

### 수정 후 동작
- **is_unknown_final_token=False**: `[init(1), chunk(50), final(1), pad(48)]` 
- **is_unknown_final_token=True**: `[init(1), chunk(50), pad(48)]` ← Same padding!

### 효과
✅ Padding이 일관되게 유지됨  
✅ Shape mismatch 방지  
✅ MCTD tree operations 안정성 보장  

## 테스트
`test_padding_consistency.py`로 검증 완료:
- Bidirectional pad_tokens: 48
- Unidirectional pad_tokens (with reserve): 48
- Difference: 0 ✓
