# Critical Review: Padding Logic Correction

## 원래 구현 (잘못됨)
```python
reserve_final_token_space=self.bidirectional_search
```

**문제점**:
- `bidirectional_search=True, is_unknown_final_token=True`일 때
- Final_token이 실제로 **없는데** 공간을 예약
- 논리적으로 일관되지 않음

## 수정된 구현 (올바름)
```python
reserve_final_token_space=use_bidirectional_sequence
```

**근거**:
- `use_bidirectional_sequence = bidirectional_search and not is_unknown_final_token`
- Final_token이 **실제로 있을 때만** 공간 예약
- 논리적으로 일관됨

## Padding 비교표

| Scenario | is_unknown_final_token | use_bid_seq | reserve_space | Padding | Total |
|----------|----------------------|-------------|---------------|---------|-------|
| 1. Bid + has final | False | True | True | 48 | 100 |
| 2. Bid + NO final | True | False | False | 49 | 100 |
| 3. Unidirectional | N/A | False | False | 49 | 100 |

## 왜 Padding 차이가 문제 없는가?

### 핵심 이유
1. **`plan_hist` shape는 `horizon`으로 결정됨**
   - `plan_hist[:, frame_stack : frame_stack + horizon]`
   - Padding과 무관!

2. **`plan_tokens`는 항상 같음**
   - `plan_tokens = ceil(horizon / frame_stack)`
   - Config로만 결정됨

3. **`current_levels`는 middle tokens만 표현**
   - Shape: `(batch_size, plan_tokens)`
   - Padding과 무관!

4. **Plan 재사용 시 재구성됨**
   - `plan_history`에서 가져온 plan은 이미 sliced
   - 새로운 padding으로 재구성
   - 매번 fresh padding

### 결론
✅ Padding이 달라도 **shape mismatch 발생 안 함**  
✅ 오히려 **논리적 일관성**이 더 중요  
✅ 사용자의 지적이 **정확했음**

## 교훈
제 원래 생각: "Shape 일관성을 위해 padding을 고정하자"
→ **잘못된 접근**: 실제로 shape는 horizon으로 결정됨

올바른 접근: "실제 token 구조에 맞게 padding을 계산하자"
→ **논리적 일관성** = 코드 이해도 향상 + 버그 감소
