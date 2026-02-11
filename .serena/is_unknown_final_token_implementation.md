# is_unknown_final_token Feature Implementation

## 목적
사용자 가설 테스트: final_token이 stabilization되어 있을 때 모델이 start와 goal 중 어디서 경로를 시작해야 할지 혼란스러워하는지 확인

## 구현 내용

### 1. 새로운 Config Flag 추가
**파일**: `configurations/algorithm/df_planning.yaml`
```yaml
is_unknown_final_token: False  # Test: Set to True if final_token stabilization causes confusion
```

### 2. 핵심 로직
**파일**: `algorithms/diffusion_forcing/df_planning.py`

#### A. Helper Function 추가 (`_construct_noise_levels`)
- **위치**: L315-346
- **기능**: noise level 배열 구성 시 final_token 포함 여부 제어
- **파라미터**: `include_final_token` - True면 final_token 포함, False면 제외

#### B. `parallel_plan` 함수 수정
- **파라미터 추가**: `is_unknown_final_token: bool = False`
- **Type hints 개선**: 모든 파라미터에 type hint 추가

#### C. 토큰 시퀀스 구성 로직
```python
use_bidirectional_sequence = self.bidirectional_search and not is_unknown_final_token
```

- `is_unknown_final_token=False`: 기존 bidirectional 방식 (final_token 포함)
- `is_unknown_final_token=True`: unidirectional 방식 (final_token 제외)

### 3. 동작 방식

#### Case 1: is_unknown_final_token=False (기본값)
```
Tokens: [init_token(start), middle_tokens, final_token(goal), padding]
Noise:  [stabilization(0), variable_levels, stabilization(0), max_noise(100)]
```
- 모델이 start와 goal 모두를 완전히 denoised된 상태로 받음
- **가설**: 이로 인해 어디서 시작할지 혼란

#### Case 2: is_unknown_final_token=True (테스트 모드)
```
Tokens: [init_token(start), middle_tokens, padding]
Noise:  [stabilization(0), variable_levels, max_noise(100)]
```
- 모델이 start만 완전히 denoised된 상태로 받음
- Goal은 middle_tokens의 일부로 포함 (variable noise)
- **가설**: 혼란 감소, 경로가 start에서 시작

### 4. 코드 리팩토링
기존 코드 반복을 방지하기 위해:
- `_construct_noise_levels` helper function 추가
- bidirectional/unidirectional 모드를 `include_final_token` 파라미터로 통합

### 5. 수정된 파일 목록
1. `algorithms/diffusion_forcing/df_planning.py`
   - `_construct_noise_levels` 함수 추가
   - `parallel_plan` 함수 시그니처 수정
   - MCTD 호출 부분 2곳 수정
   - Config 로딩 추가
   
2. `configurations/algorithm/df_planning.yaml`
   - `is_unknown_final_token` 플래그 추가

### 6. 테스트 방법

#### Step 1: 기본 동작 확인 (is_unknown_final_token=False)
```bash
# df_planning.yaml에서 is_unknown_final_token: False 확인
python main.py task=validation
```

#### Step 2: 테스트 모드 실행 (is_unknown_final_token=True)
```bash
# df_planning.yaml에서 is_unknown_final_token: True로 변경
python main.py task=validation
```

#### Step 3: 결과 비교
시각화된 경로 이미지 확인:
- `plan/plan_at_{steps}_from_start`: forward planning 경로
- `plan/plan_at_{steps}_from_goal`: backward planning 경로

**기대 결과**:
- False: 경로가 map 가운데에서 시작 (현재 문제)
- True: 경로가 start position에서 시작 (가설이 맞다면)

### 7. 구현 원칙 준수

✅ **Flag가 False일 때 기존 코드와 정확히 동일하게 동작**
- `use_bidirectional_sequence = self.bidirectional_search and not is_unknown_final_token`
- Flag=False → 기존 bidirectional 모드 그대로

✅ **코드 반복 없이 함수화**
- `_construct_noise_levels` helper function으로 중복 제거

✅ **구현사항 무관한 코드는 손대지 않음**
- 주석 유지
- 기존 로직 유지

✅ **Type Hinting 추가**
- `parallel_plan` 함수의 모든 파라미터에 type hint 추가

✅ **단계별 구현 및 테스트**
- Helper function 추가 → 컴파일 테스트
- parallel_plan 수정 → 컴파일 테스트
- Config 연결 → 통합 테스트

## 결론

이 구현은 사용자의 가설을 테스트하기 위한 실험적 기능입니다:
- **가설**: final_token stabilization이 모델을 혼란스럽게 함
- **테스트**: `is_unknown_final_token=True`로 final_token 제거
- **검증**: 경로가 start에서 올바르게 시작하는지 확인

가설이 확인되면, `is_unknown_final_token=True`를 기본값으로 설정하거나,
bidirectional_search 방식을 재검토할 수 있습니다.
