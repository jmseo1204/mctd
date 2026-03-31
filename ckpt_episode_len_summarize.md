# Checkpoint Training Parameters Summary

Generated from `outputs/downloaded/jmseo1204-seoul-national-university/mctd_eval/`

## 열 설명

| 열 | 설명 |
|---|---|
| `episode_len_eff` | 모델이 실제로 학습한 시퀀스 길이 = `raw_episode_len // jump`. `n_tokens = episode_len_eff // frame_stack` |
| `jump` | dataset 서브샘플링 stride. 학습 시 dataset이 `raw_episode_len // jump + 1` 프레임을 제공 |
| `source` | 파라미터 출처. 신뢰도 순: `ckpt_hparams` > `wandb/` (실제 훈련 run) > `wandb/hydra` (eval .hydra) > `training_config.yaml` > `no config found` |

> **주의**: `wandb/hydra (날짜)` 출처는 해당 날짜에 수행된 **eval 실행**의 `.hydra/config.yaml`에서 읽어온 것으로, 훈련 당시 config가 아닐 수 있음.
> `wandb/` 출처는 실제 WandB 훈련 run의 `files/config.yaml`로 신뢰도가 높음.

---

## 모델별 요약

| model_id | episode_len_eff | jump | frame_stack | causal | scheduling_matrix | attn_heads | source |
|---|---|---|---|---|---|---|---|
| `4tapu6is` | 1000 | 1 | 10 | False | smooth | 8 | wandb/hydra (2026-03-06) |
| `5g4vp0wm` | 1000 | 1 | 10 | False | pyramid | 4 | wandb/hydra (2026-02-22) |
| `5wy35u14` | ? | 1 | ? | ? | ? | ? | no config found |
| `71vbasu3` | ? | 1 | ? | ? | ? | ? | no config found |
| `8b3xf51l` | 800 | 1 | 10 | False | pyramid | 4 | wandb/hydra (2026-02-22) |
| `en1ddvu7` | 50 | 1 | 10 | False | pyramid | 4 | wandb/ (훈련 run 직접) |
| `eqqsopw2` | ? | 1 | ? | ? | ? | ? | no config found |
| `ij7c14bl` | ? | 1 | ? | ? | ? | ? | no config found |
| `pzt9dsm4` | 1000 | 1 | 10 | False | pyramid | 4 | wandb/hydra (2026-01-27) |
| `q940a89g` | ? | 1 | ? | ? | ? | ? | no config found |
| `t2tlk0ca` | ? | 1 | ? | ? | ? | ? | no config found |
| `train_29d_local_20260305` | 1000 | 1 | 10 | False | smooth | 8 | wandb/hydra (2026-03-05) |
| `train_2d_0ep_20260312060304_last` | 1000 | 1 | 10 | False | smooth | 8 | wandb/hydra (2026-03-12) |
| `uzrq13fa` | 200 | 5 | 10 | False | pyramid | 4 | wandb/hydra (2026-01-28) ⚠️ |
| `veii4g8t` | ? | 1 | ? | ? | ? | ? | no config found |
| `yc7ugn5p` | 200 | 1 | 10 | False | pyramid | 4 | wandb/ (훈련 run 직접) |
| `ynn5o8cb` | ? | 1 | ? | ? | ? | ? | no config found |

---

## 주요 관찰

### `uzrq13fa` ⚠️
- `jump=5`, `episode_len_eff=200` — 출처가 eval .hydra config이므로 훈련 실제값인지 불확실
- `episode_len_eff * jump = 200 * 5 = 1000` = dataset YAML의 raw episode_len과 일치
- 실제 훈련 시 모델이 처리한 토큰 수: `(1000 // 5) // 10 ≈ 20 tokens`

### `en1ddvu7`
- `episode_len_eff=50` — WandB 훈련 run에서 직접 확인 (신뢰도 높음)
- `og_antmaze_giant_stitch` 데이터셋으로 학습된 것으로 추정 (stitch의 episode_len=50)

### `yc7ugn5p`
- `episode_len_eff=200`, `jump=1` — WandB 훈련 run에서 직접 확인
- 20 tokens per sequence로 학습

### config 없는 모델들 (`5wy35u14`, `71vbasu3`, `eqqsopw2`, `ij7c14bl`, `q940a89g`, `t2tlk0ca`, `veii4g8t`, `ynn5o8cb`)
- WandB online 훈련 기록 없음, `training_config.yaml` 없음
- `training_hparams`도 없음 (on_save_checkpoint 추가 이전 checkpoint)
- eval 시 dataset YAML 기본값 사용 (og_antmaze_giant_navigate: episode_len=1000, jump=1)

---

## episode_len 로딩 우선순위 (현재 로직)

```
generate_jobs_generalized.py 기준:

[신규 ckpt - training_hparams 있음]
  ckpt_hparams['jump']         → actual_jump
  ckpt_hparams['episode_len']  → effective (이미 나눠진 값)

[레거시 ckpt - training_hparams 없음]
  model_metadata['jump']       → actual_jump  (WandB/hydra config)
  dataset_yaml['episode_len'] // actual_jump → episode_len_eff  (우선)
  model_metadata['episode_len']              → (fallback, 이미 나눠진 값)
```
