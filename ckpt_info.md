# Checkpoint Info

`outputs/downloaded/jmseo1204-seoul-national-university/mctd_eval/` 내 각 체크포인트의 학습 환경 및 설정 정보.

## 식별 방법

- `data_mean` / `data_std`: 체크포인트 state dict에서 추출 → dataset config의 `observation_mean`과 매핑
- `obs_dim`: `ckpt_scan_cache.json` 및 state dict의 `data_mean` shape에서 확인
- `model_dim`, `num_layers`: transformer `in_proj_weight` shape에서 추출
- WandB 확인 가능한 run (en1ddvu7, yc7ugn5p): API로 config 직접 확인

> ⚠️ **`episode_len` 주의**: 아래 표의 `episode_len`은 dataset config YAML의 **기본값**을 기재한 것. 실제 학습 시 Hydra 커맨드라인 override로 다른 값이 사용됐을 수 있음. 확인된 실제 값은 비고란에 표시.

---

## 체크포인트 목록

| run_id | dataset config | env_id (train) | obs_dim | episode_len | model_dim | epoch | global_step | 비고 |
|--------|---------------|----------------|---------|-------------|-----------|-------|-------------|------|
| `4tapu6is` | `og_antmaze_large_navigate` | `antmaze-large-v0` | 2 | 500 | 128 | 408 | 200,000 | |
| `5g4vp0wm` | `og_antmaze_large_navigate` | `antmaze-large-v0` | 2 | 500 | 128 | 408 | 200,000 | |
| `ij7c14bl` | `og_antmaze_large_navigate` | `antmaze-large-v0` | 2 | 500 | 128 | 81 | 40,000 | 학습 중단 (조기 ckpt) |
| `8b3xf51l` | `og_antmaze_medium_navigate` | `antmaze-medium-v0` | 2 | 500 | 128 | 408 | 200,000 | |
| `eqqsopw2` | `og_antmaze_medium_navigate` | `antmaze-medium-v0` | 2 | 500 | 128 | 408 | 200,000 | |
| `pzt9dsm4` | `og_antmaze_giant_navigate` | `antmaze-giant-v0` | 2 | 1000 | 128 | 408 | 200,000 | navigate dataset |
| `uzrq13fa` | `og_antmaze_giant_navigate` | `antmaze-giant-v0` | 2 | ~~1000~~ **200** | 128 | 408 | 200,000 | navigate dataset; 실제 학습 episode_len=200 (eval.sh .hydra/config.yaml 확인) |
| `en1ddvu7` | `og_antmaze_giant_stitch` | `antmaze-giant-v0` | 2 | 50 | 128 | 40 | 30,000 | stitch dataset; WandB 확인됨 |
| `yc7ugn5p` | `og_antmaze_giant_stitch` | `antmaze-giant-v0` | 2 | ~~50~~ **200** | 128 | 4,799 | 24,000 | stitch dataset; 실제 학습 episode_len=200 (WandB API 확인) |
| `71vbasu3` | `og_maze2d_giant_navigate` | `pointmaze-giant-v0` | 2 | 1000 | 128 | 408 | 200,000 | |
| `q940a89g` | `og_maze2d_giant_navigate` | `pointmaze-giant-v0` | 2 | 1000 | 128 | 408 | 200,000 | |
| `5wy35u14` | `og_maze2d_large_navigate` | `pointmaze-large-v0` | 2 | 500 | 128 | 408 | 200,000 | |
| `t2tlk0ca` | `og_maze2d_large_navigate` | `pointmaze-large-v0` | 2 | 500 | 128 | 408 | 200,000 | |
| `veii4g8t` | `og_maze2d_medium_navigate` | `pointmaze-medium-v0` | 2 | 500 | 128 | 408 | 200,000 | |
| `train_29d_local_20260305` | `og_antmaze_giant_navigate_fullstate` | `antmaze-giant-v0` | 29 | 1000 | 256 | 12 | 6,000 | 29D full-state, big model (9.7M params) |
| `train_2d_0ep_20260312060304_last` | `og_antmaze_giant_navigate` | `antmaze-giant-v0` | 2 | 1000 | 64 | 57 | 14,000 | small model (dim=64, 6 layers), 2026-03-12 로컬 학습 |
| `ynn5o8cb` | 알 수 없음 | 알 수 없음 | - | - | - | - | - | **파일 손상** (zip 오류) |

---

## 환경별 그룹

### AntMaze
| 환경 | dataset | run_ids |
|------|---------|---------|
| antmaze-medium-v0 | navigate | `8b3xf51l`, `eqqsopw2` |
| antmaze-large-v0 | navigate | `4tapu6is`, `5g4vp0wm`, `ij7c14bl` |
| antmaze-giant-v0 | navigate (2D) | `pzt9dsm4`, `uzrq13fa`, `train_2d_0ep_20260312060304_last` |
| antmaze-giant-v0 | navigate (29D full-state) | `train_29d_local_20260305` |
| antmaze-giant-v0 | stitch (2D) | `en1ddvu7`, `yc7ugn5p` |

### PointMaze (OGBench maze2d)
| 환경 | dataset | run_ids |
|------|---------|---------|
| pointmaze-medium-v0 | navigate | `veii4g8t` |
| pointmaze-large-v0 | navigate | `5wy35u14`, `t2tlk0ca` |
| pointmaze-giant-v0 | navigate | `71vbasu3`, `q940a89g` |

---

## 식별 근거 (data_mean 비교)

| data_mean (x, y) | dataset config | 매핑된 run_ids |
|-----------------|---------------|--------------|
| [25.44, 17.56] | og_antmaze_giant_stitch | en1ddvu7, yc7ugn5p |
| [24.52, 17.19] | og_antmaze_giant_navigate | pzt9dsm4, uzrq13fa, train_29d_local_20260305, train_2d_0ep_20260312060304_last |
| [25.19, 17.32] | og_maze2d_giant_navigate | 71vbasu3, q940a89g |
| [16.90, 11.44] | og_antmaze_large_navigate | 4tapu6is, 5g4vp0wm, ij7c14bl |
| [16.90, 11.01] | og_maze2d_large_navigate | 5wy35u14, t2tlk0ca |
| [10.08, 9.52] | og_antmaze_medium_navigate | 8b3xf51l, eqqsopw2 |
| [10.23, 9.57] | og_maze2d_medium_navigate | veii4g8t |

> **참고**: 동일 환경에 같은 model_dim/epoch를 가진 run이 여러 개 있는 경우 (e.g., `4tapu6is`/`5g4vp0wm`, `8b3xf51l`/`eqqsopw2`) 동일 학습 설정에서 서로 다른 seed 또는 실험 variant일 가능성이 높음.
> **참고**: `ynn5o8cb`은 zip 아카이브 오류로 체크포인트 로드 불가 (파일 손상).

---

## WandB 평가 로그에서 확인된 eval 환경 (cross-env 평가 포함)

일부 체크포인트는 학습 환경과 **다른** 환경에서 평가됨.

| run_id | 학습 env (train) | eval env (WandB 확인) | 비고 |
|--------|-----------------|----------------------|------|
| `uzrq13fa` | antmaze-giant-v0 (navigate) | antmaze-giant-v0 | 동일 환경 평가 |
| `5g4vp0wm` | antmaze-large-v0 (navigate) | antmaze-giant-v0 | **cross-env**: large→giant |
| `4tapu6is` | antmaze-large-v0 (navigate) | antmaze-giant-v0 | **cross-env**: large→giant |
| `8b3xf51l` | antmaze-medium-v0 (navigate) | maze2d-large-v1 (obs=[4]) | **cross-env** (비정상적 설정일 가능성) |
| `en1ddvu7`, `yc7ugn5p` | antmaze-giant-v0 (stitch) | — | WandB training run만 확인, eval 별도 |
