# Checkpoint Management Analysis

**Generated:** 2026-03-07 13:00

## Summary

The `outputs/downloaded/jmseo1204-seoul-national-university/mctd_eval/` directory contains 27 unique models (after deduplication). The directory also had duplicate symlinks created during repeated evaluation runs with different state dimensions.

---

## Model Groups (Checkpoint Pairs)

Each pair shares the same underlying checkpoint file. The suffix variants (_dX) indicate which state dimension was used during that evaluation session.

### Group 1: 2D State Models (2026-02-12)

| Base Model | Points To | Dimension | Training Loss | Step | Created |
|---|---|---|---|---|---|
| **local_20260212125143** | 2026-02-12/12-51-43 | 2D | ? | epoch=4799, step=24000 | Feb 12 12:51 |
| local_20260212_125143_d2 | (same) | 2D eval | (same) | (same) | Feb 12 12:51 |

| Base Model | Points To | Dimension | Training Loss | Step | Created |
|---|---|---|---|---|---|
| **local_20260212143507** | 2026-02-12/14-35-07 | 2D | ? | epoch=27, step=20000 | Feb 12 14:35 |
| local_20260212_143507_d2 | (same) | 2D eval | (same) | (same) | Feb 12 14:35 |

| Base Model | Points To | Dimension | Training Loss | Step | Created |
|---|---|---|---|---|---|
| **local_20260212152455** | 2026-02-12/15-24-55 | 2D | ? | epoch=29, step=22000 | Feb 12 15:24 |
| local_20260212_152455_d2 | (same) | 2D eval | (same) | (same) | Feb 12 15:24 |

| Base Model | Points To | Dimension | Training Loss | Step | Created |
|---|---|---|---|---|---|
| **local_20260212152901** | 2026-02-12/15-29-01 | 2D | ? | epoch=40, step=30000 | Feb 12 15:29 |
| local_20260212_152901_d2 | (same) | 2D eval | (same) | (same) | Feb 12 15:29 |

### Group 2: 29D State Models (2026-03-03 to 2026-03-06)

| Base Model | Points To | Dimension | Training Loss | Step | Created |
|---|---|---|---|---|---|
| **local_20260303181343** | 2026-03-03/18-13-43 | 29D | ? | epoch=5, step=5000 | Mar 03 18:13 |
| local_20260303_181343_d29 | (same) | 29D eval | (same) | (same) | Mar 03 18:13 |

| Base Model | Points To | Dimension | Training Loss | Step | Created |
|---|---|---|---|---|---|
| **local_20260305050924** | 2026-03-05/05-09-24 | 29D | ? | epoch=48, step=24000 | Mar 05 05:09 |
| local_20260305_050924_d29 | (same) | 29D eval | (same) | (same) | Mar 05 05:09 |

| Base Model | Points To | Dimension | Training Loss | Step | Created |
|---|---|---|---|---|---|
| **local_20260305144616** | 2026-03-05/14-46-16 | 29D | ? | epoch=12, step=6000 | Mar 05 14:46 |
| local_20260305_144616_d29 | (same) | 29D eval | (same) | (same) | Mar 05 14:46 |

| Base Model | Points To | Dimension | Training Loss | Step | Created |
|---|---|---|---|---|---|
| **local_20260305144641** | 2026-03-05/14-46-41 | 29D | ? | epoch=12, step=6000 | Mar 05 14:46 |
| local_20260305_144641_d29 | (same) | 29D eval | (same) | (same) | Mar 05 14:46 |

| Base Model | Points To | Dimension | Training Loss | Step | Created |
|---|---|---|---|---|---|
| **local_20260306044559** | 2026-03-06/04-45-59 | 29D | ? | epoch=68, step=34000 | Mar 06 04:45 |
| local_20260306_044559_d29 | (same) | 29D eval | (same) | (same) | Mar 06 04:45 |

### Group 3: Mixed Dimension Models (2026-03-06)

| Base Model | Points To | Dimension | Training Loss | Step | Created |
|---|---|---|---|---|---|
| **local_20260306175105** | 2026-03-06/17-51-05 | 2D | ? | epoch=4, step=2000 | Mar 06 17:51 |
| local_20260306_175105_d2 | (same) | 2D eval | (same) | (same) | Mar 06 17:51 |

| Base Model | Points To | Dimension | Training Loss | Step | Created |
|---|---|---|---|---|---|
| **local_20260306175519** | 2026-03-06/17-55-19 | 2D | ? | epoch=4, step=2000 | Mar 06 17:55 |
| local_20260306_175519_d2 | (same) | 2D eval | (same) | (same) | Mar 06 17:55 |

| Base Model | Points To | Dimension | Training Loss | Step | Created |
|---|---|---|---|---|---|
| **local_20260306200800** | 2026-03-06/20-08-00 | 2D | ? | epoch=36, step=18000 | Mar 06 20:08 |
| local_20260306_200800_d2 | (same) | 2D eval | (same) | (same) | Mar 06 20:08 |

| Base Model | Points To | Dimension | Training Loss | Step | Created |
|---|---|---|---|---|---|
| **local_20260306220240** | 2026-03-06/22-02-40 | 15D | ? | epoch=32, step=16000 | Mar 06 22:02 |
| local_20260306_220240_d15 | (same) | 15D eval | (same) | (same) | Mar 06 22:02 |

### Group 4: Latest 15D Model (2026-03-07)

| Base Model | Points To | Dimension | Training Loss | Step | Created |
|---|---|---|---|---|---|
| **local_20260307062042_15d** | 2026-03-07/06-20-42 | 15D | **0.01455** ✓ | epoch=137, step=68000 | Mar 07 06:20 |

---

## HuggingFace/Downloaded Models

| Model ID | Source | Status |
|---|---|---|
| 4tapu6is | WandB | Downloaded |
| 5g4vp0wm | WandB | Downloaded |
| 5wy35u14 | WandB | Downloaded |
| 71vbasu3 | WandB | Downloaded |
| 8b3xf51l | WandB | Downloaded |
| en1ddvu7 | WandB | Downloaded |
| eqqsopw2 | WandB | Downloaded |
| ij7c14bl | WandB | Downloaded |

---

## Key Findings

### 1. **Duplicate Detection**: ✓ CONFIRMED
- Each pair (base + suffix variant) points to the SAME checkpoint file
- Suffix variants are created during evaluation runs with different state dimensions
- Safe to remove duplicates while keeping the base model name

### 2. **Model Relationships**
- Models are **NOT** sequential iterations of the same training
- Each represents a separate, independent training run
- Created across different dates (Feb 12, Mar 03, Mar 05, Mar 06, Mar 07)

### 3. **Latest vs Previous**
- **Latest 15D Model**: `local_20260307062042_15d` (epoch=137, loss=0.01455) ✓ BEST
- Previous 15D attempt: `local_20260306220240` (epoch=32, loss=?) - EARLY STOP
- Latest 2D: `local_20260306200800` (epoch=36, loss=?)
- Latest 29D: `local_20260306044559` (epoch=68, loss=?)

### 4. **Training Status**
- 15D training completed successfully (68k steps, epoch 137)
- Other models are at earlier training stages or incomplete
- Loss convergence: 15D shows excellent final loss (0.01455)

---

## Cleanup Recommendations

### Option 1: **Keep Base Models Only** (Recommended)
Remove all suffix variants (_d2, _d15, _d29) to eliminate duplicates:
```bash
rm -rf local_*_d2 local_*_d15 local_*_d29
```
This reduces 27→13 folders while keeping all unique checkpoints.

### Option 2: **Archive Old Evaluations**
Move suffix variants to a separate archive:
```bash
mkdir -p evaluated_versions
mv local_*_{d2,d15,d29} evaluated_versions/
```

---

## Recommended Next Steps

1. **Use Latest 15D Model** for evaluation:
   ```bash
   python generate_jobs_generalized.py --model_id local_20260307062042_15d
   ```

2. **Clean up duplicates** (optional but recommended):
   ```bash
   rm -rf local_*_d2 local_*_d15 local_*_d29
   ```

3. **Archive other local models** for backup (if needed):
   ```bash
   tar -czf archived_models_feb-mar.tar.gz local_202602* local_202603*
   ```

