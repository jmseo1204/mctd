# Claude Code Session Instructions

This file contains reminders and guidelines for Claude Code when working on this MCTD repository. Read this at the start of each session.

## Before Any Prompt - Critical Checks

1. **Consult CLAUDE.md First**
   - Reference `/mnt/c/Users/USER/Desktop/test_ogbench/mctd_repo/CLAUDE.md` for architecture and patterns
   - This is the authoritative guide for the project

2. **Understand the Primary File**
   - `algorithms/diffusion_forcing/df_planning.py` is the **authoritative implementation**
   - It overrides df_base.py behavior
   - Always treat df_planning.py as the source of truth for MCTD algorithm

3. **Key Tensor Dimensions (Always Verify)**
   - Observations: `(batch, n_tokens, obs_dim)` after frame stacking
   - Actions: `(batch, n_tokens-1, action_dim)` — **one less token than observations**
   - Noise levels: `(batch, n_tokens)` aligned with diffusion timesteps
   - If modifying `_construct_sequence()`, validate these dimensions

4. **Check Line Numbers in df_planning.py**
   - Refer to the method groups in CLAUDE.md § "Working with df_planning.py" → "Code Organization"
   - This helps quickly locate code sections in the ~2600-line file

## Project Focus Areas

### Primary Focus: df_planning.py Improvements
- Understand `DiffusionForcingPlanning` class architecture
- Main entry points: `interact()`, `p_mctd_plan()`, `parallel_plan()`
- Core MCTS: `_run_mcts_search()`, `_init_mcts_tree()`, `_select_best_leaf()`
- Value computation: `calculate_values()`, `calculate_values_bidir()`

### Secondary: Supporting Components
- `tree_node.py` — TreeNode structure for MCTS
- Config management — Always use Hydra (`configurations/df_planning.yaml`)
- Experiment workflow — `exp_planning.py` for running validation/evaluation

## What To Avoid

### Code Modifications
- ❌ **Don't modify df_base.py** unless absolutely necessary (it's just infrastructure)
- ❌ **Don't hardcode parameters** — always use Hydra config
- ❌ **Don't change tensor shapes** without validating throughout the pipeline
- ❌ **Don't skip padding validation** when modifying sequence logic

### Configuration
- ❌ **Don't override Hydra defaults** in code (use config files)
- ❌ **Don't assume relative paths** work with Hydra (use absolute paths or W&B run IDs)
- ❌ **Don't forget to set WANDB_ENTITY** before running experiments

### Common Mistakes
- ❌ **Padding mode inconsistency** — If you change `padding_mode`, validate with `test_padding_consistency.py`
- ❌ **Bidirectional search misalignment** — `is_unknown_final_token` and `sequence_dividing_factor` must be consistent
- ❌ **Value computation without bounds** — `calculate_values()` is expensive; respect `mctd_num_denoising_steps` limits
- ❌ **Memory bloat with parallel_search_num** — Each instance holds its own MCTSTreeState

## df_planning.py Specific Guidelines

### When Reading Code
1. **Start with method size and location** (see line ranges in CLAUDE.md)
2. **Understand the entry point**:
   - `interact()` → `p_mctd_plan()` → `_run_mcts_search()` → plan extraction
3. **Follow the data flow**:
   - Observations → Sequence construction → Diffusion inference → Value computation → Tree search

### When Modifying Code
1. **Identify the exact section** using line numbers from CLAUDE.md
2. **Check dependencies**:
   - If modifying sequence construction, validate tensor shapes throughout
   - If modifying search logic, check value computation paths
   - If adding guidance, integrate into `_run_mcts_search()` value update
3. **Test incrementally**:
   - Use provided test files: `test_padding_consistency.py`, etc.
   - Run on small `mctd_max_search_num` and `parallel_search_num` first

### Parameter Tuning
- **Search quality vs speed**: `mctd_num_denoising_steps` and `mctd_max_search_num`
- **Parallelization**: `parallel_search_num` and `leaf_parallelization`
- **Guidance strength**: `mctd_guidance_scales` (list of scales for exploration)
- **Tree convergence**: `meeting_delta` and `early_stopping_condition`

## Common Development Workflow

### Running Experiments
```bash
# Always use Hydra for config management
python main.py +name=test_run algorithm=df_planning dataset=pointmaze experiment=validation

# Override specific parameters
python main.py +name=test algorithm=df_planning mctd_num_denoising_steps=50 parallel_search_num=4

# Offline mode (no W&B)
python main.py +name=test algorithm=df_planning wandb.mode=offline
```

### Debugging
1. **Enable offline mode** to iterate quickly without W&B
2. **Reduce search scope**: `mctd_max_search_num=10`, `parallel_search_num=1`
3. **Check tensor shapes** in `_construct_sequence()` and value computation
4. **Validate padding** with `test_padding_consistency.py`

### Testing
- Run relevant test scripts before committing: `python test_padding_consistency.py`
- Check gradient flow: `python test_gradient_flow.py`
- Validate trajectory properties: `python test_trajectory_diagnostics.py`

## File Organization Reference

**Always consult CLAUDE.md § "Key Files to Know" for:**
- Primary implementation: `df_planning.py`
- Supporting files: tree_node.py, df_base.py, exp_planning.py
- Config files: `configurations/algorithm/df_planning.yaml`

**Memory location**: `/home/jmseo1204/.claude/projects/-mnt-c-Users-USER-Desktop-test-ogbench-mctd-repo/memory/`
- Check memory files for recurring solutions and patterns

## Checklist Before Starting Work

- [ ] Read relevant section in CLAUDE.md
- [ ] Identify target file and method (with line numbers)
- [ ] Verify tensor dimensions if working with sequences
- [ ] Check if changes affect df_planning.py core methods
- [ ] Plan for validation/testing before implementing
- [ ] Ensure no hardcoded parameters are introduced

---

**Last Updated**: 2026-02-21  
**Primary Focus**: MCTD planning in df_planning.py  
**Related Docs**: CLAUDE.md, Memory files
