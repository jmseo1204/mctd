"""
Shared planning math utilities.

Keep in sync with any caller that computes plan_tokens from episode_len:
  - algorithms/diffusion_forcing/df_planning.py  (__init__ + interact)
  - scripts/generate_jobs_generalized.py          (validation block)
"""


def episode_len_to_plan_tokens(episode_len: int, jump: int, frame_stack: int) -> int:
    """Convert raw episode length to plan token count.

    Args:
        episode_len:  raw episode length (pre-jump).
        jump:         frame-skip (from ckpt training_hparams).
        frame_stack:  frames per token (from ckpt training_hparams).

    Returns:
        plan_tokens (int): number of diffusion tokens.

    Derivation:
        horizon     = episode_len // jump    # jump-adjusted frames
        plan_tokens = horizon // frame_stack
    """
    return (episode_len // jump) // frame_stack
