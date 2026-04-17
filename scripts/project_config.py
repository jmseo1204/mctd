"""
Central project configuration reader.
Reads scripts/project_config.sh so Python scripts share the same
DOCKER_USER / WANDB_ENTITY values as the shell scripts.
"""
import os
import re
from pathlib import Path
from typing import List

def _load_config() -> dict:
    cfg_path = Path(__file__).parent / "project_config.sh"
    config: dict = {}
    for line in cfg_path.read_text().splitlines():
        m = re.match(r'^(\w+)="([^"]*)"$', line.strip())
        if m:
            config[m.group(1)] = m.group(2)
    return config

_cfg = _load_config()
DOCKER_USER:    str       = _cfg.get("DOCKER_USER",    "jmseo1204")
WANDB_ENTITY:   str       = _cfg.get("WANDB_ENTITY",   "jmseo1204-seoul-national-university")
DOCKER_IMAGE:   str       = _cfg.get("DOCKER_IMAGE",   "mctd:0.1")
WANDB_PROJECT:  str       = _cfg.get("WANDB_PROJECT",  "mctd_eval")
AVAILABLE_GPUS: List[str] = (
    os.environ.get("AVAILABLE_GPUS") or _cfg.get("AVAILABLE_GPUS", "localhost:0") or "localhost:0"
).split(",")
