"""
Central project configuration reader.
Reads scripts/project_config.sh so Python scripts share the same
DOCKER_USER / WANDB_ENTITY values as the shell scripts.
"""
import re
from pathlib import Path

def _load_config() -> dict:
    cfg_path = Path(__file__).parent / "project_config.sh"
    config: dict = {}
    for line in cfg_path.read_text().splitlines():
        m = re.match(r'^(\w+)="([^"]*)"$', line.strip())
        if m:
            config[m.group(1)] = m.group(2)
    return config

_cfg = _load_config()
DOCKER_USER:  str = _cfg.get("DOCKER_USER",  "jmseo1204")
WANDB_ENTITY: str = _cfg.get("WANDB_ENTITY", "jmseo1204-seoul-national-university")
