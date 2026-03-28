"""
utils/tracer.py

Stateful context manager-based structured logger for AI research.
Follows the Structured Log Protocol (SLP) compatible with log-analysis skill.
"""

import os
import json
import time
import signal
import atexit
import logging
import traceback
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict

# ── SLP 필드 상수 ──────────────────────────────────────────────────────────────
SLP_VERSION = "1.0"

# ── DEBUG_MODE 전역 플래그 ────────────────────────────────────────────────────
DEBUG_MODE: bool = os.environ.get("DEBUG_MODE", "1").strip() == "1"


class _NullScope:
    """DEBUG_MODE=False 시 완전한 no-op context manager."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def log(self, *args, **kwargs): pass


class Tracer:
    """실험 run 단위 Stateful Logger."""

    def __init__(
        self,
        run_id: Optional[str] = None,
        purpose: str = "general_monitoring",
        log_dir: str = "logs_memory_debug",
        debug_mode: Optional[bool] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ):
        self.debug_mode = debug_mode if debug_mode is not None else DEBUG_MODE
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.purpose = purpose
        self.log_dir = Path(log_dir)
        self.extra_meta = extra_meta or {}

        self._log_path: Optional[Path] = None
        self._fh: Optional[Any] = None
        self._current_phase: str = "init"
        self._current_group: str = ""
        self._active: bool = False

    def __enter__(self):
        if not self.debug_mode:
            return self
        self._setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.debug_mode:
            return False
        if exc_type is not None:
            self.log_exception(exc_type, exc_val, exc_tb)
        self._flush_and_close()
        return False

    def _setup(self):
        if self._active:   # already open — skip re-initialization
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.log_dir / f"{self.run_id}.jsonl"
        self._fh = open(self._log_path, "a", buffering=1)
        self._active = True

        atexit.register(self._flush_and_close)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._signal_handler)
            except (OSError, ValueError):
                pass

        self._write({
            "ts": time.time(),
            "level": "INFO",
            "run_id": self.run_id,
            "phase": "init",
            "step": None,
            "tag": "run.start",
            "group": "run_meta",
            "depth": 0,
            "data": {
                "purpose": self.purpose,
                "log_path": str(self._log_path),
                "slp_version": SLP_VERSION,
                **self.extra_meta,
            },
            "source": "tracer.py:0",
        })

    def _signal_handler(self, signum, frame):
        self._flush_and_close()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def _flush_and_close(self):
        if self._active and self._fh and not self._fh.closed:
            self._write({
                "ts": time.time(),
                "level": "INFO",
                "run_id": self.run_id,
                "phase": self._current_phase,
                "step": None,
                "tag": "run.end",
                "group": "run_meta",
                "depth": 0,
                "data": {},
                "source": "tracer.py:0",
            })
            self._fh.flush()
            self._fh.close()
            self._active = False

    @contextlib.contextmanager
    def scope(self, group: str, phase: Optional[str] = None, depth: int = 0):
        if not self.debug_mode:
            yield _NullScope()
            return

        prev_group = self._current_group
        prev_phase = self._current_phase
        if phase:
            self._current_phase = phase
        self._current_group = group
        try:
            yield self
        finally:
            self._current_group = prev_group
            self._current_phase = prev_phase

    def log(
        self,
        tag: str,
        data: Dict[str, Any],
        step: Optional[int] = None,
        depth: int = 0,
        source: Optional[str] = None,
    ):
        if not self.debug_mode:
            return

        level = "INFO" if depth == 0 else "DEBUG"

        if source is None:
            import inspect
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller = frame.f_back
                source = f"{Path(caller.f_code.co_filename).name}:{caller.f_lineno}"
            else:
                source = "unknown"

        record = {
            "ts": time.time(),
            "level": level,
            "run_id": self.run_id,
            "phase": self._current_phase,
            "step": step,
            "tag": tag,
            "group": self._current_group,
            "depth": depth,
            "data": data,
            "source": source,
            "purpose": self.purpose,
        }
        self._write(record)

    def log_exception(self, exc_type, exc_val, exc_tb):
        if not self.debug_mode:
            return
        tb_str = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
        self._write({
            "ts": time.time(),
            "level": "ERROR",
            "run_id": self.run_id,
            "phase": self._current_phase,
            "step": None,
            "tag": "exception.unhandled",
            "group": "error",
            "depth": 0,
            "data": {
                "exc_type": exc_type.__name__ if exc_type else "Unknown",
                "exc_msg": str(exc_val),
                "traceback": tb_str,
            },
            "source": "tracer.py:0",
            "purpose": self.purpose,
        })

    def log_tensor_stats(self, tag: str, tensor, step: Optional[int] = None, depth: int = 1, label: Optional[str] = None):
        if not self.debug_mode:
            return
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                t = tensor.detach().float()
                data = {
                    "shape": list(t.shape),
                    "dtype": str(tensor.dtype),
                    "mean": float(t.mean()) if t.numel() > 0 else None,
                    "std": float(t.std()) if t.numel() > 1 else None,
                    "min": float(t.min()) if t.numel() > 0 else None,
                    "max": float(t.max()) if t.numel() > 0 else None,
                    "nan_count": int(torch.isnan(t).sum()),
                    "inf_count": int(torch.isinf(t).sum()),
                    "norm": float(t.norm()) if t.numel() > 0 else None,
                }
                if label:
                    data["label"] = label
                self.log(tag, data, step=step, depth=depth)
                return
        except ImportError:
            pass

    def append_record(self, tag: str, data: dict) -> None:
        """Append a single record to the log file (works even after the tracer is closed)."""
        if self._log_path is None:
            return
        record = {
            "ts": time.time(),
            "tag": tag,
            "data": data,
        }
        try:
            with open(str(self._log_path), "a") as _af:
                _af.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass

    def _write(self, record: Dict[str, Any]):
        if self._fh and not self._fh.closed:
            try:
                self._fh.write(json.dumps(record, default=str) + "\n")
            except Exception:
                pass


_default_tracer: Optional[Tracer] = None

def get_tracer() -> Optional[Tracer]:
    return _default_tracer

def set_default_tracer(tracer: Tracer):
    global _default_tracer
    _default_tracer = tracer
