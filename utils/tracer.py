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
import traceback
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict

# ── SLP 필드 상수 ──────────────────────────────────────────────────────────────
SLP_VERSION = "1.0"

# ── DEBUG_MODE 전역 플래그 ────────────────────────────────────────────────────
# 환경변수 또는 코드에서 설정 가능
DEBUG_MODE: bool = os.environ.get("DEBUG_MODE", "1").strip() == "1"


class _NullScope:
    """DEBUG_MODE=False 시 완전한 no-op context manager."""
    def __enter__(self): return self
    def __exit__(self, *_args): pass  # Context manager protocol
    def log(self, *_args, **_kwargs): pass


class Tracer:
    """
    실험 run 단위 Stateful Logger.

    사용법:
        tracer = Tracer(run_id="my_exp", purpose="loss_nan_diagnosis")
        with tracer:
            with tracer.scope("train_loop", phase="train"):
                tracer.log("loss.train.total", {"value": loss.item()}, step=step)
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        purpose: str = "general_monitoring",
        log_dir: str = "logs",
        debug_mode: Optional[bool] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ):
        self.debug_mode = debug_mode if debug_mode is not None else DEBUG_MODE
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.purpose = purpose
        self.log_dir = Path(log_dir)
        self.extra_meta = extra_meta or {}

        self._log_path: Optional[Path] = None
        self._fh: Optional[Any] = None  # file handle
        self._current_phase: str = "init"
        self._current_group: str = ""
        self._active: bool = False

    # ── 생명주기 ──────────────────────────────────────────────────────────────

    def __enter__(self):
        if not self.debug_mode:
            return self
        self._setup()
        return self

    def __exit__(self, exc_type, exc_val, _exc_tb):
        if not self.debug_mode:
            return False
        if exc_type is not None:
            self.log_exception(exc_type, exc_val, _exc_tb)
        self._flush_and_close()
        return False  # 예외를 억제하지 않음

    def _setup(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.log_dir / f"{self.run_id}.jsonl"
        self._fh = open(self._log_path, "a", buffering=1)  # line-buffered
        self._active = True

        # atexit + signal 등록: 비정상 종료 시에도 flush 보장
        atexit.register(self._flush_and_close)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._signal_handler)
            except (OSError, ValueError):
                pass  # non-main thread 등 signal 설정 불가 환경 무시

        # run 시작 메타데이터 기록
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
        # 기본 핸들러 복원 후 재발생
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

    # ── scope context manager ─────────────────────────────────────────────────

    @contextlib.contextmanager
    def scope(self, group: str, phase: Optional[str] = None, depth: int = 0):
        """
        같은 디버깅 목적의 로그를 묶는 scope.

        Args:
            group: 로그 그룹명 (예: "optimizer", "data_pipeline")
            phase: 실험 단계 (예: "train", "eval"). None이면 현재 phase 유지
            depth: 0=coarse(INFO), 1+=fine(DEBUG)

        DEBUG_MODE=False 시 완전한 no-op.
        """
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

    # ── 로깅 메서드 ──────────────────────────────────────────────────────────

    def log(
        self,
        tag: str,
        data: Dict[str, Any],
        step: Optional[int] = None,
        depth: int = 0,
        source: Optional[str] = None,
    ):
        """
        구조화된 레코드를 jsonl에 기록.

        Args:
            tag: 점(dot) 계층 구조의 고유 식별자 (예: "loss.train.total")
            data: 기록할 값 딕셔너리
            step: 현재 iteration/epoch
            depth: 0=coarse(INFO), 1+=fine(DEBUG)
            source: "파일명:라인번호" 문자열 (None이면 자동 추론)
        """
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
        """예외 발생 시 ERROR 레벨로 기록."""
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

    def log_tensor_stats(
        self,
        tag: str,
        tensor,
        step: Optional[int] = None,
        depth: int = 1,
        label: Optional[str] = None,
    ):
        """
        텐서 통계를 fine 레벨로 기록.
        PyTorch tensor를 가정하나, numpy array도 처리.

        분해-재조립 무결성: 텐서를 detach/cpu 복사본에서만 통계 추출.
        원본 tensor는 절대 변경하지 않는다.
        """
        if not self.debug_mode:
            return

        try:
            # numpy 또는 torch tensor 모두 처리
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

            import numpy as np
            arr = np.asarray(tensor).astype(float)
            data = {
                "shape": list(arr.shape),
                "dtype": str(tensor.dtype) if hasattr(tensor, "dtype") else str(arr.dtype),
                "mean": float(np.mean(arr)) if arr.size > 0 else None,
                "std": float(np.std(arr)) if arr.size > 1 else None,
                "min": float(np.min(arr)) if arr.size > 0 else None,
                "max": float(np.max(arr)) if arr.size > 0 else None,
                "nan_count": int(np.isnan(arr).sum()),
                "inf_count": int(np.isinf(arr).sum()),
                "norm": float(np.linalg.norm(arr)) if arr.size > 0 else None,
            }
            if label:
                data["label"] = label
            self.log(tag, data, step=step, depth=depth)

        except Exception as e:
            self.log(
                f"{tag}.stats_error",
                {"error": str(e)},
                step=step,
                depth=depth,
            )

    # ── 내부 write ────────────────────────────────────────────────────────────

    def _write(self, record: Dict[str, Any]):
        if self._fh and not self._fh.closed:
            try:
                self._fh.write(json.dumps(record, default=str) + "\n")
            except Exception:
                pass  # write 실패가 실험을 중단시켜서는 안 됨


# ── 편의 함수: 모듈 수준 기본 tracer ─────────────────────────────────────────
_default_tracer: Optional[Tracer] = None


def get_tracer() -> Optional[Tracer]:
    """모듈 수준의 기본 tracer를 반환. 없으면 None."""
    return _default_tracer


def set_default_tracer(tracer: Tracer):
    """모듈 수준의 기본 tracer 설정."""
    global _default_tracer
    _default_tracer = tracer
