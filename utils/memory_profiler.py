"""
Memory profiling utility for MCTS planning.

Tracks GPU and CPU memory usage during MCTS tree search operations.
Helps identify memory leaks and inefficient resource allocation.
"""

import torch
import psutil
import os
import sys
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import gc

@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: float
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_cached_mb: float
    cpu_rss_mb: float  # Resident Set Size
    cpu_vms_mb: float  # Virtual Memory Size
    tensor_count: int
    phase: str = ""
    
    def __str__(self) -> str:
        return (
            f"[GPU: alloc={self.gpu_allocated_mb:.1f}MB, "
            f"reserved={self.gpu_reserved_mb:.1f}MB, "
            f"cached={self.gpu_cached_mb:.1f}MB] "
            f"[CPU: RSS={self.cpu_rss_mb:.1f}MB, VMS={self.cpu_vms_mb:.1f}MB] "
            f"[Tensors: {self.tensor_count}] ({self.phase})"
        )

class MemoryProfiler:
    """Tracks memory usage during MCTS planning."""
    
    def __init__(self, device: torch.device, debug_level: int = 0):
        self.device = device
        self.debug_level = debug_level
        self.snapshots: Dict[str, MemorySnapshot] = {}
        self.peak_memory: Optional[MemorySnapshot] = None
        self.process = psutil.Process(os.getpid())
        
    def _get_gpu_memory_mb(self) -> Tuple[float, float, float]:
        """Get GPU memory stats in MB."""
        if torch.cuda.is_available() and isinstance(self.device, torch.device):
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6
            cached = (reserved - allocated) / 1e6
            return allocated, reserved, cached
        return 0.0, 0.0, 0.0
    
    def _get_cpu_memory_mb(self) -> Tuple[float, float]:
        """Get CPU memory stats in MB."""
        try:
            mem_info = self.process.memory_info()
            return mem_info.rss / 1e6, mem_info.vms / 1e6
        except:
            return 0.0, 0.0
    
    def _count_tensors(self) -> int:
        """Count total tensors in memory."""
        count = 0
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor):
                count += 1
        return count
    
    def snapshot(self, tag: str, phase: str = "") -> MemorySnapshot:
        """Take a memory snapshot and record it."""
        gpu_alloc, gpu_reserved, gpu_cached = self._get_gpu_memory_mb()
        cpu_rss, cpu_vms = self._get_cpu_memory_mb()
        tensor_count = self._count_tensors()
        
        snap = MemorySnapshot(
            timestamp=datetime.now().timestamp(),
            gpu_allocated_mb=gpu_alloc,
            gpu_reserved_mb=gpu_reserved,
            gpu_cached_mb=gpu_cached,
            cpu_rss_mb=cpu_rss,
            cpu_vms_mb=cpu_vms,
            tensor_count=tensor_count,
            phase=phase,
        )
        
        self.snapshots[tag] = snap
        
        if self.peak_memory is None or snap.gpu_allocated_mb > self.peak_memory.gpu_allocated_mb:
            self.peak_memory = snap
        
        if self.debug_level >= 1:
            print(f"[MEM] {tag}: {snap}", file=sys.stderr, flush=True)
        
        return snap
    
    def delta(self, tag_before: str, tag_after: str) -> Dict[str, float]:
        """Compute memory delta between two snapshots."""
        if tag_before not in self.snapshots or tag_after not in self.snapshots:
            return {}
        
        before = self.snapshots[tag_before]
        after = self.snapshots[tag_after]
        
        return {
            "gpu_alloc_delta_mb": after.gpu_allocated_mb - before.gpu_allocated_mb,
            "cpu_rss_delta_mb": after.cpu_rss_mb - before.cpu_rss_mb,
            "tensor_delta": after.tensor_count - before.tensor_count,
        }
    
    def report(self) -> str:
        """Generate a memory usage report."""
        lines = []
        lines.append("\n" + "="*60)
        lines.append("MEMORY PROFILING REPORT")
        lines.append("="*60)
        
        if self.peak_memory:
            lines.append(f"Peak GPU Memory: {self.peak_memory.gpu_allocated_mb:.1f}MB")
            lines.append(f"Peak Memory Phase: {self.peak_memory.phase}")
        
        lines.append("\nAll Snapshots:")
        for tag, snap in sorted(self.snapshots.items(), key=lambda x: x[1].timestamp):
            lines.append(f"  {tag}: {snap}")
        
        lines.append("="*60 + "\n")
        return "\n".join(lines)
    
    def clear(self):
        """Clear snapshots."""
        self.snapshots.clear()
        self.peak_memory = None


# Global profiler instance
_profiler: Optional[MemoryProfiler] = None

def init_profiler(device: torch.device, debug_level: int = 0) -> MemoryProfiler:
    """Initialize the global profiler."""
    global _profiler
    _profiler = MemoryProfiler(device, debug_level)
    return _profiler

def get_profiler() -> Optional[MemoryProfiler]:
    """Get the global profiler instance."""
    return _profiler
