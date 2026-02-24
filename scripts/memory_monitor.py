#!/usr/bin/env python3
"""
Background memory monitoring script.
Logs memory usage every 5 seconds.
"""
import subprocess
import time
import os
from datetime import datetime

def get_memory_stats():
    """Get memory stats from /proc/meminfo and ps."""
    try:
        # Get system memory
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                key, value = line.split(':')
                meminfo[key.strip()] = int(value.split()[0])
        
        mem_total = meminfo.get('MemTotal', 0)
        mem_avail = meminfo.get('MemAvailable', 0)
        mem_used = mem_total - mem_avail
        mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'mem_total_mb': mem_total // 1024,
            'mem_used_mb': mem_used // 1024,
            'mem_avail_mb': mem_avail // 1024,
            'mem_percent': mem_percent,
        }
    except Exception as e:
        return {'timestamp': datetime.now().isoformat(), 'error': str(e)}

def main():
    log_path = 'logs/memory_monitor.log'
    os.makedirs('logs', exist_ok=True)
    
    print(f"[MemoryMonitor] Logging to {log_path}")
    
    with open(log_path, 'w') as f:
        f.write("timestamp,mem_total_mb,mem_used_mb,mem_avail_mb,mem_percent\n")
        f.flush()
        
        while True:
            try:
                stats = get_memory_stats()
                if 'error' not in stats:
                    line = f"{stats['timestamp']},{stats['mem_total_mb']},{stats['mem_used_mb']},{stats['mem_avail_mb']},{stats['mem_percent']:.1f}\n"
                    f.write(line)
                    f.flush()
                time.sleep(5)
            except KeyboardInterrupt:
                print("[MemoryMonitor] Stopped")
                break
            except Exception as e:
                print(f"[MemoryMonitor] Error: {e}")
                time.sleep(5)

if __name__ == '__main__':
    main()
