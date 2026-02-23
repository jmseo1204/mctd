#!/bin/bash

# Start run_jobs.py in background
python3 run_jobs.py > /tmp/run_jobs.log 2>&1 &
RUN_JOBS_PID=$!
echo "[$(date)] Started run_jobs.py with PID: $RUN_JOBS_PID"

# Monitor for 5 minutes (300 seconds), checking every 20 seconds
for i in {1..15}; do
    sleep 20
    echo ""
    echo "=== [$(date)] 20-second checkpoint $i ==="
    
    # Check if run_jobs is still running
    if ! kill -0 $RUN_JOBS_PID 2>/dev/null; then
        echo "run_jobs.py has finished!"
        break
    fi
    
    # Check for running docker containers
    echo "[Docker Status]"
    docker ps --filter "name=exp_gpu" --format "table {{.Names}}\t{{.Status}}" 2>/dev/null || echo "No containers running"
    
    # Check latest logs
    if [ -f "logs/"*.log ]; then
        LATEST_LOG=$(ls -t logs/*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            echo "[Latest log tail from $LATEST_LOG]"
            tail -20 "$LATEST_LOG" | tail -10
        fi
    fi
    
    # Check for errors
    if grep -q "Error\|Traceback\|failed\|ERROR" /tmp/run_jobs.log 2>/dev/null; then
        echo "⚠️  ERROR detected! Showing relevant lines:"
        grep -i "error\|traceback\|failed" /tmp/run_jobs.log | tail -5
    fi
done

echo ""
echo "=== Final Status ==="
wait $RUN_JOBS_PID
EXIT_CODE=$?
echo "run_jobs.py exited with code: $EXIT_CODE"
