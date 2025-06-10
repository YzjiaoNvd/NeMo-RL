#!/bin/bash

# Script to check SLURM job status for DCP to HF conversion
OUTPUT_BASE_DIR="$1"

if [ -z "$OUTPUT_BASE_DIR" ]; then
    echo "Usage: $0 <output_base_dir>"
    echo "Example: $0 /path/to/checkpoint/dir/HF"
    exit 1
fi

echo "=== CHECKING SLURM JOB STATUS ==="
echo "Output directory: $OUTPUT_BASE_DIR"
echo ""

# Check if logs directory exists
LOGS_DIR="$OUTPUT_BASE_DIR/logs"
if [ ! -d "$LOGS_DIR" ]; then
    echo "❌ Logs directory not found: $LOGS_DIR"
    echo "This suggests no jobs were submitted or the output directory is wrong."
    exit 1
fi

echo "✅ Logs directory found: $LOGS_DIR"
echo ""

# Count log files
out_files=($(find "$LOGS_DIR" -name "step_*.out" 2>/dev/null))
err_files=($(find "$LOGS_DIR" -name "step_*.err" 2>/dev/null))

echo "📊 Job Statistics:"
echo "  Output log files: ${#out_files[@]}"
echo "  Error log files: ${#err_files[@]}"
echo ""

# Check current SLURM jobs
echo "🔍 Current SLURM jobs (dcp2hf):"
current_jobs=$(squeue -u $USER --name=dcp2hf* --format="%.10i %.15j %.8T %.10M %.6D %R" --noheader 2>/dev/null)
if [ -z "$current_jobs" ]; then
    echo "  No active dcp2hf jobs found"
else
    echo "  JobID      Name           State    Time   Nodes Reason"
    echo "  $current_jobs"
fi
echo ""

# Check recent completed jobs
echo "📋 Recent completed jobs (last 24h):"
recent_jobs=$(sacct -u $USER --starttime=$(date -d '24 hours ago' +%Y-%m-%d) --format="JobID,JobName,State,ExitCode,Start,End" --name=dcp2hf* --noheader 2>/dev/null)
if [ -z "$recent_jobs" ]; then
    echo "  No recent dcp2hf jobs found"
else
    echo "  $recent_jobs"
fi
echo ""

# Analyze log files
if [ ${#out_files[@]} -gt 0 ]; then
    echo "📝 Analyzing log files:"
    
    successful_jobs=0
    failed_jobs=0
    
    for out_file in "${out_files[@]}"; do
        step_name=$(basename "$out_file" .out)
        err_file="$LOGS_DIR/${step_name}.err"
        
        echo ""
        echo "  📄 $step_name:"
        
        # Check if job completed successfully
        if grep -q "Conversion successful" "$out_file" 2>/dev/null; then
            echo "    ✅ SUCCESS"
            ((successful_jobs++))
        elif grep -q -i "error\|failed\|exception" "$out_file" 2>/dev/null; then
            echo "    ❌ FAILED (check output log)"
            ((failed_jobs++))
            echo "    Last few lines of output:"
            tail -5 "$out_file" | sed 's/^/      /'
        elif [ -f "$err_file" ] && [ -s "$err_file" ]; then
            echo "    ❌ FAILED (check error log)"
            ((failed_jobs++))
            echo "    Error log contents:"
            tail -5 "$err_file" | sed 's/^/      /'
        elif [ -f "$out_file" ] && [ -s "$out_file" ]; then
            echo "    🔄 IN PROGRESS or UNKNOWN"
            echo "    Last few lines:"
            tail -3 "$out_file" | sed 's/^/      /'
        else
            echo "    ⚠️  Empty or missing log file"
        fi
    done
    
    echo ""
    echo "📊 Summary:"
    echo "  Successful jobs: $successful_jobs"
    echo "  Failed jobs: $failed_jobs"
    echo "  Total jobs: ${#out_files[@]}"
    
else
    echo "❌ No log files found. Jobs may not have been submitted or are still queued."
fi

# Check output directories
echo ""
echo "🎯 Checking output directories:"
output_dirs=($(find "$OUTPUT_BASE_DIR" -maxdepth 1 -name "step_*" -type d 2>/dev/null))

if [ ${#output_dirs[@]} -eq 0 ]; then
    echo "  ❌ No step_* output directories found"
    echo "  This confirms no successful conversions yet"
else
    echo "  ✅ Found ${#output_dirs[@]} output directories:"
    for dir in "${output_dirs[@]}"; do
        step_name=$(basename "$dir")
        file_count=$(find "$dir" -type f | wc -l)
        echo "    - $step_name ($file_count files)"
        
        # Check for key HF files
        if [ -f "$dir/config.json" ] && [ -f "$dir/pytorch_model.bin" ]; then
            echo "      ✅ Valid HuggingFace checkpoint"
        else
            echo "      ⚠️  Incomplete checkpoint"
        fi
    done
fi

echo ""
echo "=== DEBUGGING SUGGESTIONS ==="
if [ ${#out_files[@]} -eq 0 ]; then
    echo "1. Check if jobs were actually submitted - run the conversion script with -v for verbose output"
    echo "2. Verify SLURM account and partition settings"
    echo "3. Check if container image exists: $NEMO_RL_CONTAINER"
elif [ $failed_jobs -gt 0 ]; then
    echo "1. Review error logs in: $LOGS_DIR"
    echo "2. Check container permissions and mount points"
    echo "3. Verify the convert_dcp_to_hf.py script exists and is executable"
    echo "4. Check if the checkpoint files are accessible from compute nodes"
elif [ ${#output_dirs[@]} -eq 0 ]; then
    echo "1. Jobs may still be running - check 'squeue -u \$USER'"
    echo "2. Review output logs for any errors or warnings"
    echo "3. Check disk space in output directory"
else
    echo "✅ Everything looks good! Check individual step directories for converted models."
fi