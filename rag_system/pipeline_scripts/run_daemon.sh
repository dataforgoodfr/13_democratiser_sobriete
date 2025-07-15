#!/bin/bash

# Path to the parallel processing script
PARALLEL_SCRIPT="/home/ec2-user/13_democratiser_sobriete/rag_system/pipeline_scripts/run_parallel.sh"

# Log file for the daemon
DAEMON_LOG="/home/ec2-user/13_democratiser_sobriete/daemon.log"

# Function to check if the ingestion process is running
is_process_running() {
    pgrep -f "$PARALLEL_SCRIPT" > /dev/null
}

# Function to log messages with timestamp
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$DAEMON_LOG"
}

# Main daemon loop
log_message "Starting ingestion daemon monitor..."

while true; do
    if ! is_process_running; then
        log_message "Ingestion process not running. Relaunching..."
        # Run the parallel processing script in the background
        nohup bash "$PARALLEL_SCRIPT" > /dev/null 2>&1 &
        log_message "Ingestion process relaunched with PID $!"
    else
        log_message "Ingestion process is running"
    fi
    
    # Sleep for 30 seconds before checking again
    sleep 120
done 