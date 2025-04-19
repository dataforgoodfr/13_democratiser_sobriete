#!/bin/bash
# run_parallel.sh
#
# This script launches 4 parallel processes that each iterate over a different folder.
# For every file in a folder, it runs a python program (which takes the file name as an argument).
# If the python program fails (nonzero exit status), the filename is added to a log file.

# activate .bashrc

# Name of the Python program to be executed. Change it as needed.
PYTHON_SCRIPT="/home/ec2-user/13_democratiser_sobriete/rag_system/pipeline_scripts/historized_ingestion_pipeline.py"

# Log file for failed Python script invocations.
FAILED_LOG="/home/ec2-user/13_democratiser_sobriete/failed_files.txt"

# Clear previous failure log (or create an empty file)
> "$FAILED_LOG"

pwd

# List of folders to process. Modify these folder names as required.
FOLDERS=("test_pdf/folder1") # "test_pdf/folder2" "test_pdf/folder3" "test_pdf/folder4")

# Function to process all files in a given folder.
process_folder() {
    local folder="$1"
    echo "Processing folder: $folder"
    # Iterate over every item in the folder
    for file in "$folder"/*; do
        # Check if it is a regular file
        if [ -f "$file" ]; then
            # Run the Python script on the file.
            # You can add any additional flags or environmental variables if needed.
            python "$PYTHON_SCRIPT" --file-path "$file"
            # Capture the exit code of the python program.
            if [ $? -ne 0 ]; then
                # Append the filename to the failure log if execution fails.
                echo "$file" >> "$FAILED_LOG"
                echo "Failed processing: $file"
            fi
        fi
    done
    echo "Completed folder: $folder"
}

# Process each folder in parallel
for folder in "${FOLDERS[@]}"; do
    if [ -d "$folder" ]; then
        # Launch the folder processing in the background
        process_folder "$folder" &
    else
        echo "Folder $folder does not exist. Skipping..."
    fi
done

# Wait for all background processes to complete
wait

echo "Processing complete. Check $FAILED_LOG for files that failed."
date >> /home/guevel/projects/RAG/13_democratiser_sobriete/date.txt 
