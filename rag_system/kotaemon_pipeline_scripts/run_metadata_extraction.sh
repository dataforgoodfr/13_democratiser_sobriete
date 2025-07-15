#!/bin/bash
# run_metadata_extraction.sh
#
# This script launches 12 parallel processes that each process a different folder.
# Each folder contains PDFs that need metadata extraction and vector ingestion.
# If processing fails, the filename is logged. Successfully processed files are moved to a 'done' folder.

set -e  # Exit on any error

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    echo "🚀 Activating virtual environment..."
    source .venv/bin/activate
fi

# Configuration
PYTHON_SCRIPT="./pipeline_scripts/historized_ingestion_pipeline.py"
BASE_SCRAPING_DIR="${SCRAPING_OUTPUT_DIR:-./scraping_output}"
FAILED_LOG="failed_extractions.txt"
DONE_DIR="$BASE_SCRAPING_DIR/done"

# Create 'done' directory if it doesn't exist
mkdir -p "$DONE_DIR"

# Clear previous failure log
> "$FAILED_LOG"

echo "📊 Starting metadata extraction pipeline..."
echo "📁 Base directory: $BASE_SCRAPING_DIR"
echo "🐍 Python script: $PYTHON_SCRIPT"
echo "📋 Failed log: $FAILED_LOG"
echo "✅ Done directory: $DONE_DIR"

# Check if base directory exists
if [ ! -d "$BASE_SCRAPING_DIR" ]; then
    echo "❌ Error: Base scraping directory $BASE_SCRAPING_DIR does not exist"
    echo "   Make sure to run the scraping pipeline first!"
    exit 1
fi

# Function to process all PDF files in a given folder
process_folder() {
    local folder_id="$1"
    local folder_path="$BASE_SCRAPING_DIR/folder_$(printf "%02d" $folder_id)"
    
    echo "🔄 [Folder $folder_id] Starting processing: $folder_path"
    
    # Check if folder exists
    if [ ! -d "$folder_path" ]; then
        echo "⚠️  [Folder $folder_id] Folder does not exist: $folder_path"
        return 0
    fi
    
    # Count PDF files
    pdf_count=$(find "$folder_path" -name "*.pdf" -type f | wc -l)
    if [ "$pdf_count" -eq 0 ]; then
        echo "📭 [Folder $folder_id] No PDF files found in $folder_path"
        return 0
    fi
    
    echo "📄 [Folder $folder_id] Found $pdf_count PDF files to process"
    
    # Run the Python script on the entire folder
    if python "$PYTHON_SCRIPT" --folder-path "$folder_path"; then
        echo "✅ [Folder $folder_id] Successfully processed $pdf_count files"
        
        # Move processed files to done directory
        local done_folder="$DONE_DIR/folder_$(printf "%02d" $folder_id)"
        mkdir -p "$done_folder"
        
        # Move all PDFs to done folder
        find "$folder_path" -name "*.pdf" -type f -exec mv {} "$done_folder/" \; 2>/dev/null || true
        
        echo "📦 [Folder $folder_id] Moved processed files to $done_folder"
    else
        echo "❌ [Folder $folder_id] Processing failed for folder $folder_path"
        echo "folder_$folder_id" >> "$FAILED_LOG"
    fi
    
    echo "🏁 [Folder $folder_id] Completed processing"
}

# Export the function so background processes can use it
export -f process_folder
export BASE_SCRAPING_DIR PYTHON_SCRIPT FAILED_LOG DONE_DIR

echo ""
echo "🚀 Launching 12 parallel extraction processes..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Process each folder in parallel (0-11)
pids=()
for folder_id in {0..11}; do
    echo "🔄 Starting worker for folder_$(printf "%02d" $folder_id)..."
    process_folder "$folder_id" &
    pids+=($!)
done

echo ""
echo "⏳ Waiting for all 12 processes to complete..."
echo "   You can monitor progress by checking the logs above"
echo "   Or run: tail -f failed_extractions.txt"

# Wait for all background processes to complete
failed_count=0
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    folder_id=$i
    
    if wait "$pid"; then
        echo "✅ Folder $(printf "%02d" $folder_id) worker completed successfully"
    else
        echo "❌ Folder $(printf "%02d" $folder_id) worker failed"
        ((failed_count++))
    fi
done

echo ""
echo "🎯 METADATA EXTRACTION COMPLETED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Summary statistics
total_failed_folders=$(wc -l < "$FAILED_LOG" 2>/dev/null || echo "0")
successful_folders=$((12 - total_failed_folders))

echo "📊 PROCESSING SUMMARY:"
echo "   ✅ Successful folders: $successful_folders/12"
echo "   ❌ Failed folders: $total_failed_folders/12"

if [ "$total_failed_folders" -gt 0 ]; then
    echo "   📋 Failed folders list: $FAILED_LOG"
    echo ""
    echo "🔍 Failed folders:"
    cat "$FAILED_LOG" | while IFS= read -r folder; do
        echo "   • $folder"
    done
fi

# Count total processed files
total_done_files=$(find "$DONE_DIR" -name "*.pdf" -type f 2>/dev/null | wc -l)
echo "   📄 Total files processed: $total_done_files"

echo ""
echo "📅 Completion time: $(date)"

if [ "$failed_count" -eq 0 ]; then
    echo "🎉 All metadata extraction processes completed successfully!"
    exit 0
else
    echo "⚠️  Some processes failed. Check the logs above and $FAILED_LOG for details."
    exit 1
fi 