#!/bin/bash
# distribute_files.sh
#
# This script distributes files from the "test_pdf" folder evenly into 4 target folders.
# It uses a round-robin algorithm to assign each file to one of the folders.
# The target folders are created if they do not already exist.

# Define the source folder containing the files to distribute.
SOURCE_FOLDER="test_pdf_mobility"

# Define the target folders.
TARGET_FOLDERS=("folder1" "folder2" "folder3" "folder4")

# Create target folders if they do not exist.
for folder in "${TARGET_FOLDERS[@]}"; do
    if [ ! -d "$folder" ]; then
        mkdir -p "$folder"
        echo "Created folder: $folder"
    fi
done

echo "Distributing files from '$SOURCE_FOLDER' to target folders..."

# Initialize a counter to distribute files in a round-robin fashion.
count=0
for file in "$SOURCE_FOLDER"/*; do
    if [ -f "$file" ]; then
        index=$(( count % ${#TARGET_FOLDERS[@]} ))
        dest_folder="${TARGET_FOLDERS[$index]}"
        # Move the file from the source folder to the target folder.
        mv "$file" "$dest_folder"
        echo "Moved '$file' to '$dest_folder'"
        count=$(( count + 1 ))
    fi
done

echo "File distribution complete."
