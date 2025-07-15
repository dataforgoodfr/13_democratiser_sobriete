# Define the folder containing the new documents.
NEW_DOCS="./pdf_files"

# Define the source folder from which files will be distributed.
SOURCE_FOLDER="test_pdf_mobility"

# Create the source folder if it doesn't exist.
if [ ! -d "$SOURCE_FOLDER" ]; then
    mkdir -p "$SOURCE_FOLDER"
    echo "Created source folder: $SOURCE_FOLDER"
fi

# Initialize counter.
counter=0

# Loop through the files in the NEW_DOCS folder.
for file in "$NEW_DOCS"/*; do
    if [ -f "$file" ]; then
        mv "$file" "$SOURCE_FOLDER"
        counter=$((counter + 1))
        echo "Moved '$file' to '$SOURCE_FOLDER' ($counter of 500)"
        # Stop once 500 files have been moved.
        if [ $counter -ge 500 ]; then
            break
        fi
    fi
done

echo "Completed moving files. Total files moved: $counter."
