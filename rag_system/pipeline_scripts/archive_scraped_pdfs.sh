#!/bin/bash
# Archive Scraped PDFs Script
# Collects all PDFs from the 12 distribution folders and creates an archive for server deployment

set -e  # Exit on any error

# Parse command line arguments
KEEP_ORIGINALS=false
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [--keep-originals]"
    echo ""
    echo "Archive all scraped PDFs from distribution folders into a compressed tar.gz file"
    echo ""
    echo "Options:"
    echo "  --keep-originals    Keep original PDFs in folders after archiving (default: remove them)"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                  # Archive and remove original PDFs"
    echo "  $0 --keep-originals # Archive but keep original PDFs"
    exit 0
fi

if [ "$1" = "--keep-originals" ]; then
    KEEP_ORIGINALS=true
    echo "🔒 Keep originals mode: PDFs will NOT be removed after archiving"
fi

# Configuration
SCRAPING_OUTPUT_DIR="./scraping_output"
ARCHIVE_NAME="scraped_pdfs_$(date +%Y%m%d_%H%M%S)"
COMBINED_FOLDER="$ARCHIVE_NAME"
ARCHIVE_FILE="${ARCHIVE_NAME}.tar.gz"

echo "📦 Starting PDF archiving process..."
echo "🔍 Source directory: $SCRAPING_OUTPUT_DIR"
echo "📁 Archive name: $ARCHIVE_FILE"
echo ""

# Check if scraping output directory exists
if [ ! -d "$SCRAPING_OUTPUT_DIR" ]; then
    echo "❌ Error: Scraping output directory not found: $SCRAPING_OUTPUT_DIR"
    echo "💡 Make sure you've run the scraping process first"
    exit 1
fi

# Create combined folder
echo "📁 Creating combined folder: $COMBINED_FOLDER"
mkdir -p "$COMBINED_FOLDER"

# Initialize counters
total_files=0
total_size=0

echo "📋 Collecting PDFs from distribution folders..."

# Collect PDFs from all 12 folders
for i in $(seq -f "%02g" 0 11); do
    folder_path="$SCRAPING_OUTPUT_DIR/folder_$i"
    
    if [ -d "$folder_path" ]; then
        # Count PDF files in this folder
        pdf_count=$(find "$folder_path" -name "*.pdf" -type f | wc -l)
        
        if [ $pdf_count -gt 0 ]; then
            echo "  📂 folder_$i: $pdf_count PDFs"
            
            # Copy all PDFs to combined folder
            find "$folder_path" -name "*.pdf" -type f -exec cp {} "$COMBINED_FOLDER/" \;
            
            # Update counters
            total_files=$((total_files + pdf_count))
        else
            echo "  📂 folder_$i: 0 PDFs (empty)"
        fi
    else
        echo "  📂 folder_$i: not found (skipping)"
    fi
done

# Check if we found any PDFs
if [ $total_files -eq 0 ]; then
    echo ""
    echo "⚠️  No PDF files found in any distribution folder"
    echo "💡 Make sure the scraping process has completed successfully"
    
    # Clean up empty folder
    rmdir "$COMBINED_FOLDER" 2>/dev/null || true
    exit 1
fi

# Calculate total size of collected PDFs
echo ""
echo "📊 Calculating archive size..."
total_size_bytes=$(du -sb "$COMBINED_FOLDER" | cut -f1)
total_size_mb=$((total_size_bytes / 1024 / 1024))

echo "✅ Collection complete:"
echo "   📄 Total PDF files: $total_files"
echo "   💾 Total size: ${total_size_mb} MB"
echo ""

# Create tar.gz archive
echo "🗜️  Creating archive: $ARCHIVE_FILE"
tar -czf "$ARCHIVE_FILE" "$COMBINED_FOLDER"

# Verify archive was created
if [ -f "$ARCHIVE_FILE" ]; then
    archive_size_bytes=$(stat -c%s "$ARCHIVE_FILE" 2>/dev/null || stat -f%z "$ARCHIVE_FILE" 2>/dev/null)
    archive_size_mb=$((archive_size_bytes / 1024 / 1024))
    
    echo "✅ Archive created successfully:"
    echo "   📦 Archive file: $ARCHIVE_FILE"
    echo "   💾 Archive size: ${archive_size_mb} MB"
    echo "   🗜️  Compression ratio: $(echo "scale=1; $archive_size_mb * 100 / $total_size_mb" | bc -l)%"
else
    echo "❌ Error: Failed to create archive"
    exit 1
fi

# Clean up combined folder
echo ""
echo "🧹 Cleaning up temporary folder..."
rm -rf "$COMBINED_FOLDER"

# Remove original PDFs from distribution folders after successful archiving (optional)
if [ "$KEEP_ORIGINALS" = false ]; then
    echo ""
    echo "🗑️  Removing original PDFs from distribution folders..."
    removed_count=0

    for i in $(seq -f "%02g" 0 11); do
        folder_path="$SCRAPING_OUTPUT_DIR/folder_$i"
        
        if [ -d "$folder_path" ]; then
            # Count PDFs before removal
            pdf_count=$(find "$folder_path" -name "*.pdf" -type f | wc -l)
            
            if [ $pdf_count -gt 0 ]; then
                # Remove all PDFs from this folder
                find "$folder_path" -name "*.pdf" -type f -delete
                removed_count=$((removed_count + pdf_count))
                echo "  📂 folder_$i: removed $pdf_count PDFs"
            fi
        fi
    done

    echo ""
    echo "✅ Cleanup complete:"
    echo "   🗑️  Removed $removed_count PDF files from distribution folders"
    echo "   📁 Distribution folders are now empty and ready for next batch"
else
    echo ""
    echo "🔒 Keeping original PDFs in distribution folders (--keep-originals mode)"
fi

echo ""
if [ "$KEEP_ORIGINALS" = false ]; then
    echo "🎉 Archiving and cleanup complete!"
else
    echo "🎉 Archiving complete!"
fi
echo "📦 Archive ready for server deployment: $ARCHIVE_FILE"
echo ""
echo "💡 To extract on server:"
echo "   tar -xzf $ARCHIVE_FILE"
echo ""
echo "💡 To verify archive contents:"
echo "   tar -tzf $ARCHIVE_FILE | head -10"
echo ""
if [ "$KEEP_ORIGINALS" = false ]; then
    echo "⚠️  Note: Original PDFs have been removed from distribution folders"
    echo "   If you need to re-process, extract from archive or re-run scraping"
else
    echo "💡 Note: Original PDFs are still available in distribution folders"
    echo "   Use --keep-originals to preserve files, or run without it to clean up"
fi 