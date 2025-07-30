# 📖 Usage Examples

Practical examples for using the unified CLI for paper processing.

## 🧪 Testing & Development

### **Clean Start for Testing**
```bash
# 1. Clear any previous test data
python cli.py queue --clear --confirm

# 2. Verify queue is empty
python cli.py queue --stats

# 3. Add a small test batch
python cli.py queue --populate --limit 10

# 4. Test the scraping
python cli.py scrape --batch-size 5

# 5. Check results
python cli.py queue --stats
python cli.py queue --show-successes
```

### **Test Specific Paper**
```bash
# Test scraping a single known paper
python cli.py scrape --test-paper "https://openalex.org/W2741809807"

# Check if it was added to the right folder
ls -la scraping_output/folder_*/W2741809807.pdf
```

### **Quick Help**
```bash
# Show all available commands (beautiful rich output)
python cli.py --help

# Show help for specific commands (with examples and colored output)
python cli.py test --help
python cli.py queue --help
python cli.py scrape --help

# Version information
python cli.py --version
```

## 🚀 Production Workflows

### **Initial Production Setup**
```bash
# 1. Clear any test data
python cli.py queue --clear --confirm

# 2. Populate with production papers (adjust limit as needed)
python cli.py queue --populate --limit 50000

# 3. Check what was added
python cli.py queue --stats

# 4. Start scraping in manageable batches
python cli.py scrape --batch-size 20
```

### **Monitoring Production Progress**
```bash
# Check overall progress
python cli.py queue --stats

# See what's failing
python cli.py queue --show-failed --limit 20

# See recent successes
python cli.py queue --show-successes --limit 10

# Continue scraping if needed
python cli.py scrape --batch-size 25
```

### **Handling Failures**
```bash
# Look at failed papers in detail
python cli.py queue --show-failed --limit 50

# Reset failed papers to try again (after fixing issues)
python cli.py queue --reset-failed

# Try scraping the reset papers
python cli.py scrape --batch-size 10
```

## 🔄 Continuous Processing

### **Daily Production Run**
Create a script like `daily_scraping.sh`:

```bash
#!/bin/bash
# daily_scraping.sh - Run this daily for continuous processing

echo "📅 $(date): Starting daily scraping run"

# Check current status
python cli.py queue --stats

# Scrape a batch
python cli.py scrape --batch-size 50

# Show results
echo "📊 After scraping:"
python cli.py queue --stats

# Check if enough papers for metadata extraction
scraped_count=$(python cli.py queue --stats | grep "Successfully scraped" | grep -o '[0-9,]*' | tr -d ',')

if [ "$scraped_count" -gt 500 ]; then
    echo "🚀 Ready for metadata extraction! Run: ./run_metadata_extraction.sh"
fi

echo "✅ Daily scraping completed at $(date)"
```

### **Recovery from Interruption**
```bash
# If scraping was interrupted, just continue
python cli.py queue --stats  # See where you left off
python cli.py scrape --batch-size 30  # Continue scraping

# Reset any papers that might be stuck
python cli.py queue --reset-failed
```

## 📊 Analysis & Monitoring

### **Success Rate Analysis**
```bash
# Get detailed statistics
python cli.py queue --stats

# Look at failure patterns
python cli.py queue --show-failed --limit 100 > failures.txt

# Count common error types
grep "No PDF URL found" failures.txt | wc -l
grep "Download timeout" failures.txt | wc -l
grep "WebDriver error" failures.txt | wc -l
```

### **Folder Balance Check**
```bash
# Check if papers are distributed evenly across folders
python cli.py queue --stats | grep "folder_"

# Count actual files in each folder
for i in {00..11}; do
    count=$(ls scraping_output/folder_$i/*.pdf 2>/dev/null | wc -l)
    echo "Folder $i: $count files"
done
```

## 🏭 Large Scale Processing

### **Process 250k Papers (Example)**
```bash
# 1. Initial setup
python cli.py queue --clear --confirm
python cli.py queue --populate --limit 250000

# 2. Create a processing loop
cat > process_large_batch.sh << 'EOF'
#!/bin/bash
for batch in {1..1000}; do
    echo "=== Batch $batch ==="
    
    # Scrape 50 papers
    python cli.py scrape --batch-size 50
    
    # Check progress every 10 batches
    if [ $((batch % 10)) -eq 0 ]; then
        echo "Progress after $batch batches:"
        python cli.py queue --stats
        
        # Run metadata extraction if we have enough papers
        scraped=$(python cli.py queue --stats | grep "Successfully scraped" | grep -o '[0-9,]*' | tr -d ',')
        if [ "$scraped" -gt 1000 ]; then
            echo "Running metadata extraction..."
            ./run_metadata_extraction.sh &
        fi
    fi
    
    # Small delay to avoid overwhelming the servers
    sleep 10
done
EOF

chmod +x process_large_batch.sh
./process_large_batch.sh
```

### **Monitor Long-Running Process**
```bash
# Create a monitoring script
cat > monitor_progress.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "📊 SCRAPING MONITOR - $(date)"
    echo "================================"
    python cli.py queue --stats
    echo ""
    echo "🔄 Recent activity:"
    python cli.py queue --show-successes --limit 5
    echo ""
    echo "❌ Recent failures:"
    python cli.py queue --show-failed --limit 3
    
    sleep 60  # Update every minute
done
EOF

chmod +x monitor_progress.sh
./monitor_progress.sh
```

## 🚨 Emergency Procedures

### **Stop All Processing**
```bash
# Kill any running scraping processes
pkill -f "cli.py scrape"
pkill -f "run_metadata_extraction.sh"

# Check what's still running
ps aux | grep -E "(cli.py|historized_ingestion)"
```

### **Reset Everything**
```bash
# Clear database queue
python cli.py queue --clear --confirm

# Remove downloaded files
rm -rf scraping_output/

# Verify clean state
python cli.py queue --stats
ls -la scraping_output/ 2>/dev/null || echo "Directory doesn't exist (good)"
```

### **Database Connection Issues**
```bash
# Test database connection
python cli.py queue --stats

# Check environment variables
echo "DATABASE_URL: ${DATABASE_URL:0:50}..."  # Show first 50 chars
echo "LOGFIRE_TOKEN: ${LOGFIRE_TOKEN:0:20}..."  # Show first 20 chars
```

## 💡 Tips & Best Practices

### **CLI Command Structure**
```bash
# General pattern
python cli.py [COMMAND] [OPTIONS]

# Commands:
#   test     - Run test scraping with 10 papers
#   queue    - Manage scraping queue database
#   scrape   - Run targeted paper scraping

# Get help for any command
python cli.py [COMMAND] --help
```

### **Optimal Batch Sizes**
- **Testing**: 5-10 papers
- **Development**: 10-20 papers  
- **Production**: 20-50 papers
- **Large scale**: 50-100 papers (with monitoring)

### **Error Handling Strategy**
1. Run small batches first to identify common issues
2. Use `python cli.py queue --show-failed` to understand failure patterns
3. Fix infrastructure issues (Chrome, network, etc.)
4. Use `python cli.py queue --reset-failed` to retry failed papers
5. Monitor completion rates and adjust batch sizes

### **Resource Management**
- Monitor disk space in `scraping_output/`
- Check Chrome memory usage during long runs
- Use `--max-wait-time` to prevent hanging downloads
- Run metadata extraction in parallel once you have 500+ papers

### **Development Workflow**
```bash
# 1. Always test first
python cli.py test

# 2. Start with small batches
python cli.py queue --populate --limit 100
python cli.py scrape --batch-size 10

# 3. Monitor and adjust
python cli.py queue --stats
python cli.py queue --show-failed

# 4. Scale up gradually
python cli.py scrape --batch-size 25
python cli.py scrape --batch-size 50
``` 