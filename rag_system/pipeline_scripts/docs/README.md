# 📄 Paper Processing Pipeline

**Safe refactoring for 250k document processing with targeted scraping and parallel metadata extraction**

This is a well-organized, production-ready system for scraping academic papers from OpenAlex and extracting metadata using LLM-powered processing.

## 📁 Project Structure

```
pipeline_scripts/
├── 📂 scraping/              # Paper scraping components
│   ├── targeted_scraper.py   # ⭐ Main scraper (targets specific OpenAlex IDs)
│   ├── test_scraping.py      # Test scraping with 10 papers
│   └── __init__.py
├── 📂 database/              # Database models and management  
│   ├── models.py             # SQLModel classes and operations
│   ├── manage_queue.py       # ⭐ Central queue management CLI
│   └── __init__.py
├── 📂 docs/                  # Documentation
│   ├── README.md             # This file
│   ├── USAGE_EXAMPLES.md     # Detailed usage examples
│   └── install_chrome.md     # Chrome installation guide
├── historized_ingestion_pipeline.py  # ✏️ Original pipeline (folder mode added)
└── run_metadata_extraction.sh        # Bash script for parallel processing
```

## 🚀 Quick Start

### **Step 0: Install Chrome (REQUIRED for scraping)**
```bash
# macOS
brew install --cask google-chrome && pip install selenium webdriver-manager

# Ubuntu  
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add - && \
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' && \
sudo apt update && sudo apt install -y google-chrome-stable && \
pip install selenium webdriver-manager

# Install CLI dependencies
pip install typer[all] rich

# Test Chrome setup (optional but recommended)
python cli.py scrape --test-paper "https://openalex.org/W2741809807"
```

### **Step 1: Explore the CLI (RECOMMENDED)**
```bash
# See beautiful help with all commands
python cli.py --help

# Get detailed help for specific commands
python cli.py scrape --help
python cli.py queue --help
```

### **Step 2: Test with 10 Papers (RECOMMENDED)**
```bash
# Test the complete pipeline with just 10 papers
python cli.py test

# This will:
# 1. Create test database queue
# 2. Test scraping functionality  
# 3. Create folder structure
# 4. Show you what to expect
```

### **Step 3: Test Single File (Original Mode)**
```bash
# This still works exactly as before
python historized_ingestion_pipeline.py --file-path path/to/your.pdf
```

### **Step 4: Test Folder Mode (New)**
```bash
# New: Process entire folder
python historized_ingestion_pipeline.py --folder-path ./test_scraping_output/folder_00
```

### **Step 5: Production Scraping (When Ready)**
```bash
# Use the new targeted scraper (RECOMMENDED)
python cli.py scrape --batch-size 10

# Or test a specific paper first
python cli.py scrape --test-paper "https://openalex.org/W2741809807"

# Check progress
python cli.py queue --stats
```

### **Step 6: Run 12 Parallel Processes**
```bash
# Make executable (one time)  
chmod +x run_metadata_extraction.sh

# Run on all 12 folders
./run_metadata_extraction.sh
```

## 🎯 Components Overview

### 0. **cli.py** (NEW - Modern CLI Interface)
- 🎨 **Built with Typer and Rich** for beautiful, colored output
- 📖 **Auto-generated help** with examples and rich formatting
- 🔧 **Type-safe arguments** with automatic validation
- 💡 **Intuitive command structure** with subcommands

### 1. **scraping/targeted_scraper.py** (NEW - Main Scraper)
- 🎯 **Downloads specific papers by OpenAlex ID** from database queue
- 🌐 **Calls OpenAlex API directly** for each paper (no random search)
- 🔄 **DOI retry logic**: If OpenAlex ID fails, automatically retries with DOI
- 📁 **Distributes papers to 12 folders automatically**
- 🗄️ **Tracks progress and retry attempts in database**
- 🛡️ **Batch processing** with proper error handling and resilience

### 2. **scraping/test_scraping.py** (Test Suite)
- 🧪 **Tests with 10 papers** before full-scale processing
- 📁 **Creates folder structure** and shows expected results
- 🗄️ **Populates test database queue**
- 📊 **Shows expected results**

### 3. **database/manage_queue.py** (NEW - Queue Management CLI)
- 🧹 **Clear/reset** scraping queue for testing
- 📥 **Populate queue** from policies_abstracts_all table
- 📊 **Comprehensive statistics** and progress tracking
- 🔍 **Show failed papers** with error details
- ✅ **Show recent successes** 
- 🎛️ **Central CLI** for all queue operations

### 4. **database/models.py** (Database Support)
- 📥 **Simple queue tracking** for papers to scrape
- 🗄️ **Basic database models** (much simpler than complex version)
- 📊 **Statistics and progress tracking**

### 5. **historized_ingestion_pipeline.py** (✏️ Modified Original)
- 🔄 **Added `--folder-path` mode** for batch processing
- ✅ **Maintains backward compatibility** (single file mode still works)
- 🗄️ **Unchanged core logic** for metadata extraction

### 6. **run_metadata_extraction.sh** (Parallel Processing)
- 🔄 **Runs 12 parallel processes** for maximum throughput
- 📁 **Each process handles one folder**
- 📊 **Progress tracking and error logging**

## 🗄️ Queue Management

The unified CLI provides comprehensive database queue management:

### **Clean Up Test Data**
```bash
# Clear entire queue (for fresh start)
python cli.py queue --clear

# Reset failed entries to try again
python cli.py queue --reset-failed
```

### **Monitor Progress**
```bash
# Show comprehensive statistics
python cli.py queue --stats

# Show recent failures with error messages
python cli.py queue --show-failed --limit 10

# Show recent successes
python cli.py queue --show-successes --limit 10
```

### **Example Output**
```
📊 SCRAPING QUEUE STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 Total in queue: 1,250
✅ Successfully scraped: 1,180
❌ Failed: 45
⏳ Pending: 25
📈 Completion rate: 94.4%
📉 Failure rate: 3.6%

📁 FOLDER DISTRIBUTION:
   📂 folder_00: 98 papers
   📂 folder_01: 102 papers
   📂 folder_02: 95 papers
   ... (showing distribution across 12 folders)
   📊 Total distributed: 1,180 papers
```

## 🛠️ Troubleshooting

### **Chrome/Selenium Issues**
```bash
# Test Chrome setup first
python -c "from selenium import webdriver; print('Selenium OK')"

# If Chrome not found, install it:
# macOS:
brew install --cask google-chrome

# Ubuntu:
sudo apt install -y google-chrome-stable

# Install Python dependencies
pip install selenium webdriver-manager requests
```

### **Script Won't Run**
```bash
# Make executable
chmod +x run_metadata_extraction.sh

# Check paths
ls -la historized_ingestion_pipeline.py
```

### **Database Connection Issues**
```bash
# Test database connection
python cli.py queue --stats

# Check environment variables
echo $DATABASE_URL
```

### **No PDFs Found**
```bash
# Check folder structure
ls -la scraping_output/
find scraping_output/ -name "*.pdf" | head -10
```

### **Individual Folder Testing**
```bash
# Test single folder first
python historized_ingestion_pipeline.py --folder-path ./scraping_output/folder_00
```

### **Check Logs**
```bash
# Monitor progress
tail -f failed_extractions.txt

# Check for specific errors
grep "Error" failed_extractions.txt
```

## 🎉 Summary

This is a **minimal, safe refactoring** that:

- ✅ **Keeps your working pipeline intact**
- ✅ **Adds folder processing capability**  
- ✅ **Enables 12x parallelization**
- ✅ **Improves monitoring and error handling**
- ✅ **Maintains backward compatibility**

**Total changes**: 
- ✏️  **1 file modified**: `historized_ingestion_pipeline.py` (added folder mode)
- 📄 **8 files added**: 
  - `scraping/targeted_scraper.py` (⭐ NEW - targeted production scraper)
  - `scraping/test_scraping.py` (test with 10 papers)
  - `database/manage_queue.py` (⭐ NEW - queue management CLI)
  - `database/models.py` (database support)
  - `run_metadata_extraction.sh` (parallel processing)
  - `docs/install_chrome.md` (Chrome installation guide)
  - `docs/USAGE_EXAMPLES.md` (detailed usage examples & workflows)
  - `docs/README.md` (this documentation)
- 🚀 **Ready to process 250k documents safely!**

## 🎯 Complete Workflow for 250k Documents

```bash
# 1. Setup Chrome/Selenium (one-time)
# See install_chrome.md for detailed instructions

# 2. Test everything with 10 papers first
python cli.py test

# 3. If test works, populate production queue
python cli.py queue --populate --limit 250000

# 4. Run targeted scraping (repeat as needed)
python cli.py scrape --batch-size 50

# 5. Check scraping progress
python cli.py queue --stats

# 6. When enough papers are scraped, run extraction
./run_metadata_extraction.sh

# 7. Monitor extraction progress
tail -f failed_extractions.txt
```

The original approach still works, but now you can test safely and process folders in parallel for much better throughput.

## 📚 Additional Resources

- **📖 [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** - Detailed workflows and practical examples
- **🔧 [install_chrome.md](install_chrome.md)** - Chrome installation guide for all platforms  
- **🧪 Testing**: Start with `python cli.py test` for a safe environment
- **🗄️ Queue Management**: Use `python cli.py queue --help` for all database operations
- **🆘 Need help?** Check the troubleshooting section above or the usage examples 