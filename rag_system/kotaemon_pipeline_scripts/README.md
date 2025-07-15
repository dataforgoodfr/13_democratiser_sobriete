# 📄 Paper Processing Pipeline

**Production-ready system for scraping 250k academic papers and extracting metadata**

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
│   ├── README.md             # Detailed documentation
│   ├── USAGE_EXAMPLES.md     # Practical usage examples
│   └── install_chrome.md     # Chrome installation guide
├── 🎯 UNIFIED CLI ENTRY POINT
├── cli.py                    # python cli.py [test|queue|scrape] - unified interface
├── 🔧 CORE PROCESSING
├── historized_ingestion_pipeline.py  # ✏️ Original pipeline (folder mode added)
└── run_metadata_extraction.sh        # Bash script for parallel processing
```

## 🚀 Quick Start

### **Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install typer rich logfire sqlmodel selenium webdriver-manager
```

### **For Developers**

```bash
# 1. Test the system
python cli.py test

# 2. Check queue status
python cli.py queue --stats

# 3. Run scraping
python cli.py scrape --batch-size 10

# 4. Process with metadata extraction
./run_metadata_extraction.sh
```

### **For Production**

```bash
# 1. Populate queue from your database
python cli.py queue --populate --limit 50000

# 2. Run scraping in batches
python cli.py scrape --batch-size 50

# 3. Monitor progress
python cli.py queue --stats

# 4. Run parallel metadata extraction
./run_metadata_extraction.sh
```

## 📖 Detailed Documentation

- **📖 [docs/README.md](docs/README.md)** - Complete documentation
- **🧪 [docs/USAGE_EXAMPLES.md](docs/USAGE_EXAMPLES.md)** - Practical examples and workflows
- **🔧 [docs/install_chrome.md](docs/install_chrome.md)** - Chrome/Selenium setup guide

## 🎯 Key Features

### ✅ **Clean Architecture**
- **Separation of concerns**: scraping, database, docs
- **Modular design**: each component has clear responsibility
- **Developer-friendly**: easy to understand and extend
- **Modern CLI**: Built with Typer and Rich for beautiful, intuitive interface

### ✅ **Production Ready**
- **Targeted scraping**: downloads specific papers by OpenAlex ID
- **Queue management**: comprehensive database tracking
- **Error handling**: robust failure recovery and retry logic
- **Monitoring**: detailed statistics and progress tracking

### ✅ **Scalable Processing**
- **12x parallelization**: processes 12 folders simultaneously
- **Batch processing**: configurable batch sizes for optimal throughput
- **Resumable**: can be interrupted and resumed safely

## 🛠️ Development

### **Working with CLI**

```bash
# Get help (shows all commands with rich formatting)
python cli.py --help

# Command-specific help
python cli.py scrape --help
python cli.py queue --help

# Common usage
python cli.py scrape --batch-size 20
python cli.py queue --stats
python cli.py test

# Or work directly with modules (advanced)
python -m scraping.targeted_scraper --batch-size 20
python -m database.manage_queue --stats
python -m scraping.test_scraping
```

### **Code Organization**

- **`scraping/`**: Everything related to downloading papers
- **`database/`**: Database models, queue management, statistics
- **`docs/`**: All documentation and guides
- **Root level**: Entry points and core processing scripts

### **Adding New Features**

1. **Scraping features**: Add to `scraping/` directory
2. **Database features**: Add to `database/` directory
3. **Documentation**: Update files in `docs/`
4. **CLI commands**: Add new subcommands to `cli.py`

## ⚡ Performance

- **Original**: ~2,500 hours for 250k papers (sequential)
- **Optimized**: ~200-250 hours (12x parallel processing)
- **Throughput**: ~1,000-1,250 papers/hour with parallel processing

## 🎉 Summary

This refactored system provides:

- ✅ **Clean, organized codebase** with proper separation of concerns
- ✅ **Production-ready components** for large-scale processing
- ✅ **Comprehensive documentation** for developers and users
- ✅ **Modern Typer CLI** with beautiful Rich formatting and auto-generated help
- ✅ **Type-safe arguments** with automatic validation and intuitive interface
- ✅ **Robust error handling** and monitoring capabilities
- ✅ **Scalable architecture** ready for 250k documents

**Beautiful, developer-friendly CLI ready for production use! 🚀** 