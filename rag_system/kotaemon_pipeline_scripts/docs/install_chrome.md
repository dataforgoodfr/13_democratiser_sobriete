# 🌐 Chrome/Chromium Installation Guide

Installation instructions for paper scraping dependencies on macOS and Ubuntu.

## 🍎 macOS Installation

### **Option 1: Google Chrome (Recommended)**
```bash
# Install using Homebrew
brew install --cask google-chrome

# Or download manually from:
# https://www.google.com/chrome/
```

### **Option 2: Chromium (Open Source)**
```bash
# Install using Homebrew
brew install --cask chromium

# Or install via Homebrew core
brew install chromium
```

### **Verify Installation (macOS)**
```bash
# Check if Chrome is installed
ls -la "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Or check Chromium
ls -la "/Applications/Chromium.app/Contents/MacOS/Chromium"

# Test Chrome from command line
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --version

# Test Chromium from command line  
"/Applications/Chromium.app/Contents/MacOS/Chromium" --version
```

## 🐧 Ubuntu Installation

### **Option 1: Google Chrome (Recommended)**
```bash
# Download and install Chrome
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
sudo apt update
sudo apt install -y google-chrome-stable
```

### **Option 2: Chromium (Open Source)**
```bash
# Install Chromium (simpler)
sudo apt update
sudo apt install -y chromium-browser

# Or on newer Ubuntu versions
sudo apt install -y chromium
```

### **Option 3: Headless Setup (For Servers)**
```bash
# For headless servers, install Chrome without GUI dependencies
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
sudo apt update
sudo apt install -y google-chrome-stable

# Install additional dependencies for headless operation
sudo apt install -y xvfb
```

### **Verify Installation (Ubuntu)**
```bash
# Check if Chrome is installed
google-chrome --version

# Or check Chromium
chromium-browser --version
# or
chromium --version

# Test headless mode
google-chrome --headless --no-sandbox --dump-dom https://www.google.com > /dev/null
echo "Chrome headless test: $?"  # Should output 0 for success
```

## 🔧 Python Dependencies

### **Install Selenium and WebDriver Manager**
```bash
# Install required Python packages
pip install selenium webdriver-manager requests

# Or add to your requirements.txt
echo "selenium>=4.0.0" >> requirements.txt
echo "webdriver-manager>=3.8.0" >> requirements.txt
echo "requests>=2.25.0" >> requirements.txt

pip install -r requirements.txt
```

## 🧪 Test Your Installation

### **Quick Test Script**
Create a test file `test_chrome.py`:

```python
#!/usr/bin/env python3
"""Test Chrome/Selenium setup"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import sys

def test_chrome_setup():
    """Test if Chrome and Selenium work together"""
    
    print("🧪 Testing Chrome/Selenium setup...")
    
    try:
        # Setup Chrome options
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        
        # Try to setup ChromeDriver automatically
        print("📥 Setting up ChromeDriver...")
        service = Service(ChromeDriverManager().install())
        
        # Create driver
        print("🚀 Starting Chrome...")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Test navigation
        print("🌐 Testing navigation...")
        driver.get("https://www.google.com")
        
        # Check if page loaded
        title = driver.title
        print(f"✅ Success! Page title: {title}")
        
        # Clean up
        driver.quit()
        print("🎉 Chrome/Selenium setup is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Chrome/Selenium test failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   • Make sure Chrome or Chromium is installed")
        print("   • Try: pip install selenium webdriver-manager")
        print("   • On Ubuntu servers, you might need: sudo apt install xvfb")
        return False

if __name__ == "__main__":
    success = test_chrome_setup()
    sys.exit(0 if success else 1)
```

Run the test:
```bash
python test_chrome.py
```

## 🐳 Docker Setup (Alternative)

If you prefer using Docker:

```dockerfile
# Dockerfile for scraping environment
FROM python:3.10-slim

# Install Chrome dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    xvfb

# Install Chrome
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' \
    && apt-get update \
    && apt-get install -y google-chrome-stable

# Install Python dependencies
RUN pip install selenium webdriver-manager requests sqlmodel logfire

# Set working directory
WORKDIR /app

# Copy your scripts
COPY . .

# Run your scraping script
CMD ["python", "scraping/targeted_scraper.py"]
```

## 🚨 Common Issues & Solutions

### **Issue 1: Chrome binary not found**
```bash
# macOS - Check these paths:
ls -la "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
ls -la "/Applications/Chromium.app/Contents/MacOS/Chromium"

# Ubuntu - Check these paths:
which google-chrome
which chromium-browser
which chromium
```

### **Issue 2: ChromeDriver version mismatch**
```bash
# Let webdriver-manager handle it automatically
pip install --upgrade webdriver-manager

# Or manually download ChromeDriver:
# https://chromedriver.chromium.org/downloads
```

### **Issue 3: Permission denied**
```bash
# Make sure Chrome is executable
sudo chmod +x /usr/bin/google-chrome

# Or for Chromium
sudo chmod +x /usr/bin/chromium-browser
```

### **Issue 4: Display issues on headless servers**
```bash
# Install virtual display
sudo apt install xvfb

# Run with virtual display
xvfb-run -a python scraping/targeted_scraper.py
```

## 🎯 Quick Setup Commands

### **macOS Quick Setup**
```bash
# One-liner for macOS
brew install --cask google-chrome && pip install selenium webdriver-manager requests
```

### **Ubuntu Quick Setup**
```bash
# One-liner for Ubuntu
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add - && \
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' && \
sudo apt update && sudo apt install -y google-chrome-stable && \
pip install selenium webdriver-manager requests
```

After installation, test with:
```bash
python cli.py test
``` 