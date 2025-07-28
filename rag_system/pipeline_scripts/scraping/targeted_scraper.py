#!/usr/bin/env python3
"""
Targeted Paper Scraper
Downloads specific papers by OpenAlex ID from database queue
Adapted from the functional parts of extract_openalex.py
"""

import os
import sys
import time
import hashlib
import shutil
import argparse
from pathlib import Path

import logfire
import requests
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from database.models import (
    get_papers_to_scrape, 
    mark_paper_scraped, 
    mark_paper_failed,
    get_session,
    ScrapingQueue
)
from sqlmodel import select

# Configure logfire
LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN", "pylf_v1_us_qTtmbDFpkfhFwzTfZyZrTJcl4C4lC7FhmZ65BgJ7dLDV")
logfire.configure(token=LOGFIRE_TOKEN)

class TargetedPaperScraper:
    """Scraper that downloads specific papers by OpenAlex ID"""
    
    def __init__(self, base_output_dir="./scraping_output", max_wait_time=30):
        self.base_output_dir = Path(base_output_dir)
        self.max_wait_time = max_wait_time
        self.driver = None
        
        # Setup directories
        self.setup_directories()
        
        # Setup temporary download directory
        self.temp_dir = self.base_output_dir / "temp_downloads"
        self.temp_dir.mkdir(exist_ok=True)
        
    def setup_directories(self):
        """Setup 12 folders for distribution"""
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.folder_paths = []
        for i in range(12):
            folder_path = self.base_output_dir / f"folder_{i:02d}"
            folder_path.mkdir(exist_ok=True)
            self.folder_paths.append(folder_path)
            
        logfire.info(f"Setup directories in {self.base_output_dir}")
        
    def get_folder_for_paper(self, openalex_id: str) -> int:
        """Determine target folder using hash distribution"""
        hash_obj = hashlib.md5(openalex_id.encode())
        return int(hash_obj.hexdigest(), 16) % 12
        
    def start_webdriver(self) -> webdriver.Chrome:
        """Start Selenium webdriver - adapted from extract_openalex.py"""
        chrome_options = webdriver.ChromeOptions()
        
        # Setup chrome for various environments
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
        except Exception:
            logfire.info("Default Chrome setup failed. Setting manually")
            
            # Setup chrome for remote/server usage
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--disable-popup-blocking")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            # Try to find Chrome binary
            possible_chrome_paths = [
                "/usr/bin/google-chrome",
                "/usr/bin/chromium-browser", 
                "/snap/bin/chromium",
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            ]
            
            chrome_path = None
            for path in possible_chrome_paths:
                if os.path.exists(path):
                    chrome_path = path
                    break
                    
            if chrome_path:
                chrome_options.binary_location = chrome_path
                
            service = Service()
            driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Configure download preferences to temp directory
        prefs = {
            "download.default_directory": str(self.temp_dir),
            "savefile.default_directory": str(self.temp_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True
        }
        
        # Note: This needs to be set before creating the driver
        # Let's recreate with proper prefs
        driver.quit()
        chrome_options.add_experimental_option('prefs', prefs)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        driver.set_page_load_timeout(30)
        logfire.info("Chrome webdriver started successfully")
        return driver
        
    def clear_temp_directory(self):
        """Clear temporary download directory - from extract_openalex.py"""
        try:
            for file in self.temp_dir.glob("*"):
                if file.is_file():
                    file.unlink()
        except Exception as e:
            logfire.warning(f"Failed to clear temp directory: {e}")
            
    def get_last_downloaded_file_path(self) -> str:
        """Get the last downloaded file - adapted from extract_openalex.py"""
        files_in_dir = list(self.temp_dir.glob("*"))
        if not files_in_dir:
            return None
        return max(files_in_dir, key=lambda f: f.stat().st_ctime)
        
    def get_paper_pdf_url(self, openalex_id: str) -> str:
        """Get PDF URL for a specific OpenAlex paper ID"""
        try:
            # Call OpenAlex API for specific paper
            if not openalex_id.startswith('https://'):
                # Handle both full URLs and just IDs
                if openalex_id.startswith('W'):
                    api_url = f"https://api.openalex.org/works/https://openalex.org/{openalex_id}"
                else:
                    api_url = f"https://api.openalex.org/works/{openalex_id}"
            else:
                api_url = f"https://api.openalex.org/works/{openalex_id}"
            
            logfire.info(f"Calling OpenAlex API: {api_url}")
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            paper_data = response.json()
            
            # Extract PDF URL - adapted from get_urls_to_fetch()
            try:
                pdf_url = paper_data["best_oa_location"]["pdf_url"]
                logfire.info(f"Found PDF URL via best_oa_location: {pdf_url}")
                return pdf_url
            except (TypeError, KeyError):
                try:
                    pdf_url = paper_data["open_access"]["oa_url"]
                    logfire.info(f"Found PDF URL via open_access: {pdf_url}")
                    return pdf_url
                except (TypeError, KeyError):
                    # Look through all locations
                    for location in paper_data.get('locations', []):
                        if location.get('is_oa') and location.get('pdf_url'):
                            pdf_url = location['pdf_url']
                            logfire.info(f"Found PDF URL in locations: {pdf_url}")
                            return pdf_url
                    
                    logfire.warning(f"No PDF URL found for {openalex_id}")
                    return None
                    
        except Exception as e:
            logfire.error(f"Failed to get PDF URL for {openalex_id}: {e}")
            return None
            
    def get_paper_pdf_url_by_doi(self, doi: str) -> str:
        """Get PDF URL by DOI - adapted from scrape_all_urls method"""
        try:
            # Use OpenAlex API filter by DOI - based on search_openalex function
            params = {
                "filter": f"open_access.is_oa:true,doi:{doi}",
                "per-page": 1
            }
            
            api_url = "https://api.openalex.org/works"
            logfire.info(f"Querying OpenAlex by DOI: {doi}")
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                logfire.warning(f"No results found for DOI: {doi}")
                return None
                
            paper_data = results[0]  # Take first result
            
            # Extract PDF URL using same logic as main method
            try:
                pdf_url = paper_data["best_oa_location"]["pdf_url"]
                logfire.info(f"Found PDF URL via DOI best_oa_location: {pdf_url}")
                return pdf_url
            except (TypeError, KeyError):
                try:
                    pdf_url = paper_data["open_access"]["oa_url"]
                    logfire.info(f"Found PDF URL via DOI open_access: {pdf_url}")
                    return pdf_url
                except (TypeError, KeyError):
                    # Look through all locations
                    for location in paper_data.get('locations', []):
                        if location.get('is_oa') and location.get('pdf_url'):
                            pdf_url = location['pdf_url']
                            logfire.info(f"Found PDF URL via DOI in locations: {pdf_url}")
                            return pdf_url
                    
                    logfire.warning(f"No PDF URL found for DOI: {doi}")
                    return None
                    
        except Exception as e:
            logfire.error(f"Failed to get PDF URL by DOI {doi}: {e}")
            return None
            
    def download_pdf(self, url: str, output_file_path: str) -> str:
        """Download PDF from URL - adapted from extract_openalex.py"""
        if not self.driver:
            self.driver = self.start_webdriver()
            
        self.clear_temp_directory()
        
        try:
            logfire.info(f"Downloading PDF from: {url}")
            self.driver.get(url)
            time.sleep(10)
            
            # Check if download started
            if not list(self.temp_dir.glob("*")):
                logfire.warning("Download didn't start")
                return None
                
            # Wait for download to complete
            wait_time = 0
            while wait_time < self.max_wait_time:
                downloaded_file = self.get_last_downloaded_file_path()
                if downloaded_file and downloaded_file.suffix.lower() == '.pdf':
                    # Move to final location
                    shutil.move(str(downloaded_file), output_file_path)
                    logfire.info(f"Successfully downloaded PDF to {output_file_path}")
                    return output_file_path
                    
                time.sleep(10)
                wait_time += 10
                
            logfire.warning(f"Download timeout after {self.max_wait_time}s")
            return None
            
        except WebDriverException as e:
            logfire.error(f"WebDriver error downloading PDF: {e}")
            return None
        except Exception as e:
            logfire.error(f"Error downloading PDF: {e}")
            return None
            
    def scrape_paper(self, openalex_id: str) -> bool:
        """Scrape a single paper by OpenAlex ID with DOI retry logic"""
        try:
            logfire.info(f"Scraping paper: {openalex_id}")
            
            # Determine target folder and output path
            folder_id = self.get_folder_for_paper(openalex_id)
            target_folder = self.folder_paths[folder_id]
            paper_id = openalex_id.split('/')[-1]  # Extract W123456 from URL
            output_file_path = target_folder / f"{paper_id}.pdf"
            
            # First attempt: Get PDF URL from OpenAlex ID
            pdf_url = self.get_paper_pdf_url(openalex_id)
            downloaded_path = None
            
            if pdf_url:
                logfire.info(f"Primary attempt: trying to download from {pdf_url}")
                downloaded_path = self.download_pdf(pdf_url, str(output_file_path))
                
            # If primary attempt failed, try DOI retry
            if not downloaded_path:
                logfire.info(f"Primary scrape failed for {openalex_id}, attempting DOI retry")
                
                # Get DOI from database
                with get_session() as session:
                    stmt = select(ScrapingQueue).where(ScrapingQueue.openalex_id == openalex_id)
                    paper_from_db = session.exec(stmt).first()
                    
                    if paper_from_db and paper_from_db.doi:
                        doi = paper_from_db.doi
                        logfire.info(f"Found DOI for retry: {doi}")
                        
                        # Try to get PDF URL using DOI
                        doi_pdf_url = self.get_paper_pdf_url_by_doi(doi)
                        
                        if doi_pdf_url:
                            logfire.info(f"DOI retry: trying to download from {doi_pdf_url}")
                            downloaded_path = self.download_pdf(doi_pdf_url, str(output_file_path))
                            
                            if downloaded_path:
                                logfire.info(f"Successfully scraped {openalex_id} via DOI retry")
                            else:
                                logfire.warning(f"DOI retry download failed for {openalex_id}")
                        else:
                            logfire.warning(f"No PDF URL found via DOI retry for {openalex_id}")
                    else:
                        logfire.warning(f"No DOI available for retry on {openalex_id}")
            
            # Final result handling
            if downloaded_path:
                # Mark as successful in database
                mark_paper_scraped(openalex_id, downloaded_path, folder_id)
                logfire.info(f"Successfully scraped {openalex_id} to folder {folder_id}")
                return True
            else:
                mark_paper_failed(openalex_id, "Failed to download PDF (tried both OpenAlex ID and DOI)")
                return False
                
        except Exception as e:
            error_msg = f"Error scraping {openalex_id}: {e}"
            logfire.error(error_msg)
            mark_paper_failed(openalex_id, error_msg)
            return False
            
    def close_driver(self):
        """Close webdriver"""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logfire.info("Closed webdriver")
            except Exception as e:
                logfire.warning(f"Error closing driver: {e}")
                
    def scrape_batch(self, batch_size: int = 10) -> dict:
        """Scrape a batch of papers from database queue"""
        stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0
        }
        
        try:
            # Get papers to scrape from database
            papers_to_scrape = get_papers_to_scrape(limit=batch_size)
            
            if not papers_to_scrape:
                logfire.info("No papers found to scrape")
                return stats
                
            logfire.info(f"Found {len(papers_to_scrape)} papers to scrape")
            
            # Process each paper
            for i, paper in enumerate(papers_to_scrape, 1):
                try:
                    print(f"[{i}/{len(papers_to_scrape)}] 🔄 Scraping: {paper.openalex_id}")
                    
                    success = self.scrape_paper(paper.openalex_id)
                    
                    if success:
                        stats["successful"] += 1
                        print("   ✅ Success!")
                    else:
                        stats["failed"] += 1
                        print("   ❌ Failed")
                        
                    stats["processed"] += 1
                    
                    # Small delay between downloads
                    time.sleep(5)
                    
                except Exception as e:
                    stats["failed"] += 1
                    stats["processed"] += 1
                    logfire.error(f"Error processing paper {paper.openalex_id}: {e}")
                    print(f"   ❌ Error: {e}")
                    
            return stats
            
        except Exception as e:
            logfire.error(f"Failed to scrape batch: {e}")
            return stats
        finally:
            self.close_driver()

def main():
    parser = argparse.ArgumentParser(description='Targeted paper scraper by OpenAlex ID')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of papers to scrape')
    parser.add_argument('--output-dir', default='./scraping_output', help='Output directory')
    parser.add_argument('--max-wait-time', type=int, default=30, help='Max wait time for downloads')
    parser.add_argument('--test-paper', help='Test scraping a specific OpenAlex ID')
    
    args = parser.parse_args()
    
    try:
        scraper = TargetedPaperScraper(
            base_output_dir=args.output_dir,
            max_wait_time=args.max_wait_time
        )
        
        if args.test_paper:
            print(f"🧪 Testing single paper: {args.test_paper}")
            success = scraper.scrape_paper(args.test_paper)
            if success:
                print("✅ Test successful!")
            else:
                print("❌ Test failed")
        else:
            print("🚀 Starting targeted paper scraping...")
            stats = scraper.scrape_batch(args.batch_size)
            
            print("\n📊 BATCH RESULTS:")
            print(f"   ✅ Successful: {stats['successful']}")
            print(f"   ❌ Failed: {stats['failed']}")
            print(f"   📋 Total processed: {stats['processed']}")
            
            if stats['successful'] > 0:
                print(f"\n🎉 Successfully scraped {stats['successful']} papers!")
                print(f"📁 Files distributed across folders in: {args.output_dir}")
                print("💡 Next: Run metadata extraction with ../run_metadata_extraction.sh")
            
    except Exception as e:
        print(f"❌ Script failed: {e}")
        logfire.error(f"Main script error: {e}")

if __name__ == "__main__":
    main() 