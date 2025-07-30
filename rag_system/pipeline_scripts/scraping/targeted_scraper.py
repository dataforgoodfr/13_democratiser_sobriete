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
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading

import logfire
import requests
from tqdm import tqdm
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
    get_paper_doi
)

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
        """Start Selenium webdriver with proper download configuration"""
        chrome_options = webdriver.ChromeOptions()
        
        # Configure download preferences FIRST (before creating driver)
        prefs = {
            "download.default_directory": str(self.temp_dir.absolute()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True,
            "profile.default_content_settings.popups": 0,
            "profile.default_content_setting_values.automatic_downloads": 1
        }
        chrome_options.add_experimental_option('prefs', prefs)
        
        # Add arguments for better PDF handling
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-default-apps")
        chrome_options.add_argument("--no-first-run")
        
        # For laptop (non-headless) - better performance
        # chrome_options.add_argument("--headless=new")  # Commented out for laptop use
        
        # Enable logging for debugging downloads
        chrome_options.add_argument("--enable-logging")
        chrome_options.add_argument("--v=1")
        
        try:
            # Try automatic setup first
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            logfire.info("Chrome webdriver started with automatic setup")
        except Exception as e:
            logfire.info(f"Automatic Chrome setup failed: {e}. Trying manual setup.")
            
            # Try to find Chrome binary manually
            possible_chrome_paths = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",  # macOS
                "/usr/bin/google-chrome",  # Linux
                "/usr/bin/chromium-browser", 
                "/snap/bin/chromium"
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
            logfire.info("Chrome webdriver started with manual setup")
        
        driver.set_page_load_timeout(60)
        
        # Verify download directory is working
        logfire.info(f"Chrome configured to download to: {self.temp_dir.absolute()}")
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
        """Get the last downloaded file"""
        try:
            files_in_dir = list(self.temp_dir.glob("*"))
            if not files_in_dir:
                return None
            
            # Filter out partial downloads and get PDF files only
            pdf_files = [f for f in files_in_dir if f.suffix.lower() == '.pdf' and f.stat().st_size > 0]
            if pdf_files:
                return max(pdf_files, key=lambda f: f.stat().st_ctime)
            
            # If no PDFs, return the newest file (might be still downloading)
            return max(files_in_dir, key=lambda f: f.stat().st_ctime)
        except Exception as e:
            logfire.warning(f"Error getting last downloaded file: {e}")
            return None
        
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
        """Download PDF from URL with proper file handling"""
        if not self.driver:
            self.driver = self.start_webdriver()
            
        self.clear_temp_directory()
        
        try:
            self.driver.get(url)
            time.sleep(5)  # Initial wait for page load
            
            # Wait for download to start and complete
            wait_time = 0
            download_started = False
            
            while wait_time < self.max_wait_time:
                temp_files = list(self.temp_dir.glob("*"))
                
                if temp_files and not download_started:
                    download_started = True
                
                # Check for completed PDF file
                for temp_file in temp_files:
                    if temp_file.suffix.lower() == '.pdf' and temp_file.stat().st_size > 0:
                        # PDF download completed
                        try:
                            # Ensure target directory exists
                            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                            
                            # Move file to final location
                            shutil.move(str(temp_file), output_file_path)
                            
                            # Verify file exists at target location
                            if os.path.exists(output_file_path):
                                return output_file_path
                            else:
                                logfire.error(f"File move failed - target file not found: {output_file_path}")
                                return None
                                
                        except Exception as move_error:
                            logfire.error(f"Error moving file: {move_error}")
                            return None
                
                # Check for partial downloads (.crdownload files) - wait silently
                
                time.sleep(3)
                wait_time += 3
                
            # Timeout reached
            temp_files = list(self.temp_dir.glob("*"))
            logfire.warning(f"Download timeout after {self.max_wait_time}s. Found {len(temp_files)} files in temp dir:")
            for f in temp_files:
                logfire.warning(f"  - {f.name} ({f.stat().st_size} bytes)")
            
            return None
            
        except WebDriverException as e:
            logfire.error(f"WebDriver error downloading PDF: {e}")
            return None
        except Exception as e:
            logfire.error(f"Error downloading PDF: {e}")
            return None
            
    def scrape_paper(self, openalex_id: str, progress_callback=None) -> bool:
        """Scrape a single paper by OpenAlex ID with DOI retry logic"""
        try:
            # Determine target folder and output path
            folder_id = self.get_folder_for_paper(openalex_id)
            target_folder = self.folder_paths[folder_id]
            paper_id = openalex_id.split('/')[-1]  # Extract W123456 from URL
            output_file_path = target_folder / f"{paper_id}.pdf"
            
            # Check if already exists
            if output_file_path.exists():
                mark_paper_scraped(openalex_id, str(output_file_path), f"folder_{folder_id:02d}")
                if progress_callback:
                    progress_callback(f"✅ {paper_id} (already exists)")
                return True
            
            # First attempt: Get PDF URL from OpenAlex ID
            pdf_url = self.get_paper_pdf_url(openalex_id)
            downloaded_path = None
            
            if pdf_url:
                downloaded_path = self.download_pdf(pdf_url, str(output_file_path))
                
            # If primary attempt failed, try DOI retry
            if not downloaded_path:
                # Get DOI from database
                doi = get_paper_doi(openalex_id)
                
                if doi:
                    doi_pdf_url = self.get_paper_pdf_url_by_doi(doi)
                    
                    if doi_pdf_url:
                        downloaded_path = self.download_pdf(doi_pdf_url, str(output_file_path))
                        
                        if downloaded_path:
                            if progress_callback:
                                progress_callback(f"✅ {paper_id} (DOI retry)")
                        else:
                            if progress_callback:
                                progress_callback(f"❌ {paper_id} (DOI retry failed)")
                    else:
                        if progress_callback:
                            progress_callback(f"❌ {paper_id} (no PDF URL via DOI)")
                else:
                    if progress_callback:
                        progress_callback(f"❌ {paper_id} (no DOI available)")
            
            # Final result handling
            if downloaded_path:
                mark_paper_scraped(openalex_id, downloaded_path, f"folder_{folder_id:02d}")
                if progress_callback:
                    progress_callback(f"✅ {paper_id}")
                return True
            else:
                mark_paper_failed(openalex_id, "Failed to download PDF (tried both OpenAlex ID and DOI)")
                if progress_callback:
                    progress_callback(f"❌ {paper_id} (download failed)")
                return False
                
        except Exception as e:
            error_msg = f"Error scraping {openalex_id}: {e}"
            logfire.error(error_msg)
            mark_paper_failed(openalex_id, error_msg)
            if progress_callback:
                progress_callback(f"❌ {paper_id} (error: {str(e)[:30]}...)")
            return False
            
    def test_download_setup(self):
        """Test download directory setup"""
        print(f"🔧 Testing download setup:")
        print(f"   Base output dir: {self.base_output_dir.absolute()}")
        print(f"   Temp dir: {self.temp_dir.absolute()}")
        print(f"   Temp dir exists: {self.temp_dir.exists()}")
        
        # Test temp directory write permissions
        try:
            test_file = self.temp_dir / "test.txt"
            test_file.write_text("test")
            test_file.unlink()
            print(f"   ✅ Temp dir writable: Yes")
        except Exception as e:
            print(f"   ❌ Temp dir writable: No - {e}")
        
        # Check folder structure
        print(f"   12 folders exist:")
        for i, folder_path in enumerate(self.folder_paths):
            exists = folder_path.exists()
            print(f"      folder_{i:02d}: {'✅' if exists else '❌'}")
    
    def close_driver(self):
        """Close webdriver"""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logfire.info("Closed webdriver")
            except Exception as e:
                logfire.warning(f"Error closing driver: {e}")
                
    def scrape_batch(self, batch_size: int = 10, show_progress: bool = True) -> dict:
        """Scrape a batch of papers from database queue with progress reporting"""
        import time as time_module
        start_time = time_module.time()
        
        stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "start_time": start_time,
            "batch_size_requested": batch_size
        }
        
        try:
            # Get papers to scrape from database
            papers_to_scrape = get_papers_to_scrape(limit=batch_size)
            
            if not papers_to_scrape:
                if show_progress:
                    print("ℹ️  No papers found to scrape")
                return stats
                
            stats["batch_size_actual"] = len(papers_to_scrape)
            
            # Setup progress bar
            if show_progress:
                pbar = tqdm(
                    papers_to_scrape, 
                    desc="📄 Scraping papers",
                    unit="paper",
                    ncols=100,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                )
            else:
                pbar = papers_to_scrape
            
            # Process each paper
            for paper in pbar:
                paper_start_time = time_module.time()
                
                try:
                    # Progress callback for individual paper status
                    def update_progress(status_msg):
                        if show_progress:
                            pbar.set_postfix_str(status_msg)
                    
                    success = self.scrape_paper(paper.openalex_id, progress_callback=update_progress)
                    
                    if success:
                        stats["successful"] += 1
                    else:
                        stats["failed"] += 1
                        
                    stats["processed"] += 1
                    
                    # Update progress bar description with current stats
                    if show_progress:
                        success_rate = (stats["successful"] / stats["processed"]) * 100
                        elapsed = time_module.time() - start_time
                        avg_time = elapsed / stats["processed"] if stats["processed"] > 0 else 0
                        
                        pbar.set_description(
                            f"📄 Scraping papers ({success_rate:.1f}% success, {avg_time:.1f}s/paper)"
                        )
                    
                    # Small delay between downloads
                    time.sleep(2)
                    
                except Exception as e:
                    stats["failed"] += 1
                    stats["processed"] += 1
                    logfire.error(f"Error processing paper {paper.openalex_id}: {e}")
                    if show_progress:
                        pbar.set_postfix_str(f"❌ Error: {str(e)[:30]}...")
                        
            # Close progress bar
            if show_progress:
                pbar.close()
                        
            # Final stats calculation
            stats["total_time"] = time_module.time() - start_time
            stats["success_rate"] = (stats["successful"] / stats["processed"]) * 100 if stats["processed"] > 0 else 0
            stats["avg_time_per_paper"] = stats["total_time"] / stats["processed"] if stats["processed"] > 0 else 0
            
            # Final summary
            if show_progress and stats["processed"] > 0:
                print(f"\n📊 Batch completed: {stats['successful']}/{stats['processed']} successful "
                      f"({stats['success_rate']:.1f}%) in {stats['total_time']/60:.1f}m")
                    
            return stats
            
        except Exception as e:
            logfire.error(f"Failed to scrape batch: {e}")
            return stats
        finally:
            self.close_driver()
            
    def scrape_all_continuous(self, batch_size: int = 100, show_progress: bool = True) -> dict:
        """Continuously scrape all available papers in the database with progress reporting"""
        import time as time_module
        from database.models import get_scraping_stats
        
        overall_start_time = time_module.time()
        total_stats = {
            "total_processed": 0,
            "total_successful": 0,
            "total_failed": 0,
            "batches_completed": 0,
            "start_time": overall_start_time
        }
        
        # Get initial stats and setup overall progress bar
        initial_stats = get_scraping_stats()
        total_in_db = initial_stats.get('total', 0)
        already_scraped = initial_stats.get('scraped', 0)
        pending = initial_stats.get('pending', 0)
        
        if show_progress:
            print(f"🚀 Starting continuous scraping")
            print(f"📋 Total papers in database: {total_in_db:,}")
            print(f"✅ Already scraped: {already_scraped:,}")
            print(f"⏳ Pending: {pending:,}")
            print(f"📦 Batch size: {batch_size}")
            print()
            
            # Overall progress bar
            overall_pbar = tqdm(
                total=total_in_db,
                initial=already_scraped,
                desc="🌐 Overall progress",
                unit="paper",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )
        
        try:
            while True:
                # Process the next batch
                batch_stats = self.scrape_batch(batch_size=batch_size, show_progress=show_progress)
                
                if batch_stats["processed"] == 0:
                    if show_progress:
                        print("✅ All papers have been processed!")
                    break
                
                # Update overall stats
                total_stats["total_processed"] += batch_stats["processed"]
                total_stats["total_successful"] += batch_stats["successful"]
                total_stats["total_failed"] += batch_stats["failed"]
                total_stats["batches_completed"] += 1
                
                # Update overall progress bar
                if show_progress:
                    overall_pbar.update(batch_stats["processed"])
                    
                    # Update overall progress bar description
                    elapsed = time_module.time() - overall_start_time
                    overall_success_rate = (total_stats["total_successful"] / total_stats["total_processed"]) * 100
                    papers_per_hour = (total_stats["total_processed"] / (elapsed/3600)) if elapsed > 0 else 0
                    
                    overall_pbar.set_description(
                        f"🌐 Overall progress ({overall_success_rate:.1f}% success, {papers_per_hour:.0f}/hr)"
                    )
                    
                    # Show postfix with current batch info
                    overall_pbar.set_postfix_str(
                        f"Batch #{total_stats['batches_completed']}: {batch_stats['successful']}/{batch_stats['processed']}"
                    )
                
                # Small delay between batches
                time.sleep(1)
                        
        except KeyboardInterrupt:
            if show_progress:
                overall_pbar.close()
                elapsed = time_module.time() - overall_start_time
                print(f"\n⏹️  Interrupted by user after {elapsed/60:.1f} minutes")
                print(f"✅ Successfully processed {total_stats['total_successful']:,} papers")
                print("💡 You can resume by running the same command again")
        except Exception as e:
            logfire.error(f"Error in continuous scraping: {e}")
            if show_progress:
                overall_pbar.close()
                print(f"❌ Error in continuous scraping: {e}")
        finally:
            if show_progress and 'overall_pbar' in locals():
                overall_pbar.close()
                
        # Final summary
        total_stats["total_time"] = time_module.time() - overall_start_time
        if show_progress:
            elapsed_hours = total_stats["total_time"] / 3600
            final_success_rate = (total_stats["total_successful"] / total_stats["total_processed"]) * 100 if total_stats["total_processed"] > 0 else 0
            
            print(f"\n🏁 FINAL RESULTS:")
            print(f"   ⏱️  Total time: {total_stats['total_time']/60:.1f} minutes")
            print(f"   📊 Papers processed: {total_stats['total_processed']:,}")
            print(f"   ✅ Successful: {total_stats['total_successful']:,} ({final_success_rate:.1f}%)")
            print(f"   ❌ Failed: {total_stats['total_failed']:,}")
            print(f"   🔄 Batches completed: {total_stats['batches_completed']}")
            if elapsed_hours > 0:
                print(f"   📈 Average rate: {total_stats['total_processed']/elapsed_hours:.0f} papers/hour")
                
        return total_stats


class ParallelPaperScraper:
    """
    Parallel paper scraper that manages multiple Chrome instances for concurrent downloading
    """
    
    def __init__(self, base_output_dir: str = "./scraping_output", max_wait_time: int = 30, num_workers: int = 5):
        self.base_output_dir = Path(base_output_dir)
        self.max_wait_time = max_wait_time
        self.num_workers = num_workers
        
        # Thread-safe counters and locks
        self.progress_lock = Lock()
        self.stats_lock = Lock()
        self.scrapers = []
        
        # Setup directories
        self.setup_directories()
        
        logfire.info(f"Initialized parallel scraper with {num_workers} workers")
    
    def setup_directories(self):
        """Setup directories for all workers"""
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 12 folders for distribution
        self.folder_paths = []
        for i in range(12):
            folder_path = self.base_output_dir / f"folder_{i:02d}"
            folder_path.mkdir(exist_ok=True)
            self.folder_paths.append(folder_path)
            
        # Create temp directory for each worker
        for worker_id in range(self.num_workers):
            temp_dir = self.base_output_dir / f"temp_downloads_worker_{worker_id}"
            temp_dir.mkdir(exist_ok=True)
    
    def create_worker_scraper(self, worker_id: int) -> TargetedPaperScraper:
        """Create a scraper instance for a specific worker"""
        # Each worker gets its own temp directory to avoid conflicts
        worker_temp_dir = self.base_output_dir / f"temp_downloads_worker_{worker_id}"
        
        scraper = TargetedPaperScraper(
            base_output_dir=str(self.base_output_dir),
            max_wait_time=self.max_wait_time
        )
        
        # Override temp directory for this worker
        scraper.temp_dir = worker_temp_dir
        scraper.worker_id = worker_id
        
        return scraper
    
    def scrape_paper_worker(self, paper_data):
        """Worker function to scrape a single paper"""
        paper, worker_id, progress_callback = paper_data
        
        # Get or create scraper for this thread
        thread_id = threading.current_thread().ident
        if not hasattr(threading.current_thread(), 'scraper'):
            threading.current_thread().scraper = self.create_worker_scraper(worker_id)
        
        scraper = threading.current_thread().scraper
        
        # Scrape the paper
        try:
            success = scraper.scrape_paper(paper.openalex_id, progress_callback=progress_callback)
            return success, paper.openalex_id, worker_id
        except Exception as e:
            logfire.error(f"Worker {worker_id} error on {paper.openalex_id}: {e}")
            return False, paper.openalex_id, worker_id
    
    def scrape_batch_parallel(self, batch_size: int = 50, show_progress: bool = True) -> dict:
        """Scrape a batch of papers using parallel workers"""
        import time as time_module
        start_time = time_module.time()
        
        stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "start_time": start_time,
            "batch_size_requested": batch_size,
            "workers_used": self.num_workers
        }
        
        try:
            # Get papers to scrape
            papers_to_scrape = get_papers_to_scrape(limit=batch_size)
            
            if not papers_to_scrape:
                if show_progress:
                    print("ℹ️  No papers found to scrape")
                return stats
                
            stats["batch_size_actual"] = len(papers_to_scrape)
            
            # Setup progress bar
            if show_progress:
                pbar = tqdm(
                    total=len(papers_to_scrape),
                    desc=f"📄 Scraping papers ({self.num_workers} workers)",
                    unit="paper",
                    ncols=100,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                )
            
            # Thread-safe progress callback
            def update_progress(status_msg):
                if show_progress:
                    with self.progress_lock:
                        pbar.set_postfix_str(status_msg)
            
            # Prepare work items (paper, worker_id, callback)
            work_items = []
            for i, paper in enumerate(papers_to_scrape):
                worker_id = i % self.num_workers  # Distribute papers across workers
                work_items.append((paper, worker_id, update_progress))
            
            # Execute parallel scraping
            successful_papers = []
            failed_papers = []
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                future_to_paper = {
                    executor.submit(self.scrape_paper_worker, work_item): work_item[0] 
                    for work_item in work_items
                }
                
                # Process completed tasks
                for future in as_completed(future_to_paper):
                    paper = future_to_paper[future]
                    
                    try:
                        success, openalex_id, worker_id = future.result()
                        
                        with self.stats_lock:
                            if success:
                                stats["successful"] += 1
                                successful_papers.append(openalex_id)
                            else:
                                stats["failed"] += 1
                                failed_papers.append(openalex_id)
                            
                            stats["processed"] += 1
                            
                            # Update progress bar
                            if show_progress:
                                pbar.update(1)
                                success_rate = (stats["successful"] / stats["processed"]) * 100
                                elapsed = time_module.time() - start_time
                                avg_time = elapsed / stats["processed"] if stats["processed"] > 0 else 0
                                
                                pbar.set_description(
                                    f"📄 Scraping papers ({success_rate:.1f}% success, {avg_time:.1f}s/paper, {self.num_workers} workers)"
                                )
                    
                    except Exception as e:
                        logfire.error(f"Future error for {paper.openalex_id}: {e}")
                        with self.stats_lock:
                            stats["failed"] += 1
                            stats["processed"] += 1
                            failed_papers.append(paper.openalex_id)
                        
                        if show_progress:
                            pbar.update(1)
            
            # Close progress bar
            if show_progress:
                pbar.close()
            
            # Final stats calculation
            stats["total_time"] = time_module.time() - start_time
            stats["success_rate"] = (stats["successful"] / stats["processed"]) * 100 if stats["processed"] > 0 else 0
            stats["avg_time_per_paper"] = stats["total_time"] / stats["processed"] if stats["processed"] > 0 else 0
            stats["papers_per_second"] = stats["processed"] / stats["total_time"] if stats["total_time"] > 0 else 0
            
            # Final summary
            if show_progress and stats["processed"] > 0:
                print(f"\n📊 Parallel batch completed:")
                print(f"   🔧 Workers used: {self.num_workers}")
                print(f"   📄 Papers processed: {stats['processed']}")
                print(f"   ✅ Successful: {stats['successful']} ({stats['success_rate']:.1f}%)")
                print(f"   ❌ Failed: {stats['failed']}")
                print(f"   ⏱️  Total time: {stats['total_time']/60:.1f}m")
                print(f"   🚀 Speed: {stats['papers_per_second']:.1f} papers/second")
                print(f"   📈 Speedup: ~{stats['papers_per_second']*5:.1f}x vs single worker")
                    
            return stats
            
        except Exception as e:
            logfire.error(f"Failed to scrape parallel batch: {e}")
            return stats
        finally:
            # Cleanup all worker scrapers
            self.cleanup_workers()
    
    def cleanup_workers(self):
        """Clean up all worker Chrome instances"""
        try:
            # Get all active threads and clean up their scrapers
            for thread in threading.enumerate():
                if hasattr(thread, 'scraper') and thread.scraper:
                    try:
                        thread.scraper.close_driver()
                    except Exception as e:
                        logfire.warning(f"Error closing worker scraper: {e}")
            
            # Clean up worker temp directories
            for worker_id in range(self.num_workers):
                temp_dir = self.base_output_dir / f"temp_downloads_worker_{worker_id}"
                if temp_dir.exists():
                    try:
                        for file in temp_dir.glob("*"):
                            if file.is_file():
                                file.unlink()
                    except Exception as e:
                        logfire.warning(f"Error cleaning worker {worker_id} temp dir: {e}")
                        
            logfire.info("Cleaned up all worker instances")
            
        except Exception as e:
            logfire.error(f"Error during worker cleanup: {e}")
    
    def scrape_all_continuous_parallel(self, batch_size: int = 100, show_progress: bool = True) -> dict:
        """Continuously scrape with parallel workers"""
        import time as time_module
        from database.models import get_scraping_stats
        
        overall_start_time = time_module.time()
        total_stats = {
            "total_processed": 0,
            "total_successful": 0,
            "total_failed": 0,
            "batches_completed": 0,
            "start_time": overall_start_time,
            "workers_used": self.num_workers
        }
        
        # Get initial stats and setup overall progress bar
        initial_stats = get_scraping_stats()
        total_in_db = initial_stats.get('total', 0)
        already_scraped = initial_stats.get('scraped', 0)
        pending = initial_stats.get('pending', 0)
        
        if show_progress:
            print(f"🚀 Starting parallel continuous scraping with {self.num_workers} workers")
            print(f"📋 Total papers in database: {total_in_db:,}")
            print(f"✅ Already scraped: {already_scraped:,}")
            print(f"⏳ Pending: {pending:,}")
            print(f"📦 Batch size: {batch_size}")
            print()
            
            # Overall progress bar
            overall_pbar = tqdm(
                total=total_in_db,
                initial=already_scraped,
                desc=f"🌐 Overall progress ({self.num_workers} workers)",
                unit="paper",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )
        
        try:
            while True:
                # Process the next batch
                batch_stats = self.scrape_batch_parallel(batch_size=batch_size, show_progress=show_progress)
                
                if batch_stats["processed"] == 0:
                    if show_progress:
                        print("✅ All papers have been processed!")
                    break
                
                # Update overall stats
                total_stats["total_processed"] += batch_stats["processed"]
                total_stats["total_successful"] += batch_stats["successful"]
                total_stats["total_failed"] += batch_stats["failed"]
                total_stats["batches_completed"] += 1
                
                # Update overall progress bar
                if show_progress:
                    overall_pbar.update(batch_stats["processed"])
                    
                    # Update overall progress bar description
                    elapsed = time_module.time() - overall_start_time
                    overall_success_rate = (total_stats["total_successful"] / total_stats["total_processed"]) * 100
                    papers_per_hour = (total_stats["total_processed"] / (elapsed/3600)) if elapsed > 0 else 0
                    
                    overall_pbar.set_description(
                        f"🌐 Overall progress ({overall_success_rate:.1f}% success, {papers_per_hour:.0f}/hr, {self.num_workers}w)"
                    )
                    
                    # Show postfix with current batch info
                    overall_pbar.set_postfix_str(
                        f"Batch #{total_stats['batches_completed']}: {batch_stats['successful']}/{batch_stats['processed']} | {batch_stats.get('papers_per_second', 0):.1f}/s"
                    )
                
                # Small delay between batches
                time.sleep(1)
                        
        except KeyboardInterrupt:
            if show_progress:
                overall_pbar.close()
                elapsed = time_module.time() - overall_start_time
                print(f"\n⏹️  Interrupted by user after {elapsed/60:.1f} minutes")
                print(f"✅ Successfully processed {total_stats['total_successful']:,} papers")
                print("💡 You can resume by running the same command again")
        except Exception as e:
            logfire.error(f"Error in parallel continuous scraping: {e}")
            if show_progress:
                overall_pbar.close()
                print(f"❌ Error in parallel continuous scraping: {e}")
        finally:
            if show_progress and 'overall_pbar' in locals():
                overall_pbar.close()
                
        # Final summary
        total_stats["total_time"] = time_module.time() - overall_start_time
        if show_progress:
            elapsed_hours = total_stats["total_time"] / 3600
            final_success_rate = (total_stats["total_successful"] / total_stats["total_processed"]) * 100 if total_stats["total_processed"] > 0 else 0
            
            print(f"\n🏁 PARALLEL SCRAPING RESULTS:")
            print(f"   🔧 Workers used: {self.num_workers}")
            print(f"   ⏱️  Total time: {total_stats['total_time']/60:.1f} minutes")
            print(f"   📊 Papers processed: {total_stats['total_processed']:,}")
            print(f"   ✅ Successful: {total_stats['total_successful']:,} ({final_success_rate:.1f}%)")
            print(f"   ❌ Failed: {total_stats['total_failed']:,}")
            print(f"   🔄 Batches completed: {total_stats['batches_completed']}")
            if elapsed_hours > 0:
                avg_rate = total_stats['total_processed']/elapsed_hours
                print(f"   📈 Average rate: {avg_rate:.0f} papers/hour")
                print(f"   🚀 Estimated speedup: ~{self.num_workers}x vs single worker")
                
        return total_stats


def main():
    parser = argparse.ArgumentParser(description='Targeted paper scraper by OpenAlex ID')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of papers to scrape')
    parser.add_argument('--output-dir', default='./scraping_output', help='Output directory')
    parser.add_argument('--max-wait-time', type=int, default=30, help='Max wait time for downloads')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel Chrome workers (1=sequential, 5=recommended)')
    parser.add_argument('--test-paper', help='Test scraping a specific OpenAlex ID')
    
    args = parser.parse_args()
    
    # Configure logfire
    logfire.configure(token="pylf_v1_us_qTtmbDFpkfhFwzTfZyZrTJcl4C4lC7FhmZ65BgJ7dLDV")
    
    try:
        # Choose scraper based on worker count
        if args.workers > 1:
            print(f"🚀 Using parallel scraper with {args.workers} Chrome workers")
            scraper = ParallelPaperScraper(
                base_output_dir=args.output_dir,
                max_wait_time=args.max_wait_time,
                num_workers=args.workers
            )
            is_parallel = True
        else:
            print("🔧 Using sequential scraper (single Chrome worker)")
            scraper = TargetedPaperScraper(
                base_output_dir=args.output_dir,
                max_wait_time=args.max_wait_time
            )
            is_parallel = False
        
        if args.test_paper:
            print(f"🧪 Testing single paper: {args.test_paper}")
            # Use sequential scraper for testing
            if is_parallel:
                test_scraper = TargetedPaperScraper(
                    base_output_dir=args.output_dir,
                    max_wait_time=args.max_wait_time
                )
                success = test_scraper.scrape_paper(args.test_paper)
                test_scraper.close_driver()
            else:
                success = scraper.scrape_paper(args.test_paper)
            
            if success:
                print("✅ Test successful!")
            else:
                print("❌ Test failed")
        else:
            print(f"🚀 Starting {'parallel' if is_parallel else 'sequential'} paper scraping...")
            
            if is_parallel:
                stats = scraper.scrape_batch_parallel(args.batch_size)
            else:
                stats = scraper.scrape_batch(args.batch_size)
            
            print("\n📊 BATCH RESULTS:")
            print(f"   ✅ Successful: {stats['successful']}")
            print(f"   ❌ Failed: {stats['failed']}")
            print(f"   📋 Total processed: {stats['processed']}")
            
            if is_parallel:
                print(f"   🔧 Workers used: {args.workers}")
                if stats.get('papers_per_second'):
                    print(f"   🚀 Speed: {stats['papers_per_second']:.1f} papers/second")
            
            if stats['successful'] > 0:
                if is_parallel:
                    print(f"\n🎉 Successfully scraped {stats['successful']} papers with {args.workers} parallel workers!")
                else:
                    print(f"\n🎉 Successfully scraped {stats['successful']} papers!")
                print(f"📁 Files distributed across folders in: {args.output_dir}")
                print("💡 Next: Run metadata extraction with ../run_metadata_extraction.sh")
            
    except Exception as e:
        print(f"❌ Script failed: {e}")
        logfire.error(f"Main script error: {e}")

if __name__ == "__main__":
    main() 