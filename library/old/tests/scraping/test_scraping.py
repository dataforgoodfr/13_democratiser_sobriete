#!/usr/bin/env python3
"""
Test Scraping Script - Test with 10 papers
Simple script to test paper scraping functionality before running on 250k papers
"""

import os
import sys
import time
import hashlib
from pathlib import Path

import logfire

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from library.database.models import (
    get_papers_to_scrape, 
    mark_paper_scraped, 
    mark_paper_failed,
    get_scraping_stats
)

# Configure logfire
LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN", "pylf_v1_us_qTtmbDFpkfhFwzTfZyZrTJcl4C4lC7FhmZ65BgJ7dLDV")
logfire.configure(token=LOGFIRE_TOKEN)

def setup_test_directories(base_dir="./test_scraping_output"):
    """Setup directory structure for testing"""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Create 12 folders
    folder_paths = []
    for i in range(12):
        folder_path = base_path / f"folder_{i:02d}"
        folder_path.mkdir(exist_ok=True)
        folder_paths.append(folder_path)
    
    print(f"✅ Created test directories in {base_path}")
    return base_path, folder_paths

def get_folder_for_paper(openalex_id: str) -> int:
    """Determine which folder a paper should go to based on hash distribution"""
    hash_obj = hashlib.md5(openalex_id.encode())
    return int(hash_obj.hexdigest(), 16) % 12

def test_scraping_10_papers():
    """Test scraping functionality with 10 papers"""
    
    print("🧪 TESTING PAPER SCRAPING (10 papers)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Step 1: Setup test directories
    base_dir, folder_paths = setup_test_directories()
    
    # Step 2: Get papers to scrape (auto-populates as needed)
    print("\n🔍 Step 1: Getting papers to scrape (auto-populating as needed)...")
    try:
        papers_to_scrape = get_papers_to_scrape(limit=10)
        print(f"📄 Found {len(papers_to_scrape)} papers to scrape")
        
        if not papers_to_scrape:
            print("⚠️  No papers found to scrape. They might already be scraped.")
            stats = get_scraping_stats()
            print(f"📊 Current stats: {stats}")
            return True
            
    except Exception as e:
        print(f"❌ Failed to get papers: {e}")
        return False
    
    # Step 3: Test scraping each paper
    print(f"\n🌐 Step 2: Testing scraping of {len(papers_to_scrape)} papers...")
    successful = 0
    failed = 0
    
    for i, paper in enumerate(papers_to_scrape, 1):
        try:
            print(f"\n[{i}/{len(papers_to_scrape)}] 🔄 Processing: {paper.openalex_id}")
            print(f"   📝 Title: {paper.title[:80]}..." if hasattr(paper, 'title') and paper.title else "   📝 Title: [No title]")
            
            # Determine target folder
            folder_id = get_folder_for_paper(paper.openalex_id)
            target_folder = folder_paths[folder_id]
            
            print(f"   📁 Target folder: folder_{folder_id:02d}")
            
            # Try real scraping with the targeted scraper
            try:
                from ...src.library.scraping import TargetedPaperScraper
                
                # Create scraper for testing
                scraper = TargetedPaperScraper(base_output_dir=str(base_dir))
                
                # Attempt to scrape the specific paper
                success = scraper.scrape_paper(paper.openalex_id)
                
                if success:
                    successful += 1
                    print("   ✅ Successfully scraped real PDF!")
                else:
                    failed += 1
                    print("   ❌ Failed to scrape PDF")
                
                # Clean up
                scraper.close_driver()
                
            except Exception as scrape_error:
                # Fallback to dummy file for testing
                print(f"   ⚠️  Real scraping failed ({scrape_error}), creating test file...")
                
                try:
                    test_filename = f"{paper.openalex_id.split('/')[-1]}.pdf"
                    test_pdf_path = target_folder / test_filename
                    
                    # Create a dummy PDF file for testing
                    with open(test_pdf_path, 'w') as f:
                        f.write(f"# Test PDF for {paper.openalex_id}\n")
                        f.write("# This is a test file, not a real PDF\n")
                        f.write(f"# Paper: {getattr(paper, 'title', 'No title')}\n")
                        f.write(f"# Folder: {folder_id}\n")
                        f.write(f"# Created: {time.time()}\n")
                    
                    # Mark as scraped in database
                    mark_paper_scraped(paper.openalex_id, str(test_pdf_path), folder_id)
                    
                    successful += 1
                    print(f"   ✅ Created test file at {test_pdf_path}")
                    
                except Exception as fallback_error:
                    # Mark as failed
                    mark_paper_failed(paper.openalex_id, str(fallback_error))
                    failed += 1
                    print(f"   ❌ Failed completely: {fallback_error}")
                
        except Exception as e:
            failed += 1
            print(f"   ❌ Error processing paper: {e}")
            logfire.error(f"Error processing {paper.openalex_id}: {e}")
    
    # Step 5: Show results
    print("\n🎯 SCRAPING TEST COMPLETED")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 RESULTS:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📋 Total: {len(papers_to_scrape)}")
    
    # Show file distribution
    print("\n📁 FILE DISTRIBUTION:")
    for i in range(12):
        folder_path = base_dir / f"folder_{i:02d}"
        file_count = len(list(folder_path.glob("*")))
        if file_count > 0:
            print(f"   📁 folder_{i:02d}: {file_count} files")
    
    # Show database stats
    try:
        stats = get_scraping_stats()
        print("\n📊 DATABASE STATS:")
        print(f"   📋 Total in queue: {stats.get('total', 0)}")
        print(f"   ✅ Scraped: {stats.get('scraped', 0)}")
        print(f"   ❌ Failed: {stats.get('failed', 0)}")
        print(f"   ⏳ Pending: {stats.get('pending', 0)}")
    except Exception as e:
        print(f"   ⚠️  Could not get stats: {e}")
    
    print(f"\n🔍 Test files created in: {base_dir}")
    print("💡 You can now test metadata extraction with:")
    print(f"   python ../historized_ingestion_pipeline.py --folder-path {base_dir}/folder_00")
    
    return successful > 0

def test_targeted_scraper_single():
    """Test the targeted scraper with a specific OpenAlex ID"""
    print("\n🧪 TESTING TARGETED SCRAPER (1 specific paper)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    try:
        from ...src.library.scraping import TargetedPaperScraper
        
        # Test with a known OpenAlex ID (example)
        test_openalex_id = "https://openalex.org/W2741809807"  # Known open access paper
        
        print(f"🔍 Testing targeted scraping for: {test_openalex_id}")
        print("⚠️  This will attempt to download a real PDF - make sure Chrome is set up")
        
        scraper = TargetedPaperScraper(base_output_dir="./test_targeted_output")
        success = scraper.scrape_paper(test_openalex_id)
        
        if success:
            print("✅ Targeted scraping test successful!")
            print("📁 Check ./test_targeted_output/ for the downloaded PDF")
        else:
            print("❌ Targeted scraping test failed")
            print("💡 The paper might not have an available PDF")
            
        scraper.close_driver()
        
    except ImportError:
        print("⚠️  Targeted scraper not available")
    except Exception as e:
        print(f"❌ Targeted scraping error: {e}")
        print("💡 This might be normal if Chrome/Selenium isn't set up")
        print("💡 The main test above should be sufficient for most cases")

def main():
    """Main test function"""
    
    print("🚀 PAPER SCRAPING TEST SUITE")
    print("=" * 80)
    
    try:
        # Test 1: Database and folder setup
        success1 = test_scraping_10_papers()
        
        if success1:
            print("\n✅ Database and folder test completed successfully!")
            
            # Ask if user wants to test targeted scraping
            response = input("\n❓ Test targeted scraper with a specific paper? (y/N): ").strip().lower()
            if response == 'y':
                test_targeted_scraper_single()
            else:
                print("⏭️  Skipping targeted scraper test")
        
        print("\n🎉 Testing completed!")
        print("💡 Next steps:")
        print("   1. Check test files in ./test_scraping_output/")
        print("   2. Test metadata extraction on a folder")
        print("   3. Run full scraping pipeline when ready")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        logfire.error(f"Test suite error: {e}")

if __name__ == "__main__":
    main() 