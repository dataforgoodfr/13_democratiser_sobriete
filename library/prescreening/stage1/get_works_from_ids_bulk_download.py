"""
NOT USED IN THE END!
Script to download works from OpenAlex snapshot and keep only those in a given list of IDs.
Should allow to bypass OpenAlex API rate limits.
AI-generated.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import gzip
import logging
from io import BytesIO
from pathlib import Path
import requests
from typing import Set, List
import sys

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


MANIFEST_URL = "https://openalex.s3.amazonaws.com/data/works/manifest"


class OpenAlexFetcher:
    def __init__(self, id_list_path: str, output_dir: str = "output", 
                 batch_size: int = 10000, max_workers: int = 8):
        """
        Initialize the fetcher.
        
        Args:
            id_list_path: Path to file containing IDs (one per line)
            output_dir: Directory for output parquet files
            batch_size: Number of records per output file
            max_workers: Number of parallel download threads
        """
        self.id_list_path = id_list_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Initialize S3 client (anonymous access)
        self.s3_client = boto3.client(
            's3',
            config=Config(signature_version=UNSIGNED)
        )
        
        # Load IDs into a set for fast lookup
        logger.info(f"Loading IDs from {id_list_path}")
        self.target_ids = self._load_ids()
        logger.info(f"Loaded {len(self.target_ids)} target IDs")
        
        # Track statistics
        self.stats = {
            'files_processed': 0,
            'records_found': 0,
            'output_files': 0
        }
        
        # Buffer for accumulating records
        self.buffer = []
        
    def _load_ids(self) -> Set[str]:
        """Load target IDs from file into a set."""
        ids = set()
        with open(self.id_list_path, 'r') as f:
            for line in f:
                id_str = line.strip()
                if id_str:
                    ids.add(id_str)
        return ids
    
    def _parse_s3_url(self, url: str) -> tuple:
        """Parse S3 URL into bucket and key."""
        # Format: s3://bucket/path/to/file
        parts = url.replace('s3://', '').split('/', 1)
        return parts[0], parts[1]
    
    def _download_and_decompress(self, s3_url: str) -> List[dict]:
        """Download and decompress a single S3 file, filtering by IDs."""
        try:
            bucket, key = self._parse_s3_url(s3_url)
            
            # Download file
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            compressed_data = response['Body'].read()
            
            # Decompress
            with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as gz:
                # Parse JSONL and filter line by line for memory efficiency
                filtered_records = []
                for line in tqdm(gz):
                    record = json.loads(line)
                    
                    # Check if record ID is in our target set
                    record_id = record.get('id', '')
                    if record_id in self.target_ids:
                        filtered_records.append(record)
            
            return filtered_records
            
        except Exception as e:
            logger.error(f"Error processing {s3_url}: {e}")
            return []
    
    def _write_batch(self, records: List[dict]):
        """Write a batch of records to a parquet file."""
        if not records:
            return
        
        # Convert to PyArrow table
        table = pa.Table.from_pylist(records)
        
        # Write to parquet
        output_file = self.output_dir / f"batch_{self.stats['output_files']:05d}.parquet"
        pq.write_table(table, output_file, compression='snappy')
        
        self.stats['output_files'] += 1
        logger.info(f"Wrote {len(records)} records to {output_file}")
    
    def process_manifest(self, manifest_url: str):
        """Process all files listed in the manifest."""
        # Load manifest
        r = requests.get(manifest_url)
        manifest = r.json()
        urls = [entry['url'] for entry in manifest['entries']]
        total_files = len(urls)
        logger.info(f"Processing {total_files} files from manifest")
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(self._download_and_decompress, url): url 
                for url in urls
            }
            
            # Process completed tasks
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    records = future.result()
                    self.stats['files_processed'] += 1
                    
                    if records:
                        self.buffer.extend(records)
                        self.stats['records_found'] += len(records)
                        
                        # Write batch if buffer is full
                        if len(self.buffer) >= self.batch_size:
                            self._write_batch(self.buffer[:self.batch_size])
                            self.buffer = self.buffer[self.batch_size:]
                    
                    # Progress update
                    if self.stats['files_processed'] % 10 == 0:
                        logger.info(
                            f"Progress: {self.stats['files_processed']}/{total_files} files, "
                            f"{self.stats['records_found']} records found"
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to process {url}: {e}")
        
        # Write remaining records
        if self.buffer:
            self._write_batch(self.buffer)
        
        # Final statistics
        logger.info("=" * 60)
        logger.info("Processing complete!")
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Records found: {self.stats['records_found']}")
        logger.info(f"Output files created: {self.stats['output_files']}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <id_list_file> [max_workers]")
        print("\nExample:")
        print("  python script.py ids.txt 16")
        sys.exit(1)
    
    id_list_file = sys.argv[1]
    max_workers = int(sys.argv[2]) if len(sys.argv) > 1 else 8
    
    fetcher = OpenAlexFetcher(
        id_list_path=id_list_file,
        batch_size=100_000,  # 100k records per parquet file
        max_workers=max_workers
    )
    
    fetcher.process_manifest(MANIFEST_URL)

if __name__ == "__main__":
    main()