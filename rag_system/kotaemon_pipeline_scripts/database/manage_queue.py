#!/usr/bin/env python3
"""
Queue Management Utility
Comprehensive tool for managing the scraping queue database
"""

import argparse
import sys
from datetime import datetime
from typing import Optional

import logfire
from .models import (
    get_session,
    ScrapingQueue,
    populate_scraping_queue,
    get_scraping_stats,
    create_tables,
    reset_failed_papers
)
from sqlmodel import select, delete

# Configure logfire
LOGFIRE_TOKEN = "pylf_v1_us_qTtmbDFpkfhFwzTfZyZrTJcl4C4lC7FhmZ65BgJ7dLDV"
logfire.configure(token=LOGFIRE_TOKEN)

class QueueManager:
    """Manages scraping queue operations"""

    def __init__(self):
        """Initialize queue manager"""
        try:
            create_tables()
            logfire.info("Database connection established")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            sys.exit(1)

    def show_stats(self) -> dict:
        """Show detailed queue statistics"""
        try:
            stats = get_scraping_stats()

            print("📊 SCRAPING QUEUE STATISTICS")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"📋 Total in queue: {stats.get('total', 0):,}")
            print(f"✅ Successfully scraped: {stats.get('scraped', 0):,}")
            print(f"❌ Failed: {stats.get('failed', 0):,}")
            print(f"⏳ Pending: {stats.get('pending', 0):,}")

            if stats.get('total', 0) > 0:
                completion_rate = (stats.get('scraped', 0) / stats['total']) * 100
                failure_rate = (stats.get('failed', 0) / stats['total']) * 100
                print(f"\n📈 Completion rate: {completion_rate:.1f}%")
                print(f"📉 Failure rate: {failure_rate:.1f}%")

            # Show folder distribution
            self._show_folder_distribution()

            return stats

        except Exception as e:
            print(f"❌ Failed to get statistics: {e}")
            logfire.error(f"Stats error: {e}")
            return {}

    def _show_folder_distribution(self):
        """Show how papers are distributed across folders"""
        try:
            with get_session() as session:
                # Count papers per folder
                print("\n📁 FOLDER DISTRIBUTION:")
                total_assigned = 0

                for folder_id in range(12):
                    stmt = select(ScrapingQueue).where(
                        ScrapingQueue.assigned_folder == folder_id,
                        ScrapingQueue.scraped == True
                    )
                    count = len(session.exec(stmt).all())

                    if count > 0:
                        print(f"   📂 folder_{folder_id:02d}: {count:,} papers")
                        total_assigned += count

                if total_assigned == 0:
                    print("   📭 No papers distributed to folders yet")
                else:
                    print(f"   📊 Total distributed: {total_assigned:,} papers")

        except Exception as e:
            logfire.warning(f"Failed to show folder distribution: {e}")

    def clear_queue(self, confirm: bool = False) -> bool:
        """Clear the entire scraping queue"""
        try:
            if not confirm:
                response = input("⚠️  This will DELETE ALL entries in the scraping queue. Continue? (y/N): ")
                if response.lower() != 'y':
                    print("❌ Operation cancelled")
                    return False

            with get_session() as session:
                # Count entries before deletion
                count_stmt = select(ScrapingQueue)
                before_count = len(session.exec(count_stmt).all())

                if before_count == 0:
                    print("📭 Queue is already empty")
                    return True

                # Delete all entries
                delete_stmt = delete(ScrapingQueue)
                session.exec(delete_stmt)
                session.commit()

                print(f"🗑️  Cleared {before_count:,} entries from scraping queue")
                logfire.info(f"Cleared scraping queue: {before_count} entries")
                return True

        except Exception as e:
            print(f"❌ Failed to clear queue: {e}")
            logfire.error(f"Clear queue error: {e}")
            return False

    def reset_failed(self) -> int:
        """Reset failed entries to pending status"""
        try:
            reset_count = reset_failed_papers()

            if reset_count > 0:
                print(f"🔄 Reset {reset_count:,} failed entries to pending status")
            else:
                print("📭 No failed entries to reset")

            return reset_count

        except Exception as e:
            print(f"❌ Failed to reset failed entries: {e}")
            logfire.error(f"Reset failed error: {e}")
            return 0

    def populate_queue(self, limit: Optional[int] = None) -> int:
        """Populate queue from policies_abstracts_all table"""
        try:
            print("📥 Populating scraping queue from policies_abstracts_all...")
            added = populate_scraping_queue(limit=limit)

            if added > 0:
                print(f"✅ Added {added:,} new papers to scraping queue")
            else:
                print("📭 No new papers added (may already exist in queue)")

            # Show updated stats
            stats = get_scraping_stats()
            print(f"📊 Queue now contains {stats.get('total', 0):,} total papers")

            return added

        except Exception as e:
            print(f"❌ Failed to populate queue: {e}")
            logfire.error(f"Populate queue error: {e}")
            return 0

    def show_failed_papers(self, limit: int = 20):
        """Show recent failed papers with error messages"""
        try:
            with get_session() as session:
                failed_stmt = select(ScrapingQueue).where(
                    ScrapingQueue.error_message.isnot(None)
                ).limit(limit)

                failed_papers = session.exec(failed_stmt).all()

                if not failed_papers:
                    print("✅ No failed papers found")
                    return

                print(f"❌ FAILED PAPERS (showing {len(failed_papers)} of {limit} max):")
                print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

                for i, paper in enumerate(failed_papers, 1):
                    print(f"{i:2d}. {paper.openalex_id}")
                    if paper.error_message:
                        # Truncate long error messages
                        error = paper.error_message[:100] + "..." if len(paper.error_message) > 100 else paper.error_message
                        print(f"    💥 {error}")
                    if paper.doi:
                        print(f"    🔗 DOI: {paper.doi}")
                    print()

        except Exception as e:
            print(f"❌ Failed to show failed papers: {e}")
            logfire.error(f"Show failed papers error: {e}")

    def show_recent_successes(self, limit: int = 10):
        """Show recently scraped papers"""
        try:
            with get_session() as session:
                success_stmt = select(ScrapingQueue).where(
                    ScrapingQueue.scraped == True,
                    ScrapingQueue.scraped_at.isnot(None)
                ).order_by(ScrapingQueue.scraped_at.desc()).limit(limit)

                success_papers = session.exec(success_stmt).all()

                if not success_papers:
                    print("📭 No successful papers found")
                    return

                print(f"✅ RECENT SUCCESSES (last {len(success_papers)}):")
                print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

                for i, paper in enumerate(success_papers, 1):
                    print(f"{i:2d}. {paper.openalex_id}")
                    if paper.assigned_folder is not None:
                        print(f"    📁 folder_{paper.assigned_folder:02d}")
                    if paper.scraped_at:
                        print(f"    🕐 {paper.scraped_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    print()

        except Exception as e:
            print(f"❌ Failed to show recent successes: {e}")
            logfire.error(f"Show recent successes error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Manage scraping queue database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_queue.py --stats                    # Show statistics
  python manage_queue.py --populate --limit 1000   # Add 1000 papers
  python manage_queue.py --clear                    # Clear entire queue
  python manage_queue.py --reset-failed             # Reset failed entries
  python manage_queue.py --show-failed --limit 50  # Show 50 failed papers
        """
    )

    # Primary actions
    parser.add_argument('--stats', action='store_true', help='Show queue statistics')
    parser.add_argument('--populate', action='store_true', help='Populate queue from database')
    parser.add_argument('--clear', action='store_true', help='Clear entire queue')
    parser.add_argument('--reset-failed', action='store_true', help='Reset failed entries to pending')

    # Display options
    parser.add_argument('--show-failed', action='store_true', help='Show failed papers')
    parser.add_argument('--show-successes', action='store_true', help='Show recent successes')

    # Parameters
    parser.add_argument('--limit', type=int, help='Limit number of papers/results')
    parser.add_argument('--confirm', action='store_true', help='Skip confirmation prompts')

    args = parser.parse_args()

    # If no arguments, show stats by default
    if not any([args.stats, args.populate, args.clear, args.reset_failed,
                args.show_failed, args.show_successes]):
        args.stats = True

    try:
        manager = QueueManager()

        # Execute requested actions
        if args.clear:
            success = manager.clear_queue(confirm=args.confirm)
            if not success:
                sys.exit(1)

        if args.reset_failed:
            reset_count = manager.reset_failed()

        if args.populate:
            added = manager.populate_queue(limit=args.limit)

        if args.show_failed:
            limit = args.limit or 20
            manager.show_failed_papers(limit=limit)

        if args.show_successes:
            limit = args.limit or 10
            manager.show_recent_successes(limit=limit)

        if args.stats:
            manager.show_stats()

        print(f"\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logfire.error(f"Main error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()