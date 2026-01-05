"""
Paper Processing Pipeline CLI
Unified command-line interface for scraping, queue management, and testing
"""

from typing import Optional
import typer
from rich.console import Console
from pathlib import Path

app = typer.Typer(
    name="pipeline",
    help="Paper Processing Pipeline - 5x faster parallel scraping, queue management, and testing",
    no_args_is_help=True,
    rich_markup_mode="rich"
)

console = Console()

@app.command()
def test():
    """
    🧪 Run scraping tests with small batch
    This command will:
    - Test scraping functionality with 10 papers
    - Create folder structure
    - Show you what to expect

    Equivalent to: [cyan]python cli.py scrape --batch-size 10[/cyan]
    """
    try:
        from ...tests.scraping.test_scraping import main as test_main
        console.print("🧪 Running scraping tests...", style="bold blue")
        test_main()
    except ImportError as e:
        console.print(f"❌ Failed to import test module: {e}", style="bold red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Test error: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def scrape(
    batch_size: int = typer.Option(100, "--batch-size", help="Number of papers to scrape per batch"),
    output_dir: str = typer.Option("./scraping_output", "--output-dir", help="Output directory"),
    max_wait_time: int = typer.Option(30, "--max-wait-time", help="Max wait time for downloads (seconds)"),
    workers: int = typer.Option(1, "--workers", help="Number of parallel Chrome workers (1=sequential, 5=recommended)"),
    test_paper: Optional[str] = typer.Option(None, "--test-paper", help="Test scraping a specific OpenAlex ID"),
    all_papers: bool = typer.Option(False, "--all", help="Process ALL papers in database continuously (resumable)"),
    quiet: bool = typer.Option(False, "--quiet", help="Reduce output for large batch processing"),
    stats: bool = typer.Option(False, "--stats", help="Show scraping statistics and progress"),
    clear: bool = typer.Option(False, "--clear", help="Clear the entire scraping queue")
):
    """
    🌐 Paper scraping with DOI retry logic and parallel processing
    Automatically retries failed downloads using DOI if OpenAlex ID scraping fails.
    Papers are distributed across 12 folders for parallel metadata extraction.
Fully resumable - can be interrupted and restarted without issues.

    🚀 PARALLEL SCRAPING: Use --workers 5 for ~5x faster scraping with multiple Chrome instances.

    [bold blue]Examples:[/bold blue]
    - [cyan]python cli.py scrape --all --workers 5 --batch-size 100[/cyan] - Parallel processing (FAST!)
    - [cyan]python cli.py scrape --workers 5 --batch-size 50[/cyan] - Single parallel batch
    - [cyan]python cli.py scrape --batch-size 50[/cyan] - Single worker (sequential)
    - [cyan]python cli.py scrape --test-paper "https://openalex.org/W123"[/cyan] - Test single paper
    - [cyan]python cli.py scrape --stats[/cyan] - Show scraping progress and statistics
    - [cyan]python cli.py scrape --clear[/cyan] - Clear the entire scraping queue
    - [cyan]python cli.py scrape --all --workers 5 --quiet[/cyan] - Fast parallel processing, minimal output
    """
    try:
        # Handle clear operation
        if clear:
            from database.models import clear_scraping_queue

            console.print("⚠️  [bold red]This will DELETE ALL entries in the scraping queue![/bold red]")
            if not typer.confirm("Are you sure you want to continue?"):
                console.print("❌ Operation cancelled", style="yellow")
                return

            cleared_count = clear_scraping_queue()
            if cleared_count > 0:
                console.print(f"🗑️  Cleared {cleared_count:,} entries from scraping queue", style="green")
            else:
                console.print("📭 Scraping queue was already empty", style="blue")

            # If only clear was requested, we're done
            if not test_paper and not all_papers and not stats:
                return

        # Handle stats display
        if stats:
            from database.models import get_scraping_stats

            stats_data = get_scraping_stats()
            console.print("📊 [bold]SCRAPING STATISTICS[/bold]")
            console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            console.print(f"📋 Total papers: {stats_data.get('total', 0):,}", style="blue")
            console.print(f"✅ Successfully scraped: {stats_data.get('scraped', 0):,}", style="green")
            console.print(f"❌ Failed: {stats_data.get('failed', 0):,}", style="red")
            console.print(f"⏳ Pending: {stats_data.get('pending', 0):,}", style="yellow")

            if stats_data.get('total', 0) > 0:
                completion_rate = (stats_data.get('scraped', 0) / stats_data['total']) * 100
                failure_rate = (stats_data.get('failed', 0) / stats_data['total']) * 100
                console.print(f"📈 Completion rate: {completion_rate:.1f}%", style="blue")
                console.print(f"📉 Failure rate: {failure_rate:.1f}%", style="blue")

            # If only stats was requested, we're done
            if not test_paper and not all_papers:
                return

        # Handle scraping operations
        from scraping.targeted_scraper import TargetedPaperScraper, ParallelPaperScraper
        import logfire

        # Validate workers parameter
        if workers < 1:
            console.print("❌ Error: Number of workers must be at least 1", style="bold red")
            raise typer.Exit(1)
        elif workers > 10:
            console.print("⚠️  Warning: Using more than 10 workers may overwhelm your system", style="yellow")
            if not typer.confirm("Continue anyway?"):
                console.print("❌ Operation cancelled", style="yellow")
                raise typer.Exit(0)

        # Configure logfire
        LOGFIRE_TOKEN = "pylf_v1_us_qTtmbDFpkfhFwzTfZyZrTJcl4C4lC7FhmZ65BgJ7dLDV"
        logfire.configure(token=LOGFIRE_TOKEN)
        
        # Choose scraper based on worker count
        if workers > 1:
            console.print(f"🚀 Using parallel scraper with {workers} Chrome workers", style="bold green")
            scraper = ParallelPaperScraper(
                base_output_dir=output_dir,
                max_wait_time=max_wait_time,
                num_workers=workers
            )
            is_parallel = True
        else:
            console.print("🔧 Using sequential scraper (single Chrome worker)", style="blue")
            scraper = TargetedPaperScraper(
                base_output_dir=output_dir,
                max_wait_time=max_wait_time
            )
            is_parallel = False
        
        if test_paper:
            console.print(f"🧪 Testing single paper: {test_paper}", style="bold blue")
            # Test paper only works with sequential scraper for now
            if is_parallel:
                console.print("💡 Test mode uses sequential scraper for single paper testing", style="dim")
                test_scraper = TargetedPaperScraper(
                    base_output_dir=output_dir,
                    max_wait_time=max_wait_time
                )
                success = test_scraper.scrape_paper(test_paper)
                test_scraper.close_driver()
            else:
                success = scraper.scrape_paper(test_paper)

            if success:
                console.print("✅ Test successful!", style="bold green")
            else:
                console.print("❌ Test failed", style="bold red")
        elif all_papers:
            if is_parallel:
                console.print(f"🚀 Starting parallel continuous scraping with {workers} workers...", style="bold blue")
            else:
                console.print("🚀 Starting continuous scraping of ALL papers in database...", style="bold blue")
            console.print("💡 [bold yellow]TIP:[/bold yellow] You can interrupt with Ctrl+C and resume later", style="dim")

            if is_parallel:
                stats = scraper.scrape_all_continuous_parallel(
                    batch_size=batch_size,
                    show_progress=not quiet
                )
            else:
                stats = scraper.scrape_all_continuous(
                    batch_size=batch_size,
                    show_progress=not quiet
                )

            # Final summary
            console.print("\n🎉 [bold]CONTINUOUS SCRAPING COMPLETED![/bold]")
            console.print(f"   ✅ Successfully scraped: {stats['total_successful']:,} papers", style="green")
            console.print(f"   📊 Batches completed: {stats['batches_completed']}", style="blue")
            console.print(f"   ⏱️  Total time: {stats['total_time']/60:.1f} minutes", style="blue")

            if is_parallel:
                console.print(f"   🔧 Workers used: {stats.get('workers_used', workers)}", style="blue")

            if stats['total_successful'] > 0:
                papers_per_hour = stats['total_successful'] / (stats['total_time'] / 3600)
                console.print(f"   📈 Average rate: {papers_per_hour:.0f} papers/hour", style="blue")

                if is_parallel and workers > 1:
                    estimated_single_rate = papers_per_hour / workers
                    console.print(f"   🚀 Speedup: ~{papers_per_hour/estimated_single_rate:.1f}x vs single worker", style="green")

                console.print(f"\n📁 Files distributed across 12 folders in: {output_dir}")
                console.print("💡 Next: Run metadata extraction with [cyan]./run_metadata_extraction.sh[/cyan]")
        else:
            if is_parallel:
                console.print(f"🚀 Starting parallel batch scraping of {batch_size} papers with {workers} workers...", style="bold blue")
                stats = scraper.scrape_batch_parallel(batch_size, show_progress=not quiet)
            else:
                console.print(f"🚀 Starting single batch scraping of {batch_size} papers...", style="bold blue")
                stats = scraper.scrape_batch(batch_size, show_progress=not quiet)
            console.print("\n📊 [bold]BATCH RESULTS:[/bold]")
            console.print(f"   ✅ Successful: {stats['successful']}", style="green")
            console.print(f"   ❌ Failed: {stats['failed']}", style="red")
            console.print(f"   📋 Total processed: {stats['processed']}", style="blue")

            if is_parallel:
                console.print(f"   🔧 Workers used: {stats.get('workers_used', workers)}", style="blue")

            if stats.get('total_time'):
                console.print(f"   ⏱️  Batch time: {stats['total_time']/60:.1f} minutes", style="blue")
                console.print(f"   📈 Success rate: {stats.get('success_rate', 0):.1f}%", style="blue")

                # Show speed metrics for parallel processing
                if is_parallel and stats.get('papers_per_second'):
                    console.print(f"   🚀 Speed: {stats['papers_per_second']:.1f} papers/second", style="green")
                    if workers > 1:
                        console.print(f"   📈 Estimated speedup: ~{stats['papers_per_second']*5:.1f}x vs single worker", style="green")

            if stats['successful'] > 0:
                if is_parallel:
                    console.print(f"\n🎉 Successfully scraped {stats['successful']} papers with {workers} parallel workers!", style="bold green")
                else:
                    console.print(f"\n🎉 Successfully scraped {stats['successful']} papers!", style="bold green")

                console.print("💡 [bold yellow]TIP:[/bold yellow] Use [cyan]--all --workers 5[/cyan] for fast continuous processing")
                console.print(f"📁 Files distributed across folders in: {output_dir}")

    except ImportError as e:
        console.print(f"❌ Failed to import scraping module: {e}", style="bold red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Scraping error: {e}", style="bold red")
        raise typer.Exit(1)

@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version information")
):
    """
    📄 Paper Processing Pipeline

    A production-ready system for scraping 250k academic papers and extracting metadata.
    Features DOI retry logic for improved scraping performance and resilience.

    [bold blue]Available Commands:[/bold blue]

    • [green]test[/green] - Run test scraping with 10 papers
    • [green]scrape[/green] - Paper scraping with progress reporting
    [bold blue]Quick Examples:[/bold blue]

    • [cyan]python cli.py test[/cyan] - Test the system
    • [cyan]python cli.py scrape --stats[/cyan] - Check scraping progress
    • [cyan]python cli.py scrape --all --batch-size 100[/cyan] - Process all papers (resumable)
    • [cyan]python cli.py scrape --clear[/cyan] - Clear scraping queue
    """
    if version:
        console.print("Paper Processing Pipeline CLI v2.0.0", style="bold blue")
        raise typer.Exit()

def cli_main():
    """Entry point for CLI"""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n⏹️  Operation cancelled by user", style="bold yellow")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}", style="bold red")
        raise typer.Exit(1)

if __name__ == "__main__":
    cli_main()