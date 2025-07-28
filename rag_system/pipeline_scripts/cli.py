#!/usr/bin/env python3
"""
Paper Processing Pipeline CLI
Unified command-line interface for scraping, queue management, and testing
"""

from typing import Optional
import typer
from rich.console import Console

app = typer.Typer(
    name="pipeline",
    help="Paper Processing Pipeline - Unified CLI for scraping, queue management, and testing",
    no_args_is_help=True,
    rich_markup_mode="rich"
)

console = Console()

@app.command()
def test():
    """
    🧪 Run scraping tests with 10 papers
    
    This command will:
    - Create test database queue
    - Test scraping functionality  
    - Create folder structure
    - Show you what to expect
    """
    try:
        from scraping.test_scraping import main as test_main
        console.print("🧪 Running scraping tests...", style="bold blue")
        test_main()
    except ImportError as e:
        console.print(f"❌ Failed to import test module: {e}", style="bold red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Test error: {e}", style="bold red")
        raise typer.Exit(1)

@app.command()
def queue(
    stats: bool = typer.Option(False, "--stats", help="Show queue statistics"),
    populate: bool = typer.Option(False, "--populate", help="Populate queue from database"),
    clear: bool = typer.Option(False, "--clear", help="Clear entire queue"),
    reset_failed: bool = typer.Option(False, "--reset-failed", help="Reset failed entries to pending"),
    show_failed: bool = typer.Option(False, "--show-failed", help="Show failed papers"),
    show_successes: bool = typer.Option(False, "--show-successes", help="Show recent successes"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Limit number of papers/results"),
    confirm: bool = typer.Option(False, "--confirm", help="Skip confirmation prompts")
):
    """
    🗄️ Manage scraping queue database with DOI retry support
    
    Examples:
    - [cyan]python cli.py queue --stats[/cyan] - Show statistics
    - [cyan]python cli.py queue --populate --limit 1000[/cyan] - Add 1000 papers
    - [cyan]python cli.py queue --clear[/cyan] - Clear entire queue
    - [cyan]python cli.py queue --reset-failed[/cyan] - Reset failed entries
    - [cyan]python cli.py queue --show-failed --limit 50[/cyan] - Show failed papers
    """
    try:
        from database.manage_queue import QueueManager
        
        manager = QueueManager()
        
        # If no specific arguments, show stats by default
        if not any([stats, populate, clear, reset_failed, show_failed, show_successes]):
            stats = True
        
        # Execute requested actions
        if clear:
            success = manager.clear_queue(confirm=confirm)
            if not success:
                raise typer.Exit(1)
                
        if reset_failed:
            reset_count = manager.reset_failed()
            
        if populate:
            added = manager.populate_queue(limit=limit)
            
        if show_failed:
            limit_val = limit or 20
            manager.show_failed_papers(limit=limit_val)
            
        if show_successes:
            limit_val = limit or 10
            manager.show_recent_successes(limit=limit_val)
            
        if stats:
            manager.show_stats()
            
    except ImportError as e:
        console.print(f"❌ Failed to import queue management module: {e}", style="bold red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Queue management error: {e}", style="bold red")
        raise typer.Exit(1)

@app.command()
def scrape(
    batch_size: int = typer.Option(10, "--batch-size", help="Number of papers to scrape"),
    output_dir: str = typer.Option("./scraping_output", "--output-dir", help="Output directory"),
    max_wait_time: int = typer.Option(30, "--max-wait-time", help="Max wait time for downloads (seconds)"),
    test_paper: Optional[str] = typer.Option(None, "--test-paper", help="Test scraping a specific OpenAlex ID")
):
    """
    🌐 Run targeted paper scraping with DOI retry logic
    
    Automatically retries failed downloads using DOI if OpenAlex ID scraping fails.
    Papers are distributed across 12 folders for parallel metadata extraction.
    
    Examples:
    - [cyan]python cli.py scrape --batch-size 20[/cyan] - Scrape 20 papers from queue
    - [cyan]python cli.py scrape --test-paper "https://openalex.org/W123"[/cyan] - Test single paper
    - [cyan]python cli.py scrape --output-dir ./my_output[/cyan] - Custom output directory
    """
    try:
        from scraping.targeted_scraper import TargetedPaperScraper
        import logfire
        
        # Configure logfire
        LOGFIRE_TOKEN = "pylf_v1_us_qTtmbDFpkfhFwzTfZyZrTJcl4C4lC7FhmZ65BgJ7dLDV"
        logfire.configure(token=LOGFIRE_TOKEN)
        
        scraper = TargetedPaperScraper(
            base_output_dir=output_dir,
            max_wait_time=max_wait_time
        )
        
        if test_paper:
            console.print(f"🧪 Testing single paper: {test_paper}", style="bold blue")
            success = scraper.scrape_paper(test_paper)
            if success:
                console.print("✅ Test successful!", style="bold green")
            else:
                console.print("❌ Test failed", style="bold red")
        else:
            console.print("🚀 Starting targeted paper scraping...", style="bold blue")
            stats = scraper.scrape_batch(batch_size)
            
            console.print("\n📊 [bold]BATCH RESULTS:[/bold]")
            console.print(f"   ✅ Successful: {stats['successful']}", style="green")
            console.print(f"   ❌ Failed: {stats['failed']}", style="red")
            console.print(f"   📋 Total processed: {stats['processed']}", style="blue")
            
            if stats['successful'] > 0:
                console.print(f"\n🎉 Successfully scraped {stats['successful']} papers!", style="bold green")
                console.print(f"📁 Files distributed across folders in: {output_dir}")
                console.print("💡 Next: Run metadata extraction with [cyan]./run_metadata_extraction.sh[/cyan]")
                
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
    • [green]queue[/green] - Manage scraping queue database
    • [green]scrape[/green] - Run targeted paper scraping with DOI fallback
    
    [bold blue]Quick Examples:[/bold blue]
    
    • [cyan]python cli.py test[/cyan] - Test the system
    • [cyan]python cli.py queue --stats[/cyan] - Check queue status
    • [cyan]python cli.py scrape --batch-size 20[/cyan] - Scrape 20 papers with DOI retry
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