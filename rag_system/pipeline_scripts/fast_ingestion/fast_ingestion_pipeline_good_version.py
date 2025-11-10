import os
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool

from fast_ingestion.logging_config import configure_logging
from fast_ingestion.persist_taxonomy import (
    add_ingestion_column, get_open_alex_articles,
    get_open_alex_articles_not_ingested, update_article_ingestion_by_title)
from fast_ingestion.shortcut_indexing_pipeline import (
    DEFAULT_INGESTION_VERSION, IndexingPipelineShortCut)
from pydantic import BaseModel

# import logfire




# DOCSTORE_PATH = "/app/ktem_app_data/user_data/docstore" !!!
# Now, defined in the flowsettings.py (or in kotaemon for s3 path)

# LOGFIRE_TOKEN = "1*************************"

# qdrant_host = "116919ed-8e07-47f6-8f24-a22527d5d520.europe-west3-0.gcp.cloud.qdrant.io"
#  ! to report in flowsetting.py !

# Configure logging
logger = configure_logging()


class RelevanceScore(BaseModel):
    relevance_score: float


class ExtractionError(Exception):
    pass


def process_one_article(
    article, force_reindex: bool = False, ingestion_version: str = DEFAULT_INGESTION_VERSION
):
    title = article.title
    doi = article.doi

    article_metadatas = article.model_dump()

    logger.info(f"Processing article: {title} with DOI: {doi}")
    try:
        indexing_pipeline = IndexingPipelineShortCut()
        indexing_pipeline.get_resources_set_up()
        # 1. Kotaemon ingestion
        file_id = indexing_pipeline.run_one_file(
            file_path=doi,
            reindex=force_reindex,
            article_metadatas=article_metadatas,
            ingestion_version=ingestion_version,
        )
        logger.info(f"file_id : {file_id} - Ingestion result : OK")
        # 2. Update ingestion status on external postgres db
        update_article_ingestion_by_title(title=title, ingestion_version=ingestion_version)

    except Exception as e:
        logger.error(f"Error fetching OpenAlex article for DOI {doi}: {e}")


def main():
    parser = ArgumentParser(description="Run pdf ingestion")
    parser.add_argument(
        "-fr",
        "--force_reindex",
        action="store_true",
        help="Force to reindex all the pdf files in the folder",
    )
    parser.add_argument(
        "-iv",
        "--ingestion_version",
        default=DEFAULT_INGESTION_VERSION,
        help="Force the ingestion version",
    )
    parser.add_argument(
        "-nbp",
        "--nb_processes",
        default=os.cpu_count(),
        help="Nb processes for Multi-processing ingestion",
    )

    # logfire.configure(token=LOGFIRE_TOKEN)

    args = parser.parse_args()
    # file_path = args.file_path
    # folder_path = Path(file_path).parent # not used ?
    # logfire.notice("starting doc")
    # Add ingestion status column to external article db (if not exists)
    add_ingestion_column()

    if args.force_reindex:
        articles = get_open_alex_articles()
    else:
        articles = get_open_alex_articles_not_ingested(ingestion_status=args.ingestion_version)

    logger.info(
        f"Multi-Processing ingestion launched... (NB_PROCESSES = {int(args.nb_processes)})"
    )
    with Pool(processes=int(args.nb_processes)) as pool:
        pool.map(
            partial(
                process_one_article,
                force_reindex=args.force_reindex,
                ingestion_version=args.ingestion_version,
            ),
            articles,
        )


if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn' to avoid the lance fork-safety issue
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Method already set
        pass

    main()
