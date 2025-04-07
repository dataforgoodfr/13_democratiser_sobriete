import os
from sqlmodel import SQLModel, Field, create_engine, Session
from sqlalchemy import JSON, Column
from taxonomy.paper_taxonomy import PaperTaxonomy

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    None
)


class ArticleMetadata(SQLModel, table=True):
    __tablename__ = "article_metadata"

    id: int = Field(default=None, primary_key=True)
    title: str = Field(unique=True)
    authors: list = Field(sa_column=Column(JSON))
    abstract: str
    year_of_publication: int
    peer_reviewed: bool
    grey_literature: bool
    publication_type: str
    sufficiency_mentioned: bool
    science_type: str
    scientific_discipline: str
    regional_group: str
    geographical_scope: str
    keywords: list = Field(sa_column=Column(JSON))
    url: str | None = None
    doi: str | None = None
    source: str | None = None
    source_type: str | None = None
    source_url: str | None = None
    source_doi: str | None = None
    source_publication_date: str | None = None
    source_access_date: str | None = None
    source_publication_type: str | None = None
    source_language: str | None = None
    source_publisher: str | None = None
    source_publisher_location: str | None = None
    source_publisher_country: str | None = None
    source_publisher_contact: str | None = None
    source_publisher_contact_email: str | None = None


# Create database engine and tables
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SQLModel.metadata.create_all(engine)


def persist_article_metadata(article: PaperTaxonomy) -> int:
    """
    Persist the article metadata in the database via sqlalchemy.
    
    Args:
        article (PaperTaxonomy): The article metadata to persist
        
    Returns:
        int: The ID of the newly created article record
    """
    with Session(engine) as session:
        # Convert the article to a dictionary
        article_dict = article.model_dump()

        # Create a new ArticleMetadata instance
        db_article = ArticleMetadata(**article_dict)

        # Add and commit to database
        session.add(db_article)
        session.commit()
        session.refresh(db_article)

        return db_article.id
