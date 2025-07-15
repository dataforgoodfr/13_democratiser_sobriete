import os
from datetime import datetime

from sqlmodel import SQLModel, Field, create_engine, Session
from sqlalchemy import JSON, Column, ARRAY, String, DateTime, Integer, Float, Boolean
from taxonomy.paper_taxonomy import PaperTaxonomy

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://u4axloluqibskgvdikuy:g2rXgpHSbztokCbFxSyR@bk8htvifqendwt1wlzat-postgresql.services.clever-cloud.com:7327/bk8htvifqendwt1wlzat"
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
    studied_country: list = Field(sa_column=Column(ARRAY(String)))
    human_needs: list = Field(sa_column=Column(ARRAY(String)))
    studied_sector: list = Field(sa_column=Column(ARRAY(String)))
    studied_policy_area: list = Field(sa_column=Column(ARRAY(String)))
    natural_ressource: list = Field(sa_column=Column(ARRAY(String)))
    wellbeing: list = Field(sa_column=Column(ARRAY(String)))
    justice_consideration: list | None = Field(default=None, sa_column=Column(ARRAY(String)))
    planetary_boundaries: list | None = Field(default=None, sa_column=Column(ARRAY(String)))
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


class OpenAlexArticle(SQLModel, table=True):
    __tablename__ = "all_articles_mobility"
    id: int = Field(default=None, primary_key=True)
    openalex_id: str = Field(sa_column=Column(String))
    doi: str = Field(sa_column=Column(String))
    title: str = Field(sa_column=Column(String))
    publication_date: datetime = Field(sa_column=Column(DateTime))
    updated_date: datetime = Field(sa_column=Column(DateTime))
    created_date: datetime = Field(sa_column=Column(DateTime))
    grants: str = Field(sa_column=Column(JSON))
    language: str = Field(sa_column=Column(String))
    type: str = Field(sa_column=Column(String))
    type_crossref: str = Field(sa_column=Column(String))
    indexed_in: str = Field(sa_column=Column(String))
    cited_by_count: int = Field(sa_column=Column(Integer))
    citation_normalized_percentile: float = Field(sa_column=Column(Float))
    is_retracted: bool = Field(sa_column=Column(Boolean))
    is_paratext: bool = Field(sa_column=Column(Boolean))
    abstract: str = Field(sa_column=Column(String))
    topics: str = Field(sa_column=Column(String))
    concepts: str = Field(sa_column=Column(String))
    keywords: str = Field(sa_column=Column(String))
    openaccess_url: str = Field(sa_column=Column(String))
    # is_oa: bool = Field(sa_column=Column(Boolean))
    successfully_downloaded: bool = Field(sa_column=Column(Boolean))
    sustainable_development_goals: str = Field(sa_column=Column(String))
    author_ids: str = Field(sa_column=Column(String))
    institution_ids: str = Field(sa_column=Column(String))

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


def get_open_alex_article(pdf_path):
    """
    Retrieve an article from the database based on the OpenAlex ID extracted from a PDF filename.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        OpenAlexArticle: The article from the database if found, None otherwise
    """
    # Extract the filename without extension to get the OpenAlex ID
    openalex_id = os.path.splitext(os.path.basename(pdf_path))[0]

    with Session(engine) as session:
        # Query the database for the article with the matching OpenAlex ID
        article = session.query(OpenAlexArticle).filter(OpenAlexArticle.openalex_id == openalex_id).first()

        return article


def reconcile_metadata(openalex_metadata, llm_metadata):
    """
    Reconcile OpenAlex metadata with LLM-generated metadata.
    Preserves all values from OpenAlex metadata and only adds new values from LLM output.
    
    Args:
        openalex_metadata (OpenAlexArticle): The metadata from OpenAlex
        llm_metadata (PaperTaxonomy): The metadata generated by the LLM
        
    Returns:
        PaperTaxonomy: A merged PaperTaxonomy object with all OpenAlex values preserved
    """
    # Convert both to dictionaries for easier manipulation
    openalex_dict = openalex_metadata.model_dump() if openalex_metadata else {}
    llm_dict = llm_metadata.model_dump()
    
    # Create a new dictionary for the reconciled metadata
    reconciled_dict = {}
    
    # First, copy all fields from the LLM output
    for key, value in llm_dict.items():
        reconciled_dict[key] = value
    
    # Then, override with OpenAlex values where they exist
    # Map OpenAlex fields to PaperTaxonomy fields
    field_mapping = {
        'title': 'title',
        'doi': 'doi',
        'abstract': 'abstract',
        'publication_date': 'year_of_publication',  # Extract year from datetime
        'type': 'publication_type',
        'language': 'source_language',
        'openaccess_url': 'url',
        'keywords': 'keywords',
    }
    
    for openalex_key, taxonomy_key in field_mapping.items():
        if openalex_key in openalex_dict and openalex_dict[openalex_key] is not None:
            # Special handling for publication_date to extract year
            if openalex_key == 'publication_date':
                if isinstance(openalex_dict[openalex_key], datetime):
                    reconciled_dict[taxonomy_key] = openalex_dict[openalex_key].year
                elif isinstance(openalex_dict[openalex_key], str):
                    # Handle string format like "2023-11-13 00:00:00.000"
                    try:
                        year = int(openalex_dict[openalex_key].split('-')[0])
                        reconciled_dict[taxonomy_key] = year
                    except (ValueError, IndexError):
                        # If parsing fails, keep the LLM value
                        pass
            # Special handling for publication_type to map to valid enum values
            elif openalex_key == 'type':
                # Map OpenAlex type to valid PaperTaxonomy publication_type values
                type_mapping = {
                    'article': 'Research article',
                    'journal-article': 'Research article',
                    'book': 'Book',
                    'book-chapter': 'Chapter',
                    'conference-paper': 'Conference Paper',
                    'data-paper': 'Data paper',
                    'editorial': 'Editorial',
                    'erratum': 'Erratum',
                    'letter': 'Letter',
                    'note': 'Note',
                    'retracted-article': 'Retracted article',
                    'review': 'Review',
                    'short-survey': 'Short survey',
                    'commentary': 'Commentary ',
                    'presentation': 'Presentation',
                    'technical-report': 'Technical report',
                    'policy-report': 'Policy report',
                    'policy-brief': 'Policy brief',
                    'factsheet': 'Factsheet',
                }
                # Default to 'Research article' if no mapping found
                reconciled_dict[taxonomy_key] = type_mapping.get(openalex_dict[openalex_key], 'Research article')
            # Special handling for keywords to convert from string to list
            elif openalex_key == 'keywords':
                if isinstance(openalex_dict[openalex_key], str):
                    # Split by pipe and extract just the keyword part (before the colon)
                    keywords = []
                    for kw in openalex_dict[openalex_key].split('|'):
                        if ':' in kw:
                            keyword = kw.split(':')[0].strip()
                            if keyword:
                                keywords.append(keyword)
                        else:
                            if kw.strip():
                                keywords.append(kw.strip())
                    reconciled_dict[taxonomy_key] = keywords
                elif isinstance(openalex_dict[openalex_key], list):
                    reconciled_dict[taxonomy_key] = openalex_dict[openalex_key]
            else:
                reconciled_dict[taxonomy_key] = openalex_dict[openalex_key]
    
    # Create a new PaperTaxonomy instance with the reconciled data
    return PaperTaxonomy(**reconciled_dict)
