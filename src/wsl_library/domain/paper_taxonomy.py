from typing import Optional, Any
from pydantic import BaseModel

from wsl_library.domain.publication_taxonomy import (
    Author,
    # Author_gender,
    Publication_type,
    Science_type,
    Scientif_discipline,
)
from wsl_library.domain.geographical_taxonomy import (
    Regional_group,
    Geographical_scope,
    Studied_country,
)
from wsl_library.domain.themes_taxonomy import (
    Human_needs,
    Studied_sector,
    Studied_policy_area,
    Natural_ressource,
    Wellbeing,
    Justice_consideration,
    Planetary_boundaries,
)


class PaperTaxonomy(BaseModel):
    title: str
    authors: list[Author]
    abstract: str
    year_of_publication: int
    peer_reviewed: bool
    grey_literature: bool
    publication_type: Publication_type
    sufficiency_mentioned: bool
    science_type: Science_type
    scientific_discipline: Scientif_discipline
    regional_group: Regional_group
    geographical_scope: Geographical_scope
    studied_country: set[Studied_country]
    human_needs: set[Human_needs]
    studied_sector: set[Studied_sector]
    studied_policy_area: set[Studied_policy_area]
    natural_ressource: set[Natural_ressource]
    wellbeing: set[Wellbeing]
    justice_consideration: Optional[set[Justice_consideration]]
    planetary_boundaries: Optional[set[Planetary_boundaries]]

    ## Optional fields
    keywords: list[str]
    url: Optional[str]
    doi: Optional[str]
    source: Optional[str]
    source_type: Optional[str]
    source_url: Optional[str]
    source_doi: Optional[str]
    source_publication_date: Optional[str]
    source_access_date: Optional[str]
    source_publication_type: Optional[str]
    source_language: Optional[str]
    source_publisher: Optional[str]
    source_publisher_location: Optional[str]
    source_publisher_country: Optional[str]
    source_publisher_contact: Optional[str]
    source_publisher_contact_email: Optional[str]


class OpenAlexPaper(BaseModel):
    paper_name: str
    metadata_path: str
    pdf_path: str | None  # required, can be None
    metadata: dict | None = None # not required, can be None

# Doc for Optional / Nullable : https://docs.pydantic.dev/latest/migration/#required-optional-and-nullable-fields
class PaperWithText(BaseModel):
    openalex_paper: OpenAlexPaper
    extract_text: str
    extrated_object: list | None = None # not required, can be None
    embeddings: Any | None = None # not required, can be None
