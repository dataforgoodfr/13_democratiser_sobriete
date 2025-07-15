from typing import Optional, Any
from pydantic import BaseModel

from taxonomy.publication_taxonomy import (
    Author,
    # Author_gender,
    Publication_type,
    Science_type,
    Scientif_discipline,
)
from taxonomy.geographical_taxonomy import (
    Regional_group,
    Geographical_scope,
)

from taxonomy.geographical_taxonomy import Studied_country
from taxonomy.themes_taxonomy import Human_needs, Studied_sector, Studied_policy_area, Natural_ressource, \
    Wellbeing, Justice_consideration, Planetary_boundaries


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
    studied_country: list[Studied_country]
    human_needs: list[Human_needs]
    studied_sector: list[Studied_sector]
    studied_policy_area: list[Studied_policy_area]
    natural_ressource: list[Natural_ressource]
    wellbeing: list[Wellbeing]
    justice_consideration: Optional[list[Justice_consideration]]
    planetary_boundaries: Optional[list[Planetary_boundaries]]

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
    pass


class PaperWithText(BaseModel):
    extract_text: str
    embeddings: Optional[Any]
