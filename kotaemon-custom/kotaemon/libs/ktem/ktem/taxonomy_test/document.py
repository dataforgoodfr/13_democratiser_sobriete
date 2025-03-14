from typing import Optional
from pydantic import BaseModel
from enum import Enum

# WSL Taxonomy

class author_gender(str, Enum):
    male = 'Male'
    female = 'Female'
    unknown = 'Unknown'

class author(BaseModel):
    name: str
    gender: Optional[author_gender]

class publication_type(str, Enum):
    research_article  = 'Research article'
    article_in_press  = 'Article-in-Press (AiP)'
    book              = 'Book'
    case_study        = 'Case study'
    chapter           = 'Chapter'
    conference_paper  = 'Conference Paper'
    data_paper        = 'Data paper'
    editorial         = 'Editorial'
    erratum           = 'Erratum'
    letter            = 'Letter'
    note              = 'Note'
    retracted_article = 'Retracted article'
    review            = 'Review'
    short_survey      = 'Short survey'
    commentary        = 'Commentary '
    presentation      = 'Presentation'
    technical_report  = 'Technical report'
    policy_report     = 'Policy report'
    policy_brief      = 'Policy brief'
    factsheet         = 'Factsheet'

class science_type(str, Enum):
    natural_science = 'Natural science'
    social_science  = 'Social science'
    formal_science  = 'Formal science'

class scientif_discipline(str, Enum):
    # Social science
    anthropology         = 'Anthropology'
    criminology          = 'Criminology'
    geography            = 'Geography'
    economics            = 'Economics'
    education_sciences   = 'Education sciences'
    linguistics          = 'Linguistics'
    military_science     = 'Military science'
    management_science   = 'Management science'
    organization_studies = 'Organization studies'
    political_science    = 'Political science'
    public_health        = 'Public health'
    psychology           = 'Psychology'
    religious_studies    = 'Religious studies'
    sociology            = 'Sociology'

    # Natural science
    biology           = 'Biology'
    earth_science     = 'Earth science'
    chemistry         = 'Chemistry'
    physics           = 'Physics'
    astronomy         = 'Astronomy'
    materials_science = 'Materials science'

    # Formal science
    logic               = 'Logic'
    mathematics         = 'Mathematics'
    statistics          = 'Statistics'
    systems_science     = 'Systems science'
    data_science        = 'Data science'
    information_science = 'Information science'
    computer_science    = 'Computer science'
    cryptography        = 'Cryptography'


class paper(BaseModel):
    title: str
    authors: list[author]
    abstract: str
    year_of_publication: int
    peer_reviewed: bool
    grey_literature: bool
    publication_type: publication_type