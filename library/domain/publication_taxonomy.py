from typing import Optional
from pydantic import BaseModel
from enum import Enum

class Author_gender(str, Enum):
    male = 'Male'
    female = 'Female'
    unknown = 'Unknown'

class Author(BaseModel):
    name: str
    gender: Optional[Author_gender]

class Publication_type(str, Enum):
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

class Science_type(str, Enum):
    natural_science = 'Natural science'
    social_science  = 'Social science'
    formal_science  = 'Formal science'

class Scientif_discipline(str, Enum):
    ## TODO : Add hierarchy between science_type and scientific_discipline
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
