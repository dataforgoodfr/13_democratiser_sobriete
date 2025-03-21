from enum import Enum

class Human_needs(str, Enum):
    nutrition = "Nutrition"
    shelter_and_living_conditions = "Shelter and living conditions"
    hygiene = "Hygiene"
    clothing = "Clothing"
    healthcare = "Healthcare"
    education = "Education"
    communication_and_information = "Communication and information"
    mobility = "Mobility"

class Studied_sector(str, Enum):
    agriculture_forestry_fishing = "Agriculture, forestry, fishing"
    mining_and_quarrying = "Mining and quarrying"
    manufacturing = "Manufacturing"
    electricity_gas_steam_air_conditioning = "Electricity, gas, steam, air conditioning"
    water_sewerage_waste = "Water, sewerage, waste"
    construction = "Construction"
    trade_vehicle_motorcycle_repair = "Trade; vehicle and motorcycle repair"
    transport_storage = "Transport and storage"
    accommodation_food_services = "Accommodation and food services"
    information_communication = "Information and communication"
    financial_insurance_services = "Financial and insurance services"
    real_estate = "Real estate"
    professional_scientific_technical_services = "Professional, scientific, technical services"
    administrative_support_services = "Administrative and support services"
    public_administration_defence_social_security = (
        "Public administration, defence, social security"
    )
    education = "Education"
    health_social_work = "Health and social work"
    arts_entertainment_recreation = "Arts, entertainment, recreation"
    other_services = "Other services"
    household_employment_goods_services_for_own_use = (
        "Household employment, goods/services for own use"
    )
    extraterritorial_organizations = "Extraterritorial organizations"


class Studied_policy_area(str, Enum):
    jobs = "Jobs"
    social_rights = "Social Rights"
    economy = "Economy"
    agriculture = "Agriculture"
    cohesion = "Cohesion"
    reforms = "Reforms"
    health = "Health"
    food = "Food"
    justice = "Justice"
    equality = "Equality"
    home_affairs = "Home Affairs"
    energy = "Energy"
    finance_and_capital_markets = "Finance and Capital Markets"
    innovation_and_research = "Innovation and Research"
    culture = "Culture"
    education_and_youth = "Education and Youth"
    climate_action = "Climate Action"


class Natural_ressource(str, Enum):
    freshwater = "Freshwater"
    marine_resources = "Marine Resources"
    wetlands = "Wetlands"
    metals_and_ores = "Metals and Ores"
    non_metallic_minerals = "Non-Metallic Minerals"
    fossil_fuels = "Fossil Fuels"
    agricultural_land = "Agricultural Land"
    forests = "Forests"
    urban_land = "Urban Land"
    biomass = "Biomass"


class Wellbeing(str, Enum):
    housing = "Housing"
    jobs = "Jobs"
    education = "Education"
    civic_engagement = "Civic Engagement"
    life_satisfaction = "Life Satisfaction"
    work_life_balance = "Work-Life Balance"
    income = "Income"
    community = "Community"
    environment = "Environment"
    health = "Health"
    safety = "Safety"


class Justice_consideration(str, Enum):
    distributional = "Distributional"
    procedural = "Procedural"
    corrective = "Corrective"
    recognitional = "Recognitional"
    transitional = "Transitional"


class Planetary_boundaries(str, Enum):
    land_system_change = "Land-System Change"
    climate_change = "Climate Change"
    biosphere_integrity = "Biosphere Integrity"
    biogeochemical_flows = "Biogeochemical Flows"
    ocean_acidification = "Ocean Acidification"
    freshwater_use = "Freshwater Use"
    atmospheric_aerosol_loading = "Atmospheric Aerosol Loading"
    ozone_depletion = "Ozone Depletion"
    introduction_of_novel_entities = "Introduction of Novel Entities"