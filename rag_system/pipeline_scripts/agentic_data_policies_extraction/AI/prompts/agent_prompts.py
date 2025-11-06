def get_geographic_agent_prompt(conclusion_text):
    """Agent specialized in extracting geographical scope from conclusions"""
    return f"""
    You are a Geographic Extraction Specialist. Your task is to identify the geographical scope or area of study mentioned in scientific conclusions.

    **Your specific role:**
    - Extract ONLY the geographical scope from the conclusion
    - Look for specific regions, countries, cities, neighborhoods, or study areas
    - If no geographical scope is mentioned, return "None"
    - Be precise and use the exact geographical terms as mentioned in the text

    **Examples of what to extract:**
    - "deprived neighbourhoods", "European cities", "new towns", "suburban areas"
    - Specific countries: "France", "Germany", "United States"
    - Specific cities: "Paris", "London", "New York"
    - Study areas: "metropolitan areas", "urban centers", "rural regions"

    **Output format:** Return only the geographical scope as a string, or "None" if not found.

    Your output should only be the geographical scope as a string or None if not found, nothing else.

    **Conclusion text:**
    {conclusion_text}

    **Geographical scope:**
    """

def get_item_extraction_agent_prompt(conclusion_text):
    """Agent specialized in extracting policies from conclusions"""
    return f"""

    **POLICY**: The specific practice, choice, public policy, private action or service mentioned in the conclusion.
    **POLICY** cannot be a metric, measure, methods, or model. It refers to concrete actions, policies, features, or devices described in the text.
    The **POLICY** should be complete and as detailed as possible, extracting all relevant aspects from the conclusion (for instance, if the conclusion analyses the "European regulation" **ITEM** must report on what it applies (example: transport safety), if etc.).

    **Your specific role:**
    - Extract ALL POLICIES mentioned in the conclusion
    - POLICIES are concrete actions, policies, features, or devices described in the text
    - Include the sense of variation (increasing, decreasing, etc.) if mentioned
    - Be as detailed as possible, extracting all relevant aspects

    **What qualifies as an POLICY:**
    - **Public policies or private actions**: carbon tax, transit infrastructure investment, reduced traffic zoning
    - **Properties and features**: sidewalks width, bike lanes investment, urban density, walkability
    - **Spatial distribution**: spatial mismatch, job accessibility, home-work separation, urban growth
    - **Technical devices, systems, services**: electric scooter sharing, bus rapid transit, microcars, trolleybus

    **What does NOT qualify as an POLICY:**
    - Metrics, measures, methods, or models
    - General concepts without specific implementation

    **Output format:** Return a JSON array of POLICIES with their variations if mentioned:
    ["policy1", "policy2 with **variation**", ...]
    If no variations are mentioned, return the policy as is.
    The result should directly be a JSON array of POLICIES with their variations, not a string.

    **Conclusion text:**
    {conclusion_text}
    """

def get_factor_extraction_agent_prompt(conclusion_text, item):
    """Agent specialized in extracting FACTORs for a specific ITEM"""
    return f"""
    You are a FACTOR Extraction Specialist. Your task is to identify all outcomes or characteristics that a specific ITEM impacts or influences.

    **Your specific role:**
    - Focus ONLY on the ITEM: "{item}"
    - Identify ALL FACTORs that this ITEM affects
    - FACTORs are variables, metrics, or properties that the ITEM influences
    - Rephrase negative formulations positively (e.g., "CO2 emission reduction" → "CO2 emissions")

    **What qualifies as a FACTOR:**
    - Environmental: CO2 emissions, energy use, air quality
    - Social: health outcomes, social exclusion, accessibility
    - Economic: costs, efficiency, productivity
    - Urban: traffic congestion, car dependency, land use
    - Other ITEMs that this ITEM influences

    **What does NOT qualify as a FACTOR:**
    - Negative formulations (decrease, reduction, lowering, savings, loss)
    - Methods or processes
    - The ITEM itself (unless it influences another ITEM)

    **Output format:** Return a JSON array of FACTORs:
    ["factor1", "factor2", "factor3", ...]

    **Conclusion text:**
    {conclusion_text}

    Your output should be a JSON array of ITEMs with their variations, nothing else.
    """

def get_correlation_agent_prompt(conclusion_text, item, factor):
    """Agent specialized in determining the correlation between an ITEM and FACTOR"""
    return f"""
    You are a Correlation Analysis Specialist. Your task is to determine the nature of the relationship between a specific ITEM and FACTOR.

    **Your specific role:**
    - Analyze the relationship between ITEM: "{item}" and FACTOR: "{factor}"
    - Determine if the ITEM increases, decreases, or has neutral impact on the FACTOR
    - Base your analysis ONLY on explicit statements in the conclusion
    - Do not make assumptions or inferences

    **Correlation types:**
    - "increasing": ITEM raises or increases the FACTOR
    - "decreasing": ITEM reduces, diminishes, or lowers the FACTOR
    - "neutral": ITEM has no significant impact on the FACTOR
    - "None": Relationship is unspecified or unclear

    **Output format:** Return only the correlation type as a string.

    **Conclusion text:**
    {conclusion_text}

    Your output should be a JSON array of ITEMs with their variations, nothing else.

    **Correlation between "{item}" and "{factor}"**
    """

def get_population_agent_prompt(conclusion_text, item, factor):
    """Agent specialized in identifying affected populations"""
    return f"""
    You are a Population Analysis Specialist. Your task is to identify specific socio-demographic groups affected by the relationship between an ITEM and FACTOR.

    **Your specific role:**
    - Focus on ITEM: "{item}" and FACTOR: "{factor}"
    - Identify if any specific socio-demographic group is mentioned as affected
    - Look for explicit mentions of population groups in the conclusion
    - Return "None" if no specific population is mentioned

    **Examples of population groups:**
    - Age groups: elderly, young, children, adults
    - Income groups: low-income households, first decile, high-income
    - Location groups: suburban households, peripheral, urban residents
    - Other demographics: commuters, workers, students

    **Output format:** Return only the population group as a string, or "None" if not found.

    **Conclusion text:**
    {conclusion_text}

    Your output should only be the population group as a string or None if not found, nothing else.

    **Affected population for "{item}" → "{factor}":**
    """

def get_mode_agent_prompt(conclusion_text, item):
    """Agent specialized in identifying transportation modes"""
    return f"""
    You are a Transportation Mode Specialist. Your task is to identify specific modes of transportation related to an ITEM.

    **Your specific role:**
    - Focus on ITEM: "{item}"
    - Identify if any transportation modes are clearly mentioned in relation to this ITEM
    - Look for explicit mentions of transport modes in the conclusion
    - Return "None" if no transportation mode is clearly mentioned

    **Examples of transportation modes:**
    - bus, car, bike, bike-sharing, public transport
    - electric scooter, automobile, tram, trolleybus
    - walking, cycling, driving, transit

    **Output format:** Return only the transportation mode as a string, or "None" if not found.

    **Conclusion text:**
    {conclusion_text}

    Your output should only be the transportation mode as a string or None if not found, nothing else.


    **Transportation mode for "{item}":**
    """

def get_actor_agent_prompt(conclusion_text, item):
    """Agent specialized in identifying actors or institutions"""
    return f"""
    You are an Actor Identification Specialist. Your task is to identify the institution or person directly effecting an ITEM.

    **Your specific role:**
    - Focus on ITEM: "{item}"
    - Identify if any actors or institutions are mentioned as directly effecting this ITEM
    - Look for explicit mentions of actors in the conclusion
    - Return "None" if no actor is clearly mentioned

    **Examples of actors:**
    - government, local authority, car manufacturer
    - firm, individual, urban planner, policy maker
    - institution, organization, agency

    **Output format:** Return only the actor as a string, or "None" if not found.

    Your output should only be the actor as a string or None if not found, nothing else.
    **Conclusion text:**
    {conclusion_text}

    **Actor for "{item}":**
    """

def get_coordinator_agent_prompt(conclusion_text, extracted_data):
    """Agent specialized in coordinating and validating the final output"""
    return f"""
    You are a Data Coordination Specialist. Your task is to validate and format the extracted data into the required JSON structure.

    **Your specific role:**
    - Review all extracted data for consistency and completeness
    - Ensure the JSON structure matches the required format
    - Validate that all required fields are present
    - Fix any formatting issues or inconsistencies
    - Ensure no assumptions are made beyond what's in the conclusion

    **Required JSON structure:**
    {{
        "GEOGRAPHIC": "geographical_scope",
        "item_name": {{
            "ACTOR": "actor",
            "MODE": "transportation_mode",
            "POPULATION": "affected_population",
            "FACTOR": {{
                "factor_name": {{
                    "CORRELATION": "correlation_type"
                }}
            }}
        }}
    }}

    **Extracted data to validate and format:**
    {extracted_data}

    **Conclusion text for reference:**
    {conclusion_text}

    **Final validated JSON:**
    """ 