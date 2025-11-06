import json
import logging
from typing import Dict, List, Any
from .main_handler import get_client, get_response
from ..prompts.agent_prompts import (
    get_geographic_agent_prompt,
    get_item_extraction_agent_prompt,
    get_factor_extraction_agent_prompt,
    get_correlation_agent_prompt,
    get_population_agent_prompt,
    get_mode_agent_prompt,
    get_actor_agent_prompt,
    get_coordinator_agent_prompt
)

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """Orchestrates multiple specialized agents for high-quality data extraction"""
    
    def __init__(self):
        self.client = get_client()
        
    def _call_agent(self, prompt: str) -> str:
        """Call a single agent with a specific prompt"""
        try:
            response = get_response(self.client, prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error calling agent: {e}")
            return "None"
    
    def _parse_json_array(self, response: str) -> List[str]:
        """Parse JSON array response from agents"""
        try:
            # Clean the response and parse as JSON
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            parsed = json.loads(cleaned_response)

            print(parsed)
            if isinstance(parsed, list):
                return parsed
            else:
                logger.warning(f"Expected JSON array, got: {type(parsed)}")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response}")
            return []
    
    def extract_geographic_scope(self, conclusion_text: str) -> str:
        """Extract geographical scope using specialized agent"""
        prompt = get_geographic_agent_prompt(conclusion_text)
        return self._call_agent(prompt)
    
    def extract_items(self, conclusion_text: str) -> List[str]:
        """Extract ITEMs using specialized agent"""
        prompt = get_item_extraction_agent_prompt(conclusion_text)
        response = self._call_agent(prompt)
        return self._parse_json_array(response)
    
    def extract_factors(self, conclusion_text: str, item: str) -> List[str]:
        """Extract FACTORs for a specific ITEM using specialized agent"""
        prompt = get_factor_extraction_agent_prompt(conclusion_text, item)
        response = self._call_agent(prompt)
        return self._parse_json_array(response)
    
    def determine_correlation(self, conclusion_text: str, item: str, factor: str) -> str:
        """Determine correlation between ITEM and FACTOR using specialized agent"""
        prompt = get_correlation_agent_prompt(conclusion_text, item, factor)
        return self._call_agent(prompt)
    
    def identify_population(self, conclusion_text: str, item: str, factor: str) -> str:
        """Identify affected population using specialized agent"""
        prompt = get_population_agent_prompt(conclusion_text, item, factor)
        return self._call_agent(prompt)
    
    def identify_transport_mode(self, conclusion_text: str, item: str) -> str:
        """Identify transportation mode using specialized agent"""
        prompt = get_mode_agent_prompt(conclusion_text, item)
        return self._call_agent(prompt)
    
    def identify_actor(self, conclusion_text: str, item: str) -> str:
        """Identify actor using specialized agent"""
        prompt = get_actor_agent_prompt(conclusion_text, item)
        return self._call_agent(prompt)
    
    def coordinate_and_validate(self, conclusion_text: str, extracted_data: Dict) -> Dict:
        """Coordinate and validate final output using specialized agent"""
        prompt = get_coordinator_agent_prompt(conclusion_text, json.dumps(extracted_data, indent=2))
        response = self._call_agent(prompt)
        
        try:
            # Try to parse the response as JSON
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse coordinator response as JSON: {response}")
            return extracted_data
    
    def extract_data_with_agents(self, conclusion_text: str) -> Dict[str, Any]:
        """Main method to extract data using the multi-agent system"""
        logger.info("Starting multi-agent data extraction...")
        
        # Step 1: Extract geographical scope
        logger.info("Step 1: Extracting geographical scope...")
        geographic_scope = self.extract_geographic_scope(conclusion_text)
        
        # Step 2: Extract all ITEMs
        logger.info("Step 2: Extracting ITEMs...")
        items = self.extract_items(conclusion_text)
        
        if not items:
            logger.warning("No ITEMs found in conclusion")
            return {"GEOGRAPHIC": geographic_scope}
        
        # Step 3: For each ITEM, extract all related information
        result = {"GEOGRAPHIC": geographic_scope}
        
        for item in items:
            logger.info(f"Processing ITEM: {item}")
            
            # Extract factors for this item
            factors = self.extract_factors(conclusion_text, item)
            
            # Extract actor and mode for this item
            actor = self.identify_actor(conclusion_text, item)
            mode = self.identify_transport_mode(conclusion_text, item)
            
            item_data = {
                "ACTOR": actor,
                "MODE": mode,
                "POPULATION": "None",  # Will be updated per factor
                "FACTOR": {}
            }
            
            # For each factor, determine correlation and population
            for factor in factors:
                logger.info(f"  Processing FACTOR: {factor}")
                
                correlation = self.determine_correlation(conclusion_text, item, factor)
                population = self.identify_population(conclusion_text, item, factor)
                
                item_data["FACTOR"][factor] = {
                    "CORRELATION": correlation
                }
                
                # Update population if found for this factor
                if population != "None":
                    item_data["POPULATION"] = population
            
            result[item] = item_data
        
        # Step 4: Coordinate and validate final output
        logger.info("Step 4: Coordinating and validating final output...")
        final_result = self.coordinate_and_validate(conclusion_text, result)
        
        logger.info("Multi-agent extraction completed successfully")
        return final_result

def get_agent_orchestrator() -> AgentOrchestrator:
    """Factory function to get an agent orchestrator instance"""
    return AgentOrchestrator() 