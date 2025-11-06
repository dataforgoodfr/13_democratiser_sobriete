import json
import logging
from typing import Dict, Any
from .main_handler import get_client, get_response
from .agent_orchestrator import get_agent_orchestrator
from ..prompts.text_analyzer import get_prompt_extraction

logger = logging.getLogger(__name__)

class EnhancedMainHandler:
    """Enhanced handler that supports both single prompt and multi-agent approaches"""
    
    def __init__(self, use_agents: bool = True):
        self.use_agents = use_agents
        self.client = get_client()
        if use_agents:
            self.orchestrator = get_agent_orchestrator()
    
    def extract_data_single_prompt(self, conclusion_text: str) -> Dict[str, Any]:
        """Extract data using the original single prompt approach"""
        logger.info("Using single prompt approach...")
        
        prompt = get_prompt_extraction(conclusion_text)
        response = get_response(self.client, prompt)
        
        try:
            # Clean the response and parse as JSON
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse single prompt response as JSON: {e}")
            logger.error(f"Response was: {response}")
            return {"error": "Failed to parse response"}
    
    def extract_data_multi_agent(self, conclusion_text: str) -> Dict[str, Any]:
        """Extract data using the new multi-agent approach"""
        logger.info("Using multi-agent approach...")
        
        if not self.use_agents:
            raise ValueError("Multi-agent approach not enabled")
        
        return self.orchestrator.extract_data_with_agents(conclusion_text)
    
    def extract_data(self, conclusion_text: str, method: str = "auto") -> Dict[str, Any]:
        """Extract data using the specified method"""
        if method == "single":
            return self.extract_data_single_prompt(conclusion_text)
        elif method == "agents":
            return self.extract_data_multi_agent(conclusion_text)
        elif method == "auto":
            # Use agents by default if available, otherwise fall back to single prompt
            if self.use_agents:
                return self.extract_data_multi_agent(conclusion_text)
            else:
                return self.extract_data_single_prompt(conclusion_text)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'single', 'agents', or 'auto'")
    
    def compare_approaches(self, conclusion_text: str) -> Dict[str, Any]:
        """Compare single prompt vs multi-agent approaches"""
        logger.info("Comparing single prompt vs multi-agent approaches...")
        
        results = {}
        
        # # Single prompt approach
        # try:
        #     single_result = self.extract_data_single_prompt(conclusion_text)
        #     results["single_prompt"] = {
        #         "success": True,
        #         "result": single_result
        #     }
        # except Exception as e:
        #     logger.error(f"Single prompt approach failed: {e}")
        #     results["single_prompt"] = {
        #         "success": False,
        #         "error": str(e)
        #     }
        
        # Multi-agent approach
        if self.use_agents:
            try:
                agent_result = self.extract_data_multi_agent(conclusion_text)
                results["multi_agent"] = {
                    "success": True,
                    "result": agent_result
                }
            except Exception as e:
                logger.error(f"Multi-agent approach failed: {e}")
                results["multi_agent"] = {
                    "success": False,
                    "error": str(e)
                }
        else:
            results["multi_agent"] = {
                "success": False,
                "error": "Multi-agent approach not enabled"
            }
        
        return results

def get_enhanced_handler(use_agents: bool = True) -> EnhancedMainHandler:
    """Factory function to get an enhanced handler instance"""
    return EnhancedMainHandler(use_agents) 