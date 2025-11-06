import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pymupdf4llm
from AI.handlers.enhanced_main_handler import get_enhanced_handler
from AI.handlers.main_handler import get_client, get_response
from AI.prompts.text_analyzer import get_prompt_selection_conclusion
# Import clients from separate files
from clients.database_client import DatabaseClient
from clients.qdrant_client import QdrantClient
from utils import get_policy_text, get_pymupdf4llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_with_agents(conclusion_text: str) -> Dict[str, Any]:
    """Extract data using the multi-agent system"""
    try:
        enhanced_handler = get_enhanced_handler(use_agents=True)
        return enhanced_handler.extract_data(conclusion_text, method="agents")
    except Exception as e:
        logger.error(f"Error in agent extraction: {e}")
        raise


def store_extraction_results(
    db_client: DatabaseClient,
    openalex_id: str,
    extraction_data: Dict[str, Any],
    conclusion: str,
) -> bool:
    """
    Store extraction results in the database using execute_query

    Args:
        db_client: Database client instance
        openalex_id: OpenAlex ID of the policy
        extraction_data: Extracted data from AI system

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Execute the insert query
        db_client.save_extraction_results(
            openalex_id=openalex_id, extraction_data=extraction_data, conclusion=conclusion
        )

        logger.info(f"Successfully stored extraction data for {openalex_id}")
        return True

    except Exception as e:
        logger.error(f"Error storing extraction results: {e}")
        return False


def production_mode():
    """
    Production mode
    """
    # Initialize database client
    db_client = DatabaseClient()
    # Initialize Qdrant client
    qdrant_client = QdrantClient()

    try:
        # Individual fetching
        logger.info("=== Fetching policies individually ===")
        policies = db_client.query_policies_abstracts_all(limit=3)

        # Print results and fetch corresponding texts from Qdrant
        logger.info(f"Retrieved {len(policies)} policies:")
        for policy in policies:
            qdrant_data = get_policy_text(qdrant_client, policy)

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


def test_multi_agent_extraction():
    """Test the new multi-agent extraction system"""
    try:
        logger.info("=== Testing Multi-Agent Extraction System ===")
        files = os.listdir("test_files")

        # Initialize enhanced handler with multi-agent support
        enhanced_handler = get_enhanced_handler(use_agents=True)

        for file in files:
            logger.info(f"Processing file: {file}")

            # Extract text from PDF
            # content_md = get_pymupdf4llm(os.path.join("test_files", file))
            # full_text = "\n".join([c["text"] for c in content_md])

            # Extract conclusion using original method
            client = get_client()
            # prompt_conclusion = get_prompt_selection_conclusion(full_text)
            # conclusion_response = get_response(client, prompt_conclusion)

            logger.info(f"Extracted conclusion from {file}")
            conclusion_response = """The relationship between property market and efficiency can be examined in the context of land supply. Limited
            land supply, in particular for residential use, has been an acute problem when a land area in the city has been
            identified as having potential for development but was undeveloped, leading the land area to be considered as
            under-utilised and vacant. These vacant, under-utilised and undeveloped lands are subject to various land supply
            constraints. Past research has identified several indicators that can enhance land efficiency, namely a safe,
            attractive, and healthy environment that can at the same time offer educational opportunities, employment,
            entertainment and social attraction. Additionally, other indicators include physical ownership marketability,
            infrastructure, land use, financial deficit and pollution. All indicators were found to influence urban land
            efficiency in the context of providing housing to the city residents. Therefore, the institutional economics
            analysis is usually carried out in a descriptive way. The broad nature of formal and informal institutions, rules
            and constraints leads to a descriptive identification of formal and informal institutions, agency relations and
            agents’ decisions which constrain the supply of land for development. In particular, the analytical tool of
            institutional economics analysis framework was utilised to investigate the existence, importance and implications
            of land supply constraints on the land development process."""

            # Test both approaches
            comparison_results = enhanced_handler.compare_approaches(conclusion_response)

            # Save results
            output_file = f"extraction_results_{file.replace('.pdf', '.json')}"
            with open(output_file, "w") as f:
                json.dump(comparison_results, f, indent=2)

            logger.info(f"Results saved to {output_file}")

            # Print summary
            print(f"\n=== Results for {file} ===")
            for approach, result in comparison_results.items():
                status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
                print(f"{approach}: {status}")
                if not result["success"]:
                    print(f"  Error: {result.get('error', 'Unknown error')}")

            print("\n" + "=" * 50 + "\n")
            break

    except Exception as e:
        logger.error(f"Error in multi-agent extraction test: {e}")
        raise


def main():
    """Main function to demonstrate database operations"""

    try:
        # Initialize database client
        db_client = DatabaseClient()

        # Ensure the policy_extractions table exists
        db_client.create_policy_extractions_table()

        # Individual fetching
        logger.info("=== Fetching policies individually ===")
        files = os.listdir("test_files")

        logger.info(f"Retrieved {len(files)} files:")
        for file in files:
            content_md = get_pymupdf4llm(os.path.join("test_files", file))
            full_text = "\n".join([c["text"] for c in content_md])

            client = get_client()
            prompt_conclusion = get_prompt_selection_conclusion(full_text)
            conclusion = get_response(client, prompt_conclusion)

            # Get the prompt for the AI processing
            response = extract_with_agents(conclusion)

            # Store the extraction results in the database
            store_extraction_results(db_client, file, response, conclusion)

            logger.info(f"Extraction results for {file}:")
            logger.info(response)

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    # Uncomment the function you want to run:

    # Original main function
    # main()

    # New multi-agent test function
    test_multi_agent_extraction()
