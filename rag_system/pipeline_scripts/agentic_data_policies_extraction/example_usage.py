#!/usr/bin/env python3
"""
Example usage of the multi-agent extraction system
"""

import json
import logging

from AI.handlers.enhanced_main_handler import get_enhanced_handler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_conclusion():
    """Sample conclusion text for testing"""
    return """
    This study demonstrates that transit infrastructure investment in new towns significantly 
    reduces social exclusion and CO2 emissions. The implementation of microcars by car 
    manufacturers has shown to decrease materials use while increasing food accessibility 
    for elderly populations. Additionally, bike-sharing programs in urban centers have 
    been found to reduce traffic congestion and improve public health outcomes.
    """


def demonstrate_single_prompt():
    """Demonstrate the original single prompt approach"""
    print("=== Single Prompt Approach ===")

    enhanced_handler = get_enhanced_handler(use_agents=False)
    conclusion = example_conclusion()

    try:
        result = enhanced_handler.extract_data(conclusion, method="single")
        print("✅ Success!")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 50 + "\n")


def demonstrate_multi_agent():
    """Demonstrate the new multi-agent approach"""
    print("=== Multi-Agent Approach ===")

    enhanced_handler = get_enhanced_handler(use_agents=True)
    conclusion = example_conclusion()

    try:
        result = enhanced_handler.extract_data(conclusion, method="agents")
        print("✅ Success!")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 50 + "\n")


def demonstrate_comparison():
    """Demonstrate comparison between both approaches"""
    print("=== Comparison: Single Prompt vs Multi-Agent ===")

    enhanced_handler = get_enhanced_handler(use_agents=True)
    conclusion = example_conclusion()

    try:
        comparison = enhanced_handler.compare_approaches(conclusion)

        for approach, result in comparison.items():
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            print(f"{approach}: {status}")

            if result["success"]:
                print("Result preview:")
                result_preview = json.dumps(result["result"], indent=2)[:500] + "..."
                print(result_preview)
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")

            print()

    except Exception as e:
        print(f"❌ Comparison failed: {e}")

    print("=" * 50 + "\n")


def demonstrate_step_by_step():
    """Demonstrate the step-by-step multi-agent process"""
    print("=== Step-by-Step Multi-Agent Process ===")

    from AI.handlers.agent_orchestrator import get_agent_orchestrator

    orchestrator = get_agent_orchestrator()
    conclusion = example_conclusion()

    try:
        # Step 1: Extract geographical scope
        print("Step 1: Extracting geographical scope...")
        geographic = orchestrator.extract_geographic_scope(conclusion)
        print(f"Geographic scope: {geographic}")

        # Step 2: Extract ITEMs
        print("\nStep 2: Extracting ITEMs...")
        items = orchestrator.extract_items(conclusion)
        print(f"Items found: {items}")

        # Step 3: For each item, extract factors
        print("\nStep 3: Extracting factors for each item...")
        for item in items:
            factors = orchestrator.extract_factors(conclusion, item)
            print(f"Factors for '{item}': {factors}")

            # Step 4: For each factor, determine correlation
            for factor in factors:
                correlation = orchestrator.determine_correlation(conclusion, item, factor)
                print(f"  Correlation '{item}' → '{factor}': {correlation}")

        # Step 5: Final coordinated result
        print("\nStep 5: Final coordinated result...")
        final_result = orchestrator.extract_data_with_agents(conclusion)
        print(json.dumps(final_result, indent=2))

    except Exception as e:
        print(f"❌ Step-by-step demonstration failed: {e}")

    print("=" * 50 + "\n")


def main():
    """Run all demonstrations"""
    print("Multi-Agent Extraction System Demonstrations\n")

    # Demonstrate single prompt approach
    demonstrate_single_prompt()

    # Demonstrate multi-agent approach
    demonstrate_multi_agent()

    # Demonstrate comparison
    demonstrate_comparison()

    # Demonstrate step-by-step process
    demonstrate_step_by_step()

    print("All demonstrations completed!")


if __name__ == "__main__":
    main()
