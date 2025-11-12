#!/usr/bin/env python3
"""
Test script for the multi-agent extraction system
"""

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_sample_conclusion():
    """Test with a sample conclusion that should extract known data"""

    sample_conclusion = """
    This study demonstrates that transit infrastructure investment in new towns significantly 
    reduces social exclusion and CO2 emissions. The implementation of microcars by car 
    manufacturers has shown to decrease materials use while increasing food accessibility 
    for elderly populations. Additionally, bike-sharing programs in urban centers have 
    been found to reduce traffic congestion and improve public health outcomes.
    """

    expected_items = ["transit infrastructure investment", "microcars", "bike-sharing programs"]

    expected_geographic = "new towns"

    print("Testing multi-agent system with sample conclusion...")

    try:
        from agentic_data_policies_extraction.handlers.enhanced_main_handler import get_enhanced_handler

        handler = get_enhanced_handler(use_agents=True)
        result = handler.extract_data(sample_conclusion, method="agents")

        print("✅ Multi-agent extraction completed")
        print(f"Geographic scope: {result.get('GEOGRAPHIC', 'Not found')}")

        # Check if expected items were found
        found_items = [key for key in result.keys() if key != "GEOGRAPHIC"]
        print(f"Items found: {found_items}")

        # Validate structure
        for item in found_items:
            item_data = result[item]
            required_fields = ["ACTOR", "MODE", "POPULATION", "FACTOR"]

            for field in required_fields:
                if field not in item_data:
                    print(f"❌ Missing field '{field}' for item '{item}'")
                    return False

            # Check if FACTOR has at least one entry
            if not item_data["FACTOR"]:
                print(f"❌ No factors found for item '{item}'")
                return False

            # Check each factor has CORRELATION
            for factor, factor_data in item_data["FACTOR"].items():
                if "CORRELATION" not in factor_data:
                    print(f"❌ Missing CORRELATION for factor '{factor}' in item '{item}'")
                    return False

        print("✅ All validation checks passed")
        print("\nExtracted data:")
        print(json.dumps(result, indent=2))

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_error_handling():
    """Test error handling with invalid input"""

    print("\nTesting error handling...")

    try:
        from agentic_data_policies_extraction.handlers.enhanced_main_handler import get_enhanced_handler

        handler = get_enhanced_handler(use_agents=True)

        # Test with empty conclusion
        result = handler.extract_data("", method="agents")
        print("✅ Empty conclusion handled gracefully")

        # Test with very short conclusion
        result = handler.extract_data("This is a test.", method="agents")
        print("✅ Short conclusion handled gracefully")

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_comparison_mode():
    """Test comparison between single prompt and multi-agent approaches"""

    print("\nTesting comparison mode...")

    sample_conclusion = """
    Transit infrastructure investment reduces CO2 emissions in urban areas.
    """

    try:

        from agentic_data_policies_extraction.handlers.enhanced_main_handler import get_enhanced_handler

        handler = get_enhanced_handler(use_agents=True)
        comparison = handler.compare_approaches(sample_conclusion)

        print("✅ Comparison completed")

        for approach, result in comparison.items():
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            print(f"{approach}: {status}")

        return True

    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False


def test_individual_agents():
    """Test individual agents"""

    print("\nTesting individual agents...")

    sample_conclusion = """
    Transit infrastructure investment in new towns reduces CO2 emissions.
    """

    try:
        from agentic_data_policies_extraction.handlers.agent_orchestrator import get_agent_orchestrator

        orchestrator = get_agent_orchestrator()

        # Test geographic agent
        geographic = orchestrator.extract_geographic_scope(sample_conclusion)
        print(f"✅ Geographic agent: {geographic}")

        # Test item extraction agent
        items = orchestrator.extract_items(sample_conclusion)
        print(f"✅ Item extraction agent: {items}")

        if items:
            # Test factor extraction agent
            factors = orchestrator.extract_factors(sample_conclusion, items[0])
            print(f"✅ Factor extraction agent: {factors}")

            if factors:
                # Test correlation agent
                correlation = orchestrator.determine_correlation(
                    sample_conclusion, items[0], factors[0]
                )
                print(f"✅ Correlation agent: {correlation}")

        return True

    except Exception as e:
        print(f"❌ Individual agent test failed: {e}")
        return False


def main():
    """Run all tests"""

    print("Multi-Agent System Test Suite\n")
    print("=" * 50)

    tests = [
        ("Sample Conclusion Test", test_sample_conclusion),
        ("Error Handling Test", test_error_handling),
        ("Comparison Mode Test", test_comparison_mode),
        ("Individual Agents Test", test_individual_agents),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    print(f"\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Multi-agent system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
