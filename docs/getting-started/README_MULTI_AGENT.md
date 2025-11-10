# Multi-Agent Data Extraction System

This document describes the enhanced multi-agent system for extracting structured data from scientific paper conclusions.

## Overview

The multi-agent system replaces the original single monolithic prompt with specialized agents, each focusing on a specific extraction task. This approach improves quality, reliability, and maintainability of the extraction process.

## Architecture

### Agent Specialization

The system consists of 8 specialized agents:

1. **Geographic Extraction Agent** - Identifies geographical scope
2. **ITEM Extraction Agent** - Extracts practices, policies, features, and devices
3. **FACTOR Extraction Agent** - Identifies outcomes and characteristics affected by ITEMs
4. **Correlation Analysis Agent** - Determines relationships between ITEMs and FACTORs
5. **Population Analysis Agent** - Identifies affected socio-demographic groups
6. **Transportation Mode Agent** - Identifies related transportation modes
7. **Actor Identification Agent** - Identifies institutions or persons effecting ITEMs
8. **Data Coordination Agent** - Validates and formats final output

### System Components

- `agent_prompts.py` - Contains specialized prompts for each agent
- `agent_orchestrator.py` - Coordinates the multi-agent workflow
- `enhanced_main_handler.py` - Provides both single prompt and multi-agent approaches
- `example_usage.py` - Demonstrates usage and comparison

## Usage

### Basic Usage

```python
from .handlers.enhanced_main_handler import get_enhanced_handler

# Initialize with multi-agent support
handler = get_enhanced_handler(use_agents=True)

# Extract data using multi-agent approach
conclusion_text = "Your conclusion text here..."
result = handler.extract_data(conclusion_text, method="agents")
```

### Comparison Mode

```python
# Compare single prompt vs multi-agent approaches
comparison = handler.compare_approaches(conclusion_text)
print(comparison)
```

### Step-by-Step Process

```python
from .handlers.agent_orchestrator import get_agent_orchestrator

orchestrator = get_agent_orchestrator()

# Step 1: Extract geographical scope
geographic = orchestrator.extract_geographic_scope(conclusion_text)

# Step 2: Extract ITEMs
items = orchestrator.extract_items(conclusion_text)

# Step 3: For each item, extract factors
for item in items:
    factors = orchestrator.extract_factors(conclusion_text, item)
    
    # Step 4: Determine correlations
    for factor in factors:
        correlation = orchestrator.determine_correlation(conclusion_text, item, factor)
```

## Output Format

The system maintains the same JSON output format as the original prompt:

```json
{
    "GEOGRAPHIC": "geographical_scope",
    "item_name": {
        "ACTOR": "actor",
        "MODE": "transportation_mode",
        "POPULATION": "affected_population",
        "FACTOR": {
            "factor_name": {
                "CORRELATION": "correlation_type"
            }
        }
    }
}
```

## Benefits

### Quality Improvements

1. **Specialized Focus**: Each agent focuses on a specific extraction task
2. **Reduced Complexity**: Simpler prompts reduce cognitive load on the AI model
3. **Better Error Handling**: Individual agent failures don't affect the entire process
4. **Validation**: Coordinator agent validates final output

### Maintainability

1. **Modular Design**: Easy to modify individual agents without affecting others
2. **Clear Separation**: Each agent has a well-defined responsibility
3. **Debugging**: Easier to identify and fix issues in specific extraction tasks

### Flexibility

1. **Backward Compatibility**: Original single prompt approach still available
2. **Comparison Mode**: Can compare results between approaches
3. **Configurable**: Can enable/disable multi-agent support

## Running Examples

### Test with Sample Data

```bash
cd rag_system/pipeline_scripts
uv run python -m  agentic_data_policies_extraction.example_usage
```

### Test with PDF Files

```bash
uv run python -m  agentic_data_policies_extraction.main
```

### Production Mode

```python
from main import extract_with_agents

result = extract_with_agents(conclusion_text)
```

## Configuration

### Agent Settings

Each agent can be configured independently by modifying the prompts in `agent_prompts.py`.

### Model Configuration

The system uses the same model configuration as the original system (see `main_handler.py`).

## Error Handling

The system includes comprehensive error handling:

- Individual agent failures are logged and handled gracefully
- JSON parsing errors are caught and logged
- Fallback mechanisms ensure the system continues operating
- Detailed logging for debugging

## Performance Considerations

### Advantages

- **Parallel Processing**: Agents can potentially run in parallel (future enhancement)
- **Caching**: Individual agent results can be cached
- **Incremental Processing**: Can resume from failed steps

### Current Limitations

- **Sequential Processing**: Agents currently run sequentially
- **API Calls**: Each agent makes a separate API call
- **Latency**: Total processing time may be longer than single prompt

## Future Enhancements

1. **Parallel Processing**: Run independent agents concurrently
2. **Agent Caching**: Cache agent results for repeated extractions
3. **Dynamic Agent Selection**: Choose agents based on content type
4. **Agent Learning**: Improve agents based on validation feedback
5. **Custom Agents**: Allow users to define custom extraction agents

## Troubleshooting

### Common Issues

1. **JSON Parsing Errors**: Check agent responses for malformed JSON
2. **Empty Results**: Verify conclusion text contains extractable information
3. **API Errors**: Check model configuration and API limits

### Debug Mode

Enable detailed logging to debug agent interactions:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Migration from Single Prompt

The system is designed to be a drop-in replacement for the original single prompt approach:

```python
# Old way
from .prompts.text_analyzer import get_prompt_extraction
from .handlers.main_handler import get_client, get_response

client = get_client()
prompt = get_prompt_extraction(conclusion_text)
response = get_response(client, prompt)

# New way
from .handlers.enhanced_main_handler import get_enhanced_handler

handler = get_enhanced_handler(use_agents=True)
result = handler.extract_data(conclusion_text, method="agents")
```

The output format remains identical, ensuring seamless integration with existing systems. 