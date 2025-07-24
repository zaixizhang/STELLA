# Evaluate Response Tool

A standardized Python tool for assessing model outputs based on predefined criteria like clarity, accuracy, and relevance.

## Overview

The `evaluate_response` tool provides a comprehensive framework for quality assurance of AI-generated content. It evaluates responses across multiple dimensions and provides detailed scoring, reporting, and improvement recommendations.

## Features

- **Multi-dimensional Evaluation**: Assess clarity, accuracy, relevance
- **Length Constraints**: Check minimum and maximum length requirements
- **Keyword Analysis**: Verify presence of required keywords
- **Content Filtering**: Detect and flag forbidden words
- **Format Compliance**: Validate JSON, Markdown, Plain text, and Code formats
- **Comprehensive Reporting**: Detailed scores and improvement suggestions
- **Error Handling**: Robust input validation and error management
- **Type Safety**: Full type hints for all parameters and returns

## Installation & Usage

### Requirements
- Python 3.7+
- smolagents library

### Basic Usage

```python
from evaluate_response import evaluate_response

# Example evaluation
response = "Your model response here..."
criteria = {
    'clarity': 'high',
    'accuracy': 8,
    'relevance': 'high',
    'min_length': 100,
    'keywords': 'important, keywords, here'
}

result = evaluate_response(response, criteria)
print(f"Overall Score: {result['overall_score']}/100")
print(result['report'])
```

## Supported Criteria

| Criterion | Type | Description | Example Values |
|-----------|------|-------------|----------------|
| `clarity` | str/int/float | Text readability level | 'high', 'medium', 'low', 1-10, 0-100 |
| `accuracy` | str/int/float | Content accuracy level | 'high', 'medium', 'low', 1-10, 0-100 |
| `relevance` | str/int/float | Topic relevance level | 'high', 'medium', 'low', 1-10, 0-100 |
| `min_length` | int | Minimum character count | 50, 100, 500 |
| `max_length` | int | Maximum character count | 1000, 2000, 5000 |
| `keywords` | str | Required keywords (comma-separated) | 'AI, machine learning, data' |
| `forbidden_words` | str | Prohibited words (comma-separated) | 'inappropriate, banned, words' |
| `format` | str | Expected format type | 'json', 'markdown', 'plain', 'code' |

## Output Structure

The tool returns a dictionary containing:

```python
{
    'overall_score': 75.5,  # Float: 0-100
    'individual_scores': {  # Dict: Individual criterion scores
        'clarity': 80.0,
        'accuracy': 70.0,
        'relevance': 77.0
    },
    'report': "=== DETAILED REPORT ===\n...",  # Str: Full assessment report
    'passed_criteria': ['clarity', 'relevance'],  # List: Successful criteria
    'failed_criteria': ['accuracy'],  # List: Failed criteria
    'recommendations': [  # List: Improvement suggestions
        "Verify factual statements and provide more specific information"
    ]
}
```

## Quality Score Interpretation

- **90-100**: Excellent - Meets all quality standards
- **75-89**: Good - Meets most standards with minor issues
- **60-74**: Acceptable - Meets basic requirements but needs improvement
- **40-59**: Below Standard - Significant improvements needed
- **0-39**: Poor - Major revisions required

## Examples

### Example 1: Basic Quality Assessment
```python
criteria = {
    'clarity': 'high',
    'min_length': 100,
    'keywords': 'python, programming'
}
```

### Example 2: JSON Format Validation
```python
criteria = {
    'format': 'json',
    'min_length': 20,
    'forbidden_words': 'error, invalid'
}
```

### Example 3: Academic Content Review
```python
criteria = {
    'clarity': 9,
    'accuracy': 'high',
    'relevance': 'high',
    'min_length': 500,
    'max_length': 2000,
    'keywords': 'research, methodology, results'
}
```

## Testing

Run the comprehensive test suite:

```bash
python test_evaluate_response_comprehensive.py
```

This will demonstrate all tool capabilities across various scenarios.

## Error Handling

The tool includes robust error handling for:
- Empty or invalid responses
- Missing or malformed criteria
- Type validation errors
- Unexpected evaluation errors

## Technical Details

- **Category**: quality_assurance
- **Framework**: smolagents
- **Type Safety**: Complete type annotations
- **Logging**: Comprehensive operation logging
- **Performance**: Optimized for real-time evaluation

## License

This tool is designed for integration with AI agent systems and quality assurance workflows.