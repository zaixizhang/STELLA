"""
Quality Assurance Tool: Response Evaluation

This tool provides standardized assessment of model outputs based on
predefined criteria such as clarity, accuracy, and relevance.

Author: AI Assistant
Category: quality_assurance
"""

from typing import Dict, Any, Tuple, Union
import re
import logging
from smolagents import tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def evaluate_response(response: str, criteria: Dict[str, Union[str, float, int]]) -> Dict[str, Any]:
    """
    Evaluate a model response based on predefined criteria.
    
    This tool provides a standardized way to assess model outputs across
    multiple dimensions including clarity, accuracy, relevance, and custom criteria.
    
    Args:
        response (str): The model response to evaluate
        criteria (Dict[str, Union[str, float, int]]): Dictionary containing evaluation criteria.
            Supported keys:
            - 'clarity': Expected clarity level (1-10) or 'high'/'medium'/'low'
            - 'accuracy': Expected accuracy level (1-10) or 'high'/'medium'/'low'
            - 'relevance': Expected relevance level (1-10) or 'high'/'medium'/'low'
            - 'min_length': Minimum expected length in characters
            - 'max_length': Maximum expected length in characters
            - 'keywords': List of keywords that should be present (comma-separated string)
            - 'forbidden_words': List of words that should not be present (comma-separated string)
            - 'format': Expected format ('json', 'markdown', 'plain', 'code')
    
    Returns:
        Dict[str, Any]: Quality assessment containing:
            - 'overall_score': Overall quality score (0-100)
            - 'individual_scores': Dictionary of individual criterion scores
            - 'report': Detailed assessment report
            - 'passed_criteria': List of criteria that passed
            - 'failed_criteria': List of criteria that failed
            - 'recommendations': List of improvement suggestions
    
    Raises:
        ValueError: If response is empty or criteria is invalid
        TypeError: If input types are incorrect
    """
    
    # Input validation
    if not isinstance(response, str):
        raise TypeError("Response must be a string")
    
    if not isinstance(criteria, dict):
        raise TypeError("Criteria must be a dictionary")
    
    if not response.strip():
        raise ValueError("Response cannot be empty")
    
    if not criteria:
        raise ValueError("Criteria dictionary cannot be empty")
    
    logger.info(f"Evaluating response of length {len(response)} with criteria: {list(criteria.keys())}")
    
    # Initialize assessment results
    individual_scores = {}
    passed_criteria = []
    failed_criteria = []
    recommendations = []
    
    try:
        # Evaluate clarity
        if 'clarity' in criteria:
            clarity_score = _evaluate_clarity(response, criteria['clarity'])
            individual_scores['clarity'] = clarity_score
            if clarity_score >= 60:
                passed_criteria.append('clarity')
            else:
                failed_criteria.append('clarity')
                recommendations.append("Improve sentence structure and word choice for better clarity")
        
        # Evaluate accuracy (based on structural indicators)
        if 'accuracy' in criteria:
            accuracy_score = _evaluate_accuracy(response, criteria['accuracy'])
            individual_scores['accuracy'] = accuracy_score
            if accuracy_score >= 60:
                passed_criteria.append('accuracy')
            else:
                failed_criteria.append('accuracy')
                recommendations.append("Verify factual statements and provide more specific information")
        
        # Evaluate relevance
        if 'relevance' in criteria:
            relevance_score = _evaluate_relevance(response, criteria['relevance'])
            individual_scores['relevance'] = relevance_score
            if relevance_score >= 60:
                passed_criteria.append('relevance')
            else:
                failed_criteria.append('relevance')
                recommendations.append("Ensure content directly addresses the main topic")
        
        # Check length requirements
        if 'min_length' in criteria:
            min_length = criteria['min_length']
            if len(response) >= min_length:
                individual_scores['min_length'] = 100
                passed_criteria.append('min_length')
            else:
                individual_scores['min_length'] = (len(response) / min_length) * 100
                failed_criteria.append('min_length')
                recommendations.append(f"Expand content to meet minimum length of {min_length} characters")
        
        if 'max_length' in criteria:
            max_length = criteria['max_length']
            if len(response) <= max_length:
                individual_scores['max_length'] = 100
                passed_criteria.append('max_length')
            else:
                individual_scores['max_length'] = max(0, 100 - ((len(response) - max_length) / max_length) * 100)
                failed_criteria.append('max_length')
                recommendations.append(f"Reduce content to stay within maximum length of {max_length} characters")
        
        # Check keywords presence
        if 'keywords' in criteria:
            keywords_score = _evaluate_keywords(response, criteria['keywords'])
            individual_scores['keywords'] = keywords_score
            if keywords_score >= 60:
                passed_criteria.append('keywords')
            else:
                failed_criteria.append('keywords')
                recommendations.append("Include more relevant keywords from the specified list")
        
        # Check forbidden words
        if 'forbidden_words' in criteria:
            forbidden_score = _evaluate_forbidden_words(response, criteria['forbidden_words'])
            individual_scores['forbidden_words'] = forbidden_score
            if forbidden_score >= 90:
                passed_criteria.append('forbidden_words')
            else:
                failed_criteria.append('forbidden_words')
                recommendations.append("Remove or replace inappropriate words")
        
        # Check format compliance
        if 'format' in criteria:
            format_score = _evaluate_format(response, criteria['format'])
            individual_scores['format'] = format_score
            if format_score >= 80:
                passed_criteria.append('format')
            else:
                failed_criteria.append('format')
                recommendations.append(f"Ensure response follows {criteria['format']} format")
        
        # Calculate overall score
        if individual_scores:
            overall_score = sum(individual_scores.values()) / len(individual_scores)
        else:
            overall_score = 0
            recommendations.append("No valid criteria provided for evaluation")
        
        # Generate report
        report = _generate_report(response, overall_score, individual_scores, passed_criteria, failed_criteria)
        
        result = {
            'overall_score': round(overall_score, 2),
            'individual_scores': {k: round(v, 2) for k, v in individual_scores.items()},
            'report': report,
            'passed_criteria': passed_criteria,
            'failed_criteria': failed_criteria,
            'recommendations': recommendations
        }
        
        logger.info(f"Evaluation completed. Overall score: {result['overall_score']}")
        return result
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


def _normalize_level(level: Union[str, int, float]) -> float:
    """Normalize different level formats to a 0-100 scale."""
    if isinstance(level, str):
        level_map = {'low': 30, 'medium': 60, 'high': 90}
        return level_map.get(level.lower(), 50)
    elif isinstance(level, (int, float)):
        if 1 <= level <= 10:
            return level * 10
        elif 0 <= level <= 100:
            return level
    return 50  # Default fallback


def _evaluate_clarity(response: str, expected_level: Union[str, int, float]) -> float:
    """Evaluate response clarity based on readability metrics."""
    # Simple clarity metrics
    sentences = len(re.split(r'[.!?]+', response.strip()))
    words = len(response.split())
    
    if sentences == 0 or words == 0:
        return 0
    
    avg_sentence_length = words / sentences
    
    # Score based on sentence length (optimal: 10-20 words per sentence)
    if 10 <= avg_sentence_length <= 20:
        length_score = 100
    elif avg_sentence_length < 10:
        length_score = avg_sentence_length * 8  # Penalize too short
    else:
        length_score = max(0, 100 - (avg_sentence_length - 20) * 3)  # Penalize too long
    
    # Check for complex words (>6 characters as a simple heuristic)
    complex_words = sum(1 for word in response.split() if len(word) > 6)
    complexity_ratio = complex_words / words if words > 0 else 0
    complexity_score = max(0, 100 - complexity_ratio * 150)
    
    # Combine scores
    clarity_score = (length_score + complexity_score) / 2
    
    # Compare with expected level
    expected_score = _normalize_level(expected_level)
    if clarity_score >= expected_score * 0.8:  # 80% of expected
        return clarity_score
    else:
        return clarity_score * 0.7  # Penalize for not meeting expectations
    
    return min(100, max(0, clarity_score))


def _evaluate_accuracy(response: str, expected_level: Union[str, int, float]) -> float:
    """Evaluate response accuracy based on structural indicators."""
    # Since we can't verify factual accuracy, we use structural indicators
    
    # Check for uncertainty markers (good for accuracy)
    uncertainty_markers = ['possibly', 'might', 'could be', 'appears to', 'seems', 'likely']
    uncertainty_count = sum(1 for marker in uncertainty_markers if marker in response.lower())
    
    # Check for definitive statements without qualifiers
    definitive_patterns = re.findall(r'\b(is|are|will|must)\b', response.lower())
    definitive_ratio = len(definitive_patterns) / len(response.split()) if response.split() else 0
    
    # Check for citations or references (indicators of accuracy)
    citation_patterns = re.findall(r'\[[\d,\s-]+\]|according to|source:|ref:', response.lower())
    citation_score = min(100, len(citation_patterns) * 25)
    
    # Base score calculation
    base_score = 70  # Start with average
    if uncertainty_count > 0:
        base_score += 10  # Reward appropriate uncertainty
    if definitive_ratio > 0.5:
        base_score -= 15  # Penalize too many definitive statements
    if citation_patterns:
        base_score += citation_score * 0.3
    
    # Compare with expected level
    expected_score = _normalize_level(expected_level)
    if base_score >= expected_score * 0.8:
        return min(100, base_score)
    else:
        return base_score * 0.8
    
    return min(100, max(0, base_score))


def _evaluate_relevance(response: str, expected_level: Union[str, int, float]) -> float:
    """Evaluate response relevance based on content focus."""
    # Check for off-topic indicators
    off_topic_phrases = ['by the way', 'speaking of', 'this reminds me', 'off topic']
    off_topic_count = sum(1 for phrase in off_topic_phrases if phrase in response.lower())
    
    # Check for topic consistency (simple heuristic using word frequency)
    words = response.lower().split()
    if len(words) < 10:
        return 60  # Too short to properly assess
    
    # Calculate word frequency to check for focus
    word_freq = {}
    for word in words:
        if len(word) > 3:  # Only consider meaningful words
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Check if there are dominant themes (high frequency words)
    if word_freq:
        max_freq = max(word_freq.values())
        focus_score = min(100, (max_freq / len(words)) * 500)  # Scale appropriately
    else:
        focus_score = 50
    
    # Penalize off-topic content
    relevance_score = focus_score - (off_topic_count * 10)
    
    # Compare with expected level
    expected_score = _normalize_level(expected_level)
    if relevance_score >= expected_score * 0.8:
        return min(100, relevance_score)
    else:
        return relevance_score * 0.8
    
    return min(100, max(0, relevance_score))


def _evaluate_keywords(response: str, keywords: str) -> float:
    """Evaluate keyword presence in the response."""
    if not keywords:
        return 100
    
    keyword_list = [kw.strip().lower() for kw in keywords.split(',')]
    present_keywords = sum(1 for kw in keyword_list if kw in response.lower())
    
    return (present_keywords / len(keyword_list)) * 100 if keyword_list else 100


def _evaluate_forbidden_words(response: str, forbidden_words: str) -> float:
    """Evaluate absence of forbidden words in the response."""
    if not forbidden_words:
        return 100
    
    forbidden_list = [fw.strip().lower() for fw in forbidden_words.split(',')]
    found_forbidden = sum(1 for fw in forbidden_list if fw in response.lower())
    
    return max(0, 100 - (found_forbidden / len(forbidden_list)) * 100) if forbidden_list else 100


def _evaluate_format(response: str, expected_format: str) -> float:
    """Evaluate format compliance of the response."""
    format_lower = expected_format.lower()
    
    if format_lower == 'json':
        # Check for JSON structure
        if response.strip().startswith('{') and response.strip().endswith('}'):
            return 100
        elif '{' in response and '}' in response:
            return 70
        else:
            return 20
    
    elif format_lower == 'markdown':
        # Check for markdown elements
        md_indicators = ['#', '*', '_', '`', '[', ']', '(', ')']
        md_count = sum(1 for indicator in md_indicators if indicator in response)
        return min(100, md_count * 10)
    
    elif format_lower == 'code':
        # Check for code-like structure
        code_indicators = ['def ', 'class ', 'import ', 'from ', '{', '}', '(', ')', ';']
        code_count = sum(1 for indicator in code_indicators if indicator in response)
        return min(100, code_count * 15)
    
    elif format_lower == 'plain':
        # Plain text should avoid special formatting
        special_chars = len(re.findall(r'[#*_`\[\]{}()]', response))
        return max(0, 100 - special_chars * 5)
    
    return 70  # Default score for unknown formats


def _generate_report(response: str, overall_score: float, individual_scores: Dict[str, float], 
                    passed_criteria: list, failed_criteria: list) -> str:
    """Generate a detailed assessment report."""
    report_lines = [
        "=== RESPONSE EVALUATION REPORT ===",
        f"Response Length: {len(response)} characters",
        f"Word Count: {len(response.split())} words",
        f"Overall Score: {overall_score:.2f}/100",
        "",
        "Individual Scores:",
    ]
    
    for criterion, score in individual_scores.items():
        status = "PASS" if criterion in passed_criteria else "FAIL"
        report_lines.append(f"  {criterion}: {score:.2f}/100 [{status}]")
    
    report_lines.extend([
        "",
        f"Passed Criteria ({len(passed_criteria)}): {', '.join(passed_criteria) if passed_criteria else 'None'}",
        f"Failed Criteria ({len(failed_criteria)}): {', '.join(failed_criteria) if failed_criteria else 'None'}",
        "",
        "Quality Assessment:",
    ])
    
    if overall_score >= 90:
        report_lines.append("  Excellent - Meets all quality standards")
    elif overall_score >= 75:
        report_lines.append("  Good - Meets most quality standards with minor issues")
    elif overall_score >= 60:
        report_lines.append("  Acceptable - Meets basic requirements but needs improvement")
    elif overall_score >= 40:
        report_lines.append("  Below Standard - Significant improvements needed")
    else:
        report_lines.append("  Poor - Major revisions required")
    
    return "\n".join(report_lines)


# Example usage and testing function
def test_evaluate_response():
    """Test function to verify the tool functionality."""
    print("Testing evaluate_response tool...")
    
    # Test case 1: Basic evaluation
    test_response = "This is a clear and well-structured response that addresses the main topic effectively. It contains relevant information and maintains good readability throughout."
    
    test_criteria = {
        'clarity': 'high',
        'accuracy': 8,
        'relevance': 'high',
        'min_length': 50,
        'keywords': 'clear, structured, relevant'
    }
    
    try:
        result = evaluate_response(test_response, test_criteria)
        print(f"✓ Test 1 passed - Overall score: {result['overall_score']}")
        print(f"  Report preview: {result['report'][:100]}...")
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
    
    # Test case 2: Error handling
    try:
        evaluate_response("", {})
        print("✗ Test 2 failed - Should have raised ValueError")
    except ValueError:
        print("✓ Test 2 passed - Correctly handled empty inputs")
    except Exception as e:
        print(f"✗ Test 2 failed with unexpected error: {e}")
    
    print("Testing completed.")


if __name__ == "__main__":
    test_evaluate_response()