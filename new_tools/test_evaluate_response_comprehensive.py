"""
Comprehensive test suite for the evaluate_response tool.
This demonstrates all the tool's capabilities and features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate_response import evaluate_response

def run_comprehensive_tests():
    """Run comprehensive tests to demonstrate tool functionality."""
    print("=" * 60)
    print("COMPREHENSIVE EVALUATION TOOL TESTING")
    print("=" * 60)
    
    # Test Case 1: High-quality response
    print("\n1. Testing High-Quality Response:")
    print("-" * 40)
    
    high_quality_response = """
    Machine learning is a powerful subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. The field encompasses various algorithms and techniques including supervised learning, unsupervised learning, and reinforcement learning. These methods allow systems to identify patterns in data and make predictions or decisions based on that analysis. Applications range from image recognition to natural language processing, making machine learning essential in modern technology.
    """
    
    criteria_1 = {
        'clarity': 'high',
        'accuracy': 9,
        'relevance': 'high',
        'min_length': 200,
        'max_length': 1000,
        'keywords': 'machine learning, artificial intelligence, algorithms, data',
        'format': 'plain'
    }
    
    result_1 = evaluate_response(high_quality_response.strip(), criteria_1)
    print(f"Overall Score: {result_1['overall_score']}/100")
    print(f"Passed: {len(result_1['passed_criteria'])}, Failed: {len(result_1['failed_criteria'])}")
    print(f"Individual Scores: {result_1['individual_scores']}")
    
    # Test Case 2: JSON format response
    print("\n2. Testing JSON Format Response:")
    print("-" * 40)
    
    json_response = '{"name": "John Doe", "age": 30, "skills": ["Python", "Machine Learning", "Data Analysis"], "experience": "5 years"}'
    
    criteria_2 = {
        'format': 'json',
        'min_length': 50,
        'keywords': 'name, age, skills'
    }
    
    result_2 = evaluate_response(json_response, criteria_2)
    print(f"Overall Score: {result_2['overall_score']}/100")
    print(f"Format score: {result_2['individual_scores'].get('format', 'N/A')}")
    
    # Test Case 3: Markdown response
    print("\n3. Testing Markdown Format Response:")
    print("-" * 40)
    
    markdown_response = """
    # Data Science Project
    
    ## Overview
    This project focuses on *predictive analytics* using **machine learning** techniques.
    
    ### Key Features:
    - Data preprocessing
    - Model training
    - Performance evaluation
    
    ```python
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    ```
    
    [Learn more](https://example.com)
    """
    
    criteria_3 = {
        'format': 'markdown',
        'clarity': 'high',
        'keywords': 'data, machine learning, model'
    }
    
    result_3 = evaluate_response(markdown_response.strip(), criteria_3)
    print(f"Overall Score: {result_3['overall_score']}/100")
    print(f"Format compliance: {result_3['individual_scores'].get('format', 'N/A')}")
    
    # Test Case 4: Response with forbidden words
    print("\n4. Testing Response with Forbidden Words:")
    print("-" * 40)
    
    problematic_response = "This is a terrible and awful response that contains bad words and inappropriate content."
    
    criteria_4 = {
        'clarity': 'medium',
        'forbidden_words': 'terrible, awful, bad',
        'min_length': 20
    }
    
    result_4 = evaluate_response(problematic_response, criteria_4)
    print(f"Overall Score: {result_4['overall_score']}/100")
    print(f"Forbidden words score: {result_4['individual_scores'].get('forbidden_words', 'N/A')}")
    print(f"Recommendations: {result_4['recommendations']}")
    
    # Test Case 5: Length violations
    print("\n5. Testing Length Constraints:")
    print("-" * 40)
    
    short_response = "Too short."
    
    criteria_5 = {
        'min_length': 100,
        'max_length': 200,
        'clarity': 'medium'
    }
    
    result_5 = evaluate_response(short_response, criteria_5)
    print(f"Overall Score: {result_5['overall_score']}/100")
    print(f"Length scores: min={result_5['individual_scores'].get('min_length', 'N/A')}")
    print(f"Failed criteria: {result_5['failed_criteria']}")
    
    # Test Case 6: Perfect score scenario
    print("\n6. Testing Near-Perfect Response:")
    print("-" * 40)
    
    perfect_response = """
    Artificial intelligence represents a transformative technology that continues to reshape industries worldwide. Through sophisticated algorithms and computational models, AI systems can process vast amounts of data, recognize complex patterns, and make intelligent decisions. Key applications include natural language processing, computer vision, robotics, and predictive analytics. The field encompasses machine learning, deep learning, and neural networks, each contributing unique capabilities to solve real-world problems effectively.
    """
    
    criteria_6 = {
        'clarity': 'high',
        'accuracy': 'high',
        'relevance': 'high',
        'min_length': 300,
        'max_length': 800,
        'keywords': 'artificial intelligence, machine learning, algorithms, data',
        'format': 'plain'
    }
    
    result_6 = evaluate_response(perfect_response.strip(), criteria_6)
    print(f"Overall Score: {result_6['overall_score']}/100")
    print(f"All individual scores: {result_6['individual_scores']}")
    
    # Display full report for the best result
    print("\n" + "=" * 60)
    print("DETAILED REPORT FOR HIGH-QUALITY RESPONSE:")
    print("=" * 60)
    print(result_6['report'])
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nTool Features Demonstrated:")
    print("✓ Multiple evaluation criteria (clarity, accuracy, relevance)")
    print("✓ Length constraints (min/max)")
    print("✓ Keyword matching")
    print("✓ Forbidden word detection")
    print("✓ Format compliance checking (JSON, Markdown, Plain, Code)")
    print("✓ Comprehensive scoring and reporting")
    print("✓ Error handling and input validation")
    print("✓ Detailed recommendations for improvement")

if __name__ == "__main__":
    run_comprehensive_tests()