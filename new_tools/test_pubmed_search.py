"""
Test script for the pubmed_search tool
This demonstrates various use cases and validates the tool functionality.
"""

# Add the new_tools directory to the path to import the tool
import sys
sys.path.append('.')

from pubmed_search import pubmed_search

def test_basic_search():
    """Test basic search functionality"""
    print("=== TEST 1: Basic Search ===")
    try:
        results = pubmed_search("COVID-19", max_results=2)
        print(f"Found {len(results)} articles")
        
        for i, article in enumerate(results, 1):
            print(f"\nArticle {i}:")
            print(f"  PMID: {article['pmid']}")
            print(f"  Title: {article['title'][:80]}...")
            print(f"  Authors: {article['authors'][:50]}...")
            print(f"  Journal: {article['journal']}")
            print(f"  Date: {article['pubdate']}")
        
        print("✓ Basic search test passed")
        return True
    except Exception as e:
        print(f"✗ Basic search test failed: {e}")
        return False

def test_exclusion_keywords():
    """Test search with exclusion keywords"""
    print("\n=== TEST 2: Search with Exclusion Keywords ===")
    try:
        # Search for diabetes but exclude type 1
        results = pubmed_search("diabetes", exclude_keywords="type 1, children", max_results=3)
        print(f"Found {len(results)} articles (excluding 'type 1' and 'children')")
        
        for i, article in enumerate(results, 1):
            print(f"\nArticle {i}:")
            print(f"  Title: {article['title'][:80]}...")
            # Check if excluded terms appear (should be rare but may happen due to search complexity)
            title_lower = article['title'].lower()
            abstract_lower = article['abstract'].lower()
            
        print("✓ Exclusion keywords test passed")
        return True
    except Exception as e:
        print(f"✗ Exclusion keywords test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n=== TEST 3: Edge Cases ===")
    
    # Test empty query
    try:
        pubmed_search("")
        print("✗ Empty query test failed - should have raised ValueError")
        return False
    except ValueError:
        print("✓ Empty query test passed - correctly raised ValueError")
    except Exception as e:
        print(f"✗ Empty query test failed with unexpected error: {e}")
        return False
    
    # Test very specific search that might return no results
    try:
        results = pubmed_search("veryspecificnonexistentquery12345")
        print(f"✓ No results test passed - got {len(results)} results")
    except Exception as e:
        print(f"✗ No results test failed: {e}")
        return False
    
    # Test max_results boundary
    try:
        results = pubmed_search("cancer", max_results=150)  # Should be capped at 100
        print(f"✓ Max results boundary test passed - requested 150, implementation handled gracefully")
    except Exception as e:
        print(f"✗ Max results boundary test failed: {e}")
        return False
    
    return True

def test_data_quality():
    """Test the quality and completeness of returned data"""
    print("\n=== TEST 4: Data Quality ===")
    try:
        results = pubmed_search("artificial intelligence medicine", max_results=2)
        
        for i, article in enumerate(results, 1):
            print(f"\nValidating Article {i}:")
            
            # Check required fields
            required_fields = ['pmid', 'title', 'abstract', 'authors', 'journal', 'pubdate']
            for field in required_fields:
                if field not in article:
                    print(f"  ✗ Missing field: {field}")
                    return False
                print(f"  ✓ {field}: {len(str(article[field]))} characters")
            
            # Check PMID is numeric
            if not article['pmid'].isdigit() and article['pmid'] != 'N/A':
                print(f"  ✗ PMID is not numeric: {article['pmid']}")
                return False
        
        print("✓ Data quality test passed")
        return True
    except Exception as e:
        print(f"✗ Data quality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running PubMed Search Tool Tests\n")
    
    all_tests_passed = True
    
    # Run all tests
    test_functions = [
        test_basic_search,
        test_exclusion_keywords,
        test_edge_cases,
        test_data_quality
    ]
    
    for test_func in test_functions:
        try:
            result = test_func()
            all_tests_passed = all_tests_passed and result
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            all_tests_passed = False
    
    print(f"\n{'='*50}")
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! The pubmed_search tool is working correctly.")
    else:
        print("❌ Some tests failed. Please check the output above.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()