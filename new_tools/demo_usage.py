#!/usr/bin/env python3
"""
Demo Usage of the PubMed Search Tool

This script demonstrates practical usage scenarios for the pubmed_search tool.
"""

from pubmed_search import pubmed_search
import json
import time

def demo_basic_search():
    """Demonstrate basic search functionality"""
    print("=== DEMO 1: Basic Search ===")
    print("Searching for 'CRISPR gene editing'...")
    
    try:
        results = pubmed_search("CRISPR gene editing", max_results=3)
        
        print(f"Found {len(results)} articles:")
        for i, article in enumerate(results, 1):
            print(f"\n{i}. {article['title']}")
            print(f"   Authors: {article['authors']}")
            print(f"   Journal: {article['journal']} ({article['pubdate']})")
            print(f"   PMID: {article['pmid']}")
            print(f"   Abstract preview: {article['abstract'][:150]}...")
            
    except Exception as e:
        print(f"Error: {e}")

def demo_exclusion_search():
    """Demonstrate search with keyword exclusion"""
    print("\n\n=== DEMO 2: Search with Exclusion ===")
    print("Searching for 'machine learning' excluding 'cancer' and 'tumor'...")
    
    try:
        results = pubmed_search(
            query="machine learning",
            exclude_keywords="cancer, tumor",
            max_results=3
        )
        
        print(f"Found {len(results)} articles (excluding cancer/tumor research):")
        for i, article in enumerate(results, 1):
            print(f"\n{i}. {article['title']}")
            print(f"   Journal: {article['journal']} ({article['pubdate']})")
            
    except Exception as e:
        print(f"Error: {e}")

def demo_research_workflow():
    """Demonstrate a research workflow use case"""
    print("\n\n=== DEMO 3: Research Workflow ===")
    print("Researching 'artificial intelligence in drug discovery'...")
    
    try:
        results = pubmed_search(
            query="artificial intelligence drug discovery",
            max_results=5
        )
        
        # Analyze results
        recent_papers = [r for r in results if int(r['pubdate']) >= 2020]
        
        print(f"Total papers found: {len(results)}")
        print(f"Recent papers (2020+): {len(recent_papers)}")
        
        print("\nRecent papers summary:")
        for paper in recent_papers:
            print(f"• {paper['title']} ({paper['pubdate']})")
            
        # Save results to JSON for further analysis
        with open('new_tools/research_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to 'research_results.json'")
            
    except Exception as e:
        print(f"Error: {e}")

def demo_comparative_search():
    """Demonstrate comparative research between topics"""
    print("\n\n=== DEMO 4: Comparative Research ===")
    
    topics = [
        ("deep learning", "Deep Learning"),
        ("reinforcement learning", "Reinforcement Learning"),
        ("transfer learning", "Transfer Learning")
    ]
    
    print("Comparing recent research activity:")
    
    for query, display_name in topics:
        try:
            results = pubmed_search(
                query=f"{query} 2023[PDAT]",  # Restrict to 2023
                max_results=20
            )
            print(f"{display_name}: {len(results)} papers in 2023")
            
            # Brief pause between searches to be respectful
            time.sleep(1)
            
        except Exception as e:
            print(f"{display_name}: Error - {e}")

def main():
    """Run all demonstrations"""
    print("PubMed Search Tool - Usage Demonstrations")
    print("=" * 50)
    
    try:
        demo_basic_search()
        demo_exclusion_search()
        demo_research_workflow()
        demo_comparative_search()
        
        print("\n\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("Check the generated files and output above for results.")
        
    except Exception as e:
        print(f"Demo failed: {e}")

if __name__ == "__main__":
    main()