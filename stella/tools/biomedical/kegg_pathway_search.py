"""
KEGG Pathway Search Tool

This tool searches the KEGG database for gene involvement in pathways,
with a focus on immune-related pathways like NK cell cytotoxicity, 
apoptosis, or MHC signaling.
"""

import requests
import time
from typing import Dict, List, Union, Optional
from smolagents import tool
import json
import re


@tool
def kegg_pathway_search(gene_name: str, organism: str = "hsa") -> Dict[str, Union[str, int, List[str]]]:
    """
    Search KEGG database for gene involvement in pathways, especially immune-related ones.
    
    Args:
        gene_name (str): Name or symbol of the gene to search for
        organism (str, optional): Organism code (default: "hsa" for human)
    
    Returns:
        Dict[str, Union[str, int, List[str]]]: Dictionary containing:
            - gene: The gene name searched
            - organism: The organism code used
            - pathway_count: Number of pathways the gene is involved in
            - pathway_list: List of pathway names and IDs
            - immune_pathways: List of immune-related pathways
            - status: Success or error status
            - error: Error message if any
    """
    
    # Input validation
    if not gene_name or not isinstance(gene_name, str):
        return {
            "gene": gene_name,
            "organism": organism,
            "pathway_count": 0,
            "pathway_list": [],
            "immune_pathways": [],
            "status": "error",
            "error": "Invalid gene name provided"
        }
    
    gene_name = gene_name.strip()
    
    # Initialize result dictionary
    result = {
        "gene": gene_name,
        "organism": organism,
        "pathway_count": 0,
        "pathway_list": [],
        "immune_pathways": [],
        "status": "success",
        "error": None
    }
    
    try:
        # Step 1: Find gene entries in KEGG
        gene_search_url = f"https://rest.kegg.jp/find/genes/{gene_name}"
        
        response = requests.get(gene_search_url, timeout=10)
        response.raise_for_status()
        
        gene_entries = response.text.strip()
        
        if not gene_entries:
            result["status"] = "error"
            result["error"] = f"No gene entries found for '{gene_name}'"
            return result
        
        # Parse gene entries and filter for specified organism
        gene_ids = []
        for line in gene_entries.split('\n'):
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 2:
                    gene_id = parts[0]
                    if gene_id.startswith(f"{organism}:"):
                        gene_ids.append(gene_id)
        
        if not gene_ids:
            result["status"] = "error"
            result["error"] = f"No gene entries found for organism '{organism}'"
            return result
        
        # Step 2: Get pathways for each gene ID
        all_pathways = set()
        immune_pathways = set()
        
        # Keywords for immune-related pathways
        immune_keywords = [
            'immune', 'immunity', 'cytotoxicity', 'nk cell', 'natural killer',
            'apoptosis', 'mhc', 'antigen', 'complement', 'interferon',
            'interleukin', 'cytokine', 'inflammation', 'toll-like',
            'b cell', 't cell', 'dendritic', 'macrophage', 'neutrophil',
            'adaptive immune', 'innate immune', 'immunoglobulin'
        ]
        
        for gene_id in gene_ids[:5]:  # Limit to first 5 matches to avoid overwhelming
            try:
                # Get gene information
                gene_info_url = f"https://rest.kegg.jp/get/{gene_id}"
                gene_response = requests.get(gene_info_url, timeout=10)
                gene_response.raise_for_status()
                
                gene_info = gene_response.text
                
                # Extract pathway information
                pathway_section = False
                for line in gene_info.split('\n'):
                    line = line.strip()
                    
                    if line.startswith('PATHWAY'):
                        pathway_section = True
                        # Extract pathway from the same line
                        pathway_part = line[7:].strip()  # Remove 'PATHWAY'
                        if pathway_part:
                            pathway_match = re.match(r'(\w+)\s+(.+)', pathway_part)
                            if pathway_match:
                                pathway_id, pathway_name = pathway_match.groups()
                                pathway_entry = f"{pathway_id}: {pathway_name}"
                                all_pathways.add(pathway_entry)
                                
                                # Check if it's immune-related
                                if any(keyword in pathway_name.lower() for keyword in immune_keywords):
                                    immune_pathways.add(pathway_entry)
                    
                    elif pathway_section and line and not line[0].isupper():
                        # Continuation of pathway section
                        pathway_match = re.match(r'(\w+)\s+(.+)', line)
                        if pathway_match:
                            pathway_id, pathway_name = pathway_match.groups()
                            pathway_entry = f"{pathway_id}: {pathway_name}"
                            all_pathways.add(pathway_entry)
                            
                            # Check if it's immune-related
                            if any(keyword in pathway_name.lower() for keyword in immune_keywords):
                                immune_pathways.add(pathway_entry)
                    
                    elif pathway_section and line and line[0].isupper():
                        # End of pathway section
                        break
                
                # Small delay to be respectful to the API
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                # Continue with other gene IDs if one fails
                continue
        
        # Update result
        result["pathway_count"] = len(all_pathways)
        result["pathway_list"] = sorted(list(all_pathways))
        result["immune_pathways"] = sorted(list(immune_pathways))
        
        if not all_pathways:
            result["status"] = "warning"
            result["error"] = "No pathway information found for the gene"
    
    except requests.exceptions.RequestException as e:
        result["status"] = "error"
        result["error"] = f"Network error: {str(e)}"
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Unexpected error: {str(e)}"
    
    return result


def test_kegg_pathway_search():
    """Test function for the KEGG pathway search tool."""
    print("Testing KEGG Pathway Search Tool...")
    
    # Test with a known immune-related gene
    test_genes = ["GZMB", "PERFORIN", "CD8A", "INVALID_GENE"]
    
    for gene in test_genes:
        print(f"\n--- Testing gene: {gene} ---")
        result = kegg_pathway_search(gene)
        
        print(f"Status: {result['status']}")
        print(f"Pathway count: {result['pathway_count']}")
        
        if result['error']:
            print(f"Error: {result['error']}")
        
        if result['immune_pathways']:
            print("Immune-related pathways found:")
            for pathway in result['immune_pathways'][:3]:  # Show first 3
                print(f"  - {pathway}")
        
        if result['pathway_list'] and len(result['pathway_list']) > len(result['immune_pathways']):
            print("Other pathways found:")
            other_pathways = [p for p in result['pathway_list'] if p not in result['immune_pathways']]
            for pathway in other_pathways[:2]:  # Show first 2
                print(f"  - {pathway}")


if __name__ == "__main__":
    test_kegg_pathway_search()