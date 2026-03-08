"""
UniProt Query Tool

A tool for querying the UniProt database for protein/gene annotations,
focusing on functional descriptions, immune-related roles, and keywords
like 'immune evasion' or 'NK cell'.
"""

import time
from typing import Dict, List, Optional, Union, Any
import requests
from urllib.parse import quote
from smolagents import tool


@tool
def uniprot_query(
    query: str,
    organism: Optional[str] = None,
    limit: int = 50,
    include_isoforms: bool = False,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Query UniProt database for protein/gene annotations with focus on immune-related functions.
    
    Args:
        query (str): Search query (gene name, protein name, UniProt ID, or keywords like 'immune evasion', 'NK cell')
        organism (Optional[str]): Organism name or taxonomy ID to filter results (e.g., 'human', 'mouse', '9606')
        limit (int): Maximum number of results to return (default: 50, max: 500)
        include_isoforms (bool): Whether to include protein isoforms in results (default: False)
        timeout (int): Request timeout in seconds (default: 30)
    
    Returns:
        Dict[str, Any]: Structured dictionary containing:
            - 'success': bool indicating if query was successful
            - 'query_info': dict with query parameters used
            - 'results': list of protein entries with gene, functions, keywords
            - 'total_found': int total number of matches
            - 'error': str error message if query failed
    
    Example:
        >>> result = uniprot_query("KLRC1", organism="human")
        >>> result = uniprot_query("immune evasion", limit=20)
        >>> result = uniprot_query("NK cell receptor")
    """
    
    # Input validation
    if not query or not isinstance(query, str):
        return {
            'success': False,
            'error': 'Query must be a non-empty string',
            'query_info': {},
            'results': [],
            'total_found': 0
        }
    
    if limit < 1 or limit > 500:
        limit = min(max(limit, 1), 500)
    
    # Build query parameters
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    
    # Construct search query
    search_query = query.strip()
    
    # Add organism filter if specified
    if organism:
        # Handle common organism names
        organism_map = {
            'human': 'taxonomy_id:9606',
            'mouse': 'taxonomy_id:10090',
            'rat': 'taxonomy_id:10116',
            'zebrafish': 'taxonomy_id:7955',
            'fly': 'taxonomy_id:7227',
            'worm': 'taxonomy_id:6239',
            'yeast': 'taxonomy_id:559292'
        }
        
        if organism.lower() in organism_map:
            organism_filter = organism_map[organism.lower()]
        elif organism.isdigit():
            organism_filter = f"taxonomy_id:{organism}"
        else:
            organism_filter = f"organism_name:{organism}"
        
        search_query = f"({search_query}) AND {organism_filter}"
    
    # Add isoform filter
    if not include_isoforms:
        search_query = f"({search_query}) NOT is_isoform:true"
    
    # Define columns to retrieve
    columns = [
        'accession',
        'id',
        'gene_names',
        'gene_primary',
        'protein_name',
        'organism_name',
        'organism_id',
        'function_cc',
        'keyword',
        'go_f',  # GO molecular function
        'go_p',  # GO biological process
        'go_c',  # GO cellular component
        'cc_pathway',
        'cc_disease',
        'cc_interaction',
        'ft_domain',
        'length',
        'reviewed'
    ]
    
    params = {
        'query': search_query,
        'format': 'tsv',
        'fields': ','.join(columns),
        'size': str(limit)
    }
    
    query_info = {
        'original_query': query,
        'processed_query': search_query,
        'organism': organism,
        'limit': limit,
        'include_isoforms': include_isoforms
    }
    
    try:
        # Make API request with rate limiting consideration
        headers = {
            'User-Agent': 'UniProtQueryTool/1.0 (https://github.com/your-org/tools)',
            'Accept': 'text/tab-separated-values'
        }
        
        response = requests.get(
            base_url,
            params=params,
            headers=headers,
            timeout=timeout
        )
        
        # Handle rate limiting
        if response.status_code == 429:
            time.sleep(2)  # Wait 2 seconds before retry
            response = requests.get(
                base_url,
                params=params,
                headers=headers,
                timeout=timeout
            )
        
        response.raise_for_status()
        
        # Parse TSV response
        lines = response.text.strip().split('\n')
        if len(lines) < 2:
            return {
                'success': True,
                'query_info': query_info,
                'results': [],
                'total_found': 0,
                'error': None
            }
        
        headers_line = lines[0].split('\t')
        data_lines = lines[1:]
        
        results = []
        for line in data_lines:
            if not line.strip():
                continue
                
            values = line.split('\t')
            if len(values) != len(headers_line):
                continue  # Skip malformed lines
            
            entry = dict(zip(headers_line, values))
            
            # Process and structure the data
            processed_entry = {
                'accession': entry.get('Entry', '').strip(),
                'entry_name': entry.get('Entry Name', '').strip(),
                'gene_names': _parse_gene_names(entry.get('Gene Names', '')),
                'primary_gene': entry.get('Gene Names (primary)', '').strip(),
                'protein_name': entry.get('Protein names', '').strip(),
                'organism': {
                    'name': entry.get('Organism', '').strip(),
                    'id': entry.get('Organism (ID)', '').strip()
                },
                'function': _clean_text(entry.get('Function [CC]', '')),
                'keywords': _parse_keywords(entry.get('Keywords', '')),
                'go_annotations': {
                    'molecular_function': _parse_go_terms(entry.get('Gene Ontology (molecular function)', '')),
                    'biological_process': _parse_go_terms(entry.get('Gene Ontology (biological process)', '')),
                    'cellular_component': _parse_go_terms(entry.get('Gene Ontology (cellular component)', ''))
                },
                'pathways': _clean_text(entry.get('Pathway', '')),
                'disease_involvement': _clean_text(entry.get('Involvement in disease', '')),
                'interactions': _clean_text(entry.get('Interacts with', '')),
                'domains': _parse_domains(entry.get('Domain [FT]', '')),
                'length': entry.get('Length', '').strip(),
                'reviewed': entry.get('Reviewed', '').strip() == 'reviewed',
                'immune_related': _check_immune_relevance(entry)
            }
            
            results.append(processed_entry)
        
        return {
            'success': True,
            'query_info': query_info,
            'results': results,
            'total_found': len(results),
            'error': None
        }
        
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': f'Request timeout after {timeout} seconds',
            'query_info': query_info,
            'results': [],
            'total_found': 0
        }
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f'Network error: {str(e)}',
            'query_info': query_info,
            'results': [],
            'total_found': 0
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Unexpected error: {str(e)}',
            'query_info': query_info,
            'results': [],
            'total_found': 0
        }


def _parse_gene_names(gene_names_str: str) -> List[str]:
    """Parse gene names from UniProt format."""
    if not gene_names_str or gene_names_str.strip() == '':
        return []
    
    # Remove common prefixes and split by semicolon
    cleaned = gene_names_str.replace('Name=', '').replace('Synonyms=', '')
    names = [name.strip() for name in cleaned.split(';') if name.strip()]
    return names


def _parse_keywords(keywords_str: str) -> List[str]:
    """Parse keywords from semicolon-separated string."""
    if not keywords_str or keywords_str.strip() == '':
        return []
    
    keywords = [kw.strip() for kw in keywords_str.split(';') if kw.strip()]
    return keywords


def _parse_go_terms(go_str: str) -> List[Dict[str, str]]:
    """Parse GO terms from UniProt format."""
    if not go_str or go_str.strip() == '':
        return []
    
    terms = []
    # Split by semicolon and parse each term
    for term in go_str.split(';'):
        term = term.strip()
        if term:
            # Extract GO ID and term name if present
            if '[' in term and ']' in term:
                name = term.split('[')[0].strip()
                go_id = term.split('[')[1].split(']')[0].strip()
                terms.append({'id': go_id, 'name': name})
            else:
                terms.append({'id': '', 'name': term})
    
    return terms


def _parse_domains(domains_str: str) -> List[str]:
    """Parse protein domains from string."""
    if not domains_str or domains_str.strip() == '':
        return []
    
    domains = [domain.strip() for domain in domains_str.split(';') if domain.strip()]
    return domains


def _clean_text(text: str) -> str:
    """Clean and format text fields."""
    if not text or text.strip() == '':
        return ''
    
    # Remove extra whitespace and common formatting
    cleaned = ' '.join(text.split())
    return cleaned


def _check_immune_relevance(entry: Dict[str, str]) -> bool:
    """Check if protein entry is immune-related based on various fields."""
    immune_keywords = [
        'immune', 'immunity', 'immunoglobulin', 'antibody', 'antigen',
        'nk cell', 'natural killer', 'cytotoxic', 'cd', 'hla', 'mhc',
        'complement', 'interferon', 'interleukin', 'chemokine', 'cytokine',
        'toll-like', 'tlr', 'killer cell', 't cell', 'b cell',
        'lymphocyte', 'leukocyte', 'macrophage', 'dendritic cell',
        'inflammation', 'inflammatory', 'defense', 'pathogen',
        'viral evasion', 'immune evasion', 'immunosuppression',
        'autoimmune', 'allergy', 'hypersensitivity'
    ]
    
    # Check multiple fields for immune-related content
    fields_to_check = [
        entry.get('Protein names', ''),
        entry.get('Function [CC]', ''),
        entry.get('Keywords', ''),
        entry.get('Gene Ontology (biological process)', ''),
        entry.get('Gene Ontology (molecular function)', ''),
        entry.get('Pathway', '')
    ]
    
    combined_text = ' '.join(fields_to_check).lower()
    
    return any(keyword in combined_text for keyword in immune_keywords)


# Test function for development
def _test_uniprot_query():
    """Test function to verify tool functionality."""
    print("Testing UniProt Query Tool...")
    
    # Test 1: Gene name query
    print("\n1. Testing gene name query (KLRC1)...")
    result1 = uniprot_query("KLRC1", organism="human", limit=5)
    print(f"Success: {result1['success']}")
    print(f"Results found: {result1['total_found']}")
    if result1['results']:
        print(f"First result: {result1['results'][0]['protein_name']}")
    
    # Test 2: Keyword query
    print("\n2. Testing keyword query (NK cell receptor)...")
    result2 = uniprot_query("NK cell receptor", organism="human", limit=3)
    print(f"Success: {result2['success']}")
    print(f"Results found: {result2['total_found']}")
    
    # Test 3: Error handling
    print("\n3. Testing error handling (empty query)...")
    result3 = uniprot_query("")
    print(f"Success: {result3['success']}")
    print(f"Error: {result3['error']}")
    
    print("\nTesting completed!")


if __name__ == "__main__":
    _test_uniprot_query()