import requests
from markdownify import markdownify
import re
from pathlib import Path
import subprocess
from requests.exceptions import RequestException
from smolagents import tool
import pandas as pd
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
import math
from rapidfuzz import fuzz, process
import numpy as np

# Add the parent directory to sys.path to import gene_tools from main directory
main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)
# Change working directory to main directory
os.chdir(main_dir)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from llm import json_llm_call


def validate_genes(genes: List[str], species: str = "human") -> Tuple[List[str], List[str]]:
    """
    Validate a list of gene symbols against local HGNC database.
    
    Args:
        genes: List of gene symbols to validate
        species: Species to validate against ("human" or "mouse")
            
    Returns:
        Tuple of (valid_genes, invalid_genes)
    """
    # Only support human genes for now since we're using HGNC data
    if species != "human":
        raise ValueError("Only human gene validation is supported with HGNC data")
    
    try:
        with open("resource/hgnc_name.txt", "r") as f:
            # Skip header line and get second column (index 1)
            next(f)  # Skip header
            valid_symbols = {line.strip().split("\t")[1] for line in f}
            
        valid_genes = [gene for gene in genes if gene in valid_symbols]
        invalid_genes = [gene for gene in genes if gene not in valid_symbols]
        
    except FileNotFoundError:
        print("HGNC gene list file not found at resource/hgnc_name.txt")
        raise
        
    return valid_genes, invalid_genes

def normalize_scores(scores: List[float], temperature: float = 1.0) -> List[float]:
    """
    Normalize scores using softmax transformation with temperature control.
    
    Args:
        scores: List of numerical scores to normalize
        temperature: Controls distribution sharpness (lower = sharper contrasts)
            Lower values (e.g., 0.5) create sharper contrasts between high/low scores
            Higher values (e.g., 2.0) make the distribution more uniform
        
    Returns:
        List of normalized scores that sum to 1.0, ensuring no zero values
    """
    if not scores:
        return []
        
    # If all scores are the same, return equal probabilities
    if len(set(scores)) == 1:
        return [1.0 / len(scores)] * len(scores)
    
    # Apply softmax with temperature: exp(x/T) / sum(exp(x/T))
    # First subtract max score for numerical stability
    max_score = max(scores)
    exp_scores = [math.exp((score - max_score) / temperature) for score in scores]
    sum_exp_scores = sum(exp_scores)
    
    return [exp_score / sum_exp_scores for exp_score in exp_scores]


@tool
def kegg_pathway_search(query_list: List[str], model_name: str = "gemini-2.5-pro") -> Dict[str, Any]:
    """
    Search and extract biological pathway information from the KEGG database. The KEGG (Kyoto Encyclopedia of Genes and Genomes) database is a comprehensive resource for understanding biological systems and their functions.
    
    Args:
        query_list: List of pathway names related to biological processes or diseases
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing gene lists associated with each pathway and normalized relevance scores
    """
    
    # Prompt templates
    PROMPT_MATCH = """
    You are an assistant to biologists. Given the user query, your task is to think step by step to identify the most relevant KEGG pathway ID from the provided pathway table. Please format your response following response format and make sure it is parsable by JSON.

    User query:
    {query}

    Pathway table:
    {existing}

    Response format:
    {{
    "thoughts": <thoughts>, # your step-by-step thinking process
    "pathway_id": <pathway_id>, # Most relevant KEGG pathway ID (e.g., "hsa04630"). If no pathway is relevant, return "NA".
    }}
    """

    PROMPT_EXTRACT = """
    You are an assistant to biologists. Your task is to think step by step to extract valid gene names in the pathway information. Please format your response following response format and make sure it is parsable by JSON.

    Pathway information:
    {existing}

    Response format:
    {{
    "thoughts": <thoughts>, # your step-by-step thinking process.
    "genes": <genes>, # list of valid gene names extracted from pathway information.
    }}
    """
    
    def extract_genes(pathway_info: str) -> Dict[str, Any]:
        """Extract gene names from pathway information"""
        try:
            prompt = PROMPT_EXTRACT.format(existing=pathway_info)
            return json_llm_call(prompt, model_name=model_name)
        except Exception as e:
            print(f"Error extracting genes: {str(e)}")
            return {"thoughts": "Error in extraction", "genes": []}
    
    try:
        # Step 1: Load KEGG pathway table
        pathway_df = pd.read_csv("resource/Kegg_pathways.csv")
        
        # Step 2: Process each query
        final_results = {}
        raw_results = {}
        
        for query in query_list:
            # Use LLM to identify relevant pathway IDs
            prompt = PROMPT_MATCH.format(
                query=query,
                existing=pathway_df.to_string()
            )
            pathway_selection = json_llm_call(prompt, model_name=model_name)
            relevant_pathway_id = pathway_selection.get('pathway_id', "NA")
            
            # Store both pathway IDs and their names
            pathway_details = []
            if relevant_pathway_id != "NA":
                pathway_name = pathway_df[pathway_df['ID'] == relevant_pathway_id]['Pathways'].iloc[0]
                pathway_details.append({
                    "pathway_id": relevant_pathway_id,
                    "pathway_name": pathway_name
                })
            raw_results[query] = pathway_details
            
            # Collect gene information for each pathway
            pathway_genes = []
            for pathway_info in pathway_details:
                pathway_id = pathway_info['pathway_id']
                try:
                    url = f"https://rest.kegg.jp/get/{pathway_id}"
                    response = requests.get(url)
                    response.raise_for_status()
                    
                    # Extract GENE section
                    gene_section = ""
                    in_gene_section = False
                    for line in response.text.split('\n'):
                        if line.startswith('GENE'):
                            in_gene_section = True
                            gene_section += line + '\n'
                        elif in_gene_section and line.startswith(' '):
                            gene_section += line + '\n'
                        elif in_gene_section and not line.startswith(' '):
                            break
                    
                    # Extract genes using existing extract_genes method
                    if gene_section:
                        gene_info = extract_genes(gene_section)
                        extracted_genes = gene_info.get('genes', [])
                        
                        # Validate genes using gene_tools
                        valid_genes, invalid_genes = validate_genes(extracted_genes)
                        pathway_genes.extend(valid_genes)  # Only add valid genes
                
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching pathway {pathway_id}: {str(e)}")
                    continue
            
            # Use set to remove duplicates
            unique_genes = list(set(pathway_genes))
            
            # Assign initial raw scores of 1.0 to each gene
            raw_scores = [1.0] * len(unique_genes)
            
            # Normalize scores using gene_tools function
            normalized_scores = normalize_scores(raw_scores, temperature=1.0)
            
            # Create final results with normalized scores
            final_results[query] = [
                {"gene": gene, "score": score} 
                for gene, score in zip(unique_genes, normalized_scores)
            ]

        return {
            "status": "success",
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def disease_gene_search(
    disease_list: List[str],
    confidence_cutoff: float = 2.0,
    top_n: Optional[int] = None,
    similarity_threshold: int = 80,
    file_path: str = "resource/diseases/human_disease_integrated_full.tsv"
) -> Dict[str, Any]:
    """
    Search for genes associated with specific diseases using the DISEASES database. DISEASES is a comprehensive resource that integrates evidence on disease-gene associations from automatic text mining, manually curated literature, cancer mutation data, and genome-wide association studies.
    
    Args:
        disease_list: List of disease names to search for associated genes
        confidence_cutoff: Minimum confidence score threshold (default: 2.0)
        top_n: Maximum number of top genes to return by confidence score (None means return all)
        similarity_threshold: Minimum similarity score (0-100) for fuzzy matching (default: 80)
        file_path: Path to the disease mapping TSV file
    
    Returns:
        Dictionary containing gene lists associated with each disease and normalized confidence scores
    """
    
    def search_disease_genes(disease_list: List[str]) -> Dict[str, Dict[str, Any]]:
        """Search for genes associated with given diseases in the mapping file"""
        results = {}
        
        try:
            # Validate file existence
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Disease mapping file not found at: {file_path}")
            
            # Read the TSV file
            df = pd.read_csv(file_path, 
                           sep='\t', 
                           header=None,
                           names=['col1', 'gene_name', 'col3', 'disease', 'confidence'])
            
            # Process each disease query
            for query in disease_list:
                normalized_query = query.lower().replace('cancer', 'carcinoma')
                
                # Calculate similarity scores
                df['similarity'] = df['disease'].apply(
                    lambda x: max(
                        fuzz.token_sort_ratio(str(x).lower(), normalized_query),
                        fuzz.token_sort_ratio(str(x).lower().replace('carcinoma', 'cancer'), query.lower())
                    ) if pd.notnull(x) else 0
                )
                
                # Get the best matching disease
                best_match = df.loc[df['similarity'].idxmax()]
                best_score = best_match['similarity']
                
                # Initialize result structure with the match info, even if below threshold
                results[query] = {
                    "genes": [],
                    "scores": [],
                    "count": 0,
                    "matched_disease": best_match['disease'],  # Always include the best match
                    "similarity_score": best_score,  # Always include the score
                    "note": ""
                }
                
                # Only proceed with gene collection if above threshold
                if best_score >= similarity_threshold:
                    # Filter for the exact matched disease name
                    disease_df = df[df['disease'] == best_match['disease']]
                    # First filter by confidence score
                    score_mask = disease_df['confidence'] >= confidence_cutoff
                    filtered_df = disease_df[score_mask]
                    if len(filtered_df) > 0:
                        # Sort by confidence score
                        top_genes_df = (filtered_df.sort_values('confidence', ascending=False)
                                      .drop_duplicates('gene_name'))
                        
                        total_available = len(top_genes_df)
                        
                        # Apply top_n filter only if specified
                        if top_n is not None and total_available > top_n:
                            top_genes = top_genes_df.head(top_n)
                            matching_genes = top_genes['gene_name'].tolist()
                            raw_scores = top_genes['confidence'].tolist()
                            results[query]["note"] = f"Selected top {top_n} genes from {total_available} available genes"
                        else:
                            matching_genes = top_genes_df['gene_name'].tolist()
                            raw_scores = top_genes_df['confidence'].tolist()
                            results[query]["note"] = f"Found {total_available} genes"
                        
                        results[query].update({
                            "genes": matching_genes,
                            "scores": raw_scores,
                            "count": len(matching_genes)
                        })
                    else:
                        results[query]["note"] = f"No genes found with confidence >= {confidence_cutoff}"
                else:
                    results[query]["note"] = f"Match found but similarity ({best_score}%) below threshold ({similarity_threshold}%)"
                
        except pd.errors.EmptyDataError:
            print(f"Error: Empty file at {file_path}")
            return {disease: {"genes": [], "count": 0, "note": "Error: Empty file"} 
                   for disease in disease_list}
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return {disease: {"genes": [], "count": 0, "note": f"Error: {str(e)}"} 
                   for disease in disease_list}
            
        return results
    
    try:
        # Validate inputs
        if not disease_list:
            raise ValueError("Disease list cannot be empty")
        
        if confidence_cutoff < 0:
            raise ValueError("Confidence cutoff must be non-negative")
        
        if top_n is not None and top_n < 1:
            raise ValueError("top_n must be positive when specified")
        
        # Search for disease-gene associations
        detailed_results = search_disease_genes(disease_list)
        
        # Validate all genes found
        all_genes = []
        for result in detailed_results.values():
            all_genes.extend(result["genes"])
        valid_genes, invalid_genes = validate_genes(all_genes)
        valid_genes_set = set(valid_genes)
        
        # Filter results to only include valid genes
        final_results = {}
        for disease, result in detailed_results.items():
            # Filter to only valid genes first
            valid_gene_indices = [i for i, gene in enumerate(result["genes"]) if gene in valid_genes_set]
            valid_genes_list = [result["genes"][i] for i in valid_gene_indices]
            valid_scores_list = [result["scores"][i] for i in valid_gene_indices]
            
            # Normalize the scores if any valid genes exist
            if valid_scores_list:
                normalized_scores = normalize_scores(valid_scores_list)
            else:
                normalized_scores = []
            
            # Create entries with normalized scores
            valid_entries = [
                {"gene": gene, "score": score}
                for gene, score in zip(valid_genes_list, normalized_scores)
            ]
            final_results[disease] = valid_entries
        
        # Prepare output with all results
        return {
            "status": "success",
            "metadata": {
                "diseases_searched": len(disease_list),
                "confidence_cutoff": confidence_cutoff,
                "top_n": top_n,
                "genes_found": {
                    disease: len(final_results[disease])
                    for disease in disease_list
                },
                "invalid_genes": invalid_genes
            },
            "results": detailed_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "metadata": {},
            "results": {
                disease: {"genes": [], "count": 0, "note": f"Error: {str(e)}"} 
                for disease in disease_list
            },
            "final_results": {disease: [] for disease in disease_list}
        }


@tool
def string_database_search(
    gene_list: List[str],
    score_cutoff: float = 0.9,
    species_id: int = 9606,
    limit: int = 1000,
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Search for protein-protein interactions using the STRING database. The STRING (Search Tool for the Retrieval of Interacting Genes/Proteins) database is a biological database of known and predicted protein-protein interactions.
    
    Args:
        gene_list: List of gene names to find interaction partners for
        score_cutoff: Minimum combined score threshold (default: 0.9)
        species_id: NCBI species identifier (default: 9606 for human)
        limit: Maximum number of interaction partners per gene (default: 1000)
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing interaction partners and normalized confidence scores
    """
    from collections import defaultdict
    
    base_url = "https://version-11-5.string-db.org/api"
    output_format = "tsv-no-header"
    method = "interaction_partners"
    
    try:
        # Validate genes first
        valid_genes, invalid_genes = validate_genes(gene_list)
        if not valid_genes:
            return {
                "status": "error",
                "error_message": f"No valid genes found. Invalid genes: {invalid_genes}",
                "raw_interactions": {},
                "final_results": {}
            }

        request_url = "/".join([base_url, output_format, method])

        params = {
            "identifiers": "%0d".join(valid_genes),
            "species": species_id,
            "limit": limit,
            "caller_identity": "www.awesome_app.org"
        }

        # Make API request
        response = requests.get(request_url, params=params)
        response.raise_for_status()

        # Process results
        interactions = defaultdict(list)
        for line in response.text.strip().split("\n"):
            if not line.strip():
                continue
                
            fields = line.strip().split("\t")
            
            if len(fields) >= 6:
                query_id = fields[0]
                query_name = fields[2]
                partner_name = fields[3]
                combined_score = float(fields[5])

                if combined_score >= score_cutoff:
                    interactions[query_name].append({
                        "partner": partner_name,
                        "score": combined_score
                    })

        # Create final results with normalized scores
        final_results = {}
        raw_interactions = dict(interactions)
        
        for gene, partners in interactions.items():
            if not partners:
                continue
                
            # Get scores above cutoff
            scores = [p["score"] for p in partners]
            if not scores:
                continue
            
            # Get normalized scores
            normalized_scores = normalize_scores(scores)
            
            final_results[gene] = [
                {
                    "gene": partners[i]["partner"],
                    "score": normalized_scores[i]
                }
                for i in range(len(partners))
            ]

        return {
            "status": "success",
            "metadata": {
                "genes_searched": len(valid_genes),
                "invalid_genes": invalid_genes,
                "score_cutoff": score_cutoff,
                "interactions_found": {
                    gene: len(final_results.get(gene, []))
                    for gene in valid_genes
                }
            },
            "raw_interactions": raw_interactions,
            "final_results": final_results
        }

    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "error_message": f"Error in STRING database request: {str(e)}",
            "raw_interactions": {},
            "final_results": {}
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}",
            "raw_interactions": {},
            "final_results": {}
        }


@tool
def go_terms_search(
    query_list: List[str],
    max_candidates: int = 100,
    similarity_threshold: int = 10,
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Search for genes associated with Gene Ontology (GO) terms. Gene Ontology provides a controlled vocabulary of terms describing gene product characteristics and gene product annotation data across various databases.
    
    Args:
        query_list: List of biological processes, molecular functions, or cellular components to search for
        max_candidates: Maximum number of GO term candidates to consider (default: 100)
        similarity_threshold: Minimum similarity score (0-100) for fuzzy matching (default: 10)
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing genes associated with each GO term query and normalized relevance scores
    """
    import json
    from rapidfuzz import process, fuzz
    from rapidfuzz.utils import default_process
    
    json_directory = "resource/GO"
    
    def load_json_files() -> List[Dict[str, Any]]:
        """Load all JSON files from the GO directory"""
        data_list = []
        try:
            for file_name in os.listdir(json_directory):
                if file_name.lower().endswith(".json"):
                    file_path = os.path.join(json_directory, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data_list.append(json.load(f))
                    except Exception as e:
                        print(f"Could not parse {file_path} as JSON: {e}")
        except FileNotFoundError:
            print(f"GO directory not found: {json_directory}")
        return data_list

    def extract_all_term_names(data_files: List[Dict[str, Any]]) -> List[str]:
        """Extract all GO term names from the loaded JSON files"""
        term_names = []
        for data_file in data_files:
            if isinstance(data_file, dict):
                term_names.extend(list(data_file.keys()))
        return term_names

    def similarity_search(query: str, all_terms: List[str]) -> List[str]:
        """Find most similar GO term names using fuzzy matching"""
        matches = process.extract(
            query, 
            all_terms, 
            scorer=fuzz.token_sort_ratio, 
            processor=default_process,
            limit=max_candidates,
            score_cutoff=similarity_threshold
        )
        return [match[0] for match in matches]

    def match_terms_with_llm(queries: List[str], all_terms: List[str]) -> Dict[str, str]:
        """Use LLM to find the most relevant GO terms"""
        matched_terms = {}
        
        for query in queries:
            candidates = similarity_search(query, all_terms)
            
            if not candidates:
                matched_terms[query] = ""
                continue
            
            prompt = f"""
You are an assistant to biologists. Given a user query, your task is to identify the SINGLE most relevant Gene Ontology (GO) term from the provided list.

IMPORTANT: You must return the EXACT GO term name from the list as it appears, including any IDs, numbers, or special characters. Do not modify, simplify, or clean up the term names.

Think step by step and select only the most relevant GO term name. Please format your response following the response format and make sure it is parsable by JSON.

User query: {query}

Available GO terms:
{chr(10).join(candidates)}

Response format:
{{
    "thoughts": "your step-by-step thinking process",
    "matched_term_name": "matched GO term name EXACTLY as it appears in the list, or NA if no relevant term found"
}}
"""
            
            try:
                result = json_llm_call(prompt, model_name=model_name)
                matched_term = result.get('matched_term_name', "")
                matched_terms[query] = matched_term if matched_term != "NA" else ""
            except Exception as e:
                print(f"Error matching GO term '{query}' with LLM: {str(e)}")
                matched_terms[query] = ""
                
        return matched_terms

    def find_genes_for_terms(matched_terms: Dict[str, str], data_files: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Find genes for matched GO terms"""
        results = {}
        
        for query, term_name in matched_terms.items():
            found_genes = set()
            
            if term_name:  # Only search if we have a matched term
                for data_file in data_files:
                    if isinstance(data_file, dict) and term_name in data_file:
                        entry = data_file[term_name]
                        if isinstance(entry, dict):
                            genes = entry.get("geneSymbols", [])
                            if isinstance(genes, list):
                                found_genes.update(genes)
            
            results[query] = sorted(found_genes)
        
        return results

    try:
        # Load GO data files
        data_files = load_json_files()
        if not data_files:
            return {
                "status": "error",
                "error_message": f"No GO JSON files found in {json_directory}",
                "raw_results": {},
                "final_results": {}
            }

        # Extract all GO terms
        all_terms = extract_all_term_names(data_files)
        if not all_terms:
            return {
                "status": "error",
                "error_message": "No GO terms found in JSON files",
                "raw_results": {},
                "final_results": {}
            }

        final_results = {}
        raw_results = {}
        
        # Match terms with LLM
        matched_terms = match_terms_with_llm(query_list, all_terms)
        
        # Find genes for matched terms
        genes_by_query = find_genes_for_terms(matched_terms, data_files)
        
        for query in query_list:
            # Get genes for the current query
            found_genes = genes_by_query.get(query, [])
            
            # Validate genes 
            valid_genes, invalid_genes = validate_genes(found_genes)
            
            # Store raw results
            raw_results[query] = {
                "matched_go_term": matched_terms.get(query, ""),
                "total_genes_found": len(found_genes),
                "valid_genes": len(valid_genes),
                "invalid_genes": len(invalid_genes)
            }
            
            # Assign initial raw scores of 1.0 to each gene
            raw_scores = [1.0] * len(valid_genes)
            
            # Normalize scores
            normalized_scores = normalize_scores(raw_scores, temperature=1.0)
            
            # Create final results with normalized scores
            final_results[query] = [
                {"gene": gene, "score": score} 
                for gene, score in zip(valid_genes, normalized_scores)
            ]
        
        return {
            "status": "success",
            "metadata": {
                "queries_processed": len(query_list),
                "go_terms_matched": {
                    query: matched_terms.get(query, "")
                    for query in query_list
                },
                "genes_found": {
                    query: len(final_results.get(query, []))
                    for query in query_list
                }
            },
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def screen_data_analysis(
    input_file: str,
    p_value_threshold: float = 0.05,
    effect_size_threshold: Optional[float] = None,
    top_n: Optional[int] = None,
    score_column: Optional[str] = None,
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Process and filter genetic screen data to identify significant genes based on statistical measures. Handles various formats of screen results, supporting filtering by p-values, effect sizes, or custom scoring.
    
    Args:
        input_file: Path to the input CSV file containing screen results
        p_value_threshold: Threshold for p-values (default: 0.05)
        effect_size_threshold: Threshold for effect sizes (optional)
        top_n: Number of top genes to return (optional)
        score_column: Optional specific column to use for scoring
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing filtered genes with normalized relevance scores and processing metadata
    """
    try:
        # Load screen data - try different approaches
        try:
            # Try loading with 'id' as index
            screen_data = pd.read_csv(input_file, index_col='id')
        except (KeyError, ValueError):
            # If that fails, load normally then handle index in filter method
            screen_data = pd.read_csv(input_file)
        
        # Create a copy of the data
        data = screen_data.copy()
        
        # Determine which column to use for gene identifiers
        id_col = 'Gene' if 'Gene' in data.columns else 'id'
        if id_col in data.columns and id_col != data.index.name:
            data = data.set_index(id_col)
        
        # Filter logic based on available data
        p_value_keywords = ['p-value', 'p_value', 'pvalue', 'pval']
        effect_keywords = ['effect', 'fold', 'log2fc', 'lfc']
        
        has_p_value = any(col for col in data.columns if any(keyword in col.lower() for keyword in p_value_keywords))
        has_effect = any(col for col in data.columns if any(keyword in col.lower() for keyword in effect_keywords))
        
        if p_value_threshold is not None and has_p_value:
            # Get minimum p-value between positive and negative (if both exist)
            if all(col in data.columns for col in ['pos|p-value', 'neg|p-value']):
                data['min_p_value'] = data[['pos|p-value', 'neg|p-value']].min(axis=1)
            else:
                # Find any column with p-value in the name
                p_value_col = next((col for col in data.columns if any(keyword in col.lower() for keyword in p_value_keywords)), None)
                if p_value_col:
                    # Convert to numeric, handling any string values
                    data['min_p_value'] = pd.to_numeric(data[p_value_col], errors='coerce')
            
            # Filter by p-value threshold
            significant_genes = data[data['min_p_value'] < p_value_threshold]
            # Sort by minimum p-value
            sorted_genes = significant_genes.sort_values('min_p_value')
            # Calculate -log10(p-value) for scoring
            sorted_genes['score'] = -np.log10(sorted_genes['min_p_value'])
            
        elif effect_size_threshold is not None and has_effect:
            # Find effect size column
            effect_col = next((col for col in data.columns if any(keyword in col.lower() for keyword in effect_keywords)), None)
            
            # Use absolute values for filtering
            # Convert to numeric first
            data[effect_col] = pd.to_numeric(data[effect_col], errors='coerce')
            data['abs_effect'] = data[effect_col].abs()
            
            # Filter by effect size threshold
            significant_genes = data[data['abs_effect'] > effect_size_threshold]
            # Sort by absolute effect size (descending)
            sorted_genes = significant_genes.sort_values('abs_effect', ascending=False)
            # Use absolute effect size for scoring
            sorted_genes['score'] = sorted_genes['abs_effect']
            
        else:
            # Auto-detect best scoring method
            if has_p_value:
                # Use p-value with default threshold
                p_value_col = next((col for col in data.columns if any(keyword in col.lower() for keyword in p_value_keywords)), None)
                if p_value_col:
                    # Convert to numeric, handling any string values
                    data['min_p_value'] = pd.to_numeric(data[p_value_col], errors='coerce')
                    # Use default threshold if not provided
                    threshold = p_value_threshold if p_value_threshold is not None else 0.05
                    significant_genes = data[data['min_p_value'] < threshold]
                    sorted_genes = significant_genes.sort_values('min_p_value')
                    sorted_genes['score'] = -np.log10(sorted_genes['min_p_value'])
                else:
                    raise ValueError("P-value column detected but could not be processed")
            elif has_effect:
                # Use effect size without threshold
                effect_col = next((col for col in data.columns if any(keyword in col.lower() for keyword in effect_keywords)), None)
                if effect_col:
                    # Convert to numeric first
                    data[effect_col] = pd.to_numeric(data[effect_col], errors='coerce')
                    data['abs_effect'] = data[effect_col].abs()
                    significant_genes = data
                    sorted_genes = significant_genes.sort_values('abs_effect', ascending=False)
                    sorted_genes['score'] = sorted_genes['abs_effect']
                else:
                    raise ValueError("Effect size column detected but could not be processed")
            elif score_column is not None and score_column in data.columns:
                # Use the specified column directly
                data['score'] = data[score_column].abs()
                significant_genes = data
                sorted_genes = significant_genes.sort_values('score', ascending=False)
            else:
                raise ValueError("Could not determine scoring method from data. Available columns: " + str(list(data.columns)))
        
        # Limit to top N if specified
        if top_n is not None:
            sorted_genes = sorted_genes.head(top_n)
        
        # Validate gene names
        valid_genes, invalid_genes = validate_genes(sorted_genes.index.tolist())
        sorted_genes = sorted_genes.loc[valid_genes]
        
        # Reset index to make id a column
        sorted_genes = sorted_genes.reset_index()
        col_name = id_col if id_col == 'Gene' else 'id'
        sorted_genes = sorted_genes.rename(columns={'index': col_name})
        
        # Get input file name from path
        file_name = input_file.split('/')[-1]
        
        # Normalize scores
        scores = sorted_genes['score'].tolist()
        normalized_scores = normalize_scores(scores)
        
        # Create final results
        final_results = {
            file_name: [
                {"gene": gene_name, "score": score}
                for gene_name, score in zip(valid_genes, normalized_scores)
            ]
        }
        
        return {
            "status": "success",
            "metadata": {
                "file_processed": file_name,
                "significant_genes": len(valid_genes),
                "threshold_used": p_value_threshold if p_value_threshold is not None else effect_size_threshold,
                "invalid_genes": invalid_genes
            },
            "final_results": final_results
        }
        
    except FileNotFoundError:
        return {
            "status": "error",
            "error_message": f"Input file not found: {input_file}",
            "final_results": {}
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "final_results": {}
        }


@tool
def tcga_survival_analysis(
    cancer_types: List[str],
    threshold: float = 1.96,
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Identify genes associated with cancer survival from The Cancer Genome Atlas (TCGA) data. This tool analyzes multiple data types (CNA, methylation, gene expression, miRNA, and mutations) to find genes with significant associations to patient survival across different cancer types.
    
    Args:
        cancer_types: List of cancer types (common names like "Breast Cancer" or "Chronic Myeloid Leukemia")
        threshold: Z-score threshold for statistical significance (default: 1.96)
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing genes associated with survival for each cancer type with normalized relevance scores
    """
    from math import sqrt
    
    # TCGA codes mapping
    TCGA_CODES = {
        'LAML', 'ACC', 'BLCA', 'LGG', 'BRCA', 'CESC', 'CHOL', 'LCML', 'COAD',
        'CNTL', 'ESCA', 'FPPP', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LIHC',
        'LUAD', 'LUSC', 'DLBC', 'MESO', 'MISC', 'OV', 'PAAD', 'PCPG', 'PRAD',
        'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THYM', 'THCA', 'UCS', 'UCEC', 'UVM'
    }
    
    data_files = {
        "CNA": "resource/TCGA/Table S1 - univariate Cox models_S1A - CNA.csv",
        "Methylation": "resource/TCGA/Table S1 - univariate Cox models_S1B - DNA methylation.csv",
        "Gene Expression": "resource/TCGA/Table S1 - univariate Cox models_S1C - Gene expression.csv",
        "miRNA": "resource/TCGA/Table S1 - univariate Cox models_S1D - miRNA expression.csv",
        "Mutations": "resource/TCGA/Table S1 - univariate Cox models_S1E - Mutations.csv"
    }
    
    TCGA_PROMPT = """
You are an assistant to biologists. Given the user query, your task is to think step by step to identify the most relevant TCGA study abbreviation(s) from the provided mapping table. For some cancers like lung cancer, return all relevant subtypes. 

IMPORTANT: If the query is too general (e.g., "cancer", "tumor", "solid tumor", "carcinoma", "neoplasm"), return "TOO_GENERAL" as the tcga_code. Overly general terms should not match with all possible cancer types.

If the query is specific to a cancer type but could match multiple types (e.g. "lung cancer" matching both LUAD and LUSC), that's acceptable and you should return all relevant matches.

Please format your response following response format and make sure it is parsable by JSON.

User query:
{query}

TCGA mapping table:
LAML    Acute Myeloid Leukemia
ACC     Adrenocortical carcinoma
BLCA    Bladder Urothelial Carcinoma
LGG     Brain Lower Grade Glioma
BRCA    Breast invasive carcinoma
CESC    Cervical squamous cell carcinoma and endocervical adenocarcinoma
CHOL    Cholangiocarcinoma
LCML    Chronic Myelogenous Leukemia or Chronic Myeloid Leukemia
COAD    Colon adenocarcinoma
CNTL    Controls
ESCA    Esophageal carcinoma
FPPP    FFPE Pilot Phase II
GBM     Glioblastoma multiforme
HNSC    Head and Neck squamous cell carcinoma
KICH    Kidney Chromophobe
KIRC    Kidney renal clear cell carcinoma
KIRP    Kidney renal papillary cell carcinoma
LIHC    Liver hepatocellular carcinoma
LUAD    Lung adenocarcinoma
LUSC    Lung squamous cell carcinoma
DLBC    Lymphoid Neoplasm Diffuse Large B-cell Lymphoma
MESO    Mesothelioma
MISC    Miscellaneous
OV      Ovarian serous cystadenocarcinoma
PAAD    Pancreatic adenocarcinoma
PCPG    Pheochromocytoma and Paraganglioma
PRAD    Prostate adenocarcinoma
READ    Rectum adenocarcinoma
SARC    Sarcoma
SKCM    Skin Cutaneous Melanoma
STAD    Stomach adenocarcinoma
TGCT    Testicular Germ Cell Tumors
THYM    Thymoma
THCA    Thyroid carcinoma
UCS     Uterine Carcinosarcoma
UCEC    Uterine Corpus Endometrial Carcinoma
UVM     Uveal Melanoma

Response format:
{{
"thoughts": "your step-by-step thinking process",
"tcga_code": "Most relevant TCGA study abbreviation(s). Can be a single code (e.g., 'PAAD') or an array of codes (e.g., ['LUAD', 'LUSC']). If the query is too general, return 'TOO_GENERAL'. If no match is found, return 'NA'."
}}
"""

    def get_tcga_code(cancer_name: str) -> List[str]:
        """Convert common cancer name to TCGA study abbreviation(s) using LLM."""
        prompt = TCGA_PROMPT.format(query=cancer_name)
        response = json_llm_call(prompt, model_name=model_name)
        tcga_code = response.get('tcga_code', 'NA')
        
        # Handle too general case
        if tcga_code == "TOO_GENERAL":
            print(f"Warning: '{cancer_name}' is too general. Please specify a particular cancer type.")
            return []
            
        # Handle both single string and list responses
        if isinstance(tcga_code, list):
            valid_codes = [code for code in tcga_code if code in TCGA_CODES]
            return valid_codes if valid_codes else []
        return [tcga_code] if tcga_code in TCGA_CODES else []

    def get_survival_genes(cancer_type: str, threshold: float = 1.96):
        results = {}
        
        for category, file_path in data_files.items():
            try:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip().str.lower() 
                
                if cancer_type.lower() not in df.columns:
                    print(f"Skipping {category}: {cancer_type} not found in columns.")
                    continue
                
                gene_col = next((col for col in df.columns if "gene" in col.lower()), None)
                if gene_col is None:
                    print(f"Skipping {category}: No column with 'Gene' found.")
                    continue
                
                z_col = df.columns[-1]  # Get the last column as the Stouffer's Z-score column
                
                # Clean gene names by removing extra quotes
                df[gene_col] = df[gene_col].str.replace("'", "")
                
                df_filtered = df[abs(df[cancer_type.lower()]) > threshold][[gene_col, cancer_type.lower(), z_col]]
                results[category] = df_filtered.to_dict(orient='records')
            except FileNotFoundError:
                print(f"TCGA data file not found: {file_path}")
                continue
        
        return results

    try:
        final_results = {}
        raw_results = {}
        
        # Process all queries and collect genes first
        all_genes = set()
        for query in cancer_types:
            tcga_codes = get_tcga_code(query)
            
            if not tcga_codes:
                raw_results[query] = []
                final_results[query] = []
                continue
            
            for tcga_code in tcga_codes:
                genes_by_category = get_survival_genes(tcga_code, threshold)
                if genes_by_category:
                    # Collect all unique genes
                    for category, gene_list in genes_by_category.items():
                        all_genes.update(gene_data["gene"] for gene_data in gene_list if "gene" in gene_data)
                        
                    # Store raw results
                    if query not in raw_results:
                        raw_results[query] = []
                    raw_results[query].append({
                        "tcga_code": tcga_code,
                        "category_data": genes_by_category
                    })
        
        # Validate all unique genes at once
        valid_genes, invalid_genes = validate_genes(list(all_genes))
        valid_genes = set(valid_genes)
        
        # Process results using the pre-validated gene set
        for query in cancer_types:
            if query in raw_results:
                for result in raw_results[query]:
                    tcga_code = result["tcga_code"]
                    genes_by_category = result["category_data"]
                    
                    # Create a dictionary to store combined Z-scores for each gene
                    gene_scores = {}
                    gene_counts = {}
                    
                    # Only process valid genes
                    for category, gene_list in genes_by_category.items():
                        if gene_list:  # Check if gene_list is not empty
                            z_col = list(gene_list[0].keys())[-1]
                            for gene_data in gene_list:
                                gene = gene_data.get("gene", "")
                                if gene in valid_genes:  # Only process validated genes
                                    z_score = float(gene_data.get(z_col, 0))
                                    
                                    if gene not in gene_scores:
                                        gene_scores[gene] = 0
                                        gene_counts[gene] = 0
                                    
                                    gene_scores[gene] += z_score
                                    gene_counts[gene] += 1
                    
                    # Calculate combined Z-scores
                    combined_results = []
                    for gene in gene_scores:
                        combined_score = abs(gene_scores[gene] / sqrt(gene_counts[gene]))
                        combined_results.append({
                            "gene": gene,
                            "score": combined_score
                        })
                    
                    # Sort by combined Z-score
                    combined_results.sort(key=lambda x: x["score"], reverse=True)
                    
                    # Normalize scores
                    all_scores = [item["score"] for item in combined_results]
                    normalized_scores = normalize_scores(all_scores)
                    
                    # Update scores with normalized values
                    for i, item in enumerate(combined_results):
                        item["score"] = normalized_scores[i]
                        
                    final_results[tcga_code] = combined_results

        return {
            "status": "success",
            "metadata": {
                "cancer_types_processed": len(cancer_types),
                "tcga_codes_found": list(final_results.keys()),
                "total_genes_analyzed": len(all_genes),
                "valid_genes": len(valid_genes),
                "invalid_genes": invalid_genes
            },
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def drug_gene_network_search(
    drug_queries: List[str],
    graph_path: str = "resource/RxGrid/G_full.p",
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Identify relationships between drugs and genes using a graph-based database. This tool can match potentially misspelled drug names to their correct counterparts and extract gene relationships from the RxGrid network.
    
    Args:
        drug_queries: List of drug names (which may contain misspellings or variations)
        graph_path: Path to the pickled graph file (default: "resource/RxGrid/G_full.p")
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing matched drugs and their associated genes with relationship details
    """
    import pickle
    import difflib
    import networkx as nx
    
    try:
        # Load the graph
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)

        drug_list = [
            node for node, attrs in G.nodes(data=True)
            if attrs.get('type', '').lower() == 'compound'
        ]
        
        def find_best_drug_match(drug_query: str) -> str:
            """Find the best matching drug name, handling misspellings"""
            # Check for an exact match first
            if drug_query in drug_list:
                return drug_query
            
            # Use difflib to find the closest match
            matches = difflib.get_close_matches(drug_query, drug_list, n=1, cutoff=0.6)
            if matches:
                return matches[0]
            else:
                return None
        
        # Initialize results structure
        results = {
            "status": "success",
            "intermediate_results": {},
            "final_results": {}
        }
        
        # Process each drug query
        for drug_query in drug_queries:
            best_match = find_best_drug_match(drug_query)
            if not best_match:
                results["intermediate_results"][drug_query] = {
                    "status": "error",
                    "message": f"No drug found matching '{drug_query}'."
                }
                continue
            
            # Get neighbors and their edge data
            gene_relationships = []
            for neighbor in G.neighbors(best_match):
                if G.nodes[neighbor].get('type', '').lower() != 'compound':
                    edge_data = G.get_edge_data(best_match, neighbor)
                    gene_relationships.append({
                        'gene': neighbor,
                        'relationship': edge_data
                    })
            
            # Store in intermediate_results using the original query for reference
            results["intermediate_results"][drug_query] = {
                "status": "success",
                "best_match": best_match,
                "gene_relationships": gene_relationships
            }
            
            # First create gene relationships with initial score of 1.0
            gene_scores = [
                {
                    "gene": rel["gene"],
                    "score": 1.0
                }
                for rel in gene_relationships
            ]
            
            # Extract just the scores
            scores = [item["score"] for item in gene_scores]
            
            # Normalize the scores
            normalized_scores = normalize_scores(scores)
            
            # Update the gene_scores with normalized values
            for i, item in enumerate(gene_scores):
                item["score"] = normalized_scores[i]
            
            # Add to final results using the best_match name
            results["final_results"][best_match] = gene_scores
        
        return results
        
    except FileNotFoundError:
        return {
            "status": "error",
            "error_message": f"Graph file not found: {graph_path}",
            "intermediate_results": {},
            "final_results": {}
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "intermediate_results": {},
            "final_results": {}
        }


@tool
def pubchem_drug_gene_search(
    drug_names: List[str],
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Search and extract gene associations for drugs from the PubChem database. PubChem is a comprehensive database of chemical substances and their biological activities. This tool retrieves drug information and uses LLM analysis to extract gene targets and associations.
    
    Args:
        drug_names: List of drug names to search for gene associations
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing gene associations for each drug with normalized relevance scores
    """
    import time
    
    pugrest_base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    pugview_base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data"
    
    def get_cid_from_drug_name(drug_name: str) -> str:
        """Given a drug name, query PubChem to retrieve its CID."""
        url = f"{pugrest_base}/compound/name/{drug_name}/cids/TXT"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"Error retrieving CID for {drug_name}: {response.text}")
                return None
            cids = response.text.strip().split()
            if not cids:
                return None
            return cids[0]
        except requests.exceptions.Timeout:
            print(f"Timeout retrieving CID for {drug_name}")
            return None
        except Exception as e:
            print(f"Error retrieving CID for {drug_name}: {str(e)}")
            return None
    
    def get_compound_record_json(cid: str) -> dict:
        """Retrieve the compound record (in JSON) for the given CID using PubChem PUG-View."""
        url = f"{pugview_base}/compound/{cid}/JSON"
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                print(f"Error retrieving compound record for CID {cid}: {response.text}")
                return None
            return response.json()
        except requests.exceptions.Timeout:
            print(f"Timeout retrieving compound record for CID {cid}")
            return None
        except Exception as e:
            print(f"Error retrieving compound record for CID {cid}: {str(e)}")
            return None
    
    def extract_relevant_text_for_gene_association(record_json: dict) -> str:
        """Extract text from 'Mechanism of Action' section focusing on the actual content in the 'String' fields."""
        text_chunks = []
        max_text_length = 10000  # Maximum text length to limit LLM input size
        
        def traverse_sections(sections):
            for section in sections:
                # Check if this is the "Mechanism of Action" section
                heading = section.get('TOCHeading', '').lower()
                
                if "mechanism of action" in heading:
                    # Extract the description if present
                    if 'Description' in section:
                        text_chunks.append(section['Description'])
                    
                    # Extract text from each Information item
                    if 'Information' in section:
                        for info_item in section['Information']:
                            if 'Value' in info_item and 'StringWithMarkup' in info_item['Value']:
                                for string_item in info_item['Value']['StringWithMarkup']:
                                    if 'String' in string_item:
                                        text_chunks.append(string_item['String'])
                    
                    return  # Once we've found the Mechanism of Action section, we're done
                
                # Recurse into nested sections, if any
                if 'Section' in section:
                    traverse_sections(section['Section'])
        
        try:
            record = record_json.get('Record', {})
            sections = record.get('Section', [])
            traverse_sections(sections)
        except Exception as e:
            print(f"Error traversing record JSON: {e}")
        
        # Truncate to maximum length if needed
        result = "\n\n".join(text_chunks)
        if len(result) > max_text_length:
            return result[:max_text_length] + "... (text truncated)"
        
        if not result:
            return "No Mechanism of Action information found in the record."
            
        return result
    
    def query_llm_for_gene_extraction(drug_name: str, record_text: str) -> dict:
        """Build a prompt for the LLM to extract gene associations from the provided record text."""
        prompt = f"""
You are an assistant to pharmacologists and geneticists. Given the drug name, your task is to think step by step to identify all gene symbols or names that are mentioned as targets or associated with the drug in the provided PubChem compound record text. Please format your response following response format and make sure it is parsable by JSON.

Drug query:
{drug_name}

PubChem compound record text:
{record_text}

Response format:
{{
"thoughts": "your step-by-step analysis of the gene associations in the text",
"drug": "{drug_name}",
"gene_count": "the number of unique genes found",
"genes": "a list of gene names/symbols associated with the drug, please make sure the gene names are in the format of HGNC symbols."
}}

If no gene associations are found, return an empty list for "genes" and set "gene_count" to 0.
"""
        # Query the LLM and return its response
        response = json_llm_call(prompt, model_name=model_name)
        return response
    
    try:
        # Convert to list if input is a single drug name
        if not isinstance(drug_names, list):
            drug_names = [drug_names]
        
        results = []
        all_genes = set()
        preliminary_results = {}
        
        # Process each drug
        for drug_name in drug_names:
            # Get CID from drug name
            cid = get_cid_from_drug_name(drug_name)
            if not cid:
                results.append({
                    "status": "error",
                    "message": f"Could not find a CID for drug '{drug_name}'."
                })
                continue
            
            # Get compound record
            record_json = get_compound_record_json(cid)
            if not record_json:
                results.append({
                    "status": "error",
                    "message": f"Could not retrieve compound record for CID {cid}."
                })
                continue
            
            # Extract text from sections that might include gene/target info
            record_text = extract_relevant_text_for_gene_association(record_json)
            if not record_text:
                record_text = "No detailed gene or target information found in the record."
            
            # Query the LLM to parse the text and extract gene associations
            llm_result = query_llm_for_gene_extraction(drug_name, record_text)

            # Store gene entries temporarily
            if isinstance(llm_result, dict) and "genes" in llm_result:
                genes_list = []
                for gene in llm_result["genes"]:
                    genes_list.append({
                        "gene": gene,
                        "score": 1.0  # Initial score of 1
                    })
                    all_genes.add(gene)
                preliminary_results[drug_name] = genes_list
            
            results.append({
                "status": "success",
                "drug": drug_name,
                "cid": cid,
                "results": llm_result
            })
        
        # Validate all the genes we've collected
        valid_genes, invalid_genes = validate_genes(list(all_genes))
        valid_genes_set = set(valid_genes)
        
        # Filter final_results to only include validated genes
        final_results = {}
        for drug_name, genes_list in preliminary_results.items():
            validated_genes_list = [gene_entry for gene_entry in genes_list 
                                   if gene_entry["gene"] in valid_genes_set]
            
            # Only normalize if we have genes
            if validated_genes_list:
                # Extract the scores for normalization
                scores = [gene_entry["score"] for gene_entry in validated_genes_list]
                # Normalize the scores
                normalized_scores = normalize_scores(scores, temperature=1.0)
                
                # Update the gene entries with normalized scores
                for i, gene_entry in enumerate(validated_genes_list):
                    gene_entry["score"] = normalized_scores[i]
                
                final_results[drug_name] = validated_genes_list
        
        # Return comprehensive results
        return {
            "status": "success",
            "metadata": {
                "drugs_processed": len(drug_names),
                "successful_queries": len(preliminary_results),
                "total_genes_found": len(all_genes),
                "valid_genes": len(valid_genes),
                "invalid_genes": invalid_genes
            },
            "results": results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "results": [],
            "final_results": {}
        }


@tool
def hpo_phenotype_search(phenotype_terms: List[str], model_name: str = "gemini-2.5-pro") -> Dict[str, Any]:
    """
    Search and process Human Phenotype Ontology (HPO) database information. HPO provides a standardized vocabulary of phenotypic abnormalities encountered in human disease, allowing for the mapping of phenotypes to associated genes.
    
    Args:
        phenotype_terms: List of phenotype/disease terms to search in the HPO database
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing status, raw results, and final results
    """
    try:
        base_url = "https://ontology.jax.org/api/hp"
        final_results = {}
        raw_results = {}
        
        for query in phenotype_terms:
            print(f"\nProcessing query: '{query}'")
            
            # Get HPO terms for the query
            try:
                response = requests.get(f"{base_url}/search", params={"q": query})
                if response.status_code == 200:
                    data = response.json()
                    if "terms" in data and len(data["terms"]) > 0:
                        hpo_terms = data["terms"][:10]  # Top 10 terms
                    else:
                        print("No HPO term found for the given input.")
                        hpo_terms = []
                else:
                    print("Failed to connect to the HPO API.")
                    hpo_terms = []
            except Exception as e:
                print(f"Error fetching HPO terms: {str(e)}")
                hpo_terms = []
            
            term_details = []
            if hpo_terms:
                # Use LLM to select the most relevant term
                terms_str = "\n".join([f"{i+1}. {term['id']} - {term['name']}" 
                                     for i, term in enumerate(hpo_terms)])
                
                prompt = f"""
You are an assistant to biologists. Given a user's phenotype or disease query and a list of HPO (Human Phenotype Ontology) terms from the database, your task is to select the most relevant HPO term that matches the query. Please format your response following response format and make sure it is parsable by JSON.

User query:
{query}

Available HPO terms:
{terms_str}

Response format:
{{
"thoughts": "your step-by-step reasoning for selection",
"selected_term": "selected HPO ID (not the full term)"
}}
"""
                try:
                    llm_response = json_llm_call(prompt, model_name=model_name)
                    selected_hpo_id = llm_response.get("selected_term", "")
                    
                    # Find the term name from the original search results
                    term_name = next((term["name"] for term in hpo_terms if term["id"] == selected_hpo_id), "Unknown term")
                    print(f"Selected HPO term for '{query}': {selected_hpo_id} - {term_name}")
                    
                    term_details.append({
                        "hpo_id": selected_hpo_id,
                        "name": term_name
                    })
                except Exception as e:
                    print(f"LLM selection unavailable (using first term): {str(e)}")
                    # Fallback to first term
                    if hpo_terms:
                        term_details.append({
                            "hpo_id": hpo_terms[0]["id"],
                            "name": hpo_terms[0]["name"]
                        })
                        print(f"Using fallback HPO term: {hpo_terms[0]['id']} - {hpo_terms[0]['name']}")
            
            raw_results[query] = term_details
            
            # Collect gene information for each HPO ID
            all_genes = []
            for term_info in term_details:
                hpo_id = term_info['hpo_id']
                try:
                    annotation_url = f"https://ontology.jax.org/api/network/annotation/{hpo_id}"
                    response = requests.get(annotation_url)
                    if response.status_code == 200:
                        data = response.json()
                        if "genes" in data:
                            genes = [gene["name"] for gene in data["genes"]]
                            all_genes.extend(genes)
                except Exception as e:
                    print(f"Error fetching genes for HPO ID {hpo_id}: {str(e)}")
                    continue
            
            # Validate genes and create final results
            valid_genes, invalid_genes = validate_genes(all_genes)
            unique_valid_genes = list(set(valid_genes))
            
            if unique_valid_genes:
                initial_scores = [1.0] * len(unique_valid_genes)
                normalized_scores = normalize_scores(initial_scores)
                final_results[query] = [
                    {"gene": gene, "score": score} 
                    for gene, score in zip(unique_valid_genes, normalized_scores)
                ]
            else:
                final_results[query] = []

        return {
            "status": "success",
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def omim_disease_search(disease_terms: List[str], model_name: str = "gemini-2.5-pro") -> Dict[str, Any]:
    """
    Search and extract gene-disease relationship information from the OMIM database. The OMIM (Online Mendelian Inheritance in Man) database is a comprehensive catalog of human genes and genetic disorders, focusing on the relationship between phenotype and genotype.
    
    Args:
        disease_terms: List of disease or phenotype names to search in OMIM
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing status, raw results, and final results
    """
    try:
        base_url = "https://api.omim.org/api"
        api_key = '7TcdVyyhRWOtu58HFHJeOw'
        final_results = {}
        raw_results = {}
        
        for query in disease_terms:
            print(f"\nProcessing query: '{query}'")
            
            # Search for OMIM terms
            search_url = f"{base_url}/entry/search"
            params = {
                'search': query,
                'start': 0,
                'limit': 10,
                'format': 'json',
                'apiKey': api_key
            }
            
            try:
                response = requests.get(search_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if 'omim' in data and 'searchResponse' in data['omim'] and 'entryList' in data['omim']['searchResponse']:
                        omim_entries = data['omim']['searchResponse']['entryList'][:5]  # Top 5 entries
                    else:
                        print(f"No OMIM entries found for query: {query}")
                        omim_entries = []
                else:
                    print(f"Failed to search OMIM for query: {query}")
                    omim_entries = []
            except Exception as e:
                print(f"Error searching OMIM: {str(e)}")
                omim_entries = []
            
            raw_results[query] = omim_entries
            
            # Extract genes from OMIM entries
            all_genes = []
            for entry in omim_entries:
                omim_number = entry.get('entry', {}).get('mimNumber')
                if omim_number:
                    try:
                        # Get detailed entry information
                        detail_url = f"{base_url}/entry"
                        detail_params = {
                            'mimNumber': omim_number,
                            'include': 'geneMap',
                            'format': 'json',
                            'apiKey': api_key
                        }
                        
                        detail_response = requests.get(detail_url, params=detail_params)
                        if detail_response.status_code == 200:
                            detail_data = detail_response.json()
                            if 'omim' in detail_data and 'entryList' in detail_data['omim']:
                                for detail_entry in detail_data['omim']['entryList']:
                                    if 'entry' in detail_entry and 'geneMap' in detail_entry['entry']:
                                        gene_map = detail_entry['entry']['geneMap']
                                        if 'geneSymbols' in gene_map:
                                            all_genes.extend(gene_map['geneSymbols'].split(', '))
                    except Exception as e:
                        print(f"Error fetching details for OMIM {omim_number}: {str(e)}")
                        continue
            
            # Validate genes and create final results
            valid_genes, invalid_genes = validate_genes(all_genes)
            unique_valid_genes = list(set(valid_genes))
            
            if unique_valid_genes:
                initial_scores = [1.0] * len(unique_valid_genes)
                normalized_scores = normalize_scores(initial_scores)
                final_results[query] = [
                    {"gene": gene, "score": score} 
                    for gene, score in zip(unique_valid_genes, normalized_scores)
                ]
            else:
                final_results[query] = []

        return {
            "status": "success",
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def orphanet_rare_disease_search(disease_terms: List[str], model_name: str = "gemini-2.5-pro") -> Dict[str, Any]:
    """
    Search and process Orphanet rare disease database information. Orphanet is a comprehensive resource for information on rare diseases and orphan drugs, providing data on disease-gene associations.
    
    Args:
        disease_terms: List of rare disease names or terms to search in the Orphanet database
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing status, raw results, and final results
    """
    try:
        base_url = "https://api.orphadata.com"
        final_results = {}
        raw_results = {}
        
        for query in disease_terms:
            print(f"\nProcessing query: '{query}'")
            
            # Get Orphanet terms for the query
            try:
                from urllib.parse import quote
                encoded_term = quote(query)
                response = requests.get(
                    f"{base_url}/rd-cross-referencing/orphacodes/names/{encoded_term}?lang=en",
                    headers={"accept": "application/json"},
                    timeout=30
                )
                
                orphanet_terms = []
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("data", {}).get("results", {})
                    if results and "ORPHAcode" in results:
                        # Convert to list format for consistency
                        orphanet_terms = [results]
                    else:
                        print("No ORPHAcode found for the given term.")
                        orphanet_terms = []
                else:
                    print("Failed to connect to the Orphadata API.")
                    orphanet_terms = []
            except Exception as e:
                print(f"Error retrieving data for '{query}': {e}")
                orphanet_terms = []
            
            term_details = []
            if orphanet_terms:
                # Use LLM to select the most relevant term if needed
                if len(orphanet_terms) > 1:
                    try:
                        # Format the terms for the prompt
                        terms_str = "\n".join([f"{i+1}. ORPHAcode: {term['ORPHAcode']} - {term['PreferredTerm']}" 
                                            for i, term in enumerate(orphanet_terms)])
                        
                        prompt = f"""
You are an assistant to biologists. Given a user's rare disease query and a list of Orphanet disease terms from the database, your task is to select the most relevant Orphanet disease that matches the query. Please format your response following response format and make sure it is parsable by JSON.

User query:
{query}

Available Orphanet terms:
{terms_str}

Response format:
{{
"thoughts": "your step-by-step reasoning for selection",
"selected_term": "orphacode" 
}}
"""
                        llm_response = json_llm_call(prompt, model_name=model_name)
                        selected_code = llm_response.get("selected_term", "")
                        
                        # Find the selected term from the list
                        selected_term = None
                        for term in orphanet_terms:
                            if str(term['ORPHAcode']) == str(selected_code):
                                selected_term = term
                                break
                        
                        if not selected_term:
                            selected_term = orphanet_terms[0]  # Fallback to first term
                            
                    except Exception as e:
                        print(f"LLM selection failed: {str(e)}")
                        selected_term = orphanet_terms[0]  # Fallback to first term
                else:
                    selected_term = orphanet_terms[0]
                
                # Print selected Orphanet term for verification
                orphacode = selected_term.get("ORPHAcode")
                term_name = selected_term.get("PreferredTerm", "Unknown")
                print(f"Selected Orphanet term for '{query}':")
                print(f"  - ORPHAcode {orphacode}: {term_name}")
                
                # Store the Orphanet information
                term_details.append({
                    "orphacode": orphacode,
                    "name": term_name
                })
            
            raw_results[query] = term_details
            
            # Collect gene information for each Orphanet disease
            all_genes = []
            for term_info in term_details:
                orphacode = term_info['orphacode']
                try:
                    response = requests.get(
                        f"{base_url}/rd-associated-genes/orphacodes/{orphacode}",
                        headers={"accept": "application/json"},
                        timeout=30
                    )
                    if response.status_code == 200:
                        data = response.json()
                        # The gene associations are located in:
                        # data → results → DisorderGeneAssociation (a list of associations)
                        associations = data.get("data", {}).get("results", {}).get("DisorderGeneAssociation", [])
                        # Extract just the gene symbols from each association
                        genes = [assoc["Gene"]["Symbol"] for assoc in associations 
                               if "Gene" in assoc and "Symbol" in assoc["Gene"]]
                        all_genes.extend(genes)
                    else:
                        print("Failed to connect to the Orphadata API for gene retrieval.")
                except Exception as e:
                    print(f"Error retrieving genes for ORPHAcode '{orphacode}': {e}")
                    continue
            
            # Validate genes and create final results
            valid_genes, invalid_genes = validate_genes(all_genes)
            unique_valid_genes = list(set(valid_genes))
            
            if unique_valid_genes:
                initial_scores = [1.0] * len(unique_valid_genes)
                normalized_scores = normalize_scores(initial_scores)
                final_results[query] = [
                    {"gene": gene, "score": score} 
                    for gene, score in zip(unique_valid_genes, normalized_scores)
                ]
            else:
                final_results[query] = []

        return {
            "status": "success",
            "metadata": {
                "diseases_searched": len(disease_terms),
                "invalid_genes": sum(len(invalid_genes) for _, invalid_genes in [validate_genes(all_genes) for all_genes in []])
            },
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def coxpres_coexpression_search(gene_list: List[str]) -> Dict[str, Any]:
    """
    Search and process COXPRESdb coexpression database information. COXPRESdb provides gene function predictions based on gene coexpression relationships.
    
    Args:
        gene_list: List of gene symbols to search for coexpression partners
    
    Returns:
        Dictionary containing status, raw results, and final results
    """
    try:
        import urllib.parse
        import time
        
        final_results = {}
        raw_results = {}
        
        # Validate input genes
        valid_genes, invalid_genes = validate_genes(gene_list)
        
        if not valid_genes:
            return {
                "status": "error",
                "error_message": "No valid genes found",
                "raw_results": {},
                "final_results": {}
            }
        
        for gene in valid_genes:
            print(f"\nProcessing gene: {gene}")
            
            try:
                # Build COXPRESdb query URL
                base_url = "https://coxpresdb.jp/cgi-bin/search.cgi"
                
                # Query parameters
                params = {
                    'mode': 'human',
                    'genename': gene,
                    'format': 'text'
                }
                
                # Encode parameters
                query_string = urllib.parse.urlencode(params)
                url = f"{base_url}?{query_string}"
                
                # Send request
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Parse response
                coexpressed_genes = []
                correlation_scores = []
                
                lines = response.text.strip().split('\n')
                for line in lines[1:]:  # Skip header line
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            coexp_gene = parts[0].strip()
                            try:
                                correlation = float(parts[1].strip())
                                if correlation > 0.5:  # Only keep genes with correlation > 0.5
                                    coexpressed_genes.append(coexp_gene)
                                    correlation_scores.append(correlation)
                            except ValueError:
                                continue
                
                # Validate coexpressed genes
                if coexpressed_genes:
                    valid_coexp_genes, invalid_coexp_genes = validate_genes(coexpressed_genes)
                    
                    # Filter scores for valid genes
                    valid_scores = []
                    final_valid_genes = []
                    valid_coexp_set = set(valid_coexp_genes)
                    
                    for i, coexp_gene in enumerate(coexpressed_genes):
                        if coexp_gene in valid_coexp_set:
                            final_valid_genes.append(coexp_gene)
                            valid_scores.append(correlation_scores[i])
                    
                    # Store raw results
                    raw_results[gene] = {
                        "total_coexpressed": len(coexpressed_genes),
                        "valid_coexpressed": len(final_valid_genes),
                        "invalid_coexpressed": len(invalid_coexp_genes)
                    }
                    
                    # Normalize scores
                    if valid_scores:
                        normalized_scores = normalize_scores(valid_scores)
                        final_results[gene] = [
                            {"gene": coexp_gene, "score": norm_score}
                            for coexp_gene, norm_score in zip(final_valid_genes, normalized_scores)
                        ]
                    else:
                        final_results[gene] = []
                else:
                    raw_results[gene] = {
                        "total_coexpressed": 0,
                        "valid_coexpressed": 0,
                        "invalid_coexpressed": 0
                    }
                    final_results[gene] = []
                
                # Add delay to avoid too frequent requests
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                print(f"COXPRESdb API unavailable for gene {gene} (this is expected): {str(e)}")
                raw_results[gene] = {
                    "note": "COXPRESdb API temporarily unavailable",
                    "total_coexpressed": 0,
                    "valid_coexpressed": 0
                }
                final_results[gene] = []
                continue
            except Exception as e:
                print(f"Unexpected error for gene {gene}: {str(e)}")
                raw_results[gene] = {
                    "error": str(e),
                    "total_coexpressed": 0,
                    "valid_coexpressed": 0
                }
                final_results[gene] = []
                continue
        
        return {
            "status": "success",
            "metadata": {
                "genes_processed": len(valid_genes),
                "invalid_input_genes": invalid_genes
            },
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def ensembl_paralog_search(gene_list: List[str], species: str = "human") -> Dict[str, Any]:
    """
    Identify paralog genes using the Ensembl API. Paralogs are genes within the same species that evolved from a common ancestral gene through duplication events. This tool helps discover evolutionary relationships between genes.
    
    Args:
        gene_list: List of gene symbols to search for paralogs
        species: The species to search paralogs for (default: "human")
    
    Returns:
        Dictionary containing paralog genes for each input gene with normalized relevance scores
    """
    
    base_url = "https://rest.ensembl.org"
    species_map = {
        "human": "homo_sapiens",
        "mouse": "mus_musculus", 
        "rat": "rattus_norvegicus",
        "zebrafish": "danio_rerio"
    }
    
    def get_gene_id(gene_symbol: str) -> Optional[str]:
        """Fetch the Ensembl gene ID for a given gene symbol"""
        species_name = species_map.get(species.lower(), "homo_sapiens")
        try:
            response = requests.get(
                f"{base_url}/lookup/symbol/{species_name}/{gene_symbol}",
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("id")
            else:
                print(f"Failed to retrieve gene ID for {gene_symbol}. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error retrieving gene ID for '{gene_symbol}': {e}")
            return None
    
    def get_gene_symbol_from_id(gene_id: str) -> Optional[str]:
        """Fetch the gene symbol for a given Ensembl gene ID"""
        try:
            response = requests.get(
                f"{base_url}/lookup/id/{gene_id}",
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("display_name")
            else:
                print(f"Failed to retrieve gene symbol for ID {gene_id}. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error retrieving gene symbol for ID '{gene_id}': {e}")
            return None
    
    def get_paralogs(gene_id: str) -> List[Dict[str, Any]]:
        """Fetch paralogs for a given Ensembl gene ID"""
        species_name = species_map.get(species.lower(), "homo_sapiens")
        try:
            url = f"{base_url}/homology/id/{species_name}/{gene_id}"
            
            params = {
                "type": "paralogues",
                "format": "condensed", 
                "sequence": "none"
            }
            
            response = requests.get(
                url,
                params=params,
                headers={"Accept": "application/json"},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                homologies = data.get("data", [{}])[0].get("homologies", [])
                
                paralogs = []
                for homology in homologies:
                    paralog_id = homology.get("id")
                    if paralog_id:
                        gene_symbol = get_gene_symbol_from_id(paralog_id)
                        
                        paralog_info = {
                            "gene_symbol": gene_symbol or "",
                            "gene_id": paralog_id,
                            "similarity": homology.get("dn_ds", 0)
                        }
                        paralogs.append(paralog_info)
                
                return paralogs
            else:
                print(f"Failed to retrieve paralogs for gene ID {gene_id}. Status code: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error retrieving paralogs for gene ID '{gene_id}': {e}")
            return []
    
    def get_paralogs_by_symbol(gene_symbol: str) -> List[Dict[str, Any]]:
        """Fetch paralogs for a given gene symbol"""
        gene_id = get_gene_id(gene_symbol)
        if not gene_id:
            print(f"Could not find Ensembl gene ID for {gene_symbol}")
            return []
        
        return get_paralogs(gene_id)
    
    try:
        final_results = {}
        raw_results = {}
        
        for gene_symbol in gene_list:
            # Get paralogs for the gene
            paralogs = get_paralogs_by_symbol(gene_symbol)
            
            # Store raw results
            raw_results[gene_symbol] = paralogs
            
            # Extract paralog gene symbols for validation
            paralog_genes = [p["gene_symbol"] for p in paralogs if p["gene_symbol"]]
            
            # Validate output paralog genes
            valid_paralogs, invalid_paralogs = validate_genes(paralog_genes)
            
            # Collect paralog data with fixed initial score of 1
            valid_paralogs_data = [
                {
                    "gene": paralog["gene_symbol"],
                    "score": 1
                }
                for paralog in paralogs 
                if paralog["gene_symbol"] and paralog["gene_symbol"] in valid_paralogs
            ]
            
            # Extract scores for normalization
            initial_scores = [p["score"] for p in valid_paralogs_data]
            
            # Normalize scores
            normalized_scores = normalize_scores(initial_scores)
            
            # Apply normalized scores to results
            for i, paralog in enumerate(valid_paralogs_data):
                paralog["score"] = normalized_scores[i]
                
            # Store final results with normalized scores
            final_results[gene_symbol] = valid_paralogs_data
        
        return {
            "status": "success",
            "metadata": {
                "genes_processed": len(gene_list),
                "species": species
            },
            "raw_results": raw_results,
            "final_results": final_results
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def gsea_hallmark_search(
    query_list: List[str], 
    json_directory: str = "resource/GSEA",
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Search and extract gene set information from MSigDB JSON files. The MSigDB (Molecular Signatures Database) contains collections of annotated gene sets for use with gene set enrichment analysis including hallmark pathways, curated gene sets, and regulatory motifs.
    
    Args:
        query_list: List of pathway names, hallmarks, chromosome regions, or other gene set identifiers
        json_directory: Path to directory containing MSigDB JSON files (default: "resource/GSEA")
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing genes associated with each gene set query and normalized relevance scores
    """
    import json
    
    def load_json_files() -> List[Dict[str, Any]]:
        """Load all JSON files from the specified directory"""
        data_list = []
        try:
            for file_name in os.listdir(json_directory):
                if file_name.lower().endswith(".json"):
                    file_path = os.path.join(json_directory, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data_list.append(json.load(f))
                    except Exception as e:
                        print(f"Could not parse {file_path} as JSON: {e}")
        except FileNotFoundError:
            print(f"GSEA directory not found: {json_directory}")
        return data_list

    def extract_all_pathway_names(data_files: List[Dict[str, Any]]) -> List[str]:
        """Extract all pathway names from the loaded JSON files"""
        pathway_names = []
        for data_file in data_files:
            if isinstance(data_file, dict):
                pathway_names.extend(list(data_file.keys()))
        return pathway_names

    def match_pathways_with_llm(queries: List[str], all_pathways: List[str]) -> Dict[str, str]:
        """Use LLM to find the most relevant pathway names for multiple queries"""
        prompt = f"""
You are an assistant to biologists. Given multiple user queries, your task is to identify the SINGLE most relevant MSigDB pathway name from the provided list for EACH query.

Think step by step for each query and select only the most relevant pathway name. Please format your response following the response format and make sure it is parsable by JSON.

User queries:
{chr(10).join([f"{i+1}. {query}" for i, query in enumerate(queries)])}

Available pathway names:
{chr(10).join(all_pathways)}

Response format:
{{
"thoughts": "your step-by-step thinking process for each query",
"pathway_matches": [
    {{
        "query": "user query",
        "matched_pathway_name": "matched pathway name or NA if no relevant pathway found"
    }}
]
}}
"""
        
        try:
            result = json_llm_call(prompt, model_name=model_name)
            matched_pathways_list = result.get('pathway_matches', [])
            matched_pathways = {}
            for item in matched_pathways_list:
                if isinstance(item, dict) and 'query' in item and 'matched_pathway_name' in item:
                    matched_pathways[item['query']] = item['matched_pathway_name']
            return matched_pathways
        except Exception as e:
            print(f"Error matching pathways with LLM: {str(e)}")
            return {}

    def find_genes_for_pathways(matched_pathways: Dict[str, str], data_files: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Find genes for multiple matched pathways"""
        results = {}
        
        for query, pathway_name in matched_pathways.items():
            found_genes = set()
            
            if pathway_name and pathway_name != "NA":
                # Look up genes for the matched pathway
                for data_file in data_files:
                    if isinstance(data_file, dict) and pathway_name in data_file:
                        entry = data_file[pathway_name]
                        if isinstance(entry, dict):
                            genes = entry.get("geneSymbols", [])
                            if isinstance(genes, list):
                                found_genes.update(genes)
            
            results[query] = sorted(found_genes)
        
        return results

    try:
        # Load GSEA data files
        data_files = load_json_files()
        if not data_files:
            return {
                "status": "error",
                "error_message": f"No GSEA JSON files found in {json_directory}",
                "raw_results": {},
                "final_results": {}
            }

        # Extract all pathway names
        all_pathways = extract_all_pathway_names(data_files)
        if not all_pathways:
            return {
                "status": "error",
                "error_message": "No pathway names found in JSON files",
                "raw_results": {},
                "final_results": {}
            }

        final_results = {}
        raw_results = {}
        
        # Match all pathways with LLM
        matched_pathways = match_pathways_with_llm(query_list, all_pathways)
        
        # Find genes for all matched pathways
        genes_by_query = find_genes_for_pathways(matched_pathways, data_files)
        
        for query in query_list:
            # Get genes for the current query
            found_genes = genes_by_query.get(query, [])
            
            # Validate genes 
            valid_genes, invalid_genes = validate_genes(found_genes)
            
            # Store raw results
            raw_results[query] = {
                "matched_pathway": matched_pathways.get(query, ""),
                "total_genes_found": len(found_genes),
                "valid_genes": len(valid_genes),
                "invalid_genes": len(invalid_genes)
            }
            
            # Assign initial raw scores of 1.0 to each gene
            raw_scores = [1.0] * len(valid_genes)
            
            # Normalize scores
            normalized_scores = normalize_scores(raw_scores, temperature=1.0)
            
            # Create final results with normalized scores
            final_results[query] = [
                {"gene": gene, "score": score} 
                for gene, score in zip(valid_genes, normalized_scores)
            ]
        
        return {
            "status": "success",
            "metadata": {
                "queries_processed": len(query_list),
                "pathways_matched": {
                    query: matched_pathways.get(query, "")
                    for query in query_list
                },
                "genes_found": {
                    query: len(final_results.get(query, []))
                    for query in query_list
                }
            },
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def wikipathways_search(
    query_list: List[str], 
    json_directory: str = "resource/WikiPathways",
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Search and extract gene set information from WikiPathways JSON files. WikiPathways is a database of biological pathways maintained by the community, containing diverse pathway information across multiple organisms including metabolic pathways, signaling cascades, and disease pathways.
    
    Args:
        query_list: List of pathway names, biological processes, or other pathway identifiers
        json_directory: Path to directory containing WikiPathways JSON files (default: "resource/WikiPathways")
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing genes associated with each pathway query and normalized relevance scores
    """
    import json
    
    def load_json_files() -> List[Dict[str, Any]]:
        """Load all JSON files from the specified directory"""
        data_list = []
        try:
            for file_name in os.listdir(json_directory):
                if file_name.lower().endswith(".json"):
                    file_path = os.path.join(json_directory, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data_list.append(json.load(f))
                    except Exception as e:
                        print(f"Could not parse {file_path} as JSON: {e}")
        except FileNotFoundError:
            print(f"WikiPathways directory not found: {json_directory}")
        return data_list

    def extract_all_pathway_names(data_files: List[Dict[str, Any]]) -> List[str]:
        """Extract all pathway names from the loaded JSON files"""
        pathway_names = []
        for data_file in data_files:
            if isinstance(data_file, dict):
                pathway_names.extend(list(data_file.keys()))
        return pathway_names

    def match_pathways_with_llm(queries: List[str], all_pathways: List[str]) -> Dict[str, str]:
        """Use LLM to find the most relevant pathway names for multiple queries"""
        prompt = f"""
You are an assistant to biologists. Given multiple user queries, your task is to identify the SINGLE most relevant WikiPathways pathway name from the provided list for EACH query.

Think step by step for each query and select only the most relevant pathway name. Please format your response following the response format and make sure it is parsable by JSON.

User queries:
{chr(10).join([f"{i+1}. {query}" for i, query in enumerate(queries)])}

Available pathway names:
{chr(10).join(all_pathways)}

Response format:
{{
"thoughts": "your step-by-step thinking process for each query",
"pathway_matches": [
    {{
        "query": "user query",
        "matched_pathway_name": "matched pathway name or NA if no relevant pathway found"
    }}
]
}}
"""
        
        try:
            result = json_llm_call(prompt, model_name=model_name)
            matched_pathways_list = result.get('pathway_matches', [])
            matched_pathways = {}
            for item in matched_pathways_list:
                if isinstance(item, dict) and 'query' in item and 'matched_pathway_name' in item:
                    matched_pathways[item['query']] = item['matched_pathway_name']
            return matched_pathways
        except Exception as e:
            print(f"Error matching pathways with LLM: {str(e)}")
            return {}

    def find_genes_for_pathways(matched_pathways: Dict[str, str], data_files: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Find genes for multiple matched pathways"""
        results = {}
        
        for query, pathway_name in matched_pathways.items():
            found_genes = set()
            
            if pathway_name and pathway_name != "NA":
                # Look up genes for the matched pathway
                for data_file in data_files:
                    if isinstance(data_file, dict) and pathway_name in data_file:
                        entry = data_file[pathway_name]
                        if isinstance(entry, dict):
                            genes = entry.get("geneSymbols", [])
                            if isinstance(genes, list):
                                found_genes.update(genes)
            
            results[query] = sorted(found_genes)
        
        return results

    try:
        # Load WikiPathways data files
        data_files = load_json_files()
        if not data_files:
            return {
                "status": "error",
                "error_message": f"No WikiPathways JSON files found in {json_directory}",
                "raw_results": {},
                "final_results": {}
            }

        # Extract all pathway names
        all_pathways = extract_all_pathway_names(data_files)
        if not all_pathways:
            return {
                "status": "error",
                "error_message": "No pathway names found in JSON files",
                "raw_results": {},
                "final_results": {}
            }

        final_results = {}
        raw_results = {}
        
        # Match all pathways with LLM
        matched_pathways = match_pathways_with_llm(query_list, all_pathways)
        
        # Find genes for all matched pathways
        genes_by_query = find_genes_for_pathways(matched_pathways, data_files)
        
        for query in query_list:
            # Get genes for the current query
            found_genes = genes_by_query.get(query, [])
            
            # Validate genes 
            valid_genes, invalid_genes = validate_genes(found_genes)
            
            # Store raw results
            raw_results[query] = {
                "matched_pathway": matched_pathways.get(query, ""),
                "total_genes_found": len(found_genes),
                "valid_genes": len(valid_genes),
                "invalid_genes": len(invalid_genes)
            }
            
            # Assign initial raw scores of 1.0 to each gene
            raw_scores = [1.0] * len(valid_genes)
            
            # Normalize scores
            normalized_scores = normalize_scores(raw_scores, temperature=1.0)
            
            # Create final results with normalized scores
            final_results[query] = [
                {"gene": gene, "score": score} 
                for gene, score in zip(valid_genes, normalized_scores)
            ]
        
        return {
            "status": "success",
            "metadata": {
                "queries_processed": len(query_list),
                "pathways_matched": {
                    query: matched_pathways.get(query, "")
                    for query in query_list
                },
                "genes_found": {
                    query: len(final_results.get(query, []))
                    for query in query_list
                }
            },
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def reactome_pathway_search(
    query_list: List[str], 
    json_directory: str = "resource/Reactome",
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Search and extract gene set information from Reactome JSON files. Reactome is an open-source, curated and peer-reviewed pathway database that provides detailed information about biological pathways, reactions, and cellular processes across different species.
    
    Args:
        query_list: List of pathway names, reactions, or cellular process identifiers
        json_directory: Path to directory containing Reactome JSON files (default: "resource/Reactome")
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing genes associated with each pathway/reaction query and normalized relevance scores
    """
    import json
    
    def load_json_files() -> List[Dict[str, Any]]:
        """Load all JSON files from the specified directory"""
        data_list = []
        try:
            for file_name in os.listdir(json_directory):
                if file_name.lower().endswith(".json"):
                    file_path = os.path.join(json_directory, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data_list.append(json.load(f))
                    except Exception as e:
                        print(f"Could not parse {file_path} as JSON: {e}")
        except FileNotFoundError:
            print(f"Reactome directory not found: {json_directory}")
        return data_list

    def extract_all_pathway_names(data_files: List[Dict[str, Any]]) -> List[str]:
        """Extract all pathway names from the loaded JSON files"""
        pathway_names = []
        for data_file in data_files:
            if isinstance(data_file, dict):
                pathway_names.extend(list(data_file.keys()))
        return pathway_names

    def match_pathways_with_llm(queries: List[str], all_pathways: List[str]) -> Dict[str, str]:
        """Use LLM to find the most relevant pathway names for multiple queries"""
        prompt = f"""
You are an assistant to biologists. Given multiple user queries, your task is to identify the SINGLE most relevant Reactome pathway name from the provided list for EACH query.

Think step by step for each query and select only the most relevant pathway name. Please format your response following the response format and make sure it is parsable by JSON.

User queries:
{chr(10).join([f"{i+1}. {query}" for i, query in enumerate(queries)])}

Available pathway names:
{chr(10).join(all_pathways)}

Response format:
{{
"thoughts": "your step-by-step thinking process for each query",
"pathway_matches": [
    {{
        "query": "user query",
        "matched_pathway_name": "matched pathway name or NA if no relevant pathway found"
    }}
]
}}
"""
        
        try:
            result = json_llm_call(prompt, model_name=model_name)
            matched_pathways_list = result.get('pathway_matches', [])
            matched_pathways = {}
            for item in matched_pathways_list:
                if isinstance(item, dict) and 'query' in item and 'matched_pathway_name' in item:
                    matched_pathways[item['query']] = item['matched_pathway_name']
            return matched_pathways
        except Exception as e:
            print(f"Error matching pathways with LLM: {str(e)}")
            return {}

    def find_genes_for_pathways(matched_pathways: Dict[str, str], data_files: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Find genes for multiple matched pathways"""
        results = {}
        
        for query, pathway_name in matched_pathways.items():
            found_genes = set()
            
            if pathway_name and pathway_name != "NA":
                # Look up genes for the matched pathway
                for data_file in data_files:
                    if isinstance(data_file, dict) and pathway_name in data_file:
                        entry = data_file[pathway_name]
                        if isinstance(entry, dict):
                            genes = entry.get("geneSymbols", [])
                            if isinstance(genes, list):
                                found_genes.update(genes)
            
            results[query] = sorted(found_genes)
        
        return results

    try:
        # Load Reactome data files
        data_files = load_json_files()
        if not data_files:
            return {
                "status": "error",
                "error_message": f"No Reactome JSON files found in {json_directory}",
                "raw_results": {},
                "final_results": {}
            }

        # Extract all pathway names
        all_pathways = extract_all_pathway_names(data_files)
        if not all_pathways:
            return {
                "status": "error",
                "error_message": "No pathway names found in JSON files",
                "raw_results": {},
                "final_results": {}
            }

        final_results = {}
        raw_results = {}
        
        # Match all pathways with LLM
        matched_pathways = match_pathways_with_llm(query_list, all_pathways)
        
        # Find genes for all matched pathways
        genes_by_query = find_genes_for_pathways(matched_pathways, data_files)
        
        for query in query_list:
            # Get genes for the current query
            found_genes = genes_by_query.get(query, [])
            
            # Validate genes 
            valid_genes, invalid_genes = validate_genes(found_genes)
            
            # Store raw results
            raw_results[query] = {
                "matched_pathway": matched_pathways.get(query, ""),
                "total_genes_found": len(found_genes),
                "valid_genes": len(valid_genes),
                "invalid_genes": len(invalid_genes)
            }
            
            # Assign initial raw scores of 1.0 to each gene
            raw_scores = [1.0] * len(valid_genes)
            
            # Normalize scores
            normalized_scores = normalize_scores(raw_scores, temperature=1.0)
            
            # Create final results with normalized scores
            final_results[query] = [
                {"gene": gene, "score": score} 
                for gene, score in zip(valid_genes, normalized_scores)
            ]
        
        return {
            "status": "success",
            "metadata": {
                "queries_processed": len(query_list),
                "pathways_matched": {
                    query: matched_pathways.get(query, "")
                    for query in query_list
                },
                "genes_found": {
                    query: len(final_results.get(query, []))
                    for query in query_list
                }
            },
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def cancer_biomarkers_search(
    query_list: List[str], 
    json_path: str = "resource/Open_target/cancerbiomarkers.json",
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Search and extract gene information from Cancer Biomarkers database. The Cancer Biomarkers database features biomarkers of drug sensitivity, resistance, and toxicity for drugs targeting specific targets in cancer, curated by clinical and scientific experts in precision oncology, and classified by cancer type.
    
    Args:
        query_list: List of cancer types or disease names to search for associated biomarker genes
        json_path: Path to Cancer Biomarkers JSON file (default: "resource/Open_target/cancerbiomarkers.json")
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing genes associated with each cancer type query and normalized relevance scores
    """
    import json
    
    def load_json_data() -> List[Dict[str, Any]]:
        """Load the Cancer Biomarkers JSON data (line-delimited JSON format)"""
        data_list = []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                # Each line is a separate JSON object
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data_list.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            print(f"Cancer biomarkers file not found: {json_path}")
        except Exception as e:
            print(f"Could not parse {json_path} as JSON: {e}")
        return data_list

    def extract_all_disease_names(data: List[Dict[str, Any]]) -> List[str]:
        """Extract all unique disease names from the loaded JSON data"""
        disease_names = set()
        for entry in data:
            if isinstance(entry, dict) and "diseaseFromSource" in entry:
                disease_names.add(entry["diseaseFromSource"])
        return sorted(list(disease_names))

    def match_diseases_with_llm(queries: List[str], all_diseases: List[str]) -> Dict[str, str]:
        """Use LLM to find the most relevant disease names for multiple queries"""
        prompt = f"""
You are an assistant to oncologists. Given multiple user queries, your task is to identify the SINGLE most relevant cancer or disease name from the provided list for EACH query.

Think step by step for each query and select only the most relevant disease name. Please format your response following the response format and make sure it is parsable by JSON.

User queries:
{chr(10).join([f"{i+1}. {query}" for i, query in enumerate(queries)])}

Available disease names:
{chr(10).join(all_diseases)}

Response format:
{{
"thoughts": "your step-by-step thinking process for each query",
"disease_matches": [
    {{
        "query": "user query",
        "matched_disease_name": "matched disease name or NA if no relevant disease found"
    }}
]
}}
"""
        
        try:
            result = json_llm_call(prompt, model_name=model_name)
            matched_diseases_list = result.get('disease_matches', [])
            matched_diseases = {}
            for item in matched_diseases_list:
                if isinstance(item, dict) and 'query' in item and 'matched_disease_name' in item:
                    matched_diseases[item['query']] = item['matched_disease_name']
            return matched_diseases
        except Exception as e:
            print(f"Error matching diseases with LLM: {str(e)}")
            return {}

    def find_genes_for_diseases(matched_diseases: Dict[str, str], data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Find genes for multiple matched diseases"""
        results = {}
        
        for query, disease_name in matched_diseases.items():
            found_genes = set()
            
            if disease_name and disease_name != "NA":
                # Look up genes for the matched disease
                for entry in data:
                    if isinstance(entry, dict) and entry.get("diseaseFromSource") == disease_name:
                        target_gene = entry.get("targetFromSourceId")
                        if target_gene:
                            found_genes.add(target_gene)
            
            results[query] = sorted(found_genes)
        
        return results

    try:
        # Load Cancer Biomarkers data
        data = load_json_data()
        if not data:
            return {
                "status": "error",
                "error_message": f"No cancer biomarkers data found in {json_path}",
                "raw_results": {},
                "final_results": {}
            }

        # Extract all disease names
        all_diseases = extract_all_disease_names(data)
        if not all_diseases:
            return {
                "status": "error",
                "error_message": "No disease names found in data",
                "raw_results": {},
                "final_results": {}
            }

        final_results = {}
        raw_results = {}
        
        # Match all diseases with LLM
        matched_diseases = match_diseases_with_llm(query_list, all_diseases)
        
        # Find genes for all matched diseases
        genes_by_query = find_genes_for_diseases(matched_diseases, data)
        
        for query in query_list:
            # Get genes for the current query
            found_genes = genes_by_query.get(query, [])
            
            # Validate genes 
            valid_genes, invalid_genes = validate_genes(found_genes)
            
            # Store raw results
            raw_results[query] = {
                "matched_disease": matched_diseases.get(query, ""),
                "total_genes_found": len(found_genes),
                "valid_genes": len(valid_genes),
                "invalid_genes": len(invalid_genes)
            }
            
            # Assign initial raw scores of 1.0 to each gene
            raw_scores = [1.0] * len(valid_genes)
            
            # Normalize scores
            normalized_scores = normalize_scores(raw_scores, temperature=1.0)
            
            # Create final results with normalized scores
            final_results[query] = [
                {"gene": gene, "score": score} 
                for gene, score in zip(valid_genes, normalized_scores)
            ]
        
        return {
            "status": "success",
            "metadata": {
                "queries_processed": len(query_list),
                "diseases_matched": {
                    query: matched_diseases.get(query, "")
                    for query in query_list
                },
                "genes_found": {
                    query: len(final_results.get(query, []))
                    for query in query_list
                }
            },
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def clingen_search(
    query_list: List[str], 
    json_path: str = "resource/Open_target/clingen.json",
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Search and extract gene information from ClinGen database. The Clinical Genome Resource (ClinGen) Gene-Disease Validity Curation evaluates the strength of evidence supporting or refuting claims that variation in a particular gene causes a particular disease, providing a framework to assess clinical validity in a semi-quantitative manner.
    
    Args:
        query_list: List of diseases or conditions to search for associated genes with clinical evidence
        json_path: Path to ClinGen JSON file (default: "resource/Open_target/clingen.json")
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing genes associated with each disease query and normalized relevance scores based on clinical evidence strength
    """
    import json
    
    # Confidence to score mapping based on ClinGen classification
    confidence_scores = {
        "No reported evidence": 0.01,
        "Refuted": 0.01,
        "Disputed": 0.01,
        "Limited": 0.01,
        "Moderate": 0.5,
        "Strong": 1.0,
        "Definitive": 1.0
    }
    
    def load_json_data() -> List[Dict[str, Any]]:
        """Load the ClinGen JSON data (line-delimited JSON format)"""
        data_list = []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                # Each line is a separate JSON object
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data_list.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            print(f"ClinGen file not found: {json_path}")
        except Exception as e:
            print(f"Could not parse {json_path} as JSON: {e}")
        return data_list

    def extract_all_disease_names(data: List[Dict[str, Any]]) -> List[str]:
        """Extract all unique disease names from the loaded JSON data"""
        disease_names = set()
        for entry in data:
            if isinstance(entry, dict) and "diseaseFromSource" in entry:
                disease_names.add(entry["diseaseFromSource"])
        return sorted(list(disease_names))

    def match_diseases_with_llm(queries: List[str], all_diseases: List[str]) -> Dict[str, str]:
        """Use LLM to find the most relevant disease names for multiple queries"""
        prompt = f"""
You are an assistant to medical genetics experts. Given multiple user queries, your task is to identify the SINGLE most relevant disease name from the provided list for EACH query.

Think step by step for each query and select only the most relevant disease name. Please format your response following the response format and make sure it is parsable by JSON.

User queries:
{chr(10).join([f"{i+1}. {query}" for i, query in enumerate(queries)])}

Available disease names:
{chr(10).join(all_diseases)}

Response format:
{{
"thoughts": "your step-by-step thinking process for each query",
"disease_matches": [
    {{
        "query": "user query",
        "matched_disease_name": "matched disease name or NA if no relevant disease found"
    }}
]
}}
"""
        
        try:
            result = json_llm_call(prompt, model_name=model_name)
            matched_diseases_list = result.get('disease_matches', [])
            matched_diseases = {}
            for item in matched_diseases_list:
                if isinstance(item, dict) and 'query' in item and 'matched_disease_name' in item:
                    matched_diseases[item['query']] = item['matched_disease_name']
            return matched_diseases
        except Exception as e:
            print(f"Error matching diseases with LLM: {str(e)}")
            return {}

    def find_genes_for_diseases(matched_diseases: Dict[str, str], data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Find genes for multiple matched diseases with confidence levels"""
        results = {}
        
        for query, disease_name in matched_diseases.items():
            found_genes = []
            
            if disease_name and disease_name != "NA":
                # Look up genes for the matched disease
                for entry in data:
                    if isinstance(entry, dict) and entry.get("diseaseFromSource") == disease_name:
                        target_gene = entry.get("targetFromSourceId")
                        confidence = entry.get("confidence", "No reported evidence")
                        if target_gene:
                            found_genes.append({
                                "gene": target_gene,
                                "confidence": confidence
                            })
            
            results[query] = found_genes
        
        return results

    try:
        # Load ClinGen data
        data = load_json_data()
        if not data:
            return {
                "status": "error",
                "error_message": f"No ClinGen data found in {json_path}",
                "raw_results": {},
                "final_results": {}
            }

        # Extract all disease names
        all_diseases = extract_all_disease_names(data)
        if not all_diseases:
            return {
                "status": "error",
                "error_message": "No disease names found in data",
                "raw_results": {},
                "final_results": {}
            }

        final_results = {}
        raw_results = {}
        
        # Match all diseases with LLM
        matched_diseases = match_diseases_with_llm(query_list, all_diseases)
        
        # Find genes for all matched diseases
        genes_by_query = find_genes_for_diseases(matched_diseases, data)
        
        for query in query_list:
            # Get genes for the current query
            gene_results = genes_by_query.get(query, [])
            
            # Extract gene symbols and confidence
            found_genes = [g["gene"] for g in gene_results]
            confidence_levels = [g["confidence"] for g in gene_results]
            
            # Validate genes 
            valid_genes, invalid_genes = validate_genes(found_genes)
            
            # Store raw results
            raw_results[query] = {
                "matched_disease": matched_diseases.get(query, ""),
                "total_genes_found": len(found_genes),
                "valid_genes": len(valid_genes),
                "invalid_genes": len(invalid_genes)
            }
            
            # Create a mapping of valid genes to their confidence
            # Use a dictionary to handle duplicates - keep the highest confidence version
            valid_gene_confidence = {}
            for i, gene in enumerate(found_genes):
                if gene in valid_genes:
                    current_confidence = confidence_levels[i]
                    current_score = confidence_scores.get(current_confidence, 0.01)
                    
                    # If gene already exists, only replace if current confidence score is higher
                    if gene not in valid_gene_confidence or current_score > confidence_scores.get(valid_gene_confidence[gene], 0.01):
                        valid_gene_confidence[gene] = current_confidence
            
            # Assign raw scores based on confidence levels using deduplicated genes
            raw_scores = []
            valid_genes_list = list(valid_gene_confidence.keys())  # Using deduplicated genes
            
            for gene in valid_genes_list:
                confidence = valid_gene_confidence.get(gene, "No reported evidence")
                score = confidence_scores.get(confidence, 0.01)
                raw_scores.append(score)
            
            # Normalize scores
            normalized_scores = normalize_scores(raw_scores, temperature=1.0)
            
            # Create final results with normalized scores
            final_results[query] = [
                {"gene": gene, "score": score} 
                for gene, score in zip(valid_genes_list, normalized_scores)
            ]
        
        return {
            "status": "success",
            "metadata": {
                "queries_processed": len(query_list),
                "diseases_matched": {
                    query: matched_diseases.get(query, "")
                    for query in query_list
                },
                "genes_found": {
                    query: len(final_results.get(query, []))
                    for query in query_list
                },
                "confidence_levels": ["Definitive", "Strong", "Moderate", "Limited", "Disputed", "Refuted", "No reported evidence"]
            },
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def gene2phenotype_search(
    query_list: List[str], 
    json_path: str = "resource/Open_target/gene2phenotype.json",
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Search and extract gene information from Gene2Phenotype database. The data in Gene2Phenotype (G2P) is produced and curated from the literature by different sets of panels formed by consultant clinical geneticists. The G2P data is designed to facilitate the development, validation, curation, and distribution of large-scale, evidence-based datasets for use in diagnostic variant filtering.
    
    Args:
        query_list: List of diseases or conditions to search for associated genes
        json_path: Path to Gene2Phenotype JSON file (default: "resource/Open_target/gene2phenotype.json")
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing genes associated with each disease query and normalized relevance scores based on confidence levels
    """
    import json
    
    # Confidence to score mapping based on Gene2Phenotype classification
    confidence_scores = {
        "limited": 0.01,
        "moderate": 0.5,
        "strong": 1.0,
        "both rd and if": 1.0,
        "definitive": 1.0
    }
    
    def load_json_data() -> List[Dict[str, Any]]:
        """Load the Gene2Phenotype JSON data (line-delimited JSON format)"""
        data_list = []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                # Each line is a separate JSON object
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data_list.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            print(f"Gene2Phenotype file not found: {json_path}")
        except Exception as e:
            print(f"Could not parse {json_path} as JSON: {e}")
        return data_list

    def extract_all_disease_names(data: List[Dict[str, Any]]) -> List[str]:
        """Extract all unique disease names from the loaded JSON data"""
        disease_names = set()
        for entry in data:
            if isinstance(entry, dict) and "diseaseFromSource" in entry:
                disease_names.add(entry["diseaseFromSource"])
        return sorted(list(disease_names))

    def match_diseases_with_llm(queries: List[str], all_diseases: List[str]) -> Dict[str, str]:
        """Use LLM to find the most relevant disease names for multiple queries"""
        prompt = f"""
You are an assistant to medical genetics experts. Given multiple user queries, your task is to identify the SINGLE most relevant disease name from the provided list for EACH query.

Think step by step for each query and select only the most relevant disease name. Please format your response following the response format and make sure it is parsable by JSON.

User queries:
{chr(10).join([f"{i+1}. {query}" for i, query in enumerate(queries)])}

Available disease names:
{chr(10).join(all_diseases)}

Response format:
{{
"thoughts": "your step-by-step thinking process for each query",
"disease_matches": [
    {{
        "query": "user query",
        "matched_disease_name": "matched disease name or NA if no relevant disease found"
    }}
]
}}
"""
        
        try:
            result = json_llm_call(prompt, model_name=model_name)
            matched_diseases_list = result.get('disease_matches', [])
            matched_diseases = {}
            for item in matched_diseases_list:
                if isinstance(item, dict) and 'query' in item and 'matched_disease_name' in item:
                    matched_diseases[item['query']] = item['matched_disease_name']
            return matched_diseases
        except Exception as e:
            print(f"Error matching diseases with LLM: {str(e)}")
            return {}

    def find_genes_for_diseases(matched_diseases: Dict[str, str], data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Find genes for multiple matched diseases with confidence levels"""
        results = {}
        
        for query, disease_name in matched_diseases.items():
            found_genes = []
            
            if disease_name and disease_name != "NA":
                # Look up genes for the matched disease
                for entry in data:
                    if isinstance(entry, dict) and entry.get("diseaseFromSource") == disease_name:
                        target_gene = entry.get("targetFromSourceId")
                        confidence = entry.get("confidence", "limited")
                        if target_gene:
                            found_genes.append({
                                "gene": target_gene,
                                "confidence": confidence.lower()
                            })
            
            results[query] = found_genes
        
        return results

    try:
        # Load Gene2Phenotype data
        data = load_json_data()
        if not data:
            return {
                "status": "error",
                "error_message": f"No Gene2Phenotype data found in {json_path}",
                "raw_results": {},
                "final_results": {}
            }

        # Extract all disease names
        all_diseases = extract_all_disease_names(data)
        if not all_diseases:
            return {
                "status": "error",
                "error_message": "No disease names found in data",
                "raw_results": {},
                "final_results": {}
            }

        final_results = {}
        raw_results = {}
        
        # Match all diseases with LLM
        matched_diseases = match_diseases_with_llm(query_list, all_diseases)
        
        # Print matched diseases for quality check
        print("\n=== Disease Matching Results (Quality Check) ===")
        for query, disease in matched_diseases.items():
            print(f"Query: '{query}' → Matched: '{disease}'")
        print("===================================================\n")
        
        # Find genes for all matched diseases
        genes_by_query = find_genes_for_diseases(matched_diseases, data)
        
        for query in query_list:
            # Get genes for the current query
            gene_results = genes_by_query.get(query, [])
            
            # Extract gene symbols and confidence
            found_genes = [g["gene"] for g in gene_results]
            confidence_levels = [g["confidence"] for g in gene_results]
            
            # Validate genes 
            valid_genes, invalid_genes = validate_genes(found_genes)
            
            # Store raw results
            raw_results[query] = {
                "matched_disease": matched_diseases.get(query, ""),
                "total_genes_found": len(found_genes),
                "valid_genes": len(valid_genes),
                "invalid_genes": len(invalid_genes)
            }
            
            # Create a mapping of valid genes to their confidence
            # Use a dictionary to handle duplicates - keep the highest confidence version
            valid_gene_confidence = {}
            for i, gene in enumerate(found_genes):
                if gene in valid_genes:
                    current_confidence = confidence_levels[i]
                    current_score = confidence_scores.get(current_confidence, 0.01)
                    
                    # If gene already exists, only replace if current confidence score is higher
                    if gene not in valid_gene_confidence or current_score > confidence_scores.get(valid_gene_confidence[gene], 0.01):
                        valid_gene_confidence[gene] = current_confidence
            
            # Assign raw scores based on confidence levels using deduplicated genes
            raw_scores = []
            valid_genes_list = list(valid_gene_confidence.keys())  # Using deduplicated genes
            
            for gene in valid_genes_list:
                confidence = valid_gene_confidence.get(gene, "limited")
                score = confidence_scores.get(confidence, 0.01)
                raw_scores.append(score)
            
            # Normalize scores
            normalized_scores = normalize_scores(raw_scores, temperature=1.0)
            
            # Create final results with normalized scores
            final_results[query] = [
                {"gene": gene, "score": score} 
                for gene, score in zip(valid_genes_list, normalized_scores)
            ]
        
        return {
            "status": "success",
            "metadata": {
                "queries_processed": len(query_list),
                "diseases_matched": {
                    query: matched_diseases.get(query, "")
                    for query in query_list
                },
                "genes_found": {
                    query: len(final_results.get(query, []))
                    for query in query_list
                },
                "confidence_levels": ["definitive", "strong", "moderate", "limited", "both rd and if"]
            },
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def gene_burden_search(
    query_list: List[str], 
    json_path: str = "resource/Open_target/gene_burden.json",
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Search and extract gene information from Gene Burden database. Gene burden data comprises gene–phenotype relationships observed in gene-level association tests using rare variant collapsing analyses. These associations result from collapsing rare variants in a gene into a single burden statistic and regressing the phenotype on the burden statistic to test for the combined effects of all rare variants in that gene.
    
    Args:
        query_list: List of phenotypes or conditions to search for associated genes
        json_path: Path to Gene Burden JSON file (default: "resource/Open_target/gene_burden.json")
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing genes associated with each phenotype query and normalized relevance scores based on p-values
    """
    import json
    
    def load_json_data() -> List[Dict[str, Any]]:
        """Load the Gene Burden JSON data (line-delimited JSON format)"""
        data_list = []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                # Each line is a separate JSON object
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data_list.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            print(f"Gene Burden file not found: {json_path}")
        except Exception as e:
            print(f"Could not parse {json_path} as JSON: {e}")
        return data_list

    def extract_all_phenotype_names(data: List[Dict[str, Any]]) -> List[str]:
        """Extract all unique phenotype names from the loaded JSON data"""
        phenotype_names = set()
        for entry in data:
            if isinstance(entry, dict) and "diseaseFromSource" in entry:
                phenotype_names.add(entry["diseaseFromSource"])
        return sorted(list(phenotype_names))

    def match_phenotypes_with_llm(queries: List[str], all_phenotypes: List[str]) -> Dict[str, str]:
        """Use LLM to find the most relevant phenotype names for multiple queries"""
        prompt = f"""
You are an assistant to medical genetics experts. Given multiple user queries, your task is to identify the SINGLE most relevant phenotype name from the provided list for EACH query.

IMPORTANT: You must return the EXACT phenotype name from the list as it appears, including any IDs, numbers, or special characters (like "20002#1473#high cholesterol"). Do not modify, simplify, or clean up the phenotype names.

Think step by step for each query and select only the most relevant phenotype name. Please format your response following the response format and make sure it is parsable by JSON.

User queries:
{chr(10).join([f"{i+1}. {query}" for i, query in enumerate(queries)])}

Available phenotype names:
{chr(10).join(all_phenotypes)}

Response format:
{{
"thoughts": "your step-by-step thinking process for each query",
"phenotype_matches": [
    {{
        "query": "user query",
        "matched_phenotype_name": "matched phenotype name EXACTLY as it appears in the list, or NA if no relevant phenotype found"
    }}
]
}}
"""
        
        try:
            result = json_llm_call(prompt, model_name=model_name)
            matched_phenotypes_list = result.get('phenotype_matches', [])
            matched_phenotypes = {}
            for item in matched_phenotypes_list:
                if isinstance(item, dict) and 'query' in item and 'matched_phenotype_name' in item:
                    matched_phenotypes[item['query']] = item['matched_phenotype_name']
            return matched_phenotypes
        except Exception as e:
            print(f"Error matching phenotypes with LLM: {str(e)}")
            return {}

    def find_genes_for_phenotypes(matched_phenotypes: Dict[str, str], data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Find genes for multiple matched phenotypes with p-values"""
        results = {}
        
        for query, phenotype_name in matched_phenotypes.items():
            found_genes = []
            
            if phenotype_name and phenotype_name != "NA":
                # Look up genes for the matched phenotype
                for entry in data:
                    if isinstance(entry, dict) and entry.get("diseaseFromSource") == phenotype_name:
                        target_gene = entry.get("targetFromSourceId")
                        p_value_mantissa = entry.get("pValueMantissa")
                        p_value_exponent = entry.get("pValueExponent")
                        
                        if target_gene and p_value_mantissa is not None:
                            found_genes.append({
                                "gene": target_gene,
                                "pValueMantissa": p_value_mantissa,
                                "pValueExponent": p_value_exponent if p_value_exponent is not None else 0
                            })
            
            results[query] = found_genes
        
        return results

    try:
        # Load Gene Burden data
        data = load_json_data()
        if not data:
            return {
                "status": "error",
                "error_message": f"No Gene Burden data found in {json_path}",
                "raw_results": {},
                "final_results": {}
            }

        # Extract all phenotype names
        all_phenotypes = extract_all_phenotype_names(data)
        if not all_phenotypes:
            return {
                "status": "error",
                "error_message": "No phenotype names found in data",
                "raw_results": {},
                "final_results": {}
            }

        final_results = {}
        raw_results = {}
        
        # Match all phenotypes with LLM
        matched_phenotypes = match_phenotypes_with_llm(query_list, all_phenotypes)
        
        # Print matched phenotypes for quality check
        print("\n=== Phenotype Matching Results (Quality Check) ===")
        for query, phenotype in matched_phenotypes.items():
            print(f"Query: '{query}' → Matched: '{phenotype}'")
        print("===================================================\n")
        
        # Find genes for all matched phenotypes
        genes_by_query = find_genes_for_phenotypes(matched_phenotypes, data)
        
        for query in query_list:
            # Get genes for the current query
            gene_results = genes_by_query.get(query, [])
            
            # Extract gene symbols
            found_genes = [g["gene"] for g in gene_results]
            
            # Validate genes 
            valid_genes, invalid_genes = validate_genes(found_genes)
            
            # Store raw results
            raw_results[query] = {
                "matched_phenotype": matched_phenotypes.get(query, ""),
                "total_genes_found": len(found_genes),
                "valid_genes": len(valid_genes),
                "invalid_genes": len(invalid_genes)
            }
            
            # Create a mapping of valid genes to their p-values
            # Use a dictionary to handle duplicates - keep the most significant (smallest p-value) version
            valid_gene_pvalues = {}
            for i, gene in enumerate(found_genes):
                if gene in valid_genes:
                    mantissa = gene_results[i].get("pValueMantissa", 0)
                    exponent = gene_results[i].get("pValueExponent", 0)
                    
                    # Calculate full p-value (for comparison only)
                    full_pvalue = mantissa * (10 ** exponent) if exponent else mantissa
                    
                    # Keep the most significant (smallest) p-value for each gene
                    if gene not in valid_gene_pvalues or full_pvalue < valid_gene_pvalues[gene]["full_pvalue"]:
                        valid_gene_pvalues[gene] = {
                            "mantissa": mantissa,
                            "full_pvalue": full_pvalue
                        }
            
            # Assign raw scores based on pValueMantissa (higher mantissa means higher significance)
            raw_scores = []
            valid_genes_list = list(valid_gene_pvalues.keys())
            
            for gene in valid_genes_list:
                # Use mantissa directly as the raw score
                score = valid_gene_pvalues[gene]["mantissa"]
                raw_scores.append(score)
            
            # Normalize scores
            normalized_scores = normalize_scores(raw_scores, temperature=1.0)
            
            # Create final results with normalized scores
            final_results[query] = [
                {"gene": gene, "score": score} 
                for gene, score in zip(valid_genes_list, normalized_scores)
            ]
        
        return {
            "status": "success",
            "metadata": {
                "queries_processed": len(query_list),
                "phenotypes_matched": {
                    query: matched_phenotypes.get(query, "")
                    for query in query_list
                },
                "genes_found": {
                    query: len(final_results.get(query, []))
                    for query in query_list
                }
            },
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def intogen_search(
    query_list: List[str], 
    json_path: str = "resource/Open_target/intogen.json",
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Search and extract gene information from IntOGen database. IntOGen provides a framework to identify potential cancer driver genes using large-scale mutational data from sequenced tumor samples. By harmonising tumor sequencing data from the ICGC/TCGA Pan-Cancer Analysis of Whole Genomes (PCAWG) and other comprehensive efforts, IntOGen aims to provide a consensus assessment of cancer driver genes.
    
    Args:
        query_list: List of cancer types or related conditions to search for associated driver genes
        json_path: Path to IntOGen JSON file (default: "resource/Open_target/intogen.json")
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing cancer driver genes associated with each cancer type query and normalized relevance scores
    """
    import json
    import math
    
    def load_json_data() -> List[Dict[str, Any]]:
        """Load the IntOGen JSON data (line-delimited JSON format)"""
        data_list = []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                # Each line is a separate JSON object
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data_list.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            print(f"IntOGen file not found: {json_path}")
        except Exception as e:
            print(f"Could not parse {json_path} as JSON: {e}")
        return data_list

    def extract_all_disease_names(data: List[Dict[str, Any]]) -> List[str]:
        """Extract all unique disease names from the loaded JSON data"""
        disease_names = set()
        for entry in data:
            if isinstance(entry, dict) and "diseaseFromSource" in entry:
                disease_names.add(entry["diseaseFromSource"])
        return sorted(list(disease_names))

    def match_diseases_with_llm(queries: List[str], all_diseases: List[str]) -> Dict[str, str]:
        """Use LLM to find the most relevant disease names for multiple queries"""
        prompt = f"""
You are an assistant to cancer genomics experts. Given multiple user queries, your task is to identify the SINGLE most relevant cancer or disease name from the provided list for EACH query.

IMPORTANT: You must return the EXACT disease name from the list as it appears, including any IDs, numbers, or special characters. Do not modify, simplify, or clean up the disease names.

Think step by step for each query and select only the most relevant disease name. Please format your response following the response format and make sure it is parsable by JSON.

User queries:
{chr(10).join([f"{i+1}. {query}" for i, query in enumerate(queries)])}

Available disease names:
{chr(10).join(all_diseases)}

Response format:
{{
"thoughts": "your step-by-step thinking process for each query",
"disease_matches": [
    {{
        "query": "user query",
        "matched_disease_name": "matched disease name EXACTLY as it appears in the list, or NA if no relevant disease found"
    }}
]
}}
"""
        
        try:
            result = json_llm_call(prompt, model_name=model_name)
            matched_diseases_list = result.get('disease_matches', [])
            matched_diseases = {}
            for item in matched_diseases_list:
                if isinstance(item, dict) and 'query' in item and 'matched_disease_name' in item:
                    matched_diseases[item['query']] = item['matched_disease_name']
            return matched_diseases
        except Exception as e:
            print(f"Error matching diseases with LLM: {str(e)}")
            return {}

    def find_genes_for_diseases(matched_diseases: Dict[str, str], data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Find genes for multiple matched diseases with resource scores"""
        results = {}
        
        for query, disease_name in matched_diseases.items():
            found_genes = []
            
            if disease_name and disease_name != "NA":
                # Look up genes for the matched disease
                for entry in data:
                    if isinstance(entry, dict) and entry.get("diseaseFromSource") == disease_name:
                        target_gene = entry.get("targetFromSourceId")
                        resource_score = entry.get("resourceScore")
                        
                        if target_gene and resource_score is not None:
                            found_genes.append({
                                "gene": target_gene,
                                "resourceScore": resource_score
                            })
            
            results[query] = found_genes
        
        return results

    try:
        # Load IntOGen data
        data = load_json_data()
        if not data:
            return {
                "status": "error",
                "error_message": f"No IntOGen data found in {json_path}",
                "raw_results": {},
                "final_results": {}
            }

        # Extract all disease names
        all_diseases = extract_all_disease_names(data)
        if not all_diseases:
            return {
                "status": "error",
                "error_message": "No disease names found in data",
                "raw_results": {},
                "final_results": {}
            }

        final_results = {}
        raw_results = {}
        
        # Match all diseases with LLM
        matched_diseases = match_diseases_with_llm(query_list, all_diseases)
        
        # Print matched diseases for quality check
        print("\n=== Disease Matching Results (Quality Check) ===")
        for query, disease in matched_diseases.items():
            print(f"Query: '{query}' → Matched: '{disease}'")
        print("===================================================\n")
        
        # Find genes for all matched diseases
        genes_by_query = find_genes_for_diseases(matched_diseases, data)
        
        for query in query_list:
            # Get genes for the current query
            gene_results = genes_by_query.get(query, [])
            
            # Extract gene symbols
            found_genes = [g["gene"] for g in gene_results]
            
            # Validate genes 
            valid_genes, invalid_genes = validate_genes(found_genes)
            
            # Store raw results
            raw_results[query] = {
                "matched_disease": matched_diseases.get(query, ""),
                "total_genes_found": len(found_genes),
                "valid_genes": len(valid_genes),
                "invalid_genes": len(invalid_genes)
            }
            
            # Create a mapping of valid genes to their resource scores
            # Use a dictionary to handle duplicates - keep the highest resource score version
            valid_gene_scores = {}
            for i, gene in enumerate(found_genes):
                if gene in valid_genes:
                    resource_score = gene_results[i].get("resourceScore", 0)
                    
                    # For ties or duplicates, keep the highest resource score
                    if gene not in valid_gene_scores or resource_score > valid_gene_scores[gene]:
                        valid_gene_scores[gene] = resource_score
            
            # Assign raw scores based on -log(resourceScore)
            raw_scores = []
            valid_genes_list = list(valid_gene_scores.keys())
            
            for gene in valid_genes_list:
                # Apply -log transformation to resource score
                # Add a small epsilon to avoid log(0)
                epsilon = 1e-10
                score = -math.log(valid_gene_scores[gene] + epsilon) if valid_gene_scores[gene] > 0 else 0
                raw_scores.append(score)
            
            # Normalize scores
            normalized_scores = normalize_scores(raw_scores, temperature=1.0)
            
            # Create final results with normalized scores
            final_results[query] = [
                {"gene": gene, "score": score} 
                for gene, score in zip(valid_genes_list, normalized_scores)
            ]
        
        return {
            "status": "success",
            "metadata": {
                "queries_processed": len(query_list),
                "diseases_matched": {
                    query: matched_diseases.get(query, "")
                    for query in query_list
                },
                "genes_found": {
                    query: len(final_results.get(query, []))
                    for query in query_list
                }
            },
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def gocc_cellular_component_search(
    query_list: List[str], 
    json_directory: str = "resource/GOCC",
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Search and extract gene information from Gene Ontology Cellular Component (GOCC) database. The Gene Ontology (GO) Cellular Component ontology describes locations at the levels of subcellular structures and macromolecular complexes. This tool identifies genes associated with specific cellular components, organelles, and subcellular locations.
    
    Args:
        query_list: List of cellular component names, structures, or location identifiers to search for
        json_directory: Path to directory containing GOCC JSON files (default: "resource/GOCC")
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing genes associated with each cellular component query and normalized relevance scores
    """
    import json
    
    def load_json_files() -> List[Dict[str, Any]]:
        """Load all JSON files from the specified directory"""
        data_list = []
        try:
            for file_name in os.listdir(json_directory):
                if file_name.lower().endswith(".json"):
                    file_path = os.path.join(json_directory, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data_list.append(json.load(f))
                    except Exception as e:
                        print(f"Could not parse {file_path} as JSON: {e}")
        except FileNotFoundError:
            print(f"GOCC directory not found: {json_directory}")
        return data_list

    def extract_all_component_names(data_files: List[Dict[str, Any]]) -> List[str]:
        """Extract all cellular component names from the loaded JSON files"""
        component_names = []
        for data_file in data_files:
            if isinstance(data_file, dict):
                component_names.extend(list(data_file.keys()))
        return component_names

    def match_components_with_llm(queries: List[str], all_components: List[str]) -> Dict[str, str]:
        """Use LLM to find the most relevant cellular component names for multiple queries"""
        prompt = f"""
You are an assistant to biologists. Given multiple user queries, your task is to identify the SINGLE most relevant Gene Ontology Cellular Component (GOCC) name from the provided list for EACH query.

Think step by step for each query and select only the most relevant cellular component name. Please format your response following the response format and make sure it is parsable by JSON.

User queries:
{chr(10).join([f"{i+1}. {query}" for i, query in enumerate(queries)])}

Available cellular component names:
{chr(10).join(all_components)}

Response format:
{{
"thoughts": "your step-by-step thinking process for each query",
"component_matches": [
    {{
        "query": "user query",
        "matched_component_name": "matched cellular component name or NA if no relevant component found"
    }}
]
}}
"""
        
        try:
            result = json_llm_call(prompt, model_name=model_name)
            matched_components_list = result.get('component_matches', [])
            matched_components = {}
            for item in matched_components_list:
                if isinstance(item, dict) and 'query' in item and 'matched_component_name' in item:
                    matched_components[item['query']] = item['matched_component_name']
            return matched_components
        except Exception as e:
            print(f"Error matching components with LLM: {str(e)}")
            return {}

    def find_genes_for_components(matched_components: Dict[str, str], data_files: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Find genes for multiple matched cellular components"""
        results = {}
        
        for query, component_name in matched_components.items():
            found_genes = set()
            
            if component_name and component_name != "NA":
                # Look up genes for the matched component
                for data_file in data_files:
                    if isinstance(data_file, dict) and component_name in data_file:
                        entry = data_file[component_name]
                        if isinstance(entry, dict):
                            genes = entry.get("geneSymbols", [])
                            if isinstance(genes, list):
                                found_genes.update(genes)
            
            results[query] = sorted(found_genes)
        
        return results

    try:
        # Load GOCC data files
        data_files = load_json_files()
        if not data_files:
            return {
                "status": "error",
                "error_message": f"No GOCC JSON files found in {json_directory}",
                "raw_results": {},
                "final_results": {}
            }

        # Extract all component names
        all_components = extract_all_component_names(data_files)
        if not all_components:
            return {
                "status": "error",
                "error_message": "No cellular component names found in JSON files",
                "raw_results": {},
                "final_results": {}
            }

        final_results = {}
        raw_results = {}
        
        # Match all components with LLM
        matched_components = match_components_with_llm(query_list, all_components)
        
        # Print matched components for quality check
        print("\n=== Component Matching Results (Quality Check) ===")
        for query, component in matched_components.items():
            print(f"Query: '{query}' → Matched: '{component}'")
        print("===================================================\n")
        
        # Find genes for all matched components
        genes_by_query = find_genes_for_components(matched_components, data_files)
        
        for query in query_list:
            # Get genes for the current query
            found_genes = genes_by_query.get(query, [])
            
            # Validate genes 
            valid_genes, invalid_genes = validate_genes(found_genes)
            
            # Store raw results
            raw_results[query] = {
                "matched_component": matched_components.get(query, ""),
                "total_genes_found": len(found_genes),
                "valid_genes": len(valid_genes),
                "invalid_genes": len(invalid_genes)
            }
            
            # Assign initial raw scores of 1.0 to each gene
            raw_scores = [1.0] * len(valid_genes)
            
            # Normalize scores
            normalized_scores = normalize_scores(raw_scores, temperature=1.0)
            
            # Create final results with normalized scores
            final_results[query] = [
                {"gene": gene, "score": score} 
                for gene, score in zip(valid_genes, normalized_scores)
            ]
        
        return {
            "status": "success",
            "metadata": {
                "queries_processed": len(query_list),
                "components_matched": {
                    query: matched_components.get(query, "")
                    for query in query_list
                },
                "genes_found": {
                    query: len(final_results.get(query, []))
                    for query in query_list
                }
            },
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def clinvar_search(
    query_list: List[str], 
    json_path: str = "resource/Open_target/eva.json",
    max_candidates: int = 100,
    similarity_threshold: int = 50,
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Search and extract gene information from ClinVar database. ClinVar is a NIH public archive of reports of the relationships among human variations and phenotypes, with supporting evidence. This tool extracts gene information based on disease associations from the ClinVar database, using clinical significance and confidence levels to calculate evidence scores.
    
    Args:
        query_list: List of disease names or related conditions to search for associated genes
        json_path: Path to ClinVar JSON file (default: "resource/Open_target/eva.json")
        max_candidates: Maximum number of disease candidates to send to LLM (default: 100)
        similarity_threshold: Minimum similarity score (0-100) for fuzzy matching (default: 50)
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing genes associated with each disease query and normalized relevance scores based on clinical evidence
    """
    import json
    import time
    from rapidfuzz import process, fuzz
    
    # Cache for Ensembl ID to gene name mapping
    ensembl_id_cache = {}
    
    # Clinical significance score mapping
    clinical_significance_scores = {
        "association not found": 0.0,
        "benign": 0.0,
        "not provided": 0.0,
        "likely benign": 0.0,
        "conflicting data from submitters": 0.3,
        "conflicting interpretations of pathogenicity": 0.3,
        "low penetrance": 0.3,
        "other": 0.3,
        "uncertain risk allele": 0.3,
        "uncertain significance": 0.3,
        "established risk allele": 0.5,
        "risk factor": 0.5,
        "affects": 0.5,
        "likely pathogenic": 0.7,
        "association": 0.9,
        "confers sensitivity": 0.9,
        "drug response": 0.9,
        "protective": 0.9,
        "pathogenic": 0.9
    }
    
    # Confidence score modifiers
    confidence_score_modifiers = {
        "no assertion provided": 0.0,
        "no assertion criteria provided": 0.0,
        "no assertion for the individual variant": 0.0,
        "criteria provided, single submitter": 0.02,
        "criteria provided, conflicting interpretations": 0.02,
        "criteria provided, conflicting classifications": 0.02,
        "criteria provided, multiple submitters, no conflicts": 0.05,
        "reviewed by expert panel": 0.07,
        "practice guideline": 0.1
    }
    
    def load_json_data() -> List[Dict[str, Any]]:
        """Load the ClinVar JSON data (line-delimited JSON format)"""
        data_list = []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                # Each line is a separate JSON object
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data_list.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            print(f"ClinVar file not found: {json_path}")
        except Exception as e:
            print(f"Could not parse {json_path} as JSON: {e}")
        return data_list

    def extract_all_disease_names(data: List[Dict[str, Any]]) -> List[str]:
        """Extract all unique disease names from the loaded JSON data"""
        disease_names = set()
        for entry in data:
            if isinstance(entry, dict) and "diseaseFromSource" in entry:
                disease_names.add(entry["diseaseFromSource"])
        return sorted(list(disease_names))

    def similarity_search(query: str, all_diseases: List[str]) -> List[str]:
        """Find most similar disease names using fuzzy matching"""
        matches = process.extract(
            query, 
            all_diseases, 
            scorer=fuzz.token_sort_ratio, 
            limit=max_candidates,
            score_cutoff=similarity_threshold,
            processor=lambda s: s.lower()
        )
        return [match[0] for match in matches]

    def match_diseases_with_llm(queries: List[str], all_diseases: List[str]) -> Dict[str, str]:
        """Use LLM to find the most relevant disease names for multiple queries"""
        matched_diseases = {}
        
        for query in queries:
            # First use similarity search to get candidate matches
            candidates = similarity_search(query, all_diseases)
            print(f"Found {len(candidates)} candidate matches for '{query}'")
            
            if not candidates:
                matched_diseases[query] = ""
                continue
                
            prompt = f"""
You are an assistant to genomics experts. Given a user query, your task is to identify the SINGLE most relevant disease name from the provided list.

IMPORTANT: You must return the EXACT disease name from the list as it appears, including any IDs, numbers, or special characters. Do not modify, simplify, or clean up the disease names.

Think step by step and select only the most relevant disease name. Please format your response following the response format and make sure it is parsable by JSON.

User query:
{query}

Available disease names:
{chr(10).join(candidates)}

Response format:
{{
"thoughts": "your step-by-step thinking process",
"disease_matches": [
    {{
        "query": "{query}",
        "matched_disease_name": "matched disease name EXACTLY as it appears in the list, or NA if no relevant disease found"
    }}
]
}}
"""
            
            try:
                result = json_llm_call(prompt, model_name=model_name)
                matched_disease_list = result.get('disease_matches', [])
                if matched_disease_list and isinstance(matched_disease_list, list) and len(matched_disease_list) > 0:
                    matched_disease = matched_disease_list[0].get('matched_disease_name', "")
                    matched_diseases[query] = matched_disease if matched_disease != "NA" else ""
                else:
                    matched_diseases[query] = ""
            except Exception as e:
                print(f"Error matching disease '{query}' with LLM: {str(e)}")
                matched_diseases[query] = ""
                
        return matched_diseases

    def ensembl_id_to_gene_symbol(ensembl_id: str) -> str:
        """Convert Ensembl ID to gene symbol using Ensembl API"""
        # Check cache first
        if ensembl_id in ensembl_id_cache:
            return ensembl_id_cache[ensembl_id]
        
        try:
            # Use Ensembl REST API to fetch gene name
            url = f"https://rest.ensembl.org/lookup/id/{ensembl_id}?content-type=application/json"
            response = requests.get(url, headers={"Content-Type": "application/json"}, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                gene_symbol = data.get("display_name", ensembl_id)
                ensembl_id_cache[ensembl_id] = gene_symbol
                return gene_symbol
            
            # If we can't find the gene name, return the original ID
            ensembl_id_cache[ensembl_id] = ensembl_id
            return ensembl_id
            
        except Exception as e:
            print(f"Error converting Ensembl ID {ensembl_id} to gene symbol: {str(e)}")
            ensembl_id_cache[ensembl_id] = ensembl_id
            return ensembl_id

    def calculate_evidence_score(entry: Dict[str, Any]) -> float:
        """Calculate evidence score based on clinical significance and confidence"""
        # Get clinical significance score
        clinical_significance = "not provided"  # Default
        if "clinicalSignificances" in entry and entry["clinicalSignificances"]:
            clinical_significance = entry["clinicalSignificances"][0]
        
        base_score = clinical_significance_scores.get(clinical_significance, 0.0)
        
        # Get confidence modifier
        confidence = entry.get("confidence", "no assertion provided")
        modifier = confidence_score_modifiers.get(confidence, 0.0)
        
        # Combine base score and modifier
        return min(1.0, base_score + modifier)  # Cap at 1.0

    def find_genes_for_diseases(matched_diseases: Dict[str, str], data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Find genes for multiple matched diseases with evidence scores"""
        results = {}
        
        for query, disease_name in matched_diseases.items():
            found_genes = []
            
            if disease_name and disease_name != "NA":
                # Look up genes for the matched disease
                for entry in data:
                    if isinstance(entry, dict) and entry.get("diseaseFromSource") == disease_name:
                        ensembl_id = entry.get("targetFromSourceId")
                        
                        if ensembl_id:
                            # Calculate evidence score based on clinical significance and confidence
                            evidence_score = calculate_evidence_score(entry)
                            
                            found_genes.append({
                                "ensembl_id": ensembl_id,
                                "resourceScore": evidence_score
                            })
            
            results[query] = found_genes
        
        return results

    try:
        # Load ClinVar data
        data = load_json_data()
        if not data:
            return {
                "status": "error",
                "error_message": f"No ClinVar data found in {json_path}",
                "raw_results": {},
                "final_results": {}
            }

        # Extract all disease names
        all_diseases = extract_all_disease_names(data)
        if not all_diseases:
            return {
                "status": "error",
                "error_message": "No disease names found in data",
                "raw_results": {},
                "final_results": {}
            }

        final_results = {}
        raw_results = {}
        
        # Match all diseases with LLM
        matched_diseases = match_diseases_with_llm(query_list, all_diseases)
        
        # Print matched diseases for quality check
        print("\n=== Disease Matching Results (Quality Check) ===")
        for query, disease in matched_diseases.items():
            print(f"Query: '{query}' → Matched: '{disease}'")
        print("===================================================\n")
        
        # Find genes for all matched diseases
        genes_by_query = find_genes_for_diseases(matched_diseases, data)
        
        for query in query_list:
            # Get genes for the current query
            gene_results = genes_by_query.get(query, [])
            
            # Extract Ensembl IDs and scores
            ensembl_ids = [g["ensembl_id"] for g in gene_results]
            raw_scores = [g["resourceScore"] for g in gene_results]
            
            # Map of Ensembl IDs to their scores (keeping the highest score if duplicates)
            gene_score_map = {}
            for ensembl_id, score in zip(ensembl_ids, raw_scores):
                if ensembl_id in gene_score_map:
                    gene_score_map[ensembl_id] = max(gene_score_map[ensembl_id], score)
                else:
                    gene_score_map[ensembl_id] = score
            
            # Convert Ensembl IDs to gene symbols with rate limiting
            gene_symbols = []
            gene_scores = []
            for ensembl_id, score in gene_score_map.items():
                gene_symbol = ensembl_id_to_gene_symbol(ensembl_id)
                gene_symbols.append(gene_symbol)
                gene_scores.append(score)
                time.sleep(0.1)  # Add a small delay to avoid rate limiting
            
            # Validate genes 
            valid_genes, invalid_genes = validate_genes(gene_symbols)
            
            # Store raw results
            raw_results[query] = {
                "matched_disease": matched_diseases.get(query, ""),
                "total_ensembl_ids_found": len(ensembl_ids),
                "total_gene_symbols_converted": len(gene_symbols),
                "valid_genes": len(valid_genes),
                "invalid_genes": len(invalid_genes)
            }
            
            # Create a map of valid genes to their scores (keeping the highest score if duplicates)
            valid_gene_score_map = {}
            for gene, score in zip(gene_symbols, gene_scores):
                if gene in valid_genes:
                    if gene in valid_gene_score_map:
                        valid_gene_score_map[gene] = max(valid_gene_score_map[gene], score)
                    else:
                        valid_gene_score_map[gene] = score
            
            # Extract unique valid genes and their scores
            unique_valid_genes = list(valid_gene_score_map.keys())
            unique_raw_scores = [valid_gene_score_map[gene] for gene in unique_valid_genes]
            
            # Normalize scores
            normalized_scores = normalize_scores(unique_raw_scores, temperature=1.0)
            
            # Create final results with normalized scores
            final_results[query] = [
                {"gene": gene, "score": score} 
                for gene, score in zip(unique_valid_genes, normalized_scores)
            ]
        
        return {
            "status": "success",
            "metadata": {
                "queries_processed": len(query_list),
                "diseases_matched": {
                    query: matched_diseases.get(query, "")
                    for query in query_list
                },
                "genes_found": {
                    query: len(final_results.get(query, []))
                    for query in query_list
                },
                "api_cache_size": len(ensembl_id_cache),
                "max_candidates": max_candidates,
                "similarity_threshold": similarity_threshold
            },
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


@tool
def uniprot_variants_search(
    query_list: List[str], 
    json_path: str = "resource/Open_target/uniprot_variants.json",
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Search and extract gene information from UniProt Variants database. UniProt provides comprehensive information about proteins and their variants associated with diseases. This tool extracts gene information based on disease associations from the UniProt database, converting UniProt IDs to gene symbols.
    
    Args:
        query_list: List of disease names or related conditions to search for associated genes
        json_path: Path to UniProt variants JSON file (default: "resource/Open_target/uniprot_variants.json")
        model_name: Model name for LLM analysis, defaults to "gemini-2.5-pro"
    
    Returns:
        Dictionary containing genes associated with each disease query and normalized relevance scores
    """
    import json
    import time
    
    # Cache for UniProt ID to gene name mapping
    uniprot_id_cache = {}
    
    def load_json_data() -> List[Dict[str, Any]]:
        """Load the UniProt variants JSON data (line-delimited JSON format)"""
        data_list = []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                # Each line is a separate JSON object
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data_list.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            print(f"UniProt variants file not found: {json_path}")
        except Exception as e:
            print(f"Could not parse {json_path} as JSON: {e}")
        return data_list

    def extract_all_disease_names(data: List[Dict[str, Any]]) -> List[str]:
        """Extract all unique disease names from the loaded JSON data"""
        disease_names = set()
        for entry in data:
            if isinstance(entry, dict) and "diseaseFromSource" in entry:
                disease_names.add(entry["diseaseFromSource"])
        return sorted(list(disease_names))

    def match_diseases_with_llm(queries: List[str], all_diseases: List[str]) -> Dict[str, str]:
        """Use LLM to find the most relevant disease names for multiple queries"""
        prompt = f"""
You are an assistant to genomics experts. Given multiple user queries, your task is to identify the SINGLE most relevant disease name from the provided list for EACH query.

IMPORTANT: You must return the EXACT disease name from the list as it appears, including any IDs, numbers, or special characters. Do not modify, simplify, or clean up the disease names.

Think step by step for each query and select only the most relevant disease name. Please format your response following the response format and make sure it is parsable by JSON.

User queries:
{chr(10).join([f"{i+1}. {query}" for i, query in enumerate(queries)])}

Available disease names:
{chr(10).join(all_diseases)}

Response format:
{{
"thoughts": "your step-by-step thinking process for each query",
"disease_matches": [
    {{
        "query": "user query",
        "matched_disease_name": "matched disease name EXACTLY as it appears in the list, or NA if no relevant disease found"
    }}
]
}}
"""
        
        try:
            result = json_llm_call(prompt, model_name=model_name)
            matched_diseases_list = result.get('disease_matches', [])
            matched_diseases = {}
            for item in matched_diseases_list:
                if isinstance(item, dict) and 'query' in item and 'matched_disease_name' in item:
                    matched_diseases[item['query']] = item['matched_disease_name']
            return matched_diseases
        except Exception as e:
            print(f"Error matching diseases with LLM: {str(e)}")
            return {}

    def uniprot_id_to_gene_symbol(uniprot_id: str) -> str:
        """Convert UniProt ID to gene symbol using UniProt API"""
        # Check cache first
        if uniprot_id in uniprot_id_cache:
            return uniprot_id_cache[uniprot_id]
        
        try:
            # Use UniProt REST API to fetch gene name
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                # Extract gene name from the response
                gene_names = data.get("genes", [])
                if gene_names and "geneName" in gene_names[0]:
                    gene_symbol = gene_names[0]["geneName"].get("value", uniprot_id)
                    uniprot_id_cache[uniprot_id] = gene_symbol
                    return gene_symbol
            
            # If we can't find the gene name, return the original ID
            uniprot_id_cache[uniprot_id] = uniprot_id
            return uniprot_id
            
        except Exception as e:
            print(f"Error converting UniProt ID {uniprot_id} to gene symbol: {str(e)}")
            uniprot_id_cache[uniprot_id] = uniprot_id
            return uniprot_id

    def find_genes_for_diseases(matched_diseases: Dict[str, str], data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Find UniProt IDs for multiple matched diseases"""
        results = {}
        
        for query, disease_name in matched_diseases.items():
            found_uniprot_ids = []
            
            if disease_name and disease_name != "NA":
                # Look up UniProt IDs for the matched disease
                for entry in data:
                    if isinstance(entry, dict) and entry.get("diseaseFromSource") == disease_name:
                        uniprot_id = entry.get("targetFromSourceId")
                        if uniprot_id:
                            found_uniprot_ids.append(uniprot_id)
            
            results[query] = found_uniprot_ids
        
        return results

    try:
        # Load UniProt variants data
        data = load_json_data()
        if not data:
            return {
                "status": "error",
                "error_message": f"No UniProt variants data found in {json_path}",
                "raw_results": {},
                "final_results": {}
            }

        # Extract all disease names
        all_diseases = extract_all_disease_names(data)
        if not all_diseases:
            return {
                "status": "error",
                "error_message": "No disease names found in data",
                "raw_results": {},
                "final_results": {}
            }

        final_results = {}
        raw_results = {}
        
        # Match all diseases with LLM
        matched_diseases = match_diseases_with_llm(query_list, all_diseases)
        
        # Print matched diseases for quality check
        print("\n=== Disease Matching Results (Quality Check) ===")
        for query, disease in matched_diseases.items():
            print(f"Query: '{query}' → Matched: '{disease}'")
        print("===================================================\n")
        
        # Find UniProt IDs for all matched diseases
        uniprot_ids_by_query = find_genes_for_diseases(matched_diseases, data)
        
        for query in query_list:
            # Get UniProt IDs for the current query
            uniprot_ids = uniprot_ids_by_query.get(query, [])
            
            # Convert UniProt IDs to gene symbols with rate limiting
            gene_symbols = []
            for uniprot_id in uniprot_ids:
                gene_symbol = uniprot_id_to_gene_symbol(uniprot_id)
                gene_symbols.append(gene_symbol)
                time.sleep(0.1)  # Add a small delay to avoid rate limiting
            
            # Validate genes 
            valid_genes, invalid_genes = validate_genes(gene_symbols)
            
            # Store raw results
            raw_results[query] = {
                "matched_disease": matched_diseases.get(query, ""),
                "total_uniprot_ids_found": len(uniprot_ids),
                "total_gene_symbols_converted": len(gene_symbols),
                "valid_genes": len(valid_genes),
                "invalid_genes": len(invalid_genes)
            }
            
            # Create a set of unique valid genes (to remove duplicates)
            unique_valid_genes = list(set(valid_genes))
            
            # Since all scores are conceptually equal (all are from UniProt variants), assign equal raw scores
            raw_scores = [1.0] * len(unique_valid_genes)
            
            # Normalize scores 
            normalized_scores = normalize_scores(raw_scores, temperature=1.0)
            
            # Create final results with normalized scores
            final_results[query] = [
                {"gene": gene, "score": score} 
                for gene, score in zip(unique_valid_genes, normalized_scores)
            ]
        
        return {
            "status": "success",
            "metadata": {
                "queries_processed": len(query_list),
                "diseases_matched": {
                    query: matched_diseases.get(query, "")
                    for query in query_list
                },
                "genes_found": {
                    query: len(final_results.get(query, []))
                    for query in query_list
                },
                "api_cache_size": len(uniprot_id_cache)
            },
            "raw_results": raw_results,
            "final_results": final_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_results": {},
            "final_results": {}
        }


if __name__ == "__main__":
    # Example usage for all tools
    
    # Test gene validation
    test_genes = ['TP53', 'BRCA1', 'NOTAREALGENE', 'BRAF']
    valid, invalid = validate_genes(test_genes)
    print("Valid genes:", valid)
    print("Invalid genes:", invalid)
    
    # Test original tools
    print("\n--- Testing KEGG pathway search ---")
    kegg_result = kegg_pathway_search(['apoptosis'])
    print(f"KEGG result status: {kegg_result['status']}")
    
    print("\n--- Testing disease gene search ---")
    disease_result = disease_gene_search(['breast cancer'])
    print(f"Disease result status: {disease_result['status']}")
    
    print("\n--- Testing STRING database search ---")
    string_result = string_database_search(['TP53', 'BRCA1'])
    print(f"STRING result status: {string_result['status']}")
    
    print("\n--- Testing GO terms search ---") 
    go_result = go_terms_search(['DNA repair', 'cell cycle'])
    print(f"GO result status: {go_result['status']}")
    
    print("\n--- Testing screen data analysis ---")
    # Note: This would need an actual CSV file to work properly
    print("Screen data analysis requires a CSV file - skipping actual test")
    
    print("\n--- Testing TCGA survival analysis ---")
    tcga_result = tcga_survival_analysis(['breast cancer'])
    print(f"TCGA result status: {tcga_result['status']}")
    
    print("\n--- Testing drug gene network search ---")
    drug_network_result = drug_gene_network_search(['aspirin', 'metformin'])
    print(f"Drug network result status: {drug_network_result['status']}")
    
    print("\n--- Testing PubChem drug gene search ---")
    pubchem_result = pubchem_drug_gene_search(['aspirin'])
    print(f"PubChem result status: {pubchem_result['status']}")
    
    # Test newly added tools
    print("\n--- Testing HPO phenotype search ---")
    hpo_result = hpo_phenotype_search(['ataxia', 'seizures'])
    print(f"HPO result status: {hpo_result['status']}")
    
    print("\n--- Testing OMIM disease search ---")
    omim_result = omim_disease_search(['cystic fibrosis'])
    print(f"OMIM result status: {omim_result['status']}")
    
    print("\n--- Testing Orphanet rare disease search ---")
    orphanet_result = orphanet_rare_disease_search(['huntington disease'])
    print(f"Orphanet result status: {orphanet_result['status']}")
    
    print("\n--- Testing COXPres coexpression search ---")
    coxpres_result = coxpres_coexpression_search(['TP53', 'BRCA1'])
    print(f"COXPres result status: {coxpres_result['status']}")
    
    print("\n--- Testing Ensembl paralog search ---")
    ensembl_result = ensembl_paralog_search(['TP53', 'BRCA1'])
    print(f"Ensembl result status: {ensembl_result['status']}")
    
    print("\n--- Testing GSEA hallmark search ---")
    gsea_result = gsea_hallmark_search(['inflammatory response', 'glycolysis'])
    print(f"GSEA result status: {gsea_result['status']}")
    
    print("\n--- Testing WikiPathways search ---")
    wiki_result = wikipathways_search(['insulin signaling', 'apoptosis'])
    print(f"WikiPathways result status: {wiki_result['status']}")
    
    print("\n--- Testing Reactome pathway search ---")
    reactome_result = reactome_pathway_search(['insulin signaling', 'apoptosis'])
    print(f"Reactome result status: {reactome_result['status']}")
    
    print("\n--- Testing Cancer Biomarkers search ---")
    cancer_biomarkers_result = cancer_biomarkers_search(['breast cancer'])
    print(f"Cancer Biomarkers result status: {cancer_biomarkers_result['status']}")
    
    print("\n--- Testing ClinGen search ---")
    clingen_result = clingen_search(['cystic fibrosis'])
    print(f"ClinGen result status: {clingen_result['status']}")
    
    print("\n--- Testing Gene2Phenotype search ---")
    gene2phenotype_result = gene2phenotype_search(['Joubert syndrome', 'Kniest dysplasia'])
    print(f"Gene2Phenotype result status: {gene2phenotype_result['status']}")
    
    print("\n--- Testing Gene Burden search ---")
    gene_burden_result = gene_burden_search(['Cholesterol', 'Endometrial cancer'])
    print(f"Gene Burden result status: {gene_burden_result['status']}")
    
    print("\n--- Testing IntOGen search ---")
    intogen_result = intogen_search(['Breast cancer', 'Glioblastoma'])
    print(f"IntOGen result status: {intogen_result['status']}")
    
    print("\n--- Testing GOCC Cellular Component search ---")
    gocc_result = gocc_cellular_component_search(['nuclear membrane', 'ribosome'])
    print(f"GOCC Cellular Component result status: {gocc_result['status']}")
    
    print("\n--- Testing ClinVar search ---")
    clinvar_result = clinvar_search(['BRCA1 mutation', 'Cystic fibrosis'])
    print(f"ClinVar result status: {clinvar_result['status']}")
    
    print("\n--- Testing UniProt Variants search ---")
    uniprot_variants_result = uniprot_variants_search(['Methylglutaryl-CoA lyase deficiency', '3-ketothiolase deficiency'])
    print(f"UniProt Variants result status: {uniprot_variants_result['status']}")
    
    print("\n=== All tool tests completed ===")
    print("Note: Some tests may show 'error' status due to missing resources or API limitations, which is expected.")