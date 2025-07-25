import os, json, pickle, pandas as pd, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import AzureOpenAIEmbeddings
from tqdm.auto import tqdm
from typing import List, Dict, Union, Set, Optional, Any
import requests
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Seq import Seq
from Bio import Entrez
# Removed biomni dependency to avoid environment variable loading side effects
import traceback
import time
from smolagents import tool, OpenAIServerModel


OPENROUTER_API_KEY_STRING = "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"

# Use absolute path for schema database
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_STELLA_DIR = os.path.dirname(SCRIPT_DIR)
SCHEMA_DB_PATH = os.path.join(AGENTS_STELLA_DIR, "resource", "schema_db")

def parse_hpo_obo(obo_file_path: str) -> Dict[str, str]:
    """
    Simple HPO OBO file parser to extract term IDs and names.
    
    Args:
        obo_file_path (str): Path to the HPO OBO file
        
    Returns:
        Dict[str, str]: Dictionary mapping HPO IDs to names
    """
    hpo_dict = {}
    
    # Handle relative paths by making them absolute relative to script location
    if not os.path.isabs(obo_file_path):
        obo_file_path = os.path.join(SCRIPT_DIR, obo_file_path)
    
    try:
        if not os.path.exists(obo_file_path):
            print(f"Warning: HPO OBO file not found at {obo_file_path}")
            return hpo_dict
            
        with open(obo_file_path, 'r', encoding='utf-8') as f:
            current_term = {}
            in_term_block = False
            
            for line in f:
                line = line.strip()
                
                if line == '[Term]':
                    in_term_block = True
                    current_term = {}
                elif line == '' and in_term_block:
                    # End of term block
                    if 'id' in current_term and 'name' in current_term:
                        hpo_dict[current_term['id']] = current_term['name']
                    in_term_block = False
                elif in_term_block and ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'id':
                        current_term['id'] = value
                    elif key == 'name':
                        current_term['name'] = value
            
            # Handle last term if file doesn't end with empty line
            if in_term_block and 'id' in current_term and 'name' in current_term:
                hpo_dict[current_term['id']] = current_term['name']
                
    except Exception as e:
        print(f"Error parsing HPO OBO file: {e}")
        
    return hpo_dict

gemini_model = OpenAIServerModel(
    model_id="google/gemini-2.5-pro",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY_STRING,
    temperature=0.1,  # Lower temperature for more consistent analysis
)

def _query_gemini_for_api(prompt, schema, system_template, model=None):
    """
    Helper function to query Gemini for generating API calls based on natural language prompts.
    
    Args:
    prompt (str): Natural language query to process
    schema (dict): API schema to include in the system prompt
    system_template (str): Template string for the system prompt (should have {schema} placeholder)
    model: Gemini model instance to use (defaults to global gemini_model)
    
    Returns:
        
    dict: Dictionary with 'success', 'data' (if successful), 'error' (if failed), and optional 'raw_response'
    """
    # Use global gemini_model if none provided
    model = gemini_model
    
    try:
        if schema is not None:
            # Format the system prompt with the schema
            schema_json = json.dumps(schema, indent=2)
            system_prompt = system_template.format(schema=schema_json)
        else:
            system_prompt = system_template
        
        # Combine system prompt and user prompt for Gemini
        full_prompt = f"{system_prompt}\n\nUser query: {prompt}"
        
        # Create messages in the correct format for OpenAIServerModel
        messages = [{"role": "user", "content": full_prompt}]
        response = model(messages)
        
        # Extract content from ChatMessage response
        if hasattr(response, 'content'):
            gemini_text = response.content.strip()
        else:
            gemini_text = str(response).strip()
        
        # Find JSON boundaries (in case Gemini adds explanations)
        json_start = gemini_text.find('{')
        json_end = gemini_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_text = gemini_text[json_start:json_end]
            result = json.loads(json_text)
        else:
            # If no JSON found, try the whole response
            result = json.loads(gemini_text)
        
        return {
            "success": True,
            "data": result,
            "raw_response": gemini_text
        }
            
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return {
            "success": False,
            "error": f"Failed to parse Gemini's response: {str(e)}",
            "raw_response": gemini_text if 'gemini_text' in locals() else "No content found"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error querying Gemini: {str(e)}"
        }

# Function to map HPO terms to names
def get_hpo_names(hpo_terms: List[str], ) -> List[str]:
    """
    Retrieve the names of given HPO terms.

    Args:
        hpo_terms (List[str]): A list of HPO terms (e.g., ['HP:0001250']).

    Returns:
        
        List[str]: A list of corresponding HPO term names.
    """
    hp_dict = parse_hpo_obo(os.path.join(AGENTS_STELLA_DIR, 'resource', 'hp.obo'))

    hpo_names = []
    for term in hpo_terms:
        name = hp_dict.get(term, f"Unknown term: {term}")
        hpo_names.append(name)
    return hpo_names


def _query_rest_api(endpoint, method="GET", params=None, headers=None, json_data=None, description=None):
    """
    General helper function to query REST APIs with consistent error handling.
    
    Args:
    endpoint (str): Full URL endpoint to query
    method (str): HTTP method ("GET" or "POST")
    params (dict, optional): Query parameters to include in the URL
    headers (dict, optional): HTTP headers for the request
    json_data (dict, optional): JSON data for POST requests
    description (str, optional): Description of this query for error messages
    
    Returns:
        
    dict: Dictionary containing the result or error information
    """
    # Set default headers if not provided
    if headers is None:
        headers = {"Accept": "application/json"}
    
    # Set default description if not provided
    if description is None:
        description = f"{method} request to {endpoint}"

    url_error = None
        
    try:
        # Make the API request
        if method.upper() == "GET":
            response = requests.get(endpoint, params=params, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(endpoint, params=params, headers=headers, json=json_data)
        else:
            return {"error": f"Unsupported HTTP method: {method}"}
        
        url_error = str(response.text)
        response.raise_for_status()
        
        # Try to parse JSON response
        try:
            result = response.json()
        except ValueError:
            # Return raw text if not JSON
            result = {"raw_text": response.text}
        
        return {
            "success": True,
            "query_info": {
                "endpoint": endpoint,
                "method": method,
                "description": description
            },
            "result": result
        }
        
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        response_text = ""
        
        # Try to get more detailed error info from response
        if hasattr(e, 'response') and e.response:
            try:
                error_json = e.response.json()
                if 'messages' in error_json:
                    error_msg = "; ".join(error_json['messages'])
                elif 'message' in error_json:
                    error_msg = error_json['message']
                elif 'error' in error_json:
                    error_msg = error_json['error']
                elif 'detail' in error_json:
                    error_msg = error_json['detail']
            except:
                response_text = e.response.text
        
        return {
            "success": False,
            "error": f"API error: {error_msg}",
            "query_info": {
                "endpoint": endpoint,
                "method": method,
                "description": description
            },
            "response_url_error": url_error,
            "response_text": response_text
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "query_info": {
                "endpoint": endpoint,
                "method": method,
                "description": description
            }
        }
    
def _query_ncbi_database(
    database: str,
    search_term: str,
    result_formatter = None,
    max_results: int = 3,
) -> Dict[str, Any]:
    """
    Core function to query NCBI databases using Claude for query interpretation and NCBI eutils.
    
    Args:
    database (str): NCBI database to query (e.g., "clinvar", "gds", "geoprofiles")
    result_formatter (callable): Function to format results from the database
    api_key (str): Anthropic API key. If None, will look for ANTHROPIC_API_KEY environment variable
    model (str): Anthropic model to use
    max_results (int): Maximum number of results to return
    verbose (bool): Whether to return verbose results
    
    Returns:
        
    dict: Dictionary containing both the structured query and the results
    """
    
    # Query NCBI API using the structured search term
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esearch_params = {
        "db": database,
        "term": search_term,
        "retmode": "json",
        "retmax": 100,
        "usehistory": "y"  # Use history server to store results
    }
    
    # Get IDs of matching entries
    search_response = _query_rest_api(
        endpoint=esearch_url,
        method="GET",
        params=esearch_params,
        description="NCBI ESearch API query"
    )

    if not search_response["success"]:
        return search_response
    
    search_data = search_response["result"]
    
    # If we have results, fetch the details
    if "esearchresult" in search_data and int(search_data["esearchresult"]["count"]) > 0:
        # Extract WebEnv and query_key from the search results
        webenv = search_data["esearchresult"].get("webenv", "")
        query_key = search_data["esearchresult"].get("querykey", "")
        
        # Use WebEnv and query_key if available
        if webenv and query_key:
            # Get details using eSummary
            esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            esummary_params = {
                "db": database,
                "query_key": query_key,
                "WebEnv": webenv,
                "retmode": "json",
                "retmax": max_results
            }
            
            details_response = _query_rest_api(
                endpoint=esummary_url,
                method="GET",
                params=esummary_params,
                description="NCBI ESummary API query"
            )

            if not details_response["success"]:
                return details_response
            
            results = details_response["result"]
        
        else:
            # Fall back to direct ID fetch
            id_list = search_data["esearchresult"]["idlist"][:max_results]
            
            # Get details for each ID
            esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            esummary_params = {
                "db": database,
                "id": ",".join(id_list),
                "retmode": "json"
            }
            
            details_response = _query_rest_api(
                endpoint=esummary_url,
                method="GET",
                params=esummary_params,
                description="NCBI ESummary API query"
            )
            
            if not details_response["success"]:
                return details_response
            
            results = details_response["result"]
        
        # Format results using the provided formatter
        if result_formatter:
            formatted_results = result_formatter(results)
        else:
            formatted_results = results
        
        # Return the combined information
        return {
            "database": database,
            "query_interpretation": search_term,
            "total_results": int(search_data["esearchresult"]["count"]),
            "formatted_results": formatted_results
        }
    else:
        return {
            "database": database,
            "query_interpretation": search_term,
            "total_results": 0,
            "formatted_results": []
        }

def _format_query_results(result, options=None):
    """
    A general-purpose formatter for query function results to reduce output size.
    
    Args:
    result (dict): The original API response dictionary
    options (dict, optional): Formatting options including:
        - max_items (int): Maximum number of items to include in lists (default: 5)
        - max_depth (int): Maximum depth to traverse in nested dictionaries (default: 2)
        - include_keys (list): Only include these top-level keys (overrides exclude_keys)
        - exclude_keys (list): Exclude these keys from the output
        - summarize_lists (bool): Whether to summarize long lists (default: True)
        - truncate_strings (int): Maximum length for string values (default: 100)
    
    Returns:
        
    dict: A condensed version of the input results
    """
    def _format_value(value, depth, options):
        """
        Recursively format a value based on its type and formatting options.
        
        Args:
        value: The value to format
        depth (int): Current recursion depth
        options (dict): Formatting options
        
        Returns:
        
        Formatted value
        """
        # Base case: reached max depth
        if depth >= options['max_depth'] and (isinstance(value, dict) or isinstance(value, list)):
            if isinstance(value, dict):
                return {
                    '_summary': f'Nested dictionary with {len(value)} keys',
                    '_keys': list(value.keys())[:options['max_items']]
                }
            else:  # list
                return _summarize_list(value, options)
        
        # Process based on type
        if isinstance(value, dict):
            return _format_dict(value, depth, options)
        elif isinstance(value, list):
            return _format_list(value, depth, options)
        elif isinstance(value, str) and len(value) > options['truncate_strings']:
            return value[:options['truncate_strings']] + "... (truncated)"
        else:
            return value


    def _format_dict(d, depth, options):
        """Format a dictionary according to options."""
        result = {}
        
        # Filter keys based on include/exclude options
        keys_to_process = d.keys()
        if depth == 0 and options['include_keys']:  # Only apply at top level
            keys_to_process = [k for k in keys_to_process if k in options['include_keys']]
        elif depth == 0 and options['exclude_keys']:  # Only apply at top level
            keys_to_process = [k for k in keys_to_process if k not in options['exclude_keys']]
        
        # Process each key
        for key in keys_to_process:
            result[key] = _format_value(d[key], depth + 1, options)
        
        return result


    def _format_list(lst, depth, options):
        """Format a list according to options."""
        if options['summarize_lists'] and len(lst) > options['max_items']:
            return _summarize_list(lst, options)
        
        result = []
        for i, item in enumerate(lst):
            if i >= options['max_items']:
                remaining = len(lst) - options['max_items']
                result.append(f"... {remaining} more items (omitted)")
                break
            result.append(_format_value(item, depth + 1, options))
        
        return result


    def _summarize_list(lst, options):
        """Create a summary for a list."""
        if not lst:
            return []
        
        # Sample a few items
        sample = lst[:min(3, len(lst))]
        sample_formatted = [_format_value(item, options['max_depth'], options) for item in sample]
        
        # For homogeneous lists, provide type info
        if len(lst) > 0:
            item_type = type(lst[0]).__name__
            homogeneous = all(isinstance(item, type(lst[0])) for item in lst)
            type_info = f"all {item_type}" if homogeneous else "mixed types"
        else:
            type_info = "empty"
        
        return {
            '_summary': f"List with {len(lst)} items ({type_info})",
            '_sample': sample_formatted
        }

    if options is None:
        options = {}
    
    # Default options
    default_options = {
        'max_items': 5,
        'max_depth': 20,
        'include_keys': None,
        'exclude_keys': ['raw_response', 'debug_info', 'request_details'],
        'summarize_lists': True,
        'truncate_strings': 100
    }
    
    # Merge provided options with defaults
    for key, value in default_options.items():
        if key not in options:
            options[key] = value
    
    # Filter and format the result
    formatted = _format_value(result, 0, options)
    return formatted




@tool
def query_uniprot(prompt: str = None, endpoint: str = None, max_results: int = 5) -> dict:
    """
    Query the UniProt REST API using either natural language or a direct endpoint.
    
    Args:
        prompt: Natural language query about proteins (e.g., "Find information about human insulin")
        endpoint: Full or partial UniProt API endpoint URL to query directly 
        max_results: Maximum number of results to return
    
    Returns:
        
        Dictionary containing the query information and the UniProt API results
    
    Examples:
        - Natural language: query_uniprot(prompt="Find information about human insulin protein")
        - Direct endpoint: query_uniprot(endpoint="https://rest.uniprot.org/uniprotkb/P01308")
    """
    # Base URL for UniProt API
    base_url = "https://rest.uniprot.org"
    
    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load UniProt schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "uniprot.pkl")
        with open(schema_path, "rb") as f:
            uniprot_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are a protein biology expert specialized in using the UniProt REST API.
        
        Based on the user's natural language request, determine the appropriate UniProt REST API endpoint and parameters.
        
        UNIPROT REST API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including base URL, dataset, endpoint type, and parameters)
        2. "description": A brief description of what the query is doing
        
        SPECIAL NOTES:
        - Base URL is "https://rest.uniprot.org"
        - Search in reviewed (Swiss-Prot) entries first before using non-reviewed (TrEMBL) entries
        - Assume organism is human unless otherwise specified. Human taxonomy ID is 9606
        - Use gene_exact: for exact gene name searches
        - Use specific query fields like accession:, gene:, organism_id: in search queries
        - Use quotes for terms with spaces: organism_name:"Homo sapiens"
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=uniprot_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the full URL from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")
        
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        # Use provided endpoint directly
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = f"Direct query to provided endpoint"
    
    # Use the common REST API helper function
    api_result = _query_rest_api(
        endpoint=endpoint,
        method="GET",
        description=description
    )
    
    return api_result


@tool
def query_alphafold(
    uniprot_id: str,
    endpoint: str = "prediction",
    residue_range: str = None,
    download: bool = False,
    output_dir: str = None,
    file_format: str = "pdb",
    model_version: str = "v4",
    model_number: int = 1,
) -> dict:
    """
    Query the AlphaFold Database API for protein structure predictions.
    
    Args:
        uniprot_id: UniProt accession ID (e.g., "P12345")
        endpoint: Specific AlphaFold API endpoint to query: "prediction", "summary", or "annotations"
        residue_range: Specific residue range in format "start-end" (e.g., "1-100")
        download: Whether to download structure files
        output_dir: Directory to save downloaded files (default: current directory)
        file_format: Format of the structure file to download - "pdb" or "cif"
        model_version: AlphaFold model version - "v4" (latest) or "v3", "v2", "v1"
        model_number: Model number (1-5, with 1 being the highest confidence model)
    
    Returns:
        
        Dictionary containing both the query information and the AlphaFold results
    
    Examples:
        - Basic query: query_alphafold(uniprot_id="P53_HUMAN")
        - Download structure: query_alphafold(uniprot_id="P53_HUMAN", download=True, output_dir="./structures")
        - Get annotations: query_alphafold(uniprot_id="P53_HUMAN", endpoint="annotations")
    """
    # Base URL for AlphaFold API
    base_url = "https://alphafold.ebi.ac.uk/api"
    
    # Ensure we have a UniProt ID
    if not uniprot_id:
        return {"error": "UniProt ID is required"}
    
    # Validate endpoint
    valid_endpoints = ["prediction", "summary", "annotations"]
    if endpoint not in valid_endpoints:
        return {"error": f"Invalid endpoint. Must be one of: {', '.join(valid_endpoints)}"}
    
    # Construct the API URL based on endpoint
    if endpoint == "prediction":
        url = f"{base_url}/prediction/{uniprot_id}"
    elif endpoint == "summary":
        url = f"{base_url}/uniprot/summary/{uniprot_id}.json"
    elif endpoint == "annotations":
        if residue_range:
            url = f"{base_url}/annotations/{uniprot_id}/{residue_range}"
        else:
            url = f"{base_url}/annotations/{uniprot_id}"
    
    try:
        # Make the API request
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the response as JSON
        result = response.json()
                
        # Handle download request if specified
        download_info = None
        if download:
            # Ensure output directory exists
            if not output_dir:
                output_dir = "."
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate standard AlphaFold filename
            file_ext = file_format.lower()
            filename = f"AF-{uniprot_id}-F{model_number}-model_{model_version}.{file_ext}"
            file_path = os.path.join(output_dir, filename)
            
            # Construct download URL
            download_url = f"https://alphafold.ebi.ac.uk/files/{filename}"
            
            # Download the file
            download_response = requests.get(download_url)
            if download_response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(download_response.content)
                download_info = {
                    "success": True,
                    "file_path": file_path,
                    "url": download_url
                }
            else:
                download_info = {
                    "success": False,
                    "error": f"Failed to download file (status code: {download_response.status_code})",
                    "url": download_url
                }
        
        # Return the query information and results
        response_data = {
            "query_info": {
                "uniprot_id": uniprot_id,
                "endpoint": endpoint,
                "residue_range": residue_range,
                "url": url
            },
            "result": result
        }
        
        if download_info:
            response_data["download"] = download_info
            
        return response_data
        
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        response_text = ""
        
        # Try to get more detailed error info from response
        if hasattr(e, 'response') and e.response:
            try:
                error_json = e.response.json()
                if 'message' in error_json:
                    error_msg = error_json['message']
            except:
                response_text = e.response.text
        
        return {
            "error": f"AlphaFold API error: {error_msg}",
            "query_info": {
                "uniprot_id": uniprot_id,
                "endpoint": endpoint,
                "residue_range": residue_range,
                "url": url
            },
            "response_text": response_text
        }
    except Exception as e:
        return {
            "error": f"Error: {str(e)}",
            "query_info": {
                "uniprot_id": uniprot_id,
                "endpoint": endpoint,
                "residue_range": residue_range
            }
        }


@tool
def query_interpro(prompt: str = None, endpoint: str = None, max_results: int = 3) -> dict:
    """
    Query the InterPro REST API using natural language or a direct endpoint.
    
    Args:
        prompt: Natural language query about protein domains or families
        endpoint: Direct endpoint path or full URL (e.g., "/entry/interpro/IPR023411")
        max_results: Maximum number of results to return per page
    
    Returns:
        
        Dictionary containing both the query information and the InterPro API results
    
    Examples:
        - Natural language: query_interpro("Find information about kinase domains in InterPro")
        - Direct endpoint: query_interpro(endpoint="/entry/interpro/IPR023411")
    """
    # Base URL for InterPro API
    base_url = "https://www.ebi.ac.uk/interpro/api"
    
    # Default parameters
    format = "json"
    
    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load InterPro schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "interpro.pkl")
        with open(schema_path, "rb") as f:
            interpro_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a protein domain expert specialized in using the InterPro REST API.
        
        Based on the user's natural language request, determine the appropriate InterPro REST API endpoint.
        
        INTERPRO REST API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://www.ebi.ac.uk/interpro/api")
        2. "description": A brief description of what the query is doing
        
        SPECIAL NOTES:
        - Path components for data types: entry, protein, structure, set, taxonomy, proteome
        - Common sources: interpro, pfam, cdd, uniprot, pdb
        - Protein subtypes can be "reviewed" or "unreviewed"
        - For specific entries, use lowercase accessions (e.g., "ipr000001" instead of "IPR000001")
        - Endpoints can be hierarchical like "/entry/interpro/protein/uniprot/P04637"
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=interpro_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Extract the full URL from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")
        
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        # Process provided endpoint
        # If it's just a path, add the base URL
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        
        description = "Direct query to provided endpoint"
    
    # Add pagination parameters
    params = {"page": 1, "page_size": max_results}
    
    # Add format parameter if not json
    if format and format != "json":
        params["format"] = format
    
    # Make the API request
    api_result = _query_rest_api(
        endpoint=endpoint,
        method="GET",
        params=params,
        description=description
    )
    
    return api_result


@tool
def query_pdb(prompt: str = None, query: dict = None, max_results: int = 3) -> dict:
    """
    Query the RCSB PDB database using natural language or a direct structured query.
    
    Args:
        prompt: Natural language query about protein structures
        query: Direct structured query in RCSB Search API format (overrides prompt)
        max_results: Maximum number of results to return
    
    Returns:
        
        Dictionary containing the structured query, search results, and identifiers
    
    Examples:
        - Natural language: query_pdb("Find structures of human insulin")
        - Direct query: query_pdb(query={"query": {"type": "terminal", "service": "full_text", 
                               "parameters": {"value": "insulin"}}, "return_type": "entry"})
    """
    # Default parameters
    return_type = "entry"
    search_service = "full_text"
    
    # Generate search query from natural language if prompt is provided and query is not
    if prompt and not query:
        # Load schema from pickle file
        schema_path = os.path.join(SCHEMA_DB_PATH, "pdb.pkl")
        
        with open(schema_path, "rb") as f:
            schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a structural biology expert that creates precise RCSB PDB Search API queries based on natural language requests.
        
        SEARCH API SCHEMA:
        {schema}
        
        IMPORTANT GUIDELINES:
        1. Choose the appropriate search_service based on the query:
           - Use "text" for attribute-specific searches (REQUIRES attribute, operator, and value)
           - Use "full_text" for general keyword searches across multiple fields
           - Use appropriate specialized services for sequence, structure, motif searches
        
        2. For "text" searches, you MUST specify:
           - attribute: The specific field to search (use common_attributes from schema)
           - operator: The comparison method (exact_match, contains_words, less_or_equal, etc.)
           - value: The search term or value
        
        3. For "full_text" searches, only specify:
           - value: The search term(s)
        
        4. For combined searches, use "group" nodes with logical_operator ("and" or "or")
        
        5. Always specify the appropriate return_type based on what the user is looking for
        
        Generate a well-formed Search API query JSON object. Return ONLY the JSON with no additional explanation.
        """

        # Query Gemini to generate the search query
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return {
                "error": gemini_result["error"],
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
        
        # Get the query from Gemini's response
        query_json = gemini_result["data"]
    else:
        # Use provided query directly
        query_json = query if query else {
            "query": {
                "type": "terminal",
                "service": search_service,
                "parameters": {
                    "value": prompt
                }
            },
            "return_type": return_type
        }
    
    # Ensure return_type is set
    if "return_type" not in query_json:
        query_json["return_type"] = return_type
        
    # Add request options for pagination, but avoid conflicts with return_all_hits
    if "request_options" not in query_json:
        query_json["request_options"] = {}
    
    # Only add pagination if return_all_hits is not set
    if "return_all_hits" in query_json["request_options"] and query_json["request_options"]["return_all_hits"]:
        # Remove return_all_hits and use pagination instead for limited results
        query_json["request_options"]["return_all_hits"] = False
    
    if "paginate" not in query_json["request_options"]:
        query_json["request_options"]["paginate"] = {
            "start": 0,
            "rows": max_results
        }
    
    # Use query_rest_api to execute the search
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    api_result = _query_rest_api(
        endpoint=search_url,
        method="POST",
        json_data=query_json,
        description="PDB Search API query"
    )
    
    return api_result


@tool
def query_pdb_identifiers(identifiers: List[str], return_type: str = "entry", download: bool = False, attributes: List[str] = None) -> dict:
    """
    Retrieve detailed data and/or download files for PDB identifiers.
    
    Args:
        identifiers: List of PDB identifiers (from query_pdb)
        return_type: Type of results: "entry", "assembly", "polymer_entity", etc.
        download: Whether to download PDB structure files
        attributes: List of specific attributes to retrieve
    
    Returns:
        
        Dictionary containing the detailed data and file paths if downloaded
    
    Example:
    - Search and then get details:
        results = query_pdb("Find structures of human insulin")
        details = get_pdb_details(results["identifiers"], download=True)
    """
    if not identifiers:
        return {"error": "No identifiers provided"}
    
    try:
        # Fetch detailed data using Data API
        detailed_results = []
        for identifier in identifiers:
            try:
                # Determine the appropriate endpoint based on return_type and identifier format
                if return_type == "entry":
                    data_url = f"https://data.rcsb.org/rest/v1/core/entry/{identifier}"
                elif return_type == "polymer_entity":
                    entry_id, entity_id = identifier.split('_')
                    data_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{entry_id}/{entity_id}"
                elif return_type == "nonpolymer_entity":
                    entry_id, entity_id = identifier.split('_')
                    data_url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{entry_id}/{entity_id}"
                elif return_type == "polymer_instance":
                    entry_id, asym_id = identifier.split('.')
                    data_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{entry_id}/{asym_id}"
                elif return_type == "assembly":
                    entry_id, assembly_id = identifier.split('-')
                    data_url = f"https://data.rcsb.org/rest/v1/core/assembly/{entry_id}/{assembly_id}"
                elif return_type == "mol_definition":
                    data_url = f"https://data.rcsb.org/rest/v1/core/chem_comp/{identifier}"
                
                # Fetch data
                data_response = requests.get(data_url)
                data_response.raise_for_status()
                entity_data = data_response.json()
                
                # Filter attributes if specified
                if attributes:
                    filtered_data = {}
                    for attr in attributes:
                        parts = attr.split('.')
                        current = entity_data
                        try:
                            for part in parts[:-1]:
                                current = current[part]
                            filtered_data[attr] = current[parts[-1]]
                        except (KeyError, TypeError):
                            filtered_data[attr] = None
                    entity_data = filtered_data
                
                detailed_results.append({
                    "identifier": identifier,
                    "data": entity_data
                })
            except Exception as e:
                detailed_results.append({
                    "identifier": identifier,
                    "error": str(e)
                })
        
        # Download structure files if requested
        if download:
            for identifier in identifiers:
                if '_' in identifier or '.' in identifier or '-' in identifier:
                    # For non-entry identifiers, extract the PDB ID
                    if '_' in identifier:
                        pdb_id = identifier.split('_')[0]
                    elif '.' in identifier:
                        pdb_id = identifier.split('.')[0]
                    elif '-' in identifier:
                        pdb_id = identifier.split('-')[0]
                else:
                    pdb_id = identifier
                    
                try:
                    # Download PDB file
                    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                    pdb_response = requests.get(pdb_url)
                    
                    if pdb_response.status_code == 200:
                        # Create data directory if it doesn't exist
                        data_dir = os.path.join(os.path.dirname(__file__), "data", "pdb")
                        os.makedirs(data_dir, exist_ok=True)
                        
                        # Save PDB file
                        pdb_file_path = os.path.join(data_dir, f"{pdb_id}.pdb")
                        with open(pdb_file_path, 'wb') as pdb_file:
                            pdb_file.write(pdb_response.content)
                            
                        # Add download information to results
                        for result in detailed_results:
                            if result["identifier"] == identifier or result["identifier"].startswith(pdb_id):
                                result["pdb_file_path"] = pdb_file_path
                except Exception as e:
                    for result in detailed_results:
                        if result["identifier"] == identifier or result["identifier"].startswith(pdb_id):
                            result["download_error"] = str(e)
        
        return {
            "detailed_results": detailed_results
        }
        
    except Exception as e:
        return {
            "error": f"Error retrieving PDB details: {str(e)}"
        }
    
@tool
def query_kegg(prompt: str, endpoint: str = None, verbose: bool = True) -> dict:
    """
    Take a natural language prompt and convert it to a structured KEGG API query.
    
    Args:
        prompt: Natural language query about KEGG data (e.g., "Find human pathways related to glycolysis")
        endpoint: Direct KEGG API endpoint to query
        verbose: Whether to return detailed results
    
    Returns:
        
        Dictionary containing both the structured query and the KEGG results
    """
    base_url = "https://rest.kegg.jp"

    if not prompt and not endpoint:
        return {"error": "Either a prompt or an endpoint must be provided"}

    if prompt:
        # Load schema from pickle file
        schema_path = os.path.join(SCHEMA_DB_PATH, "kegg.pkl")
        with open(schema_path, "rb") as f:
            kegg_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a bioinformatics expert that helps convert natural language queries into KEGG API requests.
        
        Based on the user's natural language request, you will generate a structured query for the KEGG API.
        
        The KEGG API has the following general form:
        https://rest.kegg.jp/<operation>/<argument>[/<argument2>[/<argument3> ...]]
        
        Where <operation> can be one of: info, list, find, get, conv, link, ddi
        
        Here is the schema of available operations, databases, and other details:
        {schema}
        
        Output only a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://rest.kegg.jp")
        2. "description": A brief description of what the query is doing
        
        IMPORTANT: Your response must ONLY contain a JSON object with the required fields.
        
        EXAMPLES OF CORRECT OUTPUTS:
        - For "Find information about glycolysis pathway": {{"full_url": "https://rest.kegg.jp/info/pathway/hsa00010", "description": "Finding information about the glycolysis pathway"}}
        - For "Get information about the human BRCA1 gene": {{"full_url": "https://rest.kegg.jp/get/hsa:672", "description": "Retrieving information about BRCA1 gene in human"}}
        - For "List all human pathways": {{"full_url": "https://rest.kegg.jp/list/pathway/hsa", "description": "Listing all human-specific pathways"}}
        - For "Convert NCBI gene ID 672 to KEGG ID": {{"full_url": "https://rest.kegg.jp/conv/genes/ncbi-geneid:672", "description": "Converting NCBI Gene ID 672 to KEGG gene identifier"}}
        """

        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=kegg_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
            # Extract the query info from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info["full_url"]
        description = query_info["description"]

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
        
    if endpoint:
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to KEGG API"

    # Execute the KEGG API request using the helper function
    api_result = _query_rest_api(
        endpoint=endpoint,
        method="GET",
        description=description
    )

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])
    
    return api_result


@tool
def query_stringdb(prompt: str = None, endpoint: str = None, api_key: str = None,  download_image: bool = False, output_dir: str = None, verbose: bool = True) -> dict:
    """
    Query the STRING protein interaction database using natural language or direct endpoint.
    
    Args:
        prompt: Natural language query about protein interactions
        endpoint: Full URL to query directly (overrides prompt)
        api_key: Anthropic API key for processing
        model: Model to use for natural language processing
        download_image: Whether to download image results
        output_dir: Directory to save downloaded files
        verbose: Whether to return detailed results
    
    Returns:
        Dictionary containing the query results or error information
    
    Examples:
    - Natural language: query_stringdb("Show protein interactions for BRCA1 and BRCA2 in humans")
    - Direct endpoint: query_stringdb(endpoint="https://string-db.org/api/json/network?identifiers=BRCA1,BRCA2&species=9606")
    """
    # Base URL for STRING API 
    base_url = "https://version-12-0.string-db.org/api"
    
    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load STRING schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "stringdb.pkl")
        with open(schema_path, "rb") as f:
            stringdb_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are a protein interaction expert specialized in using the STRING database API.
        
        Based on the user's natural language request, determine the appropriate STRING API endpoint and parameters.
        
        STRING API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including all parameters)
        2. "description": A brief description of what the query is doing
        3. "output_format": The format of the output (json, tsv, image, svg)
        
        SPECIAL NOTES:
        - Common species IDs: 9606 (human), 10090 (mouse), 7227 (fruit fly), 4932 (yeast)
        - For protein identifiers, use either gene names (e.g., "BRCA1") or UniProt IDs (e.g., "P38398")
        - The "required_score" parameter accepts values from 0 to 1000 (higher means more stringent)
        - Add "caller_identity=bioagentos_api" as a parameter
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=stringdb_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the full URL from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")
        output_format = query_info.get("output_format", "json")
        
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        # Use direct endpoint
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to STRING API"
        output_format = "json"
        
        # Try to determine output format from URL
        if "image" in endpoint or "svg" in endpoint:
            output_format = "image"
    
    # Check if we're dealing with an image request
    is_image = output_format in ["image", "highres_image", "svg"]
    
    if is_image:
        if download_image:
            # For images, we need to handle the download manually
            try:
                response = requests.get(endpoint, stream=True)
                response.raise_for_status()
                
                # Create output directory if needed
                if not output_dir:
                    output_dir = "."
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate filename based on endpoint
                endpoint_parts = endpoint.split("/")
                filename = f"string_{endpoint_parts[-2]}_{int(time.time())}.{output_format}"
                file_path = os.path.join(output_dir, filename)
                
                # Save the image
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                
                return {
                    "success": True,
                    "query_info": {
                        "endpoint": endpoint,
                        "description": description,
                        "output_format": output_format
                    },
                    "result": {
                        "image_saved": True,
                        "file_path": file_path,
                        "content_type": response.headers.get('Content-Type')
                    }
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error downloading image: {str(e)}",
                    "query_info": {
                        "endpoint": endpoint,
                        "description": description
                    }
                }
        else:
            # Just report that an image is available but not downloaded
            return {
                "success": True,
                "query_info": {
                    "endpoint": endpoint,
                    "description": description,
                    "output_format": output_format
                },
                "result": {
                    "image_available": True,
                    "download_url": endpoint,
                    "note": "Set download_image=True to save the image"
                }
            }
    
    # For non-image requests, use the REST API helper
    api_result = _query_rest_api(
        endpoint=endpoint,
        method="GET",
        description=description
    )

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])
    
    return api_result
    

@tool
def query_paleobiology(prompt: str = None, endpoint: str = None, api_key: str = None, verbose: bool = True) -> dict:
    """
    Query the Paleobiology Database (PBDB) API using natural language or a direct endpoint.
    
    Args:
    prompt (str, required): Natural language query about fossil records
    endpoint (str, optional): API endpoint name or full URL
    api_key (str, optional): Anthropic API key. If None, will use ANTHROPIC_API_KEY env variable
    verbose (bool): Whether to return detailed results
    
    Returns:
        
    dict: Dictionary containing the query results or error information
    
    Examples:
    - Natural language: query_paleobiology("Find fossil records of Tyrannosaurus rex")
    - Direct endpoint: query_paleobiology(endpoint="data1.2/taxa/list.json?name=Tyrannosaurus")
    """
    # Base URL for PBDB API
    base_url = "https://paleobiodb.org/data1.2"
    
    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load PBDB schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "paleobiology.pkl")
        with open(schema_path, "rb") as f:
            pbdb_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are a paleobiology expert specialized in using the Paleobiology Database (PBDB) API.
        
        Based on the user's natural language request, determine the appropriate PBDB API endpoint and parameters.
        
        PBDB API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://paleobiodb.org/data1.2" and format extension)
        2. "description": A brief description of what the query is doing
        
        SPECIAL NOTES:
        - For taxonomic queries, be specific about taxonomic ranks and names
        - For geographic queries, use standard country/continent names or coordinate bounding boxes
        - For time interval queries, use standard geological time names (e.g., "Cretaceous", "Maastrichtian")
        - Use appropriate format extension (.json, .txt, .csv, .tsv) based on the query
        - If appropriate, use "vocab=pbdb" (default) or "vocab=com" (compact) parameter in the URL
        - For detailed occurrence data, include "show=paleoloc,phylo" in the parameters
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=pbdb_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the full URL from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")
        
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            # Add base URL if it's just a path
            if not endpoint.startswith('/'):
                endpoint = f"{base_url}/{endpoint}"
            else:
                endpoint = f"{base_url}{endpoint}"
        
        description = "Direct query to PBDB API"
    
    # Check if we're dealing with an image request
    is_image = endpoint.endswith('.png')
    
    if is_image:
        # For image queries, we need special handling
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            
            # Return image metadata without the binary data
            return {
                "success": True,
                "query_info": {
                    "endpoint": endpoint,
                    "description": description,
                    "format": "png"
                },
                "result": {
                    "content_type": response.headers.get('Content-Type'),
                    "size_bytes": len(response.content),
                    "note": "Binary image data not included in response"
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error retrieving image: {str(e)}",
                "query_info": {
                    "endpoint": endpoint,
                    "description": description
                }
            }
    
    # For non-image requests, use the REST API helper
    api_result = _query_rest_api(
        endpoint=endpoint,
        method="GET",
        description=description
    )

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])
    
    return api_result
    

@tool
def query_jaspar(prompt: str = None, endpoint: str = None, api_key: str = None,  verbose: bool = True) -> dict:
    """
    Query the JASPAR REST API using natural language or a direct endpoint.
    
    Args:
    prompt (str, required): Natural language query about transcription factor binding profiles
    endpoint (str, optional): API endpoint path (e.g., "/matrix/MA0002.2/") or full URL
    api_key (str, optional): Anthropic API key. If None, will use ANTHROPIC_API_KEY env variable
    verbose (bool): Whether to return detailed results
    
    Returns:
        
    dict: Dictionary containing the query results or error information
    
    Examples:
    - Natural language: query_jaspar("Find all transcription factor matrices for human")
    - Direct endpoint: query_jaspar(endpoint="/matrix/MA0002.2/")
    """
    # Base URL for JASPAR API
    base_url = "https://jaspar.elixir.no/api/v1"
    
    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load JASPAR schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "jaspar.pkl")
        with open(schema_path, "rb") as f:
            jaspar_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are a transcription factor binding site expert specialized in using the JASPAR REST API.
        
        Based on the user's natural language request, determine the appropriate JASPAR REST API endpoint and parameters.
        
        JASPAR REST API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://jaspar.elixir.no/api/v1" and any parameters)
        2. "description": A brief description of what the query is doing
        
        SPECIAL NOTES:
        - Common taxonomic groups include: vertebrates, plants, fungi, insects, nematodes, urochordates
        - Common collections include: CORE, UNVALIDATED, PENDING, etc.
        - Matrix IDs follow the format MA####.# (e.g., MA0002.2)
        - For inferring matrices from sequences, provide the protein sequence directly in the path
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=jaspar_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the full URL from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")
        
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            # Clean up endpoint format
            if not endpoint.startswith("/"):
                endpoint = "/" + endpoint
            
            # Ensure endpoint ends with /
            if not endpoint.endswith("/"):
                endpoint = endpoint + "/"
                
            # Add base URL
            endpoint = f"{base_url}{endpoint}"
        
        description = "Direct query to JASPAR API"
    
    # Execute the JASPAR API request using the helper function
    api_result = _query_rest_api(
        endpoint=endpoint,
        method="GET",
        description=description
    )
    
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])
    
    return api_result


@tool
def query_worms(prompt: str = None, endpoint: str = None, api_key: str = None,  verbose: bool = True) -> dict:
    """
    Query the World Register of Marine Species (WoRMS) REST API using natural language or a direct endpoint.
    
    Args:
    prompt (str, required): Natural language query about marine species
    endpoint (str, optional): Full URL or endpoint specification 
    api_key (str, optional): Anthropic API key. If None, will use ANTHROPIC_API_KEY env variable
    verbose (bool): Whether to return detailed results
    
    Returns:
        
    dict: Dictionary containing the query results or error information
    
    Examples:
    - Natural language: query_worms("Find information about the blue whale")
    - Direct endpoint: query_worms(endpoint="https://www.marinespecies.org/rest/AphiaRecordByName/Balaenoptera%20musculus")
    """
    # Base URL for WoRMS API
    base_url = "https://www.marinespecies.org/rest"
    
    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load WoRMS schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "worms.pkl")
        with open(schema_path, "rb") as f:
            worms_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are a marine biology expert specialized in using the World Register of Marine Species (WoRMS) API.
        
        Based on the user's natural language request, determine the appropriate WoRMS API endpoint and parameters.
        
        WORMS API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://www.marinespecies.org/rest" and any path/query parameters)
        2. "description": A brief description of what the query is doing
        
        SPECIAL NOTES:
        - For taxonomic searches, be precise with scientific names and use proper capitalization
        - For fuzzy matching, include "fuzzy=true" in the URL query parameters
        - When searching by name, prefer "AphiaRecordByName" for exact matches and "AphiaRecordsByName" for broader results
        - AphiaID is the main identifier in WoRMS (e.g., Blue Whale is 137087)
        - For multiple IDs or names, use the appropriate POST endpoint
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=worms_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the full URL and details from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")
        
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            # Add base URL if it's just a path
            if not endpoint.startswith('/'):
                endpoint = f"{base_url}/{endpoint}"
            else:
                endpoint = f"{base_url}{endpoint}"
        
        description = "Direct query to WoRMS API"

    # Execute the WoRMS API request using the helper function
    api_result = _query_rest_api(
        endpoint=endpoint,
        method='GET',
        description=description
    )

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])
    
    return api_result


@tool
def query_cbioportal(prompt: str = None, endpoint: str = None, api_key: str = None,  verbose: bool = True) -> dict:
    """
    Query the cBioPortal REST API using natural language or a direct endpoint.
    
    Args:
    prompt (str, required): Natural language query about cancer genomics data
    endpoint (str, optional): API endpoint path (e.g., "/studies/brca_tcga/patients") or full URL
    api_key (str, optional): Anthropic API key. If None, will use ANTHROPIC_API_KEY env variable
    verbose (bool): Whether to return detailed results
    
    Returns:
        
    dict: Dictionary containing the query results or error information
    
    Examples:
    - Natural language: query_cbioportal("Find mutations in BRCA1 for breast cancer")
    - Direct endpoint: query_cbioportal(endpoint="/studies/brca_tcga/molecular-profiles")
    """
    # Base URL for cBioPortal API
    base_url = "https://www.cbioportal.org/api"
    
    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load cBioPortal schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "cbioportal.pkl")
        with open(schema_path, "rb") as f:
            cbioportal_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are a cancer genomics expert specialized in using the cBioPortal REST API.
        
        Based on the user's natural language request, determine the appropriate cBioPortal REST API endpoint and parameters.
        
        CBIOPORTAL REST API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://www.cbioportal.org/api" and any parameters)
        2. "description": A brief description of what the query is doing
        
        SPECIAL NOTES:
        - For gene queries, use either Hugo symbol (e.g., "BRCA1") or Entrez ID (e.g., 672)
        - For pagination, include parameters "pageNumber" and "pageSize" if needed
        - For mutation data queries, always include appropriate sample identifiers
        - Common studies include: "brca_tcga" (breast cancer), "gbm_tcga" (glioblastoma), "luad_tcga" (lung adenocarcinoma)
        - For molecular profiles, common IDs follow pattern: "[study]_[data_type]" (e.g., "brca_tcga_mutations")
        - Consider including "projection=DETAILED" for more comprehensive results when appropriate
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=cbioportal_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the full URL from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")
        
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            # Clean up endpoint format
            if not endpoint.startswith("/"):
                endpoint = "/" + endpoint
            
            # Add base URL
            endpoint = f"{base_url}{endpoint}"
        
        description = "Direct query to cBioPortal API"
    
    # Execute the cBioPortal API request using the helper function
    api_result = _query_rest_api(
        endpoint=endpoint,
        method="GET",
        description=description
    )

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])
    
    return api_result


@tool
def query_clinvar(prompt: str = None, search_term: str = None, max_results: int = 3) -> dict:
    """
    Take a natural language prompt and convert it to a structured ClinVar query.
    
    Args:
        prompt: Natural language query about genetic variants (e.g., "Find pathogenic BRCA1 variants")
        search_term: Direct search term for ClinVar
        max_results: Maximum number of results to return
    
    Returns:
        
        Dictionary containing both the structured query and the ClinVar results
    """
        
    if not prompt and not search_term:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    if prompt:
        # Load ClinVar schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "clinvar.pkl")
        with open(schema_path, "rb") as f:
            clinvar_schema = pickle.load(f)
            
        # ClinVar system prompt template
        system_prompt_template = """
        You are a genetics research assistant that helps convert natural language queries into structured ClinVar search queries.
        
        Based on the user's natural language request, you will generate a structured search for the ClinVar database.
        
        Output only a JSON object with the following fields:
        1. "search_term": The exact search query to use with the ClinVar API
        
        IMPORTANT: Your response must ONLY contain a JSON object with the search term field.

        Your "search_term" MUST strictly follow these ClinVar search syntax rules/tags:

        {schema}

        For combining terms: Use AND, OR, NOT (must be capitalized)
        For complex logic: Use parentheses
        For terms with multiple words: use double quotes escaped with a backslash or underscore (e.g. breast_cancer[dis] or \"breast cancer\"[dis])
        Example: "BRCA1[gene] AND (pathogenic[clinsig] OR likely_pathogenic[clinsig])"


        EXAMPLES OF CORRECT QUERIES:
        - For "pathogenic BRCA1 variants": "BRCA1[gene] AND clinsig_pathogenic[prop]"
        - For "Specific RS": "rs6025[rsid]"
        - For "Combined search with multiple criteria": "BRCA1[gene] AND origin_germline[prop]"
        - For "Find variants in a specific genomic region": "17[chr] AND 43000000:44000000[chrpos37]"
        - If query asks for pathogenicity of a variant, it's asking for all possible germline classifications of the variant, so just [gene] AND [variant] is needed
        """

        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=clinvar_schema,
            system_template=system_prompt_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the full URL from Gemini's response
        query_info = gemini_result["data"]
        search_term = query_info.get("search_term", "")
    
        if not search_term:
            return {
                "error": "Failed to generate a valid search term from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }

        
    return _query_ncbi_database(
        database="clinvar",
        search_term=search_term,
        max_results=max_results,
    )

@tool
def query_geo(prompt: str = None, search_term: str = None, max_results: int = 3) -> dict:
    """
    Query the NCBI Gene Expression Omnibus (GEO) using natural language or a direct search term.
    
    Args:
        prompt: Natural language query about RNA-seq, microarray, or other expression data
        search_term: Direct search term in GEO syntax
        max_results: Maximum number of results to return
    
    Returns:
        
        Dictionary containing the query results or error information
    
    Examples:
        - Natural language: query_geo("Find RNA-seq datasets for breast cancer")
        - Direct search: query_geo(search_term="RNA-seq AND breast cancer AND gse[ETYP]")
    """

    if not prompt and not search_term:
        return {"error": "Either a prompt or a search term must be provided"}
    
    database = "gds"  # Default database
    
    if prompt:
        # Load GEO schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "geo.pkl")
        with open(schema_path, "rb") as f:
            geo_schema = pickle.load(f)
        
        # Create system prompt template
        system_template = """
        You are a bioinformatics research assistant that helps convert natural language queries into structured GEO (Gene Expression Omnibus) search queries.
        
        Based on the user's natural language request, you will generate a structured search for the GEO database.
        
        Output only a JSON object with the following fields:
        1. "search_term": The exact search query to use with the GEO API
        2. "database": The specific GEO database to search (either "gds" for GEO DataSets or "geoprofiles" for GEO Profiles)
        
        IMPORTANT: Your response must ONLY contain a JSON object with the required fields.

        Your "search_term" MUST strictly follow these GEO search syntax rules/tags:
        
        {schema}

        For combining terms: Use AND, OR, NOT (must be capitalized)
        For complex logic: Use parentheses
        For terms with multiple words: use double quotes or underscore (e.g. "breast cancer"[Title])
        Date ranges use colon format: 2015/01:2020/12[PDAT]
        
        Choose the appropriate database based on the user's query:
        - gds: GEO DataSets (contains Series, Datasets, Platforms, Samples metadata)
        - geoprofiles: GEO Profiles (contains gene expression data)
        
        If database isn't clearly specified, default to "gds" as it contains most common experiment metadata.

        EXAMPLES OF CORRECT OUTPUTS:
        - For "RNA-seq data in breast cancer": {"search_term": "RNA-seq AND breast cancer AND gse[ETYP]", "database": "gds"}
        - For "Mouse microarray data from 2020": {"search_term": "Mus musculus[ORGN] AND 2020[PDAT] AND microarray AND gse[ETYP]", "database": "gds"}
        - For "Expression profiles of TP53 in lung cancer": {"search_term": "TP53[Gene Symbol] AND lung cancer", "database": "geoprofiles"}
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=geo_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the search term and database from Gemini's response
        query_info = gemini_result["data"]
        search_term = query_info.get("search_term", "")
        database = query_info.get("database", "gds")
        
        if not search_term:
            return {
                "error": "Failed to generate a valid search term from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    
    # Execute the GEO query using the helper function
    result = _query_ncbi_database(
        database=database,
        search_term=search_term,
        max_results=max_results,
    )
    
    return result

@tool
def query_dbsnp(prompt: str = None, search_term: str = None, max_results: int = 3) -> dict:
    """
    Query the NCBI dbSNP database using natural language or a direct search term.
    
    Args:
        prompt: Natural language query about genetic variants/SNPs
        search_term: Direct search term in dbSNP syntax
        max_results: Maximum number of results to return
    
    Returns:
        
        Dictionary containing the query results or error information
    
    Examples:
        - Natural language: query_dbsnp("Find pathogenic variants in BRCA1")
        - Direct search: query_dbsnp(search_term="BRCA1[Gene Name] AND pathogenic[Clinical Significance]")
    """
 
    if not prompt and not search_term:
        return {"error": "Either a prompt or a search term must be provided"}
    
    if prompt:
        # Load dbSNP schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "dbsnp.pkl")
        with open(schema_path, "rb") as f:
            dbsnp_schema = pickle.load(f)
        
        # Create system prompt template
        system_template = """
        You are a genetics research assistant that helps convert natural language queries into structured dbSNP search queries.
        
        Based on the user's natural language request, you will generate a structured search for the dbSNP database.
        
        Output only a JSON object with the following fields:
        1. "search_term": The exact search query to use with the dbSNP API
        
        IMPORTANT: Your response must ONLY contain a JSON object with the search term field.

        Your "search_term" MUST strictly follow these dbSNP search syntax rules/tags:

        {schema}

        For combining terms: Use AND, OR, NOT (must be capitalized)
        For complex logic: Use parentheses
        For terms with multiple words: use double quotes (e.g. "breast cancer"[Disease Name])
        
        EXAMPLES OF CORRECT QUERIES:
        - For "pathogenic variants in BRCA1": "BRCA1[Gene Name] AND pathogenic[Clinical Significance]"
        - For "specific SNP rs6025": "rs6025[rs]"
        - For "SNPs in a genomic region": "17[Chromosome] AND 41196312:41277500[Base Position]"
        - For "common SNPs in EGFR": "EGFR[Gene Name] AND common[COMMON]"
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=dbsnp_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the search term from Gemini's response
        query_info = gemini_result["data"]
        search_term = query_info.get("search_term", "")
        
        if not search_term:
            return {
                "error": "Failed to generate a valid search term from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    
    # Execute the dbSNP query using the helper function
    result = _query_ncbi_database(
        database="snp",
        search_term=search_term,
        max_results=max_results,
    )
    
    return result

@tool
def query_ucsc(prompt: str = None, endpoint: str = None, verbose: bool = True) -> dict:
    """
    Query the UCSC Genome Browser API using natural language or a direct endpoint.
    
    Args:
        prompt: Natural language query about genomic data
        endpoint: Full URL or endpoint specification with parameters
        verbose: Whether to return detailed results
    
    Returns:
        
        Dictionary containing the query results or error information
    
    Examples:
        - Natural language: query_ucsc("Get DNA sequence of chromosome M positions 1-100 in human genome")
        - Direct endpoint: query_ucsc(endpoint="https://api.genome.ucsc.edu/getData/sequence?genome=hg38&chrom=chrM&start=1&end=100")
    """
    # Base URL for UCSC API
    base_url = "https://api.genome.ucsc.edu"
    
    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load UCSC schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "ucsc.pkl")
        with open(schema_path, "rb") as f:
            ucsc_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are a genomics expert specialized in using the UCSC Genome Browser API.
        
        Based on the user's natural language request, determine the appropriate UCSC Genome Browser API endpoint and parameters.
        
        UCSC GENOME BROWSER API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://api.genome.ucsc.edu" and all parameters)
        2. "description": A brief description of what the query is doing
        
        SPECIAL NOTES:
        - For chromosome names, always include the "chr" prefix (e.g., "chr1", "chrX", "chrM")
        - Genomic positions are 0-based (first base is position 0)
        - For "start" and "end" parameters, both must be provided together
        - The "maxItemsOutput" parameter can be used to limit the amount of data returned
        - Common genomes include: "hg38" (human), "mm39" (mouse), "danRer11" (zebrafish)
        - For sequence data, use "getData/sequence" endpoint
        - For chromosome listings, use "list/chromosomes" endpoint
        - For available genomes, use "list/ucscGenomes" endpoint
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=ucsc_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the full URL from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")
        
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
            
    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            # Add base URL if it's just a path
            endpoint = f"{base_url}/{endpoint}"
        
        description = "Direct query to UCSC Genome Browser API"
    
    # Execute the UCSC API request using the helper function
    api_result = _query_rest_api(
        endpoint=endpoint,
        method="GET",
        description=description
    )
    
    # Format the results if successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])
    
    return api_result


@tool
def query_ensembl(prompt: str = None, endpoint: str = None, verbose: bool = True) -> dict:
    """
    Query the Ensembl REST API using natural language or a direct endpoint.
    
    Args:
        prompt: Natural language query about genomic data
        endpoint: Direct API endpoint to query (e.g., "lookup/symbol/human/BRCA2") or full URL
        verbose: Whether to return detailed results
    
    Returns:
        
        Dictionary containing the query results or error information
    
    Examples:
        - Natural language: query_ensembl("Get information about the human BRCA2 gene")
        - Direct endpoint: query_ensembl(endpoint="lookup/symbol/homo_sapiens/BRCA2")
    """
    print("IN QUERY ENSEMBL")
    print("PROMPT: ", prompt)
    print("ENDPOINT: ", endpoint)
    # Base URL for Ensembl API
    base_url = "https://rest.ensembl.org"
    
    # Ensure we have either a prompt or an endpoint
    if not prompt and not endpoint:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load Ensembl schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "ensembl.pkl")
        with open(schema_path, "rb") as f:
            ensembl_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are a genomics and bioinformatics expert specialized in using the Ensembl REST API.
        
        Based on the user's natural language request, determine the appropriate Ensembl REST API endpoint and parameters.
        
        ENSEMBL REST API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "endpoint": The API endpoint to query (e.g., "lookup/symbol/homo_sapiens/BRCA2")
        2. "params": An object containing query parameters specific to the endpoint
        3. "description": A brief description of what the query is doing
        
        SPECIAL NOTES:
        - Chromosome region queries have a maximum length of 4900000 bp inclusive, so bp of start and end should be 4900000 bp apart. If the user's query exceeds this limit, Ensembl will return an error.
        - For symbol lookups, the format is "lookup/symbol/[species]/[symbol]"
        - To find the coordinates of a band on a chromosome, use /info/assembly/homo_sapiens/[chromosome] with parameters "band":1
        - To find the overlapping genes of a genomic region, use /overlap/region/homo_sapiens/[chromosome]:[start]-[end]
        - For sequence queries, specify the sequence type in parameters (genomic, cdna, cds, protein)
        - For converting rsID to hg38 genomic coordinates, use the "GET id/variation/[species]/[rsid]" endpoint
        - Many endpoints support "content-type" parameter for format specification (application/json, text/xml)
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=ensembl_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the endpoint and parameters from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("endpoint", "")
        params = query_info.get("params", {})
        description = query_info.get("description", "")
        
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        # Process provided endpoint
        if endpoint.startswith("http"):
            # If a full URL is provided, extract the endpoint part
            if endpoint.startswith(base_url):
                endpoint = endpoint[len(base_url):].lstrip('/')
        
        params = {}
        description = "Direct query to Ensembl API"
    
    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = endpoint[1:]
    
    # Prepare headers for JSON response
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Construct the URL
    url = f"{base_url}/{endpoint}"
    
    # Execute the Ensembl API request using the helper function
    api_result = _query_rest_api(
        endpoint=url,
        method="GET",
        params=params,
        headers=headers,
        description=description
    )
    
    # Format the results if successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])
    
    return api_result


@tool
def query_opentarget_genetics(prompt: str = None, query: str = None, variables: dict = None, api_key: str = None,  verbose: bool = True) -> dict:
    """
    Query the OpenTargets Genetics API using natural language or a direct GraphQL query.
    
    Args:
    prompt (str, required): Natural language query about genetic targets and variants
    query (str, optional): Direct GraphQL query string
    variables (dict, optional): Variables for the GraphQL query
    api_key (str, optional): Anthropic API key. If None, will use ANTHROPIC_API_KEY env variable
    verbose (bool): Whether to return detailed results
    
    Returns:
        
    dict: Dictionary containing the query results or error information
    
    Examples:
    - Natural language: query_opentarget("Get information about variant 1_154453788_C_T")
    - Direct query: query_opentarget(query="query variantInfo($variantId: String!) {...}", 
                                     variables={"variantId": "1_154453788_C_T"})
    """
    # Constants and initialization
    OPENTARGETS_URL = "https://api.genetics.opentargets.org/graphql"    
    # Ensure we have either a prompt or a query
    if prompt is None and query is None:
        return {"error": "Either a prompt or a GraphQL query must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load OpenTargets schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "opentarget_genetics.pkl")
        with open(schema_path, "rb") as f:
            opentarget_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are an expert in translating natural language requests into GraphQL queries for the OpenTargets Genetics API.
        
        Here is a schema of the main types and queries available in the OpenTargets Genetics API:
        {schema}
        
        Translate the user's natural language request into a valid GraphQL query for this API.
        Return only a JSON object with two fields:
        1. "query": The complete GraphQL query string
        2. "variables": A JSON object containing the variables needed for the query
        
        SPECIAL NOTES:
        - Variant IDs are typically in the format 'chromosome_position_ref_alt' (e.g., '1_154453788_C_T')
        - For L2G (locus-to-gene) queries, you need both a variant ID and a study ID
        - The API can provide variant information, QTLs, PheWAS results, pathogenicity scores, etc.
        - For mutations by gene, use the approved gene symbol (e.g., "BRCA1")
        - Always escape special characters, including quotes, in the query string (eg. \" instead of ")
        
        Return ONLY the JSON object with no additional text or explanations.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=opentarget_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the query and variables from Gemini's response
        query_info = gemini_result["data"]
        query = query_info.get("query", "")
        if variables is None:  # Only use Claude's variables if none provided
            variables = query_info.get("variables", {})
        
        if not query:
            return {
                "error": "Failed to generate a valid GraphQL query from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    
    # Execute the GraphQL query
    api_result = _query_rest_api(
        endpoint=OPENTARGETS_URL,
        method="POST",
        json_data={"query": query, "variables": variables or {}},
        headers={"Content-Type": "application/json"}
    )

    if not api_result["success"]:
        return api_result
    
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result

@tool
def query_opentarget(prompt: str = None, query: str = None, variables: dict = None, verbose: bool = False) -> dict:
    """
    Query the OpenTargets Platform API using natural language or a direct GraphQL query.
    
    Args:
        prompt: Natural language query about drug targets, diseases, and mechanisms
        query: Direct GraphQL query string
        variables: Variables for the GraphQL query
        verbose: Whether to return detailed results
    
    Returns:
        
        Dictionary containing the query results or error information
    
    Examples:
        - Natural language: query_opentarget("Find drug targets for Alzheimer's disease")
        - Direct query: query_opentarget(query="query diseaseAssociations($diseaseId: String!) {...}", 
                                         variables={"diseaseId": "EFO_0000249"})
    """
    # Constants and initialization
    OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"
    
    # Ensure we have either a prompt or a query
    if prompt is None and query is None:
        return {"error": "Either a prompt or a GraphQL query must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load OpenTargets schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "opentarget.pkl")
        with open(schema_path, "rb") as f:
            opentarget_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are an expert in translating natural language requests into GraphQL queries for the OpenTargets Platform API.
        
        Here is a schema of the main types and queries available in the OpenTargets Platform API:
        {schema}
        
        Translate the user's natural language request into a valid GraphQL query for this API.
        Return only a JSON object with two fields:
        1. "query": The complete GraphQL query string
        2. "variables": A JSON object containing the variables needed for the query
        
        SPECIAL NOTES:
        - Disease IDs typically use EFO ontology (e.g., "EFO_0000249" for Alzheimer's disease)
        - Target IDs typically use Ensembl IDs (e.g., "ENSG00000197386" for ENSG00000197386)
        - The API can provide information about drug-target associations, disease-target associations, etc.
        - Always limit results to a reasonable number using "first" parameter (e.g., first: 10)
        - Always escape special characters, including quotes, in the query string (eg. \\" instead of ")
        
        Return ONLY the JSON object with no additional text or explanations.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=opentarget_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the query and variables from Gemini's response
        query_info = gemini_result["data"]
        query = query_info.get("query", "")
        if variables is None:  # Only use Claude's variables if none provided
            variables = query_info.get("variables", {})
        
        if not query:
            return {
                "error": "Failed to generate a valid GraphQL query from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    
    # Execute the GraphQL query
    api_result = _query_rest_api(
        endpoint=OPENTARGETS_URL,
        method="POST",
        json_data={"query": query, "variables": variables or {}},
        headers={"Content-Type": "application/json"},
        description="OpenTargets Platform GraphQL query"
    )
    
    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])
    
    return api_result



@tool
def query_gwas_catalog(prompt: str = None, endpoint: str = None, max_results: int = 3) -> dict:
    """
    Query the GWAS Catalog API using natural language or a direct endpoint.
    
    Args:
        prompt: Natural language query about GWAS data
        endpoint: Full API endpoint to query (e.g., "https://www.ebi.ac.uk/gwas/rest/api/studies?diseaseTraitId=EFO_0001360")
        max_results: Maximum number of results to return
    
    Returns:
        
        Dictionary containing the query results or error information
    
    Examples:
        - Natural language: query_gwas_catalog("Find GWAS studies related to Type 2 diabetes")
        - Direct endpoint: query_gwas_catalog(endpoint="studies", params={"diseaseTraitId": "EFO_0001360"})
    """

    # Base URL for GWAS Catalog API
    base_url = "https://www.ebi.ac.uk/gwas/rest/api"
    
    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load GWAS Catalog schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "gwas_catalog.pkl")
        with open(schema_path, "rb") as f:
            gwas_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are a genomics expert specialized in using the GWAS Catalog API.
        
        Based on the user's natural language request, determine the appropriate GWAS Catalog API endpoint and parameters.
        
        GWAS CATALOG API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "endpoint": The API endpoint to query (e.g., "studies", "associations")
        2. "params": An object containing query parameters specific to the endpoint
        3. "description": A brief description of what the query is doing
        
        SPECIAL NOTES:
        - For disease/trait searches, consider using the "EFO" identifiers when possible
        - Common endpoints include: "studies", "associations", "singleNucleotidePolymorphisms", "efoTraits"
        - For pagination, use "size" and "page" parameters
        - For filtering by p-value, use "pvalueMax" parameter
        - GWAS Catalog uses a HAL-based REST API
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=gwas_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the endpoint and parameters from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("endpoint", "")
        params = query_info.get("params", {})
        description = query_info.get("description", "")
        
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        if endpoint is None:
            endpoint = "" # Use root endpoint
        params = {"size": max_results}
        description = f"Direct query to {endpoint}"
    
    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = endpoint[1:]
    
    # Construct the URL
    url = f"{base_url}/{endpoint}"
    
    # Execute the GWAS Catalog API request using the helper function
    api_result = _query_rest_api(
        endpoint=url,
        method="GET",
        params=params,
        description=description
    )
    
    return api_result
        


@tool
def query_gnomad(prompt: str = None, gene_symbol: str = None, verbose: bool = True) -> dict:
    """
    Query gnomAD for variants in a gene using natural language or direct gene symbol.
    
    Args:
        prompt: Natural language query about genetic variants
        gene_symbol: Gene symbol (e.g., "BRCA1")
        verbose: Whether to return detailed results
    
    Returns:
        
        Dictionary containing the query results or error information
    
    Examples:
        - Direct gene: query_gnomad(gene_symbol="BRCA1")
        - Natural language: query_gnomad(prompt="Find variants in the TP53 gene")
    """
    # Base URL for gnomAD API
    base_url = "https://gnomad.broadinstitute.org/api"

    
    # Ensure we have either a prompt or a gene_symbol
    if prompt is None and gene_symbol is None:
        return {"error": "Either a prompt or a gene_symbol must be provided"}
    
    # If using prompt, parse with Claude
    if prompt and not gene_symbol:
        # Load gnomAD schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "gnomad.pkl")
        with open(schema_path, "rb") as f:
            gnomad_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are a genomics expert specialized in using the gnomAD GraphQL API.
        
        Based on the user's natural language request, extract the gene symbol and relevant parameters and create the gnomAD GraphQL query.

        GnomAD GraphQL API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "query": The complete GraphQL query string
        
        SPECIAL NOTES:
        - The gene_symbol should be the official gene symbol (e.g., "BRCA1" not "breast cancer gene 1")
        - If no reference genome is specified, default to GRCh38
        - If no dataset is specified, default to gnomad_r4
        - Return only a single gene symbol, even if multiple are mentioned
        - Always escape special characters, including quotes, in the query string (eg. \" instead of ")


        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=gnomad_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the gene symbol from Gemini's response
        query_info = gemini_result["data"]
        query_str = query_info.get("query", "")
        
        if not query_str:
            return {
                "error": "Failed to extract a valid query from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        # Load gnomAD schema for gene_symbol substitution
        schema_path = os.path.join(SCHEMA_DB_PATH, "gnomad.pkl")
        with open(schema_path, "rb") as f:
            gnomad_schema = pickle.load(f)
            
        description = f"Query gnomAD for variants in {gene_symbol}"
        # replace BRCA1 with gene_symbol
        query_str = gnomad_schema.replace("BRCA1", gene_symbol)
    
    api_result = _query_rest_api(
        endpoint=base_url,
        method="POST",
        json_data={"query": query_str},
        headers={"Content-Type": "application/json"},
        description=description
    )
    
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result

@tool
def blast_sequence(sequence: str, database: str, program: str) -> Union[Dict[str, Union[str, float]], str]:
    """
    Identifies a DNA sequence using NCBI BLAST with improved error handling, timeout management, and debugging
    
    Args:
        sequence: The sequence to identify. If DNA, use database: core_nt, program: blastn; 
                  if protein, use database: nr, program: blastp
        database: The BLAST database to search against
        program: The BLAST program to use
    
    Returns:
        
        A dictionary containing the title, e-value, identity percentage, and coverage percentage of the best alignment
    """
    max_attempts = 1  # One initial attempt plus one retry
    attempts = 0
    max_runtime = 600  # 10 minutes in seconds
    
    while attempts < max_attempts:
        try:
            attempts += 1
            query_sequence = Seq(sequence)
            
            # Start timer
            start_time = time.time()
            
            # Submit BLAST job
            print(f"Submitting BLAST job (attempt {attempts}/{max_attempts})...")
            result_handle = NCBIWWW.qblast(program, database, query_sequence, expect=100, word_size=7, megablast=True)
            
            # Parse results with timeout check
            blast_records = NCBIXML.parse(result_handle)
            blast_record = None
            
            # Try to get the first record with timeout check
            while time.time() - start_time < max_runtime:
                try:
                    # Set a short timeout for next operation
                    blast_record = next(blast_records)  # Get first record
                    break  # Successfully got the record
                except StopIteration:
                    # No more records
                    return "No BLAST results found"
                except Exception as e:
                    # Check if we've exceeded the time limit
                    if time.time() - start_time >= max_runtime:
                        if attempts < max_attempts:
                            print("BLAST job timeout exceeded. Resubmitting...")
                            break  # Break to retry
                        else:
                            return "BLAST search failed after maximum attempts due to timeout"
                    # Brief pause before trying again
                    time.sleep(1)
            
            # Check if we timed out during record retrieval
            if blast_record is None:
                if attempts < max_attempts:
                    continue  # Retry
                else:
                    return "BLAST search failed after maximum attempts due to timeout"
            
            # Debug information
            print(f"Number of alignments found: {len(blast_record.alignments)}")
            
            if blast_record.alignments:
                for alignment in blast_record.alignments:
                    print("\nAlignment:")
                    print(f"hit_id: {alignment.hit_id}")
                    print(f"hit_def: {alignment.hit_def}")
                    print(f"accession: {alignment.accession}")
                    for hsp in alignment.hsps:
                        print(f"E-value: {hsp.expect}")
                        print(f"Score: {hsp.score}")
                        print(f"Identities: {hsp.identities}/{hsp.align_length}")

                        return {
                            'hit_id': alignment.hit_id,
                            'hit_def': alignment.hit_def,
                            'accession': alignment.accession,
                            'e_value': hsp.expect,
                            'identity': (hsp.identities / float(hsp.align_length)) * 100,
                            'coverage': len(hsp.query) / len(sequence) * 100
                        }
            else:
                return "No alignments found - sequence might be too short or low complexity"
                
        except Exception as e:
            if attempts < max_attempts:
                print(f"Error during BLAST search: {str(e)}. Retrying...")
                time.sleep(2)  # Wait briefly before retrying
            else:
                return f"Error during BLAST search after maximum attempts: {str(e)}"
    
    return "BLAST search failed after maximum attempts"
    
@tool
def query_reactome(prompt: str = None, endpoint: str = None, download: bool = False, output_dir: str = None, verbose: bool = True) -> dict:
    """
    Query the Reactome database using natural language or a direct endpoint.
    
    Args:
        prompt: Natural language query about biological pathways
        endpoint: Direct API endpoint or full URL
        download: Whether to download pathway diagrams
        output_dir: Directory to save downloaded files
        verbose: Whether to return detailed results
    
    Returns:
        
        Dictionary containing the query results or error information
    
    Examples:
        - Natural language: query_reactome("Find pathways related to DNA repair")
        - Direct endpoint: query_reactome(endpoint="data/pathways/R-HSA-73894")
    """
    # Base URLs for Reactome APIs
    content_base_url = "https://reactome.org/ContentService"
    analysis_base_url = "https://reactome.org/AnalysisService"
    
    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    # Create output directory if downloading and directory doesn't exist
    if download and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # If using prompt, parse with Claude
    if prompt:
        # Load Reactome schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "reactome.pkl")
        with open(schema_path, "rb") as f:
            reactome_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are a bioinformatics expert specialized in using the Reactome API.
        
        Based on the user's natural language request, determine the appropriate Reactome API endpoint and parameters.
        
        REACTOME API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "endpoint": The API endpoint to query (e.g., "data/pathways/PATHWAY_ID", "data/query/GENE_SYMBOL")
        2. "base": Which base URL to use ("content" for ContentService or "analysis" for AnalysisService)
        3. "params": An object containing query parameters specific to the endpoint
        4. "description": A brief description of what the query is doing
        
        SPECIAL NOTES:
        - Reactome has two primary APIs: ContentService (for retrieving specific pathway data) and AnalysisService (for analyzing gene lists)
        - For pathway queries, use "data/pathways/PATHWAY_ID" with the pathway stable identifier (e.g., R-HSA-73894)
        - For gene queries, use "data/query/GENE" with official gene symbol (e.g., "BRCA1")
        - For pathway diagrams, include "download: true" in your response if the query is for pathway visualization
        - Common human pathway IDs start with "R-HSA-"
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=reactome_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the endpoint and parameters from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("endpoint", "")
        base = query_info.get("base", "content")  # Default to ContentService
        params = query_info.get("params", {})
        description = query_info.get("description", "")
        should_download = query_info.get("download", download)  # Override download if specified
        
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        # Process provided endpoint
        if endpoint.startswith("http"):
            # Full URL already provided
            if "ContentService" in endpoint:
                base = "content"
            elif "AnalysisService" in endpoint:
                base = "analysis"
            else:
                base = "content"  # Default
        else:
            # Just endpoint provided, assume ContentService by default
            base = "content"
        
        params = {}
        description = f"Direct query to Reactome {base} API: {endpoint}"
        should_download = download
    
    # Select base URL based on API type
    base_url = content_base_url if base == "content" else analysis_base_url
    
    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = endpoint[1:]
    
    # Construct the URL
    if endpoint.startswith("http"):
        url = endpoint  # Full URL already provided
    else:
        url = f"{base_url}/{endpoint}"
    
    # Execute the Reactome API request using the helper function
    api_result = _query_rest_api(
        endpoint=url,
        method="GET",
        params=params,
        description=description
    )
    
    # Handle downloading pathway diagrams if requested
    if should_download and api_result.get("success") and "result" in api_result:
        result = api_result["result"]
        pathway_id = None
        
        # Try to extract pathway ID from result
        if isinstance(result, dict):
            pathway_id = result.get("stId") or result.get("dbId")
        
        # If we have a pathway ID and output directory, download diagram
        if pathway_id and output_dir:
            diagram_url = f"{content_base_url}/data/pathway/{pathway_id}/diagram"
            try:
                diagram_response = requests.get(diagram_url)
                diagram_response.raise_for_status()
                
                # Save diagram file
                diagram_path = os.path.join(output_dir, f"{pathway_id}_diagram.png")
                with open(diagram_path, "wb") as f:
                    f.write(diagram_response.content)
                
                api_result["diagram_path"] = diagram_path
            except Exception as e:
                api_result["diagram_error"] = f"Failed to download diagram: {str(e)}"
    
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])
    
    return api_result



@tool
def query_regulomedb(prompt: str = None, endpoint: str = None, api_key: str = None,  verbose: bool = False) -> dict:
    """
    Query the RegulomeDB database using natural language or direct variant/coordinate specification.
    
    Args:
    prompt (str, required): Natural language query about regulatory elements
    endpoint (str, optional): Direct endpoint URL or variant/coordinate specification
    api_key (str, optional): Anthropic API key. If None, will use ANTHROPIC_API_KEY env variable
    verbose (bool): Whether to return detailed results
    
    Returns:
        
    dict: Dictionary containing the query results or error information
    
    Examples:
    - Natural language: query_regulomedb("Find regulatory elements for rs35675666")
    - Direct variant: query_regulomedb(variant="rs35675666")
    - Coordinates: query_regulomedb(coordinates="chr11:5246919-5246919")
    """
    # Base URL for RegulomeDB API
    base_url = "https://regulomedb.org/regulome-search/"
    
    # Ensure we have either a prompt, variant, or coordinates
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt, variant ID, or genomic coordinates must be provided"}
    
    # If using prompt, parse with Claude
    if prompt and not endpoint:
                
        # Create system prompt template
        system_template = """
        You are a genomics expert specialized in using the RegulomeDB API.
        
        Based on the user's natural language request, extract the variant ID or genomic coordinates they want to query.
        
        Your response should be a JSON object with ONLY ONE of the following fields:
        1. "endpoint": The API endpoint to query (e.g., "https://regulomedb.org/regulome-search/?regions=chr11:5246919-5246919&genome=GRCh38")

        
        SPECIAL NOTES:
        - RegulomeDB only works with human genome data
        - Variant IDs should be rsIDs from dbSNP when possible. The endpoint should be in the format https://regulomedb.org/regulome-search/?regions=rsID&genome=GRCh38
        - Thumbnails for chip and chromatin should be in the format https://regulomedb.org/regulome-search?regions=chr11:5246919-5246919&genome=GRCh38/thumbnail=chip
        - Coordinates should be in GRCh37/hg19 format 
        - For single base queries, use the same position for start and end (e.g., "chr11:5246919-5246919")
        - Chromosome should be specified with "chr" prefix (e.g., "chr11" not just "11")
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=None,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the variant or coordinates from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("endpoint", "")
        
        if not endpoint:
            return {
                "error": "Failed to extract a valid variant ID or coordinates from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        description = f"Query RegulomeDB for {endpoint}"
    
    # Construct the request URL
    endpoint = endpoint
    
    # Execute the RegulomeDB API request using the helper function
    api_result = _query_rest_api(
        endpoint=endpoint,
        method="GET",
        headers = {'Accept': 'application/json'}
    )
    
    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])
    
    return api_result

@tool
def query_pride(prompt: str = None, endpoint: str = None, api_key: str = None,  max_results: int = 3) -> dict:
    """
    Query the PRIDE (PRoteomics IDEntifications) database using natural language or a direct endpoint.
    
    Args:
    prompt (str, required): Natural language query about proteomics data
    endpoint (str, optional): The full endpoint to query (e.g., "https://www.ebi.ac.uk/pride/ws/archive/v2/projects?keyword=breast%20cancer")
    api_key (str, optional): Anthropic API key. If None, will use ANTHROPIC_API_KEY env variable
    verbose (bool): Whether to return detailed results
    max_results (int): Maximum number of results to return
    verbose (bool): Whether to return detailed results
    
    Returns:
        
    dict: Dictionary containing the query results or error information
    
    Examples:
    - Natural language: query_pride("Find proteomics data related to breast cancer")
    - Direct endpoint: query_pride(endpoint="projects", params={"keyword": "breast cancer"})
    """
    # Base URL for PRIDE API
    base_url = "https://www.ebi.ac.uk/pride/ws/archive/v2"
    
    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load PRIDE schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "pride.pkl")
        with open(schema_path, "rb") as f:
            pride_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are a proteomics expert specialized in using the PRIDE API.
        
        Based on the user's natural language request, determine the appropriate PRIDE API endpoint and parameters.
        
        PRIDE API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "endpoint": The full url endpoint to query 
        2. "description": A brief description of what the query is doing
        
        SPECIAL NOTES:
        - PRIDE is a repository for proteomics data stored at EBI
        - Common endpoints include: "projects", "assays", "files", "proteins", "peptideevidences"
        - For searching projects, you can use parameters like "keyword", "species", "tissue", "disease"
        - For pagination, use "page" and "pageSize" parameters
        - Most results include PagingObject and FieldsObject structures
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=pride_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the endpoint and parameters from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("endpoint", "")
        params = query_info.get("params", {})
        description = query_info.get("description", "")
        
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        # Process provided endpoint
        params = {"pageSize": max_results, "page": 0}
        description = f"Direct query to PRIDE {endpoint}"
    
    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = f"{base_url}{endpoint}"
    elif not endpoint.startswith("http"):
        endpoint = f"{base_url}/{endpoint.lstrip('/')}"
    description = f"Direct query to provided endpoint"
    
    # Execute the PRIDE API request using the helper function
    api_result = _query_rest_api(
        endpoint=endpoint,
        method="GET",
        params=params,
        description=description
    )
    
    return api_result

@tool
def query_gtopdb(prompt: str = None, endpoint: str = None, api_key: str = None,  verbose: bool = True) -> dict:
    """
    Query the Guide to PHARMACOLOGY database (GtoPdb) using natural language or a direct endpoint.
    
    Args:
    prompt (str, required): Natural language query about drug targets, ligands, and interactions
    endpoint (str, optional): Full API endpoint to query (e.g., "https://www.guidetopharmacology.org/services/targets?type=GPCR&name=beta-2")
    api_key (str, optional): Anthropic API key. If None, will use ANTHROPIC_API_KEY env variable
    verbose (bool): Whether to return detailed results
    verbose (bool): Whether to return detailed results
    
    Returns:
        
    dict: Dictionary containing the query results or error information
    
    Examples:
    - Natural language: query_gtopdb("Find ligands that target the beta-2 adrenergic receptor")
    - Direct endpoint: query_gtopdb(endpoint="targets", params={"type": "GPCR", "name": "beta-2"})
    """
    # Base URL for GtoPdb API
    base_url = "https://www.guidetopharmacology.org/services"
    
    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load GtoPdb schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "gtopdb.pkl")
        with open(schema_path, "rb") as f:
            gtopdb_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are a pharmacology expert specialized in using the Guide to PHARMACOLOGY API.
        
        Based on the user's natural language request, determine the appropriate GtoPdb API endpoint and parameters.
        
        GTOPDB API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "endpoint": The full API endpoint to query
        2. "description": A brief description of what the query is doing
        
        SPECIAL NOTES:
        - Main endpoints include: "targets", "ligands", "interactions", "diseases", "refs"
        - Target types include: "GPCR", "NHR", "LGIC", "VGIC", "OtherIC", "Enzyme", "CatalyticReceptor", "Transporter", "OtherProtein"
        - Ligand types include: "Synthetic organic", "Metabolite", "Natural product", "Endogenous peptide", "Peptide", "Antibody", "Inorganic", "Approved", "Withdrawn", "Labelled", "INN"
        - Interaction types include: "Activator", "Agonist", "Allosteric modulator", "Antagonist", "Antibody", "Channel blocker", "Gating inhibitor", "Inhibitor", "Subunit-specific"
        - For specific target/ligand details, use formats like "targets/{{targetId}}" or "ligands/{{ligandId}}"
        - For subresources, use formats like "targets/{{targetId}}/interactions" or "ligands/{{ligandId}}/structure"
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=gtopdb_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the endpoint and parameters from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("endpoint", "")
        description = query_info.get("description", "")
        
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        # Process provided endpoint
        description = f"Direct query to GtoPdb {endpoint}"
    
    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = f"{base_url}{endpoint}"
    elif not endpoint.startswith("http"):
        endpoint = f"{base_url}/{endpoint.lstrip('/')}"
    description = f"Direct query to provided endpoint"

    # Execute the GtoPdb API request using the helper function
    api_result = _query_rest_api(
        endpoint=endpoint,
        method="GET",
        description=description
    )
    
    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])
    
    return api_result

@tool
def region_to_ccre_screen(coord_chrom: str, coord_start: int, coord_end: int, assembly: str = "GRCh38" ) -> str:
    """
    Given starting and ending coordinates, this function retrieves information of intersecting cCREs.
    
    Args:
        coord_chrom: Chromosome of the gene, formatted like 'chr12'
        coord_start: Starting chromosome coordinate
        coord_end: Ending chromosome coordinate
        assembly: Assembly of the genome, formatted like 'GRCh38'. Default is 'GRCh38'
    
    Returns:
        
        A detailed string explaining the steps and the intersecting cCRE data or any error encountered
    """
    steps = []
    try:
        steps.append(f"Starting cCRE data retrieval for coordinates: {coord_chrom}:{coord_start}-{coord_end} (Assembly: {assembly}).")

        # Build the URL and request payload
        url = "https://screen-beta-api.wenglab.org/dataws/cre_table"
        data = {
            "assembly": assembly,
            "coord_chrom": coord_chrom,
            "coord_start": coord_start,
            "coord_end": coord_end
        }

        steps.append("Sending POST request to API with the following data:")
        steps.append(str(data))

        # Make the request
        response = requests.post(url, json=data)

        # Check if the response is successful
        if not response.ok:
            raise Exception(f"Request failed with status code {response.status_code}. Response: {response.text}")

        steps.append("Request executed successfully. Parsing the response...")

        # Parse the JSON response
        response_json = response.json()
        if "errors" in response_json:
            raise Exception(f"API error: {response_json['errors']}")

        # Function to reduce and filter response data
        def reduce_tokens(res_json):
            # Remove unnecessary fields and round floats
            res = sorted(res_json["cres"], key=lambda x: x['dnase_zscore'], reverse=True)
            filtered_res = []

            for item in res:
                new_item = {
                    'chrom': item['chrom'],
                    'start': item['start'],
                    'len': item['len'],
                    'pct': item['pct'],
                    'ctcf_zscore': round(item['ctcf_zscore'], 2),
                    'dnase_zscore': round(item['dnase_zscore'], 2),
                    'enhancer_zscore': round(item['enhancer_zscore'], 2),
                    'promoter_zscore': round(item['promoter_zscore'], 2),
                    'accession': item['info']['accession'],
                    'isproximal': item['info']['isproximal'],
                    'concordance': item['info']['concordant'],
                    'ctcfmax': round(item['info']['ctcfmax'], 2),
                    'k4me3max': round(item['info']['k4me3max'], 2),
                    'k27acmax': round(item['info']['k27acmax'], 2)
                }
                filtered_res.append(new_item)
            return filtered_res

        # Process the response data
        filtered_data = reduce_tokens(response_json)

        if not filtered_data:
            steps.append(f"No intersecting cCREs found for coordinates: {coord_chrom}:{coord_start}-{coord_end}.")
            return "\n".join(steps + ["No cCRE data available for this genomic region."])

        # Format the result into a readable string
        ccre_data_string = f"Intersecting cCREs for {coord_chrom}:{coord_start}-{coord_end} (Assembly: {assembly}):\n"
        for i, ccre in enumerate(filtered_data, 1):
            ccre_data_string += (
                f"cCRE {i}:\n"
                f"  Chromosome: {ccre['chrom']}\n"
                f"  Start: {ccre['start']}\n"
                f"  Length: {ccre['len']}\n"
                f"  PCT: {ccre['pct']}\n"
                f"  CTCF Z-score: {ccre['ctcf_zscore']}\n"
                f"  DNase Z-score: {ccre['dnase_zscore']}\n"
                f"  Enhancer Z-score: {ccre['enhancer_zscore']}\n"
                f"  Promoter Z-score: {ccre['promoter_zscore']}\n"
                f"  Accession: {ccre['accession']}\n"
                f"  Is Proximal: {ccre['isproximal']}\n"
                f"  Concordance: {ccre['concordance']}\n"
                f"  CTCFmax: {ccre['ctcfmax']}\n"
                f"  K4me3max: {ccre['k4me3max']}\n"
                f"  K27acmax: {ccre['k27acmax']}\n\n"
            )

        steps.append(f"cCRE data successfully retrieved and formatted for {coord_chrom}:{coord_start}-{coord_end}.")
        return "\n".join(steps + [ccre_data_string])

    except Exception as e:
        steps.append(f"Exception encountered: {str(e)}")
        return "\n".join(steps + [f"Error: {str(e)}"])


@tool
def get_genes_near_ccre(accession: str, assembly: str, chromosome: str, k: int = 10) -> str:
    """
    Given a cCRE (Candidate cis-Regulatory Element), this function returns a string containing the 
    steps it performs and the k nearest genes sorted by distance.

    Args:
        accession: ENCODE Accession ID of query cCRE, e.g., EH38E1516980
        assembly: Assembly of the gene, e.g., 'GRCh38'
        chromosome: Chromosome of the gene, e.g., 'chr12'
        k: Number of nearby genes to return, sorted by distance. Default is 10

    Returns:
        
        Steps performed and the result
    """
    
    steps_log = f"Starting process with accession: {accession}, assembly: {assembly}, chromosome: {chromosome}, k: {k}\n"
    
    url = "https://screen-beta-api.wenglab.org/dataws/re_detail/nearbyGenomic"
    data = {
        "accession": accession,
        "assembly": assembly,
        "coord_chrom": chromosome
    }
    
    steps_log += "Sending POST request to API with given data.\n"
    response = requests.post(url, json=data)
    
    if not response.ok:
        steps_log += f"API request failed with response: {response.text}\n"
        return steps_log
    
    response_json = response.json()
    
    if "errors" in response_json:
        steps_log += f"API returned errors: {response_json['errors']}\n"
        return steps_log
    
    nearby_genes = response_json.get(accession, {}).get("nearby_genes", [])
    if not nearby_genes:
        steps_log += "No nearby genes found for the given accession.\n"
        return steps_log
    
    steps_log += "Successfully retrieved nearby genes. Sorting them by distance.\n"
    sorted_genes = sorted(nearby_genes, key=lambda x: x['distance'])[:k]
    
    steps_log += f"Returning the top {k} nearest genes.\n"
    steps_log += "Result:\n"
    
    for gene in sorted_genes:
        gene_name = gene.get('name', 'Unknown')
        distance = gene.get('distance', 'N/A')
        ensembl_id = gene.get('ensemblid_ver', 'N/A')
        start = gene.get('start', 'N/A')
        stop = gene.get('stop', 'N/A')
        chrom = gene.get('chrom', 'N/A')
        steps_log += f"Gene: {gene_name}, Distance: {distance}, Ensembl ID: {ensembl_id}, Chromosome: {chrom}, Start: {start}, Stop: {stop}\n"
    
    return steps_log

@tool
def query_remap(prompt: str = None, endpoint: str = None, api_key: str = None,  verbose: bool = True) -> dict:
    """
    Query the ReMap database for regulatory elements and transcription factor binding sites.
    
    Args:
    prompt (str, required): Natural language query about transcription factors and binding sites
    endpoint (str, optional): Full API endpoint to query (e.g., "https://remap.univ-amu.fr/api/v1/catalogue/tf?tf=CTCF")
    api_key (str, optional): Anthropic API key. If None, will use ANTHROPIC_API_KEY env variable
    verbose (bool): Whether to return detailed results
    verbose (bool): Whether to return detailed results
    
    Returns:
        
    dict: Dictionary containing the query results or error information
    
    Examples:
    - Natural language: query_remap("Find CTCF binding sites in chromosome 1")
    - Direct endpoint: query_remap(endpoint="catalogue/tf", params={"tf": "CTCF"})
    """
    # Base URL for ReMap API
    base_url = "https://remap.univ-amu.fr/api/v1"
    
    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load ReMap schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "remap.pkl")
        with open(schema_path, "rb") as f:
            remap_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are a genomics expert specialized in using the ReMap database API.
        
        Based on the user's natural language request, determine the appropriate ReMap API endpoint and parameters.
        
        REMAP API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "endpoint": The full url endpoint to query 
        2. "description": A brief description of what the query is doing
        
        SPECIAL NOTES:
        - ReMap is a database of regulatory regions and transcription factor binding sites based on ChIP-seq experiments
        - Common endpoints include: "catalogue/tf" (transcription factors), "catalogue/biotype" (biotypes), "browse/peaks" (binding sites)
        - For searching binding sites, you can filter by transcription factor (tf), cell line, biotype, chromosome, etc.
        - Genomic coordinates should be specified with "chr", "start", and "end" parameters
        - For limiting results, use "limit" parameter (default is 100)
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=remap_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the endpoint and parameters from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("endpoint", "")
        description = query_info.get("description", "")
        
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        # Process provided endpoint
        description = f"Direct query to ReMap {endpoint}"
    
    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = f"{base_url}{endpoint}"
    elif not endpoint.startswith("http"):
        endpoint = f"{base_url}/{endpoint.lstrip('/')}"
    description = f"Direct query to provided endpoint"
    
    # Execute the ReMap API request using the helper function
    api_result = _query_rest_api(
        endpoint=endpoint,
        method="GET",
        description=description
    )
    
    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])
    
    return api_result

@tool
def query_mpd(prompt: str = None, endpoint: str = None, api_key: str = None,  verbose: bool = True) -> dict:
    """
    Query the Mouse Phenome Database (MPD) for mouse strain phenotype data.
    
    Args:
    prompt (str, required): Natural language query about mouse phenotypes, strains, or measurements
    endpoint (str, optional): Full API endpoint to query (e.g., "https://phenomedoc.jax.org/MPD_API/strains")
    api_key (str, optional): Anthropic API key. If None, will use ANTHROPIC_API_KEY env variable
    verbose (bool): Whether to return detailed results
    verbose (bool): Whether to return detailed results
    
    Returns:
        
    dict: Dictionary containing the query results or error information
    
    Examples:
    - Natural language: query_mpd("Find phenotype data for C57BL/6J mice related to blood glucose")
    - Direct endpoint: query_mpd(endpoint="strains/C57BL/6J/measures")
    """
    # Base URL for MPD API
    base_url = "https://phenome.jax.org"
    
    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load MPD schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "mpd.pkl")
        with open(schema_path, "rb") as f:
            mpd_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are a mouse genetics expert specialized in using the Mouse Phenome Database (MPD) API.
        
        Based on the user's natural language request, determine the appropriate MPD API endpoint and parameters.
        
        MPD API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "endpoint": The full url endpoint to query (e.g. https://phenome.jax.org/api/strains)
        2. "description": A brief description of what the query is doing
        
        SPECIAL NOTES:
        - The MPD contains phenotype data for diverse strains of laboratory mice
        - Common endpoints include: "strains" (mouse strains), "measures" (phenotypic measurements), "genes" (gene info)
        - Use the url to construct the endpoint, not the endpoint name
        - Common mouse strains include: "C57BL/6J", "DBA/2J", "BALB/cJ", "A/J", "129S1/SvImJ"
        - Common phenotypic domains include: "behavior", "blood_chemistry", "body_weight", "cardiovascular", "growth", "metabolism"
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=mpd_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the endpoint and parameters from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("endpoint", "")
        description = query_info.get("description", "")
        
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        # Process provided endpoint
        description = f"Direct query to MPD {endpoint}"
    
    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = f"{base_url}{endpoint}"
    elif not endpoint.startswith("http"):
        endpoint = f"{base_url}/{endpoint.lstrip('/')}"
    description = f"Direct query to provided endpoint"

    
    # Execute the MPD API request using the helper function
    api_result = _query_rest_api(
        endpoint=endpoint,
        method="GET",
        description=description
    )
    
    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])
    
    return api_result

@tool
def query_emdb(prompt: str = None, endpoint: str = None, api_key: str = None,  verbose: bool = True) -> dict:
    """
    Query the Electron Microscopy Data Bank (EMDB) for 3D macromolecular structures.
    
    Args:
    prompt (str, required): Natural language query about EM structures and associated data
    endpoint (str, optional): Full API endpoint to query (e.g., "https://www.ebi.ac.uk/emdb/api/search")
    api_key (str, optional): Anthropic API key. If None, will use ANTHROPIC_API_KEY env variable
    verbose (bool): Whether to return detailed results
    verbose (bool): Whether to return detailed results
    
    Returns:
        
    dict: Dictionary containing the query results or error information
    
    Examples:
    - Natural language: query_emdb("Find cryo-EM structures of ribosomes at resolution better than 3Å")
    - Direct endpoint: query_emdb(endpoint="entry/EMD-10000")
    """
    # Base URL for EMDB API
    base_url = "https://www.ebi.ac.uk/emdb/api"
    
    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}
    
    # If using prompt, parse with Claude
    if prompt:
        # Load EMDB schema
        schema_path = os.path.join(SCHEMA_DB_PATH, "emdb.pkl")
        with open(schema_path, "rb") as f:
            emdb_schema = pickle.load(f)
                
        # Create system prompt template
        system_template = """
        You are a structural biology expert specialized in using the Electron Microscopy Data Bank (EMDB) API.
        
        Based on the user's natural language request, determine the appropriate EMDB API endpoint and parameters.
        
        EMDB API SCHEMA:
        {schema}
        
        Your response should be a JSON object with the following fields:
        1. "endpoint": The API endpoint to query (e.g., "search", "entry/EMD-XXXXX")
        2. "params": An object containing query parameters specific to the endpoint
        3. "description": A brief description of what the query is doing
        
        SPECIAL NOTES:
        - EMDB contains 3D macromolecular structures determined by electron microscopy
        - Common endpoints include: "search" (search for entries), "entry/EMD-XXXXX" (specific entry details)
        - For searching, you can filter by resolution, specimen, authors, release date, etc.
        - Resolution filters should be specified with "resolution_low" and "resolution_high" parameters
        - For specific entry retrieval, use the format "entry/EMD-XXXXX" where XXXXX is the EMDB ID
        - Common specimen types include: "ribosome", "virus", "membrane protein", "filament"
        
        Return ONLY the JSON object with no additional text.
        """
        
        # Query Gemini to generate the API call
        gemini_result = _query_gemini_for_api(
            prompt=prompt,
            schema=emdb_schema,
            system_template=system_template
        )
        
        if not gemini_result["success"]:
            return gemini_result
        
        # Get the endpoint and parameters from Gemini's response
        query_info = gemini_result["data"]
        endpoint = query_info.get("endpoint", "")
        params = query_info.get("params", {})
        description = query_info.get("description", "")
        
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "gemini_response": gemini_result.get("raw_response", "No response")
            }
    else:
        # Process provided endpoint
        params = {}
        description = f"Direct query to EMDB {endpoint}"
    
    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = f"{base_url}{endpoint}"
    elif not endpoint.startswith("http"):
        endpoint = f"{base_url}/{endpoint.lstrip('/')}"
    description = f"Direct query to provided endpoint"
    
    # Execute the EMDB API request using the helper function
    api_result = _query_rest_api(
        endpoint=endpoint,
        method="GET",
        params=params,
        description=description
    )
    
    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])
    
    return api_result
