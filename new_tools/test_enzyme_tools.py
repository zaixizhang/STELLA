#!/usr/bin/env python3
"""
Test script for virtual_screening_tools.py integration with manager agent

This script tests:
1. Individual tool functionality
2. Agent integration
3. Tool execution via agent
4. Error handling and edge cases
"""

import os
import sys
import json
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
main_dir = current_dir.parent
sys.path.append(str(main_dir))
sys.path.append(str(current_dir))

# Set hardcoded API key as backup
OPENROUTER_API_KEY_STRING = "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"

# Set environment variable if not already set
if not os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY_STRING
    print("✅ OPENROUTER_API_KEY set using backup key")
else:
    print("✅ OPENROUTER_API_KEY found in environment")

# Import required modules
try:
    from smolagents import CodeAgent, ToolCallingAgent
    from smolagents.models import OpenAIServerModel
    from enzyme_tools import (
        run_mmseqs_search,
        parse_mmseqs_search_results,
        run_ephod_prediction,
        run_boltz_protein_structure_prediction,
        run_clean_ec_prediction, run_catapro_prediction,
        run_prime_ogt_prediction, run_chroma_redesign,
        run_iqtree_reconstruct_phylogenetic_trees,
        run_ligandmpnn_redesign
    )

    print("✅ Successfully imported all required modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", OPENROUTER_API_KEY_STRING)


def test_environment_setup():
    """Test if environment is properly set up"""
    print("\n" + "=" * 50)
    print("🔧 TESTING ENVIRONMENT SETUP")
    print("=" * 50)

    # Check API key
    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY not found")
        return False
    else:
        print("✅ OPENROUTER_API_KEY found")

    # Check required files
    required_files = [
        "resource/Kegg_pathways.csv",
        "resource/hgnc_name.txt",
        "resource/diseases/human_disease_integrated_full.tsv"
    ]

    # Check optional files/directories
    optional_files = [
        "resource/GO"  # GO terms directory (optional)
    ]

    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            missing_files.append(file_path)

    # Check optional files
    for file_path in optional_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path} (optional)")
        else:
            print(f"⚠️  Missing: {file_path} (optional - GO search may not work)")

    if missing_files:
        print("\n⚠️  Some required files are missing. Some tests may fail.")
        return False

    return True


def test_individual_tools():
    """Test individual tool functionality"""
    print("\n" + "=" * 50)
    print("🧪 TESTING INDIVIDUAL TOOLS")
    print("=" * 50)

    # Test gene validation
    print("\n--- Testing gene validation ---")
    try:
        test_genes = ['TP53', 'BRCA1', 'FAKEGENE', 'BRAF']
        valid, invalid = validate_genes(test_genes)
        print(f"✅ Gene validation successful:")
        print(f"   Valid genes: {valid}")
        print(f"   Invalid genes: {invalid}")
    except Exception as e:
        print(f"❌ Gene validation failed: {e}")
        return False

    # Test KEGG pathway search
    print("\n--- Testing KEGG pathway search ---")
    try:
        result = kegg_pathway_search(
            query_list=["apoptosis"],
            model_name="gpt-4o-mini"
        )
        print(f"✅ KEGG search successful:")
        print(f"   Status: {result.get('status', 'unknown')}")
        if result.get('status') == 'success':
            final_results = result.get('final_results', {})
            for query, genes in final_results.items():
                print(f"   Query '{query}': Found {len(genes)} genes")
                if genes:
                    print(f"   Sample genes: {[g['gene'] for g in genes[:3]]}")
    except Exception as e:
        print(f"❌ KEGG search failed: {e}")
        return False

    # Test disease gene search
    print("\n--- Testing disease gene search ---")
    try:
        result = disease_gene_search(
            disease_list=["breast cancer"],
            confidence_cutoff=2.0,
            top_n=5
        )
        print(f"✅ Disease search successful:")
        print(f"   Status: {result.get('status', 'unknown')}")
        if result.get('status') == 'success':
            final_results = result.get('final_results', {})
            for disease, genes in final_results.items():
                print(f"   Disease '{disease}': Found {len(genes)} genes")
                if genes:
                    print(f"   Sample genes: {[g['gene'] for g in genes[:3]]}")
    except Exception as e:
        print(f"❌ Disease search failed: {e}")
        return False

    # Test STRING database search
    print("\n--- Testing STRING database search ---")
    try:
        result = string_database_search(
            gene_list=["TP53", "BRCA1"],
            score_cutoff=0.7,
            limit=10
        )
        print(f"✅ STRING search successful:")
        print(f"   Status: {result.get('status', 'unknown')}")
        if result.get('status') == 'success':
            final_results = result.get('final_results', {})
            for gene, interactions in final_results.items():
                print(f"   Gene '{gene}': Found {len(interactions)} interactions")
                if interactions:
                    print(f"   Sample interactions: {[i['gene'] for i in interactions[:3]]}")
    except Exception as e:
        print(f"❌ STRING search failed: {e}")
        return False

    # Test GO terms search
    print("\n--- Testing GO terms search ---")
    try:
        result = go_terms_search(
            query_list=["DNA repair"],
            max_candidates=50,
            model_name="gpt-4o-mini"
        )
        print(f"✅ GO search successful:")
        print(f"   Status: {result.get('status', 'unknown')}")
        if result.get('status') == 'success':
            final_results = result.get('final_results', {})
            for query, genes in final_results.items():
                print(f"   Query '{query}': Found {len(genes)} genes")
                if genes:
                    print(f"   Sample genes: {[g['gene'] for g in genes[:3]]}")
        elif 'GO' in result.get('error_message', ''):
            print(f"   ⚠️  GO data not available: {result.get('error_message')}")
    except Exception as e:
        print(f"❌ GO search failed: {e}")
        return False

    # Test screen data analysis (will likely fail due to missing file)
    print("\n--- Testing screen data analysis ---")
    try:
        result = screen_data_analysis(
            input_file="test_screen_data.csv",  # This file likely doesn't exist
            p_value_threshold=0.05,
            model_name="gpt-4o-mini"
        )
        print(f"✅ Screen analysis successful:")
        print(f"   Status: {result.get('status', 'unknown')}")
        if result.get('status') == 'success':
            final_results = result.get('final_results', {})
            print(f"   Files processed: {len(final_results)}")
        else:
            print(f"   Expected file not found: {result.get('error_message', 'Unknown error')}")
    except Exception as e:
        print(f"⚠️  Screen analysis test failed (expected): {e}")

    # Test TCGA survival analysis (will likely fail due to missing TCGA data)
    print("\n--- Testing TCGA survival analysis ---")
    try:
        result = tcga_survival_analysis(
            cancer_types=["breast cancer"],
            threshold=1.96,
            model_name="gpt-4o-mini"
        )
        print(f"✅ TCGA analysis successful:")
        print(f"   Status: {result.get('status', 'unknown')}")
        if result.get('status') == 'success':
            final_results = result.get('final_results', {})
            for cancer_code, genes in final_results.items():
                print(f"   Cancer '{cancer_code}': Found {len(genes)} genes")
        else:
            print(f"   Expected TCGA data not available: {result.get('error_message', 'Unknown error')}")
    except Exception as e:
        print(f"⚠️  TCGA analysis test failed (expected): {e}")

    # Test drug gene network search (will likely fail due to missing graph file)
    print("\n--- Testing drug gene network search ---")
    try:
        result = drug_gene_network_search(
            drug_queries=["aspirin"],
            model_name="gpt-4o-mini"
        )
        print(f"✅ Drug network search successful:")
        print(f"   Status: {result.get('status', 'unknown')}")
        if result.get('status') == 'success':
            final_results = result.get('final_results', {})
            for drug, genes in final_results.items():
                print(f"   Drug '{drug}': Found {len(genes)} gene associations")
        else:
            print(f"   Expected graph file not available: {result.get('error_message', 'Unknown error')}")
    except Exception as e:
        print(f"⚠️  Drug network search test failed (expected): {e}")

    # Test PubChem drug gene search (should work with real API)
    print("\n--- Testing PubChem drug gene search ---")
    try:
        result = pubchem_drug_gene_search(
            drug_names=["aspirin"],
            model_name="gpt-4o-mini"
        )
        print(f"✅ PubChem search successful:")
        print(f"   Status: {result.get('status', 'unknown')}")
        if result.get('status') == 'success':
            final_results = result.get('final_results', {})
            for drug, genes in final_results.items():
                print(f"   Drug '{drug}': Found {len(genes)} gene associations")
                if genes:
                    print(f"   Sample genes: {[g['gene'] for g in genes[:3]]}")
        else:
            print(f"   PubChem query issue: {result.get('error_message', 'Unknown error')}")
    except Exception as e:
        print(f"❌ PubChem search failed: {e}")
        return False

    # Test new tools with simplified tests (since many may fail due to missing data files)
    print("\n--- Testing new disease/phenotype tools ---")
    try:
        # Test HPO phenotype search
        result = hpo_phenotype_search(phenotype_terms=["ataxia"])
        print(f"✅ HPO search completed: {result.get('status', 'unknown')}")

        # Test OMIM disease search
        result = omim_disease_search(disease_terms=["breast cancer"])
        print(f"✅ OMIM search completed: {result.get('status', 'unknown')}")

        # Test Orphanet rare disease search
        result = orphanet_rare_disease_search(disease_terms=["cystic fibrosis"])
        print(f"✅ Orphanet search completed: {result.get('status', 'unknown')}")
    except Exception as e:
        print(f"⚠️  Disease/phenotype tools test failed (may be expected): {e}")

    print("\n--- Testing new gene relationship tools ---")
    try:
        # Test COXPRES coexpression search
        result = coxpres_coexpression_search(gene_list=["TP53"])
        print(f"✅ COXPRES search completed: {result.get('status', 'unknown')}")

        # Test Ensembl paralog search
        result = ensembl_paralog_search(gene_list=["TP53"])
        print(f"✅ Ensembl paralog search completed: {result.get('status', 'unknown')}")
    except Exception as e:
        print(f"⚠️  Gene relationship tools test failed (may be expected): {e}")

    print("\n--- Testing new pathway tools ---")
    try:
        # Test GSEA hallmark search
        result = gsea_hallmark_search(pathway_terms=["apoptosis"])
        print(f"✅ GSEA search completed: {result.get('status', 'unknown')}")

        # Test WikiPathways search
        result = wikipathways_search(pathway_terms=["cell cycle"])
        print(f"✅ WikiPathways search completed: {result.get('status', 'unknown')}")

        # Test Reactome pathway search
        result = reactome_pathway_search(pathway_terms=["DNA repair"])
        print(f"✅ Reactome search completed: {result.get('status', 'unknown')}")
    except Exception as e:
        print(f"⚠️  Pathway tools test failed (may be expected): {e}")

    print("\n--- Testing new clinical tools ---")
    try:
        # Test cancer biomarkers search
        result = cancer_biomarkers_search(cancer_types=["lung cancer"])
        print(f"✅ Cancer biomarkers search completed: {result.get('status', 'unknown')}")

        # Test ClinGen search
        result = clingen_search(disease_terms=["autism"])
        print(f"✅ ClinGen search completed: {result.get('status', 'unknown')}")

        # Test Gene2Phenotype search
        result = gene2phenotype_search(disease_terms=["epilepsy"])
        print(f"✅ Gene2Phenotype search completed: {result.get('status', 'unknown')}")

        # Test Gene Burden search
        result = gene_burden_search(phenotype_terms=["height"])
        print(f"✅ Gene Burden search completed: {result.get('status', 'unknown')}")

        # Test IntOGen search
        result = intogen_search(cancer_types=["breast cancer"])
        print(f"✅ IntOGen search completed: {result.get('status', 'unknown')}")
    except Exception as e:
        print(f"⚠️  Clinical tools test failed (may be expected): {e}")

    print("\n--- Testing new variant tools ---")
    try:
        # Test UniProt variants search
        result = uniprot_variants_search(disease_terms=["Alzheimer"])
        print(f"✅ UniProt variants search completed: {result.get('status', 'unknown')}")

        # Test ClinVar search
        result = clinvar_search(disease_terms=["diabetes"])
        print(f"✅ ClinVar search completed: {result.get('status', 'unknown')}")
    except Exception as e:
        print(f"⚠️  Variant tools test failed (may be expected): {e}")

    print("\n--- Testing new cellular component tools ---")
    try:
        # Test GOCC cellular component search
        result = gocc_cellular_component_search(component_terms=["mitochondria"])
        print(f"✅ GOCC search completed: {result.get('status', 'unknown')}")
    except Exception as e:
        print(f"⚠️  Cellular component tools test failed (may be expected): {e}")

    return True


def create_test_agent():
    """Create a test agent with virtual screening tools"""
    print("\n" + "=" * 50)
    print("🤖 CREATING TEST AGENT")
    print("=" * 50)

    try:
        # Create model instance
        model = OpenAIServerModel(
            model_id="openai/gpt-4o-mini",
            api_base="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            temperature=0.1
        )
        print("✅ Model instance created")

        # Create agent with virtual screening tools
        all_tools = [
            # Enzyme tools
            run_mmseqs_search, parse_mmseqs_search_results, run_ephod_prediction,
            run_boltz_protein_structure_prediction, run_clean_ec_prediction, run_catapro_prediction,
            run_prime_ogt_prediction, run_chroma_redesign, run_iqtree_reconstruct_phylogenetic_trees,
            run_ligandmpnn_redesign

        ]
        agent = ToolCallingAgent(
            tools=all_tools,
            model=model,
            max_steps=5,
            name="virtual_screening_agent",
            description="Agent for testing virtual screening tools"
        )
        print(f"✅ Agent created with {len(all_tools)} tools (original + new tools)")

        return agent

    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return None


def test_agent_tool_usage(agent):
    """Test agent using the virtual screening tools"""
    print("\n" + "=" * 50)
    print("🔬 TESTING AGENT TOOL USAGE")
    print("=" * 50)
    query_path = "/home/ubuntu/agents/agent_outputs/enzyme_tool_debug/query.fasta"
    db_path = "/home/ubuntu/agents/agent_outputs/enzyme_tool_debug/db.fasta"
    result_path = "/home/ubuntu/agents/agent_outputs/enzyme_tool_debug/results.m8"
    tmp_dir = "/home/ubuntu/agents/agent_outputs/enzyme_tool_debug/tmp_dir"

    input_fasta = "/home/ubuntu/agents/EpHod/example/test_seq.fasta"
    store_dir = "/home/ubuntu/agents/EpHod/example/"
    csv_name = "output.csv"

    bolt_output = "/home/ubuntu/agents/agent_outputs/enzyme_tool_debug/boltz_output"

    smiles_path = "/home/ubuntu/agents/agent_outputs/enzyme_tool_debug/smiles"

    pdb_path="/home/ubuntu/agents/agent_outputs/enzyme_tool_debug/fold_1194nla_model_0.cif"

    msa_path = "/home/ubuntu/agents/agent_outputs/enzyme_tool_debug/result_msa"
    test_cases = [
        # {
        #     "name": "MMseqs Search Test",
        #     "task": f"Search protein sequence {query_path} against sequence database {db_path}"
        #             f" and save the output file in {result_path} and the temporary directory is {tmp_dir}."
        #             f"Output the hit sequence.",
        #     "expected_tool": "run_mmseqs_search"
        # },
        # {
        #     "name": "Optimal pH prediction Test",
        #     "task": f"Predict the optimal pH of the sequence in {input_fasta}, and "
        #             f"store in directory {store_dir} with name {csv_name}",
        #     "expected_tool": "run_ephod_prediction"
        # },
        # {
        #     "name": "Protein structure prediction Test",
        #     "task": f"Predict the 3D structure of protein {query_path}",
        #     "expected_tool": "run_boltz_protein_structure_prediction"
        # },
        # {
        #     "name": "Predict EC number test",
        #     "task": f"Predict the EC number of enzyme {query_path}",
        #     "expected_tool": "run_clean_ec_prediction"
        # },
        # {
        #     "name": "Predict enzyme kinetic parameters test",
        #     "task": f"Predict the kcat of enzyme {query_path} with molecules {smiles_path}",
        #     "expected_tool": "run_catapro_prediction"
        # },
        # {
        #     "name": "Predict enzyme optimal growth temperature test",
        #     "task": f"Predict the optimal growth temperature enzyme {query_path}",
        #     "expected_tool": "run_prime_ogt_prediction"
        # },
        {
            "name": "Redesign enzyme test",
            "task": f"Using LigandMPNN to redesign the enzyme {pdb_path}",
            "expected_tool": "run_ligandmpnn_redesign"
        },
        # {
        #     "name": "Reconstruct phylogenomic tree test",
        #     "task": f"Reconstruct the phylogenomic tree for {msa_path}",
        #     "expected_tool": "run_iqtree_reconstruct_phylogenetic_trees"
        # },
        # {
        #     "name": "Reconstruct phylogenomic tree test",
        #     "task": f"Reconstruct the phylogenomic tree for {msa_path}",
        #     "expected_tool": "run_iqtree_reconstruct_phylogenetic_trees"
        # },

    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        print(f"Task: {test_case['task']}")

        try:
            # Run the agent task
            response = agent.run(test_case['task'], reset=True)

            # Check if response is successful
            if response:
                print(f"✅ Agent response received")
                print(f"   Response type: {type(response)}")

                # Try to extract meaningful information
                if isinstance(response, str):
                    response_preview = response[:200] + "..." if len(response) > 200 else response
                    print(f"   Response preview: {response_preview}")
                elif hasattr(response, 'output'):
                    print(f"   Response output: {response.output}")
                else:
                    print(f"   Response content: {str(response)[:200]}...")

                results.append({
                    "test": test_case['name'],
                    "status": "success",
                    "response": str(response)[:500]  # Truncate for display
                })
            else:
                print(f"❌ No response received")
                results.append({
                    "test": test_case['name'],
                    "status": "no_response",
                    "response": None
                })

        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                "test": test_case['name'],
                "status": "error",
                "error": str(e)
            })

    return results


def main():
    """Main test execution"""
    print("🧬 ENZYME TOOLS TEST SUITE")
    print("This script tests the integration of enzyme tools with the manager agent")

    # Run all tests
    agent = create_test_agent()
    if agent:
        agent_results = test_agent_tool_usage(agent)
        print(agent_results)
    # Generate report
    # overall_pass = generate_test_report(env_ok, tools_ok, integration_ok, agent_results)

    # Exit with appropriate code
    # sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
