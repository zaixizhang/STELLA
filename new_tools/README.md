# NK Gene Analyzer Tool

## Overview

The NK Gene Analyzer is a sophisticated biomedical analysis tool designed to analyze genes for their role in NK (Natural Killer) cell-mediated killing resistance in A375 melanoma cells. This tool queries multiple biomedical databases, computes evidence-based scores, and provides comprehensive analysis results.

## Features

- **Multi-Database Integration**: Queries 10+ biomedical databases in parallel
- **Evidence-Based Scoring**: Generates literature, pathway, and clinical scores
- **NK Cell Focus**: Specialized prompts targeting NK cell resistance mechanisms
- **Parallel Processing**: Efficient execution using ThreadPoolExecutor
- **Comprehensive Output**: Detailed results with evidence summaries and metadata

## Tool Files

### 1. `nk_gene_analyzer.py`
The main tool implementation with simulated database queries for testing and demonstration.

**Key Features:**
- Full smolagents @tool decorator implementation
- Simulated database responses for reliable testing
- Comprehensive scoring algorithm
- Error handling and logging

### 2. `enhanced_nk_gene_analyzer.py` 
Enhanced version with optional real database integration capability.

**Additional Features:**
- Can switch between simulated and real database queries
- Enhanced evidence extraction and scoring
- Detailed analysis metadata
- More granular scoring algorithms

### 3. `test_nk_gene_analyzer.py`
Comprehensive test suite to validate tool functionality.

**Test Coverage:**
- Basic functionality testing
- Error handling validation
- Scoring logic verification
- Database coverage confirmation
- Real-world gene testing

## Usage

### Basic Usage

```python
from nk_gene_analyzer import nk_gene_analyzer

# Define genes to analyze
genes = [
    {'gene_symbol': 'HLA-A', 'original_rank': 1},
    {'gene_symbol': 'B2M', 'original_rank': 5},
    {'gene_symbol': 'TAP1', 'original_rank': 10}
]

# Analyze genes
results = nk_gene_analyzer(genes)

# Access results
for gene_result in results['results']:
    print(f"Gene: {gene_result['gene_symbol']}")
    print(f"Combined Score: {gene_result['combined_evidence_score']}")
    print(f"Evidence: {gene_result['evidence_summary']}")
```

### Enhanced Usage

```python
from enhanced_nk_gene_analyzer import nk_gene_analyzer_enhanced

# Analyze with enhanced features
results = nk_gene_analyzer_enhanced(
    genes, 
    use_real_databases=False  # Set True for real database queries
)

# Access enhanced metadata
print(f"Analysis time: {results['metadata']['total_analysis_time']} seconds")
print(f"Database mode: {results['metadata']['database_mode']}")
```

## Input Format

The tool expects a list of dictionaries, each containing:

```python
{
    'gene_symbol': str,     # Gene symbol (e.g., 'HLA-A')
    'original_rank': int    # Original ranking position (1-200)
}
```

## Output Format

Returns a dictionary with:

```python
{
    'results': [
        {
            'gene_symbol': str,
            'original_rank': int,
            'literature_score': float,      # 0-10
            'pathway_score': float,         # 0-10
            'clinical_score': float,        # 0-10
            'rank_score': float,           # 0-10
            'combined_evidence_score': float, # 0-10
            'evidence_summary': str,
            'database_hits': dict,
            'total_hits': int,
            'analysis_metadata': dict      # Enhanced version only
        }
    ],
    'metadata': dict  # Enhanced version only
}
```

## Scoring Algorithm

### Individual Scores (0-10 scale)

1. **Literature Score** (40% weight)
   - Based on PubMed and GEO database hits
   - 0 hits = 0, 1 hit = 2, 2-3 hits = 4, 4-5 hits = 6, 6+ hits = 8, 10+ hits = 10

2. **Pathway Score** (30% weight)
   - Based on KEGG, Reactome, Ensembl, UniProt, STRING database hits
   - Similar scoring scale as literature

3. **Clinical Score** (20% weight)
   - Based on OpenTargets, ClinVar, cBioPortal database hits
   - Similar scoring scale as literature

4. **Rank Score** (10% weight)
   - Based on original gene ranking: 10 × (1 - (rank-1)/199)

### Combined Evidence Score

```
Combined Score = 0.4 × Literature + 0.3 × Pathway + 0.2 × Clinical + 0.1 × Rank
```

## Database Coverage

The tool queries these biomedical databases:

1. **PubMed** - Literature search
2. **KEGG** - Pathway analysis
3. **OpenTargets** - Drug targets and disease associations
4. **ClinVar** - Clinical variant information
5. **Ensembl** - Genomic information
6. **Reactome** - Biological pathways
7. **cBioPortal** - Cancer genomics data
8. **UniProt** - Protein information
9. **STRING** - Protein interactions
10. **GEO** - Gene expression data

## NK Cell Resistance Focus

The tool specifically targets these mechanisms:

- **MHC Class I** presentation and downregulation
- **IFN-γ signaling** pathways
- **NK cell ligands** and receptors
- **Apoptosis resistance** mechanisms
- **Cell adhesion** and migration
- **Immune evasion** strategies

## Testing

Run the comprehensive test suite:

```bash
python test_nk_gene_analyzer.py
```

The test suite validates:
- Basic functionality
- Error handling
- Scoring calculations
- Database coverage
- Real-world gene analysis

## Dependencies

- `smolagents` - For @tool decorator
- `concurrent.futures` - For parallel processing
- `logging` - For operation tracking
- Standard Python libraries (time, typing, asyncio)

## Performance

- **Parallel Execution**: All database queries run simultaneously
- **Timeout Protection**: 5-minute timeout for database queries
- **Error Resilience**: Continues analysis even if some databases fail
- **Efficient Processing**: Optimized for analyzing multiple genes

## Known NK Resistance Genes

The tool is optimized for analyzing genes known to be involved in NK cell resistance:

- **HLA genes**: HLA-A, HLA-B, HLA-C
- **Antigen processing**: B2M, TAP1, TAP2, PSMB8, PSMB9
- **IFN signaling**: IFNG, IRF1, STAT1, NLRC5
- **Immune checkpoints**: CD274, PDCD1LG2, CTLA4
- **Other mechanisms**: CALR, IDO1

## Error Handling

The tool includes robust error handling:

- Invalid input format validation
- Database query timeout protection
- Partial result generation on failures
- Comprehensive error logging
- Graceful degradation capabilities

## Future Enhancements

Potential improvements for production use:

1. **Real Database Integration**: Connect to actual biomedical APIs
2. **Caching System**: Store results to avoid redundant queries
3. **Batch Processing**: Optimize for large gene sets
4. **Custom Scoring**: Allow user-defined scoring weights
5. **Export Options**: Support multiple output formats
6. **Visualization**: Generate plots and charts
7. **API Integration**: RESTful API wrapper
8. **Database Updates**: Track database version changes