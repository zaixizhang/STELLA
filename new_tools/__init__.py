"""
PubMed Search Tool Package

This package provides a simple interface to search PubMed for scientific articles
using the NCBI E-utilities API.
"""

from .pubmed_search import pubmed_search

__all__ = ['pubmed_search']
__version__ = '1.0.0'