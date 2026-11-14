# -*- coding: utf-8 -*-
"""
New tool: query_pubmed
- Queries NCBI PubMed E-utilities for literature evidence.
- Signature: query_pubmed(args: str) -> str
- Input args: a single natural-language string; if formatted as 'gene=SYMBOL; context=TEXT', parse gene and context.
- Behavior:
  * Build a PubMed esearch term focused on human, NK cell, melanoma/A375 context, using the provided args string intelligently.
  * Use esearch.fcgi (db=pubmed, retmode=json, retmax=25) to get hit count and PMIDs.
  * If PMIDs exist, call efetch.fcgi (db=pubmed, rettype=abstract, retmode=text, retmax=5) to retrieve top abstracts' titles (up to 5).
- Output: a compact JSON-like string including:
  { "query": ..., "count": N, "pmids": [.. up to 10], "titles": [.. up to 5] }
- Robust error handling with timeouts (10s), retries (2), and graceful fallbacks.
- Dependencies: uses requests (if available) and urllib.parse; falls back to urllib.request if requests is missing.
- Keeps response under ~4KB by truncating long fields.
"""
from __future__ import annotations

import json
import re
import time
from typing import List, Dict, Optional

try:
    from smolagents import tool  # type: ignore
except Exception:  # pragma: no cover
    # Fallback no-op decorator if smolagents is unavailable
    def tool(func=None, **kwargs):  # type: ignore
        def wrapper(f):
            return f
        if func is None:
            return wrapper
        return func

try:
    import requests  # type: ignore
except Exception:
    requests = None  # We'll fall back to urllib

from urllib.parse import quote_plus

# Fallback HTTP using urllib if requests is not available
try:
    import urllib.request as urllib_request
    import urllib.error as urllib_error
except Exception:  # pragma: no cover
    urllib_request = None
    urllib_error = None

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_TIMEOUT = 10  # seconds
MAX_RETRIES = 2  # in addition to the first try (total attempts = 1 + MAX_RETRIES)

# NCBI polite usage identifiers (optional but recommended)
NCBI_TOOL = "tool_creation_agent"
NCBI_EMAIL = "tool_creation_agent@example.com"


def _http_get_json(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[Dict]:
    """GET JSON with retries and timeouts. Returns dict or None."""
    attempts = 1 + MAX_RETRIES
    for attempt in range(1, attempts + 1):
        try:
            if requests is not None:
                r = requests.get(url, timeout=timeout, headers={"User-Agent": f"{NCBI_TOOL} (mailto:{NCBI_EMAIL})"})
                r.raise_for_status()
                return r.json()
            else:
                if urllib_request is None:
                    raise RuntimeError("No HTTP client available")
                req = urllib_request.Request(url, headers={"User-Agent": f"{NCBI_TOOL} (mailto:{NCBI_EMAIL})"})
                with urllib_request.urlopen(req, timeout=timeout) as resp:
                    data = resp.read().decode("utf-8", errors="replace")
                return json.loads(data)
        except Exception:
            if attempt < attempts:
                time.sleep(0.7 * attempt)
            else:
                return None
    return None


def _http_get_text(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[str]:
    """GET text with retries and timeouts. Returns text or None."""
    attempts = 1 + MAX_RETRIES
    for attempt in range(1, attempts + 1):
        try:
            if requests is not None:
                r = requests.get(url, timeout=timeout, headers={"User-Agent": f"{NCBI_TOOL} (mailto:{NCBI_EMAIL})"})
                r.raise_for_status()
                r.encoding = r.encoding or "utf-8"
                return r.text
            else:
                if urllib_request is None:
                    raise RuntimeError("No HTTP client available")
                req = urllib_request.Request(url, headers={"User-Agent": f"{NCBI_TOOL} (mailto:{NCBI_EMAIL})"})
                with urllib_request.urlopen(req, timeout=timeout) as resp:
                    data = resp.read().decode("utf-8", errors="replace")
                return data
        except Exception:
            if attempt < attempts:
                time.sleep(0.7 * attempt)
            else:
                return None
    return None


def _sanitize_text(s: str, max_len: int) -> str:
    if s is None:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _parse_args(args: str) -> Dict[str, Optional[str]]:
    """Parse args which may be 'gene=SYMBOL; context=TEXT' or free text."""
    if not args:
        return {"gene": None, "context": None, "free": None}
    gene = None
    context = None
    m_gene = re.search(r"(?i)\bgene\s*=\s*([^;]+)", args)
    if m_gene:
        gene = m_gene.group(1).strip()
    m_ctx = re.search(r"(?i)\bcontext\s*=\s*([^;]+)", args)
    if m_ctx:
        context = m_ctx.group(1).strip()
    if gene or context:
        return {"gene": gene or None, "context": context or None, "free": None}
    # Otherwise treat entire string as free text context to AND
    return {"gene": None, "context": None, "free": args.strip()}


def _build_query(parsed: Dict[str, Optional[str]]) -> str:
    """Build a PubMed query focused on human + NK cells + melanoma/A375, extended with inputs."""
    base_parts = [
        "(Humans[MeSH Terms] OR human[Title/Abstract] OR humans[Title/Abstract])",
        '("Natural Killer Cells"[MeSH Terms] OR "NK cell"[Title/Abstract] OR "NK cells"[Title/Abstract] OR NK-cell*[Title/Abstract])',
        '(melanoma[MeSH Terms] OR melanoma[Title/Abstract] OR A375[Title/Abstract])',
    ]

    # Gene clause
    gene = parsed.get("gene")
    if gene:
        gene_clean = re.sub(r"[^A-Za-z0-9\-_/]", " ", gene).strip()
        if gene_clean:
            gene_clause = f'({gene_clean}[Gene Symbol] OR {gene_clean}[Title/Abstract] OR "{gene_clean} gene"[All Fields])'
            base_parts.append(gene_clause)

    # Context or free text clause
    ctx = parsed.get("context")
    free = parsed.get("free")
    extra = ctx or free
    if extra:
        extra = _sanitize_text(extra, 200)
        # Spread context across common fields
        extra_clause = f'("{extra}"[Title/Abstract] OR "{extra}"[All Fields])'
        base_parts.append(extra_clause)

    term = " AND ".join(base_parts)
    # Trim if overly long
    if len(term) > 900:
        term = term[: 897] + "..."
    return term


def _parse_titles_from_efetch_text(txt: str, max_titles: int = 5) -> List[str]:
    """Extract titles from efetch text (rettype=abstract, retmode=text)."""
    if not txt:
        return []
    titles: List[str] = []
    # Each record usually contains lines starting with 'TI  -'
    current_title_lines: List[str] = []
    for line in txt.splitlines():
        if line.startswith("TI  -"):
            # Flush previous
            if current_title_lines:
                t = " ".join(l.strip() for l in current_title_lines)
                t = _sanitize_text(t, 300)
                if t:
                    titles.append(t)
                current_title_lines = []
            # Start new
            current_title_lines = [line[5:].strip()]
        elif current_title_lines:
            # Continuation lines are typically indented (6 spaces) without a field tag
            if re.match(r"^[ \t]{6,}\\S", line) and not re.match(r"^[A-Z]{2}  -", line):
                current_title_lines.append(line.strip())
            else:
                # End of title block
                t = " ".join(l.strip() for l in current_title_lines)
                t = _sanitize_text(t, 300)
                if t:
                    titles.append(t)
                current_title_lines = []
        if len(titles) >= max_titles:
            break
    # Edge case: file ended while accumulating title
    if len(titles) < max_titles and current_title_lines:
        t = " ".join(l.strip() for l in current_title_lines)
        t = _sanitize_text(t, 300)
        if t:
            titles.append(t)
    # Final limit and sanitize
    return [_sanitize_text(t, 300) for t in titles[:max_titles]]


@tool
def query_pubmed(args: str) -> str:
    """Query PubMed E-utilities for literature evidence.

    Args:
        args (str): Natural-language query string. Accepts formats like
            "gene=SYMBOL; context=TEXT" or free-text keywords. The tool will
            always focus on humans, NK cells, and melanoma/A375 context, and
            intelligently combine with the provided terms.

    Returns:
        str: A compact JSON-like string with keys:
            - "query": the final PubMed search term used (possibly truncated)
            - "count": integer number of hits from esearch
            - "pmids": list of up to 10 PMIDs from esearch
            - "titles": list of up to 5 titles parsed from efetch abstract text
    """
    try:
        parsed = _parse_args(args or "")
        term = _build_query(parsed)

        # ESearch: get count + PMIDs
        esearch_url = (
            f"{NCBI_BASE}/esearch.fcgi?db=pubmed&retmode=json&retmax=25&sort=relevance"
            f"&tool={quote_plus(NCBI_TOOL)}&email={quote_plus(NCBI_EMAIL)}&term={quote_plus(term)}"
        )
        es = _http_get_json(esearch_url)
        count = 0
        idlist: List[str] = []
        if es and isinstance(es, dict):
            try:
                sr = es.get("esearchresult", {})
                count = int(sr.get("count", 0))
                idlist = list(sr.get("idlist", []) or [])
            except Exception:
                count = 0
                idlist = []

        # EFetch: fetch up to 5 abstracts in text mode to parse titles
        titles: List[str] = []
        top_ids = idlist[:5]
        if top_ids:
            id_param = ",".join(top_ids)
            efetch_url = (
                f"{NCBI_BASE}/efetch.fcgi?db=pubmed&id={quote_plus(id_param)}&rettype=abstract&retmode=text&retmax=5"
                f"&tool={quote_plus(NCBI_TOOL)}&email={quote_plus(NCBI_EMAIL)}"
            )
            txt = _http_get_text(efetch_url)
            if txt:
                titles = _parse_titles_from_efetch_text(txt, max_titles=5)

        # Build response, truncating to keep under ~4KB
        resp = {
            "query": _sanitize_text(term, 1000),
            "count": count,
            "pmids": idlist[:10],
            "titles": [_sanitize_text(t, 300) for t in titles[:5]],
        }
        out = json.dumps(resp, ensure_ascii=False, separators=(",", ":"))
        if len(out.encode("utf-8")) > 3800:
            # Try dropping titles if too large
            resp["titles"] = resp.get("titles", [])[:3]
            out = json.dumps(resp, ensure_ascii=False, separators=(",", ":"))
        if len(out.encode("utf-8")) > 3800:
            # As a last resort, shorten query further
            resp["query"] = _sanitize_text(resp["query"], 400)
            out = json.dumps(resp, ensure_ascii=False, separators=(",", ":"))
        return out
    except Exception:
        fallback = {
            "query": _sanitize_text(str(args or ""), 400),
            "count": 0,
            "pmids": [],
            "titles": [],
        }
        return json.dumps(fallback, ensure_ascii=False, separators=(",", ":"))
