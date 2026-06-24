import os
import json
import time
import requests
from flask import Blueprint, jsonify
from groq import Groq

discovery_bp = Blueprint('discovery', __name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ── SEARCH SOURCES ────────────────────────────────────────────────────────────

def search_europepmc(gene, max_results=10):
    """
    Europe PMC indexes older literature than PubMed
    including pre-1966 papers and grey literature.
    """
    results = []
    try:
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            "query": f"{gene} AND (FIRST_PDATE:[1940 TO 1990])",
            "format": "json",
            "pageSize": max_results,
            "resultType": "core",
            "sort": "CITED desc"
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        for item in data.get("resultList", {}).get("result", []):
            results.append({
                "source": "Europe PMC (pre-1990)",
                "title": item.get("title", ""),
                "authors": item.get("authorString", ""),
                "year": item.get("pubYear", ""),
                "journal": item.get("journalTitle", ""),
                "abstract": item.get("abstractText", "")[:500],
                "pmid": item.get("pmid", ""),
                "url": f"https://europepmc.org/article/MED/{item.get('pmid', '')}"
            })
    except Exception as e:
        print(f"Europe PMC error: {e}")
    return results

def search_pubmed_obscure(gene, max_results=10):
    """
    Search PubMed for low-citation papers from obscure journals
    — the ones nobody reads but that often contain real signal.
    """
    results = []
    try:
        # Search with esearch
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": f"{gene}[tiab] AND (1960[pdat]:1995[pdat])",
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }
        r = requests.get(search_url, params=params, timeout=10)
        ids = r.json().get("esearchresult", {}).get("idlist", [])

        if not ids:
            return results

        # Fetch abstracts
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
            "rettype": "abstract"
        }
        time.sleep(0.5)  # NCBI rate limit
        fr = requests.get(fetch_url, params=fetch_params, timeout=15)

        # Simple XML parsing
        from xml.etree import ElementTree as ET
        root = ET.fromstring(fr.content)

        for article in root.findall(".//PubmedArticle"):
            title_el = article.find(".//ArticleTitle")
            abstract_el = article.find(".//AbstractText")
            year_el = article.find(".//PubDate/Year")
            journal_el = article.find(".//Journal/Title")
            pmid_el = article.find(".//PMID")

            title = title_el.text if title_el is not None else ""
            abstract = abstract_el.text if abstract_el is not None else ""
            year = year_el.text if year_el is not None else ""
            journal = journal_el.text if journal_el is not None else ""
            pmid = pmid_el.text if pmid_el is not None else ""

            if title:
                results.append({
                    "source": "PubMed (pre-1995)",
                    "title": title,
                    "year": year,
                    "journal": journal,
                    "abstract": abstract[:500],
                    "pmid": pmid,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })

    except Exception as e:
        print(f"PubMed obscure error: {e}")
    return results

def search_retraction_watch(gene):
    """
    Search for retracted papers on this target.
    Retracted ≠ wrong. Often means ahead of time,
    politically inconvenient, or methodologically imperfect
    but conceptually correct.
    """
    results = []
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": f"{gene}[tiab] AND retracted[pt]",
            "retmax": 5,
            "retmode": "json"
        }
        r = requests.get(url, params=params, timeout=10)
        ids = r.json().get("esearchresult", {}).get("idlist", [])

        for pmid in ids:
            results.append({
                "source": "Retracted Paper",
                "pmid": pmid,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "note": "Retracted papers may contain valid biological insights despite methodological issues"
            })
    except Exception as e:
        print(f"Retraction search error: {e}")
    return results

def search_biorxiv(gene, max_results=5):
    """Search bioRxiv for preprints — often newer and more speculative."""
    results = []
    try:
        url = f"https://api.biorxiv.org/details/biorxiv/{gene}/0/10/json"
        r = requests.get(url, timeout=10)
        data = r.json()
        for item in data.get("collection", [])[:max_results]:
            results.append({
                "source": "bioRxiv preprint",
                "title": item.get("title", ""),
                "authors": item.get("authors", ""),
                "date": item.get("date", ""),
                "abstract": item.get("abstract", "")[:500],
                "url": f"https://www.biorxiv.org/content/{item.get('doi', '')}"
            })
    except Exception as e:
        print(f"bioRxiv error: {e}")
    return results

# ── AI SYNTHESIS ──────────────────────────────────────────────────────────────

def synthesize_discoveries(gene, all_papers, failure_cases):
    """
    Use Groq to synthesize across all found papers and
    surface serendipitous connections to current biology.
    This is the core invention — AI reading across time.
    """
    # Format papers for prompt
    papers_text = ""
    for i, p in enumerate(all_papers[:8]):
        papers_text += f"""
PAPER {i+1} ({p.get('source', '')}, {p.get('year', '')}):
Title: {p.get('title', '')}
Abstract: {p.get('abstract', '')[:300]}
---"""

    # Format relevant failures
    relevant_failures = [
        c for c in failure_cases
        if gene.lower() in c.get('target', '').lower()
        or gene.lower() in c.get('mechanistic_assessment', '').lower()
    ]
    failures_text = ""
    for f in relevant_failures[:3]:
        failures_text += f"""
TARGET: {f.get('target', '')}
FAILURE: {f.get('mechanistic_assessment', '')[:200]}
CATEGORY: {f.get('ontology_category', '')}
---"""

    prompt = f"""You are Terracotta's serendipity engine analyzing obscure and historical scientific literature.

TARGET GENE: {gene}

HISTORICAL AND OBSCURE PAPERS FOUND:
{papers_text}

KNOWN FAILURE CASES FOR THIS TARGET:
{failures_text if failures_text else "No direct failures documented yet."}

Your task: Identify the most serendipitous, non-obvious connections across these papers that a modern researcher would miss. Focus on:
1. Old findings that predict current failure modes
2. Cross-domain connections (e.g., an immunology finding relevant to oncology)
3. Forgotten hypotheses that deserve reconsideration
4. Vocabulary shifts — where old terms describe modern concepts
5. Any finding that contradicts current consensus and why that matters

Respond ONLY with this JSON:
{{
  "serendipitous_connections": [
    {{
      "connection_type": "<cross_domain|forgotten_hypothesis|contradicts_consensus|predicts_failure|vocabulary_shift>",
      "old_finding": "<what the old paper found>",
      "modern_relevance": "<why this matters for {gene} today>",
      "implication": "<what experiment or decision this should change>",
      "confidence": "<Low|Medium|High>",
      "source_hint": "<paper title or year>"
    }}
  ],
  "most_valuable_insight": "<the single most important thing a researcher studying {gene} should know from this historical record>",
  "recommended_search": "<what obscure source or search term a researcher should pursue next>"
}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a scientific intelligence engine. Find non-obvious connections across historical literature. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break
        return json.loads(raw)
    except Exception as e:
        return {
            "serendipitous_connections": [],
            "most_valuable_insight": f"Could not synthesize: {str(e)}",
            "recommended_search": f"Search Europe PMC for {gene} pre-1990"
        }

# ── ROUTES ────────────────────────────────────────────────────────────────────
@discovery_bp.route("/discover/<gene>", methods=["GET"])
def discover(gene):
    gene = gene.upper().strip()
    print(f"\n[Discovery] {gene}")

    # Load failures
    import csv
    failure_cases = []
    failures_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "terracotta_failures_100.csv"
    )
    try:
        with open(failures_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                failure_cases.append(row)
    except Exception as e:
        print(f"Could not load failures: {e}")

    # Search all sources
    print("  Searching Europe PMC (pre-1990)...")
    old_papers = search_europepmc(gene, max_results=8)

    print("  Searching PubMed (pre-1995)...")
    pubmed_old = search_pubmed_obscure(gene, max_results=8)

    print("  Searching retracted papers...")
    retracted = search_retraction_watch(gene)

    print("  Searching bioRxiv...")
    preprints = search_biorxiv(gene, max_results=5)

    all_papers = old_papers + pubmed_old + preprints

    print(f"  Found {len(all_papers)} papers total")
    print("  Synthesizing with AI...")

    # AI synthesis
    synthesis = synthesize_discoveries(gene, all_papers, failure_cases)

    return jsonify({
        "gene": gene,
        "papers_found": {
            "pre_1990_literature": len(old_papers),
            "pre_1995_pubmed": len(pubmed_old),
            "retracted_papers": len(retracted),
            "preprints": len(preprints),
            "total": len(all_papers)
        },
        "retracted_papers": retracted,
        "historical_papers": (old_papers + pubmed_old)[:5],
        "serendipitous_connections": synthesis.get("serendipitous_connections", []),
        "most_valuable_insight": synthesis.get("most_valuable_insight", ""),
        "recommended_search": synthesis.get("recommended_search", ""),
        "all_papers_raw": all_papers[:10]
    })

@discovery_bp.route("/discover/status", methods=["GET"])
def discover_status():
    return jsonify({
        "status": "operational",
        "sources": [
            "Europe PMC (pre-1990 literature)",
            "PubMed (pre-1995)",
            "Retracted papers database",
            "bioRxiv preprints"
        ]
    })
    