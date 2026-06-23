import h5py
import numpy as np
import csv
import json
from flask import Blueprint, jsonify
from sklearn.metrics.pairwise import cosine_similarity

CACHE_PATH = "/Users/sanvik/terracotta_lincs_cache.npy"
META_PATH = "/Users/sanvik/terracotta_lincs_meta.json"
FAILURES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terracotta_failures_100.csv")

serendipity_bp = Blueprint('serendipity', __name__)

# ── LOAD CACHE (6MB — totally safe) ──────────────────────────────────────────
print("Loading LINCS cache...")
_cache_available = False
try:
    _cache = np.load(CACHE_PATH)
    with open(META_PATH, 'r') as f:
        _meta = json.load(f)
    _cache_available = True
    if _cache_available:
        print(f"✓ Cache loaded: {_cache.shape[0]:,} perturbations x {_cache.shape[1]} genes")
    else:
        print("⚠ Cache not available — serendipity engine disabled")
except Exception as e:
    print(f"⚠ Cache not available: {e}")
    _cache = None
    _meta = {"symbols": [], "pert_ids_sample": []}

_symbols = _meta.get('symbols', [])
_symbol_to_col = {s: i for i, s in enumerate(_symbols)}
if _symbols:
    print(f"✓ Genes available: {_symbols}")

# ── LOAD FAILURES ─────────────────────────────────────────────────────────────
def load_failures():
    cases = []
    try:
        with open(FAILURES_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cases.append(row)
    except Exception as e:
        print(f"Warning: {e}")
    return cases

FAILURE_CASES = load_failures()
print(f"✓ Loaded {len(FAILURE_CASES)} failure cases")

# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────
def get_gene_profile(gene_symbol):
    """Get the perturbation profile for a gene from cache."""
    col = _symbol_to_col.get(gene_symbol.upper())
    if col is None:
        return None
    return _cache[:, col]

def find_correlated_genes(gene_symbol, n=10):
    """
    Find genes most correlated with query gene
    across 50,000 perturbations.
    Pure numpy — instant, RAM safe.
    """
    query = get_gene_profile(gene_symbol)
    if query is None:
        return []

    results = []
    query_norm = (query - np.mean(query)) / (np.std(query) + 1e-8)

    for symbol in _symbols:
        if symbol == gene_symbol:
            continue
        other = _cache[:, _symbol_to_col[symbol]]
        other_norm = (other - np.mean(other)) / (np.std(other) + 1e-8)
        corr = float(np.mean(query_norm * other_norm))
        results.append({"gene": symbol, "correlation": round(corr, 4)})

    results.sort(key=lambda x: abs(x['correlation']), reverse=True)
    return results[:n]

def get_top_perturbations(gene_symbol, n=10):
    """Find perturbations that most strongly affect this gene."""
    profile = get_gene_profile(gene_symbol)
    if profile is None:
        return []

    abs_profile = np.abs(profile)
    top_indices = np.argsort(abs_profile)[-n:][::-1]

    pert_ids_sample = _meta.get('pert_ids_sample', [])
    results = []
    for idx in top_indices:
        z = float(profile[idx])
        pert_id = pert_ids_sample[idx] if idx < len(pert_ids_sample) else f"pert_{idx}"
        parts = pert_id.split('_')
        cell_line = parts[1] if len(parts) > 1 else "unknown"
        results.append({
            "perturbation_id": pert_id,
            "cell_line": cell_line,
            "z_score": round(z, 3),
            "direction": "UP" if z > 0 else "DOWN"
        })
    return results

def find_failure_connections(correlated_genes):
    """Cross-reference correlated genes with failure ontology."""
    connections = []
    seen_targets = set()
    symbols = [g['gene'] for g in correlated_genes]

    for case in FAILURE_CASES:
        target = case.get('target', '').upper()
        target_key = target + case.get('indication', '')
        if target_key in seen_targets:
            continue

        for symbol in symbols:
            # Strict match — symbol must match target field specifically
            if symbol.upper() == target or target == symbol.upper():
                cat = case.get('ontology_category', '')
                code = cat.split(' - ')[0].split()[0].strip() if cat else ''
                corr_val = next(
                    (g['correlation'] for g in correlated_genes
                     if g['gene'] == symbol), 0
                )
                connections.append({
                    "connected_via_gene": symbol,
                    "correlation_strength": corr_val,
                    "failed_target": case.get('target', ''),
                    "indication": case.get('indication', ''),
                    "failure_category": code,
                    "mechanism": case.get('mechanistic_assessment', '')[:200],
                    "serendipitous_note": case.get('notes', '')[:150]
                })
                seen_targets.add(target_key)
                break

    connections.sort(key=lambda x: abs(x['correlation_strength']), reverse=True)
    return connections[:5]

def generate_experimental_suggestions(gene, correlated, connections):
    """
    Generate experimental design suggestions based on
    co-regulation patterns and failure history.
    This is the AI-native experimental design layer.
    """
    suggestions = []

    # Suggestion 1: Based on correlated genes
    if correlated:
        top_corr = correlated[0]
        direction = "co-upregulated" if top_corr['correlation'] > 0 else "inversely regulated"
        suggestions.append({
            "type": "co_regulation_experiment",
            "priority": "HIGH",
            "suggestion": f"Design a dual-perturbation experiment targeting both {gene} and {top_corr['gene']} simultaneously — they are strongly {direction} (r={top_corr['correlation']}) across 50,000 chemical perturbations, suggesting shared pathway biology.",
            "rationale": f"Strong co-regulation between {gene} and {top_corr['gene']} implies functional coupling. Independent perturbation of each may miss emergent phenotypes only visible when both are modulated."
        })

    # Suggestion 2: Based on failure connections
    if connections:
        top_conn = connections[0]
        suggestions.append({
            "type": "failure_informed_design",
            "priority": "HIGH",
            "suggestion": f"Before committing to {gene} as a therapeutic target, design a pathway compensation assay — specifically testing whether {top_conn['connected_via_gene']}-mediated pathways can rescue the phenotype when {gene} is inhibited.",
            "rationale": f"Historical failure data: {top_conn['failed_target']} failed via {top_conn['failure_category']} in {top_conn['indication']}. Given co-regulation between {gene} and {top_conn['connected_via_gene']}, this failure mode is a translational risk."
        })

    # Suggestion 3: Model system recommendation
    if connections:
        failure_cats = [c['failure_category'] for c in connections]
        if 'C2' in failure_cats or 'C3' in failure_cats:
            suggestions.append({
                "type": "model_system_warning",
                "priority": "CRITICAL",
                "suggestion": f"Do NOT use standard immunodeficient xenograft models for {gene} validation — co-regulated genes have documented C2/C3 failures indicating model non-representativeness.",
                "rationale": "Disease model non-representativeness (C2) or immunodeficient model failure (C3) in co-regulated targets strongly predicts the same failure mode will affect this target."
            })
        elif 'B3' in failure_cats:
            suggestions.append({
                "type": "redundancy_assay",
                "priority": "HIGH",
                "suggestion": f"Run a pathway redundancy screen before advancing {gene} — test 3-5 parallel pathway inhibitors simultaneously to identify compensatory escape mechanisms.",
                "rationale": "Co-regulated genes show B3 (pathway redundancy) failures. This pattern frequently propagates to targets sharing co-regulation networks."
            })

    return suggestions

# ── ROUTES ────────────────────────────────────────────────────────────────────
@serendipity_bp.route("/serendipity/<gene>", methods=["GET"])
def serendipity(gene):
    gene = gene.upper().strip()
    print(f"\n[Serendipity] {gene}")

    if not _cache_available:
        return jsonify({
            "gene": gene,
            "error": "LINCS cache not available in this deployment",
            "message": "Serendipity engine requires local LINCS data. Core scoring engine is fully operational."
        }), 503

    if gene not in _symbol_to_col:
        return jsonify({
            "gene": gene,
            "error": f"{gene} not in cache",
            "available_genes": sorted(_symbols)
        }), 404

    # Core analysis
    correlated = find_correlated_genes(gene, n=10)
    top_perts = get_top_perturbations(gene, n=8)
    connections = find_failure_connections(correlated)
    suggestions = generate_experimental_suggestions(gene, correlated, connections)

    serendipity_score = min(100,
        len(connections) * 20 +
        len(suggestions) * 15 +
        (10 if correlated else 0)
    )

    return jsonify({
        "gene": gene,
        "serendipity_score": serendipity_score,
        "perturbations_analyzed": _cache.shape[0],
        "summary": (
            f"Analyzed {gene} across {_cache.shape[0]:,} chemical perturbations. "
            f"Found {len(correlated)} co-regulated genes, "
            f"{len(connections)} failure ontology connections, "
            f"and generated {len(suggestions)} experimental design suggestions."
        ),
        "correlated_genes": correlated,
        "top_perturbing_conditions": top_perts,
        "failure_ontology_connections": connections,
        "experimental_design_suggestions": suggestions,
        "serendipitous_insight": (
            f"{gene} shares co-regulation patterns with "
            f"{', '.join(g['gene'] for g in correlated[:3])}. "
            f"{len(connections)} of these connections map to documented "
            f"translational failures — informing experimental design before "
            f"a single experiment is run."
            if connections else
            f"{gene} co-regulates with {', '.join(g['gene'] for g in correlated[:3])} "
            f"across 50,000 perturbations. No failure ontology overlaps — "
            f"this target may represent novel unexplored biology."
        )
    })

@serendipity_bp.route("/serendipity/status", methods=["GET"])
def status():
    return jsonify({
        "status": "operational",
        "lincs_cache_available": _cache_available,
        "cached_genes": len(_symbols) if _cache_available else 0,
        "perturbations_sampled": _cache.shape[0] if _cache_available else 0,
        "failure_cases": len(FAILURE_CASES),
        "ram_usage_mb": round(_cache.nbytes / 1024 / 1024, 1) if _cache_available else 0,
        "available_genes": sorted(_symbols) if _cache_available else []
    })