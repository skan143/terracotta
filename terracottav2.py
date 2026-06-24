from flask import Flask, render_template, request, jsonify
from groq import Groq
import json
import csv
import os
from collections import Counter


app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")
print(f"GROQ key loaded: {GROQ_API_KEY[:8]}...")
client = Groq(api_key=GROQ_API_KEY)

# ── CATEGORY DEFINITIONS ──────────────────────────────────────────────────────
CATEGORY_NAMES = {
    "A1": "Off-target pathway active in humans, silent in model",
    "A2": "Drug mechanism mischaracterized preclinically",
    "A3": "Target expression/function differs across species",
    "B1": "Epidemiological correlation mistaken for causal mechanism",
    "B2": "Biomarker modulated but disease mechanism unaffected",
    "B3": "Target valid but disease driven by redundant parallel pathways",
    "C1": "Genetically homogeneous model misses human heterogeneity",
    "C2": "Animal model does not recapitulate human disease mechanism",
    "C3": "Immunodeficient model removes immune/stromal context",
    "D1": "Biomarker improved by non-disease mechanism",
    "D3": "Dose required for efficacy incompatible with human safety",
    "E1": "Patient heterogeneity masked in unselected trial",
    "E2": "Responder biomarker not identified preclinically",
    "E3": "Enrolled at wrong disease stage",
    "F1": "Species-specific toxicity not visible in preclinical model",
    "F2": "Target essential for normal tissue — no therapeutic window",
}

RISK_LEVELS = {
    "A1": 85, "A2": 75, "A3": 70,
    "B1": 90, "B2": 80, "B3": 75,
    "C1": 65, "C2": 70, "C3": 60,
    "D1": 70, "D3": 65,
    "E1": 60, "E2": 65, "E3": 70,
    "F1": 80, "F2": 85,
}

# ── LOAD DATASET ──────────────────────────────────────────────────────────────
def load_failure_dataset():
    cases = []
    filepath = os.path.join(os.path.dirname(__file__), "terracotta_failures_100.csv")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cases.append(row)
        print(f"✓ Loaded {len(cases)} curated failure cases")
    except Exception as e:
        print(f"✗ Could not load dataset: {e}")
    return cases

FAILURE_CASES = load_failure_dataset()

# ── DATASET LOOKUP ────────────────────────────────────────────────────────────
def find_matching_cases(gene):
    gene_lower = gene.lower()
    matches = []
    for case in FAILURE_CASES:
        searchable = " ".join([
            case.get("target", ""),
            case.get("notes", ""),
            case.get("indication", ""),
            case.get("mechanistic_assessment", "")
        ]).lower()
        if gene_lower in searchable:
            matches.append(case)
    return matches

def extract_ground_truth(matches):
    """Extract the dominant ontology category from matched cases."""
    if not matches:
        return None, []
    
    cats = []
    for c in matches:
        raw = c.get("ontology_category", "")
        # Extract just the code e.g. "B1" from "B1 - Epidemiological..."
        code = raw.split(" - ")[0].split()[0].strip()
        if code:
            cats.append(code)
    
    if not cats:
        return None, []
    
    # Most common category wins
    dominant = Counter(cats).most_common(1)[0][0]
    return dominant, cats

def format_cases_for_prompt(matches, max_cases=12):
    lines = []
    for c in matches[:max_cases]:
        cat_raw = c.get("ontology_category", "")
        code = cat_raw.split(" - ")[0].split()[0].strip()
        lines.append(
            f"CASE: {c.get('target','')} | "
            f"{c.get('company_or_program','')} ({c.get('year','')}) | "
            f"CATEGORY: {code} | "
            f"MECHANISM: {c.get('mechanistic_assessment','')[:180]} | "
            f"KEY INSIGHT: {c.get('notes','')[:150]}"
        )
    return "\n\n".join(lines)

# ── GROQ SCORING ──────────────────────────────────────────────────────────────
def build_prompt(gene, matches, ground_truth_code):
    cases_text = format_cases_for_prompt(matches) if matches else "No direct matches — using general biological reasoning."

    ground_truth_instruction = ""
    if ground_truth_code and len(matches) >= 2:
        cat_name = CATEGORY_NAMES.get(ground_truth_code, ground_truth_code)
        ground_truth_instruction = f"""
DATASET GROUND TRUTH (MANDATORY):
The Terracotta curated dataset contains {len(matches)} documented failures for {gene}.
Expert biologists have classified the primary failure mode as: {ground_truth_code} — {cat_name}

You MUST:
- Set primary_failure_category to exactly "{ground_truth_code}"
- Set primary_failure_name to "{cat_name}"
- Explain this specific failure mode in your summary and signal layers
- Reference the specific historical cases in similar_historical_failures
Do NOT override this classification with general reasoning.
"""

    system = f"""You are Terracotta, an AI-native translational risk engine for drug discovery.
You score drug targets for translational failure risk using a curated dataset of 100 real Phase II/III failures.

TERRACOTTA ONTOLOGY:
A1: Off-target pathway active in humans, silent in model
A2: Drug mechanism mischaracterized preclinically  
A3: Target expression/function differs across species
B1: Epidemiological correlation mistaken for causal mechanism
B2: Biomarker modulated but disease mechanism unaffected
B3: Target valid but disease driven by redundant parallel pathways
C1: Genetically homogeneous model misses human heterogeneity
C2: Animal model does not recapitulate human disease mechanism
C3: Immunodeficient model removes immune/stromal context
D1: Biomarker improved by non-disease mechanism
D3: Dose required for efficacy incompatible with human safety
E1: Patient heterogeneity masked in unselected trial
E2: Responder biomarker not identified preclinically
E3: Enrolled at wrong disease stage
F1: Species-specific toxicity not visible in preclinical model
F2: Target essential for normal tissue — no therapeutic window

CURATED DATASET CASES FOR THIS TARGET:
{cases_text}

{ground_truth_instruction}

Respond ONLY with valid JSON. No markdown, no backticks, no preamble."""

    user = f"""Score this drug target for translational risk: {gene}

Return ONLY this JSON:
{{
  "gene": "{gene}",
  "overall_risk_score": <integer 0-100>,
  "risk_level": "<Low|Medium|High|Critical>",
  "primary_failure_category": "<e.g. B1>",
  "primary_failure_name": "<full category name>",
  "confidence": "<Low|Medium|High>",
  "summary": "<2-3 sentences grounded in the curated cases above>",
  "similar_historical_failures": "<cite 1-2 specific cases from the dataset with company name and year>",
  "signal_layers": {{
    "species_conservation": {{
      "score": <0-100>,
      "finding": "<one sentence>"
    }},
    "tissue_expression": {{
      "score": <0-100>,
      "finding": "<one sentence>"
    }},
    "pathway_topology": {{
      "score": <0-100>,
      "finding": "<one sentence>"
    }},
    "historical_concordance": {{
      "score": <0-100>,
      "finding": "<one sentence citing specific cases>"
    }}
  }},
  "model_recommendation": "<specific model system recommendation>",
  "kill_or_pursue": "<Kill|Caution|Pursue>",
  "kill_or_pursue_rationale": "<one sentence>"
}}"""

    return system, user

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/score", methods=["POST"])
def score():
    data = request.get_json()
    gene = data.get("gene", "").strip().upper()

    if not gene:
        return jsonify({"error": "No gene provided"}), 400

    # Step 1: Dataset lookup
    matches = find_matching_cases(gene)
    ground_truth_code, all_cats = extract_ground_truth(matches)

    # Step 2: If strong dataset signal, override LLM category directly
    forced_category = None
    forced_risk = None
    if ground_truth_code and len(matches) >= 2:
        forced_category = ground_truth_code
        forced_risk = RISK_LEVELS.get(ground_truth_code, 75)

    # Step 3: Build prompt and call Groq
    system_prompt, user_prompt = build_prompt(gene, matches, ground_truth_code)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1400
        )

        raw = response.choices[0].message.content.strip()

        # Clean markdown if present
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        result = json.loads(raw)

        # Step 4: Hard override — dataset always wins over LLM
        if forced_category:
            result["primary_failure_category"] = forced_category
            result["primary_failure_name"] = CATEGORY_NAMES.get(forced_category, forced_category)
            result["overall_risk_score"] = forced_risk
            result["dataset_ground_truth"] = True
        else:
            result["dataset_ground_truth"] = False

        result["dataset_cases_used"] = len(matches)
        result["dataset_categories_found"] = all_cats

        return jsonify(result)

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Parse error: {str(e)}", "raw": raw[:500]}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/dataset/stats", methods=["GET"])
def dataset_stats():
    """Returns stats about the curated dataset — useful for the closed loop dashboard."""
    cat_counts = Counter()
    indications = Counter()
    for c in FAILURE_CASES:
        raw = c.get("ontology_category", "").strip()
        if raw:
            parts = raw.split(" - ")[0].split()
            if parts:
                code = parts[0].strip()
                if code:
                    cat_counts[code] += 1
        ind = c.get("indication", "").strip()
        if ind:
            indications[ind] += 1

    return jsonify({
        "total_cases": len(FAILURE_CASES),
        "category_distribution": dict(cat_counts.most_common()),
        "top_indications": dict(indications.most_common(10)),
        "ontology_version": "v0.2",
        "dataset_version": "terracotta_failures_100"
    })

@app.route("/feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json()
    
    gene = data.get("gene", "").strip().upper()
    risk_score = data.get("risk_score", "")
    decision = data.get("decision", "").strip()
    next_experiment = data.get("next_experiment", "").strip()
    score_agreement = data.get("score_agreement", "").strip()
    disagreement_reason = data.get("disagreement_reason", "").strip()
    researcher_type = data.get("researcher_type", "").strip()
    
    if not gene or not decision:
        return jsonify({"error": "Gene and decision are required"}), 400
    
    import datetime
    timestamp = datetime.datetime.utcnow().isoformat()
    
    feedback_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "feedback_log.csv"
    )
    
    # Create file with header if it doesn't exist
    file_exists = os.path.exists(feedback_path)
    
    try:
        with open(feedback_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "gene", "risk_score", "decision",
                    "next_experiment", "score_agreement",
                    "disagreement_reason", "researcher_type"
                ])
            writer.writerow([
                timestamp, gene, risk_score, decision,
                next_experiment, score_agreement,
                disagreement_reason, researcher_type
            ])
        return jsonify({
            "status": "success",
            "message": "Feedback logged. Thank you — this makes Terracotta smarter.",
            "timestamp": timestamp
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/feedback/log", methods=["GET"])
def view_feedback():
    """View all submitted feedback — admin endpoint."""
    feedback_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "feedback_log.csv"
    )
    entries = []
    try:
        with open(feedback_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)
    except Exception:
        pass
    return jsonify({
        "total_submissions": len(entries),
        "entries": entries
    })



from serendipity import serendipity_bp
app.register_blueprint(serendipity_bp)
from discovery import discovery_bp
app.register_blueprint(discovery_bp)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


