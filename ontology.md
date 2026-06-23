# Terracotta Translational Failure Ontology (v0.1)

## Purpose
This document defines a structured taxonomy of *why* drug targets fail to
translate from preclinical models to human clinical efficacy. Each category
represents a distinct, mechanistically meaningful failure mode. The goal is
that two trained scientists, given the same case, would assign it to the
same category most of the time.

## Scope
This ontology classifies **translational failures** — cases where a target
appeared promising in preclinical work but did not produce clinical benefit.
It does NOT classify:
- Pure manufacturing/CMC failures
- Pure safety/toxicity failures (tracked separately)
- Business/strategic discontinuations unrelated to biology

---

## Categories

### 1. Species Conservation Failure
The target or its regulatory context differs meaningfully between the
preclinical species (commonly mouse) and humans, despite apparent sequence
or pathway conservation. The protein may be "the same gene" but behave
differently due to expression control, isoform usage, or interacting
partners that differ across species.

### 2. Tissue/Cellular Context Mismatch
The target's expression level, cell-type distribution, or functional role
differs between the tissue/cell type studied preclinically and the
disease-relevant human tissue. The target may be "real" but studied in the
wrong cellular context.

### 3. Pathway Redundancy / Compensation
The target's contribution to disease biology is buffered by parallel or
compensatory pathways in the chronic human disease state — buffering that
acute or short-duration preclinical models fail to reveal.

### 4. Disease Model Non-Representativeness
The preclinical disease model (genetic knockout, induced model, xenograft,
etc.) does not adequately recapitulate the relevant aspects of human disease
progression, chronicity, or heterogeneity.

### 5. Target Validation vs. Efficacy Gap
The target is genuinely and correctly implicated in human disease biology
(strong genetic/correlative evidence), but pharmacological modulation of the
target does not produce clinical benefit — a gap between *association* and
*causation* in the clinical context.

### 6. Pharmacology / Exposure Mismatch
The clinical failure is not primarily a target biology failure — the drug
did not achieve sufficient concentration, distribution, or duration at the
target site in humans to adequately test the hypothesis.

### 7. Patient Population Heterogeneity
The target is valid in a biologically defined subset of patients, but the
trial population was not stratified by relevant biomarkers, diluting any
true signal.

---

## Open Questions / Boundary Cases
*(To be filled in as we label real cases)*

### 8. Dual-Role Target Conflict
The target serves two opposing or competing physiological functions within
the same tissue or biological context — one disease-relevant, one
protective or homeostatic. Modulating the target achieves the intended
therapeutic effect but simultaneously disrupts the protective function,
causing net harm. This is distinct from Category 5 (where modulation
simply fails to produce benefit) and from multi-substrate collateral
effects (Category 5 subcategory, where harm comes from an unrelated
pathway). Here the harm and the benefit come from the SAME target in the
SAME tissue context.

Prototype case: Tanezumab (anti-NGF) in osteoarthritis — NGF blockade
reduced pain via TrkA nociceptors but simultaneously disrupted NGF's
protective role in joint tissue homeostasis, causing rapidly progressive
osteoarthritis.
