"""
Create the complete tiered RAG notebook structure.

Actions:
  1. Create 7 tier folders under qdrant_notebooks/
  2. Move existing notebooks to correct tier + rename with new sequence numbers
  3. Create lightweight stub notebooks for all planned-but-not-yet-built notebooks
  4. Print the full catalogue
"""
import json, uuid, shutil, os

BASE = r"C:\Users\Administrator\RAG\qdrant_notebooks"

# ─────────────────────────────────────────────────────────────────────────────
# Complete catalogue
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (seq_num, tier_folder, filename_stem, short_title, status, description)
CATALOGUE = [
    # ── Tier 1 — Chunking & Indexing ─────────────────────────────────────────
    ("01", "tier1_chunking_indexing",  "01_Simple_RAG",
     "Simple RAG",
     "DONE",
     "Baseline: fixed-size chunks, Titan V2 embeddings, Qdrant vector search, Claude generation."),

    ("02", "tier1_chunking_indexing",  "02_Semantic_Chunking",
     "Semantic Chunking",
     "BUILD_NEXT",
     "Split at meaning boundaries (cosine-similarity drop between sentences) instead of fixed char count."),

    ("03", "tier1_chunking_indexing",  "03_Hierarchical_RAG",
     "Hierarchical RAG",
     "STUB",
     "Store small child chunks for precise matching; retrieve larger parent chunks for richer context (3-5x)."),

    ("04", "tier1_chunking_indexing",  "04_Parent_Child_RAG",
     "Parent-Child RAG",
     "STUB",
     "4-level hierarchy (document -> section -> paragraph -> sentence); expand to any ancestor level."),

    ("05", "tier1_chunking_indexing",  "05_Sentence_Window_RAG",
     "Sentence Window RAG",
     "STUB",
     "NEW: Retrieve single sentences for precision; return surrounding window of sentences as LLM context."),

    ("06", "tier1_chunking_indexing",  "06_Contextual_Retrieval",
     "Contextual Retrieval",
     "STUB",
     "Anthropic pattern: prepend LLM-generated context summary to each chunk before embedding. Cuts retrieval failures 49-67%."),

    # ── Tier 2 — Retrieval Quality ────────────────────────────────────────────
    ("07", "tier2_retrieval_quality",  "07_Hybrid_Search",
     "Hybrid Search",
     "STUB",
     "Combine BM25 keyword search + vector search; catches exact-term matches that semantic search misses."),

    ("08", "tier2_retrieval_quality",  "08_HyDE",
     "HyDE (Hypothetical Document Embeddings)",
     "STUB",
     "Generate a hypothetical answer, embed it, use that embedding for retrieval. Bridges question/document vocabulary gap."),

    ("09", "tier2_retrieval_quality",  "09_Reranking",
     "Reranking",
     "STUB",
     "Wide-net retrieval (top-20) then LLM re-scores candidates and trims to top-5. Precision boost at low cost."),

    ("10", "tier2_retrieval_quality",  "10_Contextual_Compression",
     "Contextual Compression",
     "STUB",
     "After retrieval, extract only the query-relevant sentences from each chunk. Removes noise before generation."),

    ("11", "tier2_retrieval_quality",  "11_Metadata_Filtering",
     "Metadata Filtering RAG",
     "STUB",
     "NEW: Tag chunks with structured metadata (page, section, topic); filter by metadata before vector search."),

    ("12", "tier2_retrieval_quality",  "12_Multi_Document_RAG",
     "Multi-Document RAG",
     "STUB",
     "Index and query across multiple PDFs simultaneously; attribute answers to specific source documents."),

    # ── Tier 3 — Query Handling ───────────────────────────────────────────────
    ("13", "tier3_query_handling",     "13_Query_Decomposition",
     "Query Decomposition",
     "STUB",
     "Break complex multi-part questions into focused sub-questions; answer each separately then synthesize."),

    ("14", "tier3_query_handling",     "14_Step_Back_Prompting",
     "Step-Back Prompting",
     "STUB",
     "NEW: Reformulate specific question into a broader 'step-back' question; combine general + specific retrieval."),

    ("15", "tier3_query_handling",     "15_Fusion_Retrieval",
     "Fusion Retrieval (RAG Fusion)",
     "DONE",
     "Generate N query variants with Claude, retrieve for each, merge all lists with Reciprocal Rank Fusion (RRF)."),

    ("16", "tier3_query_handling",     "16_Chain_of_Thought_RAG",
     "Chain-of-Thought RAG",
     "STUB",
     "Decompose query into explicit reasoning steps; targeted retrieval at each step; build answer progressively."),

    ("17", "tier3_query_handling",     "17_ReAct_RAG",
     "ReAct RAG",
     "STUB",
     "Interleave Thought / Action / Observation cycles; model picks which tool to invoke based on each observation."),

    # ── Tier 4 — Agentic & Self-Improving ─────────────────────────────────────
    ("18", "tier4_agentic",            "18_Corrective_RAG",
     "Corrective RAG",
     "STUB",
     "Score each retrieved chunk; route to: use-as-is / refine / web-search-fallback based on confidence."),

    ("19", "tier4_agentic",            "19_Self_RAG",
     "Self RAG",
     "STUB",
     "Evaluate own answer on 4 quality dimensions; iteratively refine through self-critique until threshold met."),

    ("20", "tier4_agentic",            "20_Iterative_RAG",
     "Iterative RAG",
     "STUB",
     "Draft answer, identify content gaps, targeted follow-up retrievals to fill gaps, refine until convergence."),

    ("21", "tier4_agentic",            "21_Recursive_RAG",
     "Recursive RAG",
     "STUB",
     "Multiple retrieve-reason cycles with confidence-based stopping; iterates until sufficient info gathered."),

    ("22", "tier4_agentic",            "22_Agentic_RAG",
     "Agentic RAG",
     "STUB",
     "LLM autonomously decides when and what to retrieve; selects from tool set in a self-directed reasoning loop."),

    # ── Tier 5 — Multi-turn & Memory ──────────────────────────────────────────
    ("23", "tier5_memory_conversation","23_Memory_Augmented_RAG",
     "Memory-Augmented RAG",
     "STUB",
     "Persist conversation history; use it to resolve pronoun references and enrich retrieval across sessions."),

    ("24", "tier5_memory_conversation","24_Multi_Turn_Conversational_RAG",
     "Multi-Turn Conversational RAG",
     "STUB",
     "Dual vector stores (document KB + conversation history); retrieve relevant past turns alongside documents."),

    # ── Tier 6 — Ensemble & Meta ──────────────────────────────────────────────
    ("25", "tier6_ensemble_meta",      "25_Ensemble_RAG",
     "Ensemble RAG",
     "STUB",
     "Run multiple RAG strategies in parallel (Simple, HyDE, Fusion); aggregate answers via ranking or voting."),

    ("26", "tier6_ensemble_meta",      "26_Adaptive_RAG",
     "Adaptive RAG",
     "STUB",
     "Classify query type; dynamically route to the best tier-2/3 strategy to balance cost, latency, and quality."),

    # ── Tier 7 — Production ───────────────────────────────────────────────────
    ("27", "tier7_production",         "27_Streaming_RAG",
     "Streaming RAG",
     "STUB",
     "Stream LLM tokens as generated for low perceived latency; Strands streaming API with Qdrant retrieval."),

    ("28", "tier7_production",         "28_Caching_RAG",
     "Caching RAG",
     "STUB",
     "Cache embeddings + query results to cut redundant API calls; semantic cache for near-duplicate queries."),

    ("29", "tier7_production",         "29_Evaluation_RAG",
     "Evaluation & Metrics",
     "STUB",
     "Comprehensive metrics: precision, recall, faithfulness, answer relevance, latency, cost across all patterns."),

    ("30", "tier7_production",         "30_Complete_Pipeline",
     "Complete RAG Pipeline",
     "STUB",
     "Production-grade pipeline combining the best patterns from all tiers into one deployable AWS-native system."),
]

TIER_DESCRIPTIONS = {
    "tier1_chunking_indexing":   "Tier 1 — Chunking & Indexing Foundations",
    "tier2_retrieval_quality":   "Tier 2 — Retrieval Quality",
    "tier3_query_handling":      "Tier 3 — Query Handling",
    "tier4_agentic":             "Tier 4 — Agentic & Self-Improving",
    "tier5_memory_conversation": "Tier 5 — Multi-turn & Memory",
    "tier6_ensemble_meta":       "Tier 6 — Ensemble & Meta",
    "tier7_production":          "Tier 7 — Production",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def cid():
    return str(uuid.uuid4())[:8]

def md_cell(text):
    return {"cell_type": "markdown", "id": cid(), "metadata": {},
            "source": text.splitlines(keepends=True)}

def code_cell(lines):
    return {"cell_type": "code", "id": cid(), "metadata": {},
            "execution_count": None, "outputs": [], "source": lines}

def make_nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13.0"}
        },
        "cells": cells
    }

def write_nb(path, nb):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

def stub_nb(seq, title, tier_desc, description):
    badge = {
        "tier1_chunking_indexing":   "Tier 1 | Chunking & Indexing",
        "tier2_retrieval_quality":   "Tier 2 | Retrieval Quality",
        "tier3_query_handling":      "Tier 3 | Query Handling",
        "tier4_agentic":             "Tier 4 | Agentic & Self-Improving",
        "tier5_memory_conversation": "Tier 5 | Multi-turn & Memory",
        "tier6_ensemble_meta":       "Tier 6 | Ensemble & Meta",
        "tier7_production":          "Tier 7 | Production",
    }[tier_desc]

    overview = (
        f"# {seq} — {title}\n"
        f"\n"
        f"> **{badge}**\n"
        f"\n"
        f"## What this notebook covers\n"
        f"{description}\n"
        f"\n"
        f"## Status\n"
        f"**Planned** — will be built when the sequence reaches this notebook.\n"
        f"\n"
        f"## Tech stack (same as all notebooks)\n"
        f"| Component | Implementation |\n"
        f"|-----------|---------------|\n"
        f"| PDF Loading | `pypdf.PdfReader` |\n"
        f"| Embeddings | Amazon Bedrock Titan V2 (1024-dim) |\n"
        f"| Vector DB | Qdrant Cloud -> OpenSearch -> in-memory |\n"
        f"| LLM | AWS Strands Agent + Claude Sonnet 4.6 |\n"
    )
    cells = [
        md_cell(overview),
        code_cell([f'print("{seq} — {title}: not yet implemented. Run notebooks in sequence.")\n']),
    ]
    return make_nb(cells)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Create tier folders
# ─────────────────────────────────────────────────────────────────────────────
for folder in TIER_DESCRIPTIONS:
    os.makedirs(os.path.join(BASE, folder), exist_ok=True)
    print(f"  mkdir {folder}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Handle each catalogue entry
# ─────────────────────────────────────────────────────────────────────────────
existing_map = {
    # old path (relative to BASE) -> (seq, tier, stem)
    "01_Simple_RAG.ipynb":     ("01", "tier1_chunking_indexing", "01_Simple_RAG"),
    "03_Fusion_Retrieval.ipynb":("15", "tier3_query_handling",   "15_Fusion_Retrieval"),
}

for seq, tier, stem, title, status, desc in CATALOGUE:
    dest_dir  = os.path.join(BASE, tier)
    dest_path = os.path.join(dest_dir, f"{stem}.ipynb")

    if status == "DONE":
        # Find old file and copy (keep original too for safety)
        old_name = f"{stem.split('_', 1)[1]}.ipynb" if seq != "01" else "01_Simple_RAG.ipynb"
        # Match by stem suffix
        for old_rel, (old_seq, old_tier, old_stem) in existing_map.items():
            if old_stem == stem or (seq == "01" and "Simple_RAG" in old_stem) \
               or (seq == "15" and "Fusion" in old_stem):
                src = os.path.join(BASE, old_rel)
                if os.path.exists(src) and not os.path.exists(dest_path):
                    shutil.copy2(src, dest_path)
                    print(f"  COPY {old_rel} -> {tier}/{stem}.ipynb")
                elif os.path.exists(dest_path):
                    print(f"  EXISTS {tier}/{stem}.ipynb")
                break

    elif status == "BUILD_NEXT":
        # Will be built in the next step — create a placeholder for now
        if not os.path.exists(dest_path):
            nb = stub_nb(seq, title, tier, desc + "\n\n**Building now...**")
            write_nb(dest_path, nb)
            print(f"  PLACEHOLDER {tier}/{stem}.ipynb  [BUILD_NEXT]")
        else:
            print(f"  EXISTS {tier}/{stem}.ipynb")

    else:  # STUB
        if not os.path.exists(dest_path):
            nb = stub_nb(seq, title, tier, desc)
            write_nb(dest_path, nb)
            print(f"  STUB {tier}/{stem}.ipynb")
        else:
            print(f"  EXISTS {tier}/{stem}.ipynb")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Print catalogue
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("COMPLETE RAG NOTEBOOK CATALOGUE")
print("=" * 72)
current_tier = ""
for seq, tier, stem, title, status, desc in CATALOGUE:
    if tier != current_tier:
        print(f"\n  {TIER_DESCRIPTIONS[tier]}")
        print(f"  {'-'*50}")
        current_tier = tier
    tag = "[DONE]      " if status == "DONE" else \
          "[BUILD NEXT]" if status == "BUILD_NEXT" else \
          "[stub]      "
    print(f"    {tag}  {seq}  {title}")

new_patterns = [(s,t,st,ti,st2,d) for s,t,st,ti,st2,d in CATALOGUE if "NEW" in d]
print(f"\n  3 new patterns added (not in original 37):")
for seq, tier, stem, title, status, desc in CATALOGUE:
    if "NEW:" in desc:
        print(f"    {seq}  {title}")

print(f"\nTotal notebooks: {len(CATALOGUE)}")
print("=" * 72)
