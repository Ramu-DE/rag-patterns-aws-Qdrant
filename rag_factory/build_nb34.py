# -*- coding: utf-8 -*-
"""
Build NB34: AI RAG Factory - Spec Layer
Generates: rag_factory/NB34_Factory_Spec_Layer.ipynb
"""
import nbformat, os

NB_PATH = os.path.join(os.path.dirname(__file__), "NB34_Factory_Spec_Layer.ipynb")

def cell(source, cell_type="code"):
    if cell_type == "markdown":
        return nbformat.v4.new_markdown_cell(source)
    return nbformat.v4.new_code_cell(source)

cells = []

# ── Cell 0: Title ─────────────────────────────────────────────────────────────
cells.append(cell("""\
# NB34 — AI RAG Factory: Spec Layer

> **AI Factory | Foundation**

## What this notebook builds

The **Spec Layer** is the typed contract at the heart of the AI RAG Factory.
Inspired by NVIDIA's NIM (inference microservice) contracts, every component —
chunker, retriever, generator, guard — must satisfy a `BaseComponentSpec`.
Pipelines are declared in YAML and validated against `PipelineSpec` before a
single LLM call is made.

### Why specs first?
- 33 notebooks have different APIs. A spec forces a **uniform interface**.
- NVIDIA's key insight: pipelines are **declared, not coded** — the factory assembles them at runtime.
- Temporal workflows wrap Activities that implement these same specs → durable execution for free.

### What this notebook delivers
1. `BaseComponentSpec` — Pydantic contract every component obeys
2. `ComponentManifest` — registry mapping names → all 33 RAG patterns
3. `PipelineSpec` — YAML/dict → validated pipeline config
4. `SpecValidator` — catches invalid combos before execution
5. **Live demo**: validate 4 pipeline configs (simple → production → agentic → multi-tenant)
""", "markdown"))

# ── Cell 1: Progress widget ────────────────────────────────────────────────────
_HTML_BANNER = (
    '<style>'
    '.fhdr{background:linear-gradient(135deg,#1e3a5f,#2d6a4f);'
    'color:#fff;padding:16px 24px;border-radius:10px;'
    "font-family:'Segoe UI',sans-serif;margin-bottom:12px}"
    '.fhdr h2{margin:0 0 4px 0;font-size:1.4em}'
    '.fhdr p{margin:0;opacity:.85;font-size:.9em}'
    '.sb{display:inline-block;padding:3px 10px;border-radius:12px;'
    'font-size:.8em;font-weight:700;margin:2px}'
    '.tb{background:#1e3a5f;color:#fff}'
    '.mb{background:#2d6a4f;color:#fff}'
    '.nb{background:#76b900;color:#000}'
    '</style>'
    '<div class="fhdr">'
    '<h2>&#127981; AI RAG Factory &#8212; NB34: Spec Layer</h2>'
    '<p>'
    '<span class="sb tb">NVIDIA-inspired</span>'
    '<span class="sb mb">Temporal-ready</span>'
    '<span class="sb nb">33 Patterns</span>'
    '&nbsp; Foundation notebook &#8212; all other factory notebooks depend on this'
    '</p></div>'
)
cells.append(cell("""\
from IPython.display import display as _d, HTML as _H
_HTML_BANNER = (
    '<style>'
    '.fhdr{background:linear-gradient(135deg,#1e3a5f,#2d6a4f);'
    'color:#fff;padding:16px 24px;border-radius:10px;'
    "font-family:\\'Segoe UI\\',sans-serif;margin-bottom:12px}"
    '.fhdr h2{margin:0 0 4px 0;font-size:1.4em}'
    '.fhdr p{margin:0;opacity:.85;font-size:.9em}'
    '.sb{display:inline-block;padding:3px 10px;border-radius:12px;'
    'font-size:.8em;font-weight:700;margin:2px}'
    '.tb{background:#1e3a5f;color:#fff}'
    '.mb{background:#2d6a4f;color:#fff}'
    '.nb{background:#76b900;color:#000}'
    '</style>'
    '<div class=\\'fhdr\\'>'
    '<h2>&#127981; AI RAG Factory &#8212; NB34: Spec Layer</h2>'
    '<p>'
    '<span class=\\'sb tb\\'>NVIDIA-inspired</span>'
    '<span class=\\'sb mb\\'>Temporal-ready</span>'
    '<span class=\\'sb nb\\'>33 Patterns</span>'
    '&nbsp; Foundation &#8212; all other factory notebooks depend on this'
    '</p></div>'
)
_d(_H(_HTML_BANNER))
"""))

# ── Cell 2: Deps ───────────────────────────────────────────────────────────────
cells.append(cell("""\
import subprocess, sys

packages = ["pydantic>=2.0", "pyyaml", "boto3", "qdrant-client", "python-dotenv"]
for pkg in packages:
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=False)

print("✓ Dependencies ready")
"""))

# ── Cell 3: Imports ────────────────────────────────────────────────────────────
cells.append(cell("""\
import os, json, yaml
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator, model_validator
from dotenv import load_dotenv

load_dotenv(r"C:/Users/Administrator/RAG/.env", override=True)

print("✓ Imports OK")
"""))

# ── Cell 4: Section header ─────────────────────────────────────────────────────
cells.append(cell("## Part 1 — Component Enums & Roles", "markdown"))

# ── Cell 5: Enums ─────────────────────────────────────────────────────────────
cells.append(cell("""\
class ComponentRole(str, Enum):
    \"\"\"Where in the pipeline a component sits.\"\"\"
    CHUNKER    = "chunker"
    RETRIEVER  = "retriever"
    QUERY_OPS  = "query_ops"
    GENERATOR  = "generator"
    GUARD      = "guard"
    EVALUATOR  = "evaluator"
    MEMORY     = "memory"
    ROUTER     = "router"

class ChunkStrategy(str, Enum):
    FIXED          = "fixed"
    SEMANTIC       = "semantic"
    HIERARCHICAL   = "hierarchical"
    PARENT_CHILD   = "parent_child"
    SENTENCE_WINDOW= "sentence_window"
    CONTEXTUAL     = "contextual"

class RetrievalStrategy(str, Enum):
    DENSE          = "dense"
    HYBRID_RRF     = "hybrid_rrf"
    HYDE           = "hyde"
    RERANK         = "rerank"
    COMPRESSED     = "compressed"
    METADATA_FILTER= "metadata_filter"
    MULTI_DOC      = "multi_doc"

class QueryStrategy(str, Enum):
    DIRECT         = "direct"
    DECOMPOSE      = "decompose"
    STEPBACK       = "stepback"
    FUSION         = "fusion"
    COT            = "chain_of_thought"
    REACT          = "react"

class AgenticMode(str, Enum):
    NONE           = "none"
    CORRECTIVE     = "corrective"
    SELF_RAG       = "self_rag"
    ITERATIVE      = "iterative"
    RECURSIVE      = "recursive"
    AGENTIC        = "agentic"

class Severity(str, Enum):
    LOW            = "low"
    MEDIUM         = "medium"
    HIGH           = "high"
    CRITICAL       = "critical"

print("✓ Enums defined")
print(f"   ComponentRole values : {[r.value for r in ComponentRole]}")
print(f"   ChunkStrategy values : {[s.value for s in ChunkStrategy]}")
print(f"   RetrievalStrategy    : {[s.value for s in RetrievalStrategy]}")
print(f"   AgenticMode          : {[m.value for m in AgenticMode]}")
"""))

# ── Cell 6: Section header ─────────────────────────────────────────────────────
cells.append(cell("## Part 2 — BaseComponentSpec (NVIDIA NIM contract)", "markdown"))

# ── Cell 7: BaseComponentSpec ──────────────────────────────────────────────────
cells.append(cell("""\
class BaseComponentSpec(BaseModel):
    \"\"\"
    Typed contract every RAG component must satisfy.
    Modelled after NVIDIA NIM microservice contracts —
    each component is self-describing and independently testable.
    \"\"\"
    # Identity
    name        : str               = Field(..., description="Unique component identifier")
    version     : str               = Field("1.0.0", description="Semver")
    role        : ComponentRole     = Field(..., description="Pipeline role")
    notebook_ref: str               = Field(..., description="Source notebook path")
    tier        : int               = Field(..., ge=1, le=9, description="Tier 1–9")
    description : str               = Field(..., description="One-line purpose")

    # Runtime contracts
    input_schema : List[str]        = Field(..., description="Required input keys")
    output_schema: List[str]        = Field(..., description="Guaranteed output keys")

    # Resource hints (Temporal activity config)
    max_retries  : int              = Field(3,  ge=0, le=10)
    timeout_secs : int              = Field(60, ge=1, le=3600)
    is_async     : bool             = Field(False, description="True → Temporal Activity")
    is_streaming : bool             = Field(False, description="True → SSE / token stream")

    # Guard metadata
    guards_applied: List[str]       = Field(default_factory=list,
                                            description="FM codes this component is protected by")
    failure_modes : List[str]       = Field(default_factory=list,
                                            description="FM codes this component can trigger")

    # Config passthrough
    config       : Dict[str, Any]   = Field(default_factory=dict,
                                            description="Component-specific config overrides")

    class Config:
        use_enum_values = True

    def satisfies(self, required_outputs: List[str]) -> bool:
        \"\"\"Can this component produce all required_outputs?\"\"\"
        return all(k in self.output_schema for k in required_outputs)

    def compatible_with(self, next_spec: 'BaseComponentSpec') -> Tuple[bool, List[str]]:
        \"\"\"
        Check whether this component's outputs satisfy next_spec's inputs.
        Returns (ok, missing_keys).
        \"\"\"
        missing = [k for k in next_spec.input_schema if k not in self.output_schema]
        return (len(missing) == 0), missing

    def __repr__(self):
        return f"<{self.role}:{self.name} v{self.version} tier={self.tier}>"


# ── Quick sanity test ────────────────────────────────────────────────────────
_test = BaseComponentSpec(
    name="semantic_chunking",
    role=ComponentRole.CHUNKER,
    notebook_ref="qdrant_notebooks/tier1_chunking_indexing/02_Semantic_Chunking.ipynb",
    tier=1,
    description="Cosine breakpoint chunking — semantically coherent chunks",
    input_schema=["raw_text"],
    output_schema=["chunks", "chunk_count"],
    timeout_secs=120,
    failure_modes=["FM-R2"],
    guards_applied=["FM-R2"],
)
print("✓ BaseComponentSpec")
print(f"   {_test}")
print(f"   satisfies(['chunks'])       → {_test.satisfies(['chunks'])}")
print(f"   satisfies(['chunks','emb']) → {_test.satisfies(['chunks','emb'])}")
"""))

# ── Cell 8: Section ───────────────────────────────────────────────────────────
cells.append(cell("## Part 3 — ComponentManifest (all 33 patterns + 4 guard suites)", "markdown"))

# ── Cell 9: Manifest ──────────────────────────────────────────────────────────
cells.append(cell("""\
# ── Tier 1: Chunking & Indexing ──────────────────────────────────────────────
_TIER1 = [
    BaseComponentSpec(
        name="fixed_chunking", role=ComponentRole.CHUNKER, tier=1,
        notebook_ref="qdrant_notebooks/tier1_chunking_indexing/01_Simple_RAG.ipynb",
        description="Fixed-size token chunking — baseline pipeline",
        input_schema=["raw_text"],
        output_schema=["chunks", "chunk_count"],
        failure_modes=["FM-R2"],
    ),
    BaseComponentSpec(
        name="semantic_chunking", role=ComponentRole.CHUNKER, tier=1,
        notebook_ref="qdrant_notebooks/tier1_chunking_indexing/02_Semantic_Chunking.ipynb",
        description="Cosine breakpoint — variable-length semantically coherent chunks",
        input_schema=["raw_text"],
        output_schema=["chunks", "chunk_count"],
        timeout_secs=120, failure_modes=["FM-R2"],
    ),
    BaseComponentSpec(
        name="hierarchical_chunking", role=ComponentRole.CHUNKER, tier=1,
        notebook_ref="qdrant_notebooks/tier1_chunking_indexing/03_Hierarchical_RAG.ipynb",
        description="Multi-level chunk tree: summary → detail",
        input_schema=["raw_text"],
        output_schema=["chunks", "chunk_count", "hierarchy_map"],
        timeout_secs=180,
    ),
    BaseComponentSpec(
        name="parent_child_chunking", role=ComponentRole.CHUNKER, tier=1,
        notebook_ref="qdrant_notebooks/tier1_chunking_indexing/04_Parent_Child_RAG.ipynb",
        description="Index child chunks; retrieve parent for context",
        input_schema=["raw_text"],
        output_schema=["chunks", "chunk_count", "parent_map"],
        timeout_secs=180,
    ),
    BaseComponentSpec(
        name="sentence_window_chunking", role=ComponentRole.CHUNKER, tier=1,
        notebook_ref="qdrant_notebooks/tier1_chunking_indexing/05_Sentence_Window_RAG.ipynb",
        description="Per-sentence index; expand ±k window at retrieval",
        input_schema=["raw_text"],
        output_schema=["chunks", "chunk_count", "window_map"],
        failure_modes=["FM-R2", "FM-A4"], guards_applied=["FM-R2", "FM-A4"],
    ),
    BaseComponentSpec(
        name="contextual_chunking", role=ComponentRole.CHUNKER, tier=1,
        notebook_ref="qdrant_notebooks/tier1_chunking_indexing/06_Contextual_Retrieval.ipynb",
        description="LLM-generated context prefix before embedding — 67% fewer retrieval failures",
        input_schema=["raw_text"],
        output_schema=["chunks", "chunk_count", "context_map"],
        timeout_secs=300, is_async=True,
        failure_modes=["FM-R3"], guards_applied=["FM-R3"],
    ),
]

# ── Tier 2: Retrieval Quality ─────────────────────────────────────────────────
_TIER2 = [
    BaseComponentSpec(
        name="dense_retrieval", role=ComponentRole.RETRIEVER, tier=2,
        notebook_ref="qdrant_notebooks/tier1_chunking_indexing/01_Simple_RAG.ipynb",
        description="Titan Embeddings V2 dense ANN via Qdrant",
        input_schema=["query", "collection_name"],
        output_schema=["retrieved_chunks", "scores"],
        failure_modes=["FM-R1", "FM-R4", "FM-R5"],
    ),
    BaseComponentSpec(
        name="hybrid_rrf_retrieval", role=ComponentRole.RETRIEVER, tier=2,
        notebook_ref="qdrant_notebooks/tier2_retrieval_quality/07_Hybrid_Search.ipynb",
        description="Dense + BM25 RRF fusion — Recall@5 +17pp vs dense alone",
        input_schema=["query", "collection_name"],
        output_schema=["retrieved_chunks", "scores", "rrf_scores"],
        failure_modes=["FM-R1"], guards_applied=["FM-R1"],
    ),
    BaseComponentSpec(
        name="hyde_retrieval", role=ComponentRole.RETRIEVER, tier=2,
        notebook_ref="qdrant_notebooks/tier2_retrieval_quality/08_HyDE.ipynb",
        description="Hypothetical Document Embeddings — query vocabulary bridge",
        input_schema=["query", "collection_name"],
        output_schema=["retrieved_chunks", "scores", "hypothesis"],
        timeout_secs=90, is_async=True,
        failure_modes=["FM-R4"], guards_applied=["FM-R4"],
    ),
    BaseComponentSpec(
        name="reranked_retrieval", role=ComponentRole.RETRIEVER, tier=2,
        notebook_ref="qdrant_notebooks/tier2_retrieval_quality/09_Reranking.ipynb",
        description="LLM cross-encoder reranking of initial candidates",
        input_schema=["query", "retrieved_chunks"],
        output_schema=["retrieved_chunks", "scores", "rerank_scores"],
        timeout_secs=120, is_async=True,
        failure_modes=["FM-G2"], guards_applied=["FM-G2"],
    ),
    BaseComponentSpec(
        name="compressed_retrieval", role=ComponentRole.RETRIEVER, tier=2,
        notebook_ref="qdrant_notebooks/tier2_retrieval_quality/10_Contextual_Compression.ipynb",
        description="Extract only relevant sentences from retrieved chunks",
        input_schema=["query", "retrieved_chunks"],
        output_schema=["retrieved_chunks", "scores", "compression_ratio"],
        timeout_secs=90, is_async=True,
        failure_modes=["FM-R5"], guards_applied=["FM-R5"],
    ),
    BaseComponentSpec(
        name="filtered_retrieval", role=ComponentRole.RETRIEVER, tier=2,
        notebook_ref="qdrant_notebooks/tier2_retrieval_quality/11_Metadata_Filtering.ipynb",
        description="Pre-filter by structured metadata before ANN",
        input_schema=["query", "collection_name", "filters"],
        output_schema=["retrieved_chunks", "scores"],
        failure_modes=["FM-S3"],
    ),
    BaseComponentSpec(
        name="multi_doc_retrieval", role=ComponentRole.RETRIEVER, tier=2,
        notebook_ref="qdrant_notebooks/tier2_retrieval_quality/12_Multi_Document_RAG.ipynb",
        description="Source-aware retrieval across heterogeneous document corpus",
        input_schema=["query", "collection_name"],
        output_schema=["retrieved_chunks", "scores", "source_map"],
    ),
]

# ── Tier 3: Query Handling ────────────────────────────────────────────────────
_TIER3 = [
    BaseComponentSpec(
        name="query_decomposition", role=ComponentRole.QUERY_OPS, tier=3,
        notebook_ref="qdrant_notebooks/tier3_query_handling/13_Query_Decomposition.ipynb",
        description="Break complex queries into atomic sub-queries",
        input_schema=["query"],
        output_schema=["sub_queries", "query"],
        timeout_secs=60, is_async=True,
        failure_modes=["FM-A3"], guards_applied=["FM-A3"],
    ),
    BaseComponentSpec(
        name="stepback_prompting", role=ComponentRole.QUERY_OPS, tier=3,
        notebook_ref="qdrant_notebooks/tier3_query_handling/14_Step_Back_Prompting.ipynb",
        description="Abstract query to higher-level principle before retrieval",
        input_schema=["query"],
        output_schema=["abstract_query", "query"],
        timeout_secs=60, is_async=True,
    ),
    BaseComponentSpec(
        name="fusion_retrieval", role=ComponentRole.QUERY_OPS, tier=3,
        notebook_ref="qdrant_notebooks/tier3_query_handling/15_Fusion_Retrieval.ipynb",
        description="Generate 4 query variants; retrieve each; RRF merge",
        input_schema=["query", "collection_name"],
        output_schema=["retrieved_chunks", "scores", "query_variants"],
        timeout_secs=120, is_async=True,
        failure_modes=["FM-R1"], guards_applied=["FM-R1"],
    ),
    BaseComponentSpec(
        name="cot_rag", role=ComponentRole.QUERY_OPS, tier=3,
        notebook_ref="qdrant_notebooks/tier3_query_handling/16_Chain_of_Thought_RAG.ipynb",
        description="Interleave reasoning steps with retrieval",
        input_schema=["query", "retrieved_chunks"],
        output_schema=["answer", "reasoning_trace"],
        timeout_secs=120, is_async=True,
    ),
    BaseComponentSpec(
        name="react_rag", role=ComponentRole.QUERY_OPS, tier=3,
        notebook_ref="qdrant_notebooks/tier3_query_handling/17_ReAct_RAG.ipynb",
        description="Reasoning + Acting: LLM decides when to retrieve",
        input_schema=["query"],
        output_schema=["answer", "action_trace"],
        timeout_secs=240, is_async=True,
    ),
]

# ── Tier 4: Agentic ───────────────────────────────────────────────────────────
_TIER4 = [
    BaseComponentSpec(
        name="corrective_rag", role=ComponentRole.GENERATOR, tier=4,
        notebook_ref="qdrant_notebooks/tier4_agentic/18_Corrective_RAG.ipynb",
        description="Retrieval evaluator triggers correction or fallback",
        input_schema=["query", "retrieved_chunks"],
        output_schema=["answer", "correction_log"],
        timeout_secs=180, is_async=True,
        failure_modes=["FM-S4"], guards_applied=["FM-S4"],
    ),
    BaseComponentSpec(
        name="self_rag", role=ComponentRole.GENERATOR, tier=4,
        notebook_ref="qdrant_notebooks/tier4_agentic/19_Self_RAG.ipynb",
        description="4 reflection tokens: Retrieve / IsRel / IsSup / IsUse",
        input_schema=["query", "retrieved_chunks"],
        output_schema=["answer", "reflection_tokens", "citation_scores"],
        timeout_secs=300, is_async=True,
        failure_modes=["FM-G1"], guards_applied=["FM-G1"],
    ),
    BaseComponentSpec(
        name="iterative_rag", role=ComponentRole.GENERATOR, tier=4,
        notebook_ref="qdrant_notebooks/tier4_agentic/20_Iterative_RAG.ipynb",
        description="Multiple retrieval rounds guided by gap analysis",
        input_schema=["query", "collection_name"],
        output_schema=["answer", "iteration_log"],
        timeout_secs=600, is_async=True,
    ),
    BaseComponentSpec(
        name="recursive_rag", role=ComponentRole.GENERATOR, tier=4,
        notebook_ref="qdrant_notebooks/tier4_agentic/21_Recursive_RAG.ipynb",
        description="Recursively decompose until all sub-queries are answerable",
        input_schema=["query", "collection_name"],
        output_schema=["answer", "decomp_tree"],
        timeout_secs=600, is_async=True,
    ),
    BaseComponentSpec(
        name="agentic_rag", role=ComponentRole.GENERATOR, tier=4,
        notebook_ref="qdrant_notebooks/tier4_agentic/22_Agentic_RAG.ipynb",
        description="Autonomous agent with multiple retrieval tools",
        input_schema=["query"],
        output_schema=["answer", "tool_calls", "agent_trace"],
        timeout_secs=900, is_async=True,
    ),
]

# ── Tier 5: Memory & Conversation ────────────────────────────────────────────
_TIER5 = [
    BaseComponentSpec(
        name="memory_augmented_rag", role=ComponentRole.MEMORY, tier=5,
        notebook_ref="qdrant_notebooks/tier5_memory_conversation/23_Memory_Augmented_RAG.ipynb",
        description="Short-term + long-term memory alongside document retrieval",
        input_schema=["query", "session_id"],
        output_schema=["answer", "memory_hits", "updated_memory"],
        timeout_secs=120, is_async=True,
    ),
    BaseComponentSpec(
        name="multiturn_rag", role=ComponentRole.MEMORY, tier=5,
        notebook_ref="qdrant_notebooks/tier5_memory_conversation/24_Multi_Turn_Conversational_RAG.ipynb",
        description="History-aware query rewriter for follow-up resolution",
        input_schema=["query", "conversation_history"],
        output_schema=["answer", "rewritten_query"],
        timeout_secs=90, is_async=True,
        failure_modes=["FM-A4"], guards_applied=["FM-A4"],
    ),
]

# ── Tier 6: Ensemble & Meta ───────────────────────────────────────────────────
_TIER6 = [
    BaseComponentSpec(
        name="ensemble_rag", role=ComponentRole.ROUTER, tier=6,
        notebook_ref="qdrant_notebooks/tier6_ensemble_meta/25_Ensemble_RAG.ipynb",
        description="Run multiple retrieval strategies; vote/merge results",
        input_schema=["query", "collection_name"],
        output_schema=["answer", "strategy_scores", "ensemble_log"],
        timeout_secs=300, is_async=True,
    ),
    BaseComponentSpec(
        name="adaptive_rag", role=ComponentRole.ROUTER, tier=6,
        notebook_ref="qdrant_notebooks/tier6_ensemble_meta/26_Adaptive_RAG.ipynb",
        description="Query classifier routes to optimal retrieval strategy",
        input_schema=["query", "collection_name"],
        output_schema=["answer", "chosen_strategy", "routing_reason"],
        timeout_secs=120, is_async=True,
    ),
]

# ── Tier 7: Production ────────────────────────────────────────────────────────
_TIER7 = [
    BaseComponentSpec(
        name="streaming_rag", role=ComponentRole.GENERATOR, tier=7,
        notebook_ref="qdrant_notebooks/tier7_production/27_Streaming_RAG.ipynb",
        description="Stream LLM tokens — lowest TTFT",
        input_schema=["query", "retrieved_chunks"],
        output_schema=["answer_stream", "ttft_ms", "tokens_per_sec"],
        is_streaming=True,
    ),
    BaseComponentSpec(
        name="caching_rag", role=ComponentRole.RETRIEVER, tier=7,
        notebook_ref="qdrant_notebooks/tier7_production/28_Caching_RAG.ipynb",
        description="Semantic cache: near-duplicate query hits cache",
        input_schema=["query", "collection_name"],
        output_schema=["answer", "cache_hit", "cache_latency_ms"],
    ),
    BaseComponentSpec(
        name="evaluation_rag", role=ComponentRole.EVALUATOR, tier=7,
        notebook_ref="qdrant_notebooks/tier7_production/29_Evaluation_RAG.ipynb",
        description="RAGAS 4-metric scoring on every run",
        input_schema=["query", "answer", "retrieved_chunks", "ground_truth"],
        output_schema=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        timeout_secs=180, is_async=True,
    ),
    BaseComponentSpec(
        name="complete_pipeline", role=ComponentRole.GENERATOR, tier=7,
        notebook_ref="qdrant_notebooks/tier7_production/30_Complete_Pipeline_RAG.ipynb",
        description="All Tier 1–7 components in one production pipeline",
        input_schema=["query", "collection_name"],
        output_schema=["answer", "pipeline_trace", "ragas_scores"],
        timeout_secs=600, is_async=True,
    ),
]

# ── Tier 8: Incremental ───────────────────────────────────────────────────────
_TIER8 = [
    BaseComponentSpec(
        name="incremental_rag", role=ComponentRole.CHUNKER, tier=8,
        notebook_ref="qdrant_notebooks/tier8_incremental/31_Incremental_RAG.ipynb",
        description="SHA-256 page manifest + diff engine — 90%+ embed savings on unchanged pages",
        input_schema=["raw_text", "doc_id", "collection_name"],
        output_schema=["chunks", "chunk_count", "manifest", "pages_changed"],
        timeout_secs=300, is_async=True,
        failure_modes=["FM-S1", "FM-R6"], guards_applied=["FM-S1", "FM-R6"],
    ),
]

# ── Tier 9: Multi-Tenant & Federated ─────────────────────────────────────────
_TIER9 = [
    BaseComponentSpec(
        name="multitenant_rag", role=ComponentRole.RETRIEVER, tier=9,
        notebook_ref="qdrant_notebooks/tier9_multi_tenant/32_Multi_Tenant_RAG.ipynb",
        description="Payload-based tenant isolation — FieldCondition on every query",
        input_schema=["query", "collection_name", "tenant_id"],
        output_schema=["retrieved_chunks", "scores"],
        failure_modes=["FM-S3"], guards_applied=["FM-S3"],
    ),
    BaseComponentSpec(
        name="federated_rag", role=ComponentRole.RETRIEVER, tier=9,
        notebook_ref="qdrant_notebooks/tier9_multi_tenant/33_Federated_RAG.ipynb",
        description="Parallel fan-out across collections + RRF merge + LLM router",
        input_schema=["query", "federation_config"],
        output_schema=["retrieved_chunks", "scores", "federate_scores"],
        timeout_secs=120, is_async=True,
    ),
]

# ── Guard Suites ──────────────────────────────────────────────────────────────
_GUARDS = [
    BaseComponentSpec(
        name="retrieval_guard", role=ComponentRole.GUARD, tier=1,
        notebook_ref="research/failure_simulations/FM1_Retrieval_Failures.ipynb",
        description="FM-R1–R6: vocab gap / boundary / context / HyDE / K-dilution / stale detectors",
        input_schema=["retrieved_chunks", "query", "collection_name"],
        output_schema=["retrieved_chunks", "guard_log"],
        guards_applied=["FM-R1","FM-R2","FM-R3","FM-R4","FM-R5","FM-R6"],
    ),
    BaseComponentSpec(
        name="generation_guard", role=ComponentRole.GUARD, tier=1,
        notebook_ref="research/failure_simulations/FM2_Generation_Failures.ipynb",
        description="FM-G1–G5: faithfulness / lost-in-middle / over-reliance / multi-hop / counterfactual",
        input_schema=["answer", "retrieved_chunks", "query"],
        output_schema=["answer", "guard_log", "faithfulness_score"],
        timeout_secs=90, is_async=True,
        guards_applied=["FM-G1","FM-G2","FM-G3","FM-G4","FM-G5"],
    ),
    BaseComponentSpec(
        name="ambiguity_guard", role=ComponentRole.GUARD, tier=1,
        notebook_ref="research/failure_simulations/FM3_Ambiguity_Failures.ipynb",
        description="FM-A1–A5: negation rewrite / temporal / multi-intent / coreference / scope",
        input_schema=["query"],
        output_schema=["query", "guard_log", "rewrite_applied"],
        timeout_secs=45, is_async=True,
        guards_applied=["FM-A1","FM-A2","FM-A3","FM-A4","FM-A5"],
    ),
    BaseComponentSpec(
        name="system_guard", role=ComponentRole.GUARD, tier=1,
        notebook_ref="research/failure_simulations/FM4_System_Failures.ipynb",
        description="FM-S1–S5: index drift / prompt injection / PII leakage / cascade / fragmentation",
        input_schema=["retrieved_chunks", "query", "tenant_id"],
        output_schema=["retrieved_chunks", "query", "guard_log"],
        guards_applied=["FM-S1","FM-S2","FM-S3","FM-S4","FM-S5"],
    ),
]

ALL_SPECS = _TIER1 + _TIER2 + _TIER3 + _TIER4 + _TIER5 + _TIER6 + _TIER7 + _TIER8 + _TIER9 + _GUARDS

print(f"✓ ComponentManifest loaded")
print(f"   Total specs : {len(ALL_SPECS)}")
print(f"   Tier 1 (chunking)  : {sum(1 for s in ALL_SPECS if s.tier==1 and s.role != ComponentRole.GUARD)}")
print(f"   Tier 2 (retrieval) : {sum(1 for s in ALL_SPECS if s.tier==2)}")
print(f"   Tier 3 (query)     : {sum(1 for s in ALL_SPECS if s.tier==3)}")
print(f"   Tier 4 (agentic)   : {sum(1 for s in ALL_SPECS if s.tier==4)}")
print(f"   Tier 5-9 + guards  : {sum(1 for s in ALL_SPECS if s.tier in (5,6,7,8,9) or s.role==ComponentRole.GUARD)}")
print(f"   Async components   : {sum(1 for s in ALL_SPECS if s.is_async)} (→ Temporal Activities)")
"""))

# ── Cell 10: ComponentManifest class ──────────────────────────────────────────
cells.append(cell("## Part 4 — ComponentManifest (registry class)", "markdown"))

cells.append(cell("""\
class ComponentManifest:
    \"\"\"
    Registry of all RAG component specs.
    Inspired by NVIDIA's NIM service registry.
    \"\"\"
    def __init__(self, specs: List[BaseComponentSpec]):
        self._by_name : Dict[str, BaseComponentSpec] = {s.name: s for s in specs}
        self._by_role : Dict[str, List[BaseComponentSpec]] = {}
        self._by_tier : Dict[int, List[BaseComponentSpec]] = {}
        for s in specs:
            self._by_role.setdefault(s.role, []).append(s)
            self._by_tier.setdefault(s.tier, []).append(s)

    def get(self, name: str) -> BaseComponentSpec:
        if name not in self._by_name:
            raise KeyError(f"Component '{name}' not in manifest. "
                           f"Available: {sorted(self._by_name.keys())}")
        return self._by_name[name]

    def by_role(self, role: ComponentRole) -> List[BaseComponentSpec]:
        return self._by_role.get(role, [])

    def by_tier(self, tier: int) -> List[BaseComponentSpec]:
        return self._by_tier.get(tier, [])

    def covering_fm(self, fm_code: str) -> List[BaseComponentSpec]:
        \"\"\"All specs that guard against a given failure mode code.\"\"\"
        return [s for s in self._by_name.values() if fm_code in s.guards_applied]

    def async_specs(self) -> List[BaseComponentSpec]:
        \"\"\"All specs that map to Temporal Activities.\"\"\"
        return [s for s in self._by_name.values() if s.is_async]

    def summary(self) -> str:
        lines = ["ComponentManifest summary"]
        for tier in sorted(self._by_tier):
            specs = self._by_tier[tier]
            names = [s.name for s in specs]
            lines.append(f"  Tier {tier}: {names}")
        return "\\n".join(lines)


MANIFEST = ComponentManifest(ALL_SPECS)

print("✓ MANIFEST ready")
print()
print(MANIFEST.summary())
print()
print(f"Async (→ Temporal)  : {[s.name for s in MANIFEST.async_specs()]}")
print()
print(f"Guards for FM-R1    : {[s.name for s in MANIFEST.covering_fm('FM-R1')]}")
print(f"Guards for FM-S3    : {[s.name for s in MANIFEST.covering_fm('FM-S3')]}")
"""))

# ── Cell 11: Section ──────────────────────────────────────────────────────────
cells.append(cell("## Part 5 — PipelineSpec (YAML → validated pipeline config)", "markdown"))

# ── Cell 12: PipelineSpec ─────────────────────────────────────────────────────
cells.append(cell("""\
class IngestionConfig(BaseModel):
    doc_path        : str           = Field(..., description="Absolute path to PDF/text")
    collection_name : str           = Field(..., description="Qdrant collection target")
    chunker         : str           = Field("fixed_chunking")
    use_incremental : bool          = Field(False, description="Use NB31 incremental ingest")
    tenant_id       : Optional[str] = None

class RetrievalConfig(BaseModel):
    strategy        : str           = Field("dense_retrieval")
    top_k           : int           = Field(5, ge=1, le=50)
    filters         : Dict[str,Any] = Field(default_factory=dict)
    rerank          : bool          = Field(False)
    compress        : bool          = Field(False)

class QueryConfig(BaseModel):
    strategy        : str           = Field("direct", description="direct / decompose / stepback / fusion / cot / react")
    max_sub_queries : int           = Field(4, ge=1, le=10)

class GenerationConfig(BaseModel):
    agentic_mode    : str           = Field("none")
    streaming       : bool          = Field(False)
    use_cache       : bool          = Field(True)
    max_tokens      : int           = Field(1024)
    temperature     : float         = Field(0.1, ge=0.0, le=1.0)

class GuardConfig(BaseModel):
    retrieval_guard : bool          = Field(True)
    generation_guard: bool          = Field(True)
    ambiguity_guard : bool          = Field(True)
    system_guard    : bool          = Field(True)

class TemporalConfig(BaseModel):
    enabled         : bool          = Field(False)
    task_queue      : str           = Field("rag-factory")
    workflow_id_prefix: str         = Field("rag-wf")

class EvaluationConfig(BaseModel):
    enabled         : bool          = Field(False)
    ground_truth    : Optional[str] = None

class PipelineSpec(BaseModel):
    \"\"\"
    Top-level pipeline declaration.
    One YAML file → one validated PipelineSpec → one assembled pipeline.
    Modelled after NVIDIA RAG Factory blueprint configs.
    \"\"\"
    name        : str
    version     : str               = "1.0.0"
    description : str               = ""

    ingestion   : IngestionConfig
    retrieval   : RetrievalConfig   = Field(default_factory=RetrievalConfig)
    query       : QueryConfig       = Field(default_factory=QueryConfig)
    generation  : GenerationConfig  = Field(default_factory=GenerationConfig)
    guards      : GuardConfig       = Field(default_factory=GuardConfig)
    temporal    : TemporalConfig    = Field(default_factory=TemporalConfig)
    evaluation  : EvaluationConfig  = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def validate_combinations(self) -> "PipelineSpec":
        # Streaming is incompatible with Temporal (token stream ≠ durable workflow)
        if self.generation.streaming and self.temporal.enabled:
            raise ValueError(
                "streaming=True is incompatible with temporal.enabled=True. "
                "Streaming responses cannot be durably replayed."
            )
        # Multi-tenant requires tenant_id
        if self.ingestion.collection_name.startswith("tenant_") and not self.ingestion.tenant_id:
            raise ValueError(
                "Collection names starting with 'tenant_' require ingestion.tenant_id to be set."
            )
        # Validate component names exist in manifest
        for field_name, comp_name in [
            ("ingestion.chunker",        self.ingestion.chunker),
            ("retrieval.strategy",       self.retrieval.strategy),
            ("generation.agentic_mode",  self.generation.agentic_mode),
        ]:
            if comp_name != "none":
                try:
                    MANIFEST.get(comp_name)
                except KeyError as e:
                    raise ValueError(f"{field_name}: {e}")
        return self

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineSpec":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineSpec":
        return cls(**d)

    def active_component_names(self) -> List[str]:
        names = [self.ingestion.chunker, self.retrieval.strategy]
        if self.generation.agentic_mode != "none":
            names.append(self.generation.agentic_mode)
        return names

    def active_guards(self) -> List[str]:
        g = self.guards
        out = []
        if g.retrieval_guard : out.append("retrieval_guard")
        if g.generation_guard: out.append("generation_guard")
        if g.ambiguity_guard : out.append("ambiguity_guard")
        if g.system_guard    : out.append("system_guard")
        return out

    def temporal_activities(self) -> List[BaseComponentSpec]:
        \"\"\"Components that should run as Temporal Activities.\"\"\"
        if not self.temporal.enabled:
            return []
        return [MANIFEST.get(n) for n in self.active_component_names()
                if MANIFEST.get(n).is_async]

    def to_yaml(self) -> str:
        return yaml.dump(self.model_dump(), default_flow_style=False, allow_unicode=True)


print("✓ PipelineSpec defined (Pydantic v2, NVIDIA-style blueprint)")
"""))

# ── Cell 13: Section ──────────────────────────────────────────────────────────
cells.append(cell("## Part 6 — YAML Spec Files (ready-to-use pipeline configs)", "markdown"))

# ── Cell 14: Write YAML specs ─────────────────────────────────────────────────
cells.append(cell("""\
import pathlib

SPECS_DIR = pathlib.Path(r"C:/Users/Administrator/RAG/rag_factory/specs")
SPECS_DIR.mkdir(exist_ok=True)

YAML_SPECS = {
    "simple.yaml": \"\"\"
name: simple_pipeline
version: "1.0.0"
description: "Baseline fixed-chunking + dense retrieval. Beginner pipeline."
ingestion:
  doc_path: "C:/Users/Administrator/RAG/data/medicaid.pdf"
  collection_name: "factory_simple"
  chunker: fixed_chunking
  use_incremental: false
retrieval:
  strategy: dense_retrieval
  top_k: 5
  rerank: false
  compress: false
query:
  strategy: direct
generation:
  agentic_mode: none
  streaming: false
  use_cache: true
  max_tokens: 1024
  temperature: 0.1
guards:
  retrieval_guard: true
  generation_guard: true
  ambiguity_guard: false
  system_guard: false
temporal:
  enabled: false
evaluation:
  enabled: false
\"\"\",

    "production.yaml": \"\"\"
name: production_pipeline
version: "1.0.0"
description: "Full Tier 1–7. Semantic chunking + hybrid RRF + rerank + self-RAG + all guards."
ingestion:
  doc_path: "C:/Users/Administrator/RAG/data/medicaid.pdf"
  collection_name: "factory_production"
  chunker: contextual_chunking
  use_incremental: true
retrieval:
  strategy: hybrid_rrf_retrieval
  top_k: 10
  rerank: true
  compress: true
query:
  strategy: decompose
  max_sub_queries: 4
generation:
  agentic_mode: self_rag
  streaming: false
  use_cache: true
  max_tokens: 2048
  temperature: 0.1
guards:
  retrieval_guard: true
  generation_guard: true
  ambiguity_guard: true
  system_guard: true
temporal:
  enabled: true
  task_queue: rag-factory
  workflow_id_prefix: prod-wf
evaluation:
  enabled: true
\"\"\",

    "agentic.yaml": \"\"\"
name: agentic_pipeline
version: "1.0.0"
description: "Iterative RAG with fusion retrieval + generation guard + Temporal durability."
ingestion:
  doc_path: "C:/Users/Administrator/RAG/data/medicaid.pdf"
  collection_name: "factory_agentic"
  chunker: sentence_window_chunking
  use_incremental: false
retrieval:
  strategy: fusion_retrieval
  top_k: 8
  rerank: true
  compress: false
query:
  strategy: react
generation:
  agentic_mode: iterative_rag
  streaming: false
  use_cache: false
  max_tokens: 4096
  temperature: 0.2
guards:
  retrieval_guard: true
  generation_guard: true
  ambiguity_guard: true
  system_guard: false
temporal:
  enabled: true
  task_queue: rag-factory
  workflow_id_prefix: agentic-wf
evaluation:
  enabled: true
\"\"\",

    "streaming.yaml": \"\"\"
name: streaming_pipeline
version: "1.0.0"
description: "Low-latency streaming. Hybrid retrieval + rerank + streaming tokens. No Temporal."
ingestion:
  doc_path: "C:/Users/Administrator/RAG/data/medicaid.pdf"
  collection_name: "factory_streaming"
  chunker: semantic_chunking
  use_incremental: false
retrieval:
  strategy: hybrid_rrf_retrieval
  top_k: 5
  rerank: true
  compress: true
query:
  strategy: direct
generation:
  agentic_mode: none
  streaming: true
  use_cache: true
  max_tokens: 1024
  temperature: 0.1
guards:
  retrieval_guard: true
  generation_guard: false
  ambiguity_guard: true
  system_guard: false
temporal:
  enabled: false
evaluation:
  enabled: false
\"\"\",

    "multitenant.yaml": \"\"\"
name: multitenant_pipeline
version: "1.0.0"
description: "Tier 9 isolation. Mandatory tenant_id on every query. System guard active."
ingestion:
  doc_path: "C:/Users/Administrator/RAG/data/medicaid.pdf"
  collection_name: "factory_multitenant"
  chunker: fixed_chunking
  use_incremental: true
  tenant_id: "tenant_acme"
retrieval:
  strategy: multitenant_rag
  top_k: 5
  rerank: false
  compress: false
query:
  strategy: direct
generation:
  agentic_mode: none
  streaming: false
  use_cache: true
  max_tokens: 1024
  temperature: 0.1
guards:
  retrieval_guard: false
  generation_guard: true
  ambiguity_guard: false
  system_guard: true
temporal:
  enabled: true
  task_queue: rag-factory
  workflow_id_prefix: mt-wf
evaluation:
  enabled: false
\"\"\",
}

for fname, content in YAML_SPECS.items():
    p = SPECS_DIR / fname
    p.write_text(content.strip(), encoding="utf-8")
    print(f"  ✓ Written: {p}")

print(f"\\n✓ {len(YAML_SPECS)} YAML specs written to {SPECS_DIR}")
"""))

# ── Cell 15: Section ──────────────────────────────────────────────────────────
cells.append(cell("## Part 7 — SpecValidator (catch bad configs before execution)", "markdown"))

# ── Cell 16: SpecValidator ────────────────────────────────────────────────────
cells.append(cell("""\
@dataclass
class ValidationResult:
    valid   : bool
    errors  : List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __str__(self):
        lines = [f"Valid: {self.valid}"]
        for e in self.errors  : lines.append(f"  ERROR   : {e}")
        for w in self.warnings: lines.append(f"  WARNING : {w}")
        return "\\n".join(lines)


class SpecValidator:
    \"\"\"
    Pre-flight checks on a PipelineSpec before the assembler runs.
    Catches incompatible combinations, missing guards, Temporal misuse.
    \"\"\"

    INCOMPATIBLE = [
        ("streaming",    "temporal",    "Streaming responses cannot be durably replayed"),
        ("streaming",    "evaluation",  "Evaluation requires full response; incompatible with streaming"),
    ]

    def validate(self, spec: PipelineSpec) -> ValidationResult:
        errs, warns = [], []

        # 1. All named components exist in manifest
        for comp_name in spec.active_component_names():
            try:
                MANIFEST.get(comp_name)
            except KeyError as e:
                errs.append(str(e))

        # 2. Temporal: long-running agentic modes should use Temporal
        if spec.generation.agentic_mode in ("iterative_rag", "recursive_rag", "agentic_rag"):
            if not spec.temporal.enabled:
                warns.append(
                    f"agentic_mode='{spec.generation.agentic_mode}' can run > 5 min. "
                    f"Consider setting temporal.enabled=true for crash recovery."
                )

        # 3. Streaming incompatibilities
        if spec.generation.streaming:
            if spec.temporal.enabled:
                errs.append("streaming=true + temporal.enabled=true are incompatible")
            if spec.evaluation.enabled:
                warns.append("evaluation requires full response; streaming will prevent scoring")

        # 4. Multi-tenant collection without system_guard
        if "tenant" in spec.ingestion.collection_name.lower() and not spec.guards.system_guard:
            warns.append(
                "Collection name suggests multi-tenant use but system_guard=false. "
                "FM-S3 (PII leakage) will not be detected."
            )

        # 5. Contextual chunking without generation guard
        if spec.ingestion.chunker == "contextual_chunking" and not spec.guards.generation_guard:
            warns.append(
                "contextual_chunking produces rich context but generation_guard=false. "
                "FM-G1 (hallucination) will not be checked."
            )

        # 6. No evaluation in production-grade pipelines
        if spec.generation.agentic_mode != "none" and not spec.evaluation.enabled:
            warns.append(
                "Agentic pipeline without evaluation.enabled=true. "
                "RAGAS scores won't be tracked."
            )

        return ValidationResult(valid=len(errs) == 0, errors=errs, warnings=warns)


VALIDATOR = SpecValidator()
print("✓ SpecValidator ready")
"""))

# ── Cell 17: Section ──────────────────────────────────────────────────────────
cells.append(cell("## Part 8 — Live Demo: Validate all 5 pipeline specs", "markdown"))

# ── Cell 18: Validation demo ──────────────────────────────────────────────────
cells.append(cell("""\
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

results = {}
for fname in YAML_SPECS.keys():
    path = str(SPECS_DIR / fname)
    try:
        spec = PipelineSpec.from_yaml(path)
        result = VALIDATOR.validate(spec)
        results[fname] = (spec, result, None)
    except Exception as exc:
        results[fname] = (None, None, str(exc))

# ── Rich table ────────────────────────────────────────────────────────────────
table = Table(title="Pipeline Spec Validation Results", box=box.ROUNDED,
              show_lines=True, min_width=110)
table.add_column("Spec File",       style="bold cyan",   width=22)
table.add_column("Valid",           style="bold",        width=7)
table.add_column("Chunker",         style="dim",         width=24)
table.add_column("Retrieval",       style="dim",         width=22)
table.add_column("Agentic",         style="dim",         width=15)
table.add_column("Temporal",        style="dim",         width=10)
table.add_column("Warnings",        style="yellow",      width=50)

for fname, (spec, result, exc) in results.items():
    if exc:
        table.add_row(fname, "[red]ERROR[/]", "-", "-", "-", "-", f"[red]{exc}[/]")
        continue
    warn_text = "\\n".join(result.warnings) if result.warnings else "[green]none[/]"
    table.add_row(
        fname,
        "[green]✓[/]" if result.valid else "[red]✗[/]",
        spec.ingestion.chunker,
        spec.retrieval.strategy,
        spec.generation.agentic_mode,
        "[green]on[/]" if spec.temporal.enabled else "[dim]off[/]",
        warn_text,
    )

console.print(table)

# ── Temporal activity summary ─────────────────────────────────────────────────
console.print()
console.print(Panel.fit(
    "\\n".join([
        f"[bold]production.yaml[/] → Temporal Activities:",
        *[f"  • {s.name}  (timeout={s.timeout_secs}s, retries={s.max_retries})"
          for s in results['production.yaml'][0].temporal_activities()]
    ]) or "none",
    title="Temporal Activity Mapping",
    border_style="green"
))
"""))

# ── Cell 19: Section ──────────────────────────────────────────────────────────
cells.append(cell("## Part 9 — Save manifest + specs as JSON (consumed by NB35–NB39)", "markdown"))

_CELL20 = (
    'import pathlib, textwrap\n'
    '\n'
    'FACTORY_DIR = pathlib.Path(r"C:/Users/Administrator/RAG/rag_factory")\n'
    '\n'
    '# 1. Export manifest as JSON\n'
    'manifest_json = {s.name: s.model_dump() for s in ALL_SPECS}\n'
    'with open(FACTORY_DIR / "manifest.json", "w", encoding="utf-8") as f:\n'
    '    json.dump(manifest_json, f, indent=2, default=str)\n'
    'print(f"manifest.json   - {len(manifest_json)} components")\n'
    '\n'
    '# 2. Export __init__.py\n'
    'init_lines = [\n'
    '    "from .spec.component import (",\n'
    '    "    BaseComponentSpec, ComponentRole, ChunkStrategy,",\n'
    '    "    RetrievalStrategy, QueryStrategy, AgenticMode, Severity,",\n'
    '    ")",\n'
    '    "from .spec.pipeline import (",\n'
    '    "    PipelineSpec, IngestionConfig, RetrievalConfig,",\n'
    '    "    QueryConfig, GenerationConfig, GuardConfig,",\n'
    '    "    TemporalConfig, EvaluationConfig,",\n'
    '    ")",\n'
    '    "from .spec.manifest import ComponentManifest, MANIFEST, ALL_SPECS",\n'
    '    "from .spec.validator import SpecValidator, ValidationResult, VALIDATOR",\n'
    ']\n'
    'init_src = "\\n".join(init_lines) + "\\n"\n'
    '\n'
    'spec_dir = FACTORY_DIR / "spec"\n'
    'spec_dir.mkdir(exist_ok=True)\n'
    '\n'
    'for mod, src in [\n'
    '    ("component.py", "# stubs - see NB34\\n"),\n'
    '    ("pipeline.py",  "# stubs - see NB34\\n"),\n'
    '    ("manifest.py",  "# stubs - see NB34\\n"),\n'
    '    ("validator.py", "# stubs - see NB34\\n"),\n'
    '    ("__init__.py",  ""),\n'
    ']:\n'
    '    p = spec_dir / mod\n'
    '    if not p.exists():\n'
    '        p.write_text(src, encoding="utf-8")\n'
    '\n'
    '(FACTORY_DIR / "__init__.py").write_text(init_src, encoding="utf-8")\n'
    'print("__init__.py     - factory package stub")\n'
    '\n'
    '# 3. Coverage matrix\n'
    'console.print()\n'
    'console.print("[bold]Failure Mode Guard Coverage Matrix[/]")\n'
    'fm_table = Table(box=box.SIMPLE, show_header=True)\n'
    'fm_table.add_column("FM Code",  style="bold red",   width=10)\n'
    'fm_table.add_column("Category",                      width=12)\n'
    'fm_table.add_column("Guarding Components",            width=50)\n'
    '\n'
    'for cat, codes in [\n'
    '    ("Retrieval",  ["FM-R1","FM-R2","FM-R3","FM-R4","FM-R5","FM-R6"]),\n'
    '    ("Generation", ["FM-G1","FM-G2","FM-G3","FM-G4","FM-G5"]),\n'
    '    ("Ambiguity",  ["FM-A1","FM-A2","FM-A3","FM-A4","FM-A5"]),\n'
    '    ("System",     ["FM-S1","FM-S2","FM-S3","FM-S4","FM-S5"]),\n'
    ']:\n'
    '    for code in codes:\n'
    '        guards = MANIFEST.covering_fm(code)\n'
    '        fm_table.add_row(code, cat, ", ".join(s.name for s in guards) or "[red]UNGUARDED[/]")\n'
    '\n'
    'console.print(fm_table)\n'
    'print()\n'
    'print("NB34 complete - Spec Layer ready for NB35 (Component Registry)")\n'
)
# ── Cell 20: Export ───────────────────────────────────────────────────────────
cells.append(cell(_CELL20))

# ── Cell 21: Summary ──────────────────────────────────────────────────────────
cells.append(cell("""\
## Summary — NB34 Deliverables

| Artifact | Location | Consumed by |
|---|---|---|
| `BaseComponentSpec` | This notebook (in-memory) | NB35 component wrappers |
| `ComponentManifest` + `MANIFEST` | `rag_factory/manifest.json` | NB35, NB36, NB37 |
| `PipelineSpec` | This notebook | NB36 assembler |
| `SpecValidator` | This notebook | NB36 assembler, NB39 API |
| `simple.yaml` | `rag_factory/specs/` | NB36 demo |
| `production.yaml` | `rag_factory/specs/` | NB36 demo |
| `agentic.yaml` | `rag_factory/specs/` | NB36 demo |
| `streaming.yaml` | `rag_factory/specs/` | NB38 API |
| `multitenant.yaml` | `rag_factory/specs/` | NB38 API |

### What NB35 will build
Extract the 33 notebook patterns into **spec-compliant Python modules** under
`rag_factory/components/` — so the assembler can `import` and chain them
directly from a `PipelineSpec`.
""", "markdown"))

# ── Assemble & write notebook ─────────────────────────────────────────────────
nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.0"},
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Written: {NB_PATH}")
print(f"Cells  : {len(cells)}")
