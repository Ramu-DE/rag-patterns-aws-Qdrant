# Notes: Build Production-Ready AI-Powered Search Applications with AWS
**Source:** AWS Slide Deck — Srinivas Margasahayam & Prashant Agrawal (AWS OpenSearch Specialist SAs)

---

## 1. What Is OpenSearch?

OpenSearch is an **open-source, community-driven platform** for search, analytics, and vector storage.
It's not just a search engine — it's a full platform for:
- Full-text and semantic search
- Log analytics / observability
- Vector database operations
- AI-powered applications (RAG, recommendations)

**By the numbers:**
- 1.4B+ project downloads, 3K+ active contributors, 400+ active organizations
- Amazon OpenSearch Service wraps it as a fully managed, serverless-capable cloud service

**Why use the managed service (Amazon OpenSearch Service)?**
| Feature | What it gives you |
|---------|------------------|
| Automated | API-driven deployment, self-healing, auto-upgrades |
| Cost-optimized | Reserved instances (52% discount), tiered storage, serverless autoscale |
| Resilient | Multi-zone, 99.99% SLA, hourly snapshots |
| Secure | Fine-grained access, encryption at rest + in flight, SAML/IAM/Cognito |
| Integrated | Native connectors to DynamoDB, DocumentDB, S3, CloudWatch Logs |

---

## 2. The Evolution of Search — Four Generations

Think of search maturity as going deeper in the ocean:

```
Surface    →  Twilight   →   Midnight   →   Abyss
Keyword        Semantic       Hybrid         Agentic
BM25/TF-IDF    ML Vectors     Best of both   AI Agents + LLMs
```

### Generation 1 — Keyword (Lexical) Search
**Intuition:** A librarian who can only match the exact words you say.

- Matches terms in an inverted index (fast, interpretable)
- Scores by term frequency × inverse document frequency (BM25)
- **Fails on:** synonyms (`"heart attack"` ≠ `"myocardial infarction"`), conceptual queries (`"things to do when bored"`), ambiguous words (`"Python"` — snake or language?)

### Generation 2 — Semantic Search
**Intuition:** A librarian who understands what you *mean*, not just what you *said*.

- Converts text → dense or sparse vector → finds geometrically close vectors
- Handles synonyms, paraphrases, cross-language queries
- **Fails on:** exact codes (`ERR-4032`), domain jargon without fine-tuning, strict date/number filtering

### Generation 3 — Hybrid Search (State of the Art Today)
**Intuition:** Run keyword AND semantic in parallel, then blend the scores.

- Keyword score (BM25) + Semantic score (cosine/dot product) → normalize → combine
- Blending options:
  - **Arithmetic mean** — equal weight to both
  - **Weighted sum** — tunable, e.g. `0.6 × semantic + 0.4 × keyword`
  - **Harmonic mean** — penalizes large discrepancies between the two scores
- Consistently outperforms either method alone

### Generation 4 — Agentic Search (OpenSearch 3.3+)
**Intuition:** An AI agent that plans its own search strategy, not just executes one.

- Natural language input → agent decides which tools to call → reasons → answers
- Powered by OpenSearch ML Commons: neural search + flow pipelines + agents + persistent memory
- Still emerging; the future direction

---

## 3. Vectors 101 — The Core Intuition

### What is a vector?
A vector is just **a list of numbers that describes something**.

```
"I am attending a great webinar about vector search"
        ↓  (embedding model)
[0.743, 0.720, -0.325, 0.195, 0.835, -0.945, ...]
```

**Concrete example — a house:**
```
v1 = [5 bedrooms,  4 bathrooms,  854 sqm, $1.1M]
v2 = [4 bedrooms,  3 bathrooms,  335 sqm, $530K]
v3 = [6 bedrooms,  4 bathrooms,  530 sqm, $2.1M]
```
v1 and v3 are closer together in vector space than v1 and v2 — and they *are* more similar houses.
That's the whole idea. Put similar things close together numerically.

### What makes a good embedding?
Two properties matter:
- **Magnitude** — how strong the signal is
- **Direction** — which "concept space" the vector points toward

Unit-normalised embeddings (e.g. Bedrock Titan V2) have magnitude = 1.0, so similarity is entirely captured by direction (the angle between vectors).

### What is a vector database?
A store that lets you:
1. **Upsert** high-dimensional vectors (e.g. 1024-dim)
2. **Query** by nearest-neighbour — "find the K vectors most geometrically similar to this query vector"
3. **Filter** by metadata alongside vector similarity (e.g. `price < $2M AND similar_to_query`)

Any data type can be embedded: images, text, audio, structured records.

---

## 4. Similarity Metrics — How to Measure "Close"

| Metric | Formula | Best for |
|--------|---------|----------|
| **Cosine similarity** | `1 − (A·B)/(‖A‖·‖B‖)` — angle between vectors | Text / semantic search, document classification |
| **Euclidean (L2)** | Straight-line distance | Counts, measurements, recommendation systems |
| **Dot product** | `A·B` | Collaborative filtering; equivalent to cosine when unit-normalised |

**Intuition for cosine:** Two documents can use completely different words but if they *mean* the same thing, their vectors point in the same direction — cosine ≈ 1.

---

## 5. k-Nearest Neighbours (kNN) — Finding the Top-K Similar

### Exact kNN (brute-force)
- Compare query vector against every stored vector, rank by distance
- **Accuracy:** perfect  |  **Latency:** O(N) — seconds at scale
- Use only for small corpora or offline batch jobs

### Approximate kNN — HNSW (Hierarchical Navigable Small World)
**Intuition:** Build a multi-layer "express highway" graph over your vectors.

```
Layer 2  (coarse, few nodes)  → entry point, fast long jumps
Layer 1  (medium density)     → narrow the search neighbourhood
Layer 0  (all nodes)          → precise local search
```

At query time: enter at the top layer, greedily hop toward the query, descend, refine.
- **Latency:** milliseconds  |  **Accuracy:** 95–99% recall of exact kNN
- Trades a small recall loss for orders-of-magnitude speed gain

### Key HNSW knobs

| Parameter | Effect | Rule of thumb |
|-----------|--------|---------------|
| `ef_construction` | Index build quality — higher = better graph, slower build | Set once at index creation |
| `m` (graph connectivity) | More edges = better recall, more memory + slower indexing | Default 16; try 32 for high-recall needs |
| `ef_search` | Query-time candidate pool size — **the most undertune knob** | Start at 100, raise until recall plateaus |

---

## 6. Dense vs Sparse Encoding — Two Flavours of Semantic Search

### Dense Encoding
**Intuition:** Compress the entire meaning of a sentence into a fixed-length bullet.

- Model (BERT, Cohere, Amazon Titan, OpenAI) → fixed-length vector (384–1024 dims)
- Every dimension is non-zero; meaning distributed across all dimensions
- Stored in a vector index (HNSW, IVF)
- Similarity via cosine / dot product / L2 at query time
- Powers RAG pipelines, cross-lingual search
- **Memory-heavy** — every vector stored in RAM for fast retrieval

### Sparse Encoding
**Intuition:** Expand a document with weighted synonyms and store them in the regular word index.

- Model creates weighted term expansions: `"car"` → `{automobile: 0.8, vehicle: 0.6, sedan: 0.4, ...}`
- 30,522 dims (BERT vocab) but only ~1% non-zero per doc → uses existing Lucene inverted index
- **10× smaller index** than dense, **zero RAM overhead at query time**
- Graceful fallback: never worse than BM25 (keyword)
- Fully interpretable via the explain API
- Pre-trained model available: `opensearch-neural-sparse-v2`
- Two-phase acceleration (OpenSearch 2.15+): fast first pass, refine top-N second pass

### When to use which?
| Situation | Use |
|-----------|-----|
| Deep semantic similarity, cross-lingual, RAG | Dense |
| Speed + cost critical, interpretability needed | Sparse |
| Best results, don't want to choose | Hybrid (both in parallel) |

---

## 7. Why No Single Method Always Works

| Scenario | Keyword fails | Semantic fails |
|----------|--------------|----------------|
| `"Best laptop for students"` | "best" has no exact match | — |
| `"Heart attack"` vs `"myocardial infarction"` | Zero overlap | — |
| Product SKU `ERR-4032` | — | No training signal for arbitrary codes |
| Legal compliance exact phrasing | — | Model paraphrases, loses precision |
| `"Python"` — snake or language? | Ambiguous | Context-dependent; may still struggle |

**The takeaway:** No single method covers all scenarios. The best production systems combine approaches intelligently.

---

## 8. Tiered Vector Storage — Cost vs Speed

```
Tier            Storage      Cost     Speed       When to use
──────────────────────────────────────────────────────────────
Exact kNN       RAM          $$$$     Slowest      Small datasets needing perfect recall
In-memory       RAM          $$$      Fastest      Hot data, low-latency SLAs
Disk mode       SSD          $$       Good         Most production workloads
S3 vectors      S3           $        Slowest      Cold/archive data, massive scale
```

**The triangle trade-off — you can only pick two:**
- Minimize cost + maintain recall → use disk + 32× binary quantization
- Maximize speed + recall → in-memory HNSW, no compression
- Balance all three → in-memory + 4× scalar quantization

---

## 9. Quantization — Shrinking Vectors Without Killing Recall

| Method | Memory reduction | Recall loss |
|--------|-----------------|-------------|
| FP32 → FP16 | 2× | ~0% |
| FP32 → INT8 (scalar) | 4× | ~1% |
| FP32 → Binary + rescoring | 32× | <2% |

**Key insight:** Binary quantization with a rescoring pass (re-check top-N results with original FP32 vectors) gives 97% memory reduction with less than 2% recall loss. That is often the right default for production.

---

## 10. Full Architecture — How AI Search Works End-to-End

### Ingestion Flow
```
Source Data (text, images, PDFs)
    │
    ▼
Embedding Model (Bedrock Titan, Cohere, all-MiniLM, etc.)
    │  Produces fixed-length dense vectors
    ▼
{ "title": "...", "plot": "...", "title_v": [0.006, 0.007, ...] }
    │
    ▼
OpenSearch  (create knn_vector mapping → bulk index)
```

**Ingestion code pattern:**
```python
# 1. Call Bedrock for embedding
response = bedrock_runtime.invoke_model(
    modelId='amazon.titan-embed-text-v1',
    body=json.dumps({'inputText': text})
)
embeddings = json.loads(response['body'].read())['embedding']

# 2. Create index with knn_vector field
index_body = {
    "settings": {"index.knn": True},
    "mappings": {"properties": {
        "v_title": {"type": "knn_vector", "dimension": 1536}
    }}
}

# 3. Bulk index
client.bulk(body=actions)
```

### Search Flow
```
User Query: "uplifting underdog story"
    │
    ▼
Same Embedding Model → [0.0048, 0.0089, ...]
    │
    ▼
kNN Query to OpenSearch
    {
      "knn": { "v_title": { "vector": query_embedding, "k": 5 } },
      "filter": [{ "range": { "rating": { "gte": 8 } } }]
    }
    │
    ▼
Top-K most semantically similar results returned
```

**Important:** Always use the **same embedding model** for indexing and querying — mismatched models produce garbage results.

### End-to-end System Architecture
```
GitHub repo / documents
    │  EC2 crawls & loads embedding model
    ▼
Amazon EC2  →  Bedrock Titan (generate embeddings)
    │
    ▼
Amazon OpenSearch Serverless  (store + index vectors)
    │  kNN search on query vector
    ▼
User (web page / Streamlit app)
```

---

## 11. Amazon OpenSearch Serverless

**Problem it solves:** Provisioned clusters require you to size shards, manage node counts, handle scaling — expertise most teams don't have.

**Serverless model:**
- You create a **collection** (time-series or search type)
- AWS auto-scales **OCUs** (OpenSearch Compute Units) based on actual traffic
- Modular architecture: separate indexing nodes ↔ search nodes ↔ S3 backing store
- Pay only for OCUs consumed, not for idle capacity

**Capabilities (recent innovations):**
- 100 TB time-series collections
- 22 regions (expanded from 15)
- Data-plane audit logging
- Snapshot restore

---

## 12. Cost Optimization — Strategies That Actually Work

**50–97% memory savings** are achievable without meaningful recall loss:

1. **Quantization cascade** — FP32 → FP16 → INT8 → Binary. Start at INT8 (4× reduction, ~1% loss).
2. **Disk-based vectors** (OpenSearch 2.17+) — move vectors from RAM to SSD, up to 32× memory reduction.
3. **Sparse encoding in doc-only mode** — zero RAM at query time, uses free Lucene inverted index infrastructure.
4. **Right-size dimensions** — if your model supports Matryoshka embeddings, 768→384 dims cuts storage 50%, ~1% recall loss.
5. **Tiered storage** — hot vectors in memory, warm on disk, cold on S3. Match tier to access frequency.
6. **Serverless OCU auto-scaling** — pay for actual compute, not reserved headroom.

---

## 13. Performance Tuning — Latency Deep Dive

### HNSW Parameter Tuning
```
ef_construction  →  build-time only; higher = better recall, slower index build
m                →  graph connectivity; default 16 is fine, try 32 for high-recall
ef_search        →  THE MOST UNDER-TUNED KNOB. Start at 100, raise until recall plateaus.
```

### Query-Time Tricks
| Technique | Impact |
|-----------|--------|
| **Pre-filtering** (filter before kNN, not after) | Dramatically reduces candidate set → faster search. Use when filter selectivity is high. |
| **ANN vs exact kNN** | ANN (HNSW) = ms latency; exact = seconds. Use exact only when recall must be 100%. |
| **Request cache** | Cache repeated semantic queries — instant response for common queries. |
| **Force-merge to 1 segment** | After bulk indexing, merging to one Lucene segment reduces search latency 20–40%. |

---

## 14. Recall Optimization — Getting Better Results

| Technique | Lift |
|-----------|------|
| Hybrid BM25 + kNN | Consistently beats either alone |
| Domain fine-tuning (5K examples) | +10–15% recall vs larger generic model |
| Reranking (cross-encoder, Cohere, BGE) | Improves precision without changing the index |
| Chunking strategy (256–512 tokens, 10–20% overlap) | Better context preservation |
| Multi-vector per document (title + body + summary separately) | Catches queries that match one field but not others |
| Metadata filtering: narrow first, then semantic | Avoids wasting kNN on irrelevant documents |

**On chunking:** Too large = diluted semantics (one chunk covers too many topics). Too small = lost context (a sentence without its neighbours is meaningless). 256–512 tokens with 10–20% overlap is the sweet spot.

---

## 15. Monitoring — What to Watch

### Quality & Relevance Metrics
| Metric | What it tells you |
|--------|------------------|
| **Recall@K** | Are your top-K results actually relevant? Needs labeled data or click-through signals. |
| **NDCG / MRR** | Ranking quality — position matters, not just presence in top-K |
| **A/B testing / CTR** | Click-through rate is the ultimate real-world signal |
| **Cost per relevant result** | Combines all three triangle dimensions; best single business metric |

### Operational Metrics
| Metric | Why it matters |
|--------|----------------|
| Latency P50/P95/P99 | Track tail latency — P99 tells you the worst user experience |
| Index size & segment count | Directly correlates with query latency |
| OCU utilization (Serverless) | Are you over-provisioned or getting throttled? |
| **Embedding latency** | Bedrock/model inference is often the **real** bottleneck, not the search itself |
| Query volume patterns | Burst traffic → pre-warm indexes |

---

## 16. Decision Framework — Choosing Your Stack

### Step 1: Define your latency SLA
P99 < 200ms? < 500ms? < 2s? This determines whether you can afford exact kNN or must use ANN.

### Step 2: Set minimum acceptable recall
Recall@10 > 0.85? > 0.95? Higher recall = higher cost + more tuning work.

### Step 3: Cost becomes the optimization variable once latency + recall are met.

### By use case:
```
E-commerce (low latency + high recall)
  → In-memory HNSW, pre-filtering, aggressive caching

Internal Knowledge Base (cost matters more)
  → Disk-based vectors, sparse encoding, OCU auto-scale

Legal / Compliance (maximum recall, cost secondary)
  → Exact kNN, hybrid search, reranking pipeline

Chatbot / RAG (balance all three)
  → Hybrid search + caching, domain fine-tuned model,
    quantized vectors, tiered storage
```

---

## 17. The Search System Design Triangle — Mental Model

```
              COST
             /    \
            /  pick \
           /   two   \
         RECALL ── LATENCY
```

- **Maximize recall + minimize latency** → accept high cost (in-memory HNSW, no compression)
- **Minimize cost + maintain recall** → accept higher latency (disk + binary quantization)
- **Minimize latency + minimize cost** → accept lower recall (sparse encoding, aggressive quantization)

There is no free lunch. Every production system is a deliberate point in this triangle.

---

## 18. Workshop — What the Hands-On Builds

**Goal:** Build a movie search app comparing keyword vs semantic search side-by-side.

**Stack:** EC2 + Amazon Bedrock Titan Embeddings + Amazon OpenSearch Serverless + Streamlit

**Three modules:**
1. **Setup** — Update security groups (ports 22 + 8080), connect to EC2 via Instance Connect, verify `$AOSS_VECTORSEARCH_ENDPOINT`
2. **Load Data** — Run `movies_loader.py` — generates Titan embeddings, bulk-indexes movies into OpenSearch Serverless
3. **Search & Visualize** — Run `vector_search_bedrock.py` for CLI queries; launch Streamlit on port 8080 to compare keyword vs semantic visually

---

## Key Takeaways — One-Liners

1. **Embeddings = numbers that encode meaning.** Geometric proximity = semantic similarity.
2. **Keyword search matches characters; semantic search matches concepts.** Both have blind spots.
3. **Hybrid search wins** — run BM25 + kNN in parallel, normalize scores, combine. Best of both worlds.
4. **HNSW is the workhorse** — millisecond approximate nearest-neighbour. ef_search is the most under-tuned parameter.
5. **Quantization is free money** — binary + rescoring gives 32× memory reduction with <2% recall loss.
6. **Embedding latency is often the bottleneck**, not the vector search itself. Profile before optimizing.
7. **The triangle is real** — cost, latency, and recall. Every architecture decision is a trade-off between these three.
8. **Domain fine-tuning on 5K examples beats a larger generic model** — smaller and faster, better for your specific domain.
9. **Chunk at 256–512 tokens with 10–20% overlap** — the empirically proven sweet spot for RAG.
10. **Click-through rate is the ultimate evaluation metric** — labeled relevance and NDCG are proxies; real user behavior is truth.
