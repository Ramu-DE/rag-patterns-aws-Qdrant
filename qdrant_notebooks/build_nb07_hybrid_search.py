"""Build 07_Hybrid_Search.ipynb — full implementation."""
import json, uuid, ast

PATH = r"C:\Users\Administrator\RAG\qdrant_notebooks\tier2_retrieval_quality\07_Hybrid_Search.ipynb"
PDF  = r"C:\Users\Administrator\RAG\data\climate.pdf"

def cid(): return str(uuid.uuid4())[:8]
def md(text):
    return {"cell_type":"markdown","id":cid(),"metadata":{},"source":text.splitlines(keepends=True)}
def code(lines):
    return {"cell_type":"code","id":cid(),"metadata":{},"execution_count":None,"outputs":[],"source":lines}
def L(*lines):
    out = [l+"\n" for l in lines]
    if out: out[-1] = out[-1].rstrip("\n")
    return out

cells = []

# ── Overview ──────────────────────────────────────────────────────────────────
cells.append(md(
"# 07 — Hybrid Search\n"
"\n"
"> **Tier 2 | Retrieval Quality**\n"
"\n"
"## What is Hybrid Search?\n"
"\n"
"Pure vector search excels at semantic similarity but **misses exact-term matches**.\n"
"Pure BM25 keyword search nails exact terms but **misses paraphrases and synonyms**.\n"
"\n"
"**Hybrid Search** runs both in parallel and merges the ranked lists:\n"
"\n"
"| Method | Strength | Weakness |\n"
"|--------|----------|----------|\n"
"| **BM25** (sparse) | Exact terms, names, IDs, rare words | Misses synonyms, paraphrases |\n"
"| **Vector** (dense) | Semantic similarity, synonyms | Misses rare exact terms |\n"
"| **Hybrid** | Both | Slightly higher latency |\n"
"\n"
"## Fusion strategy — Reciprocal Rank Fusion (RRF)\n"
"\n"
"RRF merges two ranked lists without needing score normalisation:\n"
"\n"
"```\n"
"RRF_score(doc) = 1/(k + rank_bm25) + 1/(k + rank_vector)\n"
"```\n"
"\n"
"where `k=60` dampens the impact of very high ranks. Docs appearing in both\n"
"lists get a double boost; docs in only one list still contribute.\n"
"\n"
"## BM25 — pure Python, no Elasticsearch\n"
"\n"
"BM25 is implemented from scratch using term-frequency / inverse-document-frequency.\n"
"No external search engine required — just Python and the existing Qdrant collection.\n"
))

# ── Flow diagram ──────────────────────────────────────────────────────────────
cells.append(md(
"## Flow Diagram\n"
"\n"
"```mermaid\n"
"flowchart TD\n"
"    subgraph INDEX [\"🔵  INDEXING\"]\n"
"        PDF([\"📄 climate.pdf\"])\n"
"        PDF --> SPLIT[\"Fixed-size chunks\\n~500 chars\"]\n"
"        SPLIT --> BM25_IDX[(\"BM25 index\\n(in-memory)\")]\n"
"        SPLIT --> EMB[\"Embed chunks\\nTitan V2\"]\n"
"        EMB --> QDRANT[(\"Qdrant collection\")]\n"
"    end\n"
"\n"
"    subgraph RETRIEVAL [\"🟢  HYBRID RETRIEVAL\"]\n"
"        Q([\"❓ User query\"])\n"
"        Q --> BM25_S[\"BM25 keyword search\\n→ ranked list A\"]\n"
"        Q --> VEC_S[\"Vector search\\n→ ranked list B\"]\n"
"        BM25_IDX --> BM25_S\n"
"        QDRANT --> VEC_S\n"
"        BM25_S --> RRF[\"Reciprocal Rank Fusion\\nRRF(doc) = 1/(k+rank_A)\\n        + 1/(k+rank_B)\"]\n"
"        VEC_S --> RRF\n"
"        RRF --> TOPK[\"Top-K merged results\"]\n"
"    end\n"
"\n"
"    subgraph GENERATION [\"🟠  GENERATION\"]\n"
"        TOPK --> LLM[\"Strands Agent\\nClaude Sonnet 4.6\"]\n"
"        LLM --> ANS([\"✅ Answer\"])\n"
"    end\n"
"\n"
"    subgraph COMPARE [\"📊  A/B COMPARE\"]\n"
"        Q2([\"Same query\"])\n"
"        Q2 --> ONLY_VEC[\"Vector only\"]\n"
"        Q2 --> ONLY_BM[\"BM25 only\"]\n"
"        Q2 --> HYBRID[\"Hybrid (RRF)\"]\n"
"        ONLY_VEC --> SCORES[\"Compare top-1 scores\\nand answer quality\"]\n"
"        ONLY_BM --> SCORES\n"
"        HYBRID --> SCORES\n"
"    end\n"
"\n"
"    style INDEX      fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f\n"
"    style RETRIEVAL  fill:#dcfce7,stroke:#16a34a,color:#14532d\n"
"    style GENERATION fill:#ffedd5,stroke:#f97316,color:#7c2d12\n"
"    style COMPARE    fill:#faf5ff,stroke:#a855f7,color:#3b0764\n"
"```\n"
))

# ── Step 1 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 1 — Install & Import Dependencies"))

cells.append(code(L(
    "import subprocess, sys",
    'packages = ["boto3","qdrant-client","opensearch-py","requests-aws4auth","strands-agents","pypdf"]',
    'subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + packages)',
    'print("All packages ready.")',
)))

cells.append(code(L(
    "import os, json, time, uuid, re, math",
    "from collections import defaultdict",
    "from typing import List, Dict, Tuple",
    "",
    "import boto3, pypdf",
    "from strands import Agent",
    "from strands.models.bedrock import BedrockModel",
    "from qdrant_client import QdrantClient",
    "from qdrant_client.models import Distance, VectorParams, PointStruct",
    "",
    'print("Imports OK")',
)))

# ── Step 2 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 2 — Configuration"))

cells.append(code(L(
    "# AWS / Bedrock",
    'AWS_REGION      = os.getenv("AWS_DEFAULT_REGION", "us-east-1")',
    'EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"',
    'LLM_MODEL       = "us.anthropic.claude-sonnet-4-6"',
    "",
    "# Vector DB",
    'QDRANT_URL     = os.getenv("QDRANT_URL", "")',
    'QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")',
    'OPENSEARCH_URL = os.getenv("OPENSEARCH_ENDPOINT", "")',
    "",
    "# Collection",
    'COLLECTION_NAME = "hybrid_search_07"',
    "EMBEDDING_DIM   = 1024",
    "",
    "# Retrieval",
    "TOP_K_EACH = 20   # candidates from each method before fusion",
    "TOP_K_FINAL = 5   # final results returned to LLM",
    "RRF_K      = 60   # RRF damping constant",
    "",
    "# Chunking",
    "CHUNK_SIZE    = 500",
    "CHUNK_OVERLAP = 50",
    "",
    "# BM25 parameters",
    "BM25_K1 = 1.5   # term-frequency saturation",
    "BM25_B  = 0.75  # length normalisation",
    "",
    "# PDF",
    'PDF_PATH = r"' + PDF + '"',
    "",
    'print(f"Collection : {COLLECTION_NAME}")',
    'print(f"Top-K each : {TOP_K_EACH}  |  Final top-K: {TOP_K_FINAL}")',
    'print(f"RRF k      : {RRF_K}")',
    'print(f"BM25 k1={BM25_K1}, b={BM25_B}")',
    'print(f"PDF        : {PDF_PATH}")',
    'print(f"PDF exists : {os.path.exists(PDF_PATH)}")',
)))

# ── Step 3 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 3 — Vector Store"))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "class VectorStore:",
    "    def __init__(self, collection_name, qdrant_url='', qdrant_api_key='',",
    "                 opensearch_url='', region='us-east-1'):",
    "        self.name = collection_name",
    "        self._backend = None",
    "        if qdrant_url:",
    "            try:",
    "                kw = {'url': qdrant_url}",
    "                if qdrant_api_key: kw['api_key'] = qdrant_api_key",
    "                self._qdrant = QdrantClient(**kw)",
    "                self._qdrant.get_collections()",
    "                self._backend = 'qdrant_cloud'",
    "                print(f'Qdrant Cloud: {qdrant_url}')",
    "                return",
    "            except Exception as e:",
    "                print(f'Qdrant Cloud unavailable ({e}), trying next...')",
    "        if opensearch_url:",
    "            try:",
    "                from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth",
    "                creds = boto3.Session().get_credentials()",
    "                auth  = AWSV4SignerAuth(creds, region, 'aoss')",
    "                host  = opensearch_url.replace('https://','').replace('http://','')",
    "                self._os = OpenSearch(",
    "                    hosts=[{'host': host, 'port': 443}],",
    "                    http_auth=auth, use_ssl=True, verify_certs=True,",
    "                    connection_class=__import__('opensearchpy').RequestsHttpConnection,",
    "                    timeout=30)",
    "                self._os.info()",
    "                self._backend = 'opensearch'",
    "                print(f'OpenSearch: {opensearch_url}')",
    "                return",
    "            except Exception as e:",
    "                print(f'OpenSearch unavailable ({e}), falling back...')",
    "        self._qdrant  = QdrantClient(':memory:')",
    "        self._backend = 'qdrant_memory'",
    "        print('Using Qdrant in-memory')",
    "",
    "    def create_collection(self, dim=1024, force_recreate=False):",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            exists = self.name in [c.name for c in self._qdrant.get_collections().collections]",
    "            if exists and force_recreate:",
    "                self._qdrant.delete_collection(self.name); exists = False",
    "            if not exists:",
    "                self._qdrant.create_collection(",
    "                    self.name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))",
    "                print(f'Created \"{self.name}\" (dim={dim})')",
    "            else:",
    "                print(f'\"{self.name}\" already exists')",
    "        elif self._backend == 'opensearch':",
    "            if force_recreate and self._os.indices.exists(index=self.name):",
    "                self._os.indices.delete(index=self.name)",
    "            if not self._os.indices.exists(index=self.name):",
    "                self._os.indices.create(index=self.name, body={",
    "                    'settings': {'index': {'knn': True}},",
    "                    'mappings': {'properties': {",
    "                        'text':      {'type': 'text'},",
    "                        'metadata':  {'type': 'object'},",
    "                        'embedding': {",
    "                            'type': 'knn_vector', 'dimension': dim,",
    "                            'method': {'name': 'hnsw', 'space_type': 'cosinesimil',",
    "                                       'engine': 'faiss',",
    "                                       'parameters': {'ef_construction': 512, 'm': 16}}",
    "                        }}}})",
    "                print(f'Created OpenSearch index \"{self.name}\"')",
    "",
    "    def upsert(self, docs: List[Dict]) -> int:",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            pts = [PointStruct(id=str(uuid.uuid4()), vector=d['embedding'],",
    "                               payload={'text': d['text'], 'metadata': d.get('metadata',{})})",
    "                   for d in docs]",
    "            self._qdrant.upsert(collection_name=self.name, points=pts)",
    "            return len(pts)",
    "        elif self._backend == 'opensearch':",
    "            for d in docs: self._os.index(index=self.name, body=d)",
    "            time.sleep(1)",
    "            return len(docs)",
    "",
    "    def search_vector(self, qvec: List[float], top_k: int = 20) -> List[Dict]:",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            resp = self._qdrant.query_points(",
    "                collection_name=self.name, query=qvec, limit=top_k, with_payload=True)",
    "            return [{'text': p.payload.get('text',''),",
    "                     'metadata': p.payload.get('metadata',{}),",
    "                     'score': p.score, 'id': str(p.id)}",
    "                    for p in resp.points]",
    "        elif self._backend == 'opensearch':",
    "            resp = self._os.search(index=self.name, body={",
    "                'size': top_k,",
    "                'query': {'knn': {'embedding': {'vector': qvec, 'k': top_k}}},",
    "                '_source': {'excludes': ['embedding']}})",
    "            return [{'text': h['_source'].get('text',''),",
    "                     'metadata': h['_source'].get('metadata',{}),",
    "                     'score': h['_score'], 'id': h['_id']}",
    "                    for h in resp['hits']['hits']]",
    "",
    "    def count(self) -> int:",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            return self._qdrant.get_collection(self.name).points_count or 0",
    "        return 0",
    "",
    "    def delete_collection(self):",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            self._qdrant.delete_collection(self.name)",
    "        elif self._backend == 'opensearch':",
    "            self._os.indices.delete(index=self.name, ignore=[404])",
    "        print(f'Deleted \"{self.name}\"')",
    "",
    'print("VectorStore defined.")',
)))

# ── Step 4 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 4 — Bedrock Helpers"))

cells.append(code(L(
    "from typing import List",
    "",
    "bedrock_rt = boto3.client('bedrock-runtime', region_name=AWS_REGION)",
    "",
    "def embed_text(text: str) -> List[float]:",
    '    body = json.dumps({"inputText": text, "dimensions": EMBEDDING_DIM, "normalize": True})',
    "    resp = bedrock_rt.invoke_model(",
    "        modelId=EMBEDDING_MODEL, body=body,",
    '        contentType="application/json", accept="application/json")',
    '    return json.loads(resp["body"].read())["embedding"]',
    "",
    "def embed_batch(texts: List[str], label: str = '') -> List[List[float]]:",
    "    out = []",
    "    for i, t in enumerate(texts):",
    "        out.append(embed_text(t))",
    "        if (i + 1) % 20 == 0:",
    "            print(f'  {label} {i+1}/{len(texts)}')",
    "        time.sleep(0.04)",
    "    return out",
    "",
    "_model = BedrockModel(model_id=LLM_MODEL, region_name=AWS_REGION)",
    "",
    "def generate_answer(question: str, context_chunks: List[str]) -> str:",
    "    context = '\\n\\n'.join(f'[Chunk {i+1}]\\n{c}' for i, c in enumerate(context_chunks))",
    "    prompt  = (",
    "        'Use ONLY the context below to answer. '",
    "        \"If the answer is not present say 'Not found in context.'\\n\\n\"",
    "        f'Context:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:'",
    "    )",
    "    return str(Agent(",
    "        model=_model,",
    "        system_prompt='You are a precise Q&A assistant. Answer only from the provided context.'",
    "    )(prompt))",
    "",
    'test_emb = embed_text("hybrid search BM25 vector")',
    'print(f"Embedding OK — dim={len(test_emb)}")',
    'print("Strands BedrockModel ready.")',
)))

# ── Step 5 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 5 — Connect & Create Collection"))

cells.append(code(L(
    "vs = VectorStore(",
    "    collection_name=COLLECTION_NAME,",
    "    qdrant_url=QDRANT_URL,",
    "    qdrant_api_key=QDRANT_API_KEY,",
    "    opensearch_url=OPENSEARCH_URL,",
    "    region=AWS_REGION",
    ")",
    'print(f"Backend: {vs._backend}")',
    "vs.create_collection(dim=EMBEDDING_DIM, force_recreate=True)",
)))

# ── Step 6 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 6 — Load PDF & Build Chunks"))

cells.append(code(L(
    "from typing import List",
    "",
    "def recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:",
    '    separators = ["\\n\\n", "\\n", ". ", " ", ""]',
    "    def _split(text, seps):",
    "        sep, new_seps = '', []",
    "        for i, s in enumerate(seps):",
    "            if s == '' or s in text:",
    "                sep, new_seps = s, seps[i+1:]; break",
    "        parts = text.split(sep) if sep != '' else list(text)",
    "        good = []",
    "        for part in parts:",
    "            if len(part) <= chunk_size: good.append(part)",
    "            elif new_seps: good.extend(_split(part, new_seps))",
    "            else:",
    "                for k in range(0, len(part), chunk_size - chunk_overlap):",
    "                    good.append(part[k:k+chunk_size])",
    "        chunks, cur_pieces, cur_len = [], [], 0",
    "        for piece in good:",
    "            p = piece.strip()",
    "            if not p: continue",
    "            addition = len(sep) + len(p) if cur_pieces else len(p)",
    "            if cur_len + addition <= chunk_size:",
    "                cur_pieces.append(p); cur_len += addition",
    "            else:",
    "                if cur_pieces: chunks.append(sep.join(cur_pieces))",
    "                ov, ovl = [], 0",
    "                for prev in reversed(cur_pieces):",
    "                    if ovl + len(prev) + len(sep) <= chunk_overlap:",
    "                        ov.insert(0, prev); ovl += len(prev) + len(sep)",
    "                    else: break",
    "                cur_pieces = ov + [p]",
    "                cur_len = sum(len(x) + len(sep) for x in cur_pieces)",
    "        if cur_pieces: chunks.append(sep.join(cur_pieces))",
    "        return [c for c in chunks if c.strip()]",
    "    return _split(text, separators)",
    "",
    "reader    = pypdf.PdfReader(PDF_PATH)",
    'print(f"PDF    : {PDF_PATH}")',
    'print(f"Pages  : {len(reader.pages)}")',
    "full_text = ''",
    "for page in reader.pages:",
    "    full_text += (page.extract_text() or '') + '\\n\\n'",
    "chunks = recursive_split(full_text, CHUNK_SIZE, CHUNK_OVERLAP)",
    'print(f"Chunks : {len(chunks)}")',
    'print(f"Avg    : {sum(len(c) for c in chunks)/len(chunks):.0f} chars")',
)))

# ── Step 7 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 7 — Build BM25 Index (pure Python)\n\n"
    "BM25 ranks documents by how well they match a query's terms,\n"
    "taking into account term frequency, document length, and corpus-wide IDF:\n\n"
    "```\n"
    "score(doc, query) = sum over terms t:\n"
    "    IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl))\n"
    "```\n\n"
    "- **IDF** — rare terms score higher than common ones\n"
    "- **TF saturation** — doubling term frequency doesn't double the score (`k1`)\n"
    "- **Length norm** — short documents aren't penalised unfairly (`b`)\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def tokenise(text: str) -> List[str]:",
    "    return re.findall(r'[a-zA-Z0-9]+', text.lower())",
    "",
    "class BM25Index:",
    "    def __init__(self, corpus: List[str], k1: float = BM25_K1, b: float = BM25_B):",
    "        self.k1     = k1",
    "        self.b      = b",
    "        self.corpus = corpus",
    "        self.N      = len(corpus)",
    "        self.tokenised = [tokenise(doc) for doc in corpus]",
    "        self.avgdl  = sum(len(t) for t in self.tokenised) / self.N",
    "        # document frequency: how many docs contain each term",
    "        self.df: Dict[str, int] = defaultdict(int)",
    "        for tokens in self.tokenised:",
    "            for term in set(tokens):",
    "                self.df[term] += 1",
    "",
    "    def idf(self, term: str) -> float:",
    "        df = self.df.get(term, 0)",
    "        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)",
    "",
    "    def score(self, doc_tokens: List[str], query_terms: List[str], doc_len: int) -> float:",
    "        tf_map: Dict[str, int] = defaultdict(int)",
    "        for t in doc_tokens: tf_map[t] += 1",
    "        s = 0.0",
    "        for term in query_terms:",
    "            tf = tf_map.get(term, 0)",
    "            if tf == 0: continue",
    "            idf = self.idf(term)",
    "            num = tf * (self.k1 + 1)",
    "            den = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)",
    "            s  += idf * num / den",
    "        return s",
    "",
    "    def search(self, query: str, top_k: int = 20) -> List[Dict]:",
    "        q_terms = tokenise(query)",
    "        scores  = []",
    "        for i, tokens in enumerate(self.tokenised):",
    "            s = self.score(tokens, q_terms, len(tokens))",
    "            if s > 0:",
    "                scores.append((i, s))",
    "        scores.sort(key=lambda x: x[1], reverse=True)",
    "        return [",
    "            {'text': self.corpus[i], 'score': s, 'id': f'bm25_{i}',",
    "             'metadata': {'chunk_idx': i, 'source': 'bm25'}}",
    "            for i, s in scores[:top_k]",
    "        ]",
    "",
    "bm25 = BM25Index(chunks)",
    'print(f"BM25 index built over {len(chunks)} docs")',
    'print(f"Vocabulary size: {len(bm25.df):,} terms")',
    'print(f"Avg doc length : {bm25.avgdl:.0f} tokens")',
    "",
    "# Quick sanity check",
    'test_hits = bm25.search("weather forecasting methods", top_k=3)',
    'print(f"\\nBM25 test — top 3 for \'weather forecasting methods\':")',
    "for i, h in enumerate(test_hits, 1):",
    "    print(f'  [{i}] score={h[\"score\"]:.4f}  {h[\"text\"][:80]}...')",
)))

# ── Step 8 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 8 — Embed & Index in Qdrant"))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    'print(f"Embedding {len(chunks)} chunks...")',
    "t0   = time.time()",
    "embs = embed_batch(chunks, label='[chunks]')",
    "docs = [",
    "    {'text': chunks[i], 'embedding': embs[i],",
    "     'metadata': {'chunk_idx': i, 'source': 'climate.pdf'}}",
    "    for i in range(len(chunks))",
    "]",
    "vs.upsert(docs)",
    'print(f"Indexed {vs.count()} vectors in {time.time()-t0:.1f}s")',
)))

# ── Step 9 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 9 — Reciprocal Rank Fusion\n\n"
    "RRF merges two ranked lists into a single score **without normalising raw scores**.\n"
    "This is important because BM25 scores and cosine similarity scores live on\n"
    "completely different scales — you cannot simply average them.\n\n"
    "```python\n"
    "rrf_score(doc) = sum_over_lists( 1 / (k + rank_in_list) )\n"
    "```\n"
    "A doc ranked #1 in both lists gets `1/(60+1) + 1/(60+1) ≈ 0.033`.\n"
    "A doc ranked #1 in one list only gets `1/(60+1) ≈ 0.016`.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def reciprocal_rank_fusion(",
    "    list_a: List[Dict],",
    "    list_b: List[Dict],",
    "    k: int = RRF_K,",
    "    top_k: int = TOP_K_FINAL",
    ") -> List[Dict]:",
    "    scores: Dict[str, float] = defaultdict(float)",
    "    docs:   Dict[str, Dict]  = {}",
    "",
    "    for rank, doc in enumerate(list_a):",
    "        key = doc['text'][:60]   # use text prefix as dedup key",
    "        scores[key] += 1.0 / (k + rank + 1)",
    "        docs[key]    = doc",
    "",
    "    for rank, doc in enumerate(list_b):",
    "        key = doc['text'][:60]",
    "        scores[key] += 1.0 / (k + rank + 1)",
    "        if key not in docs:",
    "            docs[key] = doc",
    "",
    "    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)",
    "    return [",
    "        {**docs[key], 'rrf_score': score}",
    "        for key, score in ranked[:top_k]",
    "    ]",
    "",
    'print("reciprocal_rank_fusion() defined.")',
    "",
    "# Quick smoke test",
    'q_test = "weather analysis methods"',
    "bm_hits  = bm25.search(q_test, top_k=TOP_K_EACH)",
    "vec_hits = vs.search_vector(embed_text(q_test), top_k=TOP_K_EACH)",
    "rrf_hits = reciprocal_rank_fusion(bm_hits, vec_hits)",
    'print(f"BM25 candidates : {len(bm_hits)}")',
    'print(f"Vector candidates: {len(vec_hits)}")',
    'print(f"RRF merged top-{TOP_K_FINAL}: {len(rrf_hits)}")',
)))

# ── Step 10 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 10 — Three-Way RAG Queries"))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def rag_query(",
    "    question: str,",
    "    mode: str = 'hybrid',   # 'hybrid' | 'vector' | 'bm25'",
    "    verbose: bool = True",
    ") -> Dict:",
    "    t0   = time.time()",
    "    qvec = embed_text(question)",
    "",
    "    if mode == 'vector':",
    "        hits = vs.search_vector(qvec, top_k=TOP_K_FINAL)",
    "    elif mode == 'bm25':",
    "        hits = bm25.search(question, top_k=TOP_K_FINAL)",
    "    else:  # hybrid",
    "        bm_hits  = bm25.search(question, top_k=TOP_K_EACH)",
    "        vec_hits = vs.search_vector(qvec, top_k=TOP_K_EACH)",
    "        hits     = reciprocal_rank_fusion(bm_hits, vec_hits)",
    "",
    "    answer  = generate_answer(question, [h['text'] for h in hits])",
    "    latency = (time.time() - t0) * 1000",
    "",
    "    if verbose:",
    '        print(f"\\nQ [{mode.upper()}]: {question}")',
    '        print(f"A: {answer}")',
    "        for i, h in enumerate(hits[:3], 1):",
    "            sc = h.get('rrf_score', h.get('score', 0))",
    "            print(f\"  [{i}] score={sc:.4f}  {h['text'][:75]}...\")",
    '        print(f"  Latency: {latency:.0f}ms")',
    '        print("-" * 70)',
    "",
    "    return {'question': question, 'answer': answer, 'hits': hits,",
    "            'mode': mode, 'latency_ms': latency}",
    "",
    "test_questions = [",
    '    "What is weather forecasting and why is it important?",',
    '    "What are the main methods used in weather analysis?",',
    '    "How does climatology differ from meteorology?",',
    '    "What factors influence weather patterns and climate?",',
    "]",
    "results_log = []",
    "for q in test_questions:",
    "    results_log.append(rag_query(q, mode='hybrid'))",
)))

# ── Step 11 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 11 — A/B/C Mode Comparison\n\n"
    "Run every test question through all three modes and compare top-1 scores and latency.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "compare_qs = [",
    '    "What are the main methods used in weather analysis?",',
    '    "How does climatology differ from meteorology?",',
    '    "What factors influence weather patterns?",',
    "]",
    "",
    "print('{:<50} {:>9}  {:>9}  {:>9}  {:>8}  {:>8}  {:>8}'.format(",
    "    'Question', 'BM25 s1', 'Vec s1', 'RRF s1', 'BM ms', 'Vec ms', 'Hyb ms'))",
    "print('-' * 108)",
    "",
    "for q in compare_qs:",
    "    qvec = embed_text(q)",
    "    t0 = time.time(); bm_h  = bm25.search(q, top_k=TOP_K_FINAL);          bm_t  = (time.time()-t0)*1000",
    "    t0 = time.time(); vec_h = vs.search_vector(qvec, top_k=TOP_K_FINAL);   vec_t = (time.time()-t0)*1000",
    "    t0 = time.time()",
    "    bm_all  = bm25.search(q, top_k=TOP_K_EACH)",
    "    vec_all = vs.search_vector(qvec, top_k=TOP_K_EACH)",
    "    rrf_h   = reciprocal_rank_fusion(bm_all, vec_all)",
    "    hyb_t   = (time.time()-t0)*1000",
    "    bm_s1   = bm_h[0]['score']        if bm_h  else 0",
    "    vec_s1  = vec_h[0]['score']       if vec_h else 0",
    "    rrf_s1  = rrf_h[0]['rrf_score']   if rrf_h else 0",
    "    print('{:<50} {:>9.4f}  {:>9.4f}  {:>9.4f}  {:>7.0f}ms {:>7.0f}ms {:>7.0f}ms'.format(",
    "        q[:49], bm_s1, vec_s1, rrf_s1, bm_t, vec_t, hyb_t))",
)))

# ── Step 12 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 12 — Exact-Term vs Semantic Query Demo\n\n"
    "This demo highlights where each approach shines:\n"
    "- **Exact-term query** (rare technical term) — BM25 should win\n"
    "- **Paraphrase query** (semantic, no exact match) — Vector should win\n"
    "- **Mixed query** — Hybrid should consistently perform best\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "demo_cases = [",
    "    {",
    "        'label': 'Exact term (rare word)',",
    "        'query': 'synoptic meteorology observations',",
    "        'why'  : 'BM25 should score high — rare exact terms favour keyword search'",
    "    },",
    "    {",
    "        'label': 'Paraphrase / semantic',",
    "        'query': 'what causes the air to get hotter as you go lower in the atmosphere',",
    "        'why'  : 'Vector should win — no exact doc terms, relies on semantic similarity'",
    "    },",
    "    {",
    "        'label': 'Mixed — term + context',",
    "        'query': 'climate change temperature patterns atmospheric pressure',",
    "        'why'  : 'Hybrid should win — benefits from both exact and semantic matching'",
    "    },",
    "]",
    "",
    "for case in demo_cases:",
    "    q    = case['query']",
    "    qvec = embed_text(q)",
    "    bm_h = bm25.search(q, top_k=1)",
    "    vc_h = vs.search_vector(qvec, top_k=1)",
    "    bm_all  = bm25.search(q, top_k=TOP_K_EACH)",
    "    vec_all = vs.search_vector(qvec, top_k=TOP_K_EACH)",
    "    hy_h = reciprocal_rank_fusion(bm_all, vec_all, top_k=1)",
    "    print(f'--- {case[\"label\"]} ---')",
    "    print(f'Query  : {q}')",
    "    print(f'Why    : {case[\"why\"]}')",
    "    bm_s = bm_h[0]['score']      if bm_h else 0",
    "    vc_s = vc_h[0]['score']      if vc_h else 0",
    "    hy_s = hy_h[0]['rrf_score']  if hy_h else 0",
    "    winner = max([('BM25', bm_s), ('Vector', vc_s), ('Hybrid-RRF', hy_s)], key=lambda x: x[1])",
    "    print(f'BM25   score: {bm_s:.4f}  | {(bm_h[0][\"text\"][:60] if bm_h else \"\") }...')",
    "    print(f'Vector score: {vc_s:.4f}  | {(vc_h[0][\"text\"][:60] if vc_h else \"\")}...')",
    "    print(f'Hybrid score: {hy_s:.4f}  | {(hy_h[0][\"text\"][:60] if hy_h else \"\")}...')",
    "    print(f'Winner : {winner[0]}')",
    "    print()",
)))

# ── Step 13 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 13 — Evaluation & Metrics"))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "eval_cases = [",
    "    {'question': 'What is weather forecasting and why is it important?',",
    "     'keywords': ['forecast', 'weather', 'predict', 'atmosphere', 'climate']},",
    "    {'question': 'What are the main methods used in weather analysis?',",
    "     'keywords': ['analysis', 'synoptic', 'observation', 'data', 'pressure']},",
    "    {'question': 'How does climatology differ from meteorology?',",
    "     'keywords': ['climate', 'weather', 'long', 'study', 'atmosphere']},",
    "]",
    "",
    "modes = ['bm25', 'vector', 'hybrid']",
    "print('{:<45} {:>14}  {:>14}  {:>14}'.format('Question', 'BM25 KW%', 'Vector KW%', 'Hybrid KW%'))",
    "print('-' * 92)",
    "",
    "for case in eval_cases:",
    "    row = case['question'][:44]",
    "    kws = case['keywords']",
    "    n   = len(kws)",
    "    cols = []",
    "    for mode in modes:",
    "        r    = rag_query(case['question'], mode=mode, verbose=False)",
    "        hits = sum(1 for kw in kws if kw in r['answer'].lower())",
    "        cols.append(f'{hits}/{n} ({hits/n*100:.0f}%)')",
    "    print('{:<45} {:>14}  {:>14}  {:>14}'.format(row, cols[0], cols[1], cols[2]))",
)))

# ── Step 14 Summary ────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 14 — Summary\n\n"
    "| Component | Implementation |\n"
    "|-----------|---------------|\n"
    "| PDF | `pypdf.PdfReader` — absolute path in Step 2 |\n"
    "| Chunking | Native `recursive_split()` — 500 chars, 50 overlap |\n"
    "| BM25 index | Pure Python — no Elasticsearch, no Lucene |\n"
    "| Vector index | Bedrock Titan V2 embeddings → Qdrant |\n"
    "| Fusion | Reciprocal Rank Fusion (`k=60`) — scale-invariant |\n"
    "| Candidate pool | Top-20 from each method → fuse → Top-5 to LLM |\n"
    "| LLM | AWS Strands Agent + Claude Sonnet 4.6 |\n\n"
    "## When to use each mode\n\n"
    "| Mode | Best for |\n"
    "|------|----------|\n"
    "| BM25 only | Domain-specific terms, IDs, exact-match queries |\n"
    "| Vector only | Open-ended semantic questions, paraphrases |\n"
    "| **Hybrid** | **Production default — consistently best across query types** |\n\n"
    "### Next: **08 — HyDE** (Hypothetical Document Embeddings)\n"
))

cells.append(code(L(
    "# vs.delete_collection()  # uncomment to clean up",
    "print(f\"Collection '{COLLECTION_NAME}' in {vs._backend} — {vs.count()} vectors\")",
    'print(f"BM25 index: {len(chunks)} docs, {len(bm25.df):,} terms")',
    'print("\\nDone. Give the go-ahead for notebook 08.")',
)))

# ── Write & validate ──────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"}
    },
    "cells": cells
}

with open(PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

errors = []
for i, c in enumerate(cells):
    if c['cell_type'] == 'code':
        try:
            ast.parse(''.join(c['source']))
        except SyntaxError as e:
            errors.append((i, e.lineno, e.msg))

print(f"Written : {PATH}")
print(f"Cells   : {len(cells)}")
if errors:
    for i, ln, msg in errors:
        print(f"SYNTAX ERROR cell {i} line {ln}: {msg}")
else:
    print("All code cells parse OK")
