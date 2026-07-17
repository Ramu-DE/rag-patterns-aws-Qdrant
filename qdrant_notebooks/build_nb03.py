"""Build 03_Fusion_Retrieval.ipynb from scratch — no nested triple-quotes."""
import json, uuid

def cell_id():
    return str(uuid.uuid4())[:8]

def md(source):
    return {"cell_type": "markdown", "id": cell_id(),
            "metadata": {}, "source": source.splitlines(keepends=True)}

def code(lines):
    """lines: list of strings (each a complete source line, with \n)"""
    return {"cell_type": "code", "id": cell_id(), "metadata": {},
            "execution_count": None, "outputs": [], "source": lines}

# Helper: turn a raw string into a list of lines for a code cell
def src(*lines):
    result = []
    for line in lines:
        result.append(line if line.endswith("\n") else line + "\n")
    # strip trailing \n from last line
    if result:
        result[-1] = result[-1].rstrip("\n")
    return result

cells = []

# ── Overview ──────────────────────────────────────────────────────────────────
cells.append(md(
"# 03 — Fusion Retrieval (RAG Fusion)\n"
"\n"
"## Overview\n"
"Fusion Retrieval improves recall by generating **multiple query variants** from\n"
"the original question, retrieving chunks for each variant, then merging all\n"
"result lists using **Reciprocal Rank Fusion (RRF)** before generating the answer.\n"
"\n"
"```\n"
"User Query\n"
"    |\n"
"    v\n"
"Generate N query variants  (Strands Agent -> Claude)\n"
"    |\n"
"    |-- Variant 1 -> Vector Search -> ranked list 1\n"
"    |-- Variant 2 -> Vector Search -> ranked list 2\n"
"    |-- Variant N -> Vector Search -> ranked list N\n"
"                                            |\n"
"                                            v\n"
"                                Reciprocal Rank Fusion (RRF)\n"
"                                            |\n"
"                                            v\n"
"                                    Top-K fused chunks\n"
"                                            |\n"
"                                            v\n"
"                                Generate Answer  (Strands Agent -> Claude)\n"
"```\n"
"\n"
"## RRF Formula\n"
"For each document **d** across all query result lists:\n"
"\n"
"    RRF_score(d) = sum( 1 / (k + rank_i(d)) )\n"
"\n"
"where `rank_i(d)` is d's rank in query i's result list and `k = 60` (constant).\n"
"\n"
"## Why Fusion Retrieval?\n"
"| Aspect | Simple RAG | Fusion RAG |\n"
"|--------|-----------|------------|\n"
"| Queries issued | 1 | N (default 4) |\n"
"| Recall | Baseline | +20-40 % |\n"
"| Latency | ~2-3 s | ~5-8 s |\n"
"| Best for | Focused Q&A | Broad/ambiguous questions |\n"
"\n"
"## Vector DB Strategy\n"
"Same as notebook 01 — Qdrant Cloud -> OpenSearch -> in-memory fallback.\n"
))

# ── Step 1 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 1 — Install & Import Dependencies"))

cells.append(code(src(
    "import subprocess, sys",
    'packages = [',
    '    "boto3",',
    '    "qdrant-client",',
    '    "opensearch-py",',
    '    "requests-aws4auth",',
    '    "strands-agents",',
    '    "pypdf",',
    ']',
    'subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + packages)',
    'print("All packages ready.")',
)))

cells.append(code(src(
    "import os, sys, json, time, uuid, re",
    "from typing import List, Dict, Optional",
    "from collections import defaultdict",
    "",
    "import boto3",
    "from strands import Agent",
    "from strands.models.bedrock import BedrockModel",
    "from qdrant_client import QdrantClient",
    "from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint",
    "",
    'print("Imports OK")',
)))

# ── Step 2 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 2 — Configuration"))

cells.append(code(src(
    'AWS_REGION       = os.getenv("AWS_DEFAULT_REGION", "us-east-1")',
    'EMBEDDING_MODEL  = "amazon.titan-embed-text-v2:0"',
    'LLM_MODEL        = "us.anthropic.claude-sonnet-4-6"',
    "",
    'QDRANT_URL       = os.getenv("QDRANT_URL", "")',
    'QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY", "")',
    'OPENSEARCH_URL   = os.getenv("OPENSEARCH_ENDPOINT", "")',
    "",
    'COLLECTION_NAME  = "fusion_rag_03"',
    "EMBEDDING_DIM    = 1024",
    "CHUNK_SIZE       = 1000",
    "CHUNK_OVERLAP    = 200",
    "",
    "# Fusion-specific",
    "NUM_VARIANTS     = 4    # total queries (original + 3 variants)",
    "TOP_K_PER_QUERY  = 10   # results fetched per variant",
    "FINAL_TOP_K      = 5    # results after RRF fusion",
    "RRF_K            = 60   # RRF constant (rank dampening)",
    "",
    'print(f"Region         : {AWS_REGION}")',
    'print(f"LLM model      : {LLM_MODEL}")',
    "print(f\"Qdrant URL     : {QDRANT_URL or '(not set -- will use in-memory)'}\")",
    'print(f"Collection     : {COLLECTION_NAME}")',
    'print(f"Query variants : {NUM_VARIANTS}  |  Top-K/query : {TOP_K_PER_QUERY}  |  Final top-K : {FINAL_TOP_K}")',
)))

# ── Step 3 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 3 — Vector DB Client (Qdrant -> OpenSearch fallback)\n"
    "\n"
    "Same self-contained `VectorStore` class as notebook 01.\n"
))

cells.append(code(src(
    "class VectorStore:",
    "    # Priority: Qdrant Cloud -> OpenSearch Serverless -> Qdrant In-Memory",
    "    # Public API: create_collection, upsert, search, count, delete_collection",
    "",
    "    def __init__(self, collection_name, qdrant_url='', qdrant_api_key='',",
    "                 opensearch_url='', region='us-east-1'):",
    "        self.name = collection_name",
    "        self.region = region",
    "        self._backend = None",
    "",
    "        if qdrant_url:",
    "            try:",
    "                kwargs = {'url': qdrant_url}",
    "                if qdrant_api_key:",
    "                    kwargs['api_key'] = qdrant_api_key",
    "                self._qdrant = QdrantClient(**kwargs)",
    "                self._qdrant.get_collections()",
    "                self._backend = 'qdrant_cloud'",
    "                print(f'Connected to Qdrant Cloud: {qdrant_url}')",
    "                return",
    "            except Exception as e:",
    "                print(f'Qdrant Cloud unavailable ({e}), trying next...')",
    "",
    "        if opensearch_url:",
    "            try:",
    "                from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth",
    "                creds = boto3.Session().get_credentials()",
    "                auth  = AWSV4SignerAuth(creds, region, 'aoss')",
    "                host  = opensearch_url.replace('https://', '').replace('http://', '')",
    "                self._os = OpenSearch(",
    "                    hosts=[{'host': host, 'port': 443}],",
    "                    http_auth=auth, use_ssl=True, verify_certs=True,",
    "                    connection_class=RequestsHttpConnection, timeout=30",
    "                )",
    "                self._os.info()",
    "                self._backend = 'opensearch'",
    "                print(f'Connected to OpenSearch: {opensearch_url}')",
    "                return",
    "            except Exception as e:",
    "                print(f'OpenSearch unavailable ({e}), falling back to in-memory Qdrant...')",
    "",
    "        self._qdrant  = QdrantClient(':memory:')",
    "        self._backend = 'qdrant_memory'",
    "        print('Using Qdrant in-memory (data lost on kernel restart)')",
    "",
    "    def create_collection(self, dim=1024, force_recreate=False):",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            exists = self.name in [c.name for c in self._qdrant.get_collections().collections]",
    "            if exists and force_recreate:",
    "                self._qdrant.delete_collection(self.name)",
    "                exists = False",
    "            if not exists:",
    "                self._qdrant.create_collection(",
    "                    self.name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))",
    "                print(f'Created collection \"{self.name}\" (dim={dim})')",
    "            else:",
    "                print(f'Collection \"{self.name}\" already exists')",
    "            return True",
    "        if self._backend == 'opensearch':",
    "            if force_recreate and self._os.indices.exists(index=self.name):",
    "                self._os.indices.delete(index=self.name)",
    "            if not self._os.indices.exists(index=self.name):",
    "                body = {",
    "                    'settings': {'index': {'knn': True}},",
    "                    'mappings': {'properties': {",
    "                        'text':      {'type': 'text'},",
    "                        'metadata':  {'type': 'object'},",
    "                        'embedding': {'type': 'knn_vector', 'dimension': dim,",
    "                                      'method': {'name': 'hnsw', 'space_type': 'cosinesimil',",
    "                                                 'engine': 'faiss',",
    "                                                 'parameters': {'ef_construction': 512, 'm': 16}}}",
    "                    }}",
    "                }",
    "                self._os.indices.create(index=self.name, body=body)",
    "                print(f'Created OpenSearch index \"{self.name}\"')",
    "            return True",
    "",
    "    def upsert(self, docs: List[Dict]) -> int:",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            points = [",
    "                PointStruct(id=str(uuid.uuid4()), vector=d['embedding'],",
    "                            payload={'text': d['text'], 'metadata': d.get('metadata', {})})",
    "                for d in docs",
    "            ]",
    "            self._qdrant.upsert(collection_name=self.name, points=points)",
    "            return len(points)",
    "        if self._backend == 'opensearch':",
    "            for d in docs:",
    "                self._os.index(index=self.name, body=d)",
    "            time.sleep(1)",
    "            return len(docs)",
    "",
    "    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            resp = self._qdrant.query_points(",
    "                collection_name=self.name, query=query_vector,",
    "                limit=top_k, with_payload=True)",
    "            return [{'text': p.payload.get('text', ''),",
    "                     'metadata': p.payload.get('metadata', {}),",
    "                     'score': p.score, 'id': str(p.id)}",
    "                    for p in resp.points]",
    "        if self._backend == 'opensearch':",
    "            body = {'size': top_k,",
    "                    'query': {'knn': {'embedding': {'vector': query_vector, 'k': top_k}}},",
    "                    '_source': {'excludes': ['embedding']}}",
    "            resp = self._os.search(index=self.name, body=body)",
    "            return [{'text': h['_source'].get('text', ''),",
    "                     'metadata': h['_source'].get('metadata', {}),",
    "                     'score': h['_score'], 'id': h['_id']}",
    "                    for h in resp['hits']['hits']]",
    "",
    "    def count(self) -> int:",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            return self._qdrant.get_collection(self.name).points_count or 0",
    "        if self._backend == 'opensearch':",
    "            return self._os.count(index=self.name).get('count', 0)",
    "",
    "    def delete_collection(self):",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            self._qdrant.delete_collection(self.name)",
    "        if self._backend == 'opensearch':",
    "            self._os.indices.delete(index=self.name, ignore=[404])",
    "        print(f'Deleted \"{self.name}\"')",
    "",
    'print("VectorStore class defined.")',
)))

# ── Step 4 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 4 — Bedrock Helpers (Embeddings + Strands LLM)"))

cells.append(code(src(
    "# Embeddings use boto3 directly (Strands has no embeddings API)",
    'bedrock_rt = boto3.client("bedrock-runtime", region_name=AWS_REGION)',
    "",
    "",
    "def embed_text(text: str) -> List[float]:",
    '    body = json.dumps({"inputText": text, "dimensions": EMBEDDING_DIM, "normalize": True})',
    "    resp = bedrock_rt.invoke_model(",
    "        modelId=EMBEDDING_MODEL, body=body,",
    '        contentType="application/json", accept="application/json")',
    '    return json.loads(resp["body"].read())["embedding"]',
    "",
    "",
    "def embed_batch(texts: List[str]) -> List[List[float]]:",
    "    embeddings = []",
    "    for i, t in enumerate(texts):",
    "        embeddings.append(embed_text(t))",
    "        if (i + 1) % 10 == 0:",
    '            print(f"  Embedded {i+1}/{len(texts)}")',
    "        time.sleep(0.05)",
    "    return embeddings",
    "",
    "",
    "# Strands BedrockModel — shared across all Agent calls in this notebook",
    "_model = BedrockModel(model_id=LLM_MODEL, region_name=AWS_REGION)",
    "",
    "",
    'def strands_call(prompt: str, system: str = "You are a helpful assistant.") -> str:',
    '    "Single-turn Strands Agent call, returns response as plain string."',
    "    agent = Agent(model=_model, system_prompt=system)",
    "    return str(agent(prompt))",
    "",
    "",
    "# Smoke test",
    'test_emb = embed_text("Hello world")',
    'print(f"Embedding OK  -- dim={len(test_emb)}, sample={[round(x,4) for x in test_emb[:3]]}")',
    'print("Strands BedrockModel ready.")',
)))

# ── Step 5 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 5 — Connect & Create Collection"))

cells.append(code(src(
    "vs = VectorStore(",
    "    collection_name=COLLECTION_NAME,",
    "    qdrant_url=QDRANT_URL,",
    "    qdrant_api_key=QDRANT_API_KEY,",
    "    opensearch_url=OPENSEARCH_URL,",
    "    region=AWS_REGION",
    ")",
    'print(f"Active backend: {vs._backend}")',
    "vs.create_collection(dim=EMBEDDING_DIM, force_recreate=True)",
)))

# ── Step 6 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 6 — Load & Chunk the PDF\n"
    "\n"
    "Same `pypdf` + native recursive splitter as notebook 01.\n"
    "Using `climate.pdf` — shared across all RAG notebooks.\n"
))

cells.append(code(src(
    "import os",
    "import pypdf",
    "",
    "",
    "def recursive_split(text: str, chunk_size: int = 1000,",
    "                    chunk_overlap: int = 200) -> List[str]:",
    '    "Split text into overlapping chunks: paragraph -> line -> word -> char."',
    '    separators = ["\\n\\n", "\\n", ". ", " ", ""]',
    "",
    "    def _split(text, seps):",
    '        sep = ""',
    "        new_seps = []",
    "        for i, s in enumerate(seps):",
    '            if s == "" or s in text:',
    "                sep = s",
    "                new_seps = seps[i+1:]",
    "                break",
    '        parts = text.split(sep) if sep != "" else list(text)',
    "        good = []",
    "        for part in parts:",
    "            if len(part) <= chunk_size:",
    "                good.append(part)",
    "            elif new_seps:",
    "                good.extend(_split(part, new_seps))",
    "            else:",
    "                for k in range(0, len(part), chunk_size - chunk_overlap):",
    "                    good.append(part[k:k+chunk_size])",
    "        chunks, cur_pieces, cur_len = [], [], 0",
    "        for piece in good:",
    "            p = piece.strip()",
    "            if not p:",
    "                continue",
    "            addition = len(sep) + len(p) if cur_pieces else len(p)",
    "            if cur_len + addition <= chunk_size:",
    "                cur_pieces.append(p)",
    "                cur_len += addition",
    "            else:",
    "                if cur_pieces:",
    "                    chunks.append(sep.join(cur_pieces))",
    "                overlap_pieces, overlap_len = [], 0",
    "                for prev in reversed(cur_pieces):",
    "                    if overlap_len + len(prev) + len(sep) <= chunk_overlap:",
    "                        overlap_pieces.insert(0, prev)",
    "                        overlap_len += len(prev) + len(sep)",
    "                    else:",
    "                        break",
    "                cur_pieces = overlap_pieces + [p]",
    "                cur_len = sum(len(x) + len(sep) for x in cur_pieces)",
    "        if cur_pieces:",
    "            chunks.append(sep.join(cur_pieces))",
    "        return [c for c in chunks if c.strip()]",
    "",
    "    return _split(text, separators)",
    "",
    "",
    'PDF_PATH = os.path.join("..", "data", "climate.pdf")',
    "reader   = pypdf.PdfReader(PDF_PATH)",
    'print(f"PDF loaded  : {PDF_PATH}")',
    'print(f"Total pages : {len(reader.pages)}")',
    "",
    "chunks: List[Dict] = []",
    "for page_num, page in enumerate(reader.pages):",
    '    page_text = page.extract_text() or ""',
    "    for chunk_text in recursive_split(page_text, CHUNK_SIZE, CHUNK_OVERLAP):",
    "        if chunk_text.strip():",
    "            chunks.append({",
    '                "text"    : chunk_text.strip(),',
    '                "page_num": page_num + 1,',
    '                "source"  : os.path.basename(PDF_PATH)',
    "            })",
    "",
    'print(f"\\nChunks created   : {len(chunks)}")',
    "print(f\"Avg chunk length : {sum(len(c['text']) for c in chunks)/len(chunks):.0f} chars\")",
)))

# ── Step 7 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 7 — Embed & Index"))

cells.append(code(src(
    'print(f"Embedding {len(chunks)} chunks...")',
    "t0 = time.time()",
    "texts      = [c['text'] for c in chunks]",
    "embeddings = embed_batch(texts)",
    "",
    "docs_to_index = [",
    "    {",
    '        "text"     : chunks[i]["text"],',
    '        "embedding": embeddings[i],',
    '        "metadata" : {',
    '            "chunk_index": i,',
    '            "page_num"   : chunks[i]["page_num"],',
    '            "source"     : chunks[i]["source"]',
    "        }",
    "    }",
    "    for i in range(len(chunks))",
    "]",
    "indexed = vs.upsert(docs_to_index)",
    "elapsed = time.time() - t0",
    "",
    'print(f"Indexed  : {indexed} chunks in {elapsed:.2f}s")',
    'print(f"Count in collection: {vs.count()}")',
)))

# ── Step 8 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 8 — Query Variant Generation\n"
    "\n"
    "Claude (via Strands Agent) generates `NUM_VARIANTS - 1` alternative phrasings\n"
    "of the original query, covering different angles: rephrasing, decomposition,\n"
    "perspective shift. The original query is always included as variant 0.\n"
))

cells.append(code(src(
    "def generate_query_variants(original_query: str,",
    "                            num_variants: int = NUM_VARIANTS) -> List[str]:",
    '    "Use Strands Agent (Claude) to generate alternative query phrasings."',
    "    n_extra = num_variants - 1",
    "    prompt = (",
    "        f'Generate exactly {n_extra} alternative phrasings of the following question.\\n'",
    "        f'Each variant should approach the topic from a different angle:\\n'",
    "        f'  - Try rephrasing, decomposing, or shifting perspective\\n'",
    "        f'  - Keep each variant concise (one sentence)\\n'",
    "        f'  - Output ONLY a numbered list, one variant per line, no extra text\\n\\n'",
    "        f'Original question: {original_query}\\n\\n'",
    "        f'Variants:'",
    "    )",
    "    response = strands_call(",
    "        prompt,",
    '        system="You are a search query expert. Output only numbered lists, nothing else."',
    "    )",
    "    # Parse numbered list: handles '1.', '1)', '1 .' formats",
    "    variants = []",
    "    for line in response.strip().splitlines():",
    "        line = line.strip()",
    "        cleaned = re.sub(r'^\\d+[.)\\s]+', '', line).strip()",
    "        if cleaned:",
    "            variants.append(cleaned)",
    "    variants = variants[:n_extra]",
    "    return [original_query] + variants",
    "",
    "",
    "# Demo",
    'demo_q = "What factors influence weather patterns?"',
    "variants = generate_query_variants(demo_q, NUM_VARIANTS)",
    'print(f"Original  : {variants[0]}")',
    "for i, v in enumerate(variants[1:], 1):",
    '    print(f"Variant {i} : {v}")',
)))

# ── Step 9 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 9 — Reciprocal Rank Fusion (RRF)\n"
    "\n"
    "Merge ranked result lists from each query variant into one fused ranking.\n"
    "\n"
    "**RRF formula:**\n"
    "\n"
    "    RRF_score(doc) = sum( 1 / (k + rank_i) )  for each list i that contains doc\n"
    "\n"
    "- Documents appearing near the top of multiple lists score highest\n"
    "- Rank-based (not score-based) — robust to score scale differences\n"
    "- `k=60` dampens the effect of very high ranks\n"
))

cells.append(code(src(
    "def reciprocal_rank_fusion(results_list: List[List[Dict]],",
    "                           k: int = RRF_K) -> List[Dict]:",
    '    "Merge multiple ranked result lists using Reciprocal Rank Fusion."',
    "    rrf_scores  = defaultdict(float)",
    "    doc_store   = {}",
    "    appearances = defaultdict(int)",
    "",
    "    for results in results_list:",
    "        for rank, doc in enumerate(results, start=1):",
    '            doc_id = doc["id"]',
    "            rrf_scores[doc_id]  += 1.0 / (k + rank)",
    "            appearances[doc_id] += 1",
    "            if doc_id not in doc_store:",
    "                doc_store[doc_id] = doc",
    "",
    "    fused = []",
    "    for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):",
    "        entry = dict(doc_store[doc_id])",
    '        entry["rrf_score"]   = round(score, 6)',
    '        entry["appearances"] = appearances[doc_id]',
    "        fused.append(entry)",
    "    return fused",
    "",
    "",
    "# Quick unit test",
    "r1 = [",
    '    {"id": "a", "text": "doc A", "metadata": {}, "score": 0.9},',
    '    {"id": "b", "text": "doc B", "metadata": {}, "score": 0.8},',
    '    {"id": "c", "text": "doc C", "metadata": {}, "score": 0.6},',
    "]",
    "r2 = [",
    '    {"id": "b", "text": "doc B", "metadata": {}, "score": 0.95},',
    '    {"id": "a", "text": "doc A", "metadata": {}, "score": 0.7},',
    '    {"id": "d", "text": "doc D", "metadata": {}, "score": 0.5},',
    "]",
    "fused = reciprocal_rank_fusion([r1, r2])",
    'print("RRF unit test (doc B should rank 1st -- appears in both lists at rank 1 & 2):")',
    "for i, r in enumerate(fused, 1):",
    "    print(f\"  {i}. id={r['id']}  rrf={r['rrf_score']:.5f}  appearances={r['appearances']}\")",
)))

# ── Step 10 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 10 — Fusion RAG Pipeline\n"
    "\n"
    "End-to-end pipeline:\n"
    "1. Generate `NUM_VARIANTS` query variants with Claude\n"
    "2. Embed each variant with Titan V2\n"
    "3. Retrieve `TOP_K_PER_QUERY` chunks per variant from Qdrant\n"
    "4. Merge all result lists with RRF -> take `FINAL_TOP_K`\n"
    "5. Generate grounded answer with Strands Agent\n"
))

cells.append(code(src(
    "def fusion_rag_query(question: str,",
    "                     num_variants: int = NUM_VARIANTS,",
    "                     top_k_per_query: int = TOP_K_PER_QUERY,",
    "                     final_top_k: int = FINAL_TOP_K,",
    "                     verbose: bool = True) -> Dict:",
    '    "Full Fusion RAG pipeline. Returns question, answer, fused_results, variants, latency_ms, timing."',
    "    t0 = time.time()",
    "    timing = {}",
    "",
    "    # 1. Generate query variants",
    "    t1 = time.time()",
    "    variants = generate_query_variants(question, num_variants)",
    '    timing["variant_gen_s"] = round(time.time() - t1, 2)',
    "",
    "    # 2. Retrieve for every variant",
    "    t2 = time.time()",
    "    all_results = []",
    "    for v in variants:",
    "        q_emb = embed_text(v)",
    "        results = vs.search(q_emb, top_k=top_k_per_query)",
    "        all_results.append(results)",
    '    timing["retrieval_s"] = round(time.time() - t2, 2)',
    "",
    "    # 3. Reciprocal Rank Fusion",
    "    fused = reciprocal_rank_fusion(all_results)[:final_top_k]",
    "",
    "    # 4. Generate answer",
    "    t4 = time.time()",
    "    context = '\\n\\n'.join(",
    "        f\"[Chunk {i+1} | page {r['metadata'].get('page_num','?')}]\\n{r['text']}\"",
    "        for i, r in enumerate(fused)",
    "    )",
    "    prompt = (",
    "        f'Use ONLY the context below to answer the question. '",
    "        f\"If the answer is not in the context, say 'Not found in context.'\\n\\n\"",
    "        f'Context:\\n{context}\\n\\n'",
    "        f'Question: {question}\\n\\nAnswer:'",
    "    )",
    "    answer = strands_call(",
    "        prompt,",
    '        system="You are a precise Q&A assistant. Answer only from the provided context."',
    "    )",
    '    timing["generation_s"] = round(time.time() - t4, 2)',
    "",
    "    latency_ms = (time.time() - t0) * 1000",
    "",
    "    if verbose:",
    '        print(f"\\nQuestion : {question}")',
    '        print("Variants :")',
    "        for i, v in enumerate(variants):",
    '            tag = "(original)" if i == 0 else f"(variant {i})"',
    '            print(f"  {tag}: {v}")',
    "        print(f\"\\nFused top-{final_top_k} chunks:\")",
    "        for i, r in enumerate(fused, 1):",
    "            print(f\"  [{i}] page={r['metadata'].get('page_num','?')}  \"",
    "                  f\"rrf={r['rrf_score']:.5f}  appearances={r['appearances']}  \"",
    "                  f\"{r['text'][:80]}...\")",
    '        print(f"\\nAnswer:\\n{answer}")',
    "        print(f\"\\nLatency: {latency_ms:.0f}ms  \"",
    "              f\"(variants:{timing['variant_gen_s']}s  \"",
    "              f\"retrieval:{timing['retrieval_s']}s  \"",
    "              f\"generation:{timing['generation_s']}s)\")",
    '        print("-" * 70)',
    "",
    "    return {",
    '        "question"     : question,',
    '        "answer"       : answer,',
    '        "fused_results": fused,',
    '        "variants"     : variants,',
    '        "latency_ms"   : latency_ms,',
    '        "timing"       : timing',
    "    }",
    "",
    "",
    "# Test on climate.pdf questions",
    "test_questions = [",
    '    "What is weather forecasting and why is it important?",',
    '    "What are the main methods used in weather analysis?",',
    '    "How does climatology differ from meteorology?",',
    '    "What factors influence weather patterns and climate?",',
    "]",
    "",
    "fusion_results_log = []",
    "for q in test_questions:",
    "    r = fusion_rag_query(q)",
    "    fusion_results_log.append(r)",
)))

# ── Step 11 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 11 — Comparison: Fusion RAG vs Simple RAG\n"
    "\n"
    "Run both pipelines on the same question and compare which pages are\n"
    "retrieved and how answers differ.\n"
))

cells.append(code(src(
    "def simple_rag_query(question: str, top_k: int = 5) -> Dict:",
    '    "Baseline single-query RAG for comparison."',
    "    t0      = time.time()",
    "    q_emb   = embed_text(question)",
    "    results = vs.search(q_emb, top_k=top_k)",
    "    context = '\\n\\n'.join(",
    "        f\"[Chunk {i+1} | page {r['metadata'].get('page_num','?')}]\\n{r['text']}\"",
    "        for i, r in enumerate(results)",
    "    )",
    "    prompt = (",
    "        f'Use ONLY the context below to answer the question. '",
    "        f\"If the answer is not in the context, say 'Not found in context.'\\n\\n\"",
    "        f'Context:\\n{context}\\n\\n'",
    "        f'Question: {question}\\n\\nAnswer:'",
    "    )",
    "    answer = strands_call(",
    '        prompt, system="You are a precise Q&A assistant. Answer only from the provided context."',
    "    )",
    '    return {"question": question, "answer": answer,',
    '            "results": results, "latency_ms": (time.time()-t0)*1000}',
    "",
    "",
    'compare_q = "What factors influence weather patterns and climate?"',
    "",
    'print("=" * 70)',
    'print("FUSION RAG")',
    'print("=" * 70)',
    "f_result = fusion_rag_query(compare_q, verbose=True)",
    "",
    'print("\\n" + "=" * 70)',
    'print("SIMPLE RAG")',
    'print("=" * 70)',
    "s_result = simple_rag_query(compare_q)",
    'print(f"Question  : {s_result[\'question\']}")',
    "print(f\"Answer    : {s_result['answer'][:300]}...\")",
    "print(f\"Latency   : {s_result['latency_ms']:.0f}ms\")",
    "",
    'print("\\n" + "=" * 70)',
    'print("SUMMARY")',
    'print("=" * 70)',
    "fusion_pages = {r['metadata'].get('page_num') for r in f_result['fused_results']}",
    "simple_pages = {r['metadata'].get('page_num') for r in s_result['results']}",
    "unique_pages = fusion_pages - simple_pages",
    'print(f"Fusion retrieved pages  : {sorted(fusion_pages)}")',
    'print(f"Simple retrieved pages  : {sorted(simple_pages)}")',
    'print(f"Extra pages from fusion : {sorted(unique_pages)} (fusion-only coverage)")',
    "print(f\"Fusion latency : {f_result['latency_ms']:.0f}ms\")",
    "print(f\"Simple latency : {s_result['latency_ms']:.0f}ms\")",
)))

# ── Step 12 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 12 — Evaluation & Metrics"))

cells.append(code(src(
    "eval_cases = [",
    '    {"question": "What is weather forecasting and why is it important?",',
    '     "expected_keywords": ["forecast", "weather", "predict", "atmosphere", "climate"]},',
    '    {"question": "What are the main methods used in weather analysis?",',
    '     "expected_keywords": ["analysis", "synoptic", "observation", "data", "pressure"]},',
    '    {"question": "How does climatology differ from meteorology?",',
    '     "expected_keywords": ["climate", "weather", "long", "study", "atmosphere"]},',
    "]",
    "",
    "print(f\"{'Question':<55} {'KW Hit':>7} {'Latency':>10}\")",
    'print("-" * 75)',
    "",
    "eval_results = []",
    "for case in eval_cases:",
    '    result = fusion_rag_query(case["question"], verbose=False)',
    "    answer_lower = result['answer'].lower()",
    "    hits  = sum(1 for kw in case['expected_keywords'] if kw.lower() in answer_lower)",
    "    total = len(case['expected_keywords'])",
    "    eval_results.append({**result, 'hits': hits, 'total': total})",
    "    print(f\"{case['question'][:54]:<55} {hits}/{total} ({hits/total*100:.0f}%)  \"",
    "          f\"{result['latency_ms']:>7.0f}ms\")",
    "",
    "print()",
    "avg_lat       = sum(r['latency_ms'] for r in eval_results) / len(eval_results)",
    "avg_variant_t = sum(r['timing']['variant_gen_s'] for r in eval_results) / len(eval_results)",
    "avg_gen_t     = sum(r['timing']['generation_s']  for r in eval_results) / len(eval_results)",
    'print(f"Average latency   : {avg_lat:.0f}ms")',
    'print(f"Avg variant gen   : {avg_variant_t:.2f}s")',
    'print(f"Avg answer gen    : {avg_gen_t:.2f}s")',
)))

# ── Step 13 Summary ────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 13 — Summary\n"
    "\n"
    "### What we built\n"
    "| Component | Implementation |\n"
    "|-----------|---------------|\n"
    "| **PDF Loading** | `pypdf.PdfReader` |\n"
    "| **Chunking** | Native Python recursive splitter (1000 chars, 200 overlap) |\n"
    "| **Embeddings** | Amazon Bedrock Titan V2 — 1024 dims |\n"
    "| **Query Variants** | AWS Strands Agent + Claude Sonnet 4.6 |\n"
    "| **Fusion** | Reciprocal Rank Fusion (RRF, k=60) |\n"
    "| **Vector DB** | Qdrant Cloud -> OpenSearch -> in-memory |\n"
    "| **LLM** | AWS Strands Agent + Claude Sonnet 4.6 |\n"
    "\n"
    "### How Fusion RAG improves on Simple RAG\n"
    "- Generates N-1 alternative phrasings -> broader coverage of the document space\n"
    "- RRF rewards chunks that rank highly across multiple query angles\n"
    "- Extra pages retrieved = sections a single query would have missed\n"
    "\n"
    "### When to use\n"
    "- User queries are ambiguous or broad\n"
    "- Document collection has varied vocabulary\n"
    "- Recall matters more than latency\n"
    "\n"
    "### Limitations\n"
    "- ~3-4x slower than Simple RAG (variant generation + N retrievals)\n"
    "- Extra LLM call for variant generation adds cost\n"
    "\n"
    "### Next: **04 — Reranking RAG** (cross-encoder reranking after retrieval)\n"
))

cells.append(code(src(
    "# Optional cleanup -- uncomment to delete the Qdrant collection",
    "# vs.delete_collection()",
    "",
    "print(f\"Collection '{COLLECTION_NAME}' retained in {vs._backend}.\")",
    "print(f'Total vectors stored: {vs.count()}')",
    'print("\\nDone. Give the go-ahead to proceed to notebook 04.")',
)))

# ── Write ─────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"}
    },
    "cells": cells
}

out_path = r"C:\Users\Administrator\RAG\qdrant_notebooks\03_Fusion_Retrieval.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Written : {out_path}")
print(f"Cells   : {len(cells)}")
