"""Build 09_Reranking.ipynb — full implementation."""
import json, uuid, ast

PATH = r"C:\Users\Administrator\RAG\qdrant_notebooks\tier2_retrieval_quality\09_Reranking.ipynb"
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
"# 09 — Reranking\n"
"\n"
"> **Tier 2 | Retrieval Quality**\n"
"\n"
"## The Problem\n"
"\n"
"Vector search retrieves by **embedding similarity** — which is fast but approximate.\n"
"The top-5 results are often good but not in the optimal order:\n"
"- Chunk #3 might answer the question better than chunk #1\n"
"- A semantically similar chunk might be off-topic in context\n"
"\n"
"**Reranking** adds a second, slower but more accurate scoring pass:\n"
"\n"
"```\n"
"Stage 1 — Recall  : vector search → top-20 candidates   (fast, approximate)\n"
"Stage 2 — Precision: reranker scores each candidate      (slow, accurate)\n"
"                   → re-sorted top-5 → LLM\n"
"```\n"
"\n"
"## Two Reranking Approaches\n"
"\n"
"| Approach | How it scores | Speed | Quality |\n"
"|----------|--------------|-------|--------|\n"
"| **LLM Reranker** | Claude scores each chunk 0–10 for relevance | ~1 LLM call per chunk | Highest |\n"
"| **Cross-Encoder style** | Claude compares query+chunk jointly, outputs YES/NO | Faster per call | High |\n"
"\n"
"Both are implemented. LLM reranking is the default — no cross-encoder model weights needed.\n"
"\n"
"## Why not just use top-5 from vector search?\n"
"\n"
"Embedding models compress meaning into 1024 numbers. Information is lost.\n"
"A full LLM reading `(query, chunk)` together recovers that lost signal —\n"
"it can reason about relevance rather than just measuring vector proximity.\n"
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
"        SPLIT --> EMB[\"Embed — Titan V2\"]\n"
"        EMB --> QDRANT[(\"Qdrant\")]\n"
"    end\n"
"\n"
"    subgraph STAGE1 [\"⚡  STAGE 1 — RECALL  (fast)\"]\n"
"        Q([\"❓ Query\"])\n"
"        Q --> QEMB[\"embed(query)\"]\n"
"        QEMB --> VS[\"Vector search\\ntop-20 candidates\"]\n"
"        QDRANT --> VS\n"
"        VS --> CANDS([\"20 candidates\"])\n"
"    end\n"
"\n"
"    subgraph STAGE2 [\"🧠  STAGE 2 — PRECISION  (accurate)\"]\n"
"        CANDS --> SCORE[\"LLM scores each chunk\\n0-10 relevance to query\\n~1 call per chunk\"]\n"
"        SCORE --> RESORT[\"Re-sort by LLM score\"]\n"
"        RESORT --> TOP5([\"Top-5 reranked\"])\n"
"    end\n"
"\n"
"    subgraph GEN [\"🟠  GENERATION\"]\n"
"        TOP5 --> LLM2[\"Strands Agent\\nClaude Sonnet 4.6\"]\n"
"        LLM2 --> ANS([\"✅ Answer\"])\n"
"    end\n"
"\n"
"    subgraph COMPARE [\"📊  COMPARE\"]\n"
"        direction LR\n"
"        VEC_ONLY[\"Vector top-5\\n(no rerank)\"] \n"
"        RERANKED[\"Reranked top-5\"]\n"
"        VEC_ONLY --> DIFF[\"Score delta &\\nrank changes\"]\n"
"        RERANKED --> DIFF\n"
"    end\n"
"\n"
"    style INDEX   fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f\n"
"    style STAGE1  fill:#dcfce7,stroke:#16a34a,color:#14532d\n"
"    style STAGE2  fill:#fef9c3,stroke:#ca8a04,color:#713f12\n"
"    style GEN     fill:#ffedd5,stroke:#f97316,color:#7c2d12\n"
"    style COMPARE fill:#faf5ff,stroke:#a855f7,color:#3b0764\n"
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
    "import os, json, time, uuid, re",
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
    'COLLECTION_NAME = "reranking_rag_09"',
    "EMBEDDING_DIM   = 1024",
    "",
    "# Retrieval pipeline",
    "RECALL_K  = 20   # candidates from vector search (wide net)",
    "RERANK_K  = 5    # final results after reranking (sent to LLM)",
    "",
    "# Chunking",
    "CHUNK_SIZE    = 500",
    "CHUNK_OVERLAP = 50",
    "",
    "# PDF",
    'PDF_PATH = r"' + PDF + '"',
    "",
    'print(f"Recall K  : {RECALL_K}  candidates from vector search")',
    'print(f"Rerank K  : {RERANK_K}  final results after reranking")',
    'print(f"Collection: {COLLECTION_NAME}")',
    'print(f"PDF       : {PDF_PATH}")',
    'print(f"PDF exists: {os.path.exists(PDF_PATH)}")',
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
    "                    connection_class=RequestsHttpConnection, timeout=30)",
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
    "    def search(self, qvec: List[float], top_k: int = 20) -> List[Dict]:",
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
    'test_emb = embed_text("reranking relevance scoring")',
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
cells.append(md("## Step 6 — Load PDF, Chunk & Index"))

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
    'print(f"Chunks : {len(chunks)}  |  avg {sum(len(c) for c in chunks)//len(chunks)} chars")',
    "",
    'print(f"Embedding {len(chunks)} chunks...")',
    "t0   = time.time()",
    "embs = embed_batch(chunks, label='[index]')",
    "docs = [{'text': chunks[i], 'embedding': embs[i],",
    "         'metadata': {'chunk_idx': i, 'source': 'climate.pdf'}}",
    "        for i in range(len(chunks))]",
    "vs.upsert(docs)",
    'print(f"Indexed {vs.count()} vectors in {time.time()-t0:.1f}s")',
)))

# ── Step 7 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 7 — LLM Reranker (score 0–10)\n\n"
    "For each candidate chunk, Claude receives the query and the chunk,\n"
    "and returns a relevance score from 0–10 as a raw integer.\n\n"
    "**Prompt design:**\n"
    "- Ask for a single integer — avoids parsing complex output\n"
    "- Give explicit scoring rubric — makes scores consistent across calls\n"
    "- Score each chunk **independently** — no cross-chunk comparison needed\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "RERANK_PROMPT = (",
    "    'Score the relevance of the passage below to the question on a scale of 0-10.\\n\\n'",
    "    'Scoring rubric:\\n'",
    "    '  10 = passage directly and completely answers the question\\n'",
    "    '   7 = passage contains key information relevant to the question\\n'",
    "    '   4 = passage is tangentially related\\n'",
    "    '   1 = passage is on a related topic but does not help answer the question\\n'",
    "    '   0 = passage is completely unrelated\\n\\n'",
    "    'Question: {question}\\n\\n'",
    "    'Passage:\\n{passage}\\n\\n'",
    "    'Output ONLY a single integer 0-10. No explanation.'",
    ")",
    "",
    "def score_chunk(question: str, chunk: str) -> float:",
    "    prompt = RERANK_PROMPT.format(question=question, passage=chunk)",
    "    raw = str(Agent(",
    "        model=_model,",
    "        system_prompt='You are a relevance scoring assistant. Output only a single integer.'",
    "    )(prompt)).strip()",
    "    # Parse first integer found in response",
    "    match = re.search(r'\\d+', raw)",
    "    if match:",
    "        return min(10.0, max(0.0, float(match.group())))",
    "    return 0.0",
    "",
    "def rerank_llm(question: str, candidates: List[Dict], top_k: int = RERANK_K) -> List[Dict]:",
    "    scored = []",
    "    for i, doc in enumerate(candidates):",
    "        s = score_chunk(question, doc['text'])",
    "        scored.append({**doc, 'rerank_score': s, 'original_rank': i + 1})",
    "        time.sleep(0.05)",
    "    scored.sort(key=lambda x: x['rerank_score'], reverse=True)",
    "    for new_rank, doc in enumerate(scored, 1):",
    "        doc['new_rank'] = new_rank",
    "    return scored[:top_k]",
    "",
    "# Smoke test on 3 chunks",
    'test_q   = "What methods are used in weather analysis?"',
    "test_hits = vs.search(embed_text(test_q), top_k=3)",
    'print(f"Smoke test — scoring 3 chunks for: {test_q}")',
    "for h in test_hits:",
    "    s = score_chunk(test_q, h['text'])",
    "    print(f'  vec_score={h[\"score\"]:.4f}  rerank={s:.0f}/10  {h[\"text\"][:70]}...')",
)))

# ── Step 8 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 8 — Cross-Encoder Style Reranker (YES / NO)\n\n"
    "An alternative reranking signal: ask Claude to make a binary relevance decision.\n"
    "Faster per call than scoring, works well for filtering clearly irrelevant chunks.\n\n"
    "After filtering, remaining chunks are sorted by their original vector score.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "CE_PROMPT = (",
    "    'Does the following passage contain information that directly helps answer the question?\\n\\n'",
    "    'Question: {question}\\n\\n'",
    "    'Passage:\\n{passage}\\n\\n'",
    "    'Answer YES or NO only.'",
    ")",
    "",
    "def is_relevant(question: str, chunk: str) -> bool:",
    "    prompt = CE_PROMPT.format(question=question, passage=chunk)",
    "    raw = str(Agent(",
    "        model=_model,",
    "        system_prompt='You are a relevance filter. Answer YES or NO only.'",
    "    )(prompt)).strip().upper()",
    "    return raw.startswith('Y')",
    "",
    "def rerank_cross_encoder(question: str, candidates: List[Dict], top_k: int = RERANK_K) -> List[Dict]:",
    "    relevant = []",
    "    for i, doc in enumerate(candidates):",
    "        if is_relevant(question, doc['text']):",
    "            relevant.append({**doc, 'original_rank': i + 1, 'ce_relevant': True})",
    "        else:",
    "            pass   # filtered out",
    "        time.sleep(0.05)",
    "    # Sort retained docs by original vector score (already ordered from recall stage)",
    "    for new_rank, doc in enumerate(relevant[:top_k], 1):",
    "        doc['new_rank'] = new_rank",
    "    return relevant[:top_k]",
    "",
    'print("rerank_llm()           — score 0-10, pick top-K")',
    'print("rerank_cross_encoder() — YES/NO filter, keep orignal order")',
)))

# ── Step 9 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 9 — Full RAG Pipeline (three modes)"))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def rag_query(",
    "    question: str,",
    "    mode: str = 'rerank_llm',   # 'vector_only' | 'rerank_llm' | 'rerank_ce'",
    "    verbose: bool = True",
    ") -> Dict:",
    "    t0       = time.time()",
    "    qvec     = embed_text(question)",
    "    recall_k = RECALL_K if mode != 'vector_only' else RERANK_K",
    "    cands    = vs.search(qvec, top_k=recall_k)",
    "",
    "    if mode == 'rerank_llm':",
    "        final = rerank_llm(question, cands, top_k=RERANK_K)",
    "    elif mode == 'rerank_ce':",
    "        final = rerank_cross_encoder(question, cands, top_k=RERANK_K)",
    "    else:",
    "        final = [{**c, 'original_rank': i+1, 'new_rank': i+1}",
    "                 for i, c in enumerate(cands[:RERANK_K])]",
    "",
    "    answer  = generate_answer(question, [h['text'] for h in final])",
    "    latency = (time.time() - t0) * 1000",
    "",
    "    if verbose:",
    '        print(f"\\n[{mode.upper()}]  Q: {question}")',
    '        print(f"A: {answer}")',
    "        for h in final[:3]:",
    "            sc = h.get('rerank_score', h.get('score', 0))",
    "            mv = h.get('original_rank', '?')",
    "            nv = h.get('new_rank', '?')",
    "            print(f\"  rank {mv}->{nv}  score={sc:.2f}  {h['text'][:70]}...\")",
    '        print(f"  Latency: {latency:.0f}ms")',
    '        print("-" * 70)',
    "",
    "    return {'question': question, 'answer': answer, 'mode': mode,",
    "            'results': final, 'latency_ms': latency,",
    "            'n_recall': len(cands), 'n_final': len(final)}",
    "",
    "test_questions = [",
    '    "What is weather forecasting and why is it important?",',
    '    "What are the main methods used in weather analysis?",',
    '    "How does climatology differ from meteorology?",',
    '    "What factors influence weather patterns and climate?",',
    "]",
    "results_log = []",
    "for q in test_questions:",
    "    results_log.append(rag_query(q, mode='rerank_llm'))",
)))

# ── Step 10 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 10 — Rank Change Analysis\n\n"
    "Show how reranking reshuffles the top-K results for a single query.\n"
    "Chunks that move up significantly are those where vector similarity\n"
    "underestimated actual relevance.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "demo_q = 'What are the main methods used in weather analysis?'",
    "qvec   = embed_text(demo_q)",
    "cands  = vs.search(qvec, top_k=RECALL_K)",
    "",
    'print(f"Q: {demo_q}")',
    'print(f"Recall pool: {len(cands)} candidates")',
    'print("\\nScoring all candidates...")',
    "reranked = rerank_llm(demo_q, cands, top_k=RERANK_K)",
    "",
    "print()",
    "print('{:<5} {:<5} {:<10} {:<10}  {}'.format(",
    "    'Old#', 'New#', 'Vec score', 'LLM score', 'Chunk preview'))",
    "print('-' * 80)",
    "for doc in reranked:",
    "    moved = doc['new_rank'] - doc['original_rank']",
    "    arrow = '  ' if moved == 0 else (f'  ^{-moved}' if moved < 0 else f'  v{moved}')",
    "    print('{:<5} {:<5} {:<10.4f} {:<10.0f}  {}{}'.format(",
    "        doc['original_rank'], doc['new_rank'],",
    "        doc['score'], doc['rerank_score'],",
    "        doc['text'][:55], arrow))",
    "",
    "# Also show bottom-5 from vector search that got filtered out",
    "reranked_ids = {d['id'] for d in reranked}",
    "dropped = [c for c in cands[:10] if c['id'] not in reranked_ids]",
    'print(f"\\nDropped from top-10 vector results (not in reranked top-{RERANK_K}):")',
    "for d in dropped[:3]:",
    "    print(f\"  vec_rank={cands.index(d)+1}  vec_score={d['score']:.4f}  {d['text'][:70]}...\")",
)))

# ── Step 11 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 11 — Three-Mode Comparison\n\n"
    "Compare `vector_only`, `rerank_llm`, and `rerank_ce` on the same questions:\n"
    "keyword hit rate, latency, and number of final results.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "compare_qs = [",
    '    "What is weather forecasting and why is it important?",',
    '    "What are the main methods used in weather analysis?",',
    '    "How does climatology differ from meteorology?",',
    "]",
    "modes = ['vector_only', 'rerank_llm', 'rerank_ce']",
    "keywords_map = {",
    '    "What is weather forecasting and why is it important?": [\'forecast\',\'weather\',\'predict\',\'atmosphere\',\'climate\'],',
    '    "What are the main methods used in weather analysis?": [\'analysis\',\'synoptic\',\'observation\',\'data\',\'pressure\'],',
    '    "How does climatology differ from meteorology?": [\'climate\',\'weather\',\'long\',\'study\',\'atmosphere\'],',
    "}",
    "",
    "print('{:<50} {:>16}  {:>16}  {:>14}'.format(",
    "    'Question', 'Vector-only', 'Rerank-LLM', 'Rerank-CE'))",
    "print('-' * 100)",
    "",
    "for q in compare_qs:",
    "    kws = keywords_map[q]; n = len(kws)",
    "    cols = []",
    "    for mode in modes:",
    "        r    = rag_query(q, mode=mode, verbose=False)",
    "        hits = sum(1 for kw in kws if kw in r['answer'].lower())",
    "        cols.append(f'{hits}/{n}({hits/n*100:.0f}%) {r[\"latency_ms\"]:.0f}ms')",
    "    print('{:<50} {:>16}  {:>16}  {:>14}'.format(q[:49], cols[0], cols[1], cols[2]))",
)))

# ── Step 12 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 12 — Batch Reranking (efficiency pattern)\n\n"
    "Instead of scoring chunks one at a time, ask the LLM to rank **all candidates\n"
    "at once** in a single call — much cheaper for large recall pools.\n\n"
    "Trade-off: single-call batch ranking is slightly less accurate than\n"
    "independent per-chunk scoring, but dramatically faster (1 call vs. 20 calls).\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "BATCH_RERANK_PROMPT = (",
    "    'Below are {n} passages. Rank them from MOST to LEAST relevant to the question.\\n'",
    "    'Output ONLY a comma-separated list of passage numbers in order of relevance.\\n'",
    "    'Example output for 5 passages: 3,1,5,2,4\\n\\n'",
    "    'Question: {question}\\n\\n'",
    "    '{passages}\\n\\n'",
    "    'Ranked order (most to least relevant):'",
    ")",
    "",
    "def rerank_batch(question: str, candidates: List[Dict], top_k: int = RERANK_K) -> List[Dict]:",
    "    passages_text = '\\n\\n'.join(",
    "        f'[{i+1}] {doc[\"text\"][:200]}' for i, doc in enumerate(candidates)",
    "    )",
    "    prompt = BATCH_RERANK_PROMPT.format(",
    "        n=len(candidates), question=question, passages=passages_text",
    "    )",
    "    raw = str(Agent(",
    "        model=_model,",
    "        system_prompt='You are a passage ranking assistant. Output only comma-separated numbers.'",
    "    )(prompt)).strip()",
    "    # Parse ranked indices",
    "    nums = []",
    "    for tok in re.split(r'[,\\s]+', raw):",
    "        tok = tok.strip()",
    "        if tok.isdigit():",
    "            idx = int(tok) - 1",
    "            if 0 <= idx < len(candidates) and idx not in nums:",
    "                nums.append(idx)",
    "    # Append any missing indices at the end",
    "    for i in range(len(candidates)):",
    "        if i not in nums: nums.append(i)",
    "    ranked = [{**candidates[i], 'original_rank': i+1, 'new_rank': new_rank+1}",
    "              for new_rank, i in enumerate(nums)]",
    "    return ranked[:top_k]",
    "",
    "# Demo batch rerank vs per-chunk rerank",
    'batch_q = "What factors influence weather patterns and climate?"',
    "qvec    = embed_text(batch_q)",
    "cands   = vs.search(qvec, top_k=10)  # smaller pool for demo",
    "",
    "t0  = time.time()",
    "per_chunk = rerank_llm(batch_q, cands, top_k=RERANK_K)",
    "per_t = (time.time()-t0)*1000",
    "",
    "t0    = time.time()",
    "batch = rerank_batch(batch_q, cands, top_k=RERANK_K)",
    "bat_t = (time.time()-t0)*1000",
    "",
    'print(f"Per-chunk reranking : {per_t:.0f}ms  ({len(cands)} LLM calls)")',
    'print(f"Batch reranking     : {bat_t:.0f}ms  (1 LLM call)")',
    'print(f"\\nPer-chunk top-3:")',
    "for d in per_chunk[:3]:",
    "    print(f\"  {d['original_rank']}->{d['new_rank']}  score={d['rerank_score']:.0f}/10  {d['text'][:70]}...\")",
    'print(f"\\nBatch top-3:")',
    "for d in batch[:3]:",
    "    print(f\"  {d['original_rank']}->{d['new_rank']}  {d['text'][:70]}...\")",
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
    "print('{:<50} {:>9}  {:>11}  {:>10}'.format(",
    "    'Question', 'VecOnly', 'RerankLLM', 'RerankCE'))",
    "print('-' * 87)",
    "for case in eval_cases:",
    "    q = case['question']; kws = case['keywords']; n = len(kws)",
    "    cols = []",
    "    for mode in ['vector_only','rerank_llm','rerank_ce']:",
    "        r    = rag_query(q, mode=mode, verbose=False)",
    "        hits = sum(1 for kw in kws if kw in r['answer'].lower())",
    "        cols.append(f'{hits}/{n}({hits/n*100:.0f}%)')",
    "    print('{:<50} {:>9}  {:>11}  {:>10}'.format(q[:49], cols[0], cols[1], cols[2]))",
    "",
    "print()",
    'print(f"Recall K  : {RECALL_K}  |  Final K: {RERANK_K}")',
    'print(f"Chunks in Qdrant: {vs.count()}")',
)))

# ── Step 14 Summary ────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 14 — Summary\n\n"
    "| Component | Implementation |\n"
    "|-----------|---------------|\n"
    "| PDF | `pypdf.PdfReader` — absolute path in Step 2 |\n"
    "| Stage 1 recall | Qdrant vector search — top-20 candidates |\n"
    "| Stage 2 rerank (LLM) | Claude scores each chunk 0–10 — sort, keep top-5 |\n"
    "| Stage 2 rerank (CE) | Claude binary YES/NO filter — keep relevant, sort by vector score |\n"
    "| Batch rerank | Single LLM call ranks all candidates — 1 call vs N calls |\n"
    "| Vector DB | Qdrant Cloud → OpenSearch → in-memory |\n"
    "| LLM | AWS Strands Agent + Claude Sonnet 4.6 |\n\n"
    "## Reranking trade-offs\n\n"
    "| Approach | Latency | Accuracy | Cost |\n"
    "|----------|---------|----------|------|\n"
    "| Vector only | ~50ms | Good | Low |\n"
    "| Batch rerank | +1 LLM call | Better | Medium |\n"
    "| Per-chunk LLM | +N LLM calls | Best | High |\n"
    "| Cross-encoder | +N LLM calls | High | Medium |\n\n"
    "**Production tip:** use batch rerank for most queries;\n"
    "use per-chunk LLM scoring only for high-stakes or ambiguous queries.\n\n"
    "### Next: **10 — Contextual Compression**\n"
))

cells.append(code(L(
    "# vs.delete_collection()  # uncomment to clean up",
    "print(f\"Collection '{COLLECTION_NAME}' in {vs._backend} — {vs.count()} vectors\")",
    'print("\\nDone. Give the go-ahead for notebook 10.")',
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
