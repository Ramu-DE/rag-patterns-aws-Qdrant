"""Build 06_Contextual_Retrieval.ipynb — full implementation."""
import json, uuid, ast

PATH = r"C:\Users\Administrator\RAG\qdrant_notebooks\tier1_chunking_indexing\06_Contextual_Retrieval.ipynb"
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
"# 06 — Contextual Retrieval\n"
"\n"
"> **Tier 1 | Chunking & Indexing Foundations**\n"
"\n"
"## What is Contextual Retrieval?\n"
"\n"
"An Anthropic-published technique (Sept 2024) that reduces retrieval failures\n"
"by **49–67%**. The core insight: chunks fail to retrieve because they are\n"
"**context-stripped** — a sentence like *'The temperature dropped sharply'*\n"
"is meaningless without knowing *which document, which section, which topic*.\n"
"\n"
"**Fix:** before indexing, ask the LLM to generate a 1–2 sentence context\n"
"summary for every chunk, then **prepend it** to the chunk text before embedding.\n"
"\n"
"```\n"
"Without contextual retrieval:\n"
"  chunk = 'The temperature dropped sharply in the upper troposphere.'\n"
"\n"
"With contextual retrieval:\n"
"  chunk = 'This chunk is from a section on atmospheric temperature profiles\n"
"           in a climate science textbook discussing the tropopause.\\n\\n\n"
"           The temperature dropped sharply in the upper troposphere.'\n"
"```\n"
"\n"
"The embedding of the enriched chunk retrieves much more reliably because the\n"
"vector now encodes *topic + location + content* rather than content alone.\n"
"\n"
"## Two variants implemented here\n"
"\n"
"| Variant | What gets indexed |\n"
"|---------|------------------|\n"
"| **Baseline** | Raw fixed-size chunks (no context) |\n"
"| **Contextual** | `context_summary + '\\n\\n' + chunk_text` |\n"
"\n"
"Both use identical Qdrant collections so you can run A/B queries.\n"
"\n"
"## Cost note\n"
"Contextual retrieval calls the LLM **once per chunk** during indexing.\n"
"For 100 chunks that is ~100 LLM calls. This is a one-time indexing cost —\n"
"query latency is identical to baseline after indexing.\n"
))

# ── Flow diagram ──────────────────────────────────────────────────────────────
cells.append(md(
"## Flow Diagram\n"
"\n"
"```mermaid\n"
"flowchart TD\n"
"    subgraph BASELINE [\"⚪  BASELINE INDEX\"]\n"
"        PDF0([\"📄 climate.pdf\"])\n"
"        PDF0 --> SPLIT0[\"Fixed-size chunks\\n~500 chars\"]\n"
"        SPLIT0 --> EMB0[\"Embed chunk text\\nTitan V2\"]\n"
"        EMB0 --> QDB[(\"Qdrant\\nbaseline_col\")]\n"
"    end\n"
"\n"
"    subgraph CONTEXTUAL [\"🔵  CONTEXTUAL INDEX\"]\n"
"        PDF1([\"📄 climate.pdf\"])\n"
"        PDF1 --> SPLIT1[\"Same fixed-size chunks\"]\n"
"        SPLIT1 --> PROMPT[\"Prompt LLM:\\n'Describe where this chunk\\nfits in the document'\"]\n"
"        FULL([\"Full document text\"])\n"
"        FULL --> PROMPT\n"
"        PROMPT --> LLM1[\"Claude Sonnet 4.6\\n→ 1-2 sentence context\"]\n"
"        LLM1 --> ENRICH[\"context_summary\\n+ chunk_text\"]\n"
"        SPLIT1 --> ENRICH\n"
"        ENRICH --> EMB1[\"Embed enriched text\\nTitan V2\"]\n"
"        EMB1 --> QDC[(\"Qdrant\\ncontextual_col\")]\n"
"    end\n"
"\n"
"    subgraph QUERY [\"🟢  QUERY (same for both)\"]\n"
"        Q([\"❓ User query\"])\n"
"        Q --> QEMB[\"Embed query\\nTitan V2\"]\n"
"        QEMB --> SEARCH_B[\"Search baseline\"]\n"
"        QEMB --> SEARCH_C[\"Search contextual\"]\n"
"        QDB --> SEARCH_B\n"
"        QDC --> SEARCH_C\n"
"        SEARCH_B --> LLM_B[\"LLM → Answer A\"]\n"
"        SEARCH_C --> LLM_C[\"LLM → Answer B\"]\n"
"        LLM_B --> CMP([\"📊 A/B Compare\"])\n"
"        LLM_C --> CMP\n"
"    end\n"
"\n"
"    style BASELINE    fill:#f1f5f9,stroke:#94a3b8,color:#334155\n"
"    style CONTEXTUAL  fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f\n"
"    style QUERY       fill:#dcfce7,stroke:#16a34a,color:#14532d\n"
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
    "from typing import List, Dict, Optional",
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
    "# Two collections — one per variant",
    'COL_BASELINE    = "contextual_retrieval_06_baseline"',
    'COL_CONTEXTUAL  = "contextual_retrieval_06_contextual"',
    "EMBEDDING_DIM   = 1024",
    "TOP_K           = 5",
    "",
    "# Chunking",
    "CHUNK_SIZE      = 500",
    "CHUNK_OVERLAP   = 50",
    "",
    "# Context generation — max chars of document to pass in the prompt",
    "# Using the full document gives best context but costs more tokens.",
    "# Set to 0 to pass the full text.",
    "DOC_WINDOW_CHARS = 8000   # first 8k chars of the document as surrounding context",
    "",
    "# PDF",
    'PDF_PATH = r"' + PDF + '"',
    "",
    'print(f"Baseline collection   : {COL_BASELINE}")',
    'print(f"Contextual collection : {COL_CONTEXTUAL}")',
    'print(f"Chunk size            : {CHUNK_SIZE} chars")',
    'print(f"Doc window for LLM    : {DOC_WINDOW_CHARS} chars")',
    'print(f"PDF                   : {PDF_PATH}")',
    'print(f"PDF exists            : {os.path.exists(PDF_PATH)}")',
)))

# ── Step 3 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 3 — Vector Store"))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "class VectorStore:",
    "    def __init__(self, collection_name, qdrant_url='', qdrant_api_key='',",
    "                 opensearch_url='', region='us-east-1'):",
    "        self.name     = collection_name",
    "        self.region   = region",
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
    "    def search(self, qvec: List[float], top_k: int = 5) -> List[Dict]:",
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
    "        elif self._backend == 'opensearch':",
    "            return self._os.count(index=self.name).get('count', 0)",
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
    "        if (i + 1) % 10 == 0:",
    "            print(f'  {label} {i+1}/{len(texts)}')",
    "        time.sleep(0.04)",
    "    return out",
    "",
    "_model = BedrockModel(model_id=LLM_MODEL, region_name=AWS_REGION)",
    "",
    "def generate_answer(question: str, context_chunks: List[str]) -> str:",
    "    context = '\\n\\n'.join(f'[Context {i+1}]\\n{c}' for i, c in enumerate(context_chunks))",
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
    'test_emb = embed_text("contextual retrieval test")',
    'print(f"Embedding OK — dim={len(test_emb)}")',
    'print("Strands BedrockModel ready.")',
)))

# ── Step 5 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 5 — Connect & Create Both Collections"))

cells.append(code(L(
    "vs_base = VectorStore(",
    "    collection_name=COL_BASELINE,",
    "    qdrant_url=QDRANT_URL, qdrant_api_key=QDRANT_API_KEY,",
    "    opensearch_url=OPENSEARCH_URL, region=AWS_REGION)",
    "",
    "vs_ctx = VectorStore(",
    "    collection_name=COL_CONTEXTUAL,",
    "    qdrant_url=QDRANT_URL, qdrant_api_key=QDRANT_API_KEY,",
    "    opensearch_url=OPENSEARCH_URL, region=AWS_REGION)",
    "",
    'print(f"Backend: {vs_base._backend}")',
    "vs_base.create_collection(dim=EMBEDDING_DIM, force_recreate=True)",
    "vs_ctx.create_collection(dim=EMBEDDING_DIM,  force_recreate=True)",
)))

# ── Step 6 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 6 — Recursive Splitter & Load PDF"))

cells.append(code(L(
    "from typing import List",
    "",
    "def recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:",
    '    separators = ["\\n\\n", "\\n", ". ", " ", ""]',
    "",
    "    def _split(text: str, seps: List[str]) -> List[str]:",
    "        sep, new_seps = '', []",
    "        for i, s in enumerate(seps):",
    "            if s == '' or s in text:",
    "                sep, new_seps = s, seps[i+1:]",
    "                break",
    "        parts = text.split(sep) if sep != '' else list(text)",
    "        good: List[str] = []",
    "        for part in parts:",
    "            if len(part) <= chunk_size: good.append(part)",
    "            elif new_seps:             good.extend(_split(part, new_seps))",
    "            else:",
    "                for k in range(0, len(part), chunk_size - chunk_overlap):",
    "                    good.append(part[k : k + chunk_size])",
    "        chunks, cur_pieces, cur_len = [], [], 0",
    "        for piece in good:",
    "            p = piece.strip()",
    "            if not p: continue",
    "            addition = len(sep) + len(p) if cur_pieces else len(p)",
    "            if cur_len + addition <= chunk_size:",
    "                cur_pieces.append(p); cur_len += addition",
    "            else:",
    "                if cur_pieces: chunks.append(sep.join(cur_pieces))",
    "                overlap_pieces, overlap_len = [], 0",
    "                for prev in reversed(cur_pieces):",
    "                    if overlap_len + len(prev) + len(sep) <= chunk_overlap:",
    "                        overlap_pieces.insert(0, prev)",
    "                        overlap_len += len(prev) + len(sep)",
    "                    else: break",
    "                cur_pieces = overlap_pieces + [p]",
    "                cur_len = sum(len(x) + len(sep) for x in cur_pieces)",
    "        if cur_pieces: chunks.append(sep.join(cur_pieces))",
    "        return [c for c in chunks if c.strip()]",
    "",
    "    return _split(text, separators)",
    "",
    "# Load PDF",
    "reader    = pypdf.PdfReader(PDF_PATH)",
    'print(f"PDF    : {PDF_PATH}")',
    'print(f"Pages  : {len(reader.pages)}")',
    "full_text = ''",
    "for page in reader.pages:",
    "    full_text += (page.extract_text() or '') + '\\n\\n'",
    'print(f"Chars  : {len(full_text):,}")',
    "",
    "# Split into raw chunks",
    "raw_chunks = recursive_split(full_text, CHUNK_SIZE, CHUNK_OVERLAP)",
    'print(f"Chunks : {len(raw_chunks)}")',
    'print(f"Avg    : {sum(len(c) for c in raw_chunks)/len(raw_chunks):.0f} chars")',
    "",
    "# Document window — passed to the context-generation prompt",
    "# Using first DOC_WINDOW_CHARS chars gives the LLM enough scope info",
    "# without burning full-document tokens on every chunk call",
    "doc_window = full_text[:DOC_WINDOW_CHARS] if DOC_WINDOW_CHARS else full_text",
    'print(f"Doc window passed to LLM: {len(doc_window):,} chars")',
)))

# ── Step 7 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 7 — Generate Context Summaries (one LLM call per chunk)\n\n"
    "For every chunk, Claude receives:\n"
    "- The surrounding document scope (first 8 k chars)\n"
    "- The chunk itself\n\n"
    "It returns a 1–2 sentence description of where the chunk fits in the document.\n"
    "That summary is then **prepended** to the chunk before embedding.\n\n"
    "> **Cost:** ~1 LLM call × N chunks. For 100 chunks that is ~100 calls.\n"
    "> This happens once at index time; queries have no extra cost.\n"
))

cells.append(code(L(
    "from typing import List",
    "",
    "CONTEXT_PROMPT = (",
    "    'Here is a document excerpt for context:\\n'",
    "    '<document>\\n{doc_window}\\n</document>\\n\\n'",
    "    'Here is the specific chunk you need to contextualise:\\n'",
    "    '<chunk>\\n{chunk}\\n</chunk>\\n\\n'",
    "    'Write 1-2 sentences that describe where this chunk sits in the document '",
    "    'and what topic or concept it covers. Be specific and concise. '",
    "    'Output ONLY the description, no preamble.'",
    ")",
    "",
    "def generate_context(chunk: str, doc_win: str) -> str:",
    "    prompt = CONTEXT_PROMPT.format(doc_window=doc_win, chunk=chunk)",
    "    return str(Agent(",
    "        model=_model,",
    "        system_prompt='You are a document analyst. Generate precise chunk context descriptions.'",
    "    )(prompt)).strip()",
    "",
    "def build_contextual_chunks(chunks: List[str], doc_win: str) -> List[Dict]:",
    "    results = []",
    "    for i, chunk in enumerate(chunks):",
    "        t0  = time.time()",
    "        ctx = generate_context(chunk, doc_win)",
    "        enriched = ctx + '\\n\\n' + chunk",
    "        results.append({'raw': chunk, 'context': ctx, 'enriched': enriched})",
    "        lat = (time.time() - t0) * 1000",
    "        if (i + 1) % 5 == 0 or i == 0:",
    "            print(f'  [{i+1}/{len(chunks)}] {lat:.0f}ms | ctx: {ctx[:70]}...')",
    "    return results",
    "",
    'print(f"Generating context for {len(raw_chunks)} chunks...")',
    'print("(This takes ~1-2s per chunk — total ~2-4 min for a 100-chunk document)")',
    "t0 = time.time()",
    "contextual_chunks = build_contextual_chunks(raw_chunks, doc_window)",
    'print(f"Done in {time.time()-t0:.1f}s")',
    'print(f"\\nSample — chunk 0:")',
    'print(f"  Context : {contextual_chunks[0][\'context\']}")',
    'print(f"  Raw     : {contextual_chunks[0][\'raw\'][:120]}...")',
    'print(f"  Enriched: {contextual_chunks[0][\'enriched\'][:160]}...")',
)))

# ── Step 8 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 8 — Embed & Index Both Collections\n\n"
    "**Baseline:** embed `raw_chunks` → `COL_BASELINE`\n\n"
    "**Contextual:** embed `enriched` (context + chunk) → `COL_CONTEXTUAL`\n\n"
    "Both collections have identical structure; only the embedded text differs.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "# ── Baseline ──",
    'print("Embedding baseline (raw chunks)...")',
    "t0        = time.time()",
    "base_embs = embed_batch(raw_chunks, label='[baseline]')",
    "base_docs = [",
    "    {'text': raw_chunks[i], 'embedding': base_embs[i],",
    "     'metadata': {'chunk_idx': i, 'variant': 'baseline', 'source': 'climate.pdf'}}",
    "    for i in range(len(raw_chunks))",
    "]",
    "vs_base.upsert(base_docs)",
    'print(f"Baseline indexed: {vs_base.count()} vectors in {time.time()-t0:.1f}s")',
    "",
    "# ── Contextual ──",
    'print("Embedding contextual (enriched chunks)...")',
    "t0         = time.time()",
    "enr_texts  = [c['enriched'] for c in contextual_chunks]",
    "ctx_embs   = embed_batch(enr_texts, label='[contextual]')",
    "ctx_docs   = [",
    "    {'text': contextual_chunks[i]['raw'],   # store raw text for display",
    "     'embedding': ctx_embs[i],",
    "     'metadata': {",
    "         'chunk_idx': i,",
    "         'context'  : contextual_chunks[i]['context'],",
    "         'variant'  : 'contextual',",
    "         'source'   : 'climate.pdf'",
    "     }}",
    "    for i in range(len(contextual_chunks))",
    "]",
    "vs_ctx.upsert(ctx_docs)",
    'print(f"Contextual indexed: {vs_ctx.count()} vectors in {time.time()-t0:.1f}s")',
)))

# ── Step 9 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 9 — RAG Query (both variants)\n\n"
    "`rag_query_both()` runs the same question against baseline and contextual\n"
    "and returns both answers side-by-side for comparison.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def rag_query_single(question: str, vs: VectorStore, top_k: int = TOP_K) -> Dict:",
    "    t0      = time.time()",
    "    hits    = vs.search(embed_text(question), top_k=top_k)",
    "    answer  = generate_answer(question, [h['text'] for h in hits])",
    "    latency = (time.time() - t0) * 1000",
    "    return {'answer': answer, 'latency_ms': latency, 'hits': hits}",
    "",
    "def rag_query_both(question: str, verbose: bool = True) -> Dict:",
    "    base = rag_query_single(question, vs_base)",
    "    ctx  = rag_query_single(question, vs_ctx)",
    "    if verbose:",
    '        print(f"\\nQ: {question}")',
    '        print(f"\\n[BASELINE]  ({base[\'latency_ms\']:.0f}ms)")',
    "        print(f\"  {base['answer']}\")",
    '        print(f"\\n[CONTEXTUAL] ({ctx[\'latency_ms\']:.0f}ms)")',
    "        print(f\"  {ctx['answer']}\")",
    '        print(f"\\nTop baseline hit  : {base[\'hits\'][0][\'text\'][:80]}...")',
    '        print(f"Top contextual hit: {ctx[\'hits\'][0][\'text\'][:80]}...")',
    '        print("-" * 70)',
    "    return {'question': question, 'baseline': base, 'contextual': ctx}",
    "",
    "test_questions = [",
    '    "What is weather forecasting and why is it important?",',
    '    "What are the main methods used in weather analysis?",',
    '    "How does climatology differ from meteorology?",',
    '    "What factors influence weather patterns and climate?",',
    "]",
    "results_log = [rag_query_both(q) for q in test_questions]",
)))

# ── Step 10 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 10 — A/B Retrieval Score Comparison\n\n"
    "For each test question, compare the **top-1 similarity score** from each variant.\n"
    "A higher score from contextual = the enriched embedding is a better semantic match.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "compare_qs = [",
    '    "What is weather forecasting and why is it important?",',
    '    "What are the main methods used in weather analysis?",',
    '    "How does climatology differ from meteorology?",',
    "]",
    "",
    "print('{:<55} {:>10}  {:>11}  {:>6}'.format(",
    "    'Question', 'Base top-1', 'Ctx top-1', 'Delta'))",
    "print('-' * 90)",
    "",
    "for q in compare_qs:",
    "    qvec   = embed_text(q)",
    "    b_hits = vs_base.search(qvec, top_k=1)",
    "    c_hits = vs_ctx.search(qvec, top_k=1)",
    "    b_s    = b_hits[0]['score'] if b_hits else 0.0",
    "    c_s    = c_hits[0]['score'] if c_hits else 0.0",
    "    delta  = c_s - b_s",
    "    sign   = '+' if delta >= 0 else ''",
    "    print('{:<55} {:>10.4f}  {:>11.4f}  {:>5}{:.4f}'.format(",
    "        q[:54], b_s, c_s, sign, delta))",
)))

# ── Step 11 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 11 — Inspect Generated Contexts\n\n"
    "Show the LLM-generated context prefix for the first 5 chunks.\n"
    "This makes the enrichment visible and lets you judge quality.\n"
))

cells.append(code(L(
    'print("Generated context summaries for first 5 chunks:")',
    "print('=' * 70)",
    "for i, c in enumerate(contextual_chunks[:5]):",
    "    print(f'--- Chunk {i} ({len(c[\"raw\"])} chars) ---')",
    "    print(f'CONTEXT : {c[\"context\"]}')",
    "    print(f'RAW     : {c[\"raw\"][:120]}...')",
    "    print()",
)))

# ── Step 12 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 12 — Evaluation & Metrics"))

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
    "print('{:<45} {:>8}  {:>8}  {:>9}  {:>9}'.format(",
    "    'Question', 'Base KW', 'Ctx KW', 'Base ms', 'Ctx ms'))",
    "print('-' * 86)",
    "",
    "for case in eval_cases:",
    "    r     = rag_query_both(case['question'], verbose=False)",
    "    b_ans = r['baseline']['answer'].lower()",
    "    c_ans = r['contextual']['answer'].lower()",
    "    kws   = case['keywords']",
    "    b_hit = sum(1 for kw in kws if kw in b_ans)",
    "    c_hit = sum(1 for kw in kws if kw in c_ans)",
    "    n     = len(kws)",
    "    print('{:<45} {:>5}/{} ({:.0f}%)  {:>5}/{} ({:.0f}%)  {:>7.0f}ms  {:>7.0f}ms'.format(",
    "        case['question'][:44],",
    "        b_hit, n, b_hit/n*100,",
    "        c_hit, n, c_hit/n*100,",
    "        r['baseline']['latency_ms'],",
    "        r['contextual']['latency_ms']))",
    "",
    "print()",
    'print(f"Baseline chunks   : {vs_base.count()}")',
    'print(f"Contextual chunks : {vs_ctx.count()}")',
    "print('Note: contextual indexing cost = {} LLM calls (one-time)'.format(len(raw_chunks)))",
)))

# ── Step 13 Summary ────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 13 — Summary\n\n"
    "| Component | Implementation |\n"
    "|-----------|---------------|\n"
    "| PDF | `pypdf.PdfReader` — absolute path in Step 2 |\n"
    "| Chunking | Native `recursive_split()` — 500 chars, 50 overlap |\n"
    "| Context generation | Strands Agent + Claude Sonnet 4.6 — 1 call per chunk |\n"
    "| Context prompt | Document scope (8k chars) + chunk → 1–2 sentence description |\n"
    "| Indexed text | `context_summary + '\\n\\n' + chunk_text` |\n"
    "| Baseline | Raw chunk text indexed — no context prefix |\n"
    "| Vector DB | Qdrant Cloud → OpenSearch → in-memory |\n"
    "| LLM (answers) | AWS Strands Agent + Claude Sonnet 4.6 |\n\n"
    "## Tier 1 complete — pattern summary\n\n"
    "| Notebook | Core idea | Key knob |\n"
    "|----------|-----------|----------|\n"
    "| 01 Simple RAG | Fixed-size chunks | chunk size |\n"
    "| 02 Semantic Chunking | Split at meaning boundaries | breakpoint strategy |\n"
    "| 03 Hierarchical RAG | Search child, return parent | 2-level hierarchy |\n"
    "| 04 Parent-Child RAG | 4-level tree, any ancestor | search + return level |\n"
    "| 05 Sentence Window | Index sentences, expand window | window size W |\n"
    "| **06 Contextual Retrieval** | **LLM-enriched embeddings** | **doc window, chunk size** |\n\n"
    "### Next: **07 — Hybrid Search** (Tier 2 — Retrieval Quality)\n"
))

cells.append(code(L(
    "# vs_base.delete_collection()  # uncomment to clean up",
    "# vs_ctx.delete_collection()",
    "print(f\"Baseline   '{COL_BASELINE}'   in {vs_base._backend} — {vs_base.count()} vectors\")",
    "print(f\"Contextual '{COL_CONTEXTUAL}' in {vs_ctx._backend}  — {vs_ctx.count()} vectors\")",
    'print("\\nTier 1 complete. Give the go-ahead for notebook 07 (Tier 2).")',
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
