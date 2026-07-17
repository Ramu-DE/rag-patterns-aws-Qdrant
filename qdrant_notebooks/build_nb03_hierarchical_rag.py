"""Build 03_Hierarchical_RAG.ipynb — full implementation."""
import json, uuid, ast

PATH = r"C:\Users\Administrator\RAG\qdrant_notebooks\tier1_chunking_indexing\03_Hierarchical_RAG.ipynb"
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
"# 03 — Hierarchical RAG\n"
"\n"
"> **Tier 1 | Chunking & Indexing Foundations**\n"
"\n"
"## What is Hierarchical RAG?\n"
"Standard RAG stores one chunk size and retrieves it verbatim. That creates a tension:\n"
"- **Small chunks** → precise retrieval, but missing surrounding context for the LLM.\n"
"- **Large chunks** → rich context, but lower retrieval precision (noise dilutes the query signal).\n"
"\n"
"**Hierarchical RAG** resolves this by maintaining **two levels** in the same collection:\n"
"\n"
"| Level | Size | Purpose |\n"
"|-------|------|---------|\n"
"| **Child** (leaf) | ~200 chars | What gets *searched* — high precision |\n"
"| **Parent** (context) | ~1000 chars | What gets *sent to the LLM* — rich context |\n"
"\n"
"Every child stores a `parent_id` pointer. At query time:\n"
"1. Embed the question, search **child** vectors for top-K matches.\n"
"2. Resolve each hit's `parent_id` → fetch the full **parent** text.\n"
"3. Deduplicate parents, pass them to the LLM.\n"
"\n"
"## Pipeline\n"
"```\n"
"PDF\n"
" └─ parent chunks (~1000 chars, overlap 200)\n"
"     └─ child chunks (~200 chars, overlap 20)  <- indexed in Qdrant\n"
"         each child.payload.parent_id -> parent text stored in parent_store{}\n"
"\n"
"Query -> embed -> search children -> resolve parents -> LLM\n"
"```\n"
"\n"
"## Why two sizes?\n"
"Research (LlamaIndex, 2023) shows child retrieval + parent expansion consistently\n"
"outperforms single-level retrieval on answer faithfulness while keeping latency low.\n"
))

# ── Step 1 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 1 — Install & Import Dependencies"))

cells.append(code(L(
    "import subprocess, sys",
    'packages = ["boto3", "qdrant-client", "opensearch-py", "requests-aws4auth", "strands-agents", "pypdf"]',
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
    "from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue",
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
    'COLLECTION_NAME = "hierarchical_rag_03"',
    "EMBEDDING_DIM   = 1024",
    "TOP_K           = 5",
    "",
    "# Chunk sizes — the key hierarchy knob",
    "PARENT_CHUNK_SIZE    = 1000   # chars — rich LLM context window",
    "PARENT_CHUNK_OVERLAP = 200",
    "CHILD_CHUNK_SIZE     = 200    # chars — precise retrieval target",
    "CHILD_CHUNK_OVERLAP  = 20",
    "",
    "# PDF — absolute path",
    'PDF_PATH = r"' + PDF + '"',
    "",
    'print(f"Parent chunk : {PARENT_CHUNK_SIZE} chars  (overlap {PARENT_CHUNK_OVERLAP})")',
    'print(f"Child  chunk : {CHILD_CHUNK_SIZE} chars  (overlap {CHILD_CHUNK_OVERLAP})")',
    'print(f"Collection   : {COLLECTION_NAME}")',
    'print(f"PDF          : {PDF_PATH}")',
    'print(f"PDF exists   : {os.path.exists(PDF_PATH)}")',
)))

# ── Step 3 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 3 — Vector Store (Qdrant priority → OpenSearch → in-memory)"))

cells.append(code(L(
    "from typing import List, Dict, Optional",
    "",
    "class VectorStore:",
    "    def __init__(self, collection_name, qdrant_url='', qdrant_api_key='',",
    "                 opensearch_url='', region='us-east-1'):",
    "        self.name    = collection_name",
    "        self.region  = region",
    "        self._backend = None",
    "        if qdrant_url:",
    "            try:",
    "                kw = {'url': qdrant_url}",
    "                if qdrant_api_key:",
    "                    kw['api_key'] = qdrant_api_key",
    "                self._qdrant = QdrantClient(**kw)",
    "                self._qdrant.get_collections()",
    "                self._backend = 'qdrant_cloud'",
    "                print(f'Qdrant Cloud connected: {qdrant_url}')",
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
    "                print(f'OpenSearch connected: {opensearch_url}')",
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
    "                    self.name,",
    "                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE))",
    "                print(f'Created collection \"{self.name}\" (dim={dim})')",
    "            else:",
    "                print(f'Collection \"{self.name}\" already exists')",
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
    "                        }",
    "                    }}})",
    "                print(f'Created OpenSearch index \"{self.name}\"')",
    "",
    "    def upsert(self, docs: List[Dict]) -> int:",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            pts = [",
    "                PointStruct(id=str(uuid.uuid4()), vector=d['embedding'],",
    "                            payload={'text': d['text'], 'metadata': d.get('metadata', {})})",
    "                for d in docs",
    "            ]",
    "            self._qdrant.upsert(collection_name=self.name, points=pts)",
    "            return len(pts)",
    "        elif self._backend == 'opensearch':",
    "            for d in docs:",
    "                self._os.index(index=self.name, body=d)",
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
    'print("VectorStore class defined.")',
)))

# ── Step 4 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 4 — Bedrock Helpers (Embeddings + Strands LLM)"))

cells.append(code(L(
    "from typing import List, Dict",
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
    "    context = '\\n\\n'.join(f'[Context {i+1}]\\n{c}' for i, c in enumerate(context_chunks))",
    "    prompt  = (",
    "        'Use ONLY the context below to answer. '",
    "        \"If the answer is not there say 'Not found in context.'\\n\\n\"",
    "        f'Context:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:'",
    "    )",
    "    return str(Agent(",
    "        model=_model,",
    "        system_prompt='You are a precise Q&A assistant. Answer only from the provided context.'",
    "    )(prompt))",
    "",
    'test_emb = embed_text("hierarchical chunking test")',
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
cells.append(md(
    "## Step 6 — Recursive Text Splitter (no LangChain)\n\n"
    "Pure Python implementation of the recursive character splitter — same algorithm\n"
    "LangChain uses internally. We call it at two different sizes to build the hierarchy.\n"
))

cells.append(code(L(
    "from typing import List",
    "",
    "def recursive_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:",
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
    "            if len(part) <= chunk_size:",
    "                good.append(part)",
    "            elif new_seps:",
    "                good.extend(_split(part, new_seps))",
    "            else:",
    "                for k in range(0, len(part), chunk_size - chunk_overlap):",
    "                    good.append(part[k : k + chunk_size])",
    "",
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
    'print("recursive_split() defined.")',
)))

# ── Step 7 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 7 — Load PDF & Build the Two-Level Hierarchy\n\n"
    "1. Extract full text from each page.\n"
    "2. Split into **parent** chunks (~1000 chars).\n"
    "3. For each parent, split further into **child** chunks (~200 chars).\n"
    "4. Store parent text in an in-memory dict keyed by `parent_id`.\n"
    "5. Index only the **child** embeddings in Qdrant (with `parent_id` in payload).\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "# ── Load PDF ──",
    "reader = pypdf.PdfReader(PDF_PATH)",
    'print(f"PDF    : {PDF_PATH}")',
    'print(f"Pages  : {len(reader.pages)}")',
    "",
    "full_text = ''",
    "page_boundaries: List[Dict] = []",
    "for page_num, page in enumerate(reader.pages):",
    "    t = page.extract_text() or ''",
    "    page_boundaries.append({'start': len(full_text), 'end': len(full_text)+len(t), 'page': page_num+1})",
    "    full_text += t + '\\n\\n'",
    "",
    'print(f"Total chars: {len(full_text):,}")',
    "",
    "# ── Build parent chunks ──",
    "parent_texts = recursive_split(full_text, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP)",
    'print(f"Parent chunks: {len(parent_texts)}")',
    'print(f"Avg parent size: {sum(len(p) for p in parent_texts)/len(parent_texts):.0f} chars")',
    "",
    "# ── Build child chunks per parent ──",
    "parent_store: Dict[str, str] = {}   # parent_id -> parent_text",
    "child_docs: List[Dict] = []",
    "",
    "for p_idx, parent_text in enumerate(parent_texts):",
    "    p_id = f'parent_{p_idx:04d}'",
    "    parent_store[p_id] = parent_text",
    "",
    "    children = recursive_split(parent_text, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP)",
    "    for c_idx, child_text in enumerate(children):",
    "        child_docs.append({",
    "            'text'     : child_text,",
    "            'parent_id': p_id,",
    "            'parent_idx': p_idx,",
    "            'child_idx' : c_idx,",
    "        })",
    "",
    'print(f"Child  chunks: {len(child_docs)}")',
    'print(f"Avg child size: {sum(len(d[\'text\']) for d in child_docs)/len(child_docs):.0f} chars")',
    'print(f"Avg children per parent: {len(child_docs)/len(parent_texts):.1f}")',
    "",
    'print("\\nSample parent[0]:")',
    "print(parent_texts[0][:300], '...')",
    'print("\\nSample child[0] (from parent[0]):")',
    "print(child_docs[0]['text'][:150], '...')",
)))

# ── Step 8 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 8 — Embed Children & Index in Qdrant\n\n"
    "Only **child** vectors go into Qdrant. Each point's payload carries:\n"
    "- `text` — the child chunk text\n"
    "- `metadata.parent_id` — key into `parent_store`\n"
    "- `metadata.parent_idx`, `metadata.child_idx` — for debugging\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    'print(f"Embedding {len(child_docs)} child chunks...")',
    "t0         = time.time()",
    "child_embs = embed_batch([d['text'] for d in child_docs], label='[children]')",
    "",
    "docs_to_index = [",
    "    {",
    "        'text'     : child_docs[i]['text'],",
    "        'embedding': child_embs[i],",
    "        'metadata' : {",
    "            'parent_id' : child_docs[i]['parent_id'],",
    "            'parent_idx': child_docs[i]['parent_idx'],",
    "            'child_idx' : child_docs[i]['child_idx'],",
    "            'source'    : 'climate.pdf',",
    "            'chunk_type': 'child',",
    "        }",
    "    }",
    "    for i in range(len(child_docs))",
    "]",
    "",
    "indexed = vs.upsert(docs_to_index)",
    'print(f"Indexed {indexed} child chunks in {time.time()-t0:.1f}s")',
    'print(f"Total vectors in Qdrant: {vs.count()}")',
)))

# ── Step 9 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 9 — Hierarchical Retriever\n\n"
    "The retriever implements the two-step lookup:\n"
    "1. Search children → get precise matches.\n"
    "2. Resolve each child's `parent_id` → return the full parent text to the LLM.\n"
    "3. Deduplicate parents (multiple children can map to the same parent).\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def hierarchical_retrieve(",
    "    question: str,",
    "    top_k_children: int = TOP_K,",
    "    parent_store: Dict[str, str] = parent_store",
    ") -> Dict:",
    "    # Step 1: search children",
    "    child_hits = vs.search(embed_text(question), top_k=top_k_children)",
    "",
    "    # Step 2: resolve parents (deduplicated)",
    "    seen_parents = set()",
    "    parent_contexts: List[str] = []",
    "    child_details: List[Dict] = []",
    "",
    "    for hit in child_hits:",
    "        p_id = hit['metadata'].get('parent_id', '')",
    "        child_details.append({",
    "            'child_text': hit['text'],",
    "            'parent_id' : p_id,",
    "            'score'     : hit['score'],",
    "        })",
    "        if p_id and p_id not in seen_parents:",
    "            seen_parents.add(p_id)",
    "            parent_contexts.append(parent_store.get(p_id, hit['text']))",
    "",
    "    return {",
    "        'parent_contexts': parent_contexts,",
    "        'child_details'  : child_details,",
    "        'n_children_hit' : len(child_hits),",
    "        'n_parents_used' : len(parent_contexts),",
    "    }",
    "",
    'print("hierarchical_retrieve() defined.")',
)))

# ── Step 10 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 10 — RAG Queries"))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def rag_query(question: str, verbose: bool = True) -> Dict:",
    "    t0      = time.time()",
    "    result  = hierarchical_retrieve(question)",
    "    answer  = generate_answer(question, result['parent_contexts'])",
    "    latency = (time.time() - t0) * 1000",
    "",
    "    if verbose:",
    '        print(f"\\nQ: {question}")',
    '        print(f"A: {answer}")',
    '        print(f"  Children retrieved: {result[\'n_children_hit\']}")',
    '        print(f"  Unique parents used: {result[\'n_parents_used\']}")',
    "        for i, d in enumerate(result['child_details'][:3], 1):",
    "            print(f\"  [child {i}] pid={d['parent_id']}  score={d['score']:.4f}  {d['child_text'][:70]}...\")",
    '        print(f"  Latency: {latency:.0f}ms")',
    '        print("-" * 70)',
    "",
    "    return {",
    "        'question'       : question,",
    "        'answer'         : answer,",
    "        'latency_ms'     : latency,",
    "        'n_children_hit' : result['n_children_hit'],",
    "        'n_parents_used' : result['n_parents_used'],",
    "        'child_details'  : result['child_details'],",
    "    }",
    "",
    "test_questions = [",
    '    "What is weather forecasting and why is it important?",',
    '    "What are the main methods used in weather analysis?",',
    '    "How does climatology differ from meteorology?",',
    '    "What factors influence weather patterns and climate?",',
    "]",
    "results_log = [rag_query(q) for q in test_questions]",
)))

# ── Step 11 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 11 — Hierarchical vs. Flat Comparison\n\n"
    "Run the same questions against:\n"
    "- **Hierarchical** — search child chunks, return parent context.\n"
    "- **Flat large** — search parent-sized chunks directly.\n"
    "- **Flat small** — search child-sized chunks and return them verbatim.\n"
    "\n"
    "Compare answer quality and context token count.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "# Build two flat in-memory stores for comparison",
    'print("Building flat comparison stores...")',
    "",
    "# Flat large (parent-sized chunks)",
    "vs_flat_large = VectorStore('flat_large_compare')",
    "vs_flat_large.create_collection(dim=EMBEDDING_DIM)",
    "parent_embs = embed_batch(parent_texts, label='[flat-large]')",
    "vs_flat_large.upsert([",
    "    {'text': parent_texts[i], 'embedding': parent_embs[i],",
    "     'metadata': {'idx': i, 'chunk_type': 'parent'}}",
    "    for i in range(len(parent_texts))",
    "])",
    "",
    "# Flat small (child-sized chunks)",
    "vs_flat_small = VectorStore('flat_small_compare')",
    "vs_flat_small.create_collection(dim=EMBEDDING_DIM)",
    "vs_flat_small.upsert([",
    "    {'text': child_docs[i]['text'], 'embedding': child_embs[i],",
    "     'metadata': {'idx': i, 'chunk_type': 'child'}}",
    "    for i in range(len(child_docs))",
    "])",
    "",
    'print("Stores ready. Running comparison...")',
    "compare_q = 'What are the main methods used in weather analysis?'",
    "qvec = embed_text(compare_q)",
    "",
    "# Hierarchical",
    "t0  = time.time()",
    "hr  = hierarchical_retrieve(compare_q)",
    "h_time = (time.time() - t0)*1000",
    "h_ctx  = ' '.join(hr['parent_contexts'])",
    "",
    "# Flat large",
    "t0  = time.time()",
    "fl_hits = vs_flat_large.search(qvec, top_k=TOP_K)",
    "fl_time = (time.time() - t0)*1000",
    "fl_ctx  = ' '.join(h['text'] for h in fl_hits)",
    "",
    "# Flat small",
    "t0  = time.time()",
    "fs_hits = vs_flat_small.search(qvec, top_k=TOP_K)",
    "fs_time = (time.time() - t0)*1000",
    "fs_ctx  = ' '.join(h['text'] for h in fs_hits)",
    "",
    "print(f'\\nQ: {compare_q}')",
    "print('{:<20} {:>10}  {:>10}  {:>7}'.format('Strategy', 'Ctx chars', 'Latency', 'Chunks'))",
    "print('-' * 56)",
    "print(f\"{'Hierarchical':<20} {len(h_ctx):>10,}  {h_time:>9.0f}ms  {hr['n_children_hit']:>5} children -> {hr['n_parents_used']} parents\")",
    "print(f\"{'Flat large':<20} {len(fl_ctx):>10,}  {fl_time:>9.0f}ms  {len(fl_hits):>7}\")",
    "print(f\"{'Flat small':<20} {len(fs_ctx):>10,}  {fs_time:>9.0f}ms  {len(fs_hits):>7}\")",
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
    "print('{:<55} {:>7} {:>9} {:>7} {:>7}'.format(",
    "    'Question', 'KW Hit', 'Latency', 'C-hits', 'P-used'))",
    "print('-' * 90)",
    "",
    "eval_log = []",
    "for case in eval_cases:",
    "    r    = rag_query(case['question'], verbose=False)",
    "    low  = r['answer'].lower()",
    "    hits = sum(1 for kw in case['keywords'] if kw in low)",
    "    n    = len(case['keywords'])",
    "    eval_log.append(r)",
    "    print('{:<55} {}/{} ({:.0f}%) {:>8.0f}ms {:>6} {:>7}'.format(",
    "        case['question'][:54], hits, n, hits/n*100,",
    "        r['latency_ms'], r['n_children_hit'], r['n_parents_used']))",
    "",
    "print()",
    "print('Avg latency  : {:.0f}ms'.format(sum(r['latency_ms'] for r in eval_log)/len(eval_log)))",
    "print('Parent chunks: {}  avg size: {:.0f} chars'.format(",
    "    len(parent_texts), sum(len(p) for p in parent_texts)/len(parent_texts)))",
    "print('Child  chunks: {}  avg size: {:.0f} chars'.format(",
    "    len(child_docs), sum(len(d['text']) for d in child_docs)/len(child_docs)))",
)))

# ── Step 13 — Visualise hierarchy ─────────────────────────────────────────────
cells.append(md(
    "## Step 13 — Inspect the Hierarchy\n\n"
    "Peek at one parent and all its children to verify the parent→child relationship.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "sample_parent_idx = 2",
    "p_id = f'parent_{sample_parent_idx:04d}'",
    "p_text = parent_store[p_id]",
    "p_children = [d for d in child_docs if d['parent_id'] == p_id]",
    "",
    "print(f'=== Parent [{p_id}] ===')",
    "print(f'Length: {len(p_text)} chars')",
    "print(p_text[:400], '...')",
    "print()",
    "print(f'  -> {len(p_children)} children:')",
    "for i, c in enumerate(p_children):",
    "    print(f'     child[{i}] ({len(c[\"text\"])} chars): {c[\"text\"][:80]}...')",
)))

# ── Step 14 Summary ────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 14 — Summary\n\n"
    "| Component | Implementation |\n"
    "|-----------|---------------|\n"
    "| PDF | `pypdf.PdfReader` — absolute path hardcoded in Step 2 |\n"
    "| Splitter | Native `recursive_split()` — no LangChain |\n"
    "| Parent chunks | ~1000 chars — full LLM context, stored in-memory dict |\n"
    "| Child chunks | ~200 chars — embedded and indexed in Qdrant |\n"
    "| Retrieval | Search children → resolve parent_id → send parent to LLM |\n"
    "| Deduplication | Multiple child hits map to same parent — deduplicated |\n"
    "| Vector DB | Qdrant Cloud → OpenSearch → in-memory |\n"
    "| LLM | AWS Strands Agent + Claude Sonnet 4.6 |\n\n"
    "## Key insight\n"
    "Precision comes from **child** retrieval; context richness comes from **parent** expansion.\n"
    "This two-level design consistently beats single-level retrieval on faithfulness metrics.\n\n"
    "### Next: **04 — Parent-Child RAG** (extends to 4 levels)\n"
))

cells.append(code(L(
    "# vs.delete_collection()  # uncomment to clean up",
    "print(f\"Collection '{COLLECTION_NAME}' kept in {vs._backend} — {vs.count()} vectors\")",
    'print("\\nDone. Give the go-ahead for notebook 04.")',
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
