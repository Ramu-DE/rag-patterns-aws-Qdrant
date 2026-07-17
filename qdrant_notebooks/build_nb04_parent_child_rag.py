"""Build 04_Parent_Child_RAG.ipynb — full implementation."""
import json, uuid, ast

PATH = r"C:\Users\Administrator\RAG\qdrant_notebooks\tier1_chunking_indexing\04_Parent_Child_RAG.ipynb"
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
"# 04 — Parent-Child RAG\n"
"\n"
"> **Tier 1 | Chunking & Indexing Foundations**\n"
"\n"
"## What is Parent-Child RAG?\n"
"Notebook 03 (Hierarchical RAG) used **two levels** — child chunks for retrieval,\n"
"parent chunks for LLM context. Parent-Child RAG extends this to **four levels**:\n"
"\n"
"```\n"
"Level 0 — Document   (whole PDF, ~full text)\n"
"Level 1 — Section    (~2000 chars, major topic blocks)\n"
"Level 2 — Paragraph  (~500 chars, idea-level units)      <- indexed in Qdrant\n"
"Level 3 — Sentence   (~120 chars, atomic statements)     <- indexed in Qdrant\n"
"```\n"
"\n"
"Both **paragraph** and **sentence** chunks are indexed.\n"
"At query time you choose which level to *search* and which ancestor level to *return*.\n"
"\n"
"## Why go beyond two levels?\n"
"| Scenario | Best search level | Best return level |\n"
"|----------|------------------|------------------|\n"
"| Precise factual lookup | Sentence (L3) | Paragraph (L2) or Section (L1) |\n"
"| Broad topic summary | Paragraph (L2) | Section (L1) |\n"
"| Deep dive / long answer | Paragraph (L2) | Document (L0) |\n"
"\n"
"The four-level tree gives you flexible retrieval granularity with a single index pass.\n"
"\n"
"## Pipeline\n"
"```\n"
"PDF\n"
" └─ sections  (L1, ~2000 chars)  stored in node_store{}\n"
"     └─ paragraphs (L2, ~500 chars)  stored in node_store{} + indexed in Qdrant\n"
"         └─ sentences  (L3, ~120 chars)  stored in node_store{} + indexed in Qdrant\n"
"\n"
"Query -> embed -> search L3 sentences\n"
"      -> resolve parent_id -> L2 paragraph  (default)\n"
"      -> optionally climb to L1 section\n"
"      -> deduplicate -> LLM\n"
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
    "# Collection",
    'COLLECTION_NAME = "parent_child_rag_04"',
    "EMBEDDING_DIM   = 1024",
    "TOP_K           = 5",
    "",
    "# Four-level chunk sizes",
    "L1_SECTION_SIZE    = 2000   # chars — broad topic blocks",
    "L1_SECTION_OVERLAP = 200",
    "L2_PARA_SIZE       = 500    # chars — paragraph units (indexed)",
    "L2_PARA_OVERLAP    = 50",
    "L3_SENT_SIZE       = 120    # chars — sentence units (indexed)",
    "L3_SENT_OVERLAP    = 10",
    "",
    "# Retrieval config",
    "SEARCH_LEVEL  = 'L3'   # which level to search: 'L2' or 'L3'",
    "RETURN_LEVEL  = 'L2'   # which ancestor to return to LLM: 'L1', 'L2'",
    "",
    "# PDF",
    'PDF_PATH = r"' + PDF + '"',
    "",
    'print(f"L1 Section  : {L1_SECTION_SIZE} chars")',
    'print(f"L2 Paragraph: {L2_PARA_SIZE} chars")',
    'print(f"L3 Sentence : {L3_SENT_SIZE} chars")',
    'print(f"Search level: {SEARCH_LEVEL}  |  Return level: {RETURN_LEVEL}")',
    'print(f"PDF         : {PDF_PATH}")',
    'print(f"PDF exists  : {os.path.exists(PDF_PATH)}")',
)))

# ── Step 3 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 3 — Vector Store"))

cells.append(code(L(
    "from typing import List, Dict",
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
    "                    self.name,",
    "                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE))",
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
    "    context = '\\n\\n'.join(f'[Context {i+1}]\\n{c}' for i, c in enumerate(context_chunks))",
    "    prompt  = (",
    "        'Use ONLY the context below to answer. '",
    "        \"If the answer is not there, say 'Not found in context.'\\n\\n\"",
    "        f'Context:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:'",
    "    )",
    "    return str(Agent(",
    "        model=_model,",
    "        system_prompt='You are a precise Q&A assistant. Answer only from the provided context.'",
    "    )(prompt))",
    "",
    'test_emb = embed_text("parent child hierarchy test")',
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
cells.append(md("## Step 6 — Recursive Splitter"))

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
    "## Step 7 — Build Four-Level Hierarchy\n\n"
    "```\n"
    "L0 document\n"
    "  └─ L1 sections   (~2000 chars) — stored in node_store, NOT indexed\n"
    "      └─ L2 paragraphs (~500 chars) — stored + indexed in Qdrant\n"
    "          └─ L3 sentences (~120 chars) — stored + indexed in Qdrant\n"
    "```\n\n"
    "Every node carries:\n"
    "- `node_id` — unique key in `node_store`\n"
    "- `parent_id` — pointer one level up\n"
    "- `level` — 0/1/2/3\n"
    "- `text` — the chunk text\n"
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
    "for page in reader.pages:",
    "    full_text += (page.extract_text() or '') + '\\n\\n'",
    'print(f"Chars  : {len(full_text):,}")',
    "",
    "# ── Build hierarchy ──",
    "node_store: Dict[str, Dict] = {}  # node_id -> node dict",
    "to_index_l2: List[Dict] = []      # paragraph nodes for Qdrant",
    "to_index_l3: List[Dict] = []      # sentence nodes for Qdrant",
    "",
    "# L0 — document root",
    "doc_id = 'doc_0000'",
    "node_store[doc_id] = {'node_id': doc_id, 'level': 0, 'parent_id': None, 'text': full_text}",
    "",
    "# L1 — sections",
    "sections = recursive_split(full_text, L1_SECTION_SIZE, L1_SECTION_OVERLAP)",
    "for s_idx, s_text in enumerate(sections):",
    "    s_id = f's_{s_idx:04d}'",
    "    node_store[s_id] = {'node_id': s_id, 'level': 1, 'parent_id': doc_id, 'text': s_text}",
    "",
    "    # L2 — paragraphs",
    "    paras = recursive_split(s_text, L2_PARA_SIZE, L2_PARA_OVERLAP)",
    "    for p_idx, p_text in enumerate(paras):",
    "        p_id = f'p_{s_idx:04d}_{p_idx:04d}'",
    "        node_store[p_id] = {'node_id': p_id, 'level': 2, 'parent_id': s_id, 'text': p_text}",
    "        to_index_l2.append(node_store[p_id])",
    "",
    "        # L3 — sentences",
    "        sents = recursive_split(p_text, L3_SENT_SIZE, L3_SENT_OVERLAP)",
    "        for t_idx, t_text in enumerate(sents):",
    "            t_id = f't_{s_idx:04d}_{p_idx:04d}_{t_idx:04d}'",
    "            node_store[t_id] = {'node_id': t_id, 'level': 3, 'parent_id': p_id, 'text': t_text}",
    "            to_index_l3.append(node_store[t_id])",
    "",
    'print(f"L0 document  : 1")',
    'print(f"L1 sections  : {len(sections)}")',
    'print(f"L2 paragraphs: {len(to_index_l2)}")',
    'print(f"L3 sentences : {len(to_index_l3)}")',
    'print(f"Total nodes  : {len(node_store)}")',
    "",
    "# Quick size check",
    "avg_l1 = sum(len(n['text']) for n in node_store.values() if n['level']==1) / len(sections)",
    "avg_l2 = sum(len(n['text']) for n in to_index_l2) / len(to_index_l2)",
    "avg_l3 = sum(len(n['text']) for n in to_index_l3) / len(to_index_l3)",
    'print(f"Avg L1 size  : {avg_l1:.0f} chars")',
    'print(f"Avg L2 size  : {avg_l2:.0f} chars")',
    'print(f"Avg L3 size  : {avg_l3:.0f} chars")',
)))

# ── Step 8 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 8 — Embed & Index L2 + L3 Nodes\n\n"
    "Both paragraph (L2) and sentence (L3) vectors are stored in the **same** Qdrant collection.\n"
    "The `level` field in the payload lets us tell them apart at query time.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "# Embed L2 paragraphs",
    'print(f"Embedding {len(to_index_l2)} L2 paragraphs...")',
    "t0   = time.time()",
    "l2_embs = embed_batch([n['text'] for n in to_index_l2], label='[L2]')",
    'print(f"L2 done in {time.time()-t0:.1f}s")',
    "",
    "# Embed L3 sentences",
    'print(f"Embedding {len(to_index_l3)} L3 sentences...")',
    "t0   = time.time()",
    "l3_embs = embed_batch([n['text'] for n in to_index_l3], label='[L3]')",
    'print(f"L3 done in {time.time()-t0:.1f}s")',
    "",
    "# Build docs list",
    "all_docs: List[Dict] = []",
    "for i, node in enumerate(to_index_l2):",
    "    all_docs.append({",
    "        'text'     : node['text'],",
    "        'embedding': l2_embs[i],",
    "        'metadata' : {",
    "            'node_id'  : node['node_id'],",
    "            'parent_id': node['parent_id'],",
    "            'level'    : 2,",
    "            'source'   : 'climate.pdf',",
    "        }",
    "    })",
    "for i, node in enumerate(to_index_l3):",
    "    all_docs.append({",
    "        'text'     : node['text'],",
    "        'embedding': l3_embs[i],",
    "        'metadata' : {",
    "            'node_id'  : node['node_id'],",
    "            'parent_id': node['parent_id'],",
    "            'level'    : 3,",
    "            'source'   : 'climate.pdf',",
    "        }",
    "    })",
    "",
    "indexed = vs.upsert(all_docs)",
    'print(f"Indexed {indexed} vectors ({len(to_index_l2)} L2 + {len(to_index_l3)} L3)")',
    'print(f"Total in Qdrant: {vs.count()}")',
)))

# ── Step 9 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 9 — Parent-Child Retriever\n\n"
    "`retrieve()` accepts two parameters:\n"
    "- `search_level` — `'L2'` or `'L3'`: which vector level to query\n"
    "- `return_level` — `'L2'`, `'L1'`, or `'L0'`: which ancestor text to send to the LLM\n\n"
    "The `climb()` helper walks `parent_id` pointers up the tree until the target level.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def climb(node_id: str, target_level: int) -> str:",
    "    \"\"\"Walk parent_id pointers until we reach target_level; return that node's text.\"\"\"",
    "    node = node_store.get(node_id)",
    "    if node is None:",
    "        return ''",
    "    if node['level'] <= target_level:",
    "        return node['text']",
    "    return climb(node['parent_id'], target_level)",
    "",
    "def retrieve(",
    "    question: str,",
    "    search_level: str = SEARCH_LEVEL,",
    "    return_level: str = RETURN_LEVEL,",
    "    top_k: int = TOP_K",
    ") -> Dict:",
    "    level_int = {'L0': 0, 'L1': 1, 'L2': 2, 'L3': 3}",
    "    s_lvl = level_int[search_level]",
    "    r_lvl = level_int[return_level]",
    "",
    "    # Get all hits, filter by level",
    "    hits = vs.search(embed_text(question), top_k=top_k * 3)",
    "    hits = [h for h in hits if h['metadata'].get('level') == s_lvl][:top_k]",
    "",
    "    # Climb to return level, deduplicate",
    "    seen, contexts, details = set(), [], []",
    "    for h in hits:",
    "        nid   = h['metadata'].get('node_id', '')",
    "        ctx   = climb(nid, r_lvl)",
    "        # Use a prefix of ctx as dedup key (same parent may appear via different children)",
    "        key   = ctx[:80]",
    "        details.append({'node_id': nid, 'score': h['score'], 'text': h['text']})",
    "        if key not in seen:",
    "            seen.add(key)",
    "            contexts.append(ctx)",
    "",
    "    return {",
    "        'contexts'     : contexts,",
    "        'details'      : details,",
    "        'search_level' : search_level,",
    "        'return_level' : return_level,",
    "        'n_hits'       : len(hits),",
    "        'n_contexts'   : len(contexts),",
    "    }",
    "",
    'print("retrieve() + climb() defined.")',
)))

# ── Step 10 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 10 — RAG Queries"))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def rag_query(question: str, search_level: str = SEARCH_LEVEL,",
    "              return_level: str = RETURN_LEVEL, verbose: bool = True) -> Dict:",
    "    t0     = time.time()",
    "    result = retrieve(question, search_level=search_level, return_level=return_level)",
    "    answer = generate_answer(question, result['contexts'])",
    "    latency = (time.time() - t0) * 1000",
    "",
    "    if verbose:",
    '        print(f"\\nQ: {question}")',
    '        print(f"A: {answer}")',
    "        print(f\"  Search {result['search_level']} -> return {result['return_level']}  |  \",",
    "              f\"{result['n_hits']} hits  {result['n_contexts']} unique contexts\")",
    "        for i, d in enumerate(result['details'][:3], 1):",
    "            print(f\"  [{i}] {d['node_id']}  score={d['score']:.4f}  {d['text'][:70]}...\")",
    '        print(f"  Latency: {latency:.0f}ms")',
    '        print("-" * 70)',
    "",
    "    return {'question': question, 'answer': answer, 'latency_ms': latency,",
    "            'n_hits': result['n_hits'], 'n_contexts': result['n_contexts']}",
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
    "## Step 11 — Compare Retrieval Configurations\n\n"
    "Test all combinations of search level and return level on the same question.\n"
    "This shows how the hierarchy knob trades retrieval precision against context richness.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "compare_q   = 'What are the main methods used in weather analysis?'",
    "configs = [",
    "    ('L3', 'L3'),   # search sentence, return sentence  (like flat small)",
    "    ('L3', 'L2'),   # search sentence, return paragraph (default)",
    "    ('L3', 'L1'),   # search sentence, return section   (richest context)",
    "    ('L2', 'L2'),   # search paragraph, return paragraph",
    "    ('L2', 'L1'),   # search paragraph, return section",
    "]",
    "",
    'print(f"Q: {compare_q}")',
    "print()",
    "print('{:<12} {:<12} {:>10}  {:>10}  {:>10}'.format(",
    "    'Search', 'Return', 'Ctx chars', 'N-ctx', 'Latency'))",
    "print('-' * 60)",
    "",
    "for s_lvl, r_lvl in configs:",
    "    t0  = time.time()",
    "    res = retrieve(compare_q, search_level=s_lvl, return_level=r_lvl)",
    "    lat = (time.time() - t0) * 1000",
    "    total_chars = sum(len(c) for c in res['contexts'])",
    "    print('{:<12} {:<12} {:>10,}  {:>10}  {:>9.0f}ms'.format(",
    "        s_lvl, r_lvl, total_chars, res['n_contexts'], lat))",
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
    "print('{:<55} {:>7} {:>10} {:>7}'.format('Question', 'KW Hit', 'Latency', 'N-ctx'))",
    "print('-' * 82)",
    "eval_log = []",
    "for case in eval_cases:",
    "    r    = rag_query(case['question'], verbose=False)",
    "    low  = r['answer'].lower()",
    "    hits = sum(1 for kw in case['keywords'] if kw in low)",
    "    n    = len(case['keywords'])",
    "    eval_log.append(r)",
    "    print('{:<55} {}/{} ({:.0f}%) {:>8.0f}ms {:>6}'.format(",
    "        case['question'][:54], hits, n, hits/n*100, r['latency_ms'], r['n_contexts']))",
    "",
    "print()",
    "print('Avg latency: {:.0f}ms'.format(sum(r['latency_ms'] for r in eval_log)/len(eval_log)))",
    "print('Nodes in store: L0={}, L1={}, L2={}, L3={}'.format(",
    "    sum(1 for n in node_store.values() if n['level']==0),",
    "    sum(1 for n in node_store.values() if n['level']==1),",
    "    sum(1 for n in node_store.values() if n['level']==2),",
    "    sum(1 for n in node_store.values() if n['level']==3),",
    "))",
)))

# ── Step 13 — Tree inspect ─────────────────────────────────────────────────────
cells.append(md(
    "## Step 13 — Inspect the Tree\n\n"
    "Pick one L3 sentence and show its full ancestry: sentence → paragraph → section.\n"
))

cells.append(code(L(
    "sample_l3 = to_index_l3[5]",
    "nid = sample_l3['node_id']",
    "",
    "print('=== Ancestry of', nid, '===')",
    "node = node_store[nid]",
    "while node is not None:",
    "    snip = node['text'][:180].replace('\\n', ' ')",
    "    print(f\"  L{node['level']} [{node['node_id']}]  ({len(node['text'])} chars)\")",
    "    print(f\"    {snip}...\")",
    "    print()",
    "    node = node_store.get(node['parent_id']) if node['parent_id'] else None",
)))

# ── Step 14 Summary ────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 14 — Summary\n\n"
    "| Component | Implementation |\n"
    "|-----------|---------------|\n"
    "| PDF | `pypdf.PdfReader` — absolute path in Step 2 |\n"
    "| Splitter | Native `recursive_split()` — no LangChain |\n"
    "| L0 Document | Whole PDF text — in `node_store` only |\n"
    "| L1 Sections | ~2000 chars — in `node_store` only |\n"
    "| L2 Paragraphs | ~500 chars — `node_store` + Qdrant (level=2) |\n"
    "| L3 Sentences | ~120 chars — `node_store` + Qdrant (level=3) |\n"
    "| Retrieval | Search L3 → climb to L2 parent (configurable) |\n"
    "| Deduplication | `climb()` deduplicates by returned context prefix |\n"
    "| Vector DB | Qdrant Cloud → OpenSearch → in-memory |\n"
    "| LLM | AWS Strands Agent + Claude Sonnet 4.6 |\n\n"
    "## Key difference from notebook 03\n"
    "| | Hierarchical RAG (03) | Parent-Child RAG (04) |\n"
    "|--|---|---|\n"
    "| Levels | 2 (child + parent) | 4 (sentence/para/section/doc) |\n"
    "| Flexibility | Fixed: search child → return parent | Configurable: any search level → any ancestor |\n"
    "| Use case | General purpose | Fine-grained control over precision vs. context |\n\n"
    "### Next: **05 — Sentence Window RAG**\n"
))

cells.append(code(L(
    "# vs.delete_collection()  # uncomment to clean up",
    "print(f\"Collection '{COLLECTION_NAME}' kept in {vs._backend} — {vs.count()} vectors\")",
    'print("\\nDone. Give the go-ahead for notebook 05.")',
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
