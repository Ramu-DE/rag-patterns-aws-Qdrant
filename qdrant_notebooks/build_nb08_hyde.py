"""Build 08_HyDE.ipynb — Hypothetical Document Embeddings, full implementation."""
import json, uuid, ast

PATH = r"C:\Users\Administrator\RAG\qdrant_notebooks\tier2_retrieval_quality\08_HyDE.ipynb"
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
"# 08 — HyDE (Hypothetical Document Embeddings)\n"
"\n"
"> **Tier 2 | Retrieval Quality**\n"
"\n"
"## The Problem HyDE Solves\n"
"\n"
"There is a fundamental **vocabulary gap** between queries and documents:\n"
"\n"
"- A user asks: *\"Why does it get colder at higher altitudes?\"*\n"
"- The document says: *\"The temperature lapse rate in the troposphere averages 6.5°C/km.\"*\n"
"\n"
"These mean the same thing but share almost no words.\n"
"When you embed the query directly, its vector points toward *other questions*, not answers.\n"
"\n"
"## HyDE's Solution\n"
"\n"
"**Hypothetical Document Embeddings** (Gao et al., 2022) bridges the gap:\n"
"\n"
"1. Ask the LLM to **hallucinate** a plausible answer to the query\n"
"2. Embed the **hypothetical answer** (not the query)\n"
"3. Use that embedding for retrieval — it now lives in *answer-space*, not *question-space*\n"
"\n"
"```\n"
"Standard:  embed(query)       → search → retrieve docs\n"
"HyDE:      LLM(query)         → hypothetical_doc\n"
"           embed(hyp_doc)     → search → retrieve real docs\n"
"```\n"
"\n"
"The hypothetical document doesn't need to be factually correct —\n"
"it just needs to be **stylistically and semantically similar** to a real answer.\n"
"\n"
"## Three HyDE variants implemented\n"
"\n"
"| Variant | Description |\n"
"|---------|-------------|\n"
"| **Single HyDE** | One hypothetical doc per query |\n"
"| **Multi HyDE** | N hypothetical docs, average their embeddings |\n"
"| **HyDE + Hybrid** | Combine HyDE vector with BM25 via RRF |\n"
))

# ── Flow diagram ──────────────────────────────────────────────────────────────
cells.append(md(
"## Flow Diagram\n"
"\n"
"```mermaid\n"
"flowchart TD\n"
"    subgraph INDEX [\"🔵  INDEXING  (run once)\"]\n"
"        PDF([\"📄 climate.pdf\"])\n"
"        PDF --> SPLIT[\"Fixed-size chunks\\n~500 chars\"]\n"
"        SPLIT --> EMB[\"Embed chunks\\nTitan V2\"]\n"
"        EMB --> QDRANT[(\"Qdrant collection\")]\n"
"    end\n"
"\n"
"    subgraph STANDARD [\"⚪  STANDARD RETRIEVAL\"]\n"
"        Q0([\"❓ Query\"])\n"
"        Q0 --> QEMB0[\"embed(query)\"]\n"
"        QEMB0 --> VSEARCH0[\"Vector search\"]\n"
"        QDRANT --> VSEARCH0\n"
"        VSEARCH0 --> HITS0([\"Results\"])\n"
"    end\n"
"\n"
"    subgraph HYDE [\"🔵  HyDE RETRIEVAL\"]\n"
"        Q1([\"❓ Same query\"])\n"
"        Q1 --> LLM1[\"LLM: generate\\nhypothetical answer\"]\n"
"        LLM1 --> HYPDOC[\"Hypothetical doc\\n(may be factually wrong,\\nbut answer-shaped)\"]\n"
"        HYPDOC --> HYPEMB[\"embed(hyp_doc)\\nTitan V2\"]\n"
"        HYPEMB --> VSEARCH1[\"Vector search\\nin answer-space\"]\n"
"        QDRANT --> VSEARCH1\n"
"        VSEARCH1 --> HITS1([\"Results\"])\n"
"    end\n"
"\n"
"    subgraph MULTI [\"🟣  MULTI-HyDE\"]\n"
"        Q2([\"❓ Same query\"])\n"
"        Q2 --> LLM2A[\"Hyp doc 1\"]\n"
"        Q2 --> LLM2B[\"Hyp doc 2\"]\n"
"        Q2 --> LLM2C[\"Hyp doc N\"]\n"
"        LLM2A --> AVG[\"Average embeddings\"]\n"
"        LLM2B --> AVG\n"
"        LLM2C --> AVG\n"
"        AVG --> VSEARCH2[\"Vector search\"]\n"
"        QDRANT --> VSEARCH2\n"
"        VSEARCH2 --> HITS2([\"Results\"])\n"
"    end\n"
"\n"
"    subgraph GEN [\"🟠  GENERATION\"]\n"
"        HITS1 --> LLM3[\"Strands Agent\\nClaude Sonnet 4.6\"]\n"
"        LLM3 --> ANS([\"✅ Answer\"])\n"
"    end\n"
"\n"
"    style INDEX    fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f\n"
"    style STANDARD fill:#f1f5f9,stroke:#94a3b8,color:#334155\n"
"    style HYDE     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f\n"
"    style MULTI    fill:#faf5ff,stroke:#a855f7,color:#3b0764\n"
"    style GEN      fill:#ffedd5,stroke:#f97316,color:#7c2d12\n"
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
    "from typing import List, Dict",
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
    'COLLECTION_NAME = "hyde_rag_08"',
    "EMBEDDING_DIM   = 1024",
    "TOP_K           = 5",
    "",
    "# Chunking",
    "CHUNK_SIZE    = 500",
    "CHUNK_OVERLAP = 50",
    "",
    "# HyDE settings",
    "HYDE_N_DOCS    = 3    # number of hypothetical docs for Multi-HyDE",
    "HYDE_MAX_WORDS = 150  # target length of each hypothetical answer",
    "",
    "# PDF",
    'PDF_PATH = r"' + PDF + '"',
    "",
    'print(f"Collection   : {COLLECTION_NAME}")',
    'print(f"HyDE N docs  : {HYDE_N_DOCS}  (Multi-HyDE variant)")',
    'print(f"HyDE max words: {HYDE_MAX_WORDS}")',
    'print(f"PDF          : {PDF_PATH}")',
    'print(f"PDF exists   : {os.path.exists(PDF_PATH)}")',
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
    "def avg_embeddings(embeddings: List[List[float]]) -> List[float]:",
    "    n   = len(embeddings)",
    "    dim = len(embeddings[0])",
    "    avg = [sum(embeddings[j][i] for j in range(n)) / n for i in range(dim)]",
    "    # Re-normalise to unit length (Titan embeddings are normalised but averaging breaks that)",
    "    norm = math.sqrt(sum(x * x for x in avg))",
    "    return [x / norm for x in avg] if norm > 0 else avg",
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
    'test_emb = embed_text("hypothetical document embedding test")',
    'print(f"Embedding OK — dim={len(test_emb)}")',
    'print("avg_embeddings() defined.")',
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
    "## Step 7 — Hypothetical Document Generator\n\n"
    "The prompt asks Claude to write a **short passage** that would appear in\n"
    "a document answering the query — as if it were extracted from a real textbook.\n\n"
    "Key prompt design choices:\n"
    "- Ask for a *passage*, not an *answer* — the output should resemble a document chunk\n"
    "- Limit length to ~150 words — longer docs dilute the embedding signal\n"
    "- Tell Claude it **may be imprecise** — we want answer-space proximity, not factual accuracy\n"
))

cells.append(code(L(
    "from typing import List",
    "",
    "HYDE_PROMPT = (",
    "    'Write a short passage ({max_words} words max) that could appear in a '",
    "    'climate science or meteorology textbook as an answer to this question.\\n'",
    "    'Be specific and use technical language. '",
    "    'You may be imprecise — this is used only for document retrieval, not as a final answer.\\n\\n'",
    "    'Question: {question}\\n\\n'",
    "    'Passage:'",
    ")",
    "",
    "def generate_hypothetical_doc(question: str, max_words: int = HYDE_MAX_WORDS) -> str:",
    "    prompt = HYDE_PROMPT.format(question=question, max_words=max_words)",
    "    return str(Agent(",
    "        model=_model,",
    "        system_prompt='You are a climate science textbook author. Write concise, factual passages.'",
    "    )(prompt)).strip()",
    "",
    "def generate_hypothetical_docs(question: str, n: int = HYDE_N_DOCS) -> List[str]:",
    "    docs = []",
    "    for i in range(n):",
    "        docs.append(generate_hypothetical_doc(question))",
    "        time.sleep(0.1)",
    "    return docs",
    "",
    "# Smoke test",
    'test_q = "What causes temperature to decrease with altitude?"',
    'print(f"Generating hypothetical doc for: {test_q}")',
    "hyp = generate_hypothetical_doc(test_q)",
    'print(f"\\nHypothetical doc ({len(hyp.split())} words):")',
    "print(hyp)",
)))

# ── Step 8 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 8 — Three Retrieval Strategies\n\n"
    "### Strategy A — Standard (embed query directly)\n"
    "### Strategy B — Single HyDE (embed one hypothetical doc)\n"
    "### Strategy C — Multi-HyDE (embed N docs, average vectors)\n\n"
    "The averaged vector in Multi-HyDE is more **robust** than a single generation —\n"
    "it cancels out random hallucination noise and captures the centroid of answer-space.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def retrieve_standard(question: str, top_k: int = TOP_K) -> Dict:",
    "    qvec = embed_text(question)",
    "    hits = vs.search(qvec, top_k=top_k)",
    "    return {'hits': hits, 'hyp_docs': [], 'strategy': 'standard'}",
    "",
    "def retrieve_hyde_single(question: str, top_k: int = TOP_K) -> Dict:",
    "    hyp_doc = generate_hypothetical_doc(question)",
    "    hyp_vec = embed_text(hyp_doc)",
    "    hits    = vs.search(hyp_vec, top_k=top_k)",
    "    return {'hits': hits, 'hyp_docs': [hyp_doc], 'strategy': 'hyde_single'}",
    "",
    "def retrieve_hyde_multi(question: str, n: int = HYDE_N_DOCS, top_k: int = TOP_K) -> Dict:",
    "    hyp_docs = generate_hypothetical_docs(question, n=n)",
    "    hyp_vecs = [embed_text(d) for d in hyp_docs]",
    "    avg_vec  = avg_embeddings(hyp_vecs)",
    "    hits     = vs.search(avg_vec, top_k=top_k)",
    "    return {'hits': hits, 'hyp_docs': hyp_docs, 'strategy': 'hyde_multi'}",
    "",
    'print("retrieve_standard()   defined.")',
    'print("retrieve_hyde_single() defined.")',
    'print("retrieve_hyde_multi()  defined.")',
)))

# ── Step 9 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 9 — RAG Query (all three strategies)"))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def rag_query(",
    "    question: str,",
    "    strategy: str = 'hyde_single',  # 'standard' | 'hyde_single' | 'hyde_multi'",
    "    verbose: bool = True",
    ") -> Dict:",
    "    t0 = time.time()",
    "    if strategy == 'standard':",
    "        result = retrieve_standard(question)",
    "    elif strategy == 'hyde_single':",
    "        result = retrieve_hyde_single(question)",
    "    else:",
    "        result = retrieve_hyde_multi(question)",
    "",
    "    answer  = generate_answer(question, [h['text'] for h in result['hits']])",
    "    latency = (time.time() - t0) * 1000",
    "",
    "    if verbose:",
    '        print(f"\\n[{strategy.upper()}]  Q: {question}")',
    "        if result['hyp_docs']:",
    "            print(f\"  Hyp doc sample: {result['hyp_docs'][0][:100]}...\")",
    '        print(f"  A: {answer}")',
    "        for i, h in enumerate(result['hits'][:3], 1):",
    "            print(f\"  [{i}] score={h['score']:.4f}  {h['text'][:75]}...\")",
    '        print(f"  Latency: {latency:.0f}ms")',
    '        print("-" * 70)',
    "",
    "    return {'question': question, 'answer': answer, 'strategy': strategy,",
    "            'hits': result['hits'], 'hyp_docs': result['hyp_docs'],",
    "            'latency_ms': latency}",
    "",
    "test_questions = [",
    '    "What is weather forecasting and why is it important?",',
    '    "What are the main methods used in weather analysis?",',
    '    "How does climatology differ from meteorology?",',
    "]",
    "results_log = []",
    "for q in test_questions:",
    "    results_log.append(rag_query(q, strategy='hyde_single'))",
)))

# ── Step 10 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 10 — Three-Strategy Comparison\n\n"
    "Run each question through all three strategies. Compare:\n"
    "- Top-1 retrieval score (higher = better match in vector space)\n"
    "- Latency (standard < single HyDE < multi HyDE)\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "compare_qs = [",
    '    "What causes temperature to decrease with altitude?",',
    '    "How do weather forecasters predict precipitation?",',
    '    "What is the relationship between pressure and wind?",',
    "]",
    "",
    "strategies = ['standard', 'hyde_single', 'hyde_multi']",
    "print('{:<52} {:>10}  {:>12}  {:>11}'.format(",
    "    'Question', 'Standard', 'HyDE-single', 'HyDE-multi'))",
    "print('-' * 92)",
    "",
    "for q in compare_qs:",
    "    scores = []",
    "    lats   = []",
    "    for strat in strategies:",
    "        t0 = time.time()",
    "        if strat == 'standard':",
    "            r = retrieve_standard(q)",
    "        elif strat == 'hyde_single':",
    "            r = retrieve_hyde_single(q)",
    "        else:",
    "            r = retrieve_hyde_multi(q)",
    "        lat = (time.time() - t0) * 1000",
    "        s1  = r['hits'][0]['score'] if r['hits'] else 0",
    "        scores.append(s1)",
    "        lats.append(lat)",
    "    best = strategies[scores.index(max(scores))]",
    "    print('{:<52} {:>8.4f}  {:>9.4f}  {:>9.4f}  <- best: {}'.format(",
    "        q[:51], scores[0], scores[1], scores[2], best))",
    "    print('{:<52} {:>8.0f}ms {:>9.0f}ms {:>9.0f}ms'.format('  latency', lats[0], lats[1], lats[2]))",
    "    print()",
)))

# ── Step 11 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 11 — Inspect Hypothetical Documents\n\n"
    "See exactly what Claude generates as hypothetical answers and how different\n"
    "N=3 generations look for the same question. Diversity in the generations\n"
    "is what makes Multi-HyDE more robust — averaging covers more of answer-space.\n"
))

cells.append(code(L(
    'demo_q = "Why does air pressure decrease with altitude?"',
    'print(f"Question: {demo_q}")',
    'print("=" * 70)',
    "",
    "hyp_docs = generate_hypothetical_docs(demo_q, n=3)",
    "for i, doc in enumerate(hyp_docs, 1):",
    "    print(f'\\n--- Hypothetical Doc {i} ({len(doc.split())} words) ---')",
    "    print(doc)",
    "",
    "# Show how similar the 3 hypothetical docs are to each other",
    "vecs = [embed_text(d) for d in hyp_docs]",
    "sims = []",
    "for i in range(len(vecs)):",
    "    for j in range(i+1, len(vecs)):",
    "        sim = sum(a*b for a,b in zip(vecs[i], vecs[j]))",
    "        sims.append((i+1, j+1, sim))",
    "print(f'\\nInter-doc cosine similarities (1.0 = identical):')",
    "for a, b, s in sims:",
    "    print(f'  doc{a} <-> doc{b}: {s:.4f}')",
    "avg_vec = avg_embeddings(vecs)",
    "print(f'\\nAveraged embedding will search centroid of these {len(hyp_docs)} answer vectors.')",
)))

# ── Step 12 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 12 — HyDE + BM25 Hybrid (bonus variant)\n\n"
    "Combine HyDE's vector with BM25 keyword search via RRF.\n"
    "This variant gets the vocabulary-gap fix from HyDE **and** the exact-term\n"
    "precision from BM25 — often the strongest combination.\n"
))

cells.append(code(L(
    "import re, math",
    "from collections import defaultdict",
    "from typing import List, Dict",
    "",
    "# Minimal BM25 (same as notebook 07, inline here for self-containment)",
    "def tokenise(text: str) -> List[str]:",
    "    return re.findall(r'[a-zA-Z0-9]+', text.lower())",
    "",
    "class BM25Index:",
    "    def __init__(self, corpus: List[str], k1=1.5, b=0.75):",
    "        self.k1 = k1; self.b = b; self.corpus = corpus; self.N = len(corpus)",
    "        self.tok  = [tokenise(d) for d in corpus]",
    "        self.avgdl = sum(len(t) for t in self.tok) / self.N",
    "        self.df: Dict[str, int] = defaultdict(int)",
    "        for toks in self.tok:",
    "            for t in set(toks): self.df[t] += 1",
    "    def idf(self, t):",
    "        df = self.df.get(t, 0)",
    "        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)",
    "    def score(self, tok_d, q_terms, dl):",
    "        tf_map = defaultdict(int)",
    "        for t in tok_d: tf_map[t] += 1",
    "        s = 0.0",
    "        for t in q_terms:",
    "            tf = tf_map.get(t, 0)",
    "            if tf == 0: continue",
    "            num = tf * (self.k1 + 1)",
    "            den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)",
    "            s  += self.idf(t) * num / den",
    "        return s",
    "    def search(self, query, top_k=20):",
    "        q_terms = tokenise(query)",
    "        scored  = [(i, self.score(t, q_terms, len(t))) for i, t in enumerate(self.tok)]",
    "        scored  = [(i, s) for i, s in scored if s > 0]",
    "        scored.sort(key=lambda x: x[1], reverse=True)",
    "        return [{'text': self.corpus[i], 'score': s, 'id': f'bm25_{i}',",
    "                 'metadata': {'chunk_idx': i}} for i, s in scored[:top_k]]",
    "",
    "bm25 = BM25Index(chunks)",
    'print(f"BM25 index ready: {len(chunks)} docs")',
    "",
    "def rrf(list_a, list_b, k=60, top_k=TOP_K):",
    "    scores = defaultdict(float)",
    "    docs   = {}",
    "    for rank, doc in enumerate(list_a):",
    "        key = doc['text'][:60]; scores[key] += 1/(k+rank+1); docs[key] = doc",
    "    for rank, doc in enumerate(list_b):",
    "        key = doc['text'][:60]; scores[key] += 1/(k+rank+1)",
    "        if key not in docs: docs[key] = doc",
    "    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)",
    "    return [{**docs[k], 'rrf_score': s} for k, s in ranked[:top_k]]",
    "",
    "def retrieve_hyde_hybrid(question, n=HYDE_N_DOCS, top_k=TOP_K):",
    "    hyp_docs = generate_hypothetical_docs(question, n=n)",
    "    avg_vec  = avg_embeddings([embed_text(d) for d in hyp_docs])",
    "    vec_hits = vs.search(avg_vec, top_k=20)",
    "    bm_hits  = bm25.search(question, top_k=20)",
    "    merged   = rrf(vec_hits, bm_hits, top_k=top_k)",
    "    return {'hits': merged, 'hyp_docs': hyp_docs, 'strategy': 'hyde_hybrid'}",
    "",
    "# Quick demo",
    'q = "What are synoptic weather patterns and how are they analysed?"',
    'print(f"HyDE+Hybrid demo: {q}")',
    "r = retrieve_hyde_hybrid(q)",
    "print(f\"Top-3 results:\")",
    "for i, h in enumerate(r['hits'][:3], 1):",
    "    print(f\"  [{i}] rrf={h['rrf_score']:.4f}  {h['text'][:80]}...\")",
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
    "eval_strategies = ['standard', 'hyde_single', 'hyde_multi']",
    "print('{:<45} {:>14}  {:>14}  {:>14}'.format(",
    "    'Question', 'Standard', 'HyDE-single', 'HyDE-multi'))",
    "print('-' * 92)",
    "",
    "for case in eval_cases:",
    "    q   = case['question']",
    "    kws = case['keywords']",
    "    n   = len(kws)",
    "    cols = []",
    "    for strat in eval_strategies:",
    "        r    = rag_query(q, strategy=strat, verbose=False)",
    "        hits = sum(1 for kw in kws if kw in r['answer'].lower())",
    "        cols.append(f'{hits}/{n} ({hits/n*100:.0f}%) {r[\"latency_ms\"]:.0f}ms')",
    "    print('{:<45} {:>14}  {:>14}  {:>14}'.format(q[:44], cols[0], cols[1], cols[2]))",
)))

# ── Step 14 Summary ────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 14 — Summary\n\n"
    "| Component | Implementation |\n"
    "|-----------|---------------|\n"
    "| PDF | `pypdf.PdfReader` — absolute path in Step 2 |\n"
    "| Chunking | Native `recursive_split()` — 500 chars |\n"
    "| Hypothetical doc | Strands Agent — 'write a passage from a textbook' |\n"
    "| Single HyDE | embed(hyp_doc) → search |\n"
    "| Multi-HyDE | avg(embed(doc_1..N)) → search — more robust |\n"
    "| HyDE + Hybrid | avg HyDE vec + BM25 via RRF — strongest variant |\n"
    "| Vector DB | Qdrant Cloud → OpenSearch → in-memory |\n"
    "| LLM | AWS Strands Agent + Claude Sonnet 4.6 |\n\n"
    "## Strategy trade-offs\n\n"
    "| Strategy | Latency overhead | When to use |\n"
    "|----------|-----------------|-------------|\n"
    "| Standard | none | Low latency, query terms match doc terms |\n"
    "| HyDE single | +1 LLM call | Vocabulary gap between question and docs |\n"
    "| HyDE multi | +N LLM calls | High-stakes queries, noisy domain |\n"
    "| HyDE + Hybrid | +N LLM calls | Best quality, can afford latency |\n\n"
    "### Next: **09 — Reranking**\n"
))

cells.append(code(L(
    "# vs.delete_collection()  # uncomment to clean up",
    "print(f\"Collection '{COLLECTION_NAME}' in {vs._backend} — {vs.count()} vectors\")",
    'print("\\nDone. Give the go-ahead for notebook 09.")',
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
