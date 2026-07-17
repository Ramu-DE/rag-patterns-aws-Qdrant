"""Build 10_Contextual_Compression.ipynb — full implementation."""
import json, uuid, ast

PATH = r"C:\Users\Administrator\RAG\qdrant_notebooks\tier2_retrieval_quality\10_Contextual_Compression.ipynb"
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
"# 10 — Contextual Compression\n"
"\n"
"> **Tier 2 | Retrieval Quality**\n"
"\n"
"## The Problem\n"
"\n"
"After retrieval, each chunk is sent wholesale to the LLM — but most of a 500-char\n"
"chunk is often **noise** relative to the specific question asked:\n"
"\n"
"```\n"
"Question : \"What is the lapse rate in the troposphere?\"\n"
"\n"
"Retrieved chunk (500 chars):\n"
"  'The atmosphere is divided into several layers. The lowest layer,\n"
"   the troposphere, extends from the surface to about 12 km. Weather\n"
"   occurs primarily in this layer. The temperature lapse rate averages\n"
"   6.5 degrees C per km. The tropopause marks the boundary above which\n"
"   temperature begins to increase again in the stratosphere...'\n"
"\n"
"Only this sentence matters: 'The temperature lapse rate averages 6.5 degrees C per km.'\n"
"```\n"
"\n"
"Sending the full chunk:\n"
"- Wastes LLM context tokens\n"
"- Dilutes the relevant signal with off-topic sentences\n"
"- Increases hallucination risk when noise outnumbers signal\n"
"\n"
"## Contextual Compression Solution\n"
"\n"
"After retrieval, a **compression step** extracts only the query-relevant\n"
"portions from each chunk before passing to the LLM:\n"
"\n"
"```\n"
"Retrieve → [chunk_1, chunk_2, ..., chunk_K]\n"
"         ↓ compress each chunk\n"
"         → [relevant_extract_1, relevant_extract_2, ...]\n"
"         ↓ drop empty extracts\n"
"         → LLM generates answer from compressed context\n"
"```\n"
"\n"
"## Three Compression Strategies\n"
"\n"
"| Strategy | Method | Output |\n"
"|----------|--------|--------|\n"
"| **LLM Extract** | Claude extracts relevant sentences verbatim | Subset of original text |\n"
"| **LLM Summarise** | Claude summarises only the relevant part | Paraphrase, shorter |\n"
"| **Sentence Filter** | Split chunk into sentences, score each, keep top-N | Sentence-level granularity |\n"
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
"    subgraph RETRIEVAL [\"⚡  RETRIEVAL\"]\n"
"        Q([\"❓ Query\"])\n"
"        Q --> QEMB[\"embed(query)\"]\n"
"        QEMB --> VS[\"Vector search\\ntop-K chunks\"]\n"
"        QDRANT --> VS\n"
"        VS --> CHUNKS([\"K raw chunks\\n(full text)\"])\n"
"    end\n"
"\n"
"    subgraph COMPRESS [\"✂️  COMPRESSION  (new step)\"]\n"
"        CHUNKS --> C1[\"Chunk 1\\nLLM: extract relevant sentences\"]\n"
"        CHUNKS --> C2[\"Chunk 2\\nLLM: extract relevant sentences\"]\n"
"        CHUNKS --> CK[\"Chunk K\\nLLM: extract relevant sentences\"]\n"
"        C1 --> FILTER{\"Non-empty?\"}\n"
"        C2 --> FILTER\n"
"        CK --> FILTER\n"
"        FILTER -->|Yes| KEEP([\"Compressed\\npassages\"])\n"
"        FILTER -->|No| DROP([\"Dropped\"])\n"
"    end\n"
"\n"
"    subgraph GEN [\"🟠  GENERATION\"]\n"
"        KEEP --> LLM[\"Strands Agent\\nClaude Sonnet 4.6\"]\n"
"        LLM --> ANS([\"✅ Answer\"])\n"
"    end\n"
"\n"
"    subgraph COMPARE [\"📊  A/B COMPARE\"]\n"
"        direction LR\n"
"        FULL[\"Full chunks\\n(no compression)\"] \n"
"        COMP[\"Compressed chunks\"]\n"
"        FULL --> DIFF[\"Context size\\nvs answer quality\"]\n"
"        COMP --> DIFF\n"
"    end\n"
"\n"
"    style INDEX    fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f\n"
"    style RETRIEVAL fill:#dcfce7,stroke:#16a34a,color:#14532d\n"
"    style COMPRESS fill:#fef9c3,stroke:#ca8a04,color:#713f12\n"
"    style GEN      fill:#ffedd5,stroke:#f97316,color:#7c2d12\n"
"    style COMPARE  fill:#faf5ff,stroke:#a855f7,color:#3b0764\n"
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
    'COLLECTION_NAME = "contextual_compression_10"',
    "EMBEDDING_DIM   = 1024",
    "TOP_K           = 6    # retrieve more chunks — some may compress to nothing",
    "FINAL_K         = 4    # max compressed passages sent to LLM",
    "",
    "# Chunking",
    "CHUNK_SIZE    = 500",
    "CHUNK_OVERLAP = 50",
    "",
    "# Compression settings",
    "MIN_EXTRACT_LEN = 20   # discard extracts shorter than this (likely 'none' or empty)",
    "MAX_SUMMARY_LEN = 150  # max chars for summary-mode compression",
    "",
    "# PDF",
    'PDF_PATH = r"' + PDF + '"',
    "",
    'print(f"Collection  : {COLLECTION_NAME}")',
    'print(f"Retrieve K  : {TOP_K}  |  Final K after compression: {FINAL_K}")',
    'print(f"Min extract : {MIN_EXTRACT_LEN} chars")',
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
    "    def search(self, qvec: List[float], top_k: int = 6) -> List[Dict]:",
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
    "    context = '\\n\\n'.join(f'[Passage {i+1}]\\n{c}' for i, c in enumerate(context_chunks))",
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
    'test_emb = embed_text("contextual compression extraction")',
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
    "## Step 7 — Strategy A: LLM Extract\n\n"
    "Ask Claude to extract **verbatim sentences** from the chunk that directly\n"
    "answer the question. If nothing is relevant, it outputs `NONE`.\n\n"
    "The output stays faithful to the source — no paraphrasing, no hallucination risk.\n"
))

cells.append(code(L(
    "from typing import List, Dict, Optional",
    "",
    "EXTRACT_PROMPT = (",
    "    'Extract the sentences from the passage below that are directly relevant '",
    "    'to answering the question. Copy them verbatim.\\n'",
    "    'If no sentences are relevant, output exactly: NONE\\n\\n'",
    "    'Question: {question}\\n\\n'",
    "    'Passage:\\n{passage}\\n\\n'",
    "    'Relevant sentences (verbatim or NONE):'",
    ")",
    "",
    "def compress_extract(question: str, chunk: str) -> Optional[str]:",
    "    prompt = EXTRACT_PROMPT.format(question=question, passage=chunk)",
    "    result = str(Agent(",
    "        model=_model,",
    "        system_prompt='You are a precise text extractor. Copy sentences verbatim or output NONE.'",
    "    )(prompt)).strip()",
    "    if result.upper() == 'NONE' or len(result) < MIN_EXTRACT_LEN:",
    "        return None",
    "    return result",
    "",
    "def compress_chunks_extract(question: str, hits: List[Dict]) -> List[Dict]:",
    "    results = []",
    "    for hit in hits:",
    "        compressed = compress_extract(question, hit['text'])",
    "        results.append({",
    "            **hit,",
    "            'compressed'     : compressed,",
    "            'strategy'       : 'extract',",
    "            'original_chars' : len(hit['text']),",
    "            'compressed_chars': len(compressed) if compressed else 0,",
    "            'kept'           : compressed is not None,",
    "        })",
    "        time.sleep(0.05)",
    "    return results",
    "",
    "# Smoke test",
    'test_q = "What is the temperature lapse rate in the troposphere?"',
    "test_hits = vs.search(embed_text(test_q), top_k=2)",
    'print(f"Smoke test — extracting from 2 chunks for: {test_q}\\n")',
    "for h in test_hits:",
    "    ex = compress_extract(test_q, h['text'])",
    "    ratio = len(ex)/len(h['text'])*100 if ex else 0",
    "    print(f'Original  ({len(h[\"text\"])} chars): {h[\"text\"][:120]}...')",
    "    print(f'Extracted ({len(ex) if ex else 0} chars, {ratio:.0f}%): {ex or \"NONE\"}')",
    "    print()",
)))

# ── Step 8 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 8 — Strategy B: LLM Summarise\n\n"
    "Instead of extracting verbatim, ask Claude to **summarise only the relevant\n"
    "part** of the chunk. Produces shorter, denser context.\n\n"
    "Trade-off vs. extraction:\n"
    "- More concise — better for very noisy chunks\n"
    "- Slight hallucination risk — paraphrasing can introduce errors\n"
    "- Better for chunks with scattered relevant content\n"
))

cells.append(code(L(
    "from typing import List, Dict, Optional",
    "",
    "SUMMARISE_PROMPT = (",
    "    'Summarise the parts of the passage below that are relevant to the question.\\n'",
    "    f'Keep the summary under {MAX_SUMMARY_LEN} characters.\\n'",
    "    'If nothing is relevant output exactly: NONE\\n\\n'",
    "    'Question: {question}\\n\\n'",
    "    'Passage:\\n{passage}\\n\\n'",
    "    'Summary of relevant content (or NONE):'",
    ")",
    "",
    "def compress_summarise(question: str, chunk: str) -> Optional[str]:",
    "    prompt = SUMMARISE_PROMPT.format(question=question, passage=chunk)",
    "    result = str(Agent(",
    "        model=_model,",
    "        system_prompt='You are a concise summariser. Summarise only what is relevant or output NONE.'",
    "    )(prompt)).strip()",
    "    if result.upper() == 'NONE' or len(result) < MIN_EXTRACT_LEN:",
    "        return None",
    "    return result",
    "",
    "def compress_chunks_summarise(question: str, hits: List[Dict]) -> List[Dict]:",
    "    results = []",
    "    for hit in hits:",
    "        compressed = compress_summarise(question, hit['text'])",
    "        results.append({",
    "            **hit,",
    "            'compressed'      : compressed,",
    "            'strategy'        : 'summarise',",
    "            'original_chars'  : len(hit['text']),",
    "            'compressed_chars': len(compressed) if compressed else 0,",
    "            'kept'            : compressed is not None,",
    "        })",
    "        time.sleep(0.05)",
    "    return results",
    "",
    'print("compress_chunks_extract()    — verbatim sentence extraction")',
    'print("compress_chunks_summarise()  — concise summary of relevant part")',
)))

# ── Step 9 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 9 — Strategy C: Sentence-Level Filter\n\n"
    "No LLM needed for this one. Split each chunk into sentences, embed each,\n"
    "compute cosine similarity to the query, keep sentences above a threshold.\n\n"
    "Fastest compression strategy — no extra LLM calls at query time.\n"
))

cells.append(code(L(
    "from typing import List, Dict, Optional",
    "",
    "def split_sentences(text: str) -> List[str]:",
    "    parts = re.split(r'(?<=[.!?])\\s+', text.replace('\\n', ' '))",
    "    return [p.strip() for p in parts if len(p.strip()) > 15]",
    "",
    "def cosine_sim(a: List[float], b: List[float]) -> float:",
    "    return sum(x * y for x, y in zip(a, b))   # already L2-normalised",
    "",
    "def compress_sentence_filter(",
    "    question: str,",
    "    chunk: str,",
    "    threshold: float = 0.55,",
    "    top_n: int = 3",
    ") -> Optional[str]:",
    "    sentences = split_sentences(chunk)",
    "    if not sentences:",
    "        return None",
    "    qvec  = embed_text(question)",
    "    scored = []",
    "    for sent in sentences:",
    "        svec = embed_text(sent)",
    "        sim  = cosine_sim(qvec, svec)",
    "        scored.append((sim, sent))",
    "        time.sleep(0.02)",
    "    scored.sort(reverse=True)",
    "    kept = [s for sim, s in scored[:top_n] if sim >= threshold]",
    "    return ' '.join(kept) if kept else None",
    "",
    "def compress_chunks_sentence_filter(question: str, hits: List[Dict]) -> List[Dict]:",
    "    results = []",
    "    for hit in hits:",
    "        compressed = compress_sentence_filter(question, hit['text'])",
    "        results.append({",
    "            **hit,",
    "            'compressed'      : compressed,",
    "            'strategy'        : 'sentence_filter',",
    "            'original_chars'  : len(hit['text']),",
    "            'compressed_chars': len(compressed) if compressed else 0,",
    "            'kept'            : compressed is not None,",
    "        })",
    "    return results",
    "",
    'print("compress_chunks_sentence_filter() — embedding similarity, no extra LLM calls")',
)))

# ── Step 10 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 10 — Full RAG Pipeline (three compression modes)"))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def rag_query(",
    "    question: str,",
    "    mode: str = 'extract',  # 'none' | 'extract' | 'summarise' | 'sentence_filter'",
    "    verbose: bool = True",
    ") -> Dict:",
    "    t0   = time.time()",
    "    qvec = embed_text(question)",
    "    hits = vs.search(qvec, top_k=TOP_K)",
    "",
    "    if mode == 'none':",
    "        compressed = [{**h, 'compressed': h['text'], 'kept': True,",
    "                        'original_chars': len(h['text']),",
    "                        'compressed_chars': len(h['text']),",
    "                        'strategy': 'none'} for h in hits]",
    "    elif mode == 'extract':",
    "        compressed = compress_chunks_extract(question, hits)",
    "    elif mode == 'summarise':",
    "        compressed = compress_chunks_summarise(question, hits)",
    "    else:",
    "        compressed = compress_chunks_sentence_filter(question, hits)",
    "",
    "    kept    = [c for c in compressed if c['kept']][:FINAL_K]",
    "    dropped = len(compressed) - len(kept)",
    "    passages = [c['compressed'] for c in kept]",
    "    answer   = generate_answer(question, passages)",
    "    latency  = (time.time() - t0) * 1000",
    "",
    "    orig_chars  = sum(c['original_chars']  for c in compressed)",
    "    comp_chars  = sum(c['compressed_chars'] for c in kept)",
    "    reduction   = (1 - comp_chars / orig_chars) * 100 if orig_chars else 0",
    "",
    "    if verbose:",
    '        print(f"\\n[{mode.upper()}]  Q: {question}")',
    '        print(f"A: {answer}")',
    "        print(f\"  Retrieved: {len(hits)}  |  Kept after compress: {len(kept)}  |  Dropped: {dropped}\")",
    "        print(f\"  Context: {orig_chars} chars -> {comp_chars} chars  ({reduction:.0f}% reduction)\")",
    "        for i, c in enumerate(kept[:3], 1):",
    "            print(f\"  [kept {i}] {c['compressed_chars']}chars  {str(c['compressed'])[:75]}...\")",
    '        print(f"  Latency: {latency:.0f}ms")',
    '        print("-" * 70)',
    "",
    "    return {",
    "        'question': question, 'answer': answer, 'mode': mode,",
    "        'latency_ms': latency, 'n_retrieved': len(hits),",
    "        'n_kept': len(kept), 'n_dropped': dropped,",
    "        'orig_chars': orig_chars, 'comp_chars': comp_chars,",
    "        'reduction_pct': reduction,",
    "    }",
    "",
    "test_questions = [",
    '    "What is weather forecasting and why is it important?",',
    '    "What are the main methods used in weather analysis?",',
    '    "How does climatology differ from meteorology?",',
    '    "What factors influence weather patterns and climate?",',
    "]",
    "results_log = []",
    "for q in test_questions:",
    "    results_log.append(rag_query(q, mode='extract'))",
)))

# ── Step 11 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 11 — Compression Visualiser\n\n"
    "For a single query, show the full chunk alongside its compressed extract\n"
    "so you can see exactly what gets kept and what gets discarded.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    'demo_q = "What are the main methods used in weather analysis?"',
    "qvec   = embed_text(demo_q)",
    "demo_hits = vs.search(qvec, top_k=3)",
    "",
    'print(f"Q: {demo_q}")',
    'print("=" * 70)',
    "",
    "for i, hit in enumerate(demo_hits, 1):",
    "    ex  = compress_extract(demo_q, hit['text'])",
    "    sm  = compress_summarise(demo_q, hit['text'])",
    "    sf  = compress_sentence_filter(demo_q, hit['text'])",
    "    orig_len = len(hit['text'])",
    "    print(f'--- Chunk {i}  (vec score={hit[\"score\"]:.4f}, {orig_len} chars) ---')",
    "    print(f'ORIGINAL  : {hit[\"text\"][:200]}...')",
    "    print()",
    "    print(f'EXTRACT   ({len(ex) if ex else 0} chars): {ex or \"NONE\"}')",
    "    print(f'SUMMARISE ({len(sm) if sm else 0} chars): {sm or \"NONE\"}')",
    "    print(f'SENT FILT ({len(sf) if sf else 0} chars): {sf or \"NONE\"}')",
    "    print()",
)))

# ── Step 12 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 12 — Four-Mode Comparison\n\n"
    "Compare no compression vs. all three strategies on the same questions.\n"
    "Key metrics: context size reduction and keyword hit rate.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "compare_qs = [",
    '    "What is weather forecasting and why is it important?",',
    '    "What are the main methods used in weather analysis?",',
    '    "How does climatology differ from meteorology?",',
    "]",
    "modes = ['none', 'extract', 'summarise', 'sentence_filter']",
    "keywords_map = {",
    '    "What is weather forecasting and why is it important?":',
    "        ['forecast','weather','predict','atmosphere','climate'],",
    '    "What are the main methods used in weather analysis?":',
    "        ['analysis','synoptic','observation','data','pressure'],",
    '    "How does climatology differ from meteorology?":',
    "        ['climate','weather','long','study','atmosphere'],",
    "}",
    "",
    "print('{:<48} {:>10}  {:>10}  {:>12}  {:>13}'.format(",
    "    'Question', 'None', 'Extract', 'Summarise', 'SentFilter'))",
    "print('-' * 99)",
    "",
    "for q in compare_qs:",
    "    kws = keywords_map[q]; n = len(kws)",
    "    cols = []",
    "    for mode in modes:",
    "        r    = rag_query(q, mode=mode, verbose=False)",
    "        hits = sum(1 for kw in kws if kw in r['answer'].lower())",
    "        cols.append(f'{hits}/{n} -{r[\"reduction_pct\"]:.0f}%')",
    "    print('{:<48} {:>10}  {:>10}  {:>12}  {:>13}'.format(q[:47], *cols))",
    "",
    "print()",
    "print('Format: keyword_hits/total  -context_reduction%')",
)))

# ── Step 13 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 13 — Context Size vs Answer Quality\n\n"
    "Plot the reduction ratio vs keyword hit rate to find the\n"
    "sweet spot between compression and answer completeness.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "eval_qs = [",
    '    "What is weather forecasting and why is it important?",',
    '    "What are the main methods used in weather analysis?",',
    '    "How does climatology differ from meteorology?",',
    '    "What factors influence weather patterns and climate?",',
    "]",
    "eval_kws = {",
    '    "What is weather forecasting and why is it important?": [\'forecast\',\'weather\',\'predict\',\'atmosphere\',\'climate\'],',
    '    "What are the main methods used in weather analysis?": [\'analysis\',\'synoptic\',\'observation\',\'data\',\'pressure\'],',
    '    "How does climatology differ from meteorology?": [\'climate\',\'weather\',\'long\',\'study\',\'atmosphere\'],',
    '    "What factors influence weather patterns and climate?": [\'factors\',\'weather\',\'patterns\',\'pressure\',\'temperature\'],',
    "}",
    "",
    "summary_rows = []",
    "for mode in ['none', 'extract', 'summarise', 'sentence_filter']:",
    "    tot_kw = 0; tot_n = 0; tot_red = 0; tot_lat = 0",
    "    for q in eval_qs:",
    "        r    = rag_query(q, mode=mode, verbose=False)",
    "        kws  = eval_kws[q]",
    "        hits = sum(1 for kw in kws if kw in r['answer'].lower())",
    "        tot_kw  += hits",
    "        tot_n   += len(kws)",
    "        tot_red += r['reduction_pct']",
    "        tot_lat += r['latency_ms']",
    "    summary_rows.append({",
    "        'mode': mode,",
    "        'kw_pct'     : tot_kw / tot_n * 100,",
    "        'avg_red_pct': tot_red / len(eval_qs),",
    "        'avg_lat_ms' : tot_lat / len(eval_qs),",
    "    })",
    "",
    "print('{:<18} {:>10}  {:>14}  {:>12}'.format(",
    "    'Mode', 'KW hit %', 'Context reduc%', 'Avg latency'))",
    "print('-' * 60)",
    "for r in summary_rows:",
    "    print('{:<18} {:>10.0f}%  {:>13.0f}%  {:>10.0f}ms'.format(",
    "        r['mode'], r['kw_pct'], r['avg_red_pct'], r['avg_lat_ms']))",
)))

# ── Step 14 Summary ────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 14 — Summary\n\n"
    "| Component | Implementation |\n"
    "|-----------|---------------|\n"
    "| PDF | `pypdf.PdfReader` — absolute path in Step 2 |\n"
    "| Retrieval | Qdrant vector search — top-6 candidates |\n"
    "| Compress: Extract | Claude extracts verbatim relevant sentences |\n"
    "| Compress: Summarise | Claude summarises relevant portion (≤150 chars) |\n"
    "| Compress: Sentence filter | Cosine sim per sentence — no extra LLM call |\n"
    "| Drop threshold | Extracts shorter than 20 chars are discarded |\n"
    "| Final context | Up to 4 compressed passages sent to LLM |\n"
    "| Vector DB | Qdrant Cloud → OpenSearch → in-memory |\n"
    "| LLM | AWS Strands Agent + Claude Sonnet 4.6 |\n\n"
    "## Strategy comparison\n\n"
    "| Strategy | Extra LLM calls | Faithfulness | Context reduction |\n"
    "|----------|----------------|--------------|------------------|\n"
    "| None | 0 | N/A — full chunk | 0% |\n"
    "| Extract | K (one per chunk) | High — verbatim | 40–70% typical |\n"
    "| Summarise | K | Medium — paraphrase | 60–80% typical |\n"
    "| Sentence filter | 0 (embedding only) | High — verbatim | 30–60% typical |\n\n"
    "**Recommendation:** use `extract` for high-stakes queries where faithfulness matters;\n"
    "use `sentence_filter` when latency is tight.\n\n"
    "### Next: **11 — Metadata Filtering**\n"
))

cells.append(code(L(
    "# vs.delete_collection()  # uncomment to clean up",
    "print(f\"Collection '{COLLECTION_NAME}' in {vs._backend} — {vs.count()} vectors\")",
    'print("\\nDone. Give the go-ahead for notebook 11.")',
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
