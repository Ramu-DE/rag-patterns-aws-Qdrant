"""Build 11_Metadata_Filtering.ipynb — full implementation."""
import json, uuid, ast

PATH = r"C:\Users\Administrator\RAG\qdrant_notebooks\tier2_retrieval_quality\11_Metadata_Filtering.ipynb"
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

# ── Overview ───────────────────────────────────────────────────────────────────
cells.append(md(
"# 11 — Metadata Filtering RAG\n"
"\n"
"> **Tier 2 | Retrieval Quality**\n"
"\n"
"## The Problem\n"
"\n"
"Pure vector search retrieves the semantically closest chunks globally —\n"
"it cannot restrict results to a specific document section, page range,\n"
"or topic category. When users ask scoped questions the wrong chunks\n"
"appear at the top:\n"
"\n"
"```\n"
"Q: \"What does the introduction say about climate sensitivity?\"\n"
"\n"
"Vector search (no filter):\n"
"  Rank 1 — Chapter 7, page 42 (score 0.91)\n"
"  Rank 2 — Appendix B, page 89 (score 0.89)\n"
"  Rank 3 — Introduction, page 2  (score 0.87)  ← correct but ranked 3rd\n"
"\n"
"Filtered search (section='introduction'):\n"
"  Rank 1 — Introduction, page 2  (score 0.87)  ← only candidate from section\n"
"  Rank 2 — Introduction, page 3  (score 0.84)\n"
"```\n"
"\n"
"## Metadata Filtering Solution\n"
"\n"
"Tag every chunk with **structured payload** at index time, then apply\n"
"a `Filter` object at query time. Qdrant evaluates the filter **before**\n"
"the vector search, so it narrows the candidate pool cheaply:\n"
"\n"
"```\n"
"Index time:\n"
"  chunk → embed → Qdrant point with payload:\n"
"    { page_num: 3, section: 'introduction', topic: 'temperature',\n"
"      char_count: 487, word_count: 76, has_numbers: true }\n"
"\n"
"Query time:\n"
"  filter = Filter(must=[page_num in [1,10], topic=='temperature'])\n"
"  results = qdrant.query_points(query=embed(q), filter=filter, limit=5)\n"
"```\n"
"\n"
"## Metadata Fields\n"
"\n"
"| Field | Type | Example | Notes |\n"
"|-------|------|---------|-------|\n"
"| `page_num` | int | `3` | PDF page number |\n"
"| `char_count` | int | `487` | Chunk character length |\n"
"| `word_count` | int | `76` | Approximate word count |\n"
"| `has_numbers` | bool | `true` | Contains numeric data |\n"
"| `section` | str | `'introduction'` | Detected heading category |\n"
"| `topic` | str | `'temperature'` | Keyword-classified topic |\n"
"| `source` | str | `'climate.pdf'` | Origin file |\n"
))

# ── Mermaid diagram ────────────────────────────────────────────────────────────
cells.append(md(
"## Flow Diagram\n"
"\n"
"```mermaid\n"
"flowchart TD\n"
"    subgraph INDEX [\"🔵  INDEXING — rich payload\"]\n"
"        PDF([\"📄 climate.pdf\"])\n"
"        PDF --> PAGES[\"Extract text\\nper page\"]\n"
"        PAGES --> META[\"detect_section()\\nclassify_topic()\"]\n"
"        META --> SPLIT[\"Fixed-size chunks\\n~500 chars\"]\n"
"        SPLIT --> TAG[\"Tag each chunk:\\npage_num, section, topic,\\nchar_count, has_numbers\"]\n"
"        TAG --> EMB[\"Embed — Titan V2\"]\n"
"        EMB --> QDRANT[(\"Qdrant\\n+ payload index\")]\n"
"    end\n"
"\n"
"    subgraph FILTER [\"🟡  FILTER STRATEGIES\"]\n"
"        direction LR\n"
"        F1[\"Exact match\\nsection='introduction'\"]\n"
"        F2[\"Range\\npage_num 1-20\"]\n"
"        F3[\"Multi-condition\\ntopic='temperature'\\nAND has_numbers=true\"]\n"
"        F4[\"Dynamic\\nLLM extracts filter\\nfrom query text\"]\n"
"    end\n"
"\n"
"    subgraph QUERY [\"⚡  FILTERED RETRIEVAL\"]\n"
"        Q([\"❓ Query\"])\n"
"        Q --> QEMB[\"embed(query)\"]\n"
"        FILTER --> FOBJ[\"Filter object\"]\n"
"        QEMB --> VS[\"query_points\\n(query + filter)\"]   \n"
"        FOBJ --> VS\n"
"        QDRANT --> VS\n"
"        VS --> HITS([\"Top-K within\\nfiltered subset\"])\n"
"    end\n"
"\n"
"    subgraph GEN [\"🟠  GENERATION\"]\n"
"        HITS --> LLM[\"Strands Agent\\nClaude Sonnet 4.6\"]\n"
"        LLM --> ANS([\"✅ Answer\"])\n"
"    end\n"
"\n"
"    style INDEX   fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f\n"
"    style FILTER  fill:#fef9c3,stroke:#ca8a04,color:#713f12\n"
"    style QUERY   fill:#dcfce7,stroke:#16a34a,color:#14532d\n"
"    style GEN     fill:#ffedd5,stroke:#f97316,color:#7c2d12\n"
"```\n"
))

# ── Step 1 ─────────────────────────────────────────────────────────────────────
cells.append(md("## Step 1 — Install & Import Dependencies"))

cells.append(code(L(
    "import subprocess, sys",
    'packages = ["boto3","qdrant-client","opensearch-py","requests-aws4auth","strands-agents","pypdf"]',
    'subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + packages)',
    'print("All packages ready.")',
)))

cells.append(code(L(
    "import os, json, time, uuid, re",
    "from typing import List, Dict, Optional, Any",
    "",
    "import boto3, pypdf",
    "from strands import Agent",
    "from strands.models.bedrock import BedrockModel",
    "from qdrant_client import QdrantClient",
    "from qdrant_client.models import (",
    "    Distance, VectorParams, PointStruct,",
    "    Filter, FieldCondition, MatchValue, MatchAny, Range",
    ")",
    "",
    'print("Imports OK")',
)))

# ── Step 2 ─────────────────────────────────────────────────────────────────────
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
    'COLLECTION_NAME = "metadata_filtering_11"',
    "EMBEDDING_DIM   = 1024",
    "TOP_K           = 5",
    "",
    "# Chunking",
    "CHUNK_SIZE    = 500",
    "CHUNK_OVERLAP = 50",
    "",
    "# PDF",
    'PDF_PATH = r"' + PDF + '"',
    "",
    "# Topic taxonomy — keyword buckets",
    "TOPIC_KEYWORDS = {",
    "    'temperature' : ['temperature','warming','heat','thermal','lapse','celsius','fahrenheit'],",
    "    'ocean'       : ['ocean','sea','marine','coastal','salinity','current','tide'],",
    "    'precipitation': ['rain','precipitation','snow','flood','drought','moisture','humidity'],",
    "    'emissions'   : ['emission','carbon','greenhouse','co2','methane','fossil','aerosol'],",
    "    'policy'      : ['policy','agreement','mitigation','adaptation','target','commitment'],",
    "    'atmosphere'  : ['atmosphere','stratosphere','troposphere','ozone','pressure','wind'],",
    "}",
    "",
    'print(f"Collection : {COLLECTION_NAME}")',
    'print(f"PDF        : {PDF_PATH}")',
    'print(f"PDF exists : {os.path.exists(PDF_PATH)}")',
    'print(f"Topics     : {list(TOPIC_KEYWORDS.keys())}")',
)))

# ── Step 3 ─────────────────────────────────────────────────────────────────────
cells.append(md("## Step 3 — Vector Store (with metadata filter support)"))

cells.append(code(L(
    "from typing import List, Dict, Optional, Any",
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
    "    def create_payload_indexes(self):",
    "        if self._backend not in ('qdrant_cloud', 'qdrant_memory'):",
    "            print('Payload indexes only supported on Qdrant backend')",
    "            return",
    "        index_defs = [",
    "            ('metadata.page_num',   'integer'),",
    "            ('metadata.char_count', 'integer'),",
    "            ('metadata.word_count', 'integer'),",
    "            ('metadata.has_numbers','bool'),",
    "            ('metadata.section',    'keyword'),",
    "            ('metadata.topic',      'keyword'),",
    "        ]",
    "        for field, schema in index_defs:",
    "            self._qdrant.create_payload_index(",
    "                collection_name=self.name,",
    "                field_name=field,",
    "                field_schema=schema",
    "            )",
    "        print(f'Payload indexes created: {[f for f, _ in index_defs]}')",
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
    "    def search(",
    "        self,",
    "        qvec: List[float],",
    "        top_k: int = 5,",
    "        query_filter: Optional[Any] = None",
    "    ) -> List[Dict]:",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            resp = self._qdrant.query_points(",
    "                collection_name=self.name,",
    "                query=qvec,",
    "                limit=top_k,",
    "                query_filter=query_filter,",
    "                with_payload=True",
    "            )",
    "            return [{'text': p.payload.get('text',''),",
    "                     'metadata': p.payload.get('metadata',{}),",
    "                     'score': p.score, 'id': str(p.id)}",
    "                    for p in resp.points]",
    "        elif self._backend == 'opensearch':",
    "            body = {'size': top_k,",
    "                    'query': {'knn': {'embedding': {'vector': qvec, 'k': top_k}}},",
    "                    '_source': {'excludes': ['embedding']}}",
    "            resp = self._os.search(index=self.name, body=body)",
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
    'print("VectorStore defined (filter-aware).")',
)))

# ── Step 4 ─────────────────────────────────────────────────────────────────────
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
    'test_emb = embed_text("metadata filtering retrieval")',
    'print(f"Embedding OK — dim={len(test_emb)}")',
    'print("BedrockModel ready.")',
)))

# ── Step 5 ─────────────────────────────────────────────────────────────────────
cells.append(md("## Step 5 — Connect, Create Collection & Payload Indexes"))

cells.append(code(L(
    "vs = VectorStore(",
    "    collection_name=COLLECTION_NAME,",
    "    qdrant_url=QDRANT_URL,",
    "    qdrant_api_key=QDRANT_API_KEY,",
    "    opensearch_url=OPENSEARCH_URL,",
    "    region=AWS_REGION",
    ")",
    'print(f"Backend: {vs._backend}")',
    "",
    "vs.create_collection(dim=EMBEDDING_DIM, force_recreate=True)",
    "",
    "# Create payload indexes for fast pre-filter evaluation",
    "vs.create_payload_indexes()",
)))

# ── Step 6 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 6 — Section Detection & Topic Classification\n\n"
    "Two pure-Python helpers that run at chunk-creation time:\n\n"
    "- `detect_section(text)` — infers section category from text patterns\n"
    "- `classify_topic(text)` — keyword voting across topic buckets\n"
))

cells.append(code(L(
    "from typing import Dict, List",
    "",
    "# Section patterns (checked in priority order)",
    "SECTION_PATTERNS = [",
    "    (r'(?i)\\bintroduction\\b',                    'introduction'),",
    "    (r'(?i)\\bconclusion\\b|\\bsummary\\b',          'conclusion'),",
    "    (r'(?i)\\bmethod(?:s|ology)?\\b',               'methods'),",
    "    (r'(?i)\\bresult(?:s)?\\b|\\bfinding(?:s)?\\b',  'results'),",
    "    (r'(?i)\\bdiscussion\\b',                       'discussion'),",
    "    (r'(?i)\\breference(?:s)?\\b|\\bbiblio\\b',      'references'),",
    "    (r'(?i)\\bappendix\\b|\\bannex\\b',              'appendix'),",
    "    (r'(?i)\\babstract\\b|\\bexecutive\\s+summary\\b','abstract'),",
    "]",
    "",
    "def detect_section(text: str) -> str:",
    "    sample = text[:300]",
    "    for pattern, label in SECTION_PATTERNS:",
    "        if re.search(pattern, sample):",
    "            return label",
    "    if re.search(r'^[A-Z][A-Z\\s]{8,50}$', sample.split('\\n')[0].strip()):",
    "        return 'header'",
    "    if re.search(r'^\\d+(\\.\\d+)*\\s+\\w', sample.strip()):",
    "        return 'numbered'",
    "    return 'body'",
    "",
    "def classify_topic(text: str) -> str:",
    "    lower = text.lower()",
    "    scores = {topic: 0 for topic in TOPIC_KEYWORDS}",
    "    for topic, keywords in TOPIC_KEYWORDS.items():",
    "        for kw in keywords:",
    "            scores[topic] += lower.count(kw)",
    "    best = max(scores, key=lambda t: scores[t])",
    "    return best if scores[best] > 0 else 'general'",
    "",
    "# Spot checks",
    'samples = [',
    '    "Introduction to climate science and global warming observations.",',
    '    "Ocean temperatures have risen significantly, salinity changes observed in coastal areas.",',
    '    "Carbon dioxide emissions from fossil fuels have increased greenhouse gas concentrations.",',
    '    "The methodology involved analysis of satellite data and ground-based observations.",',
    ']',
    'print("{:<55} {:>12}  {:>12}".format("Text", "Section", "Topic"))',
    'print("-" * 82)',
    'for s in samples:',
    '    print("{:<55} {:>12}  {:>12}".format(s[:54], detect_section(s), classify_topic(s)))',
)))

# ── Step 7 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 7 — Load PDF, Extract Per-Page Text & Build Tagged Chunks\n\n"
    "Key difference from earlier notebooks: we extract text **page by page** so\n"
    "each chunk knows its `page_num`. The chunker tracks which page each character\n"
    "falls on and carries that forward into the metadata payload.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
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
    "reader = pypdf.PdfReader(PDF_PATH)",
    "n_pages = len(reader.pages)",
    'print(f"PDF    : {PDF_PATH}")',
    'print(f"Pages  : {n_pages}")',
    "",
    "# Build tagged chunks: each chunk knows its page_num",
    "tagged_chunks: List[Dict] = []",
    "for page_idx, page in enumerate(reader.pages):",
    "    page_num  = page_idx + 1",
    "    page_text = page.extract_text() or ''",
    "    page_chunks = recursive_split(page_text, CHUNK_SIZE, CHUNK_OVERLAP)",
    "    for chunk in page_chunks:",
    "        tagged_chunks.append({",
    "            'text'     : chunk,",
    "            'page_num' : page_num,",
    "            'char_count': len(chunk),",
    "            'word_count': len(chunk.split()),",
    "            'has_numbers': bool(re.search(r'\\d+\\.?\\d*', chunk)),",
    "            'section'  : detect_section(chunk),",
    "            'topic'    : classify_topic(chunk),",
    "        })",
    "",
    "total = len(tagged_chunks)",
    'print(f"Chunks : {total}  |  avg {sum(c[\"char_count\"] for c in tagged_chunks)//total} chars")',
    "",
    "# Distribution summary",
    "sections = {}; topics = {}",
    "for c in tagged_chunks:",
    "    sections[c['section']] = sections.get(c['section'], 0) + 1",
    "    topics[c['topic']]     = topics.get(c['topic'], 0)     + 1",
    "",
    'print("\\nSection distribution:")',
    "for k, v in sorted(sections.items(), key=lambda x: -x[1]):",
    '    print(f"  {k:<18} {v:>4} chunks")',
    'print("\\nTopic distribution:")',
    "for k, v in sorted(topics.items(), key=lambda x: -x[1]):",
    '    print(f"  {k:<18} {v:>4} chunks")',
)))

# ── Step 8 ─────────────────────────────────────────────────────────────────────
cells.append(md("## Step 8 — Embed & Index with Rich Metadata Payload"))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    'print(f"Embedding {len(tagged_chunks)} chunks...")',
    "t0   = time.time()",
    "texts = [c['text'] for c in tagged_chunks]",
    "embs  = embed_batch(texts, label='[index]')",
    "",
    "docs = [",
    "    {",
    "        'text'     : tagged_chunks[i]['text'],",
    "        'embedding': embs[i],",
    "        'metadata' : {",
    "            'chunk_idx'  : i,",
    "            'page_num'   : tagged_chunks[i]['page_num'],",
    "            'char_count' : tagged_chunks[i]['char_count'],",
    "            'word_count' : tagged_chunks[i]['word_count'],",
    "            'has_numbers': tagged_chunks[i]['has_numbers'],",
    "            'section'    : tagged_chunks[i]['section'],",
    "            'topic'      : tagged_chunks[i]['topic'],",
    "            'source'     : 'climate.pdf',",
    "        }",
    "    }",
    "    for i in range(len(tagged_chunks))",
    "]",
    "",
    "vs.upsert(docs)",
    'print(f"Indexed {vs.count()} vectors in {time.time()-t0:.1f}s")',
    "",
    "# Inspect a sample point's payload",
    "sample = docs[len(docs)//3]",
    'print("\\nSample payload:")',
    "for k, v in sample['metadata'].items():",
    "    print(f'  {k:<14}: {v}')",
)))

# ── Step 9 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 9 — Strategy A: Exact-Match Filter\n\n"
    "Filter by a single categorical field (`section` or `topic`) before vector search.\n"
    "Only chunks matching the field value participate in the ranking.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def rag_filtered(",
    "    question: str,",
    "    query_filter=None,",
    "    label: str = 'filtered',",
    "    verbose: bool = True",
    ") -> Dict:",
    "    t0   = time.time()",
    "    qvec = embed_text(question)",
    "    hits = vs.search(qvec, top_k=TOP_K, query_filter=query_filter)",
    "    passages = [h['text'] for h in hits]",
    "    answer   = generate_answer(question, passages) if passages else 'No results after filtering.'",
    "    latency  = (time.time() - t0) * 1000",
    "    if verbose:",
    "        print(f'\\n[{label}]  Q: {question}')",
    "        print(f'  Hits: {len(hits)}  Latency: {latency:.0f}ms')",
    "        for h in hits:",
    "            m = h['metadata']",
    "            print(f'  [p{m.get(\"page_num\",\"?\"):>3}] sec={m.get(\"section\",\"?\"):<13}'",
    "                  f' topic={m.get(\"topic\",\"?\"):<13} score={h[\"score\"]:.4f}')",
    "        print(f'  Answer: {answer[:200]}')",
    "        print('-' * 70)",
    "    return {'question': question, 'answer': answer, 'hits': hits,",
    "            'n_hits': len(hits), 'latency_ms': latency, 'label': label}",
    "",
    "# Demo: exact match on section='body' vs section='introduction'",
    'test_q = "What is the main focus of this document?"',
    "",
    "# No filter baseline",
    "rag_filtered(test_q, query_filter=None, label='no_filter')",
    "",
    "# Filter: introduction section only",
    "f_intro = Filter(must=[FieldCondition(",
    "    key='metadata.section', match=MatchValue(value='introduction'))])",
    "rag_filtered(test_q, query_filter=f_intro, label='section=introduction')",
    "",
    "# Filter: emissions topic only",
    "f_emis = Filter(must=[FieldCondition(",
    "    key='metadata.topic', match=MatchValue(value='emissions'))])",
    'rag_filtered("What causes greenhouse gas increases?",',
    "             query_filter=f_emis, label='topic=emissions')",
)))

# ── Step 10 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 10 — Strategy B: Range Filter\n\n"
    "Filter chunks by numeric fields: `page_num`, `char_count`, or `word_count`.\n"
    "Useful for 'what does the first half of the document say' type queries.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "n_pages_total = len(reader.pages)",
    "mid_page      = n_pages_total // 2",
    'print(f"Total pages: {n_pages_total}  |  Midpoint: {mid_page}")',
    "",
    "# Early pages (first quarter)",
    "f_early = Filter(must=[FieldCondition(",
    "    key='metadata.page_num',",
    "    range=Range(gte=1, lte=max(1, n_pages_total // 4)))])",
    "",
    "# Late pages (last quarter)",
    "f_late = Filter(must=[FieldCondition(",
    "    key='metadata.page_num',",
    "    range=Range(gte=n_pages_total - n_pages_total // 4, lte=n_pages_total))])",
    "",
    "# Dense chunks only (many words — likely prose, not headers)",
    "f_dense = Filter(must=[FieldCondition(",
    "    key='metadata.word_count',",
    "    range=Range(gte=60))])",
    "",
    'demo_q = "What are the key observations and findings?"',
    "",
    "r_early = rag_filtered(demo_q, query_filter=f_early,",
    "                       label=f'pages 1-{n_pages_total//4}')",
    "r_late  = rag_filtered(demo_q, query_filter=f_late,",
    "                       label=f'pages {n_pages_total - n_pages_total//4}-{n_pages_total}')",
    "r_dense = rag_filtered(demo_q, query_filter=f_dense, label='word_count>=60')",
)))

# ── Step 11 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 11 — Strategy C: Multi-Condition Filter\n\n"
    "Combine conditions with `must` (AND) and `should` (OR):\n\n"
    "- `must` = ALL conditions must hold (logical AND)\n"
    "- `should` = AT LEAST ONE condition must hold (logical OR)\n"
    "- `must_not` = NONE of the conditions must hold (logical NOT)\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range",
    "",
    "# AND: topic=temperature AND has_numbers=True",
    "f_temp_numbers = Filter(must=[",
    "    FieldCondition(key='metadata.topic',       match=MatchValue(value='temperature')),",
    "    FieldCondition(key='metadata.has_numbers', match=MatchValue(value=True)),",
    "])",
    "",
    "# OR: topic in [emissions, atmosphere]  (single field, multiple values)",
    "f_air_quality = Filter(must=[",
    "    FieldCondition(key='metadata.topic',",
    "                   match=MatchAny(any=['emissions', 'atmosphere'])),",
    "])",
    "",
    "# AND + range: early pages AND data-rich chunks",
    "f_early_dense = Filter(must=[",
    "    FieldCondition(key='metadata.page_num',   range=Range(gte=1, lte=10)),",
    "    FieldCondition(key='metadata.word_count', range=Range(gte=50)),",
    "])",
    "",
    "# NOT: exclude references/appendix sections",
    "from qdrant_client.models import Filter as QFilter",
    "f_no_refs = QFilter(must_not=[",
    "    FieldCondition(key='metadata.section',",
    "                   match=MatchAny(any=['references', 'appendix'])),",
    "])",
    "",
    'q_temp = "What quantitative evidence exists for temperature changes?"',
    'q_air  = "How do atmospheric emissions affect climate?"',
    "",
    "rag_filtered(q_temp, query_filter=f_temp_numbers,  label='temp+has_numbers')",
    "rag_filtered(q_air,  query_filter=f_air_quality,   label='emissions OR atmosphere')",
    "rag_filtered(q_temp, query_filter=f_early_dense,   label='pages 1-10 AND dense')",
    "rag_filtered(q_air,  query_filter=f_no_refs,       label='NOT references/appendix')",
)))

# ── Step 12 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 12 — Strategy D: Dynamic Filter Extraction\n\n"
    "For production systems, users rarely specify filters explicitly — they just\n"
    "ask natural-language questions. Ask Claude to extract structured filter\n"
    "parameters from the query text, then build the `Filter` object automatically.\n\n"
    "```\n"
    'Input  : "What does the introduction say about ocean warming?"\n'
    "Output : {section: 'introduction', topic: 'ocean', page_gte: null, page_lte: null}\n"
    "```\n"
))

cells.append(code(L(
    "from typing import List, Dict, Optional",
    "",
    "FILTER_EXTRACT_PROMPT = '''Extract metadata filter parameters from the question below.",
    "Return ONLY a valid JSON object with these exact keys (use null for unspecified):",
    "  page_gte   : int or null   (minimum page number the user cares about)",
    "  page_lte   : int or null   (maximum page number the user cares about)",
    "  section    : str or null   (one of: introduction, conclusion, methods, results, discussion, references, appendix, abstract, body)",
    "  topic      : str or null   (one of: temperature, ocean, precipitation, emissions, policy, atmosphere, general)",
    "  has_numbers: bool or null  (true if user asks for quantitative/numeric data)",
    "",
    "Question: {question}",
    "",
    "JSON:'''",
    "",
    "def extract_filters(question: str) -> Dict:",
    "    prompt = FILTER_EXTRACT_PROMPT.format(question=question)",
    "    raw = str(Agent(",
    "        model=_model,",
    "        system_prompt='You extract JSON metadata. Return only valid JSON, no extra text.'",
    "    )(prompt)).strip()",
    "    # Parse JSON from response",
    "    match = re.search(r'\\{[^}]+\\}', raw, re.DOTALL)",
    "    if match:",
    "        try:",
    "            return json.loads(match.group())",
    "        except json.JSONDecodeError:",
    "            pass",
    "    return {'page_gte': None, 'page_lte': None, 'section': None,",
    "            'topic': None, 'has_numbers': None}",
    "",
    "def build_filter(params: Dict) -> Optional[Filter]:",
    "    must = []",
    "    if params.get('page_gte') or params.get('page_lte'):",
    "        must.append(FieldCondition(",
    "            key='metadata.page_num',",
    "            range=Range(",
    "                gte=params.get('page_gte') or 1,",
    "                lte=params.get('page_lte') or 9999",
    "            )))",
    "    if params.get('section'):",
    "        must.append(FieldCondition(",
    "            key='metadata.section',",
    "            match=MatchValue(value=params['section'])))",
    "    if params.get('topic'):",
    "        must.append(FieldCondition(",
    "            key='metadata.topic',",
    "            match=MatchValue(value=params['topic'])))",
    "    if params.get('has_numbers') is True:",
    "        must.append(FieldCondition(",
    "            key='metadata.has_numbers',",
    "            match=MatchValue(value=True)))",
    "    return Filter(must=must) if must else None",
    "",
    "def rag_dynamic(",
    "    question: str,",
    "    verbose: bool = True",
    ") -> Dict:",
    "    params = extract_filters(question)",
    "    flt    = build_filter(params)",
    "    if verbose:",
    "        print(f'\\nQ: {question}')",
    "        print(f'Extracted filters: {params}')",
    "        print(f'Filter built: {flt is not None}')",
    "    return rag_filtered(question, query_filter=flt,",
    "                        label='dynamic', verbose=verbose)",
    "",
    "dynamic_qs = [",
    '    "What does the introduction explain about climate observations?",',
    '    "Give me quantitative data about ocean temperature changes.",',
    '    "What does the conclusion say about future policy recommendations?",',
    "]",
    "for q in dynamic_qs:",
    "    rag_dynamic(q)",
)))

# ── Step 13 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 13 — Filtered vs Unfiltered Comparison\n\n"
    "Run the same questions with no filter, static filter, and dynamic filter.\n"
    "Compare: number of hits, average page spread, and keyword hit rate.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "eval_pairs = [",
    "    {",
    '        "question"   : "What are the observed changes in atmospheric temperature?",',
    '        "keywords"   : ["temperature","change","observed","atmospheric","warming"],',
    '        "static_filter": Filter(must=[FieldCondition(',
    "            key='metadata.topic', match=MatchValue(value='temperature'))]),",
    '        "filter_label" : "topic=temperature",',
    "    },",
    "    {",
    '        "question"   : "What are the conclusions about emissions reductions?",',
    '        "keywords"   : ["conclusion","emission","reduction","target","policy"],',
    '        "static_filter": Filter(must=[FieldCondition(',
    "            key='metadata.section', match=MatchValue(value='conclusion'))]),",
    '        "filter_label" : "section=conclusion",',
    "    },",
    "    {",
    '        "question"   : "What numerical data supports ocean warming trends?",',
    '        "keywords"   : ["ocean","warming","data","degree","temperature"],',
    '        "static_filter": Filter(must=[',
    "            FieldCondition(key='metadata.topic', match=MatchValue(value='ocean')),",
    "            FieldCondition(key='metadata.has_numbers', match=MatchValue(value=True)),",
    "        ]),",
    '        "filter_label" : "ocean+numbers",',
    "    },",
    "]",
    "",
    "print('{:<46} {:>9}  {:>9}  {:>9}  {:>9}'.format(",
    "    'Question', 'NoFilter', 'Static', 'Dynamic', 'Pg spread'))",
    "print('-' * 90)",
    "",
    "for ep in eval_pairs:",
    "    q   = ep['question']",
    "    kws = ep['keywords']",
    "    n   = len(kws)",
    "",
    "    r_none    = rag_filtered(q, query_filter=None,           label='none',    verbose=False)",
    "    r_static  = rag_filtered(q, query_filter=ep['static_filter'],",
    "                             label=ep['filter_label'],       verbose=False)",
    "    r_dynamic = rag_dynamic(q, verbose=False)",
    "",
    "    def kw_hits(r):",
    "        return sum(1 for kw in kws if kw in r['answer'].lower())",
    "",
    "    def page_spread(r):",
    "        pages = [h['metadata'].get('page_num', 0) for h in r['hits']]",
    "        return max(pages) - min(pages) if len(pages) > 1 else 0",
    "",
    "    print('{:<46} {:>4}/{} hits  {:>4}/{} hits  {:>4}/{} hits  {:>5}'.format(",
    "        q[:45],",
    "        kw_hits(r_none),    n,",
    "        kw_hits(r_static),  n,",
    "        kw_hits(r_dynamic), n,",
    "        page_spread(r_none)))",
    "",
    "print()",
    "print('Format: keyword_hits/total  Page spread = max_page - min_page across top-K hits')",
)))

# ── Step 14 summary ─────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 14 — Summary\n\n"
    "| Component | Implementation |\n"
    "|-----------|---------------|\n"
    "| PDF | `pypdf.PdfReader` — page-by-page extraction preserving `page_num` |\n"
    "| Metadata fields | `page_num`, `char_count`, `word_count`, `has_numbers`, `section`, `topic` |\n"
    "| Section detection | Regex patterns on first 300 chars of each chunk |\n"
    "| Topic classification | Keyword vote across 6 topic buckets |\n"
    "| Payload indexes | `create_payload_index()` per field — accelerates pre-filter |\n"
    "| Filter A | Exact match — `FieldCondition` + `MatchValue` |\n"
    "| Filter B | Range — `FieldCondition` + `Range(gte, lte)` |\n"
    "| Filter C | Multi-condition — `Filter(must=[...])` AND / `must_not` NOT |\n"
    "| Filter D | Dynamic — LLM extracts JSON params → `build_filter()` |\n"
    "| Vector DB | Qdrant Cloud → OpenSearch → in-memory |\n"
    "| LLM | AWS Strands Agent + Claude Sonnet 4.6 |\n\n"
    "## Filter strategy comparison\n\n"
    "| Strategy | Setup | Best for | Qdrant API |\n"
    "|----------|-------|----------|------------|\n"
    "| Exact match | Low | Category-scoped Q&A | `MatchValue` |\n"
    "| Range | Low | Page/length scoping | `Range(gte, lte)` |\n"
    "| Multi-condition | Medium | Complex scoping | `must` / `should` / `must_not` |\n"
    "| Dynamic | LLM call | Natural-language filters | JSON → `build_filter()` |\n\n"
    "**When to use metadata filtering:**\n"
    "- Documents with clear structure (sections, chapters, dates)\n"
    "- Multi-document corpora — filter by `doc_id` before vector search\n"
    "- Date-scoped questions (`published_after`, `year`)\n"
    "- Compliance / auditing — restrict to authoritative sections only\n\n"
    "### Next: **12 — Maximal Marginal Relevance (MMR)**\n"
))

cells.append(code(L(
    "# vs.delete_collection()  # uncomment to clean up",
    "print(f\"Collection '{COLLECTION_NAME}' in {vs._backend} — {vs.count()} vectors\")",
    'print("\\nDone. Give the go-ahead for notebook 12.")',
)))

# ── Write & validate ───────────────────────────────────────────────────────────
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
