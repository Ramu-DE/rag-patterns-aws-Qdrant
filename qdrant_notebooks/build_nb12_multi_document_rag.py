"""Build 12_Multi_Document_RAG.ipynb — full implementation."""
import json, uuid, ast

PATH = r"C:\Users\Administrator\RAG\qdrant_notebooks\tier2_retrieval_quality\12_Multi_Document_RAG.ipynb"
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
"# 12 — Multi-Document RAG\n"
"\n"
"> **Tier 2 | Retrieval Quality**\n"
"\n"
"## The Problem\n"
"\n"
"Single-document RAG breaks down when users need to query across\n"
"a corpus of related documents:\n"
"\n"
"```\n"
"Corpus:\n"
"  doc_1 — climate.pdf            (climate science reference)\n"
"  doc_2 — weather_methods.txt    (forecasting & analysis techniques)\n"
"  doc_3 — climate_policy.txt     (policy targets & agreements)\n"
"\n"
"Q: \"Compare the scientific evidence for warming with the policy response.\"\n"
"   → needs passages from doc_1 AND doc_3\n"
"\n"
"Q: \"What forecasting methods are most accurate?\"\n"
"   → only doc_2 is relevant — doc_1/doc_3 would add noise\n"
"```\n"
"\n"
"## Multi-Document RAG Solution\n"
"\n"
"Index every document into a **single shared collection**, tagging each chunk\n"
"with `doc_id` and `doc_name`. At query time, choose a retrieval mode:\n"
"\n"
"| Mode | Description | Use when |\n"
"|------|-------------|----------|\n"
"| **Global** | Retrieve from all docs | Cross-corpus synthesis |\n"
"| **Scoped** | Restrict to one doc | Targeted Q&A |\n"
"| **Parallel** | Top-K per doc then merge | Balanced cross-doc comparison |\n"
"\n"
"Every retrieved chunk carries source attribution (`doc_name`, `page_num`)\n"
"which the LLM can cite in its answer.\n"
))

# ── Mermaid ────────────────────────────────────────────────────────────────────
cells.append(md(
"## Flow Diagram\n"
"\n"
"```mermaid\n"
"flowchart TD\n"
"    subgraph CORPUS [\"🔵  CORPUS — multiple documents\"]\n"
"        D1([\"📄 climate.pdf\"])\n"
"        D2([\"📄 weather_methods.txt\"])\n"
"        D3([\"📄 climate_policy.txt\"])\n"
"    end\n"
"\n"
"    subgraph INDEX [\"⚙️  INDEXING — shared collection\"]\n"
"        D1 --> C1[\"Chunks\\ndoc_id=doc_1\"]\n"
"        D2 --> C2[\"Chunks\\ndoc_id=doc_2\"]\n"
"        D3 --> C3[\"Chunks\\ndoc_id=doc_3\"]\n"
"        C1 --> EMB[\"Embed — Titan V2\"]\n"
"        C2 --> EMB\n"
"        C3 --> EMB\n"
"        EMB --> COL[(\"Qdrant\\nsingle collection\")]\n"
"    end\n"
"\n"
"    subgraph MODES [\"⚡  RETRIEVAL MODES\"]\n"
"        direction LR\n"
"        M1[\"🌐 Global\\nno filter → all docs\"]\n"
"        M2[\"🎯 Scoped\\nfilter doc_id=doc_2\"]\n"
"        M3[\"⚖️ Parallel\\ntop-K per doc → merge\"]\n"
"    end\n"
"\n"
"    subgraph GEN [\"🟠  ATTRIBUTED GENERATION\"]\n"
"        HITS([\"Ranked passages\\n+ doc_name, page_num\"])\n"
"        HITS --> LLM[\"Strands Agent\\nClaude Sonnet 4.6\"]\n"
"        LLM --> ANS([\"✅ Answer with citations\"])\n"
"    end\n"
"\n"
"    COL --> MODES\n"
"    MODES --> HITS\n"
"\n"
"    style CORPUS fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f\n"
"    style INDEX  fill:#dcfce7,stroke:#16a34a,color:#14532d\n"
"    style MODES  fill:#fef9c3,stroke:#ca8a04,color:#713f12\n"
"    style GEN    fill:#ffedd5,stroke:#f97316,color:#7c2d12\n"
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
    "from typing import List, Dict, Optional",
    "",
    "import boto3, pypdf",
    "from strands import Agent",
    "from strands.models.bedrock import BedrockModel",
    "from qdrant_client import QdrantClient",
    "from qdrant_client.models import (",
    "    Distance, VectorParams, PointStruct,",
    "    Filter, FieldCondition, MatchValue, MatchAny",
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
    "# Collection — one collection holds all documents",
    'COLLECTION_NAME = "multi_document_rag_12"',
    "EMBEDDING_DIM   = 1024",
    "TOP_K           = 5    # per retrieval mode",
    "TOP_K_PER_DOC   = 3    # for parallel mode",
    "",
    "# Chunking",
    "CHUNK_SIZE    = 500",
    "CHUNK_OVERLAP = 50",
    "",
    "# PDF (document 1)",
    'PDF_PATH = r"' + PDF + '"',
    "",
    'print(f"Collection : {COLLECTION_NAME}")',
    'print(f"PDF path   : {PDF_PATH}")',
    'print(f"PDF exists : {os.path.exists(PDF_PATH)}")',
)))

# ── Step 3 ─────────────────────────────────────────────────────────────────────
cells.append(md("## Step 3 — Vector Store"))

cells.append(code(L(
    "from typing import List, Dict, Optional",
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
    "    def search(",
    "        self,",
    "        qvec: List[float],",
    "        top_k: int = 5,",
    "        query_filter=None",
    "    ) -> List[Dict]:",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            resp = self._qdrant.query_points(",
    "                collection_name=self.name, query=qvec, limit=top_k,",
    "                query_filter=query_filter, with_payload=True)",
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
    "    def create_payload_indexes(self):",
    "        if self._backend not in ('qdrant_cloud', 'qdrant_memory'):",
    "            return",
    "        for field, schema in [('metadata.doc_id', 'keyword'),",
    "                               ('metadata.doc_name', 'keyword'),",
    "                               ('metadata.page_num', 'integer')]:",
    "            self._qdrant.create_payload_index(",
    "                collection_name=self.name,",
    "                field_name=field,",
    "                field_schema=schema",
    "            )",
    "        print('Payload indexes created: doc_id, doc_name, page_num')",
    "",
    "    def count(self, doc_id: Optional[str] = None) -> int:",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            if doc_id:",
    "                flt = Filter(must=[FieldCondition(",
    "                    key='metadata.doc_id', match=MatchValue(value=doc_id))])",
    "                # scroll() is universally supported; count_filter can fail on Cloud",
    "                total = 0",
    "                offset = None",
    "                while True:",
    "                    pts, offset = self._qdrant.scroll(",
    "                        collection_name=self.name,",
    "                        scroll_filter=flt,",
    "                        limit=1000,",
    "                        offset=offset,",
    "                        with_payload=False,",
    "                        with_vectors=False,",
    "                    )",
    "                    total += len(pts)",
    "                    if offset is None:",
    "                        break",
    "                return total",
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
    "        'Use ONLY the context below. Cite the source label (e.g. [doc_1, p.3]) when referencing. '",
    "        \"If not found say 'Not found in context.'\\n\\n\"",
    "        f'Context:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:'",
    "    )",
    "    return str(Agent(",
    "        model=_model,",
    "        system_prompt='You are a precise Q&A assistant. Always cite which document each fact comes from.'",
    "    )(prompt))",
    "",
    "def generate_attributed(question: str, hits: List[Dict]) -> str:",
    "    passages = []",
    "    for h in hits:",
    "        m = h['metadata']",
    "        src = f\"[{m.get('doc_id','?')}, p.{m.get('page_num','?')}]\"",
    "        passages.append(f'{src}\\n{h[\"text\"]}')",
    "    return generate_answer(question, passages)",
    "",
    'test_emb = embed_text("multi document retrieval")',
    'print(f"Embedding OK — dim={len(test_emb)}")',
    'print("BedrockModel ready.")',
)))

# ── Step 5 ─────────────────────────────────────────────────────────────────────
cells.append(md("## Step 5 — Connect & Create Shared Collection"))

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
    "vs.create_payload_indexes()",
)))

# ── Step 6 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 6 — Define the Document Corpus\n\n"
    "This notebook uses three documents in one collection:\n\n"
    "| ID | Name | Source |\n"
    "|----|------|--------|\n"
    "| `doc_1` | `climate.pdf` | PDF on disk |\n"
    "| `doc_2` | `weather_methods.txt` | Inline text (forecasting techniques) |\n"
    "| `doc_3` | `climate_policy.txt` | Inline text (policy & agreements) |\n\n"
    "Documents 2 & 3 are representative inline texts so the notebook runs without\n"
    "additional files. Swap them for real PDFs by adding a `load_pdf()` call.\n"
))

cells.append(code(L(
    "DOC_WEATHER_METHODS = '''",
    "Weather Forecasting Methods and Techniques",
    "",
    "Synoptic meteorology analyses large-scale weather patterns using surface and upper-air",
    "observations. Meteorologists interpret pressure systems, fronts, and jet streams to",
    "forecast weather up to several days ahead. The synoptic scale covers areas of roughly",
    "1000 kilometres or more.",
    "",
    "Numerical Weather Prediction (NWP) uses mathematical models of the atmosphere to",
    "simulate future states. Modern NWP models discretise the atmosphere into a three-",
    "dimensional grid and solve equations governing fluid motion, thermodynamics, and",
    "moisture. Global models run at horizontal resolutions as fine as 9 km, while regional",
    "models can reach 1-3 km resolution for local forecasting.",
    "",
    "Ensemble forecasting runs multiple model simulations with slightly perturbed initial",
    "conditions to quantify forecast uncertainty. The spread of ensemble members indicates",
    "confidence: a tight cluster signals high confidence while a wide spread signals low.",
    "Modern ensemble systems use 50 or more members.",
    "",
    "Satellite imagery provides critical input for forecast models and real-time monitoring.",
    "Geostationary satellites capture visible, infrared, and water-vapour images every",
    "15-30 minutes. Polar-orbiting satellites provide global coverage with higher spatial",
    "resolution. Passive microwave and radar instruments retrieve temperature, humidity,",
    "wind, and precipitation profiles through the atmosphere.",
    "",
    "Machine learning is increasingly applied to post-processing NWP output, correcting",
    "systematic biases in model forecasts for temperature, wind speed, and precipitation.",
    "Deep learning models trained on decades of reanalysis data can produce skilful",
    "24-hour precipitation forecasts faster than physics-based models.",
    "",
    "Radar networks detect precipitation intensity and motion. Dual-polarisation radar",
    "distinguishes rain, snow, hail, and graupel. Doppler radar measures wind speed toward",
    "or away from the radar, enabling detection of rotation in severe thunderstorms.",
    "",
    "Verification metrics assess forecast accuracy. The mean absolute error (MAE) and",
    "root-mean-square error (RMSE) measure magnitude of error. The Brier score evaluates",
    "probabilistic forecasts of binary events such as precipitation occurrence.",
    "The equitable threat score (ETS) assesses categorical yes/no forecasts against chance.",
    "'''",
    "",
    "DOC_CLIMATE_POLICY = '''",
    "Climate Policy Frameworks and International Agreements",
    "",
    "The Paris Agreement, adopted in 2015 under the UN Framework Convention on Climate",
    "Change (UNFCCC), establishes a global framework to limit warming to well below 2",
    "degrees Celsius above pre-industrial levels, with efforts to limit it to 1.5 degrees.",
    "Parties submit Nationally Determined Contributions (NDCs) that outline their climate",
    "targets and policies every five years.",
    "",
    "Carbon pricing instruments include carbon taxes and emissions trading systems (ETS).",
    "A carbon tax sets a direct price on greenhouse gas emissions, providing a clear cost",
    "signal to emitters. Cap-and-trade systems set an overall emission limit, distribute or",
    "auction allowances, and allow trading. The EU Emissions Trading System is the world's",
    "largest carbon market, covering power generation, heavy industry, and aviation.",
    "",
    "Renewable energy targets form a central pillar of national climate strategies. The",
    "European Union has set a binding target of 42.5% renewable energy by 2030. China aims",
    "to reach peak carbon dioxide emissions before 2030 and carbon neutrality by 2060.",
    "The United States committed to a 50-52% reduction in greenhouse gas emissions below",
    "2005 levels by 2030 under its revised NDC.",
    "",
    "Adaptation policy addresses the unavoidable consequences of climate change already",
    "locked in. National Adaptation Plans (NAPs) assess vulnerabilities and identify",
    "priority measures in areas such as water security, coastal protection, agricultural",
    "resilience, and public health. The Global Goal on Adaptation, agreed at COP28, calls",
    "for enhanced action and means of implementation for developing nations.",
    "",
    "Loss and damage refers to the negative impacts of climate change that cannot be",
    "adapted to. The Santiago Network facilitates technical assistance for vulnerable",
    "nations. The Fund for Responding to Loss and Damage, operationalised at COP28,",
    "provides financial resources to countries suffering climate-related losses.",
    "",
    "Climate finance flows from developed to developing countries are central to the",
    "UNFCCC process. The New Collective Quantified Goal (NCQG) negotiated at COP29 set",
    "a target of 300 billion dollars per year in public climate finance to developing",
    "countries by 2035, with a broader goal of mobilising 1.3 trillion dollars annually.",
    "'''",
    "",
    "print(f'doc_2 weather_methods.txt  : {len(DOC_WEATHER_METHODS):>5} chars')",
    "print(f'doc_3 climate_policy.txt   : {len(DOC_CLIMATE_POLICY):>5} chars')",
)))

# ── Step 7 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 7 — Chunk, Tag & Index All Documents\n\n"
    "One helper function processes any document (PDF or plain text),\n"
    "chunks it, adds `doc_id` / `doc_name` metadata, and upserts into the\n"
    "shared collection.\n"
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
    "def index_document(",
    "    doc_id: str,",
    "    doc_name: str,",
    "    text: str,",
    "    page_map: Optional[List[int]] = None   # page_num per char offset",
    ") -> int:",
    "    chunks = recursive_split(text, CHUNK_SIZE, CHUNK_OVERLAP)",
    "    embs   = embed_batch(chunks, label=f'[{doc_id}]')",
    "    docs = []",
    "    for i, (chunk, emb) in enumerate(zip(chunks, embs)):",
    "        # Approximate page number: proportional char offset",
    "        char_offset = text.find(chunk[:40])",
    "        page_num = 1",
    "        if page_map and char_offset >= 0:",
    "            idx = min(char_offset, len(page_map) - 1)",
    "            page_num = page_map[idx]",
    "        docs.append({",
    "            'text'     : chunk,",
    "            'embedding': emb,",
    "            'metadata' : {",
    "                'doc_id'    : doc_id,",
    "                'doc_name'  : doc_name,",
    "                'chunk_idx' : i,",
    "                'page_num'  : page_num,",
    "                'char_count': len(chunk),",
    "            }",
    "        })",
    "    vs.upsert(docs)",
    "    return len(docs)",
    "",
    "# ── doc_1: climate.pdf ────────────────────────────────────────────────────",
    "print('Indexing doc_1: climate.pdf')",
    "reader    = pypdf.PdfReader(PDF_PATH)",
    "full_text = ''",
    "page_map  = []   # page_map[char_offset] = page_num",
    "for pg_idx, page in enumerate(reader.pages):",
    "    pg_text = (page.extract_text() or '') + '\\n\\n'",
    "    page_map.extend([pg_idx + 1] * len(pg_text))",
    "    full_text += pg_text",
    "n1 = index_document('doc_1', 'climate.pdf', full_text, page_map)",
    "print(f'  doc_1: {n1} chunks  ({len(reader.pages)} pages)')",
    "",
    "# ── doc_2: weather methods ────────────────────────────────────────────────",
    "print('Indexing doc_2: weather_methods.txt')",
    "n2 = index_document('doc_2', 'weather_methods.txt', DOC_WEATHER_METHODS)",
    "print(f'  doc_2: {n2} chunks')",
    "",
    "# ── doc_3: climate policy ─────────────────────────────────────────────────",
    "print('Indexing doc_3: climate_policy.txt')",
    "n3 = index_document('doc_3', 'climate_policy.txt', DOC_CLIMATE_POLICY)",
    "print(f'  doc_3: {n3} chunks')",
    "",
    "print(f'\\nTotal vectors in collection: {vs.count()}')",
)))

# ── Step 8 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 8 — Per-Document Index Stats\n\n"
    "Verify chunk counts per document using Qdrant's `count()` with a doc-filter.\n"
))

cells.append(code(L(
    "doc_registry = [",
    "    {'doc_id': 'doc_1', 'doc_name': 'climate.pdf'},",
    "    {'doc_id': 'doc_2', 'doc_name': 'weather_methods.txt'},",
    "    {'doc_id': 'doc_3', 'doc_name': 'climate_policy.txt'},",
    "]",
    "",
    'print("{:<8}  {:<28}  {:>8}".format("doc_id", "doc_name", "chunks"))',
    "print('-' * 50)",
    "for d in doc_registry:",
    "    n = vs.count(doc_id=d['doc_id'])",
    "    print('{:<8}  {:<28}  {:>8}'.format(d['doc_id'], d['doc_name'], n))",
    "print('-' * 50)",
    "print('{:<8}  {:<28}  {:>8}'.format('TOTAL', '', vs.count()))",
)))

# ── Step 9 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 9 — Mode A: Global Retrieval (all documents)\n\n"
    "No filter — the vector search runs across every document in the collection.\n"
    "Result set is sorted purely by cosine similarity; the most relevant chunks\n"
    "from any document win.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def rag_global(question: str, verbose: bool = True) -> Dict:",
    "    t0   = time.time()",
    "    qvec = embed_text(question)",
    "    hits = vs.search(qvec, top_k=TOP_K, query_filter=None)",
    "    answer   = generate_attributed(question, hits)",
    "    latency  = (time.time() - t0) * 1000",
    "    if verbose:",
    "        print(f'\\n[GLOBAL]  Q: {question}')",
    "        print(f'  Hits: {len(hits)}  Latency: {latency:.0f}ms')",
    "        for h in hits:",
    "            m = h['metadata']",
    "            print(f'  [{m[\"doc_id\"]}, p.{m[\"page_num\"]:>3}]  score={h[\"score\"]:.4f}  {h[\"text\"][:60]}...')",
    "        print(f'  Answer: {answer[:300]}')",
    "        print('-' * 70)",
    "    return {'question': question, 'answer': answer,",
    "            'hits': hits, 'latency_ms': latency, 'mode': 'global'}",
    "",
    "global_qs = [",
    '    "What is the relationship between greenhouse gas emissions and temperature rise?",',
    '    "What methods are used to observe and measure atmospheric conditions?",',
    '    "What international agreements address climate change?",',
    "]",
    "for q in global_qs:",
    "    rag_global(q)",
)))

# ── Step 10 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 10 — Mode B: Scoped Retrieval (single document)\n\n"
    "Apply a `doc_id` filter to restrict search to one document.\n"
    "Useful when the user targets a specific source.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def doc_filter(doc_id: str):",
    "    return Filter(must=[FieldCondition(",
    "        key='metadata.doc_id', match=MatchValue(value=doc_id))])",
    "",
    "def rag_scoped(question: str, doc_id: str, verbose: bool = True) -> Dict:",
    "    t0   = time.time()",
    "    qvec = embed_text(question)",
    "    hits = vs.search(qvec, top_k=TOP_K, query_filter=doc_filter(doc_id))",
    "    answer  = generate_attributed(question, hits) if hits else 'No results in this document.'",
    "    latency = (time.time() - t0) * 1000",
    "    if verbose:",
    "        print(f'\\n[SCOPED doc={doc_id}]  Q: {question}')",
    "        print(f'  Hits: {len(hits)}  Latency: {latency:.0f}ms')",
    "        for h in hits:",
    "            m = h['metadata']",
    "            print(f'  [{m[\"doc_id\"]}, p.{m[\"page_num\"]:>3}]  score={h[\"score\"]:.4f}  {h[\"text\"][:60]}...')",
    "        print(f'  Answer: {answer[:300]}')",
    "        print('-' * 70)",
    "    return {'question': question, 'answer': answer,",
    "            'hits': hits, 'latency_ms': latency, 'mode': f'scoped:{doc_id}'}",
    "",
    "# Same question scoped to different docs",
    'q = "What are the key observations about temperature change?"',
    "rag_scoped(q, 'doc_1')  # climate.pdf",
    "rag_scoped(q, 'doc_2')  # weather_methods",
    "rag_scoped(q, 'doc_3')  # policy",
)))

# ── Step 11 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 11 — Mode C: Parallel Retrieval (top-K per doc, then merge)\n\n"
    "Retrieve `TOP_K_PER_DOC` from each document independently, then merge and\n"
    "re-rank by score. Guarantees every document gets representation even when\n"
    "one document dominates the global score ranking.\n\n"
    "```\n"
    "doc_1 → [rank1 0.92, rank2 0.89, rank3 0.88]  ┐\n"
    "doc_2 → [rank1 0.84, rank2 0.81, rank3 0.79]  ├─ merge & sort → top-5\n"
    "doc_3 → [rank1 0.87, rank2 0.85, rank3 0.82]  ┘\n"
    "```\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def rag_parallel(",
    "    question: str,",
    "    doc_ids: Optional[List[str]] = None,",
    "    final_k: int = 5,",
    "    verbose: bool = True",
    ") -> Dict:",
    "    if doc_ids is None:",
    "        doc_ids = [d['doc_id'] for d in doc_registry]",
    "    t0   = time.time()",
    "    qvec = embed_text(question)",
    "    all_hits: List[Dict] = []",
    "    for did in doc_ids:",
    "        hits = vs.search(qvec, top_k=TOP_K_PER_DOC, query_filter=doc_filter(did))",
    "        all_hits.extend(hits)",
    "    # Re-rank merged set by score",
    "    all_hits.sort(key=lambda h: h['score'], reverse=True)",
    "    top_hits = all_hits[:final_k]",
    "    answer   = generate_attributed(question, top_hits)",
    "    latency  = (time.time() - t0) * 1000",
    "    if verbose:",
    "        doc_counts = {}",
    "        for h in top_hits:",
    "            did = h['metadata'].get('doc_id','?')",
    "            doc_counts[did] = doc_counts.get(did, 0) + 1",
    "        print(f'\\n[PARALLEL]  Q: {question}')",
    "        print(f'  Total retrieved: {len(all_hits)}  After merge: {len(top_hits)}  Latency: {latency:.0f}ms')",
    "        print(f'  Doc distribution in top-{final_k}: {doc_counts}')",
    "        for h in top_hits:",
    "            m = h['metadata']",
    "            print(f'  [{m[\"doc_id\"]}, p.{m[\"page_num\"]:>3}]  score={h[\"score\"]:.4f}  {h[\"text\"][:60]}...')",
    "        print(f'  Answer: {answer[:300]}')",
    "        print('-' * 70)",
    "    return {'question': question, 'answer': answer,",
    "            'hits': top_hits, 'latency_ms': latency, 'mode': 'parallel'}",
    "",
    'cross_q = "How do atmospheric observations inform both forecasting and climate policy?"',
    "rag_parallel(cross_q)",
    "",
    '# Two-doc comparison: only doc_1 + doc_3 (science + policy)',
    'comp_q = "What does the scientific evidence say and what do policymakers commit to?"',
    "rag_parallel(comp_q, doc_ids=['doc_1', 'doc_3'])",
)))

# ── Step 12 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 12 — Cross-Document Comparison Table\n\n"
    "Run the same questions across all three modes and compare\n"
    "which documents contributed to the answer and keyword coverage.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "compare_qs = [",
    "    {",
    '        "q"   : "What are greenhouse gas concentrations and their effects?",',
    '        "kws" : ["greenhouse","gas","carbon","emission","warming"],',
    "    },",
    "    {",
    '        "q"   : "How is weather predicted and what are the key tools?",',
    '        "kws" : ["forecast","model","satellite","radar","ensemble"],',
    "    },",
    "    {",
    '        "q"   : "What targets have been set to reduce climate impacts?",',
    '        "kws" : ["target","agreement","reduce","policy","carbon"],',
    "    },",
    "]",
    "",
    "def dominant_doc(hits: List[Dict]) -> str:",
    "    counts = {}",
    "    for h in hits:",
    "        d = h['metadata'].get('doc_id', '?')",
    "        counts[d] = counts.get(d, 0) + 1",
    "    return max(counts, key=lambda k: counts[k]) if counts else '?'",
    "",
    "def kw_hit(answer: str, kws: List[str]) -> str:",
    "    hits = sum(1 for kw in kws if kw in answer.lower())",
    "    return f'{hits}/{len(kws)}'",
    "",
    "print('{:<50}  {:>12}  {:>12}  {:>12}'.format('Question', 'Global', 'Scoped-best', 'Parallel'))",
    "print('{:<50}  {:>12}  {:>12}  {:>12}'.format('', 'KW hits', 'KW hits', 'KW hits'))",
    "print('-' * 92)",
    "",
    "for item in compare_qs:",
    "    q   = item['q']",
    "    kws = item['kws']",
    "",
    "    r_global = rag_global(q, verbose=False)",
    "",
    "    # Best scoped = doc with highest top-1 score",
    "    best_score, best_doc, best_ans = 0, '', ''",
    "    for d in [d['doc_id'] for d in doc_registry]:",
    "        r = rag_scoped(q, d, verbose=False)",
    "        if r['hits'] and r['hits'][0]['score'] > best_score:",
    "            best_score = r['hits'][0]['score']",
    "            best_doc   = d",
    "            best_ans   = r['answer']",
    "",
    "    r_parallel = rag_parallel(q, verbose=False)",
    "",
    "    print('{:<50}  {:>12}  {:>12}  {:>12}'.format(",
    "        q[:49],",
    "        kw_hit(r_global['answer'],   kws),",
    "        kw_hit(best_ans,             kws) + ' ' + best_doc,",
    "        kw_hit(r_parallel['answer'], kws)))",
)))

# ── Step 13 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 13 — Source Attribution in Answers\n\n"
    "Show that the LLM correctly cites different documents when passages\n"
    "come from multiple sources in the same context window.\n"
))

cells.append(code(L(
    "synthesis_q = (",
    "    'Summarise: (1) what the scientific literature says about warming trends, '",
    "    '(2) how forecasting models detect these trends, '",
    "    '(3) what policy commitments exist to address them.'",
    ")",
    "",
    'print(f"Q: {synthesis_q}")',
    'print("=" * 70)',
    "",
    "# Use parallel mode to guarantee all 3 docs contribute",
    "r = rag_parallel(synthesis_q, final_k=9, verbose=False)",
    "",
    'print("Sources retrieved:")',
    "for h in r['hits']:",
    "    m = h['metadata']",
    "    print(f\"  [{m['doc_id']}, p.{m['page_num']}]  {h['text'][:80]}...\")",
    "",
    'print(f"\\nAttributed answer:\\n{r[\"answer\"]}")',
)))

# ── Step 14 summary ─────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 14 — Summary\n\n"
    "| Component | Implementation |\n"
    "|-----------|---------------|\n"
    "| Corpus | 3 documents in a single Qdrant collection |\n"
    "| Doc tagging | `doc_id`, `doc_name`, `page_num` in every chunk payload |\n"
    "| Mode A — Global | No filter; pure cosine ranking across all docs |\n"
    "| Mode B — Scoped | `FieldCondition(doc_id=X)` restricts to one document |\n"
    "| Mode C — Parallel | Top-K per doc independently, then merge & re-rank |\n"
    "| Attribution | Source label `[doc_id, p.N]` prepended to each passage |\n"
    "| Vector DB | Qdrant Cloud → OpenSearch → in-memory |\n"
    "| LLM | AWS Strands Agent + Claude Sonnet 4.6 |\n\n"
    "## When to use each mode\n\n"
    "| Mode | Best for |\n"
    "|------|----------|\n"
    "| Global | Open-ended queries — let relevance decide the source |\n"
    "| Scoped | User knows which document holds the answer |\n"
    "| Parallel | Comparative/synthesis questions — must hear from every doc |\n\n"
    "## Scaling to large corpora\n\n"
    "- Add `doc_date`, `doc_type`, `author` to payloads for richer filtering\n"
    "- Use metadata filtering (nb 11) + doc_id filter together for precise scoping\n"
    "- For hundreds of documents, add a lightweight doc-routing step first:\n"
    "  embed the query and compare against per-doc summary embeddings to select\n"
    "  the 2-3 most relevant docs before running parallel retrieval\n\n"
    "### Next: **13 — Graph RAG**\n"
))

cells.append(code(L(
    "# vs.delete_collection()  # uncomment to clean up",
    "print(f\"Collection '{COLLECTION_NAME}' in {vs._backend} — {vs.count()} vectors\")",
    'print("\\nDone. Give the go-ahead for notebook 13.")',
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
