"""Build tier1_chunking_indexing/02_Semantic_Chunking.ipynb"""
import json, uuid, os

BASE = r"C:\Users\Administrator\RAG\qdrant_notebooks"
OUT  = os.path.join(BASE, "tier1_chunking_indexing", "02_Semantic_Chunking.ipynb")

def cid(): return str(uuid.uuid4())[:8]
def md(text):
    return {"cell_type":"markdown","id":cid(),"metadata":{},
            "source":text.splitlines(keepends=True)}
def code(lines):
    return {"cell_type":"code","id":cid(),"metadata":{},
            "execution_count":None,"outputs":[],"source":lines}
def src(*lines):
    out = [l+"\n" for l in lines]
    if out: out[-1] = out[-1].rstrip("\n")
    return out

cells = []

# ── Overview ──────────────────────────────────────────────────────────────────
cells.append(md(
"# 02 — Semantic Chunking\n"
"\n"
"> **Tier 1 | Chunking & Indexing Foundations**\n"
"\n"
"## What is Semantic Chunking?\n"
"Fixed-size splitting (notebook 01) cuts text at arbitrary character counts, often\n"
"mid-sentence or mid-concept. **Semantic Chunking** instead splits at *natural meaning\n"
"boundaries* — places where the embedding similarity between consecutive sentences\n"
"drops significantly, signalling a topic transition.\n"
"\n"
"## How it works\n"
"```\n"
"PDF text\n"
"    |\n"
"    v\n"
"Split into sentences\n"
"    |\n"
"    v\n"
"Embed each sentence  (Titan V2)\n"
"    |\n"
"    v\n"
"Compute cosine similarity between adjacent sentences\n"
"    |\n"
"    v\n"
"Find breakpoints where similarity < threshold\n"
"    |\n"
"    v\n"
"Merge sentences within each segment -> semantic chunks\n"
"    |\n"
"    v\n"
"Index in Qdrant, query as usual\n"
"```\n"
"\n"
"## Breakpoint strategies\n"
"| Strategy | How | Best for |\n"
"|----------|-----|----------|\n"
"| **Percentile** | Break at the bottom P% of similarities | Long docs with varied topic density |\n"
"| **Standard deviation** | Break where drop > mean - N*std | Uniform topic density |\n"
"| **Fixed threshold** | Break where similarity < T | Known domain, tunable T |\n"
"\n"
"This notebook implements all three and compares them.\n"
"\n"
"## Why it matters\n"
"| | Fixed-size | Semantic |\n"
"|--|-----------|----------|\n"
"| Chunk boundaries | Arbitrary | At topic transitions |\n"
"| Context preserved | Sometimes cut | Always intact |\n"
"| Chunk count | Predictable | Varies with content |\n"
"| Embedding calls | Once at index | Extra at index (sentence-level) |\n"
"\n"
"## Vector DB Strategy\n"
"Qdrant Cloud -> OpenSearch Serverless -> Qdrant in-memory (same as all notebooks).\n"
))

# ── Step 1 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 1 — Install & Import Dependencies"))

cells.append(code(src(
    "import subprocess, sys",
    "packages = [",
    '    "boto3",',
    '    "qdrant-client",',
    '    "opensearch-py",',
    '    "requests-aws4auth",',
    '    "strands-agents",',
    '    "pypdf",',
    "]",
    'subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + packages)',
    'print("All packages ready.")',
)))

cells.append(code(src(
    "import os, json, time, uuid, math",
    "from typing import List, Dict, Tuple",
    "",
    "import boto3",
    "from strands import Agent",
    "from strands.models.bedrock import BedrockModel",
    "from qdrant_client import QdrantClient",
    "from qdrant_client.models import Distance, VectorParams, PointStruct",
    "",
    'print("Imports OK")',
)))

# ── Step 2 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 2 — Configuration"))

cells.append(code(src(
    'AWS_REGION      = os.getenv("AWS_DEFAULT_REGION", "us-east-1")',
    'EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"',
    'LLM_MODEL       = "us.anthropic.claude-sonnet-4-6"',
    "",
    'QDRANT_URL      = os.getenv("QDRANT_URL", "")',
    'QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "")',
    'OPENSEARCH_URL  = os.getenv("OPENSEARCH_ENDPOINT", "")',
    "",
    'COLLECTION_NAME = "semantic_chunking_02"',
    "EMBEDDING_DIM   = 1024",
    "TOP_K           = 5",
    "",
    "# Semantic chunking parameters",
    "BREAKPOINT_STRATEGY  = 'percentile'  # 'percentile' | 'std_dev' | 'threshold'",
    "BREAKPOINT_PERCENTILE = 25           # bottom 25% of similarities -> break",
    "BREAKPOINT_STD_MULT   = 1.0          # break where drop > mean - N*std",
    "BREAKPOINT_THRESHOLD  = 0.4          # fixed cosine similarity floor",
    "MAX_CHUNK_SENTENCES   = 15           # cap segment size",
    "",
    'print(f"Region         : {AWS_REGION}")',
    'print(f"LLM model      : {LLM_MODEL}")',
    "print(f\"Qdrant URL     : {QDRANT_URL or '(not set -- in-memory)'}\")",
    'print(f"Collection     : {COLLECTION_NAME}")',
    'print(f"Strategy       : {BREAKPOINT_STRATEGY}")',
)))

# ── Step 3 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 3 — Vector DB Client (Qdrant -> OpenSearch fallback)\n\n"
    "Same self-contained `VectorStore` class as notebook 01.\n"
))

cells.append(code(src(
    "class VectorStore:",
    "    # Priority: Qdrant Cloud -> OpenSearch -> Qdrant in-memory",
    "",
    "    def __init__(self, collection_name, qdrant_url='', qdrant_api_key='',",
    "                 opensearch_url='', region='us-east-1'):",
    "        self.name = collection_name",
    "        self.region = region",
    "        self._backend = None",
    "        if qdrant_url:",
    "            try:",
    "                kw = {'url': qdrant_url}",
    "                if qdrant_api_key: kw['api_key'] = qdrant_api_key",
    "                self._qdrant = QdrantClient(**kw)",
    "                self._qdrant.get_collections()",
    "                self._backend = 'qdrant_cloud'",
    "                print(f'Connected to Qdrant Cloud: {qdrant_url}')",
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
    "                print(f'Connected to OpenSearch: {opensearch_url}')",
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
    "                print(f'Created collection \"{self.name}\" (dim={dim})')",
    "            else:",
    "                print(f'Collection \"{self.name}\" already exists')",
    "            return True",
    "        if self._backend == 'opensearch':",
    "            if force_recreate and self._os.indices.exists(index=self.name):",
    "                self._os.indices.delete(index=self.name)",
    "            if not self._os.indices.exists(index=self.name):",
    "                self._os.indices.create(index=self.name, body={",
    "                    'settings': {'index': {'knn': True}},",
    "                    'mappings': {'properties': {",
    "                        'text': {'type': 'text'}, 'metadata': {'type': 'object'},",
    "                        'embedding': {'type': 'knn_vector', 'dimension': dim,",
    "                            'method': {'name': 'hnsw', 'space_type': 'cosinesimil',",
    "                                       'engine': 'faiss',",
    "                                       'parameters': {'ef_construction': 512, 'm': 16}}}}}})",
    "                print(f'Created OpenSearch index \"{self.name}\"')",
    "            return True",
    "",
    "    def upsert(self, docs: List[Dict]) -> int:",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            pts = [PointStruct(id=str(uuid.uuid4()), vector=d['embedding'],",
    "                               payload={'text': d['text'], 'metadata': d.get('metadata',{})})",
    "                   for d in docs]",
    "            self._qdrant.upsert(collection_name=self.name, points=pts)",
    "            return len(pts)",
    "        if self._backend == 'opensearch':",
    "            for d in docs: self._os.index(index=self.name, body=d)",
    "            time.sleep(1); return len(docs)",
    "",
    "    def search(self, qvec: List[float], top_k: int = 5) -> List[Dict]:",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            resp = self._qdrant.query_points(",
    "                collection_name=self.name, query=qvec, limit=top_k, with_payload=True)",
    "            return [{'text': p.payload.get('text',''),",
    "                     'metadata': p.payload.get('metadata',{}),",
    "                     'score': p.score, 'id': str(p.id)} for p in resp.points]",
    "        if self._backend == 'opensearch':",
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
    "    embeddings = []",
    "    for i, t in enumerate(texts):",
    "        embeddings.append(embed_text(t))",
    "        if (i+1) % 20 == 0:",
    "            print(f'  {label} Embedded {i+1}/{len(texts)}')",
    "        time.sleep(0.04)",
    "    return embeddings",
    "",
    "_model = BedrockModel(model_id=LLM_MODEL, region_name=AWS_REGION)",
    "",
    "def generate_answer(question: str, context_chunks: List[str]) -> str:",
    '    "Generate grounded answer via Strands Agent."',
    "    context = '\\n\\n'.join(",
    "        f'[Chunk {i+1}]\\n{c}' for i, c in enumerate(context_chunks))",
    "    prompt = (",
    "        f'Use ONLY the context below to answer the question. '",
    "        f\"If the answer is not in the context, say 'Not found in context.'\\n\\n\"",
    "        f'Context:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:'",
    "    )",
    "    return str(Agent(model=_model,",
    "        system_prompt='You are a precise Q&A assistant. Answer only from the provided context.'",
    "    )(prompt))",
    "",
    'test_emb = embed_text("climate")',
    'print(f"Embedding OK -- dim={len(test_emb)}")',
    'print("Strands BedrockModel ready.")',
)))

# ── Step 5 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 5 — Connect & Create Collection"))

cells.append(code(src(
    "vs = VectorStore(",
    "    collection_name=COLLECTION_NAME,",
    "    qdrant_url=QDRANT_URL, qdrant_api_key=QDRANT_API_KEY,",
    "    opensearch_url=OPENSEARCH_URL, region=AWS_REGION",
    ")",
    'print(f"Active backend: {vs._backend}")',
    "vs.create_collection(dim=EMBEDDING_DIM, force_recreate=True)",
)))

# ── Step 6 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 6 — Load PDF & Split into Sentences\n\n"
    "We use `pypdf` to extract text, then split into individual sentences using a\n"
    "simple regex-based splitter (no NLTK dependency).\n"
    "Each sentence keeps track of its source page.\n"
))

cells.append(code(src(
    "import os, re",
    "import pypdf",
    "",
    "def split_sentences(text: str) -> List[str]:",
    '    "Split text into sentences on .  !  ? boundaries."',
    "    # Split on sentence-ending punctuation followed by space/newline",
    "    parts = re.split(r'(?<=[.!?])\\s+', text.replace('\\n', ' '))",
    "    return [p.strip() for p in parts if len(p.strip()) > 10]",
    "",
    'PDF_PATH = os.path.join("..", "data", "climate.pdf")',
    "reader   = pypdf.PdfReader(PDF_PATH)",
    'print(f"PDF pages: {len(reader.pages)}")',
    "",
    "# Extract sentence objects: {text, page_num}",
    "all_sentences: List[Dict] = []",
    "for page_num, page in enumerate(reader.pages):",
    "    page_text = page.extract_text() or ''",
    "    for sent in split_sentences(page_text):",
    "        all_sentences.append({'text': sent, 'page_num': page_num + 1})",
    "",
    'print(f"Total sentences : {len(all_sentences)}")',
    "print(f\"Avg sentence len: {sum(len(s['text']) for s in all_sentences)/len(all_sentences):.0f} chars\")",
    'print(f"\\nSample sentence 10:")',
    "print(all_sentences[10]['text'])",
)))

# ── Step 7 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 7 — Embed All Sentences\n\n"
    "Each sentence gets its own Titan V2 embedding.\n"
    "These embeddings are used **only** to compute similarity between adjacent\n"
    "sentences to find breakpoints — they are NOT stored in Qdrant.\n"
    "The final chunk embeddings are computed separately in Step 9.\n"
))

cells.append(code(src(
    'print(f"Embedding {len(all_sentences)} sentences for similarity analysis...")',
    "t0 = time.time()",
    "sentence_texts = [s['text'] for s in all_sentences]",
    "sentence_embeddings = embed_batch(sentence_texts, label='[sentences]')",
    'print(f"Done in {time.time()-t0:.1f}s")',
)))

# ── Step 8 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 8 — Compute Adjacent Similarities & Find Breakpoints\n\n"
    "For each pair of consecutive sentences, compute their cosine similarity.\n"
    "Low similarity = topic shift = potential chunk boundary.\n\n"
    "We implement all three breakpoint strategies so you can compare them.\n"
))

cells.append(code(src(
    "def cosine_similarity(a: List[float], b: List[float]) -> float:",
    '    "Cosine similarity between two already-normalised vectors (dot product)."',
    "    return sum(x*y for x,y in zip(a,b))",
    "",
    "# Compute similarity between each adjacent pair",
    "similarities: List[float] = []",
    "for i in range(len(sentence_embeddings) - 1):",
    "    sim = cosine_similarity(sentence_embeddings[i], sentence_embeddings[i+1])",
    "    similarities.append(sim)",
    "",
    'print(f"Adjacent similarities computed: {len(similarities)}")',
    'print(f"  min  : {min(similarities):.4f}")',
    'print(f"  max  : {max(similarities):.4f}")',
    'print(f"  mean : {sum(similarities)/len(similarities):.4f}")',
    "",
    "# --- Strategy 1: percentile ---",
    "def breakpoints_percentile(sims: List[float], pct: float = 25) -> List[int]:",
    '    "Break where similarity is in the bottom pct% of all similarities."',
    "    sorted_sims = sorted(sims)",
    "    idx = max(0, int(len(sorted_sims) * pct / 100) - 1)",
    "    threshold = sorted_sims[idx]",
    "    return [i for i, s in enumerate(sims) if s <= threshold]",
    "",
    "# --- Strategy 2: standard deviation ---",
    "def breakpoints_std_dev(sims: List[float], multiplier: float = 1.0) -> List[int]:",
    '    "Break where similarity drops more than multiplier * std below the mean."',
    "    mean = sum(sims) / len(sims)",
    "    std  = math.sqrt(sum((s-mean)**2 for s in sims) / len(sims))",
    "    threshold = mean - multiplier * std",
    "    return [i for i, s in enumerate(sims) if s < threshold]",
    "",
    "# --- Strategy 3: fixed threshold ---",
    "def breakpoints_threshold(sims: List[float], threshold: float = 0.4) -> List[int]:",
    '    "Break where similarity falls below a fixed threshold."',
    "    return [i for i, s in enumerate(sims) if s < threshold]",
    "",
    "bp_pct   = breakpoints_percentile(similarities, BREAKPOINT_PERCENTILE)",
    "bp_std   = breakpoints_std_dev(similarities, BREAKPOINT_STD_MULT)",
    "bp_fixed = breakpoints_threshold(similarities, BREAKPOINT_THRESHOLD)",
    "",
    'print(f"\\nBreakpoints found:")',
    'print(f"  Percentile ({BREAKPOINT_PERCENTILE}%)  : {len(bp_pct)}")',
    'print(f"  Std dev    (x{BREAKPOINT_STD_MULT})   : {len(bp_std)}")',
    'print(f"  Fixed      (<{BREAKPOINT_THRESHOLD}) : {len(bp_fixed)}")',
)))

# ── Step 9 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 9 — Build Semantic Chunks\n\n"
    "Merge consecutive sentences between breakpoints into chunks.\n"
    "Respect `MAX_CHUNK_SENTENCES` to avoid oversized chunks.\n"
    "Then embed each final chunk with Titan V2 and index in Qdrant.\n"
))

cells.append(code(src(
    "def build_chunks(sentences: List[Dict], breakpoint_indices: List[int],",
    "                 max_sentences: int = MAX_CHUNK_SENTENCES) -> List[Dict]:",
    '    "Merge sentences into chunks, splitting at breakpoints and max-sentence cap."',
    "    chunks = []",
    "    current = []",
    "    bp_set  = set(breakpoint_indices)",
    "",
    "    for i, sent in enumerate(sentences):",
    "        current.append(sent)",
    "        at_break = (i in bp_set) or (len(current) >= max_sentences)",
    "        at_end   = (i == len(sentences) - 1)",
    "        if at_break or at_end:",
    "            chunk_text = ' '.join(s['text'] for s in current).strip()",
    "            if chunk_text:",
    "                chunks.append({",
    "                    'text'      : chunk_text,",
    "                    'page_num'  : current[0]['page_num'],",
    "                    'page_end'  : current[-1]['page_num'],",
    "                    'n_sentences': len(current),",
    "                    'source'    : 'climate.pdf',",
    "                })",
    "            current = []",
    "    return chunks",
    "",
    "# Choose active strategy",
    "if BREAKPOINT_STRATEGY == 'percentile':",
    "    active_bp = bp_pct",
    "elif BREAKPOINT_STRATEGY == 'std_dev':",
    "    active_bp = bp_std",
    "else:",
    "    active_bp = bp_fixed",
    "",
    "chunks = build_chunks(all_sentences, active_bp, MAX_CHUNK_SENTENCES)",
    'print(f"Semantic chunks created  : {len(chunks)}")',
    "print(f\"Avg chunk length         : {sum(len(c['text']) for c in chunks)/len(chunks):.0f} chars\")",
    "print(f\"Avg sentences per chunk  : {sum(c['n_sentences'] for c in chunks)/len(chunks):.1f}\")",
    'print(f"\\nSample chunk 0:")',
    "print(f\"  sentences={chunks[0]['n_sentences']}  pages={chunks[0]['page_num']}-{chunks[0]['page_end']}\")",
    "print(f\"  {chunks[0]['text'][:250]}...\")",
)))

# ── Step 10 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 10 — Embed Chunks & Index in Qdrant"))

cells.append(code(src(
    'print(f"Embedding {len(chunks)} semantic chunks...")',
    "t0 = time.time()",
    "chunk_texts      = [c['text'] for c in chunks]",
    "chunk_embeddings = embed_batch(chunk_texts, label='[chunks]')",
    "",
    "docs_to_index = [",
    "    {",
    "        'text'     : chunks[i]['text'],",
    "        'embedding': chunk_embeddings[i],",
    "        'metadata' : {",
    "            'chunk_index' : i,",
    "            'page_num'    : chunks[i]['page_num'],",
    "            'page_end'    : chunks[i]['page_end'],",
    "            'n_sentences' : chunks[i]['n_sentences'],",
    "            'source'      : chunks[i]['source'],",
    "            'strategy'    : BREAKPOINT_STRATEGY,",
    "        }",
    "    }",
    "    for i in range(len(chunks))",
    "]",
    "indexed = vs.upsert(docs_to_index)",
    "elapsed = time.time() - t0",
    'print(f"Indexed  : {indexed} chunks in {elapsed:.2f}s")',
    'print(f"Count in collection: {vs.count()}")',
)))

# ── Step 11 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 11 — RAG Query with Semantic Chunks\n\n"
    "Same retrieval-generation pipeline as notebook 01, now using semantic chunks.\n"
))

cells.append(code(src(
    "def rag_query(question: str, top_k: int = TOP_K, verbose: bool = True) -> Dict:",
    '    "Embed question, retrieve top-K semantic chunks, generate answer."',
    "    t0      = time.time()",
    "    q_emb   = embed_text(question)",
    "    results = vs.search(q_emb, top_k=top_k)",
    "    answer  = generate_answer(question, [r['text'] for r in results])",
    "    latency = (time.time() - t0) * 1000",
    "    if verbose:",
    '        print(f"\\nQuestion : {question}")',
    '        print(f"Latency  : {latency:.0f}ms")',
    '        print(f"\\nAnswer:\\n{answer}")',
    '        print(f"\\nTop sources:")',
    "        for i, r in enumerate(results[:3], 1):",
    "            pg  = r['metadata'].get('page_num','?')",
    "            ns  = r['metadata'].get('n_sentences','?')",
    "            print(f\"  [{i}] page={pg}  sentences={ns}  score={r['score']:.4f}  {r['text'][:90]}...\")",
    '        print("-" * 70)',
    "    return {'question': question, 'answer': answer,",
    "            'results': results, 'latency_ms': latency}",
    "",
    "test_questions = [",
    '    "What is weather forecasting and why is it important?",',
    '    "What are the main methods used in weather analysis?",',
    '    "How does climatology differ from meteorology?",',
    '    "What factors influence weather patterns and climate?",',
    "]",
    "results_log = []",
    "for q in test_questions:",
    "    results_log.append(rag_query(q))",
)))

# ── Step 12 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 12 — Strategy Comparison\n\n"
    "Build a separate Qdrant collection for each breakpoint strategy, index the same\n"
    "content, run the same query, and compare chunk counts and answer quality.\n"
))

cells.append(code(src(
    "compare_q    = 'What are the main methods used in weather analysis?'",
    "strategies   = [",
    "    ('percentile', bp_pct),",
    "    ('std_dev',    bp_std),",
    "    ('threshold',  bp_fixed),",
    "]",
    "",
    'print(f"Strategy comparison for: \"{compare_q}\"\\n")',
    'print(f"{\"Strategy\":<15} {\"Chunks\":>6}  {\"Avg len\":>8}  Sample top-1 answer snippet")',
    'print("-" * 80)',
    "",
    "for strat_name, bp in strategies:",
    "    s_chunks = build_chunks(all_sentences, bp, MAX_CHUNK_SENTENCES)",
    "    s_embs   = embed_batch([c['text'] for c in s_chunks], label=f'[{strat_name}]')",
    "    # temp in-memory store per strategy",
    "    tmp = VectorStore(f'tmp_{strat_name}')",
    "    tmp.create_collection(dim=EMBEDDING_DIM)",
    "    tmp.upsert([{'text': s_chunks[i]['text'], 'embedding': s_embs[i],",
    "                 'metadata': {'chunk_index': i}} for i in range(len(s_chunks))])",
    "    qe = embed_text(compare_q)",
    "    top = tmp.search(qe, top_k=1)",
    "    avg_len = sum(len(c['text']) for c in s_chunks) / len(s_chunks)",
    "    snippet = top[0]['text'][:55] if top else '(no results)'",
    "    print(f\"{strat_name:<15} {len(s_chunks):>6}  {avg_len:>8.0f}  {snippet}...\")",
)))

# ── Step 13 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 13 — Evaluation & Metrics"))

cells.append(code(src(
    "eval_cases = [",
    "    {'question': 'What is weather forecasting and why is it important?',",
    "     'expected_keywords': ['forecast', 'weather', 'predict', 'atmosphere', 'climate']},",
    "    {'question': 'What are the main methods used in weather analysis?',",
    "     'expected_keywords': ['analysis', 'synoptic', 'observation', 'data', 'pressure']},",
    "    {'question': 'How does climatology differ from meteorology?',",
    "     'expected_keywords': ['climate', 'weather', 'long', 'study', 'atmosphere']},",
    "]",
    "",
    "print(f\"{'Question':<55} {'KW Hit':>7} {'Latency':>10}\")",
    "print('-' * 75)",
    "eval_results = []",
    "for case in eval_cases:",
    "    r = rag_query(case['question'], verbose=False)",
    "    low   = r['answer'].lower()",
    "    hits  = sum(1 for kw in case['expected_keywords'] if kw in low)",
    "    total = len(case['expected_keywords'])",
    "    eval_results.append(r)",
    "    print(f\"{case['question'][:54]:<55} {hits}/{total} ({hits/total*100:.0f}%)  {r['latency_ms']:>7.0f}ms\")",
    "",
    "print()",
    "avg_lat = sum(r['latency_ms'] for r in eval_results) / len(eval_results)",
    'print(f"Average latency: {avg_lat:.0f}ms")',
    'print(f"Semantic chunks: {len(chunks)}  |  Avg size: {sum(len(c[chr(39)+chr(116)+chr(101)+chr(120)+chr(116)+chr(39)]) for c in chunks)/len(chunks):.0f} chars")',
)))

# fix the chr() hack
for cell in cells:
    if cell['cell_type'] == 'code':
        src_lines = cell['source']
        fixed = []
        for line in src_lines:
            if 'chr(' in line and "Semantic chunks" in line:
                line = line.replace(
                    "sum(len(c[chr(39)+chr(116)+chr(101)+chr(120)+chr(116)+chr(39)]) for c in chunks)",
                    "sum(len(c['text']) for c in chunks)"
                )
            fixed.append(line)
        cell['source'] = fixed

# ── Step 14 Summary ────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 14 — Summary\n\n"
    "### What we built\n"
    "| Component | Implementation |\n"
    "|-----------|---------------|\n"
    "| **PDF Loading** | `pypdf.PdfReader` |\n"
    "| **Sentence splitting** | Regex-based (no NLTK) |\n"
    "| **Sentence embeddings** | Amazon Bedrock Titan V2 (1024-dim) |\n"
    "| **Breakpoint detection** | Cosine similarity drop — percentile / std-dev / threshold |\n"
    "| **Chunk embeddings** | Amazon Bedrock Titan V2 (1024-dim) |\n"
    "| **Vector DB** | Qdrant Cloud -> OpenSearch -> in-memory |\n"
    "| **LLM** | AWS Strands Agent + Claude Sonnet 4.6 |\n\n"
    "### Semantic vs Fixed-Size chunking\n"
    "| | Fixed-size (01) | Semantic (02) |\n"
    "|--|----------------|---------------|\n"
    "| Boundary | Every N characters | Topic transitions |\n"
    "| Context preservation | Sometimes cut | Always intact |\n"
    "| Chunk count | Predictable | Content-driven |\n"
    "| Extra embedding calls | 0 | +N (sentences, index-time only) |\n\n"
    "### Three breakpoint strategies\n"
    "- **Percentile** — most robust; adapts to document's own similarity distribution\n"
    "- **Std dev** — good when document has consistent topic density\n"
    "- **Fixed threshold** — predictable but needs domain tuning\n\n"
    "### Next: **03 — Hierarchical RAG** (store small chunks, return large parent context)\n"
))

cells.append(code(src(
    "# Optional cleanup",
    "# vs.delete_collection()",
    "print(f\"Collection '{COLLECTION_NAME}' retained in {vs._backend}.\")",
    "print(f'Total vectors: {vs.count()}')",
    'print("\\nDone. Give the go-ahead to proceed to notebook 03.")',
)))

# ── Write ─────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"}
    },
    "cells": cells
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Written : {OUT}")
print(f"Cells   : {len(cells)}")
