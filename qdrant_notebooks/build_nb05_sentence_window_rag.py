"""Build 05_Sentence_Window_RAG.ipynb — full implementation with Mermaid flow diagram."""
import json, uuid, ast

PATH = r"C:\Users\Administrator\RAG\qdrant_notebooks\tier1_chunking_indexing\05_Sentence_Window_RAG.ipynb"
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
"# 05 — Sentence Window RAG\n"
"\n"
"> **Tier 1 | Chunking & Indexing Foundations**\n"
"\n"
"## What is Sentence Window RAG?\n"
"\n"
"The fundamental retrieval tension:\n"
"- **Index small** → precise match, but the LLM sees too little context.\n"
"- **Index large** → rich context, but noisy embeddings miss the exact sentence.\n"
"\n"
"**Sentence Window RAG** sidesteps this by separating the two concerns:\n"
"\n"
"| Stage | What happens |\n"
"|-------|--------------|\n"
"| **Index** | Embed every sentence individually (maximum precision) |\n"
"| **Retrieve** | Find matching sentences; then expand a *window* of ±W neighbours |\n"
"| **Generate** | Send the expanded window (rich context) to the LLM |\n"
"\n"
"The window size `W` is the single tuning knob:\n"
"- `W=0` → return only the matched sentence (flat small)\n"
"- `W=1` → matched sentence + 1 before + 1 after (3 sentences total)\n"
"- `W=2` → 5 sentences, `W=3` → 7 sentences …\n"
"\n"
"## Key difference from Parent-Child RAG (04)\n"
"Parent-Child splits text *structurally* (paragraph → sentence by chunk size).\n"
"Sentence Window expands *positionally* — it uses the sentence's absolute position\n"
"in the corpus to slice an arbitrary neighbourhood, regardless of chunk boundaries.\n"
))

# ── Mermaid flow diagram ───────────────────────────────────────────────────────
cells.append(md(
"## Flow Diagram\n"
"\n"
"The diagram below renders in VS Code's Jupyter renderer and JupyterLab ≥ 4.1.\n"
"If it shows as plain text, paste the code block into **https://mermaid.live** to view it.\n"
"\n"
"```mermaid\n"
"flowchart TD\n"
"    subgraph INDEXING [\"🔵  INDEXING  (run once)\"]\n"
"        PDF([\"📄 climate.pdf\"])\n"
"        PDF --> PAGES[\"Extract page text\\npypdf.PdfReader\"]\n"
"        PAGES --> SPLIT[\"Split into sentences\\nregex — each sentence\\ngets a position index\"]\n"
"        SPLIT --> CORPUS[(\"sentence_corpus\\n[ {text, pos, page} ]\")]\n"
"        CORPUS --> EMB[\"Embed each sentence\\nBedrock Titan V2\\n1024-dim\"]\n"
"        EMB --> QDRANT[(\"Qdrant collection\\nvector + payload\\n{text, position, page}\")]\n"
"    end\n"
"\n"
"    subgraph RETRIEVAL [\"🟢  RETRIEVAL  (per query)\"]\n"
"        Q([\"❓ User query\"])\n"
"        Q --> QEMB[\"Embed query\\nTitan V2\"]\n"
"        QEMB --> VSEARCH[\"Vector search\\ntop-K sentences\"]\n"
"        QDRANT --> VSEARCH\n"
"        VSEARCH --> HITS[\"Matched sentences\\n+ position idx\"]\n"
"        HITS --> WINDOW[\"Expand window\\nsentences\\n pos-W … pos+W\"]\n"
"        CORPUS --> WINDOW\n"
"        WINDOW --> DEDUP[\"Deduplicate &\\nmerge overlapping\\nwindows\"]\n"
"    end\n"
"\n"
"    subgraph GENERATION [\"🟠  GENERATION\"]\n"
"        DEDUP --> CTX[\"Context passages\\npassed to LLM\"]\n"
"        CTX --> LLM[\"Strands Agent\\nClaude Sonnet 4.6\"]\n"
"        LLM --> ANS([\"✅ Answer\"])\n"
"    end\n"
"\n"
"    style INDEXING  fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f\n"
"    style RETRIEVAL fill:#dcfce7,stroke:#16a34a,color:#14532d\n"
"    style GENERATION fill:#ffedd5,stroke:#f97316,color:#7c2d12\n"
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
    'COLLECTION_NAME = "sentence_window_rag_05"',
    "EMBEDDING_DIM   = 1024",
    "TOP_K           = 5    # number of sentence hits to retrieve",
    "",
    "# Window size — the key tuning knob",
    "# W=0 -> matched sentence only",
    "# W=1 -> sentence + 1 before + 1 after  (3 total)",
    "# W=2 -> sentence + 2 before + 2 after  (5 total)",
    "WINDOW_SIZE = 2",
    "",
    "# Minimum sentence length to keep (filters headers/page numbers)",
    "MIN_SENT_LEN = 15",
    "",
    "# PDF — absolute path",
    'PDF_PATH = r"' + PDF + '"',
    "",
    'print(f"Window size  : W={WINDOW_SIZE}  ({2*WINDOW_SIZE+1} sentences per hit)")',
    'print(f"Top-K hits   : {TOP_K}")',
    'print(f"Collection   : {COLLECTION_NAME}")',
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
    'test_emb = embed_text("sentence window retrieval test")',
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
    "## Step 6 — Load PDF & Build Sentence Corpus\n\n"
    "Every sentence is stored in `sentence_corpus` with its **absolute position index**.\n"
    "This position is what lets us expand the window at retrieval time.\n\n"
    "```\n"
    "sentence_corpus = [\n"
    "  { text: '...', position: 0, page_num: 1 },\n"
    "  { text: '...', position: 1, page_num: 1 },\n"
    "  ...\n"
    "]\n"
    "```\n"
    "The `position` is also stored in each Qdrant vector's payload,\n"
    "so we can look up neighbours directly from the corpus list.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def split_sentences(text: str) -> List[str]:",
    "    parts = re.split(r'(?<=[.!?])\\s+', text.replace('\\n', ' '))",
    "    return [p.strip() for p in parts if len(p.strip()) >= MIN_SENT_LEN]",
    "",
    "# Load PDF",
    "reader = pypdf.PdfReader(PDF_PATH)",
    'print(f"PDF    : {PDF_PATH}")',
    'print(f"Pages  : {len(reader.pages)}")',
    "",
    "sentence_corpus: List[Dict] = []",
    "for page_num, page in enumerate(reader.pages):",
    "    page_text = page.extract_text() or ''",
    "    for sent in split_sentences(page_text):",
    "        sentence_corpus.append({",
    "            'text'    : sent,",
    "            'position': len(sentence_corpus),   # absolute index",
    "            'page_num': page_num + 1,",
    "        })",
    "",
    'print(f"Total sentences : {len(sentence_corpus)}")',
    'print(f"Avg length      : {sum(len(s[\'text\']) for s in sentence_corpus)/len(sentence_corpus):.0f} chars")',
    "",
    "# Show a sample of 5 consecutive sentences to verify quality",
    'print("\\nSample sentences [10..14]:")',
    "for s in sentence_corpus[10:15]:",
    "    print(f\"  [{s['position']}] p{s['page_num']}: {s['text'][:90]}\")",
)))

# ── Step 7 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 7 — Embed Sentences & Index in Qdrant\n\n"
    "Each Qdrant point payload includes `position` so the retriever\n"
    "can slice `sentence_corpus[pos-W : pos+W+1]` directly.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    'print(f"Embedding {len(sentence_corpus)} sentences...")',
    "t0   = time.time()",
    "embs = embed_batch([s['text'] for s in sentence_corpus], label='[sentences]')",
    "",
    "docs = [",
    "    {",
    "        'text'     : sentence_corpus[i]['text'],",
    "        'embedding': embs[i],",
    "        'metadata' : {",
    "            'position': sentence_corpus[i]['position'],",
    "            'page_num': sentence_corpus[i]['page_num'],",
    "            'source'  : 'climate.pdf',",
    "        }",
    "    }",
    "    for i in range(len(sentence_corpus))",
    "]",
    "",
    "indexed = vs.upsert(docs)",
    'print(f"Indexed {indexed} sentences in {time.time()-t0:.1f}s")',
    'print(f"Total in Qdrant: {vs.count()}")',
)))

# ── Step 8 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 8 — Sentence Window Retriever\n\n"
    "```\n"
    "retrieve(question, window_size=W)\n"
    "  1. Embed question\n"
    "  2. Search Qdrant → top-K matched sentences + their position\n"
    "  3. For each hit: slice sentence_corpus[pos-W : pos+W+1]\n"
    "  4. Merge overlapping windows (if two hits are close, their windows overlap)\n"
    "  5. Join each window into a passage string\n"
    "  6. Return passages to LLM\n"
    "```\n\n"
    "**Overlap merging:** if two windows share any sentence positions, they are merged\n"
    "into one longer passage rather than sending duplicate text to the LLM.\n"
))

cells.append(code(L(
    "from typing import List, Dict, Set",
    "",
    "def get_window(position: int, window_size: int) -> List[Dict]:",
    "    start = max(0, position - window_size)",
    "    end   = min(len(sentence_corpus), position + window_size + 1)",
    "    return sentence_corpus[start:end]",
    "",
    "def merge_windows(windows: List[List[Dict]]) -> List[List[Dict]]:",
    "    \"\"\"Merge windows that share any sentence position.\"\"\"",
    "    if not windows:",
    "        return []",
    "    merged: List[List[Dict]] = []",
    "    for w in sorted(windows, key=lambda x: x[0]['position']):",
    "        pos_set: Set[int] = {s['position'] for s in w}",
    "        placed = False",
    "        for m in merged:",
    "            m_set = {s['position'] for s in m}",
    "            if pos_set & m_set:  # overlap",
    "                all_pos = {s['position'] for s in m + w}",
    "                merged[merged.index(m)] = [sentence_corpus[p]",
    "                                           for p in sorted(all_pos)]",
    "                placed = True",
    "                break",
    "        if not placed:",
    "            merged.append(w)",
    "    return merged",
    "",
    "def retrieve_with_window(",
    "    question: str,",
    "    window_size: int = WINDOW_SIZE,",
    "    top_k: int = TOP_K",
    ") -> Dict:",
    "    hits    = vs.search(embed_text(question), top_k=top_k)",
    "    windows = []",
    "    details = []",
    "    for h in hits:",
    "        pos = h['metadata'].get('position', 0)",
    "        w   = get_window(pos, window_size)",
    "        windows.append(w)",
    "        details.append({'matched_text': h['text'], 'position': pos, 'score': h['score']})",
    "",
    "    merged   = merge_windows(windows)",
    "    passages = [' '.join(s['text'] for s in w) for w in merged]",
    "",
    "    return {",
    "        'passages'     : passages,",
    "        'details'      : details,",
    "        'n_hits'       : len(hits),",
    "        'n_passages'   : len(passages),",
    "        'window_size'  : window_size,",
    "        'sents_per_hit': 2 * window_size + 1,",
    "    }",
    "",
    'print("retrieve_with_window() defined.")',
    'print(f"Window W={WINDOW_SIZE} returns up to {2*WINDOW_SIZE+1} sentences per hit.")',
)))

# ── Step 9 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 9 — RAG Queries"))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "def rag_query(question: str, window_size: int = WINDOW_SIZE, verbose: bool = True) -> Dict:",
    "    t0      = time.time()",
    "    result  = retrieve_with_window(question, window_size=window_size)",
    "    answer  = generate_answer(question, result['passages'])",
    "    latency = (time.time() - t0) * 1000",
    "",
    "    if verbose:",
    '        print(f"\\nQ: {question}")',
    '        print(f"A: {answer}")',
    "        print(f\"  W={window_size}  hits={result['n_hits']}  \",",
    "              f\"passages after merge={result['n_passages']}  \",",
    "              f\"{result['sents_per_hit']} sents/hit max\")",
    "        for i, d in enumerate(result['details'][:3], 1):",
    "            print(f\"  [hit {i}] pos={d['position']}  score={d['score']:.4f}  {d['matched_text'][:70]}...\")",
    '        print(f"  Latency: {latency:.0f}ms")',
    '        print("-" * 70)',
    "",
    "    return {'question': question, 'answer': answer, 'latency_ms': latency,",
    "            'n_passages': result['n_passages'], 'window_size': window_size}",
    "",
    "test_questions = [",
    '    "What is weather forecasting and why is it important?",',
    '    "What are the main methods used in weather analysis?",',
    '    "How does climatology differ from meteorology?",',
    '    "What factors influence weather patterns and climate?",',
    "]",
    "results_log = [rag_query(q) for q in test_questions]",
)))

# ── Step 10 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 10 — Window Size Comparison\n\n"
    "Run the same query with W=0, 1, 2, 3, 4 and compare:\n"
    "- Total context chars sent to LLM\n"
    "- Number of merged passages\n"
    "- Retrieval latency\n\n"
    "This lets you pick the right W for your latency/context budget.\n"
))

cells.append(code(L(
    "compare_q = 'What are the main methods used in weather analysis?'",
    'print(f"Q: {compare_q}")',
    "print()",
    "print('{:<6} {:>8}  {:>12}  {:>10}  {:>10}'.format(",
    "    'W', 'Sents', 'Ctx chars', 'Passages', 'Latency'))",
    "print('-' * 54)",
    "",
    "for w in [0, 1, 2, 3, 4]:",
    "    t0  = time.time()",
    "    res = retrieve_with_window(compare_q, window_size=w)",
    "    lat = (time.time() - t0) * 1000",
    "    total_chars = sum(len(p) for p in res['passages'])",
    "    print('{:<6} {:>8}  {:>12,}  {:>10}  {:>9.0f}ms'.format(",
    "        w, res['sents_per_hit'], total_chars, res['n_passages'], lat))",
)))

# ── Step 11 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 11 — Visualise a Window Expansion\n\n"
    "Pick one query, show the matched sentence,\n"
    "then highlight how the window expands around it.\n"
))

cells.append(code(L(
    "from typing import List, Dict",
    "",
    "demo_q = 'What are the main methods used in weather analysis?'",
    "demo_hit = vs.search(embed_text(demo_q), top_k=1)[0]",
    "demo_pos = demo_hit['metadata'].get('position', 0)",
    "",
    'print(f"Query   : {demo_q}")',
    'print(f"Best match at position {demo_pos}:")',
    "print(f\"  '{demo_hit['text'][:100]}...'\")",
    "print()",
    "",
    "for w in [0, 1, 2]:",
    "    window = get_window(demo_pos, w)",
    "    print(f'--- Window W={w} ({len(window)} sentences) ---')",
    "    for s in window:",
    "        marker = '>>> ' if s['position'] == demo_pos else '    '",
    "        print(f\"{marker}[{s['position']}] {s['text'][:90]}\")",
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
    "print('{:<55} {:>7} {:>10} {:>8}'.format('Question','KW Hit','Latency','Passages'))",
    "print('-' * 84)",
    "eval_log = []",
    "for case in eval_cases:",
    "    r    = rag_query(case['question'], verbose=False)",
    "    low  = r['answer'].lower()",
    "    hits = sum(1 for kw in case['keywords'] if kw in low)",
    "    n    = len(case['keywords'])",
    "    eval_log.append(r)",
    "    print('{:<55} {}/{} ({:.0f}%) {:>8.0f}ms {:>7}'.format(",
    "        case['question'][:54], hits, n, hits/n*100, r['latency_ms'], r['n_passages']))",
    "",
    "print()",
    "print('Avg latency   : {:.0f}ms'.format(sum(r['latency_ms'] for r in eval_log)/len(eval_log)))",
    "print('Total sentences indexed: {}'.format(vs.count()))",
    "print('Window size W={} ({} sents/hit)'.format(WINDOW_SIZE, 2*WINDOW_SIZE+1))",
)))

# ── Step 13 Summary ────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 13 — Summary\n\n"
    "| Component | Implementation |\n"
    "|-----------|---------------|\n"
    "| PDF | `pypdf.PdfReader` — absolute path in Step 2 |\n"
    "| Sentence splitter | Regex on `.!?` — no NLTK |\n"
    "| Index unit | Individual sentences + absolute `position` in payload |\n"
    "| Retrieval | Top-K sentence search |\n"
    "| Window expansion | `sentence_corpus[pos-W : pos+W+1]` |\n"
    "| Overlap merge | Windows sharing any position are merged into one passage |\n"
    "| Window size | `W=2` default → 5 sentences per hit |\n"
    "| Vector DB | Qdrant Cloud → OpenSearch → in-memory |\n"
    "| LLM | AWS Strands Agent + Claude Sonnet 4.6 |\n\n"
    "## Pattern comparison so far\n\n"
    "| Notebook | Index unit | Context unit | Key knob |\n"
    "|----------|-----------|-------------|----------|\n"
    "| 02 Semantic Chunking | Semantic chunk | Same chunk | breakpoint strategy |\n"
    "| 03 Hierarchical RAG | Child (~200 chars) | Parent (~1000 chars) | fixed 2 levels |\n"
    "| 04 Parent-Child RAG | Paragraph / sentence | Any ancestor level | search+return level |\n"
    "| **05 Sentence Window** | **Sentence** | **±W neighbours** | **window size W** |\n\n"
    "### Next: **06 — Contextual Retrieval**\n"
))

cells.append(code(L(
    "# vs.delete_collection()  # uncomment to clean up",
    "print(f\"Collection '{COLLECTION_NAME}' in {vs._backend} — {vs.count()} vectors\")",
    'print("\\nDone. Give the go-ahead for notebook 06.")',
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
