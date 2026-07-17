"""Rebuild 02_Semantic_Chunking.ipynb from scratch with hardcoded absolute PDF path."""
import json, uuid, ast

path = r"C:\Users\Administrator\RAG\qdrant_notebooks\tier1_chunking_indexing\02_Semantic_Chunking.ipynb"
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
"# 02 — Semantic Chunking\n"
"\n"
"> **Tier 1 | Chunking & Indexing Foundations**\n"
"\n"
"## What is Semantic Chunking?\n"
"Fixed-size splitting cuts text at arbitrary character counts, often mid-sentence.\n"
"**Semantic Chunking** splits at *meaning boundaries* — where cosine similarity\n"
"between consecutive sentences drops, signalling a topic transition.\n"
"\n"
"## Pipeline\n"
"```\n"
"PDF -> sentences -> embed each sentence (Titan V2)\n"
"    -> cosine sim between adjacent sentences\n"
"    -> breakpoints where sim drops sharply\n"
"    -> merge sentences into semantic chunks\n"
"    -> embed chunks -> index in Qdrant -> query\n"
"```\n"
"\n"
"## Three breakpoint strategies\n"
"| Strategy | Rule |\n"
"|----------|------|\n"
"| **Percentile** | Break where sim is in bottom P% |\n"
"| **Std dev** | Break where sim < mean - N*std |\n"
"| **Fixed threshold** | Break where sim < T |\n"
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
    "import os, json, time, uuid, math, re",
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
    "# RAG parameters",
    'COLLECTION_NAME       = "semantic_chunking_02"',
    "EMBEDDING_DIM         = 1024",
    "TOP_K                 = 5",
    "",
    "# Semantic chunking",
    "BREAKPOINT_STRATEGY   = 'percentile'  # 'percentile' | 'std_dev' | 'threshold'",
    "BREAKPOINT_PERCENTILE = 25",
    "BREAKPOINT_STD_MULT   = 1.0",
    "BREAKPOINT_THRESHOLD  = 0.4",
    "MAX_CHUNK_SENTENCES   = 15",
    "",
    "# PDF — absolute path",
    'PDF_PATH = r"' + PDF + '"',
    "",
    'print(f"Region    : {AWS_REGION}")',
    'print(f"LLM       : {LLM_MODEL}")',
    'print(f"Collection: {COLLECTION_NAME}")',
    'print(f"Strategy  : {BREAKPOINT_STRATEGY}")',
    'print(f"PDF       : {PDF_PATH}")',
    'print(f"PDF exists: {os.path.exists(PDF_PATH)}")',
)))

# ── Step 3 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 3 — Vector DB Client (Qdrant -> OpenSearch fallback)"))

cells.append(code(L(
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
    "                if qdrant_api_key:",
    "                    kw['api_key'] = qdrant_api_key",
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
    "                self._qdrant.delete_collection(self.name)",
    "                exists = False",
    "            if not exists:",
    "                self._qdrant.create_collection(",
    "                    self.name,",
    "                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE))",
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
    "                        'text': {'type': 'text'},",
    "                        'metadata': {'type': 'object'},",
    "                        'embedding': {",
    "                            'type': 'knn_vector', 'dimension': dim,",
    "                            'method': {",
    "                                'name': 'hnsw', 'space_type': 'cosinesimil',",
    "                                'engine': 'faiss',",
    "                                'parameters': {'ef_construction': 512, 'm': 16}",
    "                            }",
    "                        }",
    "                    }}})",
    "                print(f'Created OpenSearch index \"{self.name}\"')",
    "            return True",
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
    "        if self._backend == 'opensearch':",
    "            for d in docs:",
    "                self._os.index(index=self.name, body=d)",
    "            time.sleep(1)",
    "            return len(docs)",
    "",
    "    def search(self, qvec: List[float], top_k: int = 5) -> List[Dict]:",
    "        if self._backend in ('qdrant_cloud', 'qdrant_memory'):",
    "            resp = self._qdrant.query_points(",
    "                collection_name=self.name, query=qvec, limit=top_k, with_payload=True)",
    "            return [",
    "                {'text': p.payload.get('text', ''),",
    "                 'metadata': p.payload.get('metadata', {}),",
    "                 'score': p.score, 'id': str(p.id)}",
    "                for p in resp.points",
    "            ]",
    "        if self._backend == 'opensearch':",
    "            resp = self._os.search(index=self.name, body={",
    "                'size': top_k,",
    "                'query': {'knn': {'embedding': {'vector': qvec, 'k': top_k}}},",
    "                '_source': {'excludes': ['embedding']}})",
    "            return [",
    "                {'text': h['_source'].get('text', ''),",
    "                 'metadata': h['_source'].get('metadata', {}),",
    "                 'score': h['_score'], 'id': h['_id']}",
    "                for h in resp['hits']['hits']",
    "            ]",
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

cells.append(code(L(
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
    "        \"If not found say 'Not found in context.'\\n\\n\"",
    "        f'Context:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:'",
    "    )",
    "    return str(Agent(",
    "        model=_model,",
    "        system_prompt='You are a precise Q&A assistant. Answer only from the provided context.'",
    "    )(prompt))",
    "",
    'test_emb = embed_text("climate")',
    'print(f"Embedding OK -- dim={len(test_emb)}")',
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
    'print(f"Active backend: {vs._backend}")',
    "vs.create_collection(dim=EMBEDDING_DIM, force_recreate=True)",
)))

# ── Step 6 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 6 — Load PDF & Split into Sentences\n\n"
    "Load `climate.pdf` with `pypdf` (absolute path set in Step 2),\n"
    "then split each page into sentences using a regex splitter.\n"
))

cells.append(code(L(
    "def split_sentences(text: str) -> List[str]:",
    "    parts = re.split(r'(?<=[.!?])\\s+', text.replace('\\n', ' '))",
    "    return [p.strip() for p in parts if len(p.strip()) > 10]",
    "",
    "reader = pypdf.PdfReader(PDF_PATH)",
    'print(f"PDF   : {PDF_PATH}")',
    'print(f"Pages : {len(reader.pages)}")',
    "",
    "all_sentences: List[Dict] = []",
    "for page_num, page in enumerate(reader.pages):",
    "    page_text = page.extract_text() or ''",
    "    for sent in split_sentences(page_text):",
    "        all_sentences.append({'text': sent, 'page_num': page_num + 1})",
    "",
    'print(f"Total sentences : {len(all_sentences)}")',
    "print(f\"Avg sentence len: {sum(len(s['text']) for s in all_sentences)/len(all_sentences):.0f} chars\")",
    'print("\\nSample sentence 10:")',
    "print(all_sentences[10]['text'])",
)))

# ── Step 7 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 7 — Embed All Sentences\n\n"
    "Embed each sentence with Titan V2 to detect topic-shift breakpoints.\n"
    "These embeddings are NOT stored in Qdrant — only used to find boundaries.\n"
))

cells.append(code(L(
    'print(f"Embedding {len(all_sentences)} sentences for breakpoint detection...")',
    "t0                  = time.time()",
    "sentence_texts      = [s['text'] for s in all_sentences]",
    "sentence_embeddings = embed_batch(sentence_texts, label='[sentences]')",
    'print(f"Done in {time.time()-t0:.1f}s")',
)))

# ── Step 8 ────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 8 — Compute Adjacent Similarities & Find Breakpoints\n\n"
    "For each consecutive sentence pair, compute cosine similarity.\n"
    "Low similarity = topic shift = chunk boundary.\n"
))

cells.append(code(L(
    "def cosine_sim(a: List[float], b: List[float]) -> float:",
    "    return sum(x * y for x, y in zip(a, b))  # vectors already L2-normalised",
    "",
    "similarities = [",
    "    cosine_sim(sentence_embeddings[i], sentence_embeddings[i + 1])",
    "    for i in range(len(sentence_embeddings) - 1)",
    "]",
    "",
    'print("Similarity stats:")',
    'print(f"  min  : {min(similarities):.4f}")',
    'print(f"  max  : {max(similarities):.4f}")',
    'print(f"  mean : {sum(similarities)/len(similarities):.4f}")',
    "",
    "def bp_percentile(sims: List[float], pct: float = 25) -> List[int]:",
    "    thr = sorted(sims)[max(0, int(len(sims) * pct / 100) - 1)]",
    "    return [i for i, s in enumerate(sims) if s <= thr]",
    "",
    "def bp_std_dev(sims: List[float], mult: float = 1.0) -> List[int]:",
    "    mean = sum(sims) / len(sims)",
    "    std  = math.sqrt(sum((s - mean) ** 2 for s in sims) / len(sims))",
    "    return [i for i, s in enumerate(sims) if s < mean - mult * std]",
    "",
    "def bp_threshold(sims: List[float], thr: float = 0.4) -> List[int]:",
    "    return [i for i, s in enumerate(sims) if s < thr]",
    "",
    "bp_pct   = bp_percentile(similarities, BREAKPOINT_PERCENTILE)",
    "bp_std   = bp_std_dev(similarities, BREAKPOINT_STD_MULT)",
    "bp_fixed = bp_threshold(similarities, BREAKPOINT_THRESHOLD)",
    "",
    'print(f"Breakpoints found:")',
    'print(f"  percentile ({BREAKPOINT_PERCENTILE}%) : {len(bp_pct)}")',
    'print(f"  std_dev    (x{BREAKPOINT_STD_MULT})  : {len(bp_std)}")',
    'print(f"  threshold  (<{BREAKPOINT_THRESHOLD}): {len(bp_fixed)}")',
)))

# ── Step 9 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 9 — Build Semantic Chunks & Index in Qdrant"))

cells.append(code(L(
    "def build_chunks(sentences: List[Dict], breakpoints: List[int],",
    "                 max_sents: int = MAX_CHUNK_SENTENCES) -> List[Dict]:",
    "    chunks, cur, bp_set = [], [], set(breakpoints)",
    "    for i, sent in enumerate(sentences):",
    "        cur.append(sent)",
    "        if i in bp_set or len(cur) >= max_sents or i == len(sentences) - 1:",
    "            text = ' '.join(s['text'] for s in cur).strip()",
    "            if text:",
    "                chunks.append({",
    "                    'text'       : text,",
    "                    'page_num'   : cur[0]['page_num'],",
    "                    'page_end'   : cur[-1]['page_num'],",
    "                    'n_sentences': len(cur),",
    "                })",
    "            cur = []",
    "    return chunks",
    "",
    "active_bp = {'percentile': bp_pct, 'std_dev': bp_std, 'threshold': bp_fixed}[BREAKPOINT_STRATEGY]",
    "chunks    = build_chunks(all_sentences, active_bp)",
    "",
    'print(f"Semantic chunks created  : {len(chunks)}")',
    "print(f\"Avg chunk length         : {sum(len(c['text']) for c in chunks)/len(chunks):.0f} chars\")",
    "print(f\"Avg sentences per chunk  : {sum(c['n_sentences'] for c in chunks)/len(chunks):.1f}\")",
    'print("\\nSample chunk 0:")',
    "print(f\"  pages {chunks[0]['page_num']}-{chunks[0]['page_end']}, {chunks[0]['n_sentences']} sentences\")",
    "print(f\"  {chunks[0]['text'][:250]}...\")",
    "",
    "# Embed chunks and index",
    'print(f"\\nEmbedding {len(chunks)} chunks...")',
    "t0         = time.time()",
    "chunk_embs = embed_batch([c['text'] for c in chunks], label='[chunks]')",
    "docs       = [",
    "    {",
    "        'text'     : chunks[i]['text'],",
    "        'embedding': chunk_embs[i],",
    "        'metadata' : {",
    "            'chunk_index': i,",
    "            'page_num'   : chunks[i]['page_num'],",
    "            'page_end'   : chunks[i]['page_end'],",
    "            'n_sentences': chunks[i]['n_sentences'],",
    "            'source'     : 'climate.pdf',",
    "            'strategy'   : BREAKPOINT_STRATEGY,",
    "        }",
    "    }",
    "    for i in range(len(chunks))",
    "]",
    "indexed = vs.upsert(docs)",
    'print(f"Indexed {indexed} chunks in {time.time()-t0:.1f}s  |  total in Qdrant: {vs.count()}")',
)))

# ── Step 10 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 10 — RAG Queries"))

cells.append(code(L(
    "def rag_query(question: str, top_k: int = TOP_K, verbose: bool = True) -> Dict:",
    "    t0      = time.time()",
    "    results = vs.search(embed_text(question), top_k=top_k)",
    "    answer  = generate_answer(question, [r['text'] for r in results])",
    "    latency = (time.time() - t0) * 1000",
    "    if verbose:",
    '        print(f"\\nQ: {question}")',
    '        print(f"A: {answer}")',
    "        for i, r in enumerate(results[:3], 1):",
    "            pg = r['metadata'].get('page_num', '?')",
    "            ns = r['metadata'].get('n_sentences', '?')",
    "            print(f\"  [{i}] page={pg}  sents={ns}  score={r['score']:.4f}  {r['text'][:80]}...\")",
    '        print(f"  Latency: {latency:.0f}ms")',
    '        print("-" * 70)',
    "    return {'question': question, 'answer': answer, 'results': results, 'latency_ms': latency}",
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
    "## Step 11 — Strategy Comparison\n\n"
    "Build one in-memory store per strategy, run the same query, compare chunk counts.\n"
))

cells.append(code(L(
    "compare_q = 'What are the main methods used in weather analysis?'",
    "strat_map = [('percentile', bp_pct), ('std_dev', bp_std), ('threshold', bp_fixed)]",
    "",
    "print('Strategy comparison for: ' + compare_q)",
    "print('{:<15} {:>7}  {:>9}  {}'.format('Strategy', 'Chunks', 'Avg len', 'Top-1 snippet'))",
    "print('-' * 80)",
    "for name, bp in strat_map:",
    "    sc  = build_chunks(all_sentences, bp)",
    "    se  = embed_batch([c['text'] for c in sc], label='[' + name + ']')",
    "    tmp = VectorStore('tmp_' + name)",
    "    tmp.create_collection(dim=EMBEDDING_DIM)",
    "    tmp.upsert([{'text': sc[i]['text'], 'embedding': se[i],",
    "                 'metadata': {'i': i}} for i in range(len(sc))])",
    "    top  = tmp.search(embed_text(compare_q), top_k=1)",
    "    avg  = sum(len(c['text']) for c in sc) / len(sc)",
    "    snip = top[0]['text'][:55] if top else '(no results)'",
    "    print('{:<15} {:>7}  {:>9.0f}  {}...'.format(name, len(sc), avg, snip))",
)))

# ── Step 12 ────────────────────────────────────────────────────────────────────
cells.append(md("## Step 12 — Evaluation & Metrics"))

cells.append(code(L(
    "eval_cases = [",
    "    {'question': 'What is weather forecasting and why is it important?',",
    "     'expected_keywords': ['forecast', 'weather', 'predict', 'atmosphere', 'climate']},",
    "    {'question': 'What are the main methods used in weather analysis?',",
    "     'expected_keywords': ['analysis', 'synoptic', 'observation', 'data', 'pressure']},",
    "    {'question': 'How does climatology differ from meteorology?',",
    "     'expected_keywords': ['climate', 'weather', 'long', 'study', 'atmosphere']},",
    "]",
    "print('{:<55} {:>7} {:>10}'.format('Question', 'KW Hit', 'Latency'))",
    "print('-' * 75)",
    "eval_log = []",
    "for case in eval_cases:",
    "    r    = rag_query(case['question'], verbose=False)",
    "    low  = r['answer'].lower()",
    "    hits = sum(1 for kw in case['expected_keywords'] if kw in low)",
    "    n    = len(case['expected_keywords'])",
    "    eval_log.append(r)",
    "    print('{:<55} {}/{} ({:.0f}%) {:>8.0f}ms'.format(",
    "        case['question'][:54], hits, n, hits / n * 100, r['latency_ms']))",
    "print()",
    "print('Avg latency : {:.0f}ms'.format(sum(r['latency_ms'] for r in eval_log) / len(eval_log)))",
    "print('Chunks used : {}  |  Avg size: {:.0f} chars'.format(",
    "    len(chunks), sum(len(c['text']) for c in chunks) / len(chunks)))",
)))

# ── Step 13 Summary ────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 13 — Summary\n\n"
    "| Component | Implementation |\n"
    "|-----------|---------------|\n"
    "| PDF | `pypdf.PdfReader` — absolute path hardcoded in Step 2 |\n"
    "| Sentences | Regex splitter — no NLTK dependency |\n"
    "| Sentence embeddings | Bedrock Titan V2 — breakpoint detection only |\n"
    "| Breakpoints | Cosine similarity drop — 3 strategies |\n"
    "| Chunk embeddings | Bedrock Titan V2 — stored in Qdrant |\n"
    "| Vector DB | Qdrant Cloud -> OpenSearch -> in-memory |\n"
    "| LLM | AWS Strands Agent + Claude Sonnet 4.6 |\n\n"
    "### Next: **03 — Hierarchical RAG**\n"
))

cells.append(code(L(
    "# vs.delete_collection()  # uncomment to clean up",
    "print(f\"Collection '{COLLECTION_NAME}' kept in {vs._backend} -- {vs.count()} vectors\")",
    'print("\\nDone. Give the go-ahead for notebook 03.")',
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

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

errors = []
for i, c in enumerate(cells):
    if c['cell_type'] == 'code':
        try:
            ast.parse(''.join(c['source']))
        except SyntaxError as e:
            errors.append((i, e.lineno, e.msg))

print(f"Written : {path}")
print(f"Cells   : {len(cells)}")
if errors:
    for i, ln, msg in errors:
        print(f"SYNTAX ERROR cell {i} line {ln}: {msg}")
else:
    print("All code cells parse OK")
