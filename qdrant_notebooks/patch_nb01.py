"""Patch notebook 01 to load climate.pdf and use climate-relevant questions."""
import json, uuid

path = r"C:\Users\Administrator\RAG\qdrant_notebooks\01_Simple_RAG.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)

# Ensure every cell has an id
for cell in nb["cells"]:
    if "id" not in cell:
        cell["id"] = str(uuid.uuid4())[:8]

# Helper: find a cell by a unique string in its source
def find_cell(marker):
    for i, cell in enumerate(nb["cells"]):
        if marker in "".join(cell["source"]):
            return i, cell
    return None, None

# ── Step 6 markdown ───────────────────────────────────────────────────────────
idx, cell = find_cell("## Step 6")
if cell:
    cell["source"] = [
        "## Step 6 — Load & Chunk the PDF\n",
        "\n",
        "We load `data/climate.pdf` (a 13-page Advanced Climatology / Weather Forecasting paper)\n",
        "using `PyPDFLoader`, then split into overlapping chunks with `RecursiveCharacterTextSplitter`.\n",
        "\n",
        "**Reuse pattern:** all subsequent notebooks use the same PDF — just change `PDF_PATH`.\n",
        "\n",
        "| Parameter | Value | Why |\n",
        "|-----------|-------|-----|\n",
        "| `CHUNK_SIZE` | 1000 chars | Fits comfortably in Claude's context |\n",
        "| `CHUNK_OVERLAP` | 200 chars | Preserves context across chunk boundaries |\n",
        "| Metadata | page_num + source | Enables citation in answers |\n"
    ]

# ── Step 6 code — replace sample docs with PDF loading ───────────────────────
idx, cell = find_cell("# Sample corpus")
if cell:
    cell["source"] = [
        "import os\n",
        "from langchain_community.document_loaders import PyPDFLoader\n",
        "from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
        "\n",
        "# Path to the shared PDF — used by all RAG notebooks in this series\n",
        "PDF_PATH = os.path.join(\"..\", \"data\", \"climate.pdf\")\n",
        "\n",
        "# Load every page (each page becomes one LangChain Document)\n",
        "loader = PyPDFLoader(PDF_PATH)\n",
        "pages  = loader.load()\n",
        "print(f\"PDF loaded   : {PDF_PATH}\")\n",
        "print(f\"Total pages  : {len(pages)}\")\n",
        "print(f\"\\nPage 1 preview (first 400 chars):\")\n",
        "print(pages[0].page_content[:400])\n",
        "\n",
        "# Chunk each page — RecursiveCharacterTextSplitter tries paragraph\n",
        "# boundaries first, then sentence, then word, then character\n",
        "splitter = RecursiveCharacterTextSplitter(\n",
        "    chunk_size      = CHUNK_SIZE,\n",
        "    chunk_overlap   = CHUNK_OVERLAP,\n",
        "    length_function = len,\n",
        "    separators      = [\"\\n\\n\", \"\\n\", \". \", \" \", \"\"]\n",
        ")\n",
        "\n",
        "chunks = []\n",
        "for page_num, page in enumerate(pages):\n",
        "    for chunk_text in splitter.split_text(page.page_content):\n",
        "        if chunk_text.strip():          # skip blank / whitespace-only chunks\n",
        "            chunks.append({\n",
        "                \"text\"    : chunk_text.strip(),\n",
        "                \"page_num\": page_num + 1,   # 1-based\n",
        "                \"source\"  : os.path.basename(PDF_PATH)\n",
        "            })\n",
        "\n",
        "print(f\"\\nChunks created   : {len(chunks)}\")\n",
        "print(f\"Avg chunk length : {sum(len(c['text']) for c in chunks)/len(chunks):.0f} chars\")\n",
        "print(f\"\\nSample chunk (chunk 0):\")\n",
        "print(chunks[0]['text'][:300])\n"
    ]

# ── Indexing cell — update metadata fields to include page_num ────────────────
idx, cell = find_cell("docs_to_index")
if cell:
    cell["source"] = [
        "print(f\"Generating embeddings for {len(chunks)} chunks...\")\n",
        "t0 = time.time()\n",
        "\n",
        "texts      = [c[\"text\"] for c in chunks]\n",
        "embeddings = embed_batch(texts)\n",
        "\n",
        "# Build index records — metadata carries page number for citation\n",
        "docs_to_index = [\n",
        "    {\n",
        "        \"text\"     : chunks[i][\"text\"],\n",
        "        \"embedding\": embeddings[i],\n",
        "        \"metadata\" : {\n",
        "            \"chunk_index\": i,\n",
        "            \"page_num\"   : chunks[i][\"page_num\"],\n",
        "            \"source\"     : chunks[i][\"source\"]\n",
        "        }\n",
        "    }\n",
        "    for i in range(len(chunks))\n",
        "]\n",
        "\n",
        "indexed = vs.upsert(docs_to_index)\n",
        "elapsed = time.time() - t0\n",
        "\n",
        "print(f\"\\nIndexed  : {indexed} chunks\")\n",
        "print(f\"Time     : {elapsed:.2f}s  ({elapsed/len(chunks):.2f}s per chunk)\")\n",
        "print(f\"Count in Qdrant collection: {vs.count()}\")\n"
    ]

# ── Query cell — climate questions + show page citations ─────────────────────
idx, cell = find_cell("test_questions")
if cell:
    cell["source"] = [
        "def rag_query(question: str, top_k: int = TOP_K, verbose: bool = True) -> Dict:\n",
        "    \"\"\"\n",
        "    Full RAG pipeline for one question.\n",
        "    Steps: embed question -> vector search -> format context -> Claude generates answer.\n",
        "    Returns dict with question, answer, sources, latency_ms.\n",
        "    \"\"\"\n",
        "    t0 = time.time()\n",
        "\n",
        "    # A: embed the question with Titan V2\n",
        "    q_embedding = embed_text(question)\n",
        "\n",
        "    # B: retrieve top-K most similar chunks from Qdrant\n",
        "    results = vs.search(q_embedding, top_k=top_k)\n",
        "\n",
        "    # C: generate answer — Claude sees the retrieved chunks as context\n",
        "    context_texts = [r[\"text\"] for r in results]\n",
        "    answer = generate_answer(question, context_texts)\n",
        "\n",
        "    latency_ms = (time.time() - t0) * 1000\n",
        "\n",
        "    if verbose:\n",
        "        print(f\"\\nQuestion : {question}\")\n",
        "        print(f\"Latency  : {latency_ms:.0f}ms\")\n",
        "        print(f\"\\nAnswer:\\n{answer}\")\n",
        "        print(f\"\\nTop sources (with page citations):\")\n",
        "        for i, r in enumerate(results[:3], 1):\n",
        "            page = r['metadata'].get('page_num', '?')\n",
        "            print(f\"  [{i}] page={page}  score={r['score']:.4f}  {r['text'][:100]}...\")\n",
        "        print(\"-\" * 70)\n",
        "\n",
        "    return {\"question\": question, \"answer\": answer,\n",
        "            \"sources\": results, \"latency_ms\": latency_ms}\n",
        "\n",
        "\n",
        "# Climate / weather forecasting questions matched to climate.pdf content\n",
        "test_questions = [\n",
        "    \"What is weather forecasting and why is it important?\",\n",
        "    \"What are the main methods used in weather analysis?\",\n",
        "    \"How does climatology differ from meteorology?\",\n",
        "    \"What factors influence weather patterns and climate?\",\n",
        "]\n",
        "\n",
        "results_log = []\n",
        "for q in test_questions:\n",
        "    r = rag_query(q)\n",
        "    results_log.append(r)\n"
    ]

# ── Evaluation cell — climate-domain keywords ────────────────────────────────
idx, cell = find_cell("eval_cases")
if cell:
    cell["source"] = [
        "# Keyword-coverage evaluation: did the answer contain expected domain terms?\n",
        "eval_cases = [\n",
        "    {\"question\": \"What is weather forecasting and why is it important?\",\n",
        "     \"expected_keywords\": [\"forecast\", \"weather\", \"predict\", \"atmosphere\", \"climate\"]},\n",
        "    {\"question\": \"What are the main methods used in weather analysis?\",\n",
        "     \"expected_keywords\": [\"analysis\", \"synoptic\", \"observation\", \"data\", \"pressure\"]},\n",
        "    {\"question\": \"How does climatology differ from meteorology?\",\n",
        "     \"expected_keywords\": [\"climate\", \"weather\", \"long\", \"study\", \"atmosphere\"]},\n",
        "]\n",
        "\n",
        "print(f\"{'Question':<55} {'KW Hit':>7} {'Latency':>10}\")\n",
        "print(\"-\" * 75)\n",
        "\n",
        "for case in eval_cases:\n",
        "    result = rag_query(case[\"question\"], verbose=False)\n",
        "    answer_lower = result[\"answer\"].lower()\n",
        "    hits  = sum(1 for kw in case[\"expected_keywords\"] if kw.lower() in answer_lower)\n",
        "    total = len(case[\"expected_keywords\"])\n",
        "    print(f\"{case['question'][:54]:<55} {hits}/{total} ({hits/total*100:.0f}%) {result['latency_ms']:>8.0f}ms\")\n",
        "\n",
        "print()\n",
        "avg_lat = sum(r[\"latency_ms\"] for r in results_log) / len(results_log)\n",
        "print(f\"Average latency across all queries: {avg_lat:.0f}ms\")\n"
    ]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook 01 patched successfully — climate.pdf wired in.")
