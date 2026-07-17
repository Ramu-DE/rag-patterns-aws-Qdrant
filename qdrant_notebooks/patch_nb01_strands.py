"""
Rewrite notebook 01 to use AWS Strands framework instead of LangChain.

Changes made:
  - Remove: langchain, langchain-community, langchain-text-splitters from install cell
  - Add:    strands-agents to install cell
  - Remove: LangChain imports (RecursiveCharacterTextSplitter)
  - Add:    strands BedrockModel + Agent imports
  - Step 6: Replace PyPDFLoader + RecursiveCharacterTextSplitter
            with pypdf.PdfReader + native Python recursive splitter
  - Bedrock helpers: Replace raw boto3 invoke_model for generation
                     with Strands Agent (still use boto3 for embeddings,
                     since Strands has no embeddings API)
  - Summary cell: update component table
"""

import json, uuid

path = r"C:\Users\Administrator\RAG\qdrant_notebooks\01_Simple_RAG.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)

# Ensure every cell has an id
for cell in nb["cells"]:
    if "id" not in cell:
        cell["id"] = str(uuid.uuid4())[:8]


def find_cell(marker):
    for i, cell in enumerate(nb["cells"]):
        if marker in "".join(cell["source"]):
            return i, cell
    return None, None


# ── Step 1: Install cell — swap LangChain for strands-agents ──────────────────
idx, cell = find_cell("langchain")
if cell and cell["cell_type"] == "code":
    cell["source"] = [
        "# Install required packages (safe to re-run)\n",
        "import subprocess, sys\n",
        "packages = [\n",
        '    "boto3",\n',
        '    "qdrant-client",\n',
        '    "opensearch-py",\n',
        '    "requests-aws4auth",\n',
        '    "strands-agents",   # AWS Strands — replaces LangChain\n',
        '    "pypdf",            # Direct PDF reading\n',
        "]\n",
        "subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"--quiet\"] + packages)\n",
        'print("All packages ready.")\n',
    ]
    print("Patched: install cell")


# ── Step 2: Imports cell — remove LangChain, add Strands ─────────────────────
idx, cell = find_cell("from langchain_text_splitters")
if cell and cell["cell_type"] == "code":
    cell["source"] = [
        "import os, sys, json, time, uuid\n",
        "from typing import List, Dict, Optional, Tuple\n",
        "\n",
        "import boto3\n",
        "\n",
        "# AWS Strands — agent framework used for LLM generation\n",
        "from strands import Agent\n",
        "from strands.models.bedrock import BedrockModel\n",
        "\n",
        "# Qdrant\n",
        "from qdrant_client import QdrantClient\n",
        "from qdrant_client.models import (\n",
        "    Distance, VectorParams, PointStruct, ScoredPoint\n",
        ")\n",
        "\n",
        'print("Imports OK")\n',
    ]
    print("Patched: imports cell")


# ── Step 6 markdown — update description ────────────────────────────────────
idx, cell = find_cell("## Step 6")
if cell and cell["cell_type"] == "markdown":
    cell["source"] = [
        "## Step 6 — Load & Chunk the PDF\n",
        "\n",
        "We load `data/climate.pdf` (a 13-page Advanced Climatology paper) using **`pypdf`** directly\n",
        "(no LangChain dependency), then split into overlapping chunks with a native Python\n",
        "recursive character splitter — the same algorithm used by LangChain, re-implemented\n",
        "without any third-party dependency.\n",
        "\n",
        "**Reuse pattern:** all subsequent notebooks use the same PDF — just change `PDF_PATH`.\n",
        "\n",
        "| Parameter | Value | Why |\n",
        "|-----------|-------|-----|\n",
        "| `CHUNK_SIZE` | 1000 chars | Fits comfortably in Claude's context |\n",
        "| `CHUNK_OVERLAP` | 200 chars | Preserves context across chunk boundaries |\n",
        "| Metadata | page_num + source | Enables citation in answers |\n",
    ]
    print("Patched: Step 6 markdown")


# ── Step 6 code — replace LangChain loaders with pypdf + native splitter ─────
idx, cell = find_cell("from langchain_community.document_loaders")
if cell and cell["cell_type"] == "code":
    cell["source"] = [
        "import os\n",
        "import pypdf\n",
        "\n",
        "# ── Native recursive character splitter (no LangChain) ─────────────────────\n",
        "def _split_text(text: str, separators: List[str], chunk_size: int,\n",
        "                chunk_overlap: int) -> List[str]:\n",
        "    \"\"\"Recursively split text by preferred separators, merge small pieces.\"\"\"\n",
        "    # Pick the first separator that actually appears in the text\n",
        "    sep = \"\"\n",
        "    rest = separators\n",
        "    for s in separators:\n",
        "        if s == \"\" or s in text:\n",
        "            sep = s\n",
        "            rest = separators[separators.index(s) + 1:]\n",
        "            break\n",
        "\n",
        "    splits = text.split(sep) if sep else list(text)\n",
        "\n",
        "    merged: List[str] = []\n",
        "    current = \"\"\n",
        "    for piece in splits:\n",
        "        candidate = (current + sep + piece).strip() if current else piece.strip()\n",
        "        if len(candidate) <= chunk_size:\n",
        "            current = candidate\n",
        "        else:\n",
        "            if current:\n",
        "                merged.append(current)\n",
        "            # piece itself may be too large — recurse with next separator\n",
        "            if len(piece) > chunk_size and rest:\n",
        "                merged.extend(_split_text(piece, rest, chunk_size, chunk_overlap))\n",
        "            else:\n",
        "                current = piece.strip()\n",
        "    if current:\n",
        "        merged.append(current)\n",
        "\n",
        "    # Apply overlap: slide a window over the merged list\n",
        "    if chunk_overlap == 0 or len(merged) <= 1:\n",
        "        return [m for m in merged if m]\n",
        "\n",
        "    final: List[str] = []\n",
        "    i = 0\n",
        "    while i < len(merged):\n",
        "        chunk = merged[i]\n",
        "        j = i + 1\n",
        "        while j < len(merged) and len(chunk) + len(sep) + len(merged[j]) <= chunk_size:\n",
        "            chunk = (chunk + sep + merged[j]).strip()\n",
        "            j += 1\n",
        "        final.append(chunk)\n",
        "        # advance so the next window overlaps by chunk_overlap characters\n",
        "        advance = max(1, len(chunk) - chunk_overlap)\n",
        "        chars_skipped = 0\n",
        "        while i < len(merged) - 1 and chars_skipped < advance:\n",
        "            chars_skipped += len(merged[i])\n",
        "            i += 1\n",
        "    return [f for f in final if f]\n",
        "\n",
        "\n",
        "def recursive_split(text: str, chunk_size: int = 1000,\n",
        "                    chunk_overlap: int = 200) -> List[str]:\n",
        "    separators = [\"\\n\\n\", \"\\n\", \". \", \" \", \"\"]\n",
        "    return _split_text(text, separators, chunk_size, chunk_overlap)\n",
        "\n",
        "\n",
        "# ── Load PDF with pypdf ────────────────────────────────────────────────────\n",
        "PDF_PATH = os.path.join(\"..\", \"data\", \"climate.pdf\")\n",
        "\n",
        "reader = pypdf.PdfReader(PDF_PATH)\n",
        "print(f\"PDF loaded   : {PDF_PATH}\")\n",
        "print(f\"Total pages  : {len(reader.pages)}\")\n",
        "print(f\"\\nPage 1 preview (first 400 chars):\")\n",
        "print(reader.pages[0].extract_text()[:400])\n",
        "\n",
        "# ── Chunk every page ──────────────────────────────────────────────────────\n",
        "chunks: List[Dict] = []\n",
        "for page_num, page in enumerate(reader.pages):\n",
        "    page_text = page.extract_text() or \"\"\n",
        "    for chunk_text in recursive_split(page_text, CHUNK_SIZE, CHUNK_OVERLAP):\n",
        "        if chunk_text.strip():\n",
        "            chunks.append({\n",
        "                \"text\"    : chunk_text.strip(),\n",
        "                \"page_num\": page_num + 1,\n",
        "                \"source\"  : os.path.basename(PDF_PATH)\n",
        "            })\n",
        "\n",
        "print(f\"\\nChunks created   : {len(chunks)}\")\n",
        "print(f\"Avg chunk length : {sum(len(c['text']) for c in chunks)/len(chunks):.0f} chars\")\n",
        "print(f\"\\nSample chunk (chunk 0):\")\n",
        "print(chunks[0]['text'][:300])\n",
    ]
    print("Patched: Step 6 code cell")


# ── Bedrock helpers cell — keep embed_text/embed_batch as boto3,
#    replace generate_answer with Strands Agent ────────────────────────────────
idx, cell = find_cell("bedrock_rt = boto3.client")
if cell and cell["cell_type"] == "code":
    cell["source"] = [
        "# ── Embeddings: still use boto3 directly (Strands has no embeddings API) ────\n",
        "bedrock_rt = boto3.client(\"bedrock-runtime\", region_name=AWS_REGION)\n",
        "\n",
        "\n",
        "def embed_text(text: str) -> List[float]:\n",
        "    \"\"\"1024-dim Titan V2 embedding for a single string.\"\"\"\n",
        "    body = json.dumps({\"inputText\": text, \"dimensions\": EMBEDDING_DIM, \"normalize\": True})\n",
        "    resp = bedrock_rt.invoke_model(\n",
        "        modelId=EMBEDDING_MODEL, body=body,\n",
        "        contentType=\"application/json\", accept=\"application/json\"\n",
        "    )\n",
        "    return json.loads(resp[\"body\"].read())[\"embedding\"]\n",
        "\n",
        "\n",
        "def embed_batch(texts: List[str]) -> List[List[float]]:\n",
        "    \"\"\"Embed a list of texts; Titan V2 processes one at a time.\"\"\"\n",
        "    embeddings = []\n",
        "    for i, t in enumerate(texts):\n",
        "        embeddings.append(embed_text(t))\n",
        "        if (i + 1) % 10 == 0:\n",
        "            print(f\"  Embedded {i+1}/{len(texts)}\")\n",
        "        time.sleep(0.05)\n",
        "    return embeddings\n",
        "\n",
        "\n",
        "# ── Generation: use AWS Strands BedrockModel + Agent ─────────────────────────\n",
        "_strands_model = BedrockModel(\n",
        "    model_config={\"model_id\": LLM_MODEL},\n",
        "    region_name=AWS_REGION\n",
        ")\n",
        "\n",
        "\n",
        "def generate_answer(question: str, context_chunks: List[str]) -> str:\n",
        "    \"\"\"\n",
        "    Generate a grounded answer with AWS Strands Agent.\n",
        "    The agent receives retrieved chunks as context and answers using only them.\n",
        "    \"\"\"\n",
        "    context = \"\\n\\n\".join(\n",
        "        f\"[Chunk {i+1}]\\n{chunk}\" for i, chunk in enumerate(context_chunks)\n",
        "    )\n",
        "    prompt = (\n",
        "        f\"Use ONLY the context below to answer the question. \"\n",
        "        f\"If the answer is not in the context, say 'Not found in context.'\\n\\n\"\n",
        "        f\"Context:\\n{context}\\n\\n\"\n",
        "        f\"Question: {question}\\n\\nAnswer:\"\n",
        "    )\n",
        "    agent = Agent(\n",
        "        model=_strands_model,\n",
        "        system_prompt=\"You are a precise Q&A assistant. Answer only from the provided context.\"\n",
        "    )\n",
        "    result = agent(prompt)\n",
        "    return str(result)\n",
        "\n",
        "\n",
        "# Quick smoke test for embeddings\n",
        "test_emb = embed_text(\"Hello world\")\n",
        "print(f\"Embedding OK  — dim={len(test_emb)}, sample={[round(x, 4) for x in test_emb[:3]]}\")\n",
        "print(\"Strands Agent configured — BedrockModel ready\")\n",
    ]
    print("Patched: Bedrock helpers cell")


# ── Summary cell — update component table ─────────────────────────────────────
idx, cell = find_cell("RecursiveCharacterTextSplitter")
if cell and cell["cell_type"] == "markdown":
    cell["source"] = [
        "## Step 10 — Summary\n",
        "\n",
        "### What we built\n",
        "| Component | Implementation |\n",
        "|-----------|---------------|\n",
        "| **PDF Loading** | `pypdf.PdfReader` (direct, no LangChain) |\n",
        "| **Chunking** | Native Python recursive splitter (1000 chars, 200 overlap) |\n",
        "| **Embeddings** | Amazon Bedrock Titan V2 — 1024 dims |\n",
        "| **Vector DB** | Qdrant Cloud (falls back to OpenSearch → in-memory) |\n",
        "| **LLM** | AWS Strands `Agent` + Bedrock Claude Sonnet 4.6 |\n",
        "| **Retrieval** | Cosine similarity, top-5 |\n",
        "\n",
        "### Strands framework role\n",
        "- `BedrockModel` wraps the Bedrock runtime config (model ID, region, credentials)\n",
        "- `Agent` manages the generation call — system prompt, message routing, response extraction\n",
        "- Embeddings still use boto3 directly (Strands has no embeddings API)\n",
        "\n",
        "### Strengths\n",
        "- No LangChain dependency — pure AWS stack (Strands + Bedrock + Qdrant)\n",
        "- Same recursive chunking algorithm, implemented natively\n",
        "- Good baseline for comparison with advanced patterns\n",
        "\n",
        "### Limitations\n",
        "- No reranking — top-K may include low-quality matches\n",
        "- Fixed chunk size — may cut across important context boundaries\n",
        "- Single query — no query expansion or reformulation\n",
        "\n",
        "### Next: **02 — Graph RAG** (entities + relationships, multi-hop queries)\n",
    ]
    print("Patched: summary cell")


with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("\nNotebook 01 patched — LangChain replaced with AWS Strands + pypdf.")
