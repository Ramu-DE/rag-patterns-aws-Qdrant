# Query Decomposition RAG — Why It's Needed & When to Use It

## The Core Problem with Standard RAG

Standard RAG embeds your question as a single vector and retrieves the top-K closest chunks.
This breaks down the moment a question contains more than one concept.

```
Question: "What causes sea level rise, how does ice melt contribute,
           and what are the consequences for coastal cities?"

Single embedding → blurry average of all three concepts
                 → retrieves chunks that are sort-of relevant to everything
                 → focused on nothing
```

The vector search cannot serve three masters at once. You get mediocre coverage
of each sub-topic instead of strong coverage of any.

---

## What Query Decomposition Fixes

The LLM first breaks the complex question into focused sub-questions.
Each sub-question gets its own dedicated retrieval pass.

```
Complex Question
      │
      ▼
  LLM Decomposer
      │
      ├─ sub-Q1: "What causes sea level rise?"
      │       └─ embed → top-K hits (thermal expansion passages)
      │
      ├─ sub-Q2: "How does ice melt contribute to sea level rise?"
      │       └─ embed → top-K hits (glacier/ice sheet passages)
      │
      └─ sub-Q3: "What are the impacts on coastal cities?"
              └─ embed → top-K hits (flooding/displacement passages)
                    │
                    ▼
            Merge + Deduplicate
                    │
                    ▼
            LLM Synthesises one answer
            with page citations from all sub-topics
```

Each sub-question is sharp and focused — its embedding hits exactly the right passages.

---

## Where It Sits in the Notebook Series

| Notebook | Pattern           | What it improves                          |
|----------|-------------------|-------------------------------------------|
| 01       | Simple RAG        | Baseline                                  |
| 07       | Hybrid Search     | HOW you search (BM25 + vectors)           |
| 08       | HyDE              | WHAT you embed (hypothetical answer)      |
| 09       | Reranking         | HOW you score results after retrieval     |
| 13       | Query Decomp      | HOW MANY retrieval passes you make        |

Query Decomposition is a **query handling** technique (Tier 3).
The problem is not the vector store or embeddings — it is that a single query vector
cannot represent a multi-concept question fairly.

---

## Real-World Use Cases

### 1. Research & Document Q&A
**Scenario:** A researcher asks:
> "How have Arctic temperatures changed, what effect does this have on the jet stream,
> and how does that influence extreme weather in mid-latitudes?"

Three distinct retrieval targets. Decomposition ensures each concept pulls its own evidence.

---

### 2. Legal & Compliance Document Search
**Scenario:** A lawyer asks:
> "What are the GDPR requirements for data retention, what penalties apply for violations,
> and how do they differ from CCPA?"

Each clause has its own location in a large document corpus.
A single query would retrieve vague overlap rather than precise answers to each point.

---

### 3. Medical / Clinical Information Systems
**Scenario:** A clinician asks:
> "What are the symptoms of Type 2 diabetes, what are the first-line treatments,
> and what complications arise if untreated?"

Symptoms, treatments, and complications live in completely different sections of medical literature.
Decomposition retrieves each independently then synthesises a unified clinical summary.

---

### 4. Financial Report Analysis
**Scenario:** An analyst asks:
> "What was the revenue growth last quarter, what drove the cost increase,
> and how does this compare to industry benchmarks?"

Three different tables/sections in an earnings report. One vector misses at least one of them.

---

### 5. Customer Support Knowledge Bases
**Scenario:** A user asks:
> "How do I set up two-factor authentication, what happens if I lose my phone,
> and how do I contact support if I'm locked out?"

Three separate FAQ entries. Decomposition retrieves all three; simple RAG likely
surfaces only the first or most prominent one.

---

### 6. Technical Documentation
**Scenario:** A developer asks:
> "How do I configure the connection pool, what are the retry settings,
> and how should I handle timeout errors?"

Config, retry, and error-handling are documented in separate sections.
Decomposition guarantees all three are retrieved.

---

## When NOT to Use It

| Situation                          | Reason to skip decomposition                  |
|------------------------------------|-----------------------------------------------|
| Simple single-concept questions    | Adds latency with no retrieval benefit        |
| Real-time / low-latency endpoints  | 2 LLM calls + N embed calls adds ~2–4s        |
| Very short documents               | Full document fits in context anyway          |
| Questions with only one entity     | "What is CO2?" needs one retrieval pass       |

**Rule of thumb:** If the question contains *and*, *also*, *both*, *compare*,
*how does X affect Y*, or asks for multiple distinct facts — decompose.

---

## Signals in a Question That Call for Decomposition

```
"... and ..."          → at least 2 sub-questions
"compare X with Y"     → retrieve X evidence + Y evidence separately
"causes ... effects"   → cause chain = multiple retrieval targets
"what ... why ... how" → three distinct information needs in one sentence
"relationship between X and Y and Z" → N-way concept retrieval
```

---

## Key Numbers from This Notebook

| Metric                  | Simple RAG | Decomposed RAG          |
|-------------------------|------------|-------------------------|
| Retrieval passes        | 1          | N (one per sub-question)|
| Unique pages covered    | ~2–3       | ~5–8 (wider coverage)   |
| LLM calls               | 1          | 2 (decompose + synthesise) |
| Typical latency delta   | baseline   | +2–4 seconds            |
| Answer completeness     | partial    | full multi-concept      |
