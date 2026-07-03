# 🎬 Movie Search Engine with AWS OpenSearch & Bedrock

A production-ready semantic search engine for movies using **AWS OpenSearch Serverless** and **AWS Bedrock** for embeddings and AI-powered recommendations.

## 🚀 Features

- **Semantic Search**: Natural language queries using vector embeddings
- **Hybrid Search**: Combines semantic similarity with keyword matching
- **AI-Powered Answers**: Claude Opus generates personalized movie recommendations
- **Filtered Search**: Search by genre, year, rating
- **Real-time**: Instant search results with OpenSearch Serverless
- **Streamlit UI**: Interactive web interface

## 🏗️ Architecture

```
┌─────────────────┐
│   Streamlit UI  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐      ┌──────────────────┐
│  Movie Search   │ ───→ │  AWS Bedrock     │
│     Engine      │      │  - Titan Embed V2│
└────────┬────────┘      │  - Claude Opus   │
         │               └──────────────────┘
         ↓
┌─────────────────┐
│   OpenSearch    │
│   Serverless    │
│  (Vector Store) │
└─────────────────┘
```

## 📊 Tech Stack

**Infrastructure:**
- AWS OpenSearch Serverless (vector search)
- AWS Bedrock (embeddings & LLM)

**Models:**
- **Embeddings**: Amazon Titan Text Embeddings V2 (1024-dim)
- **LLM**: Claude Opus 4.1 (cross-region inference profile)

**Backend:**
- Python 3.12+
- opensearch-py
- boto3

**Frontend:**
- Streamlit

## 🛠️ Setup

### Prerequisites

- AWS Account with Bedrock access
- AWS credentials configured
- Python 3.8+

### Installation

1. **Install dependencies:**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install boto3 opensearch-py requests requests-aws4auth streamlit pandas numpy
   ```

2. **Create OpenSearch Serverless collection:**
   ```bash
   python3 infrastructure/create_opensearch.py
   ```

   This creates:
   - OpenSearch Serverless collection
   - Security policies (encryption, network, data access)
   - Configuration file (`config.json`)

3. **Index movie data:**
   ```bash
   python3 indexer/movies_loader.py
   ```

   This:
   - Creates the `movies` index with vector field
   - Generates embeddings using Bedrock
   - Indexes movies into OpenSearch

4. **Run the app:**
   ```bash
   streamlit run 0_Home.py
   ```

   Access at: http://localhost:8501

## 🎯 Usage

### Command Line Test

```bash
python3 test_search.py
```

### Streamlit UI

1. Navigate to **Semantic Search** page
2. Enter a natural language query:
   - "movies about space exploration"
   - "psychological thrillers with plot twists"
   - "feel-good movies about friendship"
3. Choose search mode (semantic or hybrid)
4. Enable AI-powered answers for recommendations

### Python API

```python
from utils.movie_search import MovieSearch
import json

# Load config
with open('config.json') as f:
    config = json.load(f)

# Initialize search
searcher = MovieSearch(config)

# Search with AI answer
results, answer = searcher.search_and_answer(
    query="movies about time travel",
    search_type="hybrid",
    top_k=5,
    use_claude=True
)

# Print results
for movie in results:
    print(f"{movie['title']} ({movie['year']}) - {movie['rating']}/10")

print(f"\nAI: {answer['answer']}")
```

## 📁 Project Structure

```
vector-engine-demos/
├── 0_Home.py                  # Streamlit home page
├── pages/
│   └── 1_Semantic_Search.py   # Search interface
├── indexer/
│   └── movies_loader.py       # Data indexing pipeline
├── infrastructure/
│   └── create_opensearch.py   # OpenSearch setup
├── utils/
│   ├── bedrock_embeddings.py  # Embedding utilities
│   └── movie_search.py        # Search engine
├── data/
│   └── sample-movies-small.json  # Sample data
├── config.json                # OpenSearch config (generated)
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### Embedding Models

Change in `utils/bedrock_embeddings.py`:

```python
# Default: Amazon Titan V2
model_id = 'amazon.titan-embed-text-v2:0'

# Alternative: Cohere (if available)
# model_id = 'cohere.embed-english-v3'
```

### Claude Models

Available cross-region inference profiles:
- `us.anthropic.claude-opus-4-1-20250805-v1:0` (most capable)
- `us.anthropic.claude-3-haiku-20240307-v1:0` (fast, economical)

## 🔍 How It Works

### 1. Embedding Generation

```
Movie Data → Bedrock Titan → 1024-dim Vector
"Plot: A team explores wormhole..." → [0.123, -0.456, ...]
```

### 2. Semantic Search

```
User Query → Embedding → Vector Search → Top Results
"space movies" → [0.234, -0.567, ...] → Cosine Similarity → Interstellar
```

### 3. Hybrid Search

Combines:
- **Vector similarity** (plot embeddings)
- **Keyword matching** (title, genre, actors)

### 4. RAG with Claude

```
Query + Search Results → Claude → Natural Language Answer
"Best sci-fi?" + [Interstellar, Matrix...] → "I recommend Interstellar because..."
```

## 📊 Performance

- **Indexing**: ~10 movies/second (with embeddings)
- **Search latency**: <100ms (semantic search)
- **Embedding dimension**: 1024
- **Search algorithm**: HNSW (Hierarchical Navigable Small World)

## 🎨 Example Queries

| Query | Top Result | Why It Works |
|-------|-----------|--------------|
| "movies about redemption and hope" | The Shawshank Redemption | Semantic understanding of themes |
| "space survival epic" | Interstellar | Combined plot + genre matching |
| "mind-bending reality" | The Matrix | Conceptual similarity |
| "epic fantasy quest" | Lord of the Rings | Genre + plot semantics |

## 🐛 Troubleshooting

### "Access denied" error with Bedrock

- Ensure your IAM role has `AmazonBedrockFullAccess` policy
- Check that the model is available in your region

### "Document ID not supported"

- OpenSearch Serverless doesn't support custom document IDs
- Use auto-generated IDs (handled in code)

### "Refresh policy not supported"

- OpenSearch Serverless doesn't support `refresh=True`
- Documents are eventually consistent (handled in code)

### Search returns no results

- Wait 5-10 seconds after indexing for documents to be searchable
- Check index exists: `aws opensearch list-domain-names --region us-west-2`

## 📝 License

MIT

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📚 Resources

- [AWS OpenSearch Serverless](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless.html)
- [AWS Bedrock](https://docs.aws.amazon.com/bedrock/)
- [Claude Models](https://docs.anthropic.com/claude/docs)
- [Vector Search](https://opensearch.org/docs/latest/search-plugins/knn/index/)

---

**Built with using AWS OpenSearch Serverless & Bedrock**
