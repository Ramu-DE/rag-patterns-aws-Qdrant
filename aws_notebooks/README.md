# AWS RAG Pattern Notebooks

Comprehensive collection of Retrieval-Augmented Generation (RAG) patterns converted to AWS stack.

## 🏗️ AWS Stack

All notebooks use:
- **AWS OpenSearch Serverless**: Vector storage and retrieval
- **Amazon Bedrock Titan**: Text embeddings (1024-dim)
- **Amazon Bedrock Claude Opus**: Answer generation and reasoning
- **boto3 + opensearch-py**: AWS SDK integration

## 📚 Available Patterns

### ✅ Completed (10/37) - Phase 1 COMPLETE!

| # | Pattern | Description | Complexity | Best For |
|---|---------|-------------|------------|----------|
| 01 | **Simple RAG** | Foundational RAG pattern | ⭐ Basic | Getting started, baseline |
| 02 | **Graph RAG** | Knowledge graph-based retrieval | ⭐⭐⭐ Advanced | Multi-hop reasoning, relationships |
| 03 | **Fusion Retrieval** | Multi-query with RRF merging | ⭐⭐ Intermediate | Complex queries, better recall |
| 04 | **Reranking** | Two-stage retrieval with LLM scoring | ⭐⭐ Intermediate | Precision-critical applications |
| 05 | **HyDE** | Hypothetical document embeddings | ⭐⭐ Intermediate | Vocabulary mismatch, technical domains |
| 06 | **Contextual Compression** | Extract query-relevant snippets | ⭐⭐ Intermediate | Long documents, token reduction |
| 07 | **Semantic Chunking** | Meaning-based chunk boundaries | ⭐⭐ Intermediate | Context preservation, quality |
| 08 | **Adaptive RAG** | Intelligent strategy routing | ⭐⭐⭐ Advanced | Production, cost optimization |
| 09 | **Query Decomposition** | Break complex queries into sub-questions | ⭐⭐ Intermediate | Multi-faceted questions, comparisons |
| 10 | **Recursive RAG** | Iterative refinement with follow-ups | ⭐⭐⭐ Advanced | Research queries, deep exploration |

### 🚧 Remaining (27/37)

Remaining patterns to convert (32):
- Contextual Compression
- Semantic Chunking
- Adaptive RAG
- Multi-modal RAG
- Agentic RAG
- Corrective RAG
- Self-RAG
- Query Decomposition
- Recursive RAG
- Prompt Compression
- Long Context RAG
- Parallel RAG
- Sequential RAG
- Tree of Thoughts RAG
- Chain of Thought RAG
- ReAct RAG
- LangGraph RAG
- Memory-augmented RAG
- Ensemble RAG
- Iterative RAG
- Few-shot RAG
- Zero-shot RAG
- Cross-lingual RAG
- Multi-document RAG
- Hierarchical RAG
- Streaming RAG
- Batch RAG
- Real-time RAG
- Cached RAG
- Filtered RAG
- Weighted RAG
- And 1 more...

## 🎯 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install boto3 opensearch-py langchain matplotlib networkx pandas numpy

# Configure AWS credentials
aws configure
```

### Running a Notebook

```bash
# Launch Jupyter
jupyter notebook

# Open any notebook (e.g., 01_Simple_RAG_AWS.ipynb)
# Run cells sequentially
```

### Configuration Required

Before running notebooks, ensure you have:

1. **OpenSearch Collection**: Create via `vector-engine-demos/infrastructure/create_opensearch.py`
2. **Bedrock Access**: Enable Titan and Claude models in your AWS account
3. **Config File**: `vector-engine-demos/config.json` with OpenSearch endpoint

## 📖 Notebook Structure

Each notebook follows a consistent format:

### 1️⃣ Overview
- Pattern description
- Use cases (when to use / when not to use)
- Architecture diagram (Mermaid)

### 2️⃣ Setup
- Imports
- Configuration
- AWS service initialization

### 3️⃣ Implementation
- Core pattern logic
- Code with detailed explanations
- Helper functions

### 4️⃣ Demonstration
- Example queries
- Results visualization
- Performance metrics

### 5️⃣ Evaluation
- Quality metrics
- Latency measurement
- Cost estimation

### 6️⃣ Comparison
- vs Simple RAG (when applicable)
- Trade-offs analysis

### 7️⃣ Summary
- Key takeaways
- Limitations
- Best practices
- Next steps

## 🔧 Shared Utilities

All notebooks use the `aws_utils` module:

```python
from aws_utils.opensearch_manager import OpenSearchManager
from aws_utils.bedrock_client import BedrockEmbeddings, BedrockLLM, BedrockRAG
from aws_utils.rag_evaluator import RAGEvaluator
from aws_utils.diagram_generator import generate_mermaid_diagram
```

### Key Components

- **OpenSearchManager**: Index creation, document indexing, vector/hybrid search
- **BedrockEmbeddings**: Titan embedding generation (batch + single)
- **BedrockLLM**: Claude model invocation with context
- **BedrockRAG**: Complete RAG pipeline (index + query)
- **RAGEvaluator**: Metrics (precision, recall, F1, MRR, NDCG, latency)

## 💰 Cost Considerations

### Per-Query Costs (Approximate)

| Pattern | Embeddings | LLM Calls | Total Cost |
|---------|-----------|-----------|------------|
| Simple RAG | 1 Titan | 1 Claude Opus | ~$0.05 |
| Graph RAG | 1 Titan | 2-3 Claude | ~$0.10-0.15 |
| Fusion Retrieval | 4 Titan | 2 Claude | ~$0.06 |

### Cost Optimization Tips

1. **Use Haiku for simple tasks**: 20x cheaper than Opus
2. **Batch embeddings**: Reduce API overhead
3. **Cache results**: Reuse for similar queries
4. **Hybrid search**: Combine with keyword search (free)
5. **Adaptive complexity**: Start simple, escalate if needed

## ⚡ Performance Benchmarks

Typical latencies on sample data (15 documents):

| Pattern | Indexing | Query | Total |
|---------|----------|-------|-------|
| Simple RAG | ~5s | ~1-2s | ~6-7s |
| Graph RAG | ~15s | ~3-5s | ~18-20s |
| Fusion Retrieval | ~5s | ~3-4s | ~8-9s |

**Note**: Latencies scale with document count and query complexity.

## 🎓 Learning Path

**Recommended order for learning:**

1. **Start**: `01_Simple_RAG_AWS.ipynb` - Understand basics
2. **Intermediate**: `03_Fusion_Retrieval_AWS.ipynb` - Improve recall
3. **Advanced**: `02_Graph_RAG_AWS.ipynb` - Multi-hop reasoning
4. **Next**: Reranking, HyDE, Contextual Compression (coming soon)

## 🔍 Pattern Selection Guide

### By Query Type

- **Factual lookup**: Simple RAG
- **Relationship questions**: Graph RAG
- **Complex/ambiguous**: Fusion Retrieval
- **Precision critical**: Reranking (coming soon)
- **Long documents**: Contextual Compression (coming soon)

### By Constraints

- **Speed critical**: Simple RAG
- **Budget critical**: Simple RAG with Haiku
- **Quality critical**: Fusion + Reranking
- **Scale critical**: Hierarchical RAG (coming soon)

## 🛠️ Customization

### Changing Models

```python
# Use Haiku for faster/cheaper responses
LLM_MODEL = 'us.anthropic.claude-haiku-3-20241022-v1:0'

# Use different embedding model (if available)
EMBEDDING_MODEL = 'cohere.embed-english-v3'  # Requires access
```

### Adjusting Parameters

```python
# Simple RAG
CHUNK_SIZE = 500  # Smaller chunks = more precise
TOP_K = 10  # More results = better recall

# Fusion Retrieval
NUM_QUERY_VARIANTS = 3  # Fewer = faster
RRF_K = 60  # Standard value

# Graph RAG
MAX_HOPS = 3  # More hops = more context
```

## 📊 Evaluation Metrics

All notebooks include evaluation:

### Retrieval Metrics
- **Precision**: Relevant docs / Retrieved docs
- **Recall**: Retrieved relevant / Total relevant
- **F1 Score**: Harmonic mean of precision & recall
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain

### Performance Metrics
- **Latency**: Time to generate answer
- **Cost**: API call estimation
- **Throughput**: Queries per second

### Quality Metrics
- **Answer quality**: LLM-as-judge evaluation
- **Faithfulness**: Answer grounded in context
- **Relevance**: Answer addresses question

## 🐛 Troubleshooting

### Common Issues

**1. OpenSearch connection error**
```python
# Check endpoint in config.json
# Ensure IAM permissions are correct
# Verify security policies in OpenSearch console
```

**2. Bedrock model access**
```bash
# Enable models in Bedrock console
# Use correct model IDs with region prefix
```

**3. Rate limiting**
```python
# Add delays between requests
time.sleep(0.1)  # Between embedding calls
```

**4. Memory issues**
```python
# Process in smaller batches
batch_size = 10  # Reduce if needed
```

## 📚 Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [OpenSearch Serverless Guide](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless.html)
- [RAG Papers Collection](https://github.com/hymie122/RAG-Survey)
- [LangChain Documentation](https://python.langchain.com/docs/)

## 🤝 Contributing

Contributions welcome! To add a new pattern:

1. Follow the standard notebook structure
2. Include detailed explanations for each cell
3. Add Mermaid architecture diagram
4. Provide evaluation metrics
5. Document when to use / when not to use

## 📝 License

MIT License - See repository root for details

---

**Status**: 10/37 patterns converted (27%) | Last updated: 2026-07-03

## 📈 Conversion Progress

```
Progress: [██████████░░░░░░░░░░░░░░░░░░░░░░░░░] 27%

Phase 1 - Core Patterns (10/10): ✅ COMPLETE!
  ✅ Simple RAG
  ✅ Graph RAG  
  ✅ Fusion Retrieval
  ✅ Reranking
  ✅ HyDE
  ✅ Contextual Compression
  ✅ Semantic Chunking
  ✅ Adaptive RAG
  ✅ Query Decomposition
  ✅ Recursive RAG

Phase 2 - Advanced Patterns (0/12): Ready to start
Phase 3 - Specialized (0/10): Not started
Phase 4 - Production (0/5): Not started
```
