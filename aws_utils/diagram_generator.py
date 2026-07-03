"""
Diagram Generator for RAG Patterns
Creates Mermaid flow diagrams for visualization
"""

from typing import List, Dict, Optional


def generate_simple_rag_diagram() -> str:
    """Generate Mermaid diagram for Simple RAG"""
    return """
```mermaid
graph TB
    A[User Query] --> B[Generate Query Embedding<br/>Bedrock Titan]
    B --> C[Vector Search<br/>OpenSearch]
    C --> D[Retrieve Top-K Documents]
    D --> E[Format Context]
    E --> F[Generate Answer<br/>Claude Opus]
    F --> G[Return Answer]

    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style E fill:#fff9c4
    style F fill:#ffe0b2
    style G fill:#c8e6c9
```
"""


def generate_graph_rag_diagram() -> str:
    """Generate Mermaid diagram for Graph RAG"""
    return """
```mermaid
graph TB
    A[Documents] --> B[Extract Entities<br/>Claude]
    B --> C[Build Knowledge Graph]
    C --> D[Store in OpenSearch<br/>Nodes + Edges]

    E[User Query] --> F[Identify Query Entities]
    F --> G[Graph Traversal<br/>Find Connected Nodes]
    G --> H[Retrieve Context]
    H --> I[Generate Answer<br/>Claude]

    D -.-> G

    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style E fill:#e1f5ff
    style F fill:#fff3e0
    style G fill:#f3e5f5
    style H fill:#e8f5e9
    style I fill:#ffe0b2
```
"""


def generate_fusion_retrieval_diagram() -> str:
    """Generate Mermaid diagram for Fusion Retrieval"""
    return """
```mermaid
graph TB
    A[User Query] --> B[Generate Query Variants<br/>Claude]
    B --> C1[Query 1]
    B --> C2[Query 2]
    B --> C3[Query 3]

    C1 --> D1[Retrieve Results 1]
    C2 --> D2[Retrieve Results 2]
    C3 --> D3[Retrieve Results 3]

    D1 --> E[Reciprocal Rank Fusion]
    D2 --> E
    D3 --> E

    E --> F[Merged & Ranked Results]
    F --> G[Generate Answer<br/>Claude]

    style A fill:#e1f5ff
    style B fill:#fff3e0
    style E fill:#f3e5f5
    style F fill:#e8f5e9
    style G fill:#ffe0b2
```
"""


def generate_reranking_diagram() -> str:
    """Generate Mermaid diagram for Reranking"""
    return """
```mermaid
graph TB
    A[User Query] --> B[Initial Retrieval<br/>OpenSearch]
    B --> C[Get Top 20 Candidates]
    C --> D[Rerank with Claude<br/>Relevance Scoring]
    D --> E[Select Top 5]
    E --> F[Generate Answer<br/>Claude]

    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#ffe0b2
    style E fill:#e8f5e9
    style F fill:#c8e6c9
```
"""


def generate_hyde_diagram() -> str:
    """Generate Mermaid diagram for HyDE"""
    return """
```mermaid
graph TB
    A[User Query] --> B[Generate Hypothetical Answer<br/>Claude]
    B --> C[Create Embedding of<br/>Hypothetical Answer]
    C --> D[Vector Search<br/>OpenSearch]
    D --> E[Retrieve Relevant Docs]
    E --> F[Generate Real Answer<br/>Claude + Context]

    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style E fill:#fff9c4
    style F fill:#ffe0b2
```
"""


def generate_custom_diagram(pattern_name: str, steps: List[Dict[str, str]]) -> str:
    """
    Generate custom Mermaid diagram for any RAG pattern

    Args:
        pattern_name: Name of the RAG pattern
        steps: List of {"id": "A", "label": "Step description", "color": "#hex"}

    Returns:
        Mermaid diagram string
    """
    diagram = f"```mermaid\ngraph TB\n"

    # Generate nodes and connections
    for i, step in enumerate(steps):
        node_id = step.get('id', chr(65 + i))  # A, B, C, ...
        label = step['label']

        # Add node
        diagram += f"    {node_id}[{label}]\n"

    # Add connections
    for i in range(len(steps) - 1):
        from_id = steps[i].get('id', chr(65 + i))
        to_id = steps[i + 1].get('id', chr(65 + i + 1))
        diagram += f"    {from_id} --> {to_id}\n"

    # Add styling
    diagram += "\n"
    for i, step in enumerate(steps):
        node_id = step.get('id', chr(65 + i))
        color = step.get('color', '#e1f5ff')
        diagram += f"    style {node_id} fill:{color}\n"

    diagram += "```"
    return diagram


def get_pattern_diagram(pattern_name: str) -> str:
    """
    Get diagram for a specific RAG pattern

    Args:
        pattern_name: Name of the pattern

    Returns:
        Mermaid diagram string
    """
    diagrams = {
        'simple_rag': generate_simple_rag_diagram(),
        'graph_rag': generate_graph_rag_diagram(),
        'fusion_retrieval': generate_fusion_retrieval_diagram(),
        'reranking': generate_reranking_diagram(),
        'hyde': generate_hyde_diagram(),
    }

    return diagrams.get(pattern_name.lower(), generate_simple_rag_diagram())


def generate_comparison_diagram(patterns: List[str]) -> str:
    """Generate comparison diagram for multiple patterns"""
    diagram = """
```mermaid
graph LR
    Q[Query] --> P1[Simple RAG]
    Q --> P2[Graph RAG]
    Q --> P3[Fusion]
    Q --> P4[Reranking]
    Q --> P5[HyDE]

    P1 --> R[Results]
    P2 --> R
    P3 --> R
    P4 --> R
    P5 --> R

    R --> E[Evaluation]

    style Q fill:#e1f5ff
    style R fill:#c8e6c9
    style E fill:#ffe0b2
```
"""
    return diagram


def generate_architecture_overview() -> str:
    """Generate overall RAG architecture with AWS components"""
    return """
```mermaid
graph TB
    subgraph "AWS Bedrock"
        B1[Titan Embeddings<br/>V2 - 1024 dim]
        B2[Claude Opus 4.1<br/>Answer Generation]
    end

    subgraph "AWS OpenSearch Serverless"
        OS1[Vector Index<br/>HNSW]
        OS2[Document Store]
    end

    subgraph "RAG Pipeline"
        R1[Document Processing]
        R2[Chunking]
        R3[Query Processing]
        R4[Retrieval]
        R5[Generation]
    end

    D[Documents] --> R1
    R1 --> R2
    R2 --> B1
    B1 --> OS1
    R2 --> OS2

    Q[Query] --> R3
    R3 --> B1
    B1 --> R4
    R4 --> OS1
    OS1 --> R5
    OS2 --> R5
    R5 --> B2
    B2 --> A[Answer]

    style D fill:#e1f5ff
    style Q fill:#e1f5ff
    style A fill:#c8e6c9
```
"""


# Export main function
generate_mermaid_diagram = get_pattern_diagram
