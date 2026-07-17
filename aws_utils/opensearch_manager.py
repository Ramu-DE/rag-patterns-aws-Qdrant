"""
OpenSearch Manager for RAG Patterns
Handles all OpenSearch Serverless operations.

Auto-fallback: if no OpenSearch endpoint is found or the connection fails,
automatically delegates all operations to QdrantManager (local in-memory or
remote Qdrant server).  All notebooks continue to use OpenSearchManager
with zero code changes.

Fallback priority:
  1. Explicit collection_endpoint argument
  2. config.json on disk (../vector-engine-demos/config.json etc.)
  3. OPENSEARCH_ENDPOINT environment variable
  4. Qdrant fallback (QDRANT_URL env var, or in-memory if not set)
"""

import os
import boto3
import json
from typing import List, Dict, Any, Optional
import time


class OpenSearchManager:
    """
    Manage vector store operations for RAG patterns.
    Backed by OpenSearch Serverless when available, otherwise Qdrant.
    """

    def __init__(self,
                 collection_endpoint: str = None,
                 region: str = 'us-east-1',
                 index_name: str = 'rag_documents'):
        """
        Initialize vector store manager.

        Args:
            collection_endpoint: OpenSearch endpoint URL. If None, tries
                                  config.json, then OPENSEARCH_ENDPOINT env
                                  var, then falls back to Qdrant.
            region: AWS region
            index_name: Index / collection name for documents
        """
        self.region = region
        self.index_name = index_name
        self._backend = None  # 'opensearch' or 'qdrant'

        # --- resolve endpoint ---
        if collection_endpoint is None:
            collection_endpoint = os.environ.get("OPENSEARCH_ENDPOINT")

        if collection_endpoint is None:
            config_paths = [
                '../vector-engine-demos/config.json',
                'config.json',
                '../config.json',
                '../../vector-engine-demos/config.json'
            ]
            for config_path in config_paths:
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        collection_endpoint = config['endpoint']
                        break
                except:
                    continue

        # --- try OpenSearch ---
        if collection_endpoint:
            try:
                self._init_opensearch(collection_endpoint)
                self._backend = 'opensearch'
                return
            except Exception as e:
                print(f"⚠ OpenSearch unavailable ({e}), falling back to Qdrant")

        # --- fall back to Qdrant ---
        self._init_qdrant()
        self._backend = 'qdrant'

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_opensearch(self, collection_endpoint: str):
        from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

        self.endpoint = collection_endpoint
        self.host = collection_endpoint.replace('https://', '').replace('http://', '')

        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, self.region, 'aoss')
        self.client = OpenSearch(
            hosts=[{'host': self.host, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )
        # lightweight ping to confirm connectivity
        self.client.info()
        print(f"✓ OpenSearchManager connected to {collection_endpoint}")

    def _init_qdrant(self):
        from .qdrant_manager import QdrantManager
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")
        self._qdrant = QdrantManager(
            index_name=self.index_name,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        self.endpoint = qdrant_url or ":memory:"
        self.client = None  # not used in qdrant mode

    # ------------------------------------------------------------------
    # Public API — delegates to Qdrant when backend == 'qdrant'
    # ------------------------------------------------------------------

    def create_index(self,
                     embedding_dim: int = 1024,
                     index_name: str = None,
                     force_recreate: bool = False) -> bool:
        if self._backend == 'qdrant':
            return self._qdrant.create_index(embedding_dim, index_name, force_recreate)

        if index_name:
            self.index_name = index_name

        if force_recreate and self.client.indices.exists(index=self.index_name):
            print(f"Deleting existing index: {self.index_name}")
            self.client.indices.delete(index=self.index_name)

        if self.client.indices.exists(index=self.index_name):
            print(f"Index {self.index_name} already exists")
            return True

        index_body = {
            "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 512}},
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "metadata": {"type": "object"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "faiss",
                            "parameters": {"ef_construction": 512, "m": 16}
                        }
                    }
                }
            }
        }
        try:
            self.client.indices.create(index=self.index_name, body=index_body)
            print(f"✓ Created index: {self.index_name}")
            return True
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            return False

    def index_documents(self,
                        documents: List[Dict[str, Any]],
                        batch_size: int = 100) -> int:
        if self._backend == 'qdrant':
            return self._qdrant.index_documents(documents, batch_size)

        success_count = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            for doc in batch:
                try:
                    self.client.index(index=self.index_name, body=doc)
                    success_count += 1
                except Exception as e:
                    print(f"Error indexing document: {e}")
            if (i + batch_size) % 100 == 0:
                print(f"Indexed {min(i + batch_size, len(documents))}/{len(documents)} documents")

        time.sleep(2)
        print(f"✓ Indexed {success_count} documents")
        return success_count

    def vector_search(self,
                      query_embedding: List[float],
                      top_k: int = 5,
                      min_score: float = 0.0,
                      filters: Optional[Dict] = None) -> List[Dict]:
        if self._backend == 'qdrant':
            return self._qdrant.vector_search(query_embedding, top_k, min_score, filters)

        query = {
            "script_score": {
                "query": filters if filters else {"match_all": {}},
                "script": {
                    "source": "knn_score", "lang": "knn",
                    "params": {"field": "embedding", "query_value": query_embedding,
                               "space_type": "cosinesimil"}
                }
            }
        }
        search_body = {"size": top_k, "query": query, "_source": {"excludes": ["embedding"]}}
        try:
            response = self.client.search(index=self.index_name, body=search_body)
            results = []
            for hit in response['hits']['hits']:
                if hit['_score'] >= min_score:
                    results.append({'text': hit['_source'].get('text', ''),
                                    'metadata': hit['_source'].get('metadata', {}),
                                    'score': hit['_score'], 'id': hit['_id']})
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def hybrid_search(self,
                      query_text: str,
                      query_embedding: List[float],
                      top_k: int = 5,
                      semantic_weight: float = 0.7) -> List[Dict]:
        if self._backend == 'qdrant':
            return self._qdrant.hybrid_search(query_text, query_embedding, top_k, semantic_weight)

        keyword_weight = 1.0 - semantic_weight
        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {"script_score": {
                            "query": {"match_all": {}},
                            "script": {"source": "knn_score", "lang": "knn",
                                       "params": {"field": "embedding",
                                                  "query_value": query_embedding,
                                                  "space_type": "cosinesimil"}},
                            "boost": semantic_weight}},
                        {"match": {"text": {"query": query_text, "boost": keyword_weight}}}
                    ]
                }
            },
            "_source": {"excludes": ["embedding"]}
        }
        try:
            response = self.client.search(index=self.index_name, body=search_body)
            return [{'text': h['_source'].get('text', ''),
                     'metadata': h['_source'].get('metadata', {}),
                     'score': h['_score'], 'id': h['_id']}
                    for h in response['hits']['hits']]
        except Exception as e:
            print(f"Error during hybrid search: {e}")
            return []

    def delete_index(self, index_name: str = None):
        if self._backend == 'qdrant':
            return self._qdrant.delete_index(index_name)
        idx = index_name or self.index_name
        try:
            if self.client.indices.exists(index=idx):
                self.client.indices.delete(index=idx)
                print(f"✓ Deleted index: {idx}")
        except Exception as e:
            print(f"Error deleting index: {e}")

    def get_document_count(self, index_name: str = None) -> int:
        if self._backend == 'qdrant':
            return self._qdrant.get_document_count(index_name)
        idx = index_name or self.index_name
        try:
            return self.client.count(index=idx)['count']
        except:
            return 0

    def get_document_by_id(self, doc_id: str, index_name: str = None) -> Optional[Dict]:
        if self._backend == 'qdrant':
            return self._qdrant.get_document_by_id(doc_id, index_name)
        idx = index_name or self.index_name
        try:
            return self.client.get(index=idx, id=doc_id)['_source']
        except:
            return None
