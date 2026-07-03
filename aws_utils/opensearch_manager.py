"""
OpenSearch Manager for RAG Patterns
Handles all OpenSearch Serverless operations
"""

import boto3
import json
from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import time


class OpenSearchManager:
    """Manage OpenSearch Serverless operations for RAG patterns"""

    def __init__(self,
                 collection_endpoint: str = None,
                 region: str = 'us-west-2',
                 index_name: str = 'rag_documents'):
        """
        Initialize OpenSearch manager

        Args:
            collection_endpoint: OpenSearch endpoint (from config.json)
            region: AWS region
            index_name: Index name for documents
        """
        self.region = region
        self.index_name = index_name

        # Load from config if endpoint not provided
        if collection_endpoint is None:
            try:
                with open('../vector-engine-demos/config.json', 'r') as f:
                    config = json.load(f)
                    collection_endpoint = config['endpoint']
            except:
                raise ValueError("No OpenSearch endpoint provided and config.json not found")

        self.endpoint = collection_endpoint
        self.host = collection_endpoint.replace('https://', '').replace('http://', '')

        # Initialize client
        self.client = self._get_client()

    def _get_client(self) -> OpenSearch:
        """Create OpenSearch client with AWS auth"""
        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, self.region, 'aoss')

        client = OpenSearch(
            hosts=[{'host': self.host, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )
        return client

    def create_index(self,
                     embedding_dim: int = 1024,
                     index_name: str = None,
                     force_recreate: bool = False) -> bool:
        """
        Create vector index

        Args:
            embedding_dim: Dimension of embeddings
            index_name: Override default index name
            force_recreate: Delete existing index if exists

        Returns:
            Success status
        """
        if index_name:
            self.index_name = index_name

        # Delete if exists and force recreate
        if force_recreate and self.client.indices.exists(index=self.index_name):
            print(f"Deleting existing index: {self.index_name}")
            self.client.indices.delete(index=self.index_name)

        if self.client.indices.exists(index=self.index_name):
            print(f"Index {self.index_name} already exists")
            return True

        # Create index with vector field
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 512
                }
            },
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
                            "parameters": {
                                "ef_construction": 512,
                                "m": 16
                            }
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
        """
        Index documents with embeddings

        Args:
            documents: List of dicts with 'text', 'embedding', 'metadata'
            batch_size: Batch size for indexing

        Returns:
            Number of documents indexed
        """
        success_count = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            for doc in batch:
                try:
                    self.client.index(
                        index=self.index_name,
                        body=doc
                    )
                    success_count += 1
                except Exception as e:
                    print(f"Error indexing document: {e}")

            if (i + batch_size) % 100 == 0:
                print(f"Indexed {min(i + batch_size, len(documents))}/{len(documents)} documents")

        # Wait for documents to be searchable
        time.sleep(2)

        print(f"✓ Indexed {success_count} documents")
        return success_count

    def vector_search(self,
                     query_embedding: List[float],
                     top_k: int = 5,
                     min_score: float = 0.0,
                     filters: Optional[Dict] = None) -> List[Dict]:
        """
        Perform vector similarity search

        Args:
            query_embedding: Query vector
            top_k: Number of results
            min_score: Minimum similarity score
            filters: Optional metadata filters

        Returns:
            List of matching documents with scores
        """
        query = {
            "script_score": {
                "query": filters if filters else {"match_all": {}},
                "script": {
                    "source": "knn_score",
                    "lang": "knn",
                    "params": {
                        "field": "embedding",
                        "query_value": query_embedding,
                        "space_type": "cosinesimil"
                    }
                }
            }
        }

        search_body = {
            "size": top_k,
            "query": query,
            "_source": {"excludes": ["embedding"]}
        }

        try:
            response = self.client.search(index=self.index_name, body=search_body)

            results = []
            for hit in response['hits']['hits']:
                if hit['_score'] >= min_score:
                    results.append({
                        'text': hit['_source'].get('text', ''),
                        'metadata': hit['_source'].get('metadata', {}),
                        'score': hit['_score'],
                        'id': hit['_id']
                    })

            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def hybrid_search(self,
                     query_text: str,
                     query_embedding: List[float],
                     top_k: int = 5,
                     semantic_weight: float = 0.7) -> List[Dict]:
        """
        Hybrid search combining vector and text search

        Args:
            query_text: Text query
            query_embedding: Query vector
            top_k: Number of results
            semantic_weight: Weight for semantic vs text (0-1)

        Returns:
            List of matching documents
        """
        keyword_weight = 1.0 - semantic_weight

        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "knn_score",
                                    "lang": "knn",
                                    "params": {
                                        "field": "embedding",
                                        "query_value": query_embedding,
                                        "space_type": "cosinesimil"
                                    }
                                },
                                "boost": semantic_weight
                            }
                        },
                        {
                            "match": {
                                "text": {
                                    "query": query_text,
                                    "boost": keyword_weight
                                }
                            }
                        }
                    ]
                }
            },
            "_source": {"excludes": ["embedding"]}
        }

        try:
            response = self.client.search(index=self.index_name, body=search_body)

            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'text': hit['_source'].get('text', ''),
                    'metadata': hit['_source'].get('metadata', {}),
                    'score': hit['_score'],
                    'id': hit['_id']
                })

            return results
        except Exception as e:
            print(f"Error during hybrid search: {e}")
            return []

    def delete_index(self, index_name: str = None):
        """Delete an index"""
        idx = index_name or self.index_name
        try:
            if self.client.indices.exists(index=idx):
                self.client.indices.delete(index=idx)
                print(f"✓ Deleted index: {idx}")
        except Exception as e:
            print(f"Error deleting index: {e}")

    def get_document_count(self, index_name: str = None) -> int:
        """Get total document count in index"""
        idx = index_name or self.index_name
        try:
            response = self.client.count(index=idx)
            return response['count']
        except:
            return 0

    def get_document_by_id(self, doc_id: str, index_name: str = None) -> Optional[Dict]:
        """Retrieve document by ID"""
        idx = index_name or self.index_name
        try:
            response = self.client.get(index=idx, id=doc_id)
            return response['_source']
        except:
            return None
