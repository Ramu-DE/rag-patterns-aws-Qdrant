"""
Qdrant backend for RAG Patterns — same interface as OpenSearchManager.
Used as fallback when no OpenSearch endpoint is available.
Supports both local in-memory mode and a remote Qdrant server.
"""

import uuid
import time
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    ScoredPoint
)


class QdrantManager:
    """
    Drop-in replacement for OpenSearchManager backed by Qdrant.
    Constructor signature is intentionally compatible so OpenSearchManager
    can delegate to this class transparently.
    """

    def __init__(self,
                 collection_endpoint: str = None,
                 region: str = 'us-east-1',
                 index_name: str = 'rag_documents',
                 qdrant_url: str = None,
                 qdrant_api_key: str = None):
        """
        Args:
            collection_endpoint: ignored (kept for interface compatibility)
            region: ignored (kept for interface compatibility)
            index_name: Qdrant collection name
            qdrant_url: Remote Qdrant URL e.g. http://localhost:6333.
                        If None, uses in-memory mode.
            qdrant_api_key: API key for Qdrant Cloud
        """
        self.index_name = index_name
        self.region = region
        self.endpoint = qdrant_url or ":memory:"
        self._embedding_dim = 1024

        if qdrant_url:
            kwargs = {"url": qdrant_url}
            if qdrant_api_key:
                kwargs["api_key"] = qdrant_api_key
            self.client = QdrantClient(**kwargs)
            print(f"✓ QdrantManager connected to {qdrant_url}")
        else:
            self.client = QdrantClient(":memory:")
            print("✓ QdrantManager using in-memory Qdrant (data lost on restart)")

    # ------------------------------------------------------------------
    # Index / Collection management
    # ------------------------------------------------------------------

    def create_index(self,
                     embedding_dim: int = 1024,
                     index_name: str = None,
                     force_recreate: bool = False) -> bool:
        if index_name:
            self.index_name = index_name
        self._embedding_dim = embedding_dim

        existing = [c.name for c in self.client.get_collections().collections]

        if self.index_name in existing:
            if force_recreate:
                print(f"Deleting existing collection: {self.index_name}")
                self.client.delete_collection(self.index_name)
            else:
                print(f"Collection {self.index_name} already exists")
                return True

        try:
            self.client.create_collection(
                collection_name=self.index_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
            )
            print(f"✓ Created collection: {self.index_name}")
            return True
        except Exception as e:
            print(f"❌ Error creating collection: {e}")
            return False

    def delete_index(self, index_name: str = None):
        name = index_name or self.index_name
        try:
            self.client.delete_collection(name)
            print(f"✓ Deleted collection: {name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(self,
                        documents: List[Dict[str, Any]],
                        batch_size: int = 100) -> int:
        success_count = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            points = []
            for doc in batch:
                embedding = doc.get("embedding")
                if not embedding:
                    continue
                payload = {
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {})
                }
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload
                ))

            try:
                self.client.upsert(collection_name=self.index_name, points=points)
                success_count += len(points)
            except Exception as e:
                print(f"Error indexing batch: {e}")

            indexed = min(i + batch_size, len(documents))
            if indexed % 100 == 0 or indexed == len(documents):
                print(f"Indexed {indexed}/{len(documents)} documents")

        time.sleep(0.1)
        print(f"✓ Indexed {success_count} documents")
        return success_count

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def vector_search(self,
                      query_embedding: List[float],
                      top_k: int = 5,
                      min_score: float = 0.0,
                      filters: Optional[Dict] = None) -> List[Dict]:
        qdrant_filter = self._build_filter(filters) if filters else None

        try:
            resp = self.client.query_points(
                collection_name=self.index_name,
                query=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                score_threshold=min_score if min_score > 0.0 else None,
                with_payload=True
            )
            return [self._hit_to_dict(h) for h in resp.points]
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def hybrid_search(self,
                      query_text: str,
                      query_embedding: List[float],
                      top_k: int = 5,
                      semantic_weight: float = 0.7) -> List[Dict]:
        """
        Qdrant doesn't have native BM25, so we approximate hybrid search by:
        1. Running vector search for top_k * 3 candidates
        2. Re-scoring by keyword overlap
        3. Blending scores and returning top_k
        """
        try:
            resp = self.client.query_points(
                collection_name=self.index_name,
                query=query_embedding,
                limit=top_k * 3,
                with_payload=True
            )
            candidates = resp.points

            query_terms = set(query_text.lower().split())
            keyword_weight = 1.0 - semantic_weight
            results = []

            for hit in candidates:
                text = hit.payload.get("text", "")
                text_terms = set(text.lower().split())
                overlap = len(query_terms & text_terms)
                kw_score = overlap / max(len(query_terms), 1)
                blended = semantic_weight * hit.score + keyword_weight * kw_score
                results.append({**self._hit_to_dict(hit), "score": blended})

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"Error during hybrid search: {e}")
            return []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_document_count(self, index_name: str = None) -> int:
        name = index_name or self.index_name
        try:
            info = self.client.get_collection(name)
            return info.points_count or 0
        except:
            return 0

    def get_document_by_id(self, doc_id: str, index_name: str = None) -> Optional[Dict]:
        name = index_name or self.index_name
        try:
            results = self.client.retrieve(
                collection_name=name,
                ids=[doc_id],
                with_payload=True
            )
            if results:
                return results[0].payload
            return None
        except:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hit_to_dict(self, hit: ScoredPoint) -> Dict:
        return {
            "text": hit.payload.get("text", ""),
            "metadata": hit.payload.get("metadata", {}),
            "score": hit.score,
            "id": str(hit.id)
        }

    def _build_filter(self, filters: Dict) -> Filter:
        """Convert a simple dict filter to Qdrant Filter (best-effort)."""
        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict):
                gte = value.get("gte")
                lte = value.get("lte")
                if gte is not None or lte is not None:
                    conditions.append(FieldCondition(
                        key=f"metadata.{key}",
                        range=Range(gte=gte, lte=lte)
                    ))
            else:
                conditions.append(FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=value)
                ))
        return Filter(must=conditions) if conditions else None
