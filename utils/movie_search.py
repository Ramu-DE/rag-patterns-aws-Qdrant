"""
Movie Search with Semantic Vector Search and Claude RAG
Supports hybrid search (keyword + semantic) with AI-powered answers
"""

import boto3
import json
from typing import List, Dict, Optional, Tuple
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from .bedrock_embeddings import BedrockEmbeddings


class MovieSearch:
    """Semantic movie search with Claude RAG"""

    def __init__(self, config: Dict, index_name: str = 'movies'):
        """Initialize search client"""

        self.config = config
        self.index_name = index_name
        self.region = config.get('region', 'us-west-2')

        # Initialize components
        self.embedder = BedrockEmbeddings(region_name=self.region)
        self.client = self._get_opensearch_client()
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=self.region)

    def _get_opensearch_client(self) -> OpenSearch:
        """Create OpenSearch client"""

        endpoint = self.config['endpoint']
        host = endpoint.replace('https://', '').replace('http://', '')

        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, self.region, 'aoss')

        return OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> List[Dict]:
        """
        Pure semantic search using vector similarity

        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)

        Returns:
            List of movie documents with scores
        """

        # Generate query embedding
        query_embedding = self.embedder.embed_text(query, input_type="search_query")

        # KNN search using script_score
        search_body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "knn_score",
                        "lang": "knn",
                        "params": {
                            "field": "plot_embedding",
                            "query_value": query_embedding,
                            "space_type": "cosinesimil"
                        }
                    }
                }
            },
            "_source": {
                "excludes": ["plot_embedding"]
            }
        }

        response = self.client.search(index=self.index_name, body=search_body)

        results = []
        for hit in response['hits']['hits']:
            score = hit['_score']
            if score >= min_score:
                results.append({
                    **hit['_source'],
                    'search_score': score
                })

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7
    ) -> List[Dict]:
        """
        Hybrid search combining semantic + keyword matching

        Args:
            query: Search query
            top_k: Number of results
            semantic_weight: Weight for semantic vs keyword (0-1)

        Returns:
            List of movie documents with combined scores
        """

        # Generate query embedding
        query_embedding = self.embedder.embed_text(query, input_type="search_query")

        keyword_weight = 1.0 - semantic_weight

        # Use script_score for vector + text matching
        search_body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "should": [
                                {"match_all": {}},
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["title^3", "plot^2", "genre", "director", "actors"],
                                        "type": "best_fields"
                                    }
                                }
                            ]
                        }
                    },
                    "script": {
                        "source": "knn_score",
                        "lang": "knn",
                        "params": {
                            "field": "plot_embedding",
                            "query_value": query_embedding,
                            "space_type": "cosinesimil"
                        }
                    }
                }
            },
            "_source": {
                "excludes": ["plot_embedding"]
            }
        }

        response = self.client.search(index=self.index_name, body=search_body)

        results = []
        for hit in response['hits']['hits']:
            results.append({
                **hit['_source'],
                'search_score': hit['_score']
            })

        return results

    def filter_search(
        self,
        query: str,
        genre: Optional[List[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        min_rating: Optional[float] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Semantic search with filters

        Args:
            query: Search query
            genre: Filter by genre(s)
            min_year: Minimum year
            max_year: Maximum year
            min_rating: Minimum rating
            top_k: Number of results

        Returns:
            Filtered search results
        """

        query_embedding = self.embedder.embed_text(query, input_type="search_query")

        # Build filter clauses
        filters = []

        if genre:
            filters.append({"terms": {"genre": genre}})

        if min_year or max_year:
            year_range = {}
            if min_year:
                year_range["gte"] = min_year
            if max_year:
                year_range["lte"] = max_year
            filters.append({"range": {"year": year_range}})

        if min_rating:
            filters.append({"range": {"rating": {"gte": min_rating}}})

        # Build query with filters
        search_body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": [{"match_all": {}}],
                            "filter": filters
                        }
                    },
                    "script": {
                        "source": "knn_score",
                        "lang": "knn",
                        "params": {
                            "field": "plot_embedding",
                            "query_value": query_embedding,
                            "space_type": "cosinesimil"
                        }
                    }
                }
            },
            "_source": {
                "excludes": ["plot_embedding"]
            }
        }

        response = self.client.search(index=self.index_name, body=search_body)

        results = []
        for hit in response['hits']['hits']:
            results.append({
                **hit['_source'],
                'search_score': hit['_score']
            })

        return results[:top_k]

    def generate_answer_with_claude(
        self,
        query: str,
        search_results: List[Dict],
        model_id: str = "anthropic.claude-sonnet-4-6"
    ) -> Dict[str, str]:
        """
        Generate natural language answer using Claude with RAG

        Args:
            query: User's question
            search_results: Retrieved movie documents
            model_id: Claude model to use

        Returns:
            Dict with answer and reasoning
        """

        # Build context from search results
        context_parts = []
        for i, movie in enumerate(search_results, 1):
            context_parts.append(
                f"{i}. **{movie['title']}** ({movie['year']})\n"
                f"   - Genre: {', '.join(movie['genre']) if isinstance(movie['genre'], list) else movie['genre']}\n"
                f"   - Director: {movie['director']}\n"
                f"   - Rating: {movie['rating']}/10\n"
                f"   - Plot: {movie['plot']}\n"
            )

        context = "\n".join(context_parts)

        # Build prompt
        prompt = f"""You are a helpful movie recommendation assistant. Based on the following movies from our database, answer the user's question.

Search Results:
{context}

User Question: {query}

Provide a helpful, conversational answer. If recommending movies, explain why they match the query. Be specific and reference the movies by title."""

        # Call Claude
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })

        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body
            )

            response_body = json.loads(response['body'].read())
            answer = response_body['content'][0]['text']

            return {
                "answer": answer,
                "model": model_id,
                "num_sources": len(search_results)
            }

        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "model": model_id,
                "num_sources": len(search_results)
            }

    def search_and_answer(
        self,
        query: str,
        search_type: str = "hybrid",
        top_k: int = 5,
        use_claude: bool = True,
        claude_model: str = "anthropic.claude-sonnet-4-6"
    ) -> Tuple[List[Dict], Optional[Dict]]:
        """
        Complete search with optional AI answer generation

        Args:
            query: Search query
            search_type: "semantic", "hybrid", or "keyword"
            top_k: Number of results
            use_claude: Whether to generate answer with Claude
            claude_model: Claude model to use

        Returns:
            Tuple of (search_results, claude_answer)
        """

        # Perform search
        if search_type == "semantic":
            results = self.semantic_search(query, top_k=top_k)
        elif search_type == "hybrid":
            results = self.hybrid_search(query, top_k=top_k)
        else:
            raise ValueError(f"Unknown search type: {search_type}")

        # Generate answer if requested
        answer = None
        if use_claude and results:
            answer = self.generate_answer_with_claude(query, results, model_id=claude_model)

        return results, answer
