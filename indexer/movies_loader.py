#!/usr/bin/env python3
"""
Movie Data Loader with Vector Embeddings
Loads movie data into OpenSearch Serverless with Bedrock embeddings
"""

import boto3
import json
import sys
import os
from typing import List, Dict
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from requests_aws4auth import AWS4Auth

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.bedrock_embeddings import BedrockEmbeddings


class MovieIndexer:
    """Index movies with vector embeddings into OpenSearch"""

    def __init__(self, config_path: str = 'config.json'):
        """Initialize indexer with configuration"""

        # Load OpenSearch config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.index_name = 'movies'
        self.region = self.config.get('region', 'us-west-2')

        # Initialize Bedrock embeddings
        print("Initializing Bedrock embeddings...")
        self.embedder = BedrockEmbeddings(region_name=self.region)
        model_info = self.embedder.get_model_info()
        print(f"Using embedding model: {model_info['model_id']}")
        print(f"Embedding dimension: {model_info['embedding_dimension']}")

        # Initialize OpenSearch client
        print("Connecting to OpenSearch...")
        self.client = self._get_opensearch_client()

    def _get_opensearch_client(self) -> OpenSearch:
        """Create OpenSearch client with AWS auth"""

        # Extract host from endpoint
        endpoint = self.config['endpoint']
        host = endpoint.replace('https://', '').replace('http://', '')

        # Get AWS credentials
        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, self.region, 'aoss')

        client = OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )

        return client

    def create_index(self):
        """Create index with vector field mapping"""

        embedding_dim = self.embedder.get_embedding_dimension()

        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 512
                }
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "year": {"type": "integer"},
                    "genre": {"type": "keyword"},
                    "director": {"type": "text"},
                    "actors": {"type": "text"},
                    "plot": {"type": "text"},
                    "rating": {"type": "float"},
                    "runtime": {"type": "integer"},
                    "plot_embedding": {
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
                    },
                    "combined_text": {"type": "text"}
                }
            }
        }

        try:
            if self.client.indices.exists(index=self.index_name):
                print(f"Index '{self.index_name}' already exists. Deleting...")
                self.client.indices.delete(index=self.index_name)

            print(f"Creating index '{self.index_name}'...")
            response = self.client.indices.create(
                index=self.index_name,
                body=index_body
            )
            print(f"✓ Index created successfully")
            return True

        except Exception as e:
            print(f"❌ Error creating index: {e}")
            return False

    def prepare_movie_text(self, movie: Dict) -> str:
        """Prepare combined text for embedding"""

        # Combine relevant fields for rich semantic search
        parts = [
            f"Title: {movie['title']}",
            f"Plot: {movie['plot']}",
            f"Genre: {', '.join(movie['genre']) if isinstance(movie['genre'], list) else movie['genre']}",
            f"Director: {movie['director'] if isinstance(movie['director'], str) else ', '.join(movie['director'])}",
            f"Actors: {', '.join(movie['actors']) if isinstance(movie['actors'], list) else movie['actors']}"
        ]

        return " | ".join(parts)

    def index_movies(self, movies_file: str, batch_size: int = 10):
        """Index movies with embeddings"""

        # Load movies
        print(f"\nLoading movies from {movies_file}...")
        with open(movies_file, 'r') as f:
            movies = json.load(f)

        print(f"Loaded {len(movies)} movies")

        # Prepare texts for embedding
        print("\nPreparing movie texts...")
        movie_texts = [self.prepare_movie_text(m) for m in movies]

        # Generate embeddings in batches
        print("\nGenerating embeddings with Bedrock...")
        embeddings = self.embedder.embed_batch(
            movie_texts,
            input_type="search_document",
            batch_size=batch_size
        )

        # Index documents
        print(f"\nIndexing {len(movies)} movies...")
        success_count = 0
        error_count = 0

        for i, (movie, embedding) in enumerate(zip(movies, embeddings)):
            try:
                doc = {
                    **movie,
                    'plot_embedding': embedding,
                    'combined_text': movie_texts[i]
                }

                # OpenSearch Serverless doesn't support custom IDs or refresh
                self.client.index(
                    index=self.index_name,
                    body=doc
                )
                success_count += 1

                if (i + 1) % 10 == 0:
                    print(f"  Indexed {i + 1}/{len(movies)} movies...")

            except Exception as e:
                print(f"❌ Error indexing movie {movie.get('title', 'Unknown')}: {e}")
                error_count += 1

        print(f"\n✅ Indexing complete!")
        print(f"  Success: {success_count}")
        print(f"  Errors: {error_count}")

        return success_count, error_count

    def verify_index(self):
        """Verify index and show stats"""

        try:
            # Get index stats
            stats = self.client.indices.stats(index=self.index_name)
            doc_count = stats['indices'][self.index_name]['total']['docs']['count']

            print(f"\n📊 Index Statistics:")
            print(f"  Index: {self.index_name}")
            print(f"  Document count: {doc_count}")

            # Sample document
            if doc_count > 0:
                response = self.client.search(
                    index=self.index_name,
                    body={"query": {"match_all": {}}, "size": 1}
                )

                if response['hits']['hits']:
                    sample = response['hits']['hits'][0]['_source']
                    print(f"\n📄 Sample document:")
                    print(f"  Title: {sample.get('title')}")
                    print(f"  Year: {sample.get('year')}")
                    print(f"  Embedding dimension: {len(sample.get('plot_embedding', []))}")

            return True

        except Exception as e:
            print(f"❌ Error verifying index: {e}")
            return False


def main():
    """Main execution"""

    print("=" * 70)
    print("Movie Data Loader with Vector Embeddings")
    print("=" * 70)
    print()

    # Check for config file
    config_path = 'config.json'
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("Please run create_opensearch.py first")
        sys.exit(1)

    # Initialize indexer
    try:
        indexer = MovieIndexer(config_path)
    except Exception as e:
        print(f"❌ Failed to initialize indexer: {e}")
        sys.exit(1)

    # Create index
    if not indexer.create_index():
        sys.exit(1)

    # Index movies
    movies_file = 'data/sample-movies-small.json'
    success, errors = indexer.index_movies(movies_file)

    if success > 0:
        # Verify
        indexer.verify_index()

        print("\n" + "=" * 70)
        print("✅ Movie indexing complete!")
        print("=" * 70)
        print("\nNext: Run the Streamlit app for semantic search")
        print("  streamlit run 0_Home.py")
    else:
        print("\n❌ Indexing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
