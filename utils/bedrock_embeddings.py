"""
Bedrock embedding utilities for vector search
Supports multiple embedding models with intelligent selection
"""

import boto3
import json
from typing import List, Dict, Optional
import time

class BedrockEmbeddings:
    """Generate embeddings using AWS Bedrock models"""

    def __init__(
        self,
        region_name: str = 'us-west-2',
        model_id: Optional[str] = None
    ):
        self.client = boto3.client('bedrock-runtime', region_name=region_name)
        self.region = region_name

        # Auto-select best embedding model if not specified
        if model_id is None:
            # Amazon Titan V2 - good quality, AWS native
            self.model_id = 'amazon.titan-embed-text-v2:0'
            self.embedding_dim = 1024
            self.model_provider = 'amazon'
        else:
            self.model_id = model_id
            self._set_model_config()

    def _set_model_config(self):
        """Set model-specific configuration"""
        if 'cohere' in self.model_id:
            self.model_provider = 'cohere'
            if 'v4' in self.model_id:
                self.embedding_dim = 1024
            else:
                self.embedding_dim = 1024
        elif 'titan' in self.model_id:
            self.model_provider = 'amazon'
            if 'v2' in self.model_id:
                self.embedding_dim = 1024
            else:
                self.embedding_dim = 1536

    def embed_text(
        self,
        text: str,
        input_type: str = "search_document"
    ) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed
            input_type: Type of input - "search_document" or "search_query"

        Returns:
            List of floats representing the embedding vector
        """
        if self.model_provider == 'cohere':
            body = json.dumps({
                "texts": [text],
                "input_type": input_type,
                "truncate": "END"
            })
        elif self.model_provider == 'amazon':
            body = json.dumps({
                "inputText": text
            })
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )

            response_body = json.loads(response['body'].read())

            if self.model_provider == 'cohere':
                return response_body['embeddings'][0]
            elif self.model_provider == 'amazon':
                return response_body['embedding']

        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    def embed_batch(
        self,
        texts: List[str],
        input_type: str = "search_document",
        batch_size: int = 96
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching

        Args:
            texts: List of texts to embed
            input_type: Type of input - "search_document" or "search_query"
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            if self.model_provider == 'cohere':
                body = json.dumps({
                    "texts": batch,
                    "input_type": input_type,
                    "truncate": "END"
                })

                try:
                    response = self.client.invoke_model(
                        modelId=self.model_id,
                        body=body,
                        contentType='application/json',
                        accept='application/json'
                    )

                    response_body = json.loads(response['body'].read())
                    all_embeddings.extend(response_body['embeddings'])

                except Exception as e:
                    print(f"Error in batch {i//batch_size + 1}: {e}")
                    # Fall back to individual processing
                    for text in batch:
                        try:
                            emb = self.embed_text(text, input_type)
                            all_embeddings.append(emb)
                        except:
                            # Use zero vector as fallback
                            all_embeddings.append([0.0] * self.embedding_dim)

            elif self.model_provider == 'amazon':
                # Titan doesn't support batch processing
                for text in batch:
                    try:
                        emb = self.embed_text(text, input_type)
                        all_embeddings.append(emb)
                        time.sleep(0.1)  # Rate limiting
                    except:
                        all_embeddings.append([0.0] * self.embedding_dim)

            print(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

        return all_embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of embeddings for this model"""
        return self.embedding_dim

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model"""
        return {
            "model_id": self.model_id,
            "provider": self.model_provider,
            "embedding_dimension": self.embedding_dim,
            "region": self.region
        }
