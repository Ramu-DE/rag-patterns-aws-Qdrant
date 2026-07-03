"""
Bedrock Client for RAG Patterns
Handles embeddings and LLM calls
"""

import boto3
import json
from typing import List, Dict, Optional, Union
import time


class BedrockEmbeddings:
    """Amazon Bedrock embeddings client"""

    def __init__(self,
                 region_name: str = 'us-west-2',
                 model_id: str = 'amazon.titan-embed-text-v2:0'):
        """
        Initialize Bedrock embeddings

        Args:
            region_name: AWS region
            model_id: Embedding model ID
        """
        self.client = boto3.client('bedrock-runtime', region_name=region_name)
        self.model_id = model_id
        self.embedding_dim = 1024  # Titan V2 dimension

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for single text

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        body = json.dumps({
            "inputText": text
        })

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )

            response_body = json.loads(response['body'].read())
            return response_body['embedding']

        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * self.embedding_dim

    def embed_batch(self,
                   texts: List[str],
                   batch_size: int = 1,
                   show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts
            batch_size: Process one at a time for Titan
            show_progress: Show progress

        Returns:
            List of embeddings
        """
        embeddings = []

        for i, text in enumerate(texts):
            embedding = self.embed_text(text)
            embeddings.append(embedding)

            if show_progress and (i + 1) % 10 == 0:
                print(f"Embedded {i + 1}/{len(texts)} texts")

            # Rate limiting
            time.sleep(0.1)

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim


class BedrockLLM:
    """Amazon Bedrock LLM client for Claude models"""

    def __init__(self,
                 region_name: str = 'us-west-2',
                 model_id: str = 'us.anthropic.claude-opus-4-1-20250805-v1:0',
                 temperature: float = 0.7,
                 max_tokens: int = 2000):
        """
        Initialize Bedrock LLM

        Args:
            region_name: AWS region
            model_id: Claude model ID
            temperature: Sampling temperature
            max_tokens: Max tokens in response
        """
        self.client = boto3.client('bedrock-runtime', region_name=region_name)
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self,
                prompt: str,
                system_prompt: Optional[str] = None,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None) -> str:
        """
        Generate response from Claude

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens

        Returns:
            Generated text
        """
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": messages
        }

        if system_prompt:
            body["system"] = system_prompt

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )

            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']

        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: {str(e)}"

    def generate_with_context(self,
                             query: str,
                             context_docs: List[str],
                             system_prompt: Optional[str] = None) -> str:
        """
        Generate response with retrieved context (RAG)

        Args:
            query: User query
            context_docs: Retrieved documents for context
            system_prompt: Optional system instruction

        Returns:
            Generated answer
        """
        # Build context
        context = "\n\n".join([
            f"Document {i+1}:\n{doc}"
            for i, doc in enumerate(context_docs)
        ])

        # Build prompt
        prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""

        return self.generate(prompt, system_prompt)

    def chat(self,
            messages: List[Dict[str, str]],
            system_prompt: Optional[str] = None) -> str:
        """
        Multi-turn chat conversation

        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."}
            system_prompt: Optional system prompt

        Returns:
            Assistant response
        """
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages
        }

        if system_prompt:
            body["system"] = system_prompt

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )

            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']

        except Exception as e:
            print(f"Error in chat: {e}")
            return f"Error: {str(e)}"


class BedrockRAG:
    """Complete RAG system using Bedrock and OpenSearch"""

    def __init__(self,
                 opensearch_manager,
                 embedding_model: str = 'amazon.titan-embed-text-v2:0',
                 llm_model: str = 'us.anthropic.claude-opus-4-1-20250805-v1:0',
                 region: str = 'us-west-2'):
        """
        Initialize RAG system

        Args:
            opensearch_manager: OpenSearchManager instance
            embedding_model: Bedrock embedding model
            llm_model: Bedrock LLM model
            region: AWS region
        """
        self.opensearch = opensearch_manager
        self.embedder = BedrockEmbeddings(region, embedding_model)
        self.llm = BedrockLLM(region, llm_model)

    def index_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> int:
        """
        Index documents with embeddings

        Args:
            texts: List of text documents
            metadatas: Optional metadata for each document

        Returns:
            Number of documents indexed
        """
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.embedder.embed_batch(texts)

        documents = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            doc = {
                'text': text,
                'embedding': embedding,
                'metadata': metadatas[i] if metadatas else {}
            }
            documents.append(doc)

        return self.opensearch.index_documents(documents)

    def query(self,
             question: str,
             top_k: int = 5,
             return_sources: bool = True) -> Union[str, tuple]:
        """
        Query the RAG system

        Args:
            question: User question
            top_k: Number of documents to retrieve
            return_sources: Return source documents

        Returns:
            Answer (and optionally source documents)
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_text(question)

        # Retrieve relevant documents
        results = self.opensearch.vector_search(query_embedding, top_k=top_k)

        if not results:
            answer = "I couldn't find any relevant information to answer your question."
            return (answer, []) if return_sources else answer

        # Extract texts
        context_docs = [r['text'] for r in results]

        # Generate answer
        answer = self.llm.generate_with_context(question, context_docs)

        if return_sources:
            return answer, results
        return answer
