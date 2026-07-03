"""
RAG Evaluator for Pattern Comparison
Metrics and evaluation tools
"""

import time
from typing import List, Dict, Any
import numpy as np


class RAGEvaluator:
    """Evaluate RAG system performance"""

    def __init__(self):
        self.results = []

    def evaluate_retrieval(self,
                          retrieved_docs: List[str],
                          relevant_docs: List[str]) -> Dict[str, float]:
        """
        Evaluate retrieval quality

        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of known relevant document IDs

        Returns:
            Metrics dict with precision, recall, F1
        """
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)

        true_positives = len(retrieved_set & relevant_set)
        false_positives = len(retrieved_set - relevant_set)
        false_negatives = len(relevant_set - retrieved_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    def calculate_mrr(self, rankings: List[List[bool]]) -> float:
        """
        Calculate Mean Reciprocal Rank

        Args:
            rankings: List of ranking lists (True if relevant)

        Returns:
            MRR score
        """
        reciprocal_ranks = []

        for ranking in rankings:
            for i, is_relevant in enumerate(ranking, 1):
                if is_relevant:
                    reciprocal_ranks.append(1.0 / i)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def calculate_ndcg(self, relevance_scores: List[List[float]], k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain

        Args:
            relevance_scores: List of relevance score lists
            k: Cutoff rank

        Returns:
            NDCG@k score
        """
        def dcg(scores):
            return sum([(2**score - 1) / np.log2(idx + 2) for idx, score in enumerate(scores[:k])])

        ndcg_scores = []
        for scores in relevance_scores:
            ideal_scores = sorted(scores, reverse=True)
            dcg_score = dcg(scores)
            idcg_score = dcg(ideal_scores)
            ndcg = dcg_score / idcg_score if idcg_score > 0 else 0.0
            ndcg_scores.append(ndcg)

        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    def measure_latency(self, rag_function, *args, **kwargs) -> tuple:
        """
        Measure RAG system latency

        Args:
            rag_function: Function to measure
            *args, **kwargs: Arguments to pass

        Returns:
            (result, latency_ms)
        """
        start_time = time.time()
        result = rag_function(*args, **kwargs)
        latency_ms = (time.time() - start_time) * 1000

        return result, latency_ms

    def evaluate_answer_quality(self,
                                generated_answer: str,
                                reference_answer: str,
                                bedrock_llm=None) -> Dict[str, Any]:
        """
        Evaluate answer quality using LLM as judge

        Args:
            generated_answer: Generated answer
            reference_answer: Ground truth answer
            bedrock_llm: BedrockLLM instance for evaluation

        Returns:
            Quality metrics
        """
        if bedrock_llm is None:
            # Simple metrics without LLM
            return {
                'length': len(generated_answer),
                'reference_length': len(reference_answer),
                'length_ratio': len(generated_answer) / len(reference_answer) if len(reference_answer) > 0 else 0
            }

        # Use LLM as judge
        judge_prompt = f"""
Evaluate the quality of the generated answer compared to the reference answer.

Generated Answer:
{generated_answer}

Reference Answer:
{reference_answer}

Rate the generated answer on a scale of 1-5 for:
1. Accuracy (factual correctness)
2. Completeness (covers all key points)
3. Relevance (stays on topic)
4. Coherence (well-structured and clear)

Provide scores and brief justification in JSON format:
{{
    "accuracy": <score>,
    "completeness": <score>,
    "relevance": <score>,
    "coherence": <score>,
    "justification": "<explanation>"
}}
"""

        try:
            response = bedrock_llm.generate(judge_prompt, temperature=0.1)
            import json
            scores = json.loads(response)
            return scores
        except:
            return {
                'accuracy': 0,
                'completeness': 0,
                'relevance': 0,
                'coherence': 0,
                'justification': 'Error in evaluation'
            }

    def compare_patterns(self,
                        pattern_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple RAG patterns

        Args:
            pattern_results: Dict of {pattern_name: metrics}

        Returns:
            Comparison DataFrame
        """
        import pandas as pd

        comparison_data = []

        for pattern_name, metrics in pattern_results.items():
            comparison_data.append({
                'Pattern': pattern_name,
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1': metrics.get('f1_score', 0),
                'Latency (ms)': metrics.get('latency_ms', 0),
                'Answer Quality': metrics.get('answer_quality', 0)
            })

        df = pd.DataFrame(comparison_data)
        return df.sort_values('F1', ascending=False)

    def generate_evaluation_report(self,
                                   pattern_name: str,
                                   metrics: Dict[str, Any]) -> str:
        """
        Generate markdown evaluation report

        Args:
            pattern_name: Name of the pattern
            metrics: Evaluation metrics

        Returns:
            Markdown formatted report
        """
        report = f"""
## Evaluation Report: {pattern_name}

### Retrieval Metrics
- **Precision**: {metrics.get('precision', 0):.3f}
- **Recall**: {metrics.get('recall', 0):.3f}
- **F1 Score**: {metrics.get('f1_score', 0):.3f}

### Performance
- **Latency**: {metrics.get('latency_ms', 0):.2f} ms
- **Documents Retrieved**: {metrics.get('docs_retrieved', 0)}

### Answer Quality
- **Accuracy**: {metrics.get('accuracy', 'N/A')}
- **Completeness**: {metrics.get('completeness', 'N/A')}
- **Relevance**: {metrics.get('relevance', 'N/A')}
- **Coherence**: {metrics.get('coherence', 'N/A')}

### Cost Estimate
- **Embedding API Calls**: {metrics.get('embedding_calls', 0)}
- **LLM API Calls**: {metrics.get('llm_calls', 0)}
- **Estimated Cost**: ${metrics.get('estimated_cost', 0):.4f}

"""
        return report
