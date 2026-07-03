"""
Search Analytics Dashboard
Visualize search quality, performance, and usage patterns
"""

import streamlit as st
import json
import os
import sys
import pandas as pd
from datetime import datetime
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.movie_search import MovieSearch

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Search Analytics Dashboard")
st.markdown("Monitor search quality, performance, and usage")

# Load configuration
config_path = 'config.json'
if not os.path.exists(config_path):
    st.error("❌ OpenSearch not configured.")
    st.stop()

with open(config_path, 'r') as f:
    config = json.load(f)

@st.cache_resource
def get_search_client():
    return MovieSearch(config)

try:
    searcher = get_search_client()
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Index Overview",
    "🔍 Search Quality",
    "⚡ Performance",
    "🎯 Embedding Analysis"
])

with tab1:
    st.header("Index Overview")

    col1, col2, col3, col4 = st.columns(4)

    try:
        # Get index stats
        stats_response = searcher.client.cat.indices(
            index='movies',
            format='json'
        )

        if stats_response:
            stats = stats_response[0]
            doc_count = int(stats.get('docs.count', 0))
            store_size = stats.get('store.size', 'N/A')
            health = stats.get('health', 'N/A')

            with col1:
                st.metric("Total Movies", doc_count, help="Number of documents indexed")

            with col2:
                st.metric("Index Size", store_size, help="Storage used")

            with col3:
                st.metric("Health", health, help="Index health status")

            with col4:
                st.metric("Embedding Dim", "1024", help="Vector dimension")

        # Get sample documents
        sample_response = searcher.client.search(
            index='movies',
            body={
                "size": 100,
                "query": {"match_all": {}},
                "_source": ["title", "year", "rating", "genre"]
            }
        )

        if sample_response['hits']['hits']:
            movies = []
            for hit in sample_response['hits']['hits']:
                src = hit['_source']
                movies.append({
                    'Title': src.get('title', 'N/A'),
                    'Year': src.get('year', 'N/A'),
                    'Rating': src.get('rating', 'N/A'),
                    'Genre': ', '.join(src.get('genre', [])) if isinstance(src.get('genre'), list) else src.get('genre', 'N/A')
                })

            df = pd.DataFrame(movies)

            st.markdown("---")
            st.subheader("📊 Collection Statistics")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Rating Distribution**")
                if 'Rating' in df.columns and df['Rating'].dtype in ['float64', 'int64']:
                    rating_counts = df['Rating'].value_counts().sort_index()
                    st.bar_chart(rating_counts)
                else:
                    st.info("Rating data not available")

                st.markdown("**Year Distribution**")
                if 'Year' in df.columns:
                    year_bins = pd.cut(df['Year'], bins=5)
                    year_counts = year_bins.value_counts().sort_index()
                    st.bar_chart(year_counts)

            with col2:
                st.markdown("**Genre Distribution**")
                # Extract all genres
                all_genres = []
                for genres_str in df['Genre']:
                    if isinstance(genres_str, str) and genres_str != 'N/A':
                        all_genres.extend([g.strip() for g in genres_str.split(',')])

                if all_genres:
                    genre_counts = pd.Series(all_genres).value_counts()
                    st.bar_chart(genre_counts)
                else:
                    st.info("Genre data not available")

                st.markdown("**Average Rating by Decade**")
                if 'Year' in df.columns and 'Rating' in df.columns:
                    df['Decade'] = (df['Year'] // 10) * 10
                    decade_ratings = df.groupby('Decade')['Rating'].mean()
                    st.bar_chart(decade_ratings)

            st.markdown("---")
            st.subheader("📋 All Movies")
            st.dataframe(
                df.sort_values('Rating', ascending=False),
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        st.error(f"Error loading index stats: {e}")

with tab2:
    st.header("Search Quality Analysis")

    st.markdown("""
    Test search quality across different query types to understand
    how well the semantic search understands user intent.
    """)

    # Test queries
    test_queries = [
        {
            "category": "Thematic",
            "query": "movies about friendship and redemption",
            "expected": ["The Shawshank Redemption", "Lord of the Rings"]
        },
        {
            "category": "Genre-based",
            "query": "dark psychological thriller",
            "expected": ["Fight Club", "The Matrix"]
        },
        {
            "category": "Plot-based",
            "query": "space exploration and survival",
            "expected": ["Interstellar", "Star Wars"]
        },
        {
            "category": "Emotion-based",
            "query": "uplifting and inspirational",
            "expected": ["Forrest Gump", "Shawshank Redemption"]
        }
    ]

    if st.button("🧪 Run Quality Tests", type="primary"):
        results = []

        progress_bar = st.progress(0)
        status = st.empty()

        for i, test in enumerate(test_queries):
            status.text(f"Testing: {test['query']}")

            try:
                start_time = time.time()
                search_results = searcher.semantic_search(
                    query=test['query'],
                    top_k=5,
                    min_score=0.0
                )
                latency = (time.time() - start_time) * 1000

                top_titles = [r['title'] for r in search_results[:3]]

                # Calculate relevance (simplified)
                relevance_score = sum(
                    1 for expected in test['expected']
                    if any(expected.lower() in title.lower() for title in top_titles)
                ) / len(test['expected'])

                results.append({
                    "Category": test['category'],
                    "Query": test['query'],
                    "Top Result": top_titles[0] if top_titles else "N/A",
                    "Relevance": f"{relevance_score:.0%}",
                    "Latency (ms)": f"{latency:.0f}"
                })

            except Exception as e:
                results.append({
                    "Category": test['category'],
                    "Query": test['query'],
                    "Top Result": f"Error: {str(e)[:50]}",
                    "Relevance": "N/A",
                    "Latency (ms)": "N/A"
                })

            progress_bar.progress((i + 1) / len(test_queries))

        status.text("✅ Tests complete!")

        st.markdown("### 📊 Test Results")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 💡 Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Search Quality**")
            st.info("""
            - **Thematic queries**: Tests understanding of abstract concepts
            - **Genre-based**: Tests classification accuracy
            - **Plot-based**: Tests content matching
            - **Emotion-based**: Tests sentiment understanding
            """)

        with col2:
            st.markdown("**How to Improve**")
            st.info("""
            - Add more training data
            - Fine-tune embedding model
            - Adjust hybrid search weights
            - Implement user feedback loop
            """)

with tab3:
    st.header("Performance Metrics")

    st.markdown("### ⚡ Latency Testing")

    num_queries = st.slider("Number of test queries", 5, 50, 10)

    if st.button("🏃 Run Performance Test", type="primary"):
        test_query = "movies about adventure"

        latencies = {
            "semantic": [],
            "hybrid": []
        }

        progress = st.progress(0)
        status = st.empty()

        for i in range(num_queries):
            status.text(f"Running test {i+1}/{num_queries}")

            # Test semantic search
            start = time.time()
            try:
                searcher.semantic_search(test_query, top_k=5)
                latencies["semantic"].append((time.time() - start) * 1000)
            except:
                pass

            # Test hybrid search
            start = time.time()
            try:
                searcher.hybrid_search(test_query, top_k=5)
                latencies["hybrid"].append((time.time() - start) * 1000)
            except:
                pass

            progress.progress((i + 1) / num_queries)

        status.text("✅ Performance test complete!")

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Semantic Search**")
            if latencies["semantic"]:
                avg_latency = sum(latencies["semantic"]) / len(latencies["semantic"])
                st.metric("Avg Latency", f"{avg_latency:.0f} ms")
                st.metric("Min Latency", f"{min(latencies['semantic']):.0f} ms")
                st.metric("Max Latency", f"{max(latencies['semantic']):.0f} ms")

                st.line_chart(latencies["semantic"])

        with col2:
            st.markdown("**Hybrid Search**")
            if latencies["hybrid"]:
                avg_latency = sum(latencies["hybrid"]) / len(latencies["hybrid"])
                st.metric("Avg Latency", f"{avg_latency:.0f} ms")
                st.metric("Min Latency", f"{min(latencies['hybrid']):.0f} ms")
                st.metric("Max Latency", f"{max(latencies['hybrid']):.0f} ms")

                st.line_chart(latencies["hybrid"])

        st.markdown("---")
        st.info("""
        **Performance Tips:**
        - OpenSearch Serverless: Auto-scales with load
        - Vector search: <100ms typical latency
        - Hybrid search: Slightly slower due to multi-phase query
        - Embedding generation: ~50-100ms via Bedrock
        """)

with tab4:
    st.header("Embedding & Vector Analysis")

    st.markdown("""
    Understanding the vector space and embedding quality helps optimize search performance.
    """)

    st.markdown("### 🧬 Embedding Model Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Current Configuration:**
        - **Model**: Amazon Titan Text Embeddings V2
        - **Dimension**: 1024
        - **Space Type**: Cosine Similarity
        - **Index Type**: HNSW (Hierarchical Navigable Small World)
        """)

    with col2:
        st.markdown("""
        **Performance Characteristics:**
        - **Latency**: ~50ms per embedding
        - **Batch Size**: Up to 96 texts
        - **Max Input**: 8192 tokens
        - **Cost**: $0.0001 per 1K tokens
        """)

    st.markdown("---")
    st.markdown("### 🎯 Similarity Distribution")

    if st.button("📊 Analyze Vector Similarity"):
        with st.spinner("Computing pairwise similarities..."):
            try:
                # Get all movies with embeddings
                response = searcher.client.search(
                    index='movies',
                    body={
                        "size": 20,
                        "query": {"match_all": {}},
                        "_source": ["title", "plot_embedding"]
                    }
                )

                if response['hits']['hits']:
                    import numpy as np

                    movies = []
                    embeddings = []

                    for hit in response['hits']['hits']:
                        src = hit['_source']
                        if 'plot_embedding' in src and 'title' in src:
                            movies.append(src['title'])
                            embeddings.append(np.array(src['plot_embedding']))

                    if len(embeddings) >= 2:
                        # Compute cosine similarities
                        from numpy import dot
                        from numpy.linalg import norm

                        similarities = []
                        for i in range(len(embeddings)):
                            for j in range(i+1, len(embeddings)):
                                sim = dot(embeddings[i], embeddings[j]) / (
                                    norm(embeddings[i]) * norm(embeddings[j])
                                )
                                similarities.append({
                                    'Movie 1': movies[i],
                                    'Movie 2': movies[j],
                                    'Similarity': sim
                                })

                        # Sort by similarity
                        similarities.sort(key=lambda x: x['Similarity'], reverse=True)

                        st.markdown("**Most Similar Movie Pairs:**")
                        sim_df = pd.DataFrame(similarities[:10])
                        sim_df['Similarity'] = sim_df['Similarity'].apply(lambda x: f"{x:.3f}")
                        st.dataframe(sim_df, use_container_width=True, hide_index=True)

                        st.markdown("**Least Similar Movie Pairs:**")
                        dissim_df = pd.DataFrame(similarities[-10:])
                        dissim_df['Similarity'] = dissim_df['Similarity'].apply(lambda x: f"{x:.3f}")
                        st.dataframe(dissim_df, use_container_width=True, hide_index=True)

                        # Distribution
                        st.markdown("**Similarity Distribution:**")
                        sim_values = [s['Similarity'] for s in similarities]
                        st.bar_chart(pd.Series(sim_values).value_counts(bins=10).sort_index())

                        st.info(f"""
                        **Analysis:** Computed {len(similarities)} pairwise similarities.
                        - High similarity (>0.8): Movies are very similar in theme/plot
                        - Medium similarity (0.6-0.8): Related but distinct movies
                        - Low similarity (<0.6): Different themes/genres
                        """)
                    else:
                        st.warning("Need at least 2 movies with embeddings")

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")
    st.markdown("### 🔧 Optimization Recommendations")

    st.info("""
    **To improve search quality:**

    1. **Add more data**: More movies = better coverage of vector space
    2. **Hybrid search tuning**: Adjust semantic vs keyword weights
    3. **Embedding model**: Try Cohere Embed v4 for better quality (if available)
    4. **HNSW parameters**: Tune `ef_construction` and `m` for speed vs accuracy
    5. **Query preprocessing**: Expand queries, remove stop words
    6. **User feedback**: Collect relevance signals to fine-tune
    """)
