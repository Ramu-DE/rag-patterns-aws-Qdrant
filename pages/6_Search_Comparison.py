"""
Search Comparison Tool
Side-by-side comparison of different search approaches
"""

import streamlit as st
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.movie_search import MovieSearch

st.set_page_config(
    page_title="Search Comparison",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .comparison-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 10px;
        height: 100%;
    }
    .semantic-box {
        border-left: 4px solid #667eea;
    }
    .keyword-box {
        border-left: 4px solid #f093fb;
    }
    .hybrid-box {
        border-left: 4px solid #4facfe;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Search Comparison Tool")
st.markdown("Compare different search approaches side-by-side")

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

# Search Input
st.markdown("### 🔍 Enter Your Query")

query = st.text_input(
    "Search query",
    placeholder="e.g., movies about friendship and redemption",
    help="Enter a natural language query to see how different search methods perform"
)

col1, col2 = st.columns([3, 1])
with col1:
    num_results = st.slider("Number of results per method", 3, 10, 5)
with col2:
    if st.button("⚖️ Compare Search Methods", type="primary", use_container_width=True):
        if query:
            st.session_state.compare_query = query
            st.session_state.compare_results = None
        else:
            st.warning("Please enter a search query")

# Example Queries
st.markdown("---")
with st.expander("💡 Try These Example Queries"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Thematic Queries:**
        - movies about redemption
        - films exploring loneliness
        - stories of personal growth
        """)

    with col2:
        st.markdown("""
        **Genre + Mood:**
        - dark psychological thriller
        - uplifting adventure
        - mind-bending sci-fi
        """)

    with col3:
        st.markdown("""
        **Plot Elements:**
        - space exploration survival
        - time travel paradox
        - heist with twist ending
        """)

# Perform Comparison
if 'compare_query' in st.session_state and st.session_state.compare_query:
    query = st.session_state.compare_query

    if 'compare_results' not in st.session_state or st.session_state.compare_results is None:
        with st.spinner("Running comparison across all search methods..."):
            results = {}
            latencies = {}

            try:
                # Semantic Search
                start = time.time()
                semantic_results = searcher.semantic_search(query, top_k=num_results, min_score=0.0)
                latencies['semantic'] = (time.time() - start) * 1000
                results['semantic'] = semantic_results

                # Hybrid Search
                start = time.time()
                hybrid_results = searcher.hybrid_search(query, top_k=num_results)
                latencies['hybrid'] = (time.time() - start) * 1000
                results['hybrid'] = hybrid_results

                # Keyword Search
                start = time.time()
                keyword_body = {
                    "size": num_results,
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^3", "plot^2", "genre", "director", "actors"]
                        }
                    },
                    "_source": {"excludes": ["plot_embedding"]}
                }
                keyword_response = searcher.client.search(index='movies', body=keyword_body)
                latencies['keyword'] = (time.time() - start) * 1000
                results['keyword'] = [
                    {**hit['_source'], 'search_score': hit['_score']}
                    for hit in keyword_response['hits']['hits']
                ]

                st.session_state.compare_results = {
                    'results': results,
                    'latencies': latencies
                }

            except Exception as e:
                st.error(f"❌ Comparison error: {e}")
                st.stop()

    # Display Results
    if 'compare_results' in st.session_state and st.session_state.compare_results:
        data = st.session_state.compare_results
        results = data['results']
        latencies = data['latencies']

        st.markdown("---")
        st.markdown(f"### 📊 Results for: *\"{query}\"*")

        # Performance Metrics
        st.markdown("#### ⚡ Performance Comparison")

        perf_col1, perf_col2, perf_col3 = st.columns(3)

        with perf_col1:
            st.markdown("""
            <div class="metric-card" style="border-left: 4px solid #667eea;">
                <h4>Semantic Search</h4>
                <h2>{:.0f}ms</h2>
                <p>Latency</p>
            </div>
            """.format(latencies['semantic']), unsafe_allow_html=True)

        with perf_col2:
            st.markdown("""
            <div class="metric-card" style="border-left: 4px solid #4facfe;">
                <h4>Hybrid Search</h4>
                <h2>{:.0f}ms</h2>
                <p>Latency</p>
            </div>
            """.format(latencies['hybrid']), unsafe_allow_html=True)

        with perf_col3:
            st.markdown("""
            <div class="metric-card" style="border-left: 4px solid #f093fb;">
                <h4>Keyword Search</h4>
                <h2>{:.0f}ms</h2>
                <p>Latency</p>
            </div>
            """.format(latencies['keyword']), unsafe_allow_html=True)

        # Side-by-Side Results
        st.markdown("---")
        st.markdown("#### 📋 Side-by-Side Results")

        result_cols = st.columns(3)

        # Column headers
        with result_cols[0]:
            st.markdown('<div class="comparison-box semantic-box">', unsafe_allow_html=True)
            st.markdown("### 🧠 Semantic Search")
            st.caption("Vector similarity - understands meaning")
            st.markdown('</div>', unsafe_allow_html=True)

        with result_cols[1]:
            st.markdown('<div class="comparison-box hybrid-box">', unsafe_allow_html=True)
            st.markdown("### ⚡ Hybrid Search")
            st.caption("Combines semantic + keyword")
            st.markdown('</div>', unsafe_allow_html=True)

        with result_cols[2]:
            st.markdown('<div class="comparison-box keyword-box">', unsafe_allow_html=True)
            st.markdown("### 🔤 Keyword Search")
            st.caption("Traditional text matching")
            st.markdown('</div>', unsafe_allow_html=True)

        # Results
        max_results = max(len(results['semantic']), len(results['hybrid']), len(results['keyword']))

        for i in range(max_results):
            st.markdown("---")
            cols = st.columns(3)

            # Semantic
            with cols[0]:
                if i < len(results['semantic']):
                    movie = results['semantic'][i]
                    st.markdown(f"**{i+1}. {movie['title']}** ({movie['year']})")
                    st.caption(f"⭐ {movie['rating']}/10 | Score: {movie['search_score']:.3f}")
                    with st.expander("Details"):
                        st.markdown(f"_{movie['plot'][:150]}..._")
                        genres = movie.get('genre', [])
                        if isinstance(genres, list):
                            st.markdown(f"**Genres:** {', '.join(genres)}")
                else:
                    st.markdown("*No more results*")

            # Hybrid
            with cols[1]:
                if i < len(results['hybrid']):
                    movie = results['hybrid'][i]
                    st.markdown(f"**{i+1}. {movie['title']}** ({movie['year']})")
                    st.caption(f"⭐ {movie['rating']}/10 | Score: {movie['search_score']:.3f}")
                    with st.expander("Details"):
                        st.markdown(f"_{movie['plot'][:150]}..._")
                        genres = movie.get('genre', [])
                        if isinstance(genres, list):
                            st.markdown(f"**Genres:** {', '.join(genres)}")
                else:
                    st.markdown("*No more results*")

            # Keyword
            with cols[2]:
                if i < len(results['keyword']):
                    movie = results['keyword'][i]
                    st.markdown(f"**{i+1}. {movie['title']}** ({movie['year']})")
                    st.caption(f"⭐ {movie['rating']}/10 | Score: {movie['search_score']:.3f}")
                    with st.expander("Details"):
                        st.markdown(f"_{movie['plot'][:150]}..._")
                        genres = movie.get('genre', [])
                        if isinstance(genres, list):
                            st.markdown(f"**Genres:** {', '.join(genres)}")
                else:
                    st.markdown("*No more results*")

        # Analysis
        st.markdown("---")
        st.markdown("#### 🔬 Analysis")

        # Overlap Analysis
        semantic_titles = set([m['title'] for m in results['semantic']])
        hybrid_titles = set([m['title'] for m in results['hybrid']])
        keyword_titles = set([m['title'] for m in results['keyword']])

        all_titles = semantic_titles | hybrid_titles | keyword_titles
        common_all = semantic_titles & hybrid_titles & keyword_titles

        analysis_col1, analysis_col2 = st.columns(2)

        with analysis_col1:
            st.markdown("**📊 Result Overlap:**")
            st.info(f"""
            - **Common to all 3**: {len(common_all)} movies
            - **Semantic ∩ Hybrid**: {len(semantic_titles & hybrid_titles)} movies
            - **Semantic ∩ Keyword**: {len(semantic_titles & keyword_titles)} movies
            - **Hybrid ∩ Keyword**: {len(hybrid_titles & keyword_titles)} movies
            - **Unique results**: {len(all_titles)} total distinct
            """)

        with analysis_col2:
            st.markdown("**💡 Insights:**")

            if len(semantic_titles & keyword_titles) / len(semantic_titles) < 0.5:
                st.warning("""
                **Semantic search found different results!**

                This suggests the query relies on understanding meaning
                rather than exact keyword matches. Semantic search is
                capturing thematic relevance.
                """)
            else:
                st.success("""
                **High overlap between methods!**

                For this query, semantic and keyword search agree.
                This often happens when the query contains specific
                terms that appear in relevant documents.
                """)

        # Recommendations
        st.markdown("---")
        st.markdown("#### 🎯 Which Search Method to Use?")

        rec_col1, rec_col2, rec_col3 = st.columns(3)

        with rec_col1:
            st.info("""
            **Use Semantic Search When:**
            - Queries describe concepts/themes
            - Natural language questions
            - Want to understand intent
            - Looking for similar ideas

            *Example: "movies about hope"*
            """)

        with rec_col2:
            st.success("""
            **Use Hybrid Search When:**
            - Want best of both worlds
            - Default recommendation
            - Balances meaning + precision
            - Most versatile option

            *Example: "sci-fi space thriller"*
            """)

        with rec_col3:
            st.warning("""
            **Use Keyword Search When:**
            - Looking for specific terms
            - Exact phrase matching
            - Known titles/names
            - Precision over recall

            *Example: "The Matrix 1999"*
            """)

        # Reset button
        st.markdown("---")
        if st.button("🔄 Try Another Query", use_container_width=True):
            st.session_state.compare_results = None
            st.session_state.compare_query = None
            st.rerun()

# Information Section
if 'compare_results' not in st.session_state or st.session_state.compare_results is None:
    st.markdown("---")
    st.markdown("## 📚 Understanding Search Methods")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🧠 Semantic Search

        **How it works:**
        - Converts text to vector embeddings
        - Finds similar vectors using cosine similarity
        - Understands context and meaning

        **Strengths:**
        - Understands synonyms
        - Captures themes
        - Intent-based

        **Best for:**
        - Abstract queries
        - Natural language
        - Thematic search
        """)

    with col2:
        st.markdown("""
        ### ⚡ Hybrid Search

        **How it works:**
        - Combines semantic + keyword scores
        - Weighted combination of both
        - Best of both approaches

        **Strengths:**
        - Balanced relevance
        - Handles various query types
        - Most versatile

        **Best for:**
        - General use
        - Mixed queries
        - Default choice
        """)

    with col3:
        st.markdown("""
        ### 🔤 Keyword Search

        **How it works:**
        - BM25 algorithm
        - Term frequency analysis
        - Exact text matching

        **Strengths:**
        - Fast
        - Precise for exact terms
        - Well-understood

        **Best for:**
        - Known terms
        - Specific names
        - Exact matches
        """)

# Sidebar
with st.sidebar:
    st.markdown("### ⚖️ Comparison Guide")

    st.markdown("""
    **How to use:**
    1. Enter a search query
    2. Click "Compare"
    3. Review results side-by-side
    4. Analyze differences

    **What to look for:**
    - Which method found more results?
    - Are results similar or different?
    - Which seems most relevant?
    - Performance differences?
    """)

    st.markdown("---")
    st.markdown("### 💡 Tips")

    st.info("""
    **Good comparison queries:**
    - Abstract themes
    - Natural language
    - Multi-word concepts
    - Genre + mood combinations

    These show the biggest differences
    between search methods.
    """)

    if 'compare_results' in st.session_state and st.session_state.compare_results:
        st.markdown("---")
        st.success(f"✅ Compared {num_results} results per method")
