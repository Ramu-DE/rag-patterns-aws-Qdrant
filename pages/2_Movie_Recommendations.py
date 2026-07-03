"""
AI-Powered Movie Recommendation System
Find movies similar to ones you love using vector similarity
"""

import streamlit as st
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.movie_search import MovieSearch

st.set_page_config(
    page_title="Movie Recommendations",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI-Powered Movie Recommendations")
st.markdown("Discover movies similar to your favorites using vector embeddings")

# Load configuration
config_path = 'config.json'
if not os.path.exists(config_path):
    st.error("❌ OpenSearch not configured. Please run setup first.")
    st.stop()

with open(config_path, 'r') as f:
    config = json.load(f)

@st.cache_resource
def get_search_client():
    return MovieSearch(config)

try:
    searcher = get_search_client()
except Exception as e:
    st.error(f"❌ Error connecting to OpenSearch: {e}")
    st.stop()

# Sidebar
st.sidebar.header("Recommendation Settings")

num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=3,
    max_value=10,
    value=5
)

min_score = st.sidebar.slider(
    "Minimum Similarity Score",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Higher = more similar movies only"
)

use_claude = st.sidebar.checkbox(
    "AI Explanation",
    value=True,
    help="Claude explains why these movies are similar"
)

st.sidebar.markdown("---")
st.sidebar.info("""
**How it works:**

1. Select or search for a movie
2. AI finds similar movies using vector embeddings
3. Results ranked by semantic similarity
4. Claude explains the connections
""")

# Main content
tab1, tab2 = st.tabs(["🔍 Find Recommendations", "📊 How It Works"])

with tab1:
    st.markdown("### Select a Movie")

    # Method selection
    method = st.radio(
        "How would you like to choose a movie?",
        ["Select from list", "Describe a movie"],
        horizontal=True
    )

    if method == "Select from list":
        # Get all movies from index
        try:
            all_movies_response = searcher.client.search(
                index='movies',
                body={
                    "size": 100,
                    "query": {"match_all": {}},
                    "_source": ["title", "year", "plot", "rating"]
                }
            )

            all_movies = [
                {
                    "title": hit["_source"]["title"],
                    "year": hit["_source"]["year"],
                    "plot": hit["_source"]["plot"],
                    "rating": hit["_source"]["rating"]
                }
                for hit in all_movies_response["hits"]["hits"]
            ]

            movie_options = [
                f"{m['title']} ({m['year']}) - ⭐ {m['rating']}"
                for m in all_movies
            ]

            selected = st.selectbox(
                "Choose a movie:",
                movie_options,
                help="Select a movie to find similar recommendations"
            )

            if selected:
                selected_idx = movie_options.index(selected)
                selected_movie = all_movies[selected_idx]

                st.markdown("---")
                st.markdown("### 🎬 Selected Movie")

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**{selected_movie['title']}** ({selected_movie['year']})")
                    st.markdown(selected_movie['plot'])
                with col2:
                    st.metric("Rating", f"{selected_movie['rating']}/10")

                if st.button("🎯 Find Similar Movies", type="primary", use_container_width=True):
                    with st.spinner("Finding similar movies..."):
                        try:
                            # Use the movie's plot as query
                            results = searcher.semantic_search(
                                query=selected_movie['plot'],
                                top_k=num_recommendations + 1,  # +1 to exclude original
                                min_score=min_score
                            )

                            # Filter out the selected movie
                            recommendations = [
                                r for r in results
                                if r['title'] != selected_movie['title']
                            ][:num_recommendations]

                            if recommendations:
                                st.markdown("---")
                                st.markdown("### 🎯 Recommended Movies")

                                # AI Explanation
                                if use_claude:
                                    explanation_query = f"Why are these movies similar to {selected_movie['title']}?"
                                    answer = searcher.generate_answer_with_claude(
                                        explanation_query,
                                        recommendations,
                                        model_id="us.anthropic.claude-opus-4-1-20250805-v1:0"
                                    )

                                    if answer and "Error" not in answer['answer']:
                                        st.info(f"🤖 **AI Analysis:** {answer['answer']}")
                                        st.markdown("---")

                                # Show recommendations
                                for i, movie in enumerate(recommendations, 1):
                                    with st.expander(
                                        f"{i}. {movie['title']} ({movie['year']}) - Similarity: {movie['search_score']:.3f}",
                                        expanded=(i == 1)
                                    ):
                                        col1, col2 = st.columns([2, 1])

                                        with col1:
                                            st.markdown(f"**Plot:** {movie['plot']}")

                                            if 'director' in movie:
                                                st.markdown(f"**Director:** {movie['director']}")

                                            if 'actors' in movie:
                                                actors = movie['actors']
                                                if isinstance(actors, list):
                                                    actors = ", ".join(actors)
                                                st.markdown(f"**Actors:** {actors}")

                                        with col2:
                                            if 'genre' in movie:
                                                genres = movie['genre']
                                                if isinstance(genres, list):
                                                    for g in genres:
                                                        st.markdown(f"`{g}`")
                                                else:
                                                    st.markdown(f"`{genres}`")

                                            st.metric("Rating", f"{movie['rating']}/10")
                                            st.metric("Similarity", f"{movie['search_score']:.1%}")
                            else:
                                st.warning("No similar movies found. Try lowering the similarity threshold.")

                        except Exception as e:
                            st.error(f"❌ Error finding recommendations: {e}")
                            import traceback
                            st.code(traceback.format_exc())

        except Exception as e:
            st.error(f"❌ Error loading movies: {e}")

    else:  # Describe a movie
        movie_description = st.text_area(
            "Describe the type of movie you're looking for:",
            placeholder="e.g., A sci-fi movie about space exploration with emotional depth and stunning visuals",
            height=100
        )

        if st.button("🎯 Find Recommendations", type="primary", use_container_width=True):
            if movie_description:
                with st.spinner("Finding movies that match your description..."):
                    try:
                        results, answer = searcher.search_and_answer(
                            query=movie_description,
                            search_type="semantic",
                            top_k=num_recommendations,
                            use_claude=use_claude,
                            claude_model="us.anthropic.claude-opus-4-1-20250805-v1:0"
                        )

                        if results:
                            if answer and answer['answer']:
                                st.markdown("### 🤖 AI Recommendations")
                                st.info(answer['answer'])
                                st.markdown("---")

                            st.markdown("### 🎬 Matching Movies")

                            for i, movie in enumerate(results, 1):
                                with st.expander(
                                    f"{i}. {movie['title']} ({movie['year']}) - ⭐ {movie['rating']}/10",
                                    expanded=(i == 1)
                                ):
                                    st.markdown(f"**Plot:** {movie['plot']}")
                                    st.markdown(f"**Relevance Score:** {movie['search_score']:.3f}")
                        else:
                            st.warning("No matching movies found.")

                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            else:
                st.warning("Please describe the type of movie you're looking for.")

with tab2:
    st.markdown("""
    ## 🧠 How AI-Powered Recommendations Work

    This recommendation system uses **vector embeddings** and **semantic similarity** to find movies that are
    conceptually similar, going far beyond simple genre or keyword matching.

    ### The Process

    1. **Vector Embeddings**
       - Each movie's plot, title, and metadata is converted to a 1024-dimensional vector
       - Similar movies have vectors that point in similar directions
       - Powered by Amazon Titan Text Embeddings V2

    2. **Semantic Similarity**
       - When you select a movie, we find other movies with similar vectors
       - Measured using cosine similarity (0 = unrelated, 1 = identical)
       - Captures themes, emotions, and narrative structures

    3. **AI Explanation**
       - Claude Opus analyzes the recommended movies
       - Explains thematic connections and similarities
       - Provides context for why recommendations make sense

    ### Why This Works Better

    **Traditional Approach:**
    - Genre matching: "Both are sci-fi" ✗
    - Keyword matching: "Both mention 'space'" ✗
    - Collaborative filtering: "Users who liked X also liked Y" ✗

    **AI-Powered Approach:**
    - Understands themes: redemption, hope, friendship ✓
    - Captures emotions: dark, uplifting, tense ✓
    - Recognizes narrative patterns: hero's journey, plot twists ✓
    - Multi-dimensional: considers plot, themes, tone together ✓

    ### Example

    **Input:** "Interstellar"

    **Why similar movies are recommended:**
    - **Gravity**: Space survival, isolation, human resilience
    - **Arrival**: Emotional sci-fi, time themes, human connection
    - **The Martian**: Space survival, problem-solving, hope

    The AI understands these aren't just "space movies" - they share deeper thematic elements
    about human perseverance and emotional stakes in extreme circumstances.

    ### Technical Details

    - **Embedding Model**: Amazon Titan Text Embeddings V2
    - **Vector Dimension**: 1024
    - **Similarity Metric**: Cosine similarity
    - **Index Type**: HNSW (Hierarchical Navigable Small World)
    - **Search Latency**: <100ms
    - **LLM**: Claude Opus 4.1 for explanations
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 💡 Use Cases

        - **Personalized recommendations**
        - **Content discovery**
        - **"More like this" features**
        - **Playlist/collection generation**
        - **Similar item search**
        """)

    with col2:
        st.markdown("""
        ### 🎯 Benefits

        - **Better relevance** than genre matching
        - **Understands themes** and emotions
        - **No training data** needed
        - **Fast** real-time search
        - **Explainable** with AI
        """)
