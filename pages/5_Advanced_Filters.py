"""
Advanced Search Filters
Comprehensive filtering with genre, year, rating, and more
"""

import streamlit as st
import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.movie_search import MovieSearch

st.set_page_config(
    page_title="Advanced Search Filters",
    page_icon="🔧",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .filter-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("🔧 Advanced Search Filters")
st.markdown("Fine-tune your search with comprehensive filtering options")

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
    st.success("✅ Connected to OpenSearch")
except Exception as e:
    st.error(f"❌ Error connecting: {e}")
    st.stop()

# Initialize session state
if 'filter_results' not in st.session_state:
    st.session_state.filter_results = None

# Main Layout
col_filters, col_results = st.columns([1, 2])

with col_filters:
    st.markdown("### 🎛️ Filter Options")

    # Query
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Search Query")
    search_query = st.text_input(
        "Search terms",
        placeholder="e.g., space adventure",
        help="Enter keywords or natural language query"
    )

    search_mode = st.radio(
        "Search Mode",
        ["Semantic", "Hybrid", "Keyword Only"],
        horizontal=True,
        help="Semantic understands meaning, Hybrid combines both"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Genre Filter
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown("#### 🎭 Genre")

    # Get available genres
    try:
        all_genres = set()
        response = searcher.client.search(
            index='movies',
            body={
                "size": 100,
                "query": {"match_all": {}},
                "_source": ["genre"]
            }
        )

        for hit in response['hits']['hits']:
            genres = hit['_source'].get('genre', [])
            if isinstance(genres, list):
                all_genres.update(genres)
            elif genres:
                all_genres.add(genres)

        available_genres = sorted(list(all_genres))
    except:
        available_genres = ["Action", "Adventure", "Drama", "Sci-Fi", "Fantasy", "Crime"]

    selected_genres = st.multiselect(
        "Select genres",
        options=available_genres,
        help="Select one or more genres"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Year Range
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown("#### 📅 Year Range")

    current_year = datetime.now().year
    year_range = st.slider(
        "Release year",
        min_value=1970,
        max_value=current_year,
        value=(1990, current_year),
        help="Filter movies by release year"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Rating Filter
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown("#### ⭐ Minimum Rating")

    min_rating = st.slider(
        "Minimum IMDb rating",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1,
        help="Only show movies with rating >= this value"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Runtime Filter
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown("#### ⏱️ Runtime")

    runtime_filter = st.checkbox("Filter by runtime", value=False)

    if runtime_filter:
        runtime_range = st.slider(
            "Runtime (minutes)",
            min_value=60,
            max_value=240,
            value=(90, 180),
            help="Filter by movie duration"
        )
    else:
        runtime_range = None
    st.markdown('</div>', unsafe_allow_html=True)

    # Sort Options
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown("#### 📊 Sort By")

    sort_by = st.selectbox(
        "Sort results by",
        ["Relevance", "Rating (High to Low)", "Rating (Low to High)",
         "Year (Newest)", "Year (Oldest)", "Title (A-Z)"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Number of Results
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown("#### 📦 Results")

    num_results = st.slider(
        "Number of results",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Search Button
    search_button = st.button(
        "🔍 Apply Filters & Search",
        type="primary",
        use_container_width=True
    )

    # Reset Button
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.session_state.filter_results = None
        st.rerun()

with col_results:
    st.markdown("### 🎬 Search Results")

    if search_button or st.session_state.filter_results is not None:
        if not search_query:
            search_query = ""  # Empty query for browse mode

        with st.spinner("Searching with filters..."):
            try:
                # Perform filtered search
                if search_mode == "Semantic" or search_mode == "Hybrid":
                    results = searcher.filter_search(
                        query=search_query if search_query else "movies",
                        genre=selected_genres if selected_genres else None,
                        min_year=year_range[0],
                        max_year=year_range[1],
                        min_rating=min_rating,
                        top_k=num_results
                    )
                else:
                    # Keyword only search with filters
                    filters = []

                    if selected_genres:
                        filters.append({"terms": {"genre": selected_genres}})

                    filters.append({"range": {"year": {"gte": year_range[0], "lte": year_range[1]}}})
                    filters.append({"range": {"rating": {"gte": min_rating}}})

                    if runtime_filter and runtime_range:
                        filters.append({"range": {"runtime": {"gte": runtime_range[0], "lte": runtime_range[1]}}})

                    search_body = {
                        "size": num_results,
                        "query": {
                            "bool": {
                                "must": [
                                    {"multi_match": {
                                        "query": search_query if search_query else "*",
                                        "fields": ["title", "plot", "genre", "director"]
                                    }}
                                ] if search_query else [{"match_all": {}}],
                                "filter": filters
                            }
                        },
                        "_source": {"excludes": ["plot_embedding"]}
                    }

                    response = searcher.client.search(index='movies', body=search_body)
                    results = [hit['_source'] for hit in response['hits']['hits']]

                # Apply runtime filter if needed (post-processing)
                if runtime_filter and runtime_range and results:
                    results = [r for r in results if runtime_range[0] <= r.get('runtime', 0) <= runtime_range[1]]

                # Apply sorting
                if results:
                    if sort_by == "Rating (High to Low)":
                        results.sort(key=lambda x: x.get('rating', 0), reverse=True)
                    elif sort_by == "Rating (Low to High)":
                        results.sort(key=lambda x: x.get('rating', 0))
                    elif sort_by == "Year (Newest)":
                        results.sort(key=lambda x: x.get('year', 0), reverse=True)
                    elif sort_by == "Year (Oldest)":
                        results.sort(key=lambda x: x.get('year', 0))
                    elif sort_by == "Title (A-Z)":
                        results.sort(key=lambda x: x.get('title', ''))

                st.session_state.filter_results = results

                # Display results
                if results:
                    st.success(f"✅ Found {len(results)} movies matching your criteria")

                    # Results summary
                    st.markdown("---")

                    for i, movie in enumerate(results, 1):
                        st.markdown(f'<div class="result-card">', unsafe_allow_html=True)

                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"### {i}. {movie['title']} ({movie['year']})")
                            st.markdown(f"**Plot:** {movie['plot']}")

                            if 'director' in movie:
                                director = movie['director']
                                if isinstance(director, list):
                                    director = ", ".join(director)
                                st.markdown(f"**Director:** {director}")

                            if 'actors' in movie:
                                actors = movie['actors']
                                if isinstance(actors, list):
                                    actors = ", ".join(actors[:3])  # First 3 actors
                                st.markdown(f"**Cast:** {actors}")

                        with col2:
                            st.metric("Rating", f"⭐ {movie['rating']}/10")
                            st.metric("Runtime", f"{movie.get('runtime', 'N/A')} min")

                            genres = movie.get('genre', [])
                            if isinstance(genres, list):
                                for g in genres[:3]:
                                    st.markdown(f"`{g}`")
                            else:
                                st.markdown(f"`{genres}`")

                            if 'search_score' in movie:
                                st.caption(f"Relevance: {movie['search_score']:.3f}")

                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("")

                    # Export option
                    st.markdown("---")
                    if st.button("📥 Export Results", use_container_width=True):
                        import pandas as pd
                        df = pd.DataFrame(results)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            "search_results.csv",
                            "text/csv",
                            key='download-csv'
                        )

                else:
                    st.warning("No movies found matching your criteria. Try adjusting the filters.")

            except Exception as e:
                st.error(f"❌ Search error: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

    else:
        # Empty state
        st.info("""
        👈 **Configure your filters on the left**, then click "Apply Filters & Search"

        **Tips:**
        - Leave search query empty to browse all movies
        - Select multiple genres to find movies in any of those genres
        - Adjust the rating slider to filter quality
        - Use year range to focus on specific eras
        """)

        st.markdown("### 📋 Example Filter Combinations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **High-Rated Recent Sci-Fi:**
            - Genre: Sci-Fi
            - Year: 2000+
            - Rating: 8.0+
            - Sort: Rating (High to Low)
            """)

            st.markdown("""
            **Classic Dramas:**
            - Genre: Drama
            - Year: 1970-2000
            - Rating: 7.5+
            - Sort: Year (Oldest)
            """)

        with col2:
            st.markdown("""
            **Action-Packed Blockbusters:**
            - Genre: Action, Adventure
            - Runtime: 120-180 min
            - Rating: 7.0+
            - Sort: Relevance
            """)

            st.markdown("""
            **Hidden Gems:**
            - Any Genre
            - Rating: 8.5+
            - Year: Any
            - Sort: Rating (High to Low)
            """)

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Filter Guide")

    st.markdown("""
    **Search Modes:**
    - **Semantic**: Understands meaning
    - **Hybrid**: Meaning + keywords
    - **Keyword**: Traditional search

    **Filter Tips:**
    - Combine multiple filters for precision
    - Use year range for era-specific search
    - Higher rating = better quality
    - Multiple genres = OR logic
    """)

    st.markdown("---")
    st.markdown("### 📊 Current Filters")

    if search_query:
        st.info(f"Query: {search_query}")
    if selected_genres:
        st.info(f"Genres: {', '.join(selected_genres)}")
    st.info(f"Years: {year_range[0]}-{year_range[1]}")
    st.info(f"Min Rating: {min_rating}+")
    if runtime_filter and runtime_range:
        st.info(f"Runtime: {runtime_range[0]}-{runtime_range[1]} min")
