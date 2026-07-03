"""
Search History Management
Track and save user searches with session state
"""

import streamlit as st
from datetime import datetime
import json

class SearchHistory:
    """Manage search history and saved searches"""

    def __init__(self):
        # Initialize session state
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []

        if 'saved_searches' not in st.session_state:
            st.session_state.saved_searches = []

    def add_search(self, query: str, search_type: str, num_results: int = 0):
        """Add a search to history"""
        search_entry = {
            'query': query,
            'type': search_type,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'results_count': num_results
        }

        # Avoid duplicates of recent searches
        if not st.session_state.search_history or \
           st.session_state.search_history[0]['query'] != query:
            st.session_state.search_history.insert(0, search_entry)

            # Keep only last 20 searches
            st.session_state.search_history = st.session_state.search_history[:20]

    def get_history(self, limit: int = 10):
        """Get recent search history"""
        return st.session_state.search_history[:limit]

    def clear_history(self):
        """Clear all search history"""
        st.session_state.search_history = []

    def save_search(self, query: str, search_type: str, name: str = None):
        """Save a search for later"""
        if name is None:
            name = query[:50]  # Use query as name if not provided

        saved_entry = {
            'name': name,
            'query': query,
            'type': search_type,
            'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Check if already saved
        if not any(s['query'] == query for s in st.session_state.saved_searches):
            st.session_state.saved_searches.append(saved_entry)

    def get_saved_searches(self):
        """Get all saved searches"""
        return st.session_state.saved_searches

    def remove_saved_search(self, query: str):
        """Remove a saved search"""
        st.session_state.saved_searches = [
            s for s in st.session_state.saved_searches
            if s['query'] != query
        ]

    def export_history_json(self):
        """Export history as JSON"""
        return json.dumps({
            'history': st.session_state.search_history,
            'saved': st.session_state.saved_searches,
            'exported_at': datetime.now().isoformat()
        }, indent=2)

    def get_popular_searches(self, limit: int = 5):
        """Get most common searches"""
        query_counts = {}
        for search in st.session_state.search_history:
            query = search['query']
            query_counts[query] = query_counts.get(query, 0) + 1

        sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        return [q for q, _ in sorted_queries[:limit]]
