"""
Movie Search Demo - Enhanced Home Page
OpenSearch Serverless with AWS Bedrock Embeddings and Claude RAG
"""

import streamlit as st
import json
import os

st.set_page_config(
    page_title="AI-Powered Movie Search",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .stat-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .quick-search {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎬 AI-Powered Movie Search</div>', unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; font-size: 1.2rem; color: #666;'>
Powered by AWS OpenSearch Serverless & Amazon Bedrock
</p>
""", unsafe_allow_html=True)

# Check configuration
config_path = 'config.json'
config_exists = os.path.exists(config_path)

if config_exists:
    with open(config_path, 'r') as f:
        config = json.load(f)

# Quick Search Bar
st.markdown("---")
st.markdown("### 🔍 Quick Search")

col1, col2 = st.columns([4, 1])
with col1:
    quick_query = st.text_input(
        "Search movies...",
        placeholder="Try: 'movies about friendship' or 'dark psychological thriller'",
        label_visibility="collapsed"
    )
with col2:
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if quick_query:
            st.switch_page("pages/1_Semantic_Search.py")

# System Status
st.markdown("---")

if config_exists:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="stat-box">
            <h2>✅</h2>
            <h3>Active</h3>
            <p>System Status</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-box">
            <h2>10</h2>
            <h3>Movies</h3>
            <p>In Database</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stat-box">
            <h2>1024</h2>
            <h3>Dimensions</h3>
            <p>Vector Size</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="stat-box">
            <h2>4</h2>
            <h3>Features</h3>
            <p>AI Capabilities</p>
        </div>
        """, unsafe_allow_html=True)

    # Configuration Details
    with st.expander("📋 System Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Collection:** {config.get('collection_name', 'N/A')}")
            st.info(f"**Region:** {config.get('region', 'N/A')}")
        with col2:
            st.info(f"**Endpoint:** Available")
            st.info(f"**Status:** {config.get('status', 'Active')}")
else:
    st.error("❌ **OpenSearch not configured.** Please run setup first.")
    st.code("python3 infrastructure/create_opensearch.py", language="bash")

# Features Section
st.markdown("---")
st.markdown("## 🚀 AI-Powered Features")

# Feature Cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🔍 Semantic Search</h3>
        <p><strong>Natural Language Understanding</strong></p>
        <p>Search using natural language queries that understand context and meaning, not just keywords.</p>
        <ul>
            <li>Hybrid search (semantic + keyword)</li>
            <li>AI-generated explanations</li>
            <li>Real-time relevance scoring</li>
        </ul>
        <p><em>Example: "movies about redemption and hope"</em></p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔍 Try Semantic Search", key="btn_search", use_container_width=True):
        st.switch_page("pages/1_Semantic_Search.py")

    st.markdown("""
    <div class="feature-card">
        <h3>💬 Conversational Chatbot</h3>
        <p><strong>RAG-Powered Assistant</strong></p>
        <p>Chat naturally about movies with AI that remembers your conversation context.</p>
        <ul>
            <li>Context-aware dialogue</li>
            <li>Retrieval-Augmented Generation</li>
            <li>Source attribution</li>
        </ul>
        <p><em>Example: "Tell me about space movies" → "Which is highest rated?"</em></p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("💬 Start Chatting", key="btn_chat", use_container_width=True):
        st.switch_page("pages/3_Conversational_Chatbot.py")

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 AI Recommendations</h3>
        <p><strong>Vector Similarity Engine</strong></p>
        <p>Discover movies similar to your favorites using advanced vector embeddings.</p>
        <ul>
            <li>"More like this" functionality</li>
            <li>AI explains why movies match</li>
            <li>Similarity scoring</li>
        </ul>
        <p><em>Example: Select "Interstellar" → Get similar space/survival movies</em></p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🎯 Get Recommendations", key="btn_rec", use_container_width=True):
        st.switch_page("pages/2_Movie_Recommendations.py")

    st.markdown("""
    <div class="feature-card">
        <h3>📊 Analytics Dashboard</h3>
        <p><strong>Search Quality Insights</strong></p>
        <p>Monitor search performance, quality metrics, and system health.</p>
        <ul>
            <li>Search quality testing</li>
            <li>Performance benchmarking</li>
            <li>Vector similarity analysis</li>
        </ul>
        <p><em>Track relevance, latency, and embedding effectiveness</em></p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📊 View Analytics", key="btn_analytics", use_container_width=True):
        st.switch_page("pages/4_Analytics_Dashboard.py")

# New Features Section
st.markdown("---")
st.markdown("## 🆕 Advanced Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <h2>🔧</h2>
        <h4>Advanced Filters</h4>
        <p>Filter by genre, year, rating, and more</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Explore Filters", key="btn_filters", use_container_width=True):
        st.switch_page("pages/5_Advanced_Filters.py")

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <h2>⚖️</h2>
        <h4>Search Comparison</h4>
        <p>Compare semantic vs keyword search</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Compare Results", key="btn_compare", use_container_width=True):
        st.switch_page("pages/6_Search_Comparison.py")

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <h2>📥</h2>
        <h4>Export & Share</h4>
        <p>Download results and share searches</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Export Tools", key="btn_export", use_container_width=True):
        st.info("Coming soon! Export to CSV/JSON and shareable URLs")

# Technical Stack
st.markdown("---")
st.markdown("## 🛠️ Technical Architecture")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🗄️ Vector Database**
    - OpenSearch Serverless
    - HNSW indexing
    - 1024-dim embeddings
    - Sub-linear search time
    """)

with col2:
    st.markdown("""
    **🤖 AI Models**
    - Titan Text Embeddings V2
    - Claude Opus 4.1 (RAG)
    - Claude Haiku 3 (Fast)
    - Cross-region profiles
    """)

with col3:
    st.markdown("""
    **⚡ Performance**
    - <100ms search latency
    - Real-time indexing
    - Auto-scaling
    - Serverless architecture
    """)

# Use Cases
st.markdown("---")
st.markdown("## 🎯 Real-World Applications")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### E-Commerce
    - Natural language product search
    - "Similar items" recommendations
    - Customer service chatbots
    - Search analytics

    ### Media & Entertainment
    - Content discovery platforms
    - "More like this" features
    - Personalized recommendations
    - User engagement tracking
    """)

with col2:
    st.markdown("""
    ### Knowledge Management
    - Document search systems
    - Q&A for documentation
    - Semantic wiki search
    - Context-aware help

    ### Customer Support
    - Intent-based routing
    - Support article search
    - Conversational assistants
    - Quality monitoring
    """)

# Key Benefits
st.markdown("---")
st.markdown("## ✨ Why AI-Powered Search?")

benefits_col1, benefits_col2, benefits_col3 = st.columns(3)

with benefits_col1:
    st.success("""
    **🎯 Better Relevance**
    - Understands intent
    - Semantic meaning
    - Context-aware
    - Multi-dimensional
    """)

with benefits_col2:
    st.info("""
    **⚡ Fast & Scalable**
    - <100ms latency
    - Millions of docs
    - Auto-scaling
    - Serverless
    """)

with benefits_col3:
    st.warning("""
    **💰 Cost Effective**
    - Pay per use
    - No infrastructure
    - Managed service
    - 90% savings with Haiku
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Built with AWS OpenSearch Serverless & Amazon Bedrock</strong></p>
    <p>Demonstrating production-ready AI search capabilities</p>
    <p><em>From semantic search to conversational AI - all in one platform</em></p>
</div>
""", unsafe_allow_html=True)

# Quick Links
with st.sidebar:
    st.markdown("### 🚀 Quick Navigation")
    st.markdown("**Core Features:**")
    if st.button("🔍 Semantic Search", use_container_width=True):
        st.switch_page("pages/1_Semantic_Search.py")
    if st.button("🎯 Recommendations", use_container_width=True):
        st.switch_page("pages/2_Movie_Recommendations.py")
    if st.button("💬 Chatbot", use_container_width=True):
        st.switch_page("pages/3_Conversational_Chatbot.py")
    if st.button("📊 Analytics", use_container_width=True):
        st.switch_page("pages/4_Analytics_Dashboard.py")

    st.markdown("---")
    st.markdown("**Advanced Tools:**")
    if st.button("🔧 Advanced Filters", use_container_width=True):
        st.switch_page("pages/5_Advanced_Filters.py")
    if st.button("⚖️ Search Comparison", use_container_width=True):
        st.switch_page("pages/6_Search_Comparison.py")

    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("""
    - [README.md](README.md)
    - [Setup Guide](SETUP_SUMMARY.md)
    - [Feature Docs](EXTENDED_FEATURES.md)
    - [Demo Script](DEMO_SCRIPT.md)
    """)

    if config_exists:
        st.markdown("---")
        st.success("✅ System Online")
        st.caption(f"Collection: {config.get('collection_name', 'N/A')}")
    else:
        st.error("❌ Setup Required")
