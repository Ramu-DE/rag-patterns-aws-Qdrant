"""
Conversational Movie Chatbot with RAG
Chat naturally about movies with context-aware AI
"""

import streamlit as st
import json
import os
import sys
import boto3
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.movie_search import MovieSearch

st.set_page_config(
    page_title="Movie Chatbot",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Conversational Movie Assistant")
st.markdown("Chat naturally about movies with AI-powered search and memory")

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

@st.cache_resource
def get_bedrock_client():
    return boto3.client('bedrock-runtime', region_name='us-west-2')

try:
    searcher = get_search_client()
    bedrock = get_bedrock_client()
except Exception as e:
    st.error(f"❌ Error connecting: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.context = []

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

# Sidebar
st.sidebar.header("Chat Settings")

model_option = st.sidebar.selectbox(
    "Claude Model",
    [
        "Opus 4.1 (Most Capable)",
        "Haiku 3 (Fast & Economical)"
    ]
)

model_id = (
    "us.anthropic.claude-opus-4-1-20250805-v1:0"
    if "Opus" in model_option
    else "us.anthropic.claude-3-haiku-20240307-v1:0"
)

search_results_count = st.sidebar.slider(
    "Search Results for Context",
    min_value=1,
    max_value=10,
    value=3,
    help="Number of movies to retrieve for context"
)

if st.sidebar.button("🔄 Clear Chat History"):
    st.session_state.messages = []
    st.session_state.context = []
    st.session_state.chat_started = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 Example Questions

- What movies are about redemption?
- Tell me about Interstellar
- Suggest a dark psychological thriller
- What's similar to The Matrix?
- I want a feel-good comedy
- Movies with time travel themes
""")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show search context if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources Used"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"{i}. **{source['title']}** ({source['year']}) - ⭐ {source['rating']}")

# Chat input
if prompt := st.chat_input("Ask me anything about movies..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_started = True

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Search for relevant movies
                search_results = searcher.hybrid_search(
                    query=prompt,
                    top_k=search_results_count
                )

                # Build conversation context
                context_parts = []
                if search_results:
                    context_parts.append("**Movies from our database that might be relevant:**\n")
                    for i, movie in enumerate(search_results, 1):
                        context_parts.append(
                            f"{i}. **{movie['title']}** ({movie['year']}) - {movie['rating']}/10\n"
                            f"   Plot: {movie['plot']}\n"
                            f"   Genre: {', '.join(movie['genre']) if isinstance(movie['genre'], list) else movie['genre']}\n"
                        )

                context = "\n".join(context_parts)

                # Build conversation history
                conversation_history = []
                for msg in st.session_state.messages[-6:]:  # Last 3 exchanges
                    if msg["role"] != "assistant" or "sources" not in msg:
                        conversation_history.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })

                # Create system prompt
                system_prompt = """You are a knowledgeable and friendly movie assistant. You have access to a movie database and can help users discover films, answer questions about movies, and provide recommendations.

Guidelines:
- Be conversational and engaging
- Reference specific movies from the database when relevant
- Explain why you're recommending certain movies
- Ask clarifying questions if the user's request is vague
- Be honest if you don't have information about something
- Keep responses concise but informative (2-3 paragraphs max)
"""

                # Build full prompt
                if context:
                    user_message = f"{context}\n\nUser Question: {prompt}"
                else:
                    user_message = prompt

                # Call Claude
                messages = [
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]

                # Add conversation history (if exists)
                if len(conversation_history) > 1:
                    messages = conversation_history[:-1] + messages

                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "system": system_prompt,
                    "messages": messages
                }

                response = bedrock.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body)
                )

                response_body = json.loads(response['body'].read())
                assistant_message = response_body['content'][0]['text']

                # Display response
                st.markdown(assistant_message)

                # Add to chat history
                message_data = {
                    "role": "assistant",
                    "content": assistant_message
                }

                if search_results:
                    message_data["sources"] = search_results

                st.session_state.messages.append(message_data)

                # Show sources
                if search_results:
                    with st.expander("📚 Sources Used"):
                        for i, source in enumerate(search_results, 1):
                            st.markdown(
                                f"{i}. **{source['title']}** ({source['year']}) - "
                                f"⭐ {source['rating']} "
                                f"(Relevance: {source['search_score']:.3f})"
                            )

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Info section at bottom
if not st.session_state.chat_started:
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🌟 Features

        - **Conversational Memory**: Remembers your conversation
        - **Context-Aware**: Searches database for relevant movies
        - **Natural Language**: Chat like you would with a friend
        - **Smart Recommendations**: AI understands your preferences
        - **Source Citations**: See which movies informed the answer
        """)

    with col2:
        st.markdown("""
        ### 🎯 How to Use

        1. Ask any question about movies
        2. AI searches the database for relevant content
        3. Claude generates a conversational response
        4. Continue the conversation naturally
        5. Ask follow-up questions or refine your search

        **Example Flow:**
        - "Tell me about sci-fi movies"
        - "Which one has the best ratings?"
        - "What's it about?"
        """)

    st.markdown("---")

    st.info("""
    **💡 Tip:** This chatbot uses **Retrieval-Augmented Generation (RAG)** - it searches the movie database
    first, then uses that information to provide accurate, grounded answers. This prevents hallucination
    and ensures responses are based on real data.
    """)
