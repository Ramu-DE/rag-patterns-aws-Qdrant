#!/usr/bin/env python3
"""
Quick test of movie search functionality
"""

import json
import time
from utils.movie_search import MovieSearch

def main():
    print("=" * 70)
    print("Movie Search Test")
    print("=" * 70)
    print()

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Initialize search
    print("Initializing search...")
    searcher = MovieSearch(config)
    print("✓ Search client initialized")
    print()

    # Wait for indexing to complete
    print("Waiting for documents to be searchable...")
    time.sleep(5)

    # Test queries
    test_queries = [
        "movies about friendship and redemption",
        "space exploration and survival",
        "psychological thriller with plot twists"
    ]

    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Query: {query}")
        print('=' * 70)

        try:
            # Semantic search
            results, answer = searcher.search_and_answer(
                query=query,
                search_type="hybrid",
                top_k=3,
                use_claude=True,
                claude_model="us.anthropic.claude-opus-4-1-20250805-v1:0"  # Cross-region inference profile
            )

            if results:
                print(f"\n🎬 Top Results:")
                for i, movie in enumerate(results, 1):
                    print(f"  {i}. {movie['title']} ({movie['year']}) - ⭐ {movie['rating']}")
                    print(f"     Score: {movie['search_score']:.3f}")

                if answer:
                    print(f"\n🤖 AI Answer:")
                    print(f"  {answer['answer'][:200]}...")
            else:
                print("  No results found")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
