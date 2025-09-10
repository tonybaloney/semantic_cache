# Semantic Cache Tools

A Python library for semantic caching using embeddings and approximate nearest neighbor (ANN) search. Cache function results based on semantic similarity of inputs rather than exact matches.

## Features

- **Semantic Caching Decorator**: Cache function results based on semantic similarity of string inputs
- **FuzzyDict**: Dictionary with approximate key matching using embeddings
- **FuzzyLruCache**: LRU cache with fuzzy matching capabilities
- **Fast Similarity Search**: Uses Spotify's Annoy library for efficient approximate nearest neighbor search

## Installation

```bash
pip install git+https://github.com/tonybaloney/semantic_cache.git
```

## Quick Start

Use the `@semantic_cache` decorator like the `functools.lru_cache` decorator. It can be used to cache anything with a single `str` parameter and is particularly useful for caching LLM calls.

Adjust the `max_distance` settings based on your requirements.

### Basic Semantic Caching

```python
import openai
from semantic_cache import semantic_cache

client = openai.Client()  # this can be anything

@semantic_cache(
    embed_func=lambda text: tuple(
        client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
    ),
    max_distance=0.80,  # Cosine similarity threshold: 0.8 or higher for matches
    max_size=512 # max cache entries
)
def ask_ai(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

# These similar questions will likely hit the cache
answer1 = ask_ai("What is the capital of France?")
answer2 = ask_ai("What's the capital city of France?")  # Likely cache hit
```

### Using with Ollama (Local LLMs)

```python
import openai
from semantic_cache import semantic_cache

# Connect to local Ollama instance
client = openai.Client(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

@semantic_cache(
    embed_func=lambda text: tuple(
        client.embeddings.create(
            model="nomic-embed-text:latest",
            input=text
        ).data[0].embedding
    ),
    max_distance=0.9,  # High cosine similarity threshold
    max_size=10
)
def local_llm_query(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama3.2:latest",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
```

## Advanced Usage

### FuzzyDict - Dictionary with Approximate Matching

```python
from semantic_cache import FuzzyDict

def embed_func(text: str) -> tuple[float, ...]:
    return tuple(
        client.embeddings.create(
            model="nomic-embed-text:latest",
            input=text
        ).data[0].embedding
    )

# Create a fuzzy dictionary
fuzzy_dict = FuzzyDict(
    max_distance=0.8,  # Cosine similarity threshold (0.8 or higher matches)
    embed_func=embed_func
)

# Add some entries
fuzzy_dict["hello"] = "greeting"
fuzzy_dict["goodbye"] = "farewell"

# Exact matches work as expected
assert fuzzy_dict["hello"] == "greeting"

# Approximate matches also work
assert fuzzy_dict["hallo"] == "greeting"  # Close to "hello"
assert "helo" in fuzzy_dict  # Missing 'l', but close enough

# Distant keys won't match
assert "xyz" not in fuzzy_dict
```

### FuzzyLruCache - LRU Cache with Fuzzy Matching

```python
from semantic_cache import FuzzyLruCache

cache = FuzzyLruCache(
    embed_func=embed_func,
    capacity=3,  # Only keep 3 items
    max_distance=1.0  # Very high similarity required (near exact matches)
)

# Add items
cache.put("first", "value1")
cache.put("second", "value2")
cache.put("third", "value3")

# Access items (affects LRU order)
assert cache.get("first") == "value1"

# Add another item - should evict least recently used
cache.put("fourth", "value4")
assert cache.get("second") is None  # Evicted (was LRU)
assert cache.get("first") == "value1"  # Still present
```

## Performance

The library uses the Annoy (Approximate Nearest Neighbors) library for efficient similarity search:

- **Exact matches**: O(1) lookup time
- **Approximate matches**: O(E + log m) average case using Annoy's angular metric
- **Memory efficient**: Embeddings cached to avoid recomputation
- **No fallback**: Uses only Annoy for similarity search (no linear scan fallback)

### Time Complexity Summary

For a cache with `m` items and embeddings of dimension `d`:

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Insert    | O(E) amortized | E = embedding computation cost |
| Lookup    | O(E + log m) | Uses Annoy's fast ANN search |
| Delete    | O(1) | Immediate removal |

Where `E` is the cost of computing embeddings (typically O(d) for simple embeddings).

### Distance Metrics

The library uses **cosine similarity** for `max_distance` parameters:
- `max_distance=1.0`: Only identical vectors match
- `max_distance=0.8`: Vectors with cosine similarity ≥ 0.8 match  
- `max_distance=0.0`: Only orthogonal vectors match
- `max_distance=-1.0`: All vectors match (including opposite directions)

Internally, the library converts cosine similarity to cosine distance for Annoy's angular metric using the formula: `cosine_distance = sqrt(2 - 2 * cosine_similarity)`

## Requirements

- Python 3.9+
- annoy>=1.17.0
- openai (optional, for OpenAI embeddings)

## Development

```bash
git clone https://github.com/tonybaloney/semantic_cache.git
cd semantic_cache
pip install -e .[dev]
pytest
```
