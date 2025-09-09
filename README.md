# Semantic Cache Tools

A Python library for semantic caching using embeddings and approximate nearest neighbor search. Cache function results based on semantic similarity of inputs rather than exact matches.

## Features

- **Semantic Caching Decorator**: Cache function results based on semantic similarity of string inputs
- **FuzzyDict**: Dictionary with approximate key matching using embeddings
- **FuzzyLruCache**: LRU cache with fuzzy matching capabilities
- **Fast Similarity Search**: Uses Annoy library for efficient approximate nearest neighbor search
- **Flexible Embedding Functions**: Support for any embedding function (OpenAI, Ollama, custom)

## Installation

```bash
pip install git+https://github.com/tonybaloney/semantic_cache.git
```

## Quick Start

### Basic Semantic Caching

```python
from semantic_cache import semantic_cache

# Simple hash-based similarity (for demonstration)
@semantic_cache(
    embed_func=lambda x: (hash(x.lower()),),
    max_size=100,
    max_distance=0.0  # Only exact matches
)
def expensive_function(query: str) -> str:
    # Simulate expensive computation
    return f"Result for: {query}"

# These will hit the cache due to case-insensitive matching
result1 = expensive_function("What is Python?")
result2 = expensive_function("what is python?")  # Cache hit!
assert result1 == result2
```

### Using with OpenAI Embeddings

```python
import openai
from semantic_cache import semantic_cache

client = openai.Client()

@semantic_cache(
    embed_func=lambda text: tuple(
        client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        ).data[0].embedding
    ),
    max_distance=0.80,  # Allow similar queries
    max_size=50
)
def ask_ai(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
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
    max_distance=0.9,
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

def simple_embedding(text: str) -> tuple[float, ...]:
    # Simple character-based embedding for demo
    return tuple(float(ord(c)) for c in text.lower()[:10])

# Create a fuzzy dictionary
fuzzy_dict = FuzzyDict(
    max_distance=10.0,
    embed_func=simple_embedding
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
    embed_func=lambda x: (hash(x.lower()),),
    capacity=3,  # Only keep 3 items
    max_distance=0.0
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
- **Approximate matches**: O(log n) average case with Annoy, O(n) fallback without Annoy
- **Memory efficient**: Embeddings cached to avoid recomputation

### Time Complexity Summary

For a cache with `m` items and embeddings of dimension `d`:

| Operation | With Annoy | Without Annoy |
|-----------|------------|---------------|
| Insert    | O(E) amortized | O(E) amortized |
| Lookup    | O(E + log m) | O(E + m·d) |
| Delete    | O(1) | O(1) |

Where `E` is the cost of computing embeddings (typically O(d) for simple embeddings).

## Requirements

- Python 3.9+
- annoy>=1.17.0
- openai (optional, for OpenAI embeddings)

## Development

```bash
git clone https://github.com/yourusername/semantic_cache_tools.git
cd semantic_cache_tools
pip install -e .[dev]
pytest
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## Examples Repository

See the `tests/` directory for more examples:
- `test_ollama.py`: Integration with Ollama/local LLMs
- `test_mock.py`: Comprehensive test suite with various scenarios

