from typing import Generator
from src.semantic_cache import semantic_cache, FuzzyDict, FuzzyLruCache

# A mock function to give the lower-case hash as the embedding
# and yield two different answers
# Should return the first answer for lower/upper case variations of the same question

@semantic_cache(embed_func=lambda x: (hash(x.lower()),), max_size=10, min_distance=0.0)
def mock_function(x: str) -> Generator[str, None, None]:
    yield "the first answer"

    yield "the second answer"


def test_fuzzy_dict():
    f = FuzzyDict(min_distance=0.1, embed_func=lambda x: (hash(x.lower()),))
    f["Hello"] = "World"
    assert "hello" in f
    assert f["hello"] == "World"

    assert "yo" not in f


def test_fuzzy_lru_cache():
    cache = FuzzyLruCache(embed_func=lambda x: (hash(x.lower()),), capacity=2, min_distance=0.0)
    cache.put("Hello", "World")
    assert cache.get("hello") == "World"

    cache.put("Foo", "Bar")
    assert cache.get("foo") == "Bar"

    cache.put("New", "Entry")
    assert cache.get("hello") is None  # Evicted due to LRU policy
    assert cache.get("foo") == "Bar"
    assert cache.get("new") == "Entry"


def test_case_match():
    results = list(mock_function("What is the capital of France?"))
    assert results == ["the first answer"]

    results = list(mock_function("What is the capital of france?"))
    assert results == ["the first answer"]  # Cached result
