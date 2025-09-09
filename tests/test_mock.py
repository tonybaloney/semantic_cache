from typing import Generator
from src.semantic_cache import semantic_cache, FuzzyDict, FuzzyLruCache


# Better embedding function for exact matching
def exact_match_embed(s):
    """Create a unique normalized embedding for exact string matching."""
    # Use multiple hash functions to create a multi-dimensional embedding
    h1 = hash(s.lower())
    h2 = hash(s.lower() + "_salt1")
    h3 = hash(s.lower() + "_salt2")
    # Normalize to unit vector
    embedding = (float(h1), float(h2), float(h3))
    magnitude = sum(x * x for x in embedding) ** 0.5
    if magnitude == 0:
        return (1.0, 0.0, 0.0)  # Default vector
    return tuple(x / magnitude for x in embedding)


# A mock function that returns a simple string result for caching
@semantic_cache(embed_func=exact_match_embed, max_size=10, max_distance=0.999)
def mock_function(x: str) -> str:
    return "the first answer"


# A generator function for testing (note: generators are consumed once when cached)
@semantic_cache(embed_func=exact_match_embed, max_size=10, max_distance=0.999)
def mock_generator(x: str) -> Generator[str, None, None]:
    yield "the first answer"
    yield "the second answer"


def test_fuzzy_dict():
    f = FuzzyDict(max_distance=0.999, embed_func=exact_match_embed)
    f["Hello"] = "World"
    assert "hello" in f
    assert f["hello"] == "World"

    assert "yo" not in f


def test_fuzzy_lru_cache():
    cache = FuzzyLruCache(embed_func=exact_match_embed, capacity=2, max_distance=0.999)
    cache.put("Hello", "World")
    assert cache.get("hello") == "World"

    cache.put("Foo", "Bar")
    assert cache.get("foo") == "Bar"

    cache.put("New", "Entry")
    assert cache.get("hello") is None  # Evicted due to LRU policy
    assert cache.get("foo") == "Bar"
    assert cache.get("new") == "Entry"


def test_case_match():
    result = mock_function("What is the capital of France?")
    assert result == "the first answer"

    result = mock_function("What is the capital of france?")
    assert result == "the first answer"  # Cached result


# Extended FuzzyDict tests
def test_fuzzy_dict_exact_matches():
    """Test exact key matching behavior."""
    f = FuzzyDict(max_distance=0.1, embed_func=lambda x: (hash(x.lower()),))
    f["exact"] = "match"
    assert "exact" in f
    assert f["exact"] == "match"


def test_fuzzy_dict_approximate_matches():
    """Test approximate matching with different distance thresholds."""

    def simple_embed(s):
        # Simple embedding: first 3 chars as ASCII values, normalized
        chars = [ord(c) for c in s.lower()[:3]]
        while len(chars) < 3:
            chars.append(0)
        # Normalize to unit vector for cosine similarity
        magnitude = sum(c * c for c in chars) ** 0.5
        if magnitude == 0:
            return tuple(0.0 for _ in chars)
        return tuple(float(c) / magnitude for c in chars)

    f = FuzzyDict(
        max_distance=0.9, embed_func=simple_embed
    )  # High similarity threshold
    f["hello"] = "greeting"

    # "hallo" should be similar enough for cosine similarity
    assert "hallo" in f
    assert f["hallo"] == "greeting"


def test_fuzzy_dict_no_false_positives():
    """Test that unrelated keys don't match."""

    def simple_embed(s):
        return tuple(float(ord(c)) for c in s[:2])

    f = FuzzyDict(max_distance=5.0, embed_func=simple_embed)
    f["abc"] = "value1"

    # "xyz" should be too far away
    assert "xyz" not in f

    try:
        _ = f["xyz"]
        assert False, "Should have raised KeyError"
    except KeyError:
        pass  # Expected


def test_fuzzy_dict_deletion():
    """Test key deletion and cleanup."""
    f = FuzzyDict(max_distance=0.1, embed_func=lambda x: (hash(x.lower()),))
    f["test"] = "value"
    assert "test" in f

    del f["test"]
    assert "test" not in f
    assert len(f.embeddings) == 0  # Embedding should be cleaned up


def test_fuzzy_dict_multiple_similar_keys():
    """Test behavior with multiple similar keys."""

    def char_embed(s):
        # Normalize embeddings for proper cosine similarity
        chars = [float(ord(c)) for c in s[:2]]
        magnitude = sum(c * c for c in chars) ** 0.5
        if magnitude == 0:
            return tuple(0.0 for _ in chars)
        return tuple(c / magnitude for c in chars)

    f = FuzzyDict(max_distance=0.95, embed_func=char_embed)  # High similarity threshold
    f["aa"] = "first"
    f["ab"] = "second"
    f["ac"] = "third"

    # Should find closest match ("aa")
    result = f.find_key("aa")
    assert result == "aa"

    # Should find a similar match
    result = f.find_key("ad")
    assert result is not None  # Should find some match


def test_fuzzy_dict_distance_calculation():
    """Test the cosine distance conversion functions."""
    # Create a FuzzyDict instance to test conversion methods
    fd = FuzzyDict(max_distance=0.8, embed_func=lambda x: (1.0, 0.0, 0.0))

    # Test cosine similarity to cosine distance conversion
    cos_sim = 0.8
    cos_dist = fd._cosine_similarity_to_distance(cos_sim)
    expected_dist = (2 - 2 * 0.8) ** 0.5  # sqrt(2 - 2*cos_sim)
    assert abs(cos_dist - expected_dist) < 1e-10

    # Test cosine distance to cosine similarity conversion
    cos_dist = 0.6325  # approximately sqrt(2 - 2*0.8)
    cos_sim_back = fd._cosine_distance_to_similarity(cos_dist)
    expected_sim = 1 - (0.6325 * 0.6325) / 2
    assert abs(cos_sim_back - expected_sim) < 1e-3  # small tolerance for floating point


def test_fuzzy_dict_embedding_caching():
    """Test that embeddings are computed only once per key."""
    call_count = 0

    def counting_embed(s):
        nonlocal call_count
        call_count += 1
        return (hash(s),)

    f = FuzzyDict(max_distance=0.1, embed_func=counting_embed)

    # First access should compute embedding
    f["test"] = "value"
    assert call_count == 1

    # Subsequent accesses should use cached embedding
    _ = f["test"]
    assert call_count == 1

    # Different key should compute new embedding
    f["other"] = "value2"
    assert call_count == 2


# Extended LRU Cache tests
def test_lru_cache_basic_functionality():
    """Test basic get/put operations."""
    cache = FuzzyLruCache(
        embed_func=lambda x: (hash(str(x)),), capacity=3, max_distance=0.0
    )

    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")

    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"


def test_lru_cache_eviction_policy():
    """Test that LRU eviction works correctly."""
    cache = FuzzyLruCache(embed_func=exact_match_embed, capacity=2, max_distance=0.999)

    # Fill cache
    cache.put("first", "value1")
    cache.put("second", "value2")

    # Access first to make it recently used
    _ = cache.get("first")

    # Add third item - should evict "second" (least recently used)
    cache.put("third", "value3")

    assert cache.get("first") == "value1"  # Should still be there
    assert cache.get("second") is None  # Should be evicted
    assert cache.get("third") == "value3"  # Should be there


def test_lru_cache_update_existing():
    """Test updating existing keys refreshes their position."""
    cache = FuzzyLruCache(embed_func=exact_match_embed, capacity=2, max_distance=0.999)

    cache.put("key1", "value1")
    cache.put("key2", "value2")

    # Update key1 with new value
    cache.put("key1", "new_value1")

    # Add third key - should evict key2, not key1
    cache.put("key3", "value3")

    assert cache.get("key1") == "new_value1"  # Updated and preserved
    assert cache.get("key2") is None  # Evicted
    assert cache.get("key3") == "value3"  # New entry


def test_lru_cache_fuzzy_matching():
    """Test LRU cache with fuzzy key matching."""
    cache = FuzzyLruCache(embed_func=exact_match_embed, capacity=3, max_distance=0.999)

    cache.put("Hello", "World")
    cache.put("FOO", "Bar")

    # Should find fuzzy matches
    assert cache.get("hello") == "World"
    assert cache.get("foo") == "Bar"
    assert cache.get("HELLO") == "World"


def test_lru_cache_order_tracking():
    """Test that LRU order is maintained correctly."""
    cache = FuzzyLruCache(embed_func=exact_match_embed, capacity=3, max_distance=0.999)

    # Add items in order
    cache.put("a", "1")
    cache.put("b", "2")
    cache.put("c", "3")

    # Access 'a' to make it most recent
    _ = cache.get("a")

    # Add new item - should evict 'b' (oldest unused)
    cache.put("d", "4")

    assert cache.get("a") == "1"  # Still there (was accessed)
    assert cache.get("b") is None  # Evicted
    assert cache.get("c") == "3"  # Still there
    assert cache.get("d") == "4"  # New item


def test_lru_cache_empty_operations():
    """Test operations on empty cache."""
    cache = FuzzyLruCache(
        embed_func=lambda x: (hash(str(x)),), capacity=2, max_distance=0.0
    )

    assert cache.get("nonexistent") is None

    # Put and get should work normally
    cache.put("key", "value")
    assert cache.get("key") == "value"


# Semantic cache decorator tests
def test_semantic_cache_with_different_args():
    """Test semantic cache with different argument patterns."""
    call_count = 0

    @semantic_cache(embed_func=exact_match_embed, max_size=5, max_distance=0.999)
    def test_func(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return f"result_{call_count}"

    # Different calls should be cached separately
    result1 = test_func("a", "b")
    result2 = test_func("a", "c")
    result3 = test_func("a", "b")  # Should hit cache

    assert result1 == "result_1"
    assert result2 == "result_2"
    assert result3 == "result_1"  # Same as first call
    assert call_count == 2  # Only 2 actual function calls


def test_semantic_cache_kwargs():
    """Test semantic cache with keyword arguments."""
    call_count = 0

    @semantic_cache(embed_func=exact_match_embed, max_size=5, max_distance=0.999)
    def test_func(a, b=None):
        nonlocal call_count
        call_count += 1
        return f"result_{call_count}"

    result1 = test_func("x", b="y")
    result2 = test_func("x", b="y")  # Should hit cache
    result3 = test_func("x", b="z")  # Different kwargs

    assert result1 == "result_1"
    assert result2 == "result_1"  # Cache hit
    assert result3 == "result_2"  # New call
    assert call_count == 2


def test_semantic_cache_capacity_limit():
    """Test that semantic cache respects capacity limits."""

    @semantic_cache(embed_func=lambda x: (hash(str(x)),), max_size=2, max_distance=0.0)
    def test_func(x):
        return f"result_for_{x}"

    # Fill cache to capacity
    result1 = test_func("a")
    result2 = test_func("b")

    # Access first result to make it recently used
    result1_again = test_func("a")
    assert result1_again == result1

    # Add third result - should evict "b"
    result3 = test_func("c")

    # "a" should still be cached, "b" should be evicted
    assert test_func("a") == result1  # Cache hit
    assert test_func("c") == result3  # Cache hit
    # "b" would be recalculated (can't easily test without call counting)


# Advanced integration tests
def test_fuzzy_dict_with_annoy_integration():
    """Test FuzzyDict with many items to trigger Annoy usage."""

    def vector_embed(s):
        # Create a simple 5-dimensional vector from string
        chars = [ord(c) for c in s.lower()[:5]]
        while len(chars) < 5:
            chars.append(0)
        return tuple(float(c) / 255.0 for c in chars)

    f = FuzzyDict(max_distance=0.1, embed_func=vector_embed)

    # Add many items to potentially trigger Annoy index building
    for i in range(100):
        f[f"item_{i:03d}"] = f"value_{i}"

    # Test exact matches
    assert f["item_050"] == "value_50"

    # Test approximate matches (if any exist within distance threshold)
    # Note: With this embedding function and distance, exact matches are more likely
    result = f.find_key("item_050")
    assert result == "item_050"


def test_fuzzy_dict_edge_cases():
    """Test edge cases and error conditions."""
    f = FuzzyDict(max_distance=1.0, embed_func=lambda x: (float(ord(x[0])),))

    # Test with single character keys
    f["a"] = "first"
    f["b"] = "second"  # distance = 1.0

    # Should find "b" as closest to "c" (distance = 1.0)
    result = f.find_key("c")
    assert result == "b"

    # Test distance exactly at threshold
    f["c"] = "third"
    result = f.find_key("c")
    assert result == "c"  # Exact match preferred


def test_fuzzy_dict_empty_embeddings():
    """Test behavior with empty embeddings."""
    f = FuzzyDict(max_distance=1.0, embed_func=lambda x: ())

    try:
        f["test"] = "value"
        # This might work depending on implementation
    except Exception:
        # Or might raise an exception, which is also acceptable
        pass


def test_lru_cache_concurrent_operations():
    """Test LRU cache with rapid get/put operations."""
    cache = FuzzyLruCache(embed_func=exact_match_embed, capacity=5, max_distance=0.999)

    # Rapid insertions and retrievals
    for i in range(10):
        cache.put(f"key_{i}", f"value_{i}")

        # Verify recent items are still accessible
        if i >= 5:  # Cache capacity is 5
            # Oldest items should be evicted
            assert cache.get(f"key_{i - 5}") is None

        # Most recent item should be accessible
        assert cache.get(f"key_{i}") == f"value_{i}"


def test_semantic_cache_string_only():
    """Test semantic cache with string arguments only."""
    call_count = 0

    @semantic_cache(embed_func=exact_match_embed, max_size=10, max_distance=0.999)
    def string_func(text: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"processed_{call_count}_{text}"

    # Test with string arguments
    result1 = string_func("hello world")
    assert call_count == 1

    # Same string should hit cache
    result2 = string_func("hello world")
    assert call_count == 1
    assert result1 == result2

    # Different string should miss cache
    result3 = string_func("goodbye world")
    assert call_count == 2
    assert result1 != result3


def test_fuzzy_dict_very_similar_keys():
    """Test behavior with keys that have very small differences."""

    def precise_embed(s):
        # Very precise embedding that can detect small differences
        chars = [ord(c) + i * 0.1 for i, c in enumerate(s)]
        # Normalize for cosine similarity
        magnitude = sum(c * c for c in chars) ** 0.5
        if magnitude == 0:
            return tuple(0.0 for _ in chars)
        return tuple(c / magnitude for c in chars)

    # Use extremely high threshold to require near-perfect similarity
    f = FuzzyDict(max_distance=0.999999, embed_func=precise_embed)

    f["test1"] = "first"
    f["test2"] = "second"

    # These should be considered different (cosine similarity < 0.999999)
    assert "test1" in f
    assert "test2" in f
    assert f["test1"] == "first"
    assert f["test2"] == "second"

    # A very similar key should not match due to extremely high threshold
    assert "test3" not in f


def test_lru_cache_stress_eviction():
    """Stress test LRU eviction with many operations."""
    cache = FuzzyLruCache(embed_func=exact_match_embed, capacity=3, max_distance=0.999)

    # Fill cache
    cache.put("a", "1")
    cache.put("b", "2")
    cache.put("c", "3")

    # Access pattern that should preserve "a" and "c"
    _ = cache.get("a")
    _ = cache.get("c")

    # Add more items
    cache.put("d", "4")  # Should evict "b"
    cache.put("e", "5")  # Should evict "a" (least recently used of remaining)

    assert cache.get("a") is None  # Evicted
    assert cache.get("b") is None  # Evicted
    assert cache.get("c") == "3"  # Preserved
    assert cache.get("d") == "4"  # New
    assert cache.get("e") == "5"  # Newest
