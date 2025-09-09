"""
Test script to demonstrate Annoy performance benefits for larger datasets.
Run this to see the difference between linear scan and ANN search.
"""

from src.semantic_cache import FuzzyDict
import time
import random

def random_embed(s):
    """Create a 10-dimensional random embedding based on string hash."""
    random.seed(hash(s))
    return tuple(random.random() for _ in range(10))

def test_performance():
    print("Testing FuzzyDict performance with Annoy integration")
    print("=" * 50)
    
    # Create FuzzyDict with moderate min_distance
    f = FuzzyDict(min_distance=0.3, embed_func=random_embed)
    
    # Add many items
    n_items = 1000
    print(f"Adding {n_items} items...")
    start = time.time()
    for i in range(n_items):
        f[f'key_{i:04d}'] = f'value_{i}'
    add_time = time.time() - start
    print(f"Added {n_items} items in {add_time:.3f}s")
    
    # Test lookups
    test_keys = [f'key_{i:04d}' for i in range(0, n_items, 100)]  # Every 100th key
    
    print(f"\nTesting {len(test_keys)} lookups...")
    start = time.time()
    found = 0
    for key in test_keys:
        result = f.find_key(key)
        if result:
            found += 1
    lookup_time = time.time() - start
    
    print(f"Found {found}/{len(test_keys)} keys in {lookup_time:.3f}s")
    print(f"Average lookup time: {lookup_time/len(test_keys)*1000:.2f}ms")
    print(f"Annoy index was {'built' if f._ann_index else 'not built'}")
    
    if f._ann_index:
        print(f"Annoy index contains {f._ann_index.get_n_items()} items")

if __name__ == "__main__":
    test_performance()
