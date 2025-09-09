from typing import OrderedDict, TypeVar, Callable

__all__ = ["semantic_cache", "FuzzyDict", "FuzzyLruCache"]

TKey = TypeVar('TKey')
TValue = TypeVar('TValue')
GetEmbeddingFunc = Callable[[str], tuple[float, ...]]


class FuzzyDict(dict[TKey, TValue]):
    embeddings: dict[TKey, tuple[float, ...]]

    def __init__(self, min_distance: float, embed_func: GetEmbeddingFunc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_distance = min_distance
        self.embed_func = embed_func
        self.embeddings: dict[TKey, tuple[float, ...]] = {}

    def __setitem__(self, key: TKey, value: TValue) -> None:
        if key not in self.embeddings:
            self.embeddings[key] = self.embed_func(str(key))
        return super().__setitem__(key, value)

    def __delitem__(self, key: TKey) -> None:
        # guard in case key is not present in embeddings for some reason
        if key in self.embeddings:
            del self.embeddings[key]
        return super().__delitem__(key)

    @classmethod
    def distance(cls, emb1: tuple[float, ...], emb2: tuple[float, ...]) -> float:
        if len(emb1) != len(emb2):
            raise ValueError("Embeddings must be of the same length")
        return sum((a - b) ** 2 for a, b in zip(emb1, emb2)) ** 0.5

    def __contains__(self, key: TKey) -> bool:
        if super().__contains__(key):  # exact match
            return True
        key_embedding = self.embed_func(str(key))
        for embedding in self.embeddings.values():
            if self.distance(embedding, key_embedding) < self.min_distance:
                return True
        return False

    def find_key(self, key: TKey):
        """
        Return the stored key that matches `key` exactly or approximately,
        or None if no match exists.
        """
        if super().__contains__(key):
            return key
        key_embedding = self.embed_func(str(key))
        best = None
        best_dist = None
        for existing_key, embedding in self.embeddings.items():
            dist = self.distance(embedding, key_embedding)
            if dist < self.min_distance:
                if best is None or dist < best_dist:
                    best, best_dist = existing_key, dist
        return best

    def __getitem__(self, key: TKey) -> TValue:
        if super().__contains__(key):  # exact match
            return super().__getitem__(key)
        key_embedding = self.embed_func(str(key))
        results = []
        for existing_key, embedding in self.embeddings.items():
            dist = self.distance(embedding, key_embedding)
            if dist < self.min_distance:
                results.append((existing_key, dist))
        if not results:
            raise KeyError(f"No approximate match found for key: {key}")
        # Return the value for the first approximate match found sorted by distance
        results.sort(key=lambda k: k[1])
        return super().__getitem__(results[0][0])



class FuzzyLruCache:
    cache: FuzzyDict[str, str]

    def __init__(self, embed_func: GetEmbeddingFunc, capacity: int = 128, min_distance: float = 0.01):
        self.capacity = capacity
        self.cache = FuzzyDict(min_distance=min_distance, embed_func=embed_func)
        # Use an OrderedDict as an O(1) LRU tracking structure:
        # keys map to None; newest keys are at the end.
        self.order = OrderedDict()

    def get(self, key):
        # find the actual stored key (exact or approximate)
        match_key = self.cache.find_key(key)
        if match_key is not None:
            # Move the matched key to the end (most recently used) in O(1)
            if match_key in self.order:
                self.order.pop(match_key)
            self.order[match_key] = None
            return self.cache[match_key]
        return None

    def put(self, key, value):
        # If the exact key already exists, we'll refresh its position.
        # If approximate matching would find an existing key we treat the
        # provided key as a new insertion (consistent with original behavior).
        if key in self.cache:  # exact match
            # refresh its position
            if key in self.order:
                self.order.pop(key)
            self.order[key] = None
        else:
            # Evict oldest if full
            if len(self.cache) >= self.capacity:
                # popitem(last=False) removes the oldest item in O(1)
                try:
                    oldest_key, _ = self.order.popitem(last=False)
                except KeyError:
                    oldest_key = None
                if oldest_key is not None:
                    # remove associated cache entry
                    if oldest_key in self.cache:
                        del self.cache[oldest_key]
            # insert new item
            self.cache[key] = value
            self.order[key] = None


def semantic_cache(embed_func: GetEmbeddingFunc,
                   min_distance: float = 0.01,
                   max_size: int = 128):
    cache = FuzzyLruCache(embed_func=embed_func,
                     capacity=max_size,
                     min_distance=min_distance)
    def inner(func):
        def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result

            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        return wrapper
    return inner


