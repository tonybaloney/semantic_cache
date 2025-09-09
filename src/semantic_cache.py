from typing import TypeVar, Callable

__all__ = ["semantic_cache", "FuzzyDict", "FuzzyLruCache"]

TKey = TypeVar('TKey')
TValue = TypeVar('TValue')
GetEmbeddingFunc = Callable[[str], tuple[float, ...]]


class FuzzyDict(dict[TKey, TValue]):
    embeddings: dict[TKey, tuple[float, ...]] = {}

    def __init__(self, min_distance: float, embed_func: GetEmbeddingFunc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_distance = min_distance
        self.embed_func = embed_func

    def __setitem__(self, key: TKey, value: TValue) -> None:
        if key not in self.embeddings:
            self.embeddings[key] = self.embed_func(str(key))
        return super().__setitem__(key, value)

    def __delitem__(self, key: TKey) -> None:
        del self.embeddings[key]
        return super().__delitem__(key)

    @classmethod
    def distance(cls, emb1: tuple[float, ...], emb2: tuple[float, ...]) -> float:
        if len(emb1) != len(emb2):
            raise ValueError("Embeddings must be of the same length")
        return sum((a - b) ** 2 for a, b in zip(emb1, emb2)) ** 0.5

    def __contains__(self, key: TKey) -> bool:
        if super().__contains__(key): # exact match
            return True
        key_embedding = self.embed_func(str(key))
        for embedding in self.embeddings.values():
            if self.distance(embedding, key_embedding) <= self.min_distance:
                return True
        return False

    def __getitem__(self, key: TKey) -> TValue:
        if super().__contains__(key): # exact match
            return super().__getitem__(key)
        key_embedding = self.embed_func(str(key))
        results = []
        for existing_key, embedding in self.embeddings.items():
            if distance := self.distance(embedding, key_embedding) <= self.min_distance:
                results.append((existing_key, distance))
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
        self.order = []

    def get(self, key):
        if key in self.cache: # exact match and approx match
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache: # exact match, pop to end
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest_key = self.order.pop(0)
            del self.cache[oldest_key]
        self.cache[key] = value # insert new item
        self.order.append(key)


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


