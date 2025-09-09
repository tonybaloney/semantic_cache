from typing import OrderedDict, TypeVar, Callable

try:
    from annoy import AnnoyIndex
except ImportError:
    AnnoyIndex = None

__all__ = ["semantic_cache", "FuzzyDict", "FuzzyLruCache"]

TKey = TypeVar("TKey")
TValue = TypeVar("TValue")
GetEmbeddingFunc = Callable[[str], tuple[float, ...]]


class FuzzyDict(dict[TKey, TValue]):
    """
    A dictionary that supports both exact and approximate key matching using embeddings.

    Uses Annoy (Approximate Nearest Neighbors) for fast similarity search when available,
    with fallback to linear scan for smaller datasets or when Annoy is not installed.

    Time Complexity Summary:
    - m = number of items currently stored
    - d = embedding dimension
    - E = cost of computing embedding (typically O(d))

    With Annoy (large datasets):
    - Insert: O(E) amortized
    - Delete: O(1)
    - Lookup: O(E + log m) on average
    - Contains: O(E + log m) on average

    Without Annoy (small datasets or fallback):
    - Insert: O(E) amortized
    - Delete: O(1)
    - Lookup: O(E + m·d) worst case
    - Contains: O(E + m·d) worst case
    """

    embeddings: dict[TKey, tuple[float, ...]]

    def __init__(
        self, max_distance: float, embed_func: GetEmbeddingFunc, *args, **kwargs
    ):
        """
        Initialize FuzzyDict with embedding function and similarity threshold.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            max_distance: Minimum cosine similarity for keys to be considered similar
                         (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)
            embed_func: Function that converts keys to embedding vectors
        """
        super().__init__(*args, **kwargs)
        self.max_distance = max_distance
        self.embed_func = embed_func
        self.embeddings: dict[TKey, tuple[float, ...]] = {}

        # Annoy-related fields
        self._ann_index = None  # type: AnnoyIndex | None
        self._ann_id_for_key: dict[TKey, int] = {}
        self._key_for_ann_id: dict[int, TKey] = {}
        self._next_ann_id = 0
        self._ann_index_dirty = False

    def _build_ann_index(self, dim: int):
        """
        Build or rebuild the Annoy index from current embeddings.

        Time Complexity: O(m·d·log m) where m=items, d=dimensions
        Space Complexity: O(m·d) for the index

        This is called lazily when needed for approximate searches.
        """
        if AnnoyIndex is None:
            return

        idx = AnnoyIndex(dim, "euclidean")
        for key, emb in self.embeddings.items():
            ann_id = self._ann_id_for_key.get(key)
            if ann_id is None:
                ann_id = self._next_ann_id
                self._next_ann_id += 1
                self._ann_id_for_key[key] = ann_id
                self._key_for_ann_id[ann_id] = key
            idx.add_item(ann_id, list(emb))

        if len(self.embeddings) > 0:
            idx.build(10)  # 10 trees for good balance
            self._ann_index = idx
        else:
            self._ann_index = None
        self._ann_index_dirty = False

    def __setitem__(self, key: TKey, value: TValue) -> None:
        """
        Set item in dictionary, computing embedding if needed.

        Time Complexity:
        - O(E) where E = cost of embed_func (typically O(d))
        - Amortized O(1) if embedding already computed

        Space Complexity: O(d) for storing the embedding
        """
        if key not in self.embeddings:
            emb = self.embed_func(str(key))
            self.embeddings[key] = emb
            if AnnoyIndex is not None:
                # lazily initialize ids; mark dirty so we rebuild before queries
                if key not in self._ann_id_for_key:
                    self._ann_id_for_key[key] = self._next_ann_id
                    self._key_for_ann_id[self._next_ann_id] = key
                    self._next_ann_id += 1
                self._ann_index_dirty = True
        return super().__setitem__(key, value)

    def __delitem__(self, key: TKey) -> None:
        """
        Delete item from dictionary and its embedding.

        Time Complexity: O(1) average case
        Space Complexity: O(1)

        Note: Marks Annoy index as dirty for rebuild on next search.
        """
        # guard in case key is not present in embeddings for some reason
        if key in self.embeddings:
            del self.embeddings[key]
            if key in self._ann_id_for_key:
                ann_id = self._ann_id_for_key.pop(key)
                self._key_for_ann_id.pop(ann_id, None)
                self._ann_index_dirty = True
        return super().__delitem__(key)

    @classmethod
    def distance(cls, emb1: tuple[float, ...], emb2: tuple[float, ...]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Returns cosine similarity ranging from -1 to 1:
        - 1.0: identical vectors (exact match)
        - 0.0: orthogonal vectors (no similarity)
        - -1.0: opposite vectors (maximum dissimilarity)

        Time Complexity: O(d) where d = embedding dimension
        Space Complexity: O(1)
        """
        if len(emb1) != len(emb2):
            raise ValueError("Embeddings must be of the same length")
        
        # Compute dot product
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        
        # Compute magnitudes
        magnitude1 = sum(a * a for a in emb1) ** 0.5
        magnitude2 = sum(b * b for b in emb2) ** 0.5
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

    def __contains__(self, key: TKey) -> bool:
        """
        Check if key exists exactly or approximately in the dictionary.

        Time Complexity:
        - O(1) for exact matches
        - With Annoy: O(E + log m) average case for approximate matches
        - Without Annoy: O(E + m·d) worst case for approximate matches
        where E = embed_func cost, m = items, d = embedding dimension

        Space Complexity: O(d) for temporary embedding
        """
        if super().__contains__(key):  # exact match
            return True
        key_embedding = self.embed_func(str(key))
        for embedding in self.embeddings.values():
            if self.distance(embedding, key_embedding) >= self.max_distance:
                return True
        return False

    def find_key(self, key: TKey):
        """
        Return the stored key that matches `key` exactly or approximately,
        or None if no match exists.

        Time Complexity:
        - O(1) for exact matches
        - With Annoy: O(E + log m) average case for approximate matches
        - Without Annoy: O(E + m·d) worst case for approximate matches

        Space Complexity: O(d) for temporary embedding

        This is the core method used by the cache for similarity matching.
        """
        if super().__contains__(key):
            return key
        key_embedding = self.embed_func(str(key))

        # Try ANN search first if available
        if AnnoyIndex is not None and self.embeddings:
            # get dimension from first embedding
            dim = len(next(iter(self.embeddings.values())))
            if self._ann_index is None or self._ann_index_dirty:
                self._build_ann_index(dim)
            if self._ann_index is not None:
                # request top-k neighbors (k small like 10)
                try:
                    ids, dists = self._ann_index.get_nns_by_vector(
                        list(key_embedding), 10, include_distances=True
                    )
                    best = None
                    best_dist = None
                    for ann_id, dist in zip(ids, dists):
                        candidate_key = self._key_for_ann_id.get(ann_id)
                        if candidate_key is None:
                            continue
                        # Annoy distances are approximate; compute exact similarity
                        candidate_emb = self.embeddings[candidate_key]
                        exact = self.distance(candidate_emb, key_embedding)
                        if exact >= self.max_distance:
                            if best_dist is None or exact > best_dist:
                                best, best_dist = candidate_key, exact
                    return best
                except Exception:
                    # fallback to linear scan if ANN fails
                    pass

        # Fallback: linear scan
        best = None
        best_dist = None
        for existing_key, embedding in self.embeddings.items():
            dist = self.distance(embedding, key_embedding)
            if dist >= self.max_distance:
                if best_dist is None or dist > best_dist:
                    best, best_dist = existing_key, dist
        return best

    def __getitem__(self, key: TKey) -> TValue:
        """
        Get item by exact or approximate key match.

        Time Complexity:
        - O(1) for exact matches
        - O(E + m·d + m log m) worst case for approximate matches
        where E = embed_func cost, m = items, d = embedding dimension

        The m log m term comes from sorting results by distance.
        Could be optimized to O(E + m·d) by finding minimum in single pass.

        Space Complexity: O(m) in worst case (if all items match)

        Raises KeyError if no match found within max_distance.
        """
        if super().__contains__(key):  # exact match
            return super().__getitem__(key)
        key_embedding = self.embed_func(str(key))
        results = []
        for existing_key, embedding in self.embeddings.items():
            dist = self.distance(embedding, key_embedding)
            if dist >= self.max_distance:
                results.append((existing_key, dist))
        if not results:
            raise KeyError(f"No approximate match found for key: {key}")
        # Return the value for the best approximate match (highest cosine similarity)
        results.sort(key=lambda k: k[1], reverse=True)
        return super().__getitem__(results[0][0])


class FuzzyLruCache:
    """
    LRU cache with fuzzy key matching using embeddings.

    Combines FuzzyDict for approximate matching with LRU eviction policy.
    Uses OrderedDict for O(1) LRU operations.

    Time Complexity:
    - get(): O(E + log m) with Annoy, O(E + m·d) without (plus O(1) LRU update)
    - put(): O(E + log m) with Annoy, O(E + m·d) without (plus O(1) LRU ops)
    where E = embed_func cost, m = capacity, d = embedding dimension

    Space Complexity: O(capacity · d) for embeddings + O(capacity) for LRU order
    """

    cache: FuzzyDict[str, str]

    def __init__(
        self,
        embed_func: GetEmbeddingFunc,
        capacity: int = 128,
        max_distance: float = 0.8,
    ):
        self.capacity = capacity
        self.cache = FuzzyDict(max_distance=max_distance, embed_func=embed_func)
        # Use an OrderedDict as an O(1) LRU tracking structure:
        # keys map to None; newest keys are at the end.
        self.order = OrderedDict()

    def get(self, key):
        """
        Get value for key (exact or approximate match).

        Time Complexity: O(E + log m) with Annoy, O(E + m·d) without
        Space Complexity: O(d) for temporary embedding

        Updates LRU order if match found.
        """
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
        """
        Put key-value pair in cache with LRU eviction.

        Time Complexity: O(E + log m) with Annoy, O(E + m·d) without
        Space Complexity: O(d) for new embedding

        If exact key exists, refreshes its position.
        If cache is full, evicts least recently used item.
        """
        # If the exact key already exists, we'll refresh its position.
        # If approximate matching would find an existing key we treat the
        # provided key as a new insertion (consistent with original behavior).
        if key in self.cache:  # exact match
            # update the value and refresh its position
            self.cache[key] = value
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
            self.order[key] = None


def semantic_cache(
    embed_func: GetEmbeddingFunc, max_distance: float = 0.8, max_size: int = 128
):
    """
    Decorator that caches function results with semantic key matching.

    Uses fuzzy matching on function arguments converted to embeddings,
    allowing cache hits for "similar" argument combinations.

    Time Complexity per function call:
    - Cache hit: O(E + log m) with Annoy, O(E + m·d) without
    - Cache miss: O(E + m·d) + function execution time
    where E = embed_func cost, m = cache size, d = embedding dimension

    Space Complexity: O(max_size · d) for cached embeddings

    Args:
        embed_func: Function that converts arguments to embedding vectors
        max_distance: Minimum cosine similarity for arguments to be considered similar
                     (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)
        max_size: Maximum number of cached results (LRU eviction)

    Returns:
        Decorator function that caches results with semantic matching

    Example:
        @semantic_cache(
            embed_func=lambda x: hash_to_vector(str(x)),
            max_distance=0.8,
            max_size=1000
        )
        def expensive_function(query: str) -> str:
            # ... expensive computation ...
            return result
    """
    cache = FuzzyLruCache(
        embed_func=embed_func, capacity=max_size, max_distance=max_distance
    )

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
