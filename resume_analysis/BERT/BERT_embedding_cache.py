#!/usr/bin/env python3
"""
Efficient disk-based caching for BERT embeddings using HDF5.

HDF5 (Hierarchical Data Format) is like a file system within a file:
- Can store multiple datasets in one file
- Supports partial loading (only load what you need)
- Very efficient for large numpy arrays
- Built-in compression support
"""

import h5py
import numpy as np
import hashlib
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import time
from tqdm import tqdm


class BertEmbeddingCache:
    """
    Efficient storage-based cache for BERT embeddings using HDF5.

    Features:
    - Order-independent caching (can retrieve subsets efficiently)
    - Handles 1-10M embeddings easily
    - Simple API
    - Persistent across program runs
    - Automatic model-specific caching with dimension detection
    """

    # Model dimension mapping
    MODEL_DIMENSIONS = {
        "bert-base-uncased": 768,
        "bert-base-cased": 768,
        "bert-large-uncased": 1024,
        "bert-large-cased": 1024,
        "roberta-base": 768,
        "roberta-large": 1024,
        "distilbert-base-uncased": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-MiniLM-L12-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
        "sentence-transformers/paraphrase-mpnet-base-v2": 768,
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": 384,
    }

    def __init__(
        self,
        cache_dir: str = ".cache/bert_embeddings",
        model_name: str = None,
        embedding_dim: int = None,
        compression: bool = True,
        preload_to_memory: bool = True,
    ):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cache files
            model_name: Model name to determine embedding dim and cache file
            embedding_dim: Override dimension (auto-detected if None)
            compression: Whether to compress embeddings (saves ~50% space)
            preload_to_memory: Load entire cache into memory on init (default True for speed)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.compression = compression
        self.preload_to_memory = preload_to_memory

        # Determine embedding dimension
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        elif model_name and model_name in self.MODEL_DIMENSIONS:
            self.embedding_dim = self.MODEL_DIMENSIONS[model_name]
        elif model_name:
            # Try to detect from model name patterns
            if "large" in model_name.lower():
                self.embedding_dim = 1024
            elif "minilm-l6" in model_name.lower():
                self.embedding_dim = 384
            elif "minilm-l12" in model_name.lower():
                self.embedding_dim = 384
            else:
                self.embedding_dim = 768  # Default to base size
            print(
                f"Auto-detected embedding dimension {self.embedding_dim} for {model_name}"
            )
        else:
            self.embedding_dim = 768  # Default

        # Create model-specific cache file name
        if model_name:
            safe_model_name = model_name.replace("/", "_").replace("-", "_")
            self.cache_file = (
                self.cache_dir
                / f"embeddings_{safe_model_name}_{self.embedding_dim}d.h5"
            )
            self.index_file = (
                self.cache_dir
                / f"index_{safe_model_name}_{self.embedding_dim}d.json"
            )
        else:
            self.cache_file = (
                self.cache_dir / f"embeddings_{self.embedding_dim}d.h5"
            )
            self.index_file = (
                self.cache_dir / f"index_{self.embedding_dim}d.json"
            )

        self.index = self._load_index()

        # Initialize HDF5 file if needed
        self._init_h5_file()

        # Memory cache for entire dataset
        self._memory_cache = None
        if preload_to_memory and self.cache_file.exists():
            self._preload_cache()

    def _init_h5_file(self):
        """Initialize HDF5 file structure if it doesn't exist."""
        if not self.cache_file.exists():
            with h5py.File(self.cache_file, "w") as f:
                # Create resizable datasets
                # maxshape=(None,) means unlimited size in that dimension
                f.create_dataset(
                    "embeddings",
                    shape=(0, self.embedding_dim),
                    maxshape=(None, self.embedding_dim),
                    dtype="float32",
                    compression="gzip" if self.compression else None,
                    compression_opts=4 if self.compression else None,
                )

                # Store metadata as variable-length strings
                dt = h5py.special_dtype(vlen=str)
                f.create_dataset(
                    "hashes", shape=(0,), maxshape=(None,), dtype=dt
                )

                # Store sentences for debugging/verification
                f.create_dataset(
                    "sentences", shape=(0,), maxshape=(None,), dtype=dt
                )

    def _load_index(self) -> Dict[str, int]:
        """Load the hash->position index from disk."""
        if self.index_file.exists():
            # print(f'{self.index_file=}')
            with open(self.index_file, "r") as f:
                return json.load(f)
        return {}

    def _preload_cache(self):
        """Preload entire cache into memory for faster access."""
        print(f"Preloading cache into memory...")
        start_time = time.time()

        with h5py.File(self.cache_file, "r") as f:
            self._memory_cache = f["embeddings"][:]

        load_time = time.time() - start_time
        size_mb = self._memory_cache.nbytes / (1024 * 1024)
        print(
            f"Preloaded {self._memory_cache.shape[0]} embeddings ({size_mb:.1f}MB) in {load_time:.1f}s"
        )

    def _save_index(self):
        """Save the hash->position index to disk."""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f)

    def _generate_hash(
        self, sentence: str, model_name: str, pooling: str
    ) -> str:
        """Generate a unique hash for a sentence+model+pooling combination."""
        key = f"{sentence}|{model_name}|{pooling}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def get_many(
        self, sentences: List[str], model_name: str, pooling: str
    ) -> Tuple[List[bool], np.ndarray]:
        """
        Retrieve embeddings for multiple sentences.

        Args:
            sentences: List of sentences
            model_name: BERT model name
            pooling: Pooling strategy

        Returns:
            (cached_mask, embeddings) where:
            - cached_mask[i] = True if sentences[i] was in cache
            - embeddings[i] = embedding if cached, zeros if not cached
        """
        n = len(sentences)
        cached_mask = [False] * n
        embeddings = np.zeros((n, self.embedding_dim), dtype=np.float32)

        t_st = time.time()
        print(f"Starting to generate hases: {len(sentences)=}")
        # Generate hashes
        hashes = [
            self._generate_hash(s, model_name, pooling)
            for s in tqdm(sentences, desc="Generating hashes")
        ]
        print(f"Hashed {len(sentences)} sentences in {time.time() - t_st:.1f}s")

        # Find which ones are in cache
        positions_to_load = []
        indices_to_fill = []

        for i, hash_val in enumerate(hashes):
            if hash_val in self.index:
                position = self.index[hash_val]
                positions_to_load.append(position)
                indices_to_fill.append(i)
                cached_mask[i] = True

        print(f"About to load embeddings: {time.time() - t_st:.1f}s")

        # Use memory cache if available
        if self._memory_cache is not None and positions_to_load:
            print(
                f"Using preloaded memory cache for {len(positions_to_load)} embeddings"
            )
            load_start = time.time()

            # Direct memory access is very fast
            for pos, fill_idx in zip(positions_to_load, indices_to_fill):
                embeddings[fill_idx] = self._memory_cache[pos]

            load_time = time.time() - load_start
            if load_time > 0:
                print(
                    f"Retrieved from memory in {load_time:.3f}s ({len(positions_to_load)/load_time:.0f} embeddings/sec)"
                )
            else:
                print(
                    f"Retrieved embeddings from memory instantly ({len(positions_to_load)=}; {load_time=:.5f})"
                )
        elif positions_to_load:
            print(f"{self.cache_file=   }")
            # Batch load from HDF5 (efficient!)
            with h5py.File(self.cache_file, "r") as f:
                embeddings_ds = f["embeddings"]
                total_embeddings = embeddings_ds.shape[0]

                # Determine loading strategy based on proportion of data needed
                load_ratio = len(positions_to_load) / max(total_embeddings, 1)
                print(
                    f"Loading {len(positions_to_load)}/{total_embeddings} embeddings ({load_ratio*100:.1f}%)"
                )

                # If we need more than 50% of the data, or more than 100k embeddings,
                # it's faster to load the entire dataset
                if load_ratio > 0.5 or len(positions_to_load) > 100000:
                    print(
                        f"Loading entire dataset (more efficient for {load_ratio*100:.1f}% of data)"
                    )
                    load_start = time.time()

                    # Load all embeddings into memory
                    all_embeddings = embeddings_ds[:]

                    load_time = time.time() - load_start
                    print(
                        f"Loaded entire dataset in {load_time:.1f}s ({all_embeddings.shape[0]/load_time:.0f} embeddings/sec)"
                    )

                    # Fill from the full array (much faster than HDF5 indexing)
                    for i, (pos, fill_idx) in enumerate(
                        zip(positions_to_load, indices_to_fill)
                    ):
                        embeddings[fill_idx] = all_embeddings[pos]

                else:
                    # For smaller subsets, use fancy indexing
                    print(
                        f"Using selective loading for {len(positions_to_load)} embeddings"
                    )

                    # HDF5 requires indices to be sorted for fancy indexing
                    sorted_indices = sorted(
                        range(len(positions_to_load)),
                        key=lambda i: positions_to_load[i],
                    )
                    sorted_positions = [
                        positions_to_load[i] for i in sorted_indices
                    ]

                    # Batch the loading for very large requests
                    batch_size = 10000  # Load in chunks to avoid memory issues

                    if len(sorted_positions) > batch_size:
                        print(f"Loading in batches of {batch_size}")
                        loaded_embeddings = []

                        for i in tqdm(
                            range(0, len(sorted_positions), batch_size),
                            desc="Loading batches",
                            leave=False,
                        ):
                            batch_positions = sorted_positions[
                                i : i + batch_size
                            ]
                            batch_embeddings = embeddings_ds[batch_positions]
                            loaded_embeddings.append(batch_embeddings)

                        loaded_embeddings = np.vstack(loaded_embeddings)
                    else:
                        # Small enough to load in one go
                        loaded_embeddings = embeddings_ds[sorted_positions]

                    print(f"Loaded embeddings: {loaded_embeddings.shape=}")

                    # Fill results in original order
                    for orig_idx, sorted_idx in enumerate(sorted_indices):
                        fill_idx = indices_to_fill[orig_idx]
                        embeddings[fill_idx] = loaded_embeddings[sorted_idx]
        print(f"Did all hashing and HDF5 in {time.time() - t_st:.1f}s")
        return cached_mask, embeddings

    def put_many(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        model_name: str,
        pooling: str,
    ):
        """
        Store multiple embeddings in the cache.

        Args:
            sentences: List of sentences
            embeddings: Numpy array of shape (n_sentences, embedding_dim)
            model_name: BERT model name
            pooling: Pooling strategy
        """
        if len(sentences) != len(embeddings):
            raise ValueError(
                "Number of sentences must match number of embeddings"
            )

        # Generate hashes
        hashes = [
            self._generate_hash(s, model_name, pooling) for s in sentences
        ]

        # Filter out ones already in cache
        to_add_indices = []
        to_add_hashes = []
        to_add_sentences = []

        for i, hash_val in enumerate(hashes):
            if hash_val not in self.index:
                to_add_indices.append(i)
                to_add_hashes.append(hash_val)
                to_add_sentences.append(sentences[i])

        if not to_add_indices:
            return  # Everything already cached

        # Add to HDF5
        with h5py.File(self.cache_file, "a") as f:
            embeddings_ds = f["embeddings"]
            hashes_ds = f["hashes"]
            sentences_ds = f["sentences"]

            # Current size
            current_size = embeddings_ds.shape[0]
            new_size = current_size + len(to_add_indices)

            # Resize datasets
            embeddings_ds.resize((new_size, self.embedding_dim))
            hashes_ds.resize((new_size,))
            sentences_ds.resize((new_size,))

            # Add new data
            new_embeddings = embeddings[to_add_indices]
            embeddings_ds[current_size:new_size] = new_embeddings
            hashes_ds[current_size:new_size] = to_add_hashes
            sentences_ds[current_size:new_size] = to_add_sentences

            # Update index
            for i, hash_val in enumerate(
                tqdm(to_add_hashes, desc="Updating cache index", leave=False)
            ):
                self.index[hash_val] = current_size + i

        # Save updated index
        self._save_index()

        # Update memory cache if preloaded
        if self._memory_cache is not None:
            # Expand memory cache
            print("Updating memory cache...")
            old_cache = self._memory_cache
            self._memory_cache = np.zeros(
                (new_size, self.embedding_dim), dtype=np.float32
            )
            self._memory_cache[:current_size] = old_cache
            self._memory_cache[current_size:new_size] = new_embeddings

    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        with h5py.File(self.cache_file, "r") as f:
            n_embeddings = f["embeddings"].shape[0]
            file_size_mb = self.cache_file.stat().st_size / (1024 * 1024)

        return {
            "n_embeddings": n_embeddings,
            "file_size_mb": round(file_size_mb, 2),
            "compression": self.compression,
            "embedding_dim": self.embedding_dim,
            "avg_bytes_per_embedding": round(
                file_size_mb * 1024 * 1024 / max(n_embeddings, 1), 2
            ),
        }

    def clear_cache(self):
        """Clear the entire cache."""
        if self.cache_file.exists():
            self.cache_file.unlink()
        if self.index_file.exists():
            self.index_file.unlink()
        self.index = {}
        self._init_h5_file()
        print("Cache cleared!")


def get_bert_embeddings_cached(
    sentences: List[str],
    model_name: str = "bert-base-uncased",
    pooling: str = "mean",
    batch_size: int = 128,
    cache: Optional[BertEmbeddingCache] = None,
) -> np.ndarray:
    """
    Get BERT embeddings with caching. Drop-in replacement for get_bert_embeddings_batch.

    This is the main function you'll use!

    Args:
        sentences: List of sentences to embed
        model_name: BERT model to use
        pooling: Pooling strategy
        batch_size: Batch size for computing new embeddings
        cache: Cache instance (creates default if None)

    Returns:
        Embeddings array of shape (n_sentences, embedding_dim)
    """
    # Import here to avoid circular dependency
    from BERT.BERT_core_gpu import get_bert_embeddings_batch

    # Use model-specific cache if none provided
    if cache is None:
        cache = BertEmbeddingCache(model_name=model_name)

    # Check cache
    cached_mask, embeddings = cache.get_many(sentences, model_name, pooling)

    # Find what needs to be computed
    missing_indices = [
        i for i, is_cached in enumerate(cached_mask) if not is_cached
    ]

    if missing_indices:
        # Get sentences that need computation
        missing_sentences = [sentences[i] for i in missing_indices]

        print(
            f"Computing {len(missing_sentences)} new embeddings ({len(sentences)-len(missing_sentences)} from cache)"
        )

        # Compute new embeddings
        new_embeddings = get_bert_embeddings_batch(
            missing_sentences,
            model_name=model_name,
            batch_size=batch_size,
            pooling_strategy=pooling,
        )

        # Store in cache
        cache.put_many(missing_sentences, new_embeddings, model_name, pooling)

        # Fill in the results
        for i, idx in enumerate(missing_indices):
            embeddings[idx] = new_embeddings[i]
    else:
        print(f"All {len(sentences)} embeddings retrieved from cache!")

    return embeddings


# Example usage and testing
if __name__ == "__main__":
    print("Testing BERT embedding cache with HDF5...")
    print("=" * 60)

    # Create cache instance for bert-base-uncased
    cache = BertEmbeddingCache(model_name="bert-base-uncased")

    # Test sentences
    test_sentences = [
        "The cat sat on the mat.",
        "Dogs are loyal animals.",
        "Machine learning is fascinating.",
        "The cat sat on the mat.",  # Duplicate!
        "Python is a great language.",
    ]

    print(f"Test sentences: {len(test_sentences)} (including 1 duplicate)")

    # First call - will compute all
    print("\nFirst call (all new):")
    # Mock embeddings for testing (replace with actual BERT)
    mock_embeddings = np.random.randn(5, 768).astype(np.float32)
    cache.put_many(test_sentences[:3], mock_embeddings[:3], "bert-base", "mean")

    # Second call - some cached
    print("\nChecking cache (should find 3):")
    cached_mask, retrieved = cache.get_many(test_sentences, "bert-base", "mean")
    print(f"Cached: {sum(cached_mask)} out of {len(test_sentences)}")
    print(f"Cached positions: {[i for i, m in enumerate(cached_mask) if m]}")

    # Test different order
    print("\nTesting different order:")
    reordered = test_sentences[2:] + test_sentences[:2]
    cached_mask2, _ = cache.get_many(reordered, "bert-base", "mean")
    print(f"Reordered sentences: {len(reordered)}")
    print(f"Cached: {sum(cached_mask2)} (order independence works!)")

    # Stats
    print("\nCache statistics:")
    stats = cache.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
