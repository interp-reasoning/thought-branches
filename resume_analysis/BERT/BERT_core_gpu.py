#!/usr/bin/env python3
"""
GPU-enabled BERT embeddings with automatic caching using pkld decorator.
Uses the last hidden layer and [CLS] token for sentence embeddings.
"""

from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Tuple
from pkld import pkld
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache to avoid reloading
_model_cache = {}
_tokenizer_cache = {}

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def _get_model_and_tokenizer(model_name: str = "bert-base-uncased"):
    """Get or load model and tokenizer with caching."""
    if model_name not in _model_cache:
        logger.info(f"Loading BERT model: {model_name}")
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
        _model_cache[model_name] = AutoModel.from_pretrained(model_name)
        # IMPORTANT: Move model to GPU
        _model_cache[model_name] = _model_cache[model_name].to(device)
        # Set to eval mode
        _model_cache[model_name].eval()
        logger.info(f"Model loaded on {device}")

    return _model_cache[model_name], _tokenizer_cache[model_name]


def get_bert_embedding(
    text: str,
    model_name: str = "bert-base-uncased",
    pooling_strategy: str = "cls",
    layers: Union[int, List[int]] = -1,
) -> np.ndarray:
    """
    Get BERT embedding for a text string with automatic caching.

    Args:
        text: The text to embed
        model_name: BERT model to use (default: bert-base-uncased)
        pooling_strategy: How to pool token embeddings
            - "cls": Use [CLS] token embedding (default)
            - "mean": Mean pooling across all tokens
            - "max": Max pooling across all tokens
        layers: Which layer(s) to use
            - -1: Last hidden layer (default)
            - int: Specific layer index
            - List[int]: Average across multiple layers

    Returns:
        Numpy array of the embedding
    """
    # Get model and tokenizer
    model, tokenizer = _get_model_and_tokenizer(model_name)

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # IMPORTANT: Move inputs to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract hidden states
    hidden_states = (
        outputs.hidden_states
    )  # Tuple of (num_layers + 1) x (batch, seq_len, hidden_dim)

    # Select layer(s)
    if isinstance(layers, int):
        if layers == -1:
            # Last layer
            embeddings = hidden_states[-1]
        else:
            embeddings = hidden_states[layers]
    else:
        # Average across multiple layers
        selected_layers = [hidden_states[i] for i in layers]
        embeddings = torch.stack(selected_layers).mean(dim=0)

    # Apply pooling strategy
    if pooling_strategy == "cls":
        # Use [CLS] token (first token)
        pooled = embeddings[:, 0, :].squeeze()
    elif pooling_strategy == "mean":
        # Mean pooling (excluding padding)
        attention_mask = inputs["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = (sum_embeddings / sum_mask).squeeze()
    elif pooling_strategy == "max":
        # Max pooling
        pooled, _ = torch.max(embeddings, dim=1)
        pooled = pooled.squeeze()
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

    # Convert to numpy (move to CPU first)
    return pooled.cpu().numpy()


def get_bert_embeddings_batch(
    texts: List[str],
    model_name: str = "bert-base-uncased",
    pooling_strategy: str = "cls",
    layers: Union[int, List[int]] = -1,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Get BERT embeddings for multiple texts with automatic caching.

    Args:
        texts: List of texts to embed
        model_name: BERT model to use
        pooling_strategy: How to pool token embeddings
        layers: Which layer(s) to use
        batch_size: Process texts in batches of this size

    Returns:
        Numpy array of shape (n_texts, embedding_dim)
    """
    # Get model and tokenizer
    model, tokenizer = _get_model_and_tokenizer(model_name)

    all_embeddings = []

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Getting BERT embeddings"):
        batch_texts = texts[i : i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        )

        # IMPORTANT: Move inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract hidden states
        hidden_states = outputs.hidden_states

        # Select layer(s)
        if isinstance(layers, int):
            if layers == -1:
                embeddings = hidden_states[-1]
            else:
                embeddings = hidden_states[layers]
        else:
            selected_layers = [hidden_states[i] for i in layers]
            embeddings = torch.stack(selected_layers).mean(dim=0)

        # Apply pooling strategy
        if pooling_strategy == "cls":
            pooled = embeddings[:, 0, :]
        elif pooling_strategy == "mean":
            attention_mask = inputs["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        elif pooling_strategy == "max":
            pooled, _ = torch.max(embeddings, dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

        # Move to CPU before converting to numpy
        all_embeddings.append(pooled.cpu().numpy())

    # Concatenate all batches
    return np.vstack(all_embeddings)


def get_bert_token_embeddings(
    text: str, model_name: str = "bert-base-uncased", layers: Union[int, List[int]] = -1
) -> Tuple[List[str], np.ndarray]:
    """
    Get token-level BERT embeddings with automatic caching.

    Args:
        text: The text to embed
        model_name: BERT model to use
        layers: Which layer(s) to use

    Returns:
        Tuple of (tokens, embeddings)
        - tokens: List of token strings
        - embeddings: Array of shape (n_tokens, embedding_dim)
    """
    # Get model and tokenizer
    model, tokenizer = _get_model_and_tokenizer(model_name)

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # IMPORTANT: Move inputs to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract hidden states
    hidden_states = outputs.hidden_states

    # Select layer(s)
    if isinstance(layers, int):
        if layers == -1:
            embeddings = hidden_states[-1]
        else:
            embeddings = hidden_states[layers]
    else:
        selected_layers = [hidden_states[i] for i in layers]
        embeddings = torch.stack(selected_layers).mean(dim=0)

    # Get tokens
    token_ids = inputs["input_ids"][0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Remove padding
    attention_mask = inputs["attention_mask"][0].cpu()
    valid_length = attention_mask.sum().item()
    tokens = tokens[:valid_length]
    embeddings = embeddings[0, :valid_length, :]

    return tokens, embeddings.cpu().numpy()



# Example usage
if __name__ == "__main__":
    import time

    print("GPU-Enabled BERT Embeddings Demo\n")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # Test 1: Basic embedding
    print("1. Testing basic embedding generation...")
    text = "Machine learning is a subset of artificial intelligence."

    start = time.time()
    embedding = get_bert_embedding(text)
    time1 = time.time() - start

    print(f"   Text: '{text}'")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Time: {time1:.3f}s")

    # Test cached call
    start = time.time()
    embedding2 = get_bert_embedding(text)
    time2 = time.time() - start
    print(f"   Cached call time: {time2:.3f}s (speedup: {time1/time2:.1f}x)")

    # Test GPU speedup with batch processing
    print("\n2. Testing GPU speedup with batch processing...")
    test_texts = [f"This is test sentence number {i}." for i in range(100)]

    start = time.time()
    embeddings = get_bert_embeddings_batch(test_texts, batch_size=32)
    gpu_time = time.time() - start

    print(f"   Processed {len(test_texts)} texts in {gpu_time:.3f}s")
    print(f"   Average time per text: {gpu_time/len(test_texts)*1000:.1f}ms")

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n   GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
