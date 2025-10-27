from collections import defaultdict
import json
import sys
import os

from tqdm import tqdm

from BERT.BERT_embedding_cache import get_bert_embeddings_cached
from resampler.sentence_splitter import (
    split_into_paragraphs_safe,
    string_to_sentences,
)


# Add parent directory to path to allow imports when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state

# from resampler.BERT_core_cached import get_bert_embeddings_batch_cached
from pkld import pkld

# from resampler.sentence_splitter import split_into_paragraphs_safe, string_to_sentences


class WeightedKMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(X))

        # Initialize centers randomly
        rng = check_random_state(self.random_state)
        idx = rng.choice(len(X), self.n_clusters, replace=False)
        self.cluster_centers_ = X[idx].copy()

        for _ in tqdm(
            range(self.max_iter),
            desc="Weighted KMeans itr",
            total=self.max_iter,
        ):
            # Assign points to nearest center
            distances = euclidean_distances(X, self.cluster_centers_)
            self.labels_ = np.argmin(distances, axis=1)

            # Update centers with weighted mean
            new_centers = np.zeros_like(self.cluster_centers_)
            for k in range(self.n_clusters):
                mask = self.labels_ == k
                if np.any(mask):
                    weights_k = sample_weight[mask]
                    new_centers[k] = np.average(
                        X[mask], weights=weights_k, axis=0
                    )
                else:
                    new_centers[k] = self.cluster_centers_[k]

            # Check convergence
            if np.allclose(new_centers, self.cluster_centers_, atol=self.tol):
                break

            self.cluster_centers_ = new_centers

        return self

    def predict(self, X):
        distances = euclidean_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)


@pkld(store="both")
def get_sentence_kmeans_data(
    chunks_check=1, n_clusters=200, do_paragraphs=False
):
    """Get kmeans cluster centers and labels - returns just the data, not the model object."""
    print("Getting sentence kmeans data")
    yes_no = ["yes", "no"]
    sentence2cnt = defaultdict(int)
    bad_no_think = 0
    for yn in yes_no:
        dir_in = f"blackmail_rollouts/qwq-32b/temperature_0.7_top_p_0.95/{yn}_base_solution"
        scenario_dirs = os.listdir(dir_in)
        for scenario_dir in scenario_dirs:
            scenario_dir = os.path.join(dir_in, scenario_dir)
            for chunk_idx in range(chunks_check):
                fp_in = f"{scenario_dir}/chunk_{chunk_idx}/solutions.json"
                with open(fp_in, "r") as f:
                    solutions = json.load(f)
                for solution in solutions:
                    rollout = solution["rollout"]
                    if "</think>" not in rollout:
                        bad_no_think += 1
                        continue
                    rollout = rollout.split("</think>")[0]
                    if do_paragraphs:
                        paragraphs, paragraph_positions = (
                            split_into_paragraphs_safe(rollout, allow_0=True)
                        )
                        sentences, _ = paragraphs, paragraph_positions
                    else:
                        sentences, _ = string_to_sentences(rollout)
                    for sentence in sentences:
                        sentence2cnt[sentence] += 1

    print(f"Bad no think in Uzay data: {bad_no_think}")
    num_unique = len(sentence2cnt)
    total_cnt = sum(sentence2cnt.values())
    print(f"Number of unique sentences: {num_unique} (total cnt: {total_cnt})")

    sentences_all = sorted(list(sentence2cnt))
    embeddings = get_bert_embeddings_batch_cached(
        sentences_all,
        batch_size=512,
        # batch_size=64,
    )
    print(f"Embeddings shape: {embeddings.shape}")
    print("Computing weighted kmeans...")

    kmeans = WeightedKMeans(n_clusters=n_clusters, random_state=42)
    weights = np.array([sentence2cnt[s] for s in sentences_all])

    # Otherwise, for uniform weights:
    kmeans.fit(embeddings, sample_weight=weights)

    # Return just the data needed for prediction
    return {
        "cluster_centers": kmeans.cluster_centers_,
        "n_clusters": n_clusters,
        "labels": kmeans.labels_,
    }


def get_sentence_kmeans(chunks_check=1, n_clusters=200, do_paragraphs=False):
    """Get a kmeans model by reconstructing it from cached data."""
    kmeans_data = get_sentence_kmeans_data(
        chunks_check=chunks_check,
        n_clusters=n_clusters,
        do_paragraphs=do_paragraphs,
    )

    # Reconstruct the model
    kmeans = WeightedKMeans(n_clusters=n_clusters, random_state=42)
    kmeans.cluster_centers_ = kmeans_data["cluster_centers"]
    kmeans.labels_ = kmeans_data["labels"]

    return kmeans


def get_sentence_kmeans_from_base(
    base_responses, n_clusters=50, do_paragraphs=True
):
    kmeans_data = get_sentence_kmeans_data_from_base(
        base_responses, n_clusters=n_clusters, do_paragraphs=do_paragraphs
    )
    kmeans = WeightedKMeans(n_clusters=n_clusters, random_state=42)
    kmeans.cluster_centers_ = kmeans_data["cluster_centers"]
    kmeans.labels_ = kmeans_data["labels"]
    return kmeans


@pkld(overwrite=False)
def get_sentence_kmeans_data_from_base(
    base_responses, n_clusters=50, do_paragraphs=True
):
    sentence2cnt = defaultdict(int)
    for response in base_responses:
        reasoning = response["reasoning"]
        assert isinstance(reasoning, str)
        assert reasoning != ""
        if do_paragraphs:
            paragraphs, _ = split_into_paragraphs_safe(reasoning, allow_0=True)
        else:
            paragraphs, _ = string_to_sentences(reasoning)
        for paragraph in paragraphs:
            sentence2cnt[paragraph] += 1

    num_unique = len(sentence2cnt)
    total_cnt = sum(sentence2cnt.values())
    print(f"Number of unique sentences: {num_unique} (total cnt: {total_cnt})")

    sentences_all = sorted(list(sentence2cnt))
    embeddings = get_bert_embeddings_batch_cached(
        sentences_all,
        batch_size=512,
        # batch_size=64,
    )
    print(f"Embeddings shape: {embeddings.shape}")
    print("Computing weighted kmeans...")

    kmeans = WeightedKMeans(n_clusters=n_clusters, random_state=42)
    weights = np.array([sentence2cnt[s] for s in sentences_all])

    # Otherwise, for uniform weights:
    kmeans.fit(embeddings, sample_weight=weights)

    # Return just the data needed for prediction
    return {
        "cluster_centers": kmeans.cluster_centers_,
        "n_clusters": n_clusters,
        "labels": kmeans.labels_,
    }


@pkld(overwrite=False)
def get_all_sentences_from_new_responses(
    sentence2new_responses, do_paragraphs=False
):
    sentence2cnt = defaultdict(int)
    for sentence, new_responses in tqdm(
        sentence2new_responses.items(),
        total=len(sentence2new_responses),
        desc="Gathering all sentences for kmeans",
    ):
        # new_responses is a list of response dicts
        for response in new_responses:
            if isinstance(response, dict) and "reasoning" in response:
                reasoning = response["reasoning"]
            else:
                # If it's not a dict, skip it
                continue
            if do_paragraphs:
                new_sentences, _ = split_into_paragraphs_safe(
                    reasoning, allow_0=True
                )
            else:
                new_sentences, _ = string_to_sentences(reasoning)
            for new_sentence in new_sentences:
                sentence2cnt[new_sentence] += 1
    return sentence2cnt


# @pkld(overwrite=False)
def get_cluster_mappers(
    sentence2data,
    n_clusters=200,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
):

    sentences_all = sorted(list(sentence2data))

    print(f"Getting embeddings {len(sentences_all)}")
    embeddings = get_bert_embeddings_cached(
        sentences_all, model_name=embedding_model
    )
    weights = np.array([sentence2data[s].count for s in sentences_all])
    kmeans = WeightedKMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings, sample_weight=weights)

    nums = predict_clusters(kmeans, embeddings)
    print("Organizing labels")
    num2highest_cnt = {}
    num2most_sentence = {}
    for sentence, num in zip(sentences_all, nums):
        cnt = sentence2data[sentence].count
        if num not in num2highest_cnt:
            num2highest_cnt[num] = cnt
            num2most_sentence[num] = sentence
        else:
            if cnt > num2highest_cnt[num]:
                num2highest_cnt[num] = cnt
                num2most_sentence[num] = sentence

    all2most = {}
    for sentence, num in zip(sentences_all, nums):
        assert num in num2highest_cnt
        most = num2most_sentence[num]
        all2most[sentence] = most
    print(f"Got all2most: {len(all2most)}")
    return all2most


def get_all2most_cluster2commons(all2cluster, sentence2cnt_base):
    print("Getting most2common")
    cluster2sentences = {}

    # Group sentences by their "most" value
    for sentence, cluster_idx in tqdm(
        all2cluster.items(), total=len(all2cluster), desc="Getting most2common"
    ):
        if cluster_idx not in cluster2sentences:
            cluster2sentences[cluster_idx] = []
        cluster2sentences[cluster_idx].append(sentence)

    # Sort sentences for each "most" by count (descending)
    cluster2commons = {}
    for cluster_idx, sentences in cluster2sentences.items():
        # Sort by count in descending order (highest first)
        sorted_sentences = sorted(
            sentences, key=lambda s: sentence2cnt_base[s], reverse=True
        )
        cluster2commons[cluster_idx] = sorted_sentences

    all2most = {}
    most2commons = {}
    for sentence, cluster_idx in all2cluster.items():
        all2most[sentence] = cluster2commons[cluster_idx][0]
        most2commons[sentence] = cluster2commons[cluster_idx]

    print(f"Got all2most: {len(all2most)}")
    print(f"Got most2commons: {len(most2commons)}")

    return all2most, most2commons


def predict_clusters(kmeans, embeddings, return_distances=False):
    """
    Predict cluster assignments for new embeddings using a trained WeightedKMeans model.

    Parameters:
    -----------
    kmeans : WeightedKMeans
        Trained WeightedKMeans model with cluster_centers_
    embeddings : np.ndarray
        Single embedding (1D array) or batch of embeddings (2D array)
    return_distances : bool
        If True, also return distances to assigned clusters

    Returns:
    --------
    labels : np.ndarray
        Cluster assignments (single int or array of ints)
    distances : np.ndarray (optional)
        Distances to assigned clusters
    """
    # Ensure embeddings is 2D
    embeddings = np.atleast_2d(embeddings)

    # Calculate distances to all cluster centers
    distances_matrix = euclidean_distances(embeddings, kmeans.cluster_centers_)

    # Find nearest cluster for each embedding
    labels = np.argmin(distances_matrix, axis=1)

    # Get distances to assigned clusters
    min_distances = np.min(distances_matrix, axis=1)

    # If single embedding was passed, return scalar
    if len(embeddings) == 1:
        labels = labels[0]
        min_distances = min_distances[0]

    if return_distances:
        return labels, min_distances
    return labels
