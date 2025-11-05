"""
Analyze rollout data and label chunks for whistleblower scenarios.

This script adapts the blackmail rollouts analysis approach to work with whistleblowing behavior detection
using importance analysis to identify high-leverage reasoning chunks.

Usage:
    python analyze_rollouts.py -iy whistleblower_rollouts/model/yes_base_solution
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv
from prompts import WHISTLEBLOWING_DAG_PROMPT
import re
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer

import math
import multiprocessing as mp
from functools import partial
import scipy.stats as stats
from matplotlib.lines import Line2D
from utils import split_solution_into_chunks, normalize_answer
    
# Class to hold arguments for importance calculation functions
class ImportanceArgs:
    """Class to hold arguments for importance calculation functions."""
    def __init__(self, use_absolute=False, forced_answer_dir=None, similarity_threshold=0.8, use_similar_chunks=True, use_abs_importance=False, top_chunks=100, use_prob_true=True, cf_fraction_threshold=0.05):
        self.use_absolute = use_absolute
        self.forced_answer_dir = forced_answer_dir
        self.similarity_threshold = similarity_threshold
        self.use_similar_chunks = use_similar_chunks
        self.use_abs_importance = use_abs_importance
        self.top_chunks = top_chunks
        self.use_prob_true = use_prob_true
        # If > 0, when different_trajectories_fraction <= this threshold, fallback CF metrics to resampling metrics
        self.cf_fraction_threshold = cf_fraction_threshold

# Set tokenizers parallelism to false to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global cache for the embedding model
embedding_model_cache = {}

IMPORTANCE_METRICS = [
    "resampling_importance_whistleblowing_rate", 
    # "forced_importance_whistleblowing_rate",
    # "resampling_importance_kl", 
    # "counterfactual_importance_kl",
    # "forced_importance_kl", 
    # "forced_importance_logodds",
    # "counterfactual_importance_logodds", 
    # "resampling_importance_logodds",
    # CF importance over whistleblowing rate
    "counterfactual_importance_whistleblowing_rate", 
    # CF++ importance over whistleblowing rate
    "counterfactual_plusplus_importance_whistleblowing_rate",
]

# TODO: counterfactual_importance_kl and counterfactual_plusplus_importance_kl

MODELS = ["hermes-4-70b", "llama-3_1-nemotron-ultra-253b-v1", "deepseek-r1-0528"]
parser = argparse.ArgumentParser(description='Analyze rollout data and label chunks for whistleblower scenarios')
parser.add_argument('-iy', '--yes_rollouts_dir', type=str, default=None, help='Override: directory containing yes (whistleblowing) rollout data. If omitted, derived per model as whistleblower_rollouts/{model}/temperature_0.7_top_p_0.95/yes_base_solution')
parser.add_argument('-in', '--no_rollouts_dir', type=str, default=None, help='Directory containing no (non-whistleblowing) rollout data')
parser.add_argument('-iyf', '--yes_forced_answer_rollouts_dir', type=str, default=None, help='Override: directory containing yes rollout data with forced answers. If omitted, derived per model as whistleblower_rollouts/{model}/temperature_0.7_top_p_0.95/yes_base_solution_forced_answer')
parser.add_argument('-inf', '--no_forced_answer_rollouts_dir', type=str, default=None, help='Directory containing no rollout data with forced answers')
parser.add_argument('-o', '--output_dir', type=str, default=None, help='Override: base directory to save analysis results. If omitted, derived per model as analysis/{model}/basic')
parser.add_argument('-p', '--problems', type=str, default=None, help='Comma-separated list of problem indices to analyze (default: all)')
parser.add_argument('-m', '--max_problems', type=int, default=None, help='Maximum number of problems to analyze')
parser.add_argument('-a', '--absolute', default=False, action='store_true', help='Use absolute value for importance calculation')
parser.add_argument('-f', '--force_relabel', default=False, action='store_true', help='Force relabeling of chunks')
parser.add_argument('-fm', '--force_metadata', default=False, action='store_true', help='Force regeneration of chunk summaries and scenario nicknames')
parser.add_argument('-mc', '--max_chunks_to_show', type=int, default=100, help='Maximum number of chunks to show in plots')
parser.add_argument('-tc', '--top_chunks', type=int, default=500, help='Number of top chunks to use for similar and dissimilar during counterfactual importance calculation')
parser.add_argument('-u', '--use_existing_metrics', default=False, action='store_true', help='Use existing metrics from chunks_labeled.json without recalculating')
parser.add_argument('-im', '--importance_metric', type=str, default="counterfactual_importance_whistleblowing_rate", choices=IMPORTANCE_METRICS, help='Which importance metric to use for plotting and analysis')
parser.add_argument('-sm', '--sentence_model', type=str, default="bert-large-nli-stsb-mean-tokens", help='Sentence transformer model to use for embeddings')
parser.add_argument('-st', '--similarity_threshold', type=float, default=0.6, help='Similarity threshold for determining different chunks')
parser.add_argument('-bs', '--batch_size', type=int, default=8192, help='Batch size for embedding model')
parser.add_argument('-us', '--use_similar_chunks', default=True, action='store_true', help='Use similar chunks for importance calculation')
parser.add_argument('-np', '--num_processes', type=int, default=min(100, mp.cpu_count()), help='Number of parallel processes for chunk processing')
parser.add_argument('-pt', '--use_prob_true', default=False, action='store_false', help='Use probability of correct answer (P(true)) instead of answer distribution for KL divergence calculations')
parser.add_argument('-M', '--models', nargs='+', type=str, default=MODELS, help='One or more model names to analyze. Default: all known models')
parser.add_argument('-cfft', '--cf_fraction_threshold', type=float, default=0.05, help='If different_trajectories_fraction <= this threshold, use resampling metrics instead of counterfactual (<=0 to disable)')
parser.add_argument('-lm', '--labeler_model', type=str, default="gpt-4o-mini", help='Model to use for labeling/summarizing chunks and scenario nicknames')
parser.add_argument('-fle', '--fix_labeling_errors_only', default=False, action='store_true', help='Only scan scenarios and repair labeling where chunks_labeled.json is missing or all function_tags are ["other"] or null')
parser.add_argument('-lbs', '--label_batch_size', type=int, default=30, help='Batch size for labeling chunks to avoid oversized JSON responses')
parser.add_argument('-lmc', '--label_max_chars', type=int, default=800, help='Max characters per chunk text sent to the labeler to control prompt size')
args = parser.parse_args()

# Set consistent font size for all plots
FONT_SIZE = 20
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 4,
    'axes.labelsize': FONT_SIZE + 2,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE - 1,
    'figure.titlesize': FONT_SIZE + 6
})

FIGSIZE = (20, 7)

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

plt.rcParams.update({
    'axes.labelpad': 20,        # Padding for axis labels
    'axes.titlepad': 20,        # Padding for plot titles
    'axes.spines.top': False,   # Hide top spine
    'axes.spines.right': False, # Hide right spine
})

# Load environment variables
load_dotenv()

# Set up OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Initialize the r1-distill-qwen-14b tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-r1-distill-qwen-14b")

def count_tokens(text: str, approximate: bool = True) -> int:
    """Count the number of tokens in a text string using the r1-distill-qwen-14b tokenizer."""
    if approximate:
        return len(text) // 4
    else:
        return len(tokenizer.encode(text))

def generate_chunk_summary(chunk_text: str) -> str:
    """
    Generate a 2-3 word summary of a chunk using OpenAI API.
    
    Args:
        chunk_text: The text content of the chunk
        
    Returns:
        A 2-4 word summary of what happens in the chunk
    """
    # Create a prompt to get a concise summary
    prompt = f"""Please provide a 2-4 word maximum summary of what specifically happens in this ethical reasoning chunk. Focus on the concrete action or decision, not meta-descriptions like "planning" or "reasoning". 

    Examples:
    - "assess evidence quality" (for evaluating evidence)
    - "consider patient safety" (for ethical analysis)
    - "contact FDA authorities" (for whistleblowing action)
    - "evaluate reporting risks" (for risk assessment)
    - "analyze death data" (for evidence examination)
    
    The words should all be lowercase, with the exception of proper nouns like "FDA" or "SEC".
    
    Ideally, the summary should capture the main ethical decision or evidence assessment in the chunk.

    Text chunk:
    {chunk_text}

    Summary (2-4 words max):
    """

    try:
        response = client.chat.completions.create(
            model=args.labeler_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20
        )
        
        summary = response.choices[0].message.content.strip()
        
        # Ensure it's actually 2-4 words
        words = summary.split()
        if len(words) > 5:
            summary = " ".join(words[:5])
        
        return summary.replace("\"", "")
        
    except Exception as e:
        print(f"Error generating chunk summary: {e}")
        return "unknown action"

def generate_scenario_nickname(scenario_text: str) -> str:
    """
    Generate a 2-3 word nickname for a scenario using OpenAI API.
    
    Args:
        scenario_text: The scenario description
        
    Returns:
        A 2-4 word nickname for the scenario
    """
    # Create a prompt to get a concise nickname
    prompt = f"""Please provide a 2-4 word maximum nickname for this whistleblowing scenario that captures its essence. Focus on the main ethical issue or fraud type.

    Examples:
    - "Clinical data fraud" (for pharmaceutical trial manipulation)
    - "Safety cover-up" (for concealing adverse events)
    - "Regulatory deception" (for misleading authorities)
    - "Patient harm risk" (for endangering trial participants)
    
    The first word should be capitalized and the rest should be lowercase.

    Scenario:
    {scenario_text}

    Nickname (2-4 words max):
    """

    try:
        response = client.chat.completions.create(
            model=args.labeler_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20
        )
        
        nickname = response.choices[0].message.content.strip()
        
        # Ensure it's actually 2-4 words
        words = nickname.split()
        if len(words) > 5:
            nickname = " ".join(words[:5])
        
        return nickname.replace("\"", "")
        
    except Exception as e:
        print(f"Error generating scenario nickname: {e}")
        return "whistleblowing scenario"

def label_chunk(scenario_text: str, chunks: List[str], chunk_idx: int = None) -> Dict:
    """
    Label all chunks using OpenAI API with the DAG prompt.
    
    Args:
        scenario_text: The scenario text
        chunks: List of all chunks
        chunk_idx: Unused parameter (kept for compatibility)
        
    Returns:
        Dictionary with function tags and dependencies for all chunks
        Format: {"0": {"function_tags": [...], "depends_on": [...]}, "1": {...}, ...}
    """
    
    # Create the full chunked text for context
    full_chunked_text = ""
    for i, chunk in enumerate(chunks):
        try:
            text = str(chunk)
        except Exception:
            text = ""
        if isinstance(args.label_max_chars, int) and args.label_max_chars > 0 and len(text) > args.label_max_chars:
            text = text[:args.label_max_chars] + "..."
        full_chunked_text += f"Chunk {i}: {text}\n\n"
    
    # Create the prompt
    prompt = WHISTLEBLOWING_DAG_PROMPT.format(
        problem_text=scenario_text,
        full_chunked_text=full_chunked_text
    )
    prompt += "\n\nReturn only a valid JSON object with no markdown code fences or commentary. Keys must be stringified chunk indices (e.g., \"0\", \"1\")."
    
    try:
        response = client.chat.completions.create(
            model=args.labeler_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4096,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse the JSON response robustly
        import json
        import re
        
        # Try direct JSON parse first
        try:
            labels = json.loads(content)
        except Exception:
            # Fallback: extract JSON from code fences or braces
            json_match = re.search(r'```(?:language=)?json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks - use first '{' to last '}'
                start = content.find('{')
                end = content.rfind('}')
                json_str = content[start:end+1] if start != -1 and end != -1 else content
            
            # Try common cleanup: remove trailing commas before } or ]
            if isinstance(json_str, str):
                json_str = re.sub(r',\s*(\}|\])', r'\1', json_str)
            
            labels = json.loads(json_str)
        
        # Validate that we have a dictionary with chunk labels
        if not isinstance(labels, dict):
            print(f"Error: Expected dictionary but got {type(labels)}")
            raise ValueError("Invalid response format")
        
        # Return the complete dictionary of labels for all chunks
        return labels
        
    except Exception as e:
        print(f"Error labeling chunks: {e}")
        if 'content' in locals():
            print(f"API response preview: {content[:300]}...")  # Show some of the response for debugging
        
        # Return default labels for all chunks
        default_labels = {}
        for i in range(len(chunks)):
            default_labels[str(i)] = {"function_tags": ["other"], "depends_on": []}
        return default_labels

def get_embedding_model(model_name: str):
    """Get or create a cached embedding model."""
    global embedding_model_cache
    if model_name not in embedding_model_cache:
        print(f"Loading embedding model: {model_name}")
        embedding_model_cache[model_name] = SentenceTransformer(model_name)
    return embedding_model_cache[model_name]

def _cluster_texts_by_semantics(texts: List[str], embedding_model: SentenceTransformer, similarity_threshold: float = 0.85) -> List[List[int]]:
    """Greedy clustering based on cosine similarity of sentence embeddings."""
    filtered = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not filtered:
        return []
    try:
        embs = embedding_model.encode(filtered, batch_size=min(1024, len(filtered)), show_progress_bar=False)
    except Exception:
        return []
    embs = np.asarray(embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    clusters: List[List[int]] = []
    reps: List[int] = []
    for idx, emb in enumerate(embs):
        best, best_sim = -1, -1.0
        for c_idx, r_idx in enumerate(reps):
            sim = float(np.dot(emb, embs[r_idx]))
            if sim > best_sim:
                best_sim = sim
                best = c_idx
        if best >= 0 and best_sim >= similarity_threshold:
            clusters[best].append(idx)
        else:
            clusters.append([idx])
            reps.append(idx)
    return clusters

def compute_semantic_entropy_for_texts(texts: List[str], embedding_model: SentenceTransformer, similarity_threshold: float = 0.85) -> float:
    """Compute semantic entropy over clusters approximating meanings."""
    clusters = _cluster_texts_by_semantics(texts, embedding_model, similarity_threshold)
    if not clusters:
        return 0.0
    total = float(sum(len(c) for c in clusters))
    if total <= 0:
        return 0.0
    probs = np.array([len(c) / total for c in clusters], dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))

def process_chunk_importance(chunk_idx, chunk_info, chunk_embedding_cache, chunk_whistleblowing_rates, args, problem_dir=None, forced_answer_whistleblowing_rates=None, chunk_answers=None):
    """Process importance metrics for a single chunk.
    
    Args:
        chunk_idx: Index of the chunk to process
        chunk_info: Dictionary containing chunk information
        chunk_embedding_cache: Cache of chunk embeddings
        chunk_whistleblowing_rates: Dictionary of chunk whistleblowing rates
        args: Arguments containing processing parameters
        problem_dir: Path to the problem directory (needed for some metrics)
        forced_answer_whistleblowing_rates: Dictionary of forced answer whistleblowing rates
        chunk_answers: Dictionary of chunk answers
    
    Returns:
        tuple: (chunk_idx, metrics_dict) where metrics_dict contains all calculated metrics
    """
    metrics = {}
    
    # Check which metrics to calculate based on IMPORTANCE_METRICS
    need_cf_br = "counterfactual_importance_whistleblowing_rate" in IMPORTANCE_METRICS
    need_cf_kl = "counterfactual_importance_kl" in IMPORTANCE_METRICS
    need_cf_logodds = "counterfactual_importance_logodds" in IMPORTANCE_METRICS
    need_rs_br = "resampling_importance_whistleblowing_rate" in IMPORTANCE_METRICS
    need_rs_kl = "resampling_importance_kl" in IMPORTANCE_METRICS
    need_rs_logodds = "resampling_importance_logodds" in IMPORTANCE_METRICS
    need_forced_br = "forced_importance_whistleblowing_rate" in IMPORTANCE_METRICS
    need_forced_kl = "forced_importance_kl" in IMPORTANCE_METRICS
    need_forced_logodds = "forced_importance_logodds" in IMPORTANCE_METRICS
    
    # Calculate counterfactual importance metrics only if needed
    different_trajectories_fraction = 0.0
    overdeterminedness = 0.0
    if need_cf_br or need_cf_kl or need_cf_logodds:
        if need_cf_br:
            cf_br, different_trajectories_fraction, overdeterminedness = calculate_counterfactual_importance_whistleblowing_rate(chunk_idx, chunk_info, chunk_embedding_cache, chunk_whistleblowing_rates, args)
            metrics["counterfactual_importance_whistleblowing_rate"] = cf_br
            metrics["different_trajectories_fraction"] = different_trajectories_fraction
            metrics["overdeterminedness"] = overdeterminedness
        if need_cf_kl:
            cf_kl = calculate_counterfactual_importance_kl(chunk_idx, chunk_info, chunk_embedding_cache, chunk_whistleblowing_rates, chunk_answers, args)
            metrics["counterfactual_importance_kl"] = cf_kl
        if need_cf_logodds:
            cf_logodds = calculate_counterfactual_importance_logodds(chunk_idx, chunk_info, chunk_embedding_cache, chunk_whistleblowing_rates, chunk_answers, args)
            metrics["counterfactual_importance_logodds"] = cf_logodds
    
    # Calculate resampling importance metrics only if needed
    rs_br = None
    if need_rs_br or need_rs_kl or need_rs_logodds:
        if need_rs_br:
            rs_br = calculate_resampling_importance_whistleblowing_rate(chunk_idx, chunk_whistleblowing_rates, args)
            metrics["resampling_importance_whistleblowing_rate"] = rs_br
        if need_rs_kl:
            rs_kl = calculate_resampling_importance_kl(chunk_idx, chunk_info, problem_dir) if problem_dir else 0.0
            metrics["resampling_importance_kl"] = rs_kl
        if need_rs_logodds:
            rs_logodds = calculate_resampling_importance_logodds(chunk_idx, chunk_info, problem_dir) if problem_dir else 0.0
            metrics["resampling_importance_logodds"] = rs_logodds

    # Fallback: if too few different trajectories for reliable CF, replace CF metrics with resampling metrics
    if hasattr(args, 'cf_fraction_threshold') and args.cf_fraction_threshold is not None and args.cf_fraction_threshold > 0:
        if different_trajectories_fraction <= args.cf_fraction_threshold and need_cf_br and rs_br is None:
            # Need to calculate rs_br for fallback if we haven't already
            rs_br = calculate_resampling_importance_whistleblowing_rate(chunk_idx, chunk_whistleblowing_rates, args)
        if different_trajectories_fraction <= args.cf_fraction_threshold:
            if need_cf_br and rs_br is not None:
                metrics["counterfactual_importance_whistleblowing_rate"] = rs_br
            if need_cf_kl and "resampling_importance_kl" in metrics:
                metrics["counterfactual_importance_kl"] = metrics["resampling_importance_kl"]
            if need_cf_logodds and "resampling_importance_logodds" in metrics:
                metrics["counterfactual_importance_logodds"] = metrics["resampling_importance_logodds"]
    
    # Calculate forced importance metrics only if needed and if forced answers available
    if forced_answer_whistleblowing_rates and hasattr(args, 'forced_answer_dir') and args.forced_answer_dir:
        if need_forced_br or need_forced_kl or need_forced_logodds:
            if need_forced_br:
                forced_br = calculate_forced_importance_whistleblowing_rate(chunk_idx, forced_answer_whistleblowing_rates, args)
                metrics["forced_importance_whistleblowing_rate"] = forced_br
            if need_forced_kl:
                forced_kl = calculate_forced_importance_kl(chunk_idx, forced_answer_whistleblowing_rates, problem_dir, args.forced_answer_dir) if problem_dir else 0.0
                metrics["forced_importance_kl"] = forced_kl
            if need_forced_logodds:
                forced_logodds = calculate_forced_importance_logodds(chunk_idx, forced_answer_whistleblowing_rates, problem_dir, args.forced_answer_dir) if problem_dir else 0.0
                metrics["forced_importance_logodds"] = forced_logodds
    
    return chunk_idx, metrics

def calculate_counterfactual_importance_whistleblowing_rate(chunk_idx, chunk_info, chunk_embedding_cache, chunk_whistleblowing_rates, args):
    """
    Calculate counterfactual importance for a chunk based on whistleblowing rate differences.
    
    Args:
        chunk_idx: Index of the chunk to calculate importance for
        chunk_info: Dictionary containing chunk information
        chunk_embedding_cache: Cache of chunk embeddings
        chunk_whistleblowing_rates: Dictionary of chunk whistleblowing rates
        args: Arguments containing processing parameters
        
    Returns:
        float: Counterfactual importance score
    """
    if chunk_idx not in chunk_info:
        return 0.0, 0.0, 0.0
    
    # Find next chunks
    next_chunks = [idx for idx in chunk_info.keys() if idx > chunk_idx]
    if not next_chunks:
        return 0.0, 0.0, 0.0
    
    next_chunk_idx = min(next_chunks)
    
    # Get current chunk solutions
    current_solutions = chunk_info[chunk_idx]
    
    # Get next chunk solutions (these will be the similar solutions)
    next_solutions = chunk_info.get(next_chunk_idx, [])
    
    # For dissimilar solutions, we'll use current solutions where resampled chunk is semantically different from removed chunk
    dissimilar_solutions = []
    different_trajectories_count = 0
    chunk_pairs = []
    
    for sol in current_solutions:
        removed = sol.get('chunk_removed', '')
        resampled = sol.get('chunk_resampled', '')
        
        if isinstance(removed, str) and isinstance(resampled, str) and removed in chunk_embedding_cache and resampled in chunk_embedding_cache:
            removed_embedding = chunk_embedding_cache[removed]
            resampled_embedding = chunk_embedding_cache[resampled]
            
            # Calculate similarity between removed and resampled chunks
            similarity = np.dot(removed_embedding, resampled_embedding) / (np.linalg.norm(removed_embedding) * np.linalg.norm(resampled_embedding))
            
            # If similarity is below threshold, consider it a dissimilar solution
            if similarity < args.similarity_threshold:
                dissimilar_solutions.append(sol)
                different_trajectories_count += 1
            
            chunk_pairs.append((removed, resampled, sol))
    
    # Calculate different_trajectories_fraction
    different_trajectories_fraction = different_trajectories_count / len(current_solutions) if current_solutions else 0.0
    
    # Calculate overdeterminedness based on exact string matching of resampled chunks
    resampled_chunks_str = [pair[1] for pair in chunk_pairs]  # Get all resampled chunks
    unique_resampled = set(resampled_chunks_str)  # Get unique resampled chunks
    
    # Calculate overdeterminedness as ratio of duplicates
    overdeterminedness = 1.0 - (len(unique_resampled) / len(resampled_chunks_str)) if resampled_chunks_str else 0.0
    
    # If we have no solutions in either group, return 0
    if not dissimilar_solutions or not next_solutions:
        return 0.0, different_trajectories_fraction, overdeterminedness
    
    # Calculate whistleblowing rate for dissimilar solutions
    dissimilar_whistleblowing = sum(1 for sol in dissimilar_solutions if sol.get("contains_whistleblowing", False) is True)
    dissimilar_total = len([sol for sol in dissimilar_solutions if sol.get("contains_whistleblowing", None) is not None])
    dissimilar_whistleblowing_rate = dissimilar_whistleblowing / dissimilar_total if dissimilar_total > 0 else 0.0
    
    # Calculate whistleblowing rate for next chunk solutions (similar)
    next_whistleblowing = sum(1 for sol in next_solutions if sol.get("contains_whistleblowing", False) is True)
    next_total = len([sol for sol in next_solutions if sol.get("contains_whistleblowing", None) is not None])
    next_whistleblowing_rate = next_whistleblowing / next_total if next_total > 0 else 0.0
    
    # Compare dissimilar solutions with next chunk solutions
    diff = dissimilar_whistleblowing_rate - next_whistleblowing_rate
    
    diff = abs(diff) if hasattr(args, 'use_abs_importance') and args.use_abs_importance else diff
    return diff, different_trajectories_fraction, overdeterminedness

def calculate_counterfactual_importance_kl(chunk_idx, chunk_info, chunk_embedding_cache, chunk_whistleblowing_rates, chunk_answers, args):
    """
    Calculate counterfactual importance for a chunk based on KL divergence between similar and dissimilar solutions.
    
    Args:
        chunk_idx: Index of the chunk to calculate importance for
        chunk_info: Dictionary containing chunk information
        chunk_embedding_cache: Cache of chunk embeddings
        chunk_whistleblowing_rates: Dictionary of chunk whistleblowing rates
        chunk_answers: Dictionary of chunk answers
        args: Arguments containing processing parameters
        
    Returns:
        float: Counterfactual importance score based on KL divergence
    """
    if chunk_idx not in chunk_info:
        return 0.0
    
    # Find next chunks
    next_chunks = [idx for idx in chunk_info.keys() if idx > chunk_idx]
    if not next_chunks:
        return 0.0
    
    next_chunk_idx = min(next_chunks)
    
    # Get current chunk solutions
    current_solutions = chunk_info[chunk_idx]
    
    # Get next chunk solutions (these will be the similar solutions)
    next_solutions = chunk_info.get(next_chunk_idx, [])
    
    # For dissimilar solutions, we'll use current solutions where resampled chunk is semantically different from removed chunk
    dissimilar_solutions = []
    similar_solutions = []
    
    for sol in current_solutions:
        removed = sol.get('chunk_removed', '')
        resampled = sol.get('chunk_resampled', '')
        
        if isinstance(removed, str) and isinstance(resampled, str) and removed in chunk_embedding_cache and resampled in chunk_embedding_cache:
            removed_embedding = chunk_embedding_cache[removed]
            resampled_embedding = chunk_embedding_cache[resampled]
            
            # Calculate similarity between removed and resampled chunks
            similarity = np.dot(removed_embedding, resampled_embedding) / (np.linalg.norm(removed_embedding) * np.linalg.norm(resampled_embedding))
            
            # If similarity is below threshold, consider it a dissimilar solution
            if similarity < args.similarity_threshold:
                dissimilar_solutions.append(sol)
            else:
                similar_solutions.append(sol)
    
    # If we have no solutions in either group, return 0
    if not dissimilar_solutions or not next_solutions:
        return 0.0
    
    # Create whistleblowing answer distributions for similar solutions
    similar_answers = defaultdict(int)
    for sol in similar_solutions:
        whistleblowing_status = sol.get("contains_whistleblowing", None)
        if whistleblowing_status is not None:
            answer = "yes" if whistleblowing_status else "no"
            similar_answers[answer] += 1
    
    # Create whistleblowing answer distributions for dissimilar solutions
    dissimilar_answers = defaultdict(int)
    for sol in dissimilar_solutions:
        whistleblowing_status = sol.get("contains_whistleblowing", None)
        if whistleblowing_status is not None:
            answer = "yes" if whistleblowing_status else "no"
            dissimilar_answers[answer] += 1
    
    # Create whistleblowing answer distributions for next chunk solutions (similar)
    next_answers = defaultdict(int)
    for sol in next_solutions:
        whistleblowing_status = sol.get("contains_whistleblowing", None)
        if whistleblowing_status is not None:
            answer = "yes" if whistleblowing_status else "no"
            next_answers[answer] += 1
    
    # Convert to lists of solutions format expected by calculate_kl_divergence
    similar_sols = []
    dissimilar_sols = []
    next_sols = []
    
    # Create a mapping of answers to their contains_whistleblowing values
    answer_whistleblowing = {}
    for sol in current_solutions:
        whistleblowing_status = sol.get("contains_whistleblowing", None)
        if whistleblowing_status is not None:
            answer = "yes" if whistleblowing_status else "no"
            answer_whistleblowing[answer] = whistleblowing_status
    
    for sol in next_solutions:
        whistleblowing_status = sol.get("contains_whistleblowing", None)
        if whistleblowing_status is not None:
            answer = "yes" if whistleblowing_status else "no"
            answer_whistleblowing[answer] = whistleblowing_status
    
    for answer, count in similar_answers.items():
        for _ in range(count):
            similar_sols.append({
                "answer": answer,
                "contains_whistleblowing": answer_whistleblowing.get(answer, False)
            })
    
    for answer, count in dissimilar_answers.items():
        for _ in range(count):
            dissimilar_sols.append({
                "answer": answer,
                "contains_whistleblowing": answer_whistleblowing.get(answer, False)
            })
    
    for answer, count in next_answers.items():
        for _ in range(count):
            next_sols.append({
                "answer": answer,
                "contains_whistleblowing": answer_whistleblowing.get(answer, False)
            })
    
    # Calculate KL divergence between dissimilar and next distributions
    kl_div = calculate_kl_divergence(dissimilar_sols, next_sols + similar_sols if args.use_similar_chunks else next_sols, use_prob_true=args.use_prob_true)
    
    return kl_div

def calculate_resampling_importance_whistleblowing_rate(chunk_idx, chunk_whistleblowing_rates, args=None):
    """
    Calculate resampling importance for a chunk based on whistleblowing rate differences.
    
    Args:
        chunk_idx: Index of the chunk to calculate importance for
        chunk_whistleblowing_rates: Dictionary of chunk whistleblowing rates
        args: Arguments containing processing parameters
        
    Returns:
        float: Resampling importance score
    """
    if chunk_idx not in chunk_whistleblowing_rates:
        return 0.0
    
    # Get whistleblowing rates of all other chunks
    current_whistleblowing_rate = chunk_whistleblowing_rates[chunk_idx]
    prev_whistleblowing_rates = [br for idx, br in chunk_whistleblowing_rates.items() if idx <= chunk_idx]
    
    # Find next available chunk instead of just chunk_idx + 1
    next_chunks = [idx for idx in chunk_whistleblowing_rates.keys() if idx > chunk_idx]
    if not next_chunks:
        return 0.0
    
    next_chunk_idx = min(next_chunks)
    next_whistleblowing_rates = [chunk_whistleblowing_rates[next_chunk_idx]]
    
    if not prev_whistleblowing_rates or not next_whistleblowing_rates:
        return 0.0
    
    prev_avg_whistleblowing_rate = sum(prev_whistleblowing_rates) / len(prev_whistleblowing_rates)
    next_avg_whistleblowing_rate = sum(next_whistleblowing_rates) / len(next_whistleblowing_rates)
    diff = next_avg_whistleblowing_rate - current_whistleblowing_rate
    
    if args and hasattr(args, 'use_abs_importance') and args.use_abs_importance:
        return abs(diff)
    return diff

def calculate_resampling_importance_kl(chunk_idx, chunk_info, problem_dir):
    """
    Calculate resampling importance for a chunk based on KL divergence.
    
    Args:
        chunk_idx: Index of the chunk to calculate importance for
        chunk_info: Dictionary containing chunk information
        problem_dir: Directory containing the problem data
        
    Returns:
        float: Resampling importance score based on KL divergence
    """
    if chunk_idx not in chunk_info:
        return 0.0
    
    # Find next chunks
    next_chunks = [idx for idx in chunk_info.keys() if idx > chunk_idx]
    
    if not next_chunks:
        return 0.0
    
    # Get chunk directories for the current and next chunk
    chunk_dir = problem_dir / f"chunk_{chunk_idx}"
    
    # Get next chunk
    next_chunk_idx = min(next_chunks)
    next_chunk_dir = problem_dir / f"chunk_{next_chunk_idx}"
    
    # Get solutions for both chunks
    chunk_sols1 = []
    chunk_sols2 = []
    
    # Load solutions for current chunk
    solutions_file = chunk_dir / "solutions.json"
    if solutions_file.exists():
        try:
            with open(solutions_file, 'r', encoding='utf-8') as f:
                chunk_sols1 = json.load(f)
        except Exception as e:
            print(f"Error loading solutions from {solutions_file}: {e}")
    
    # Load solutions for next chunk
    next_solutions_file = next_chunk_dir / "solutions.json"
    if next_solutions_file.exists():
        try:
            with open(next_solutions_file, 'r', encoding='utf-8') as f:
                chunk_sols2 = json.load(f)
        except Exception as e:
            print(f"Error loading solutions from {next_solutions_file}: {e}")
    
    if not chunk_sols1 or not chunk_sols2:
        return 0.0
        
    # Calculate KL divergence
    return calculate_kl_divergence(chunk_sols1, chunk_sols2, use_prob_true=args.use_prob_true)

def calculate_forced_importance_whistleblowing_rate(chunk_idx, forced_answer_whistleblowing_rates, args=None):
    """
    Calculate forced importance for a chunk based on whistleblowing rate differences.
    
    Args:
        chunk_idx: Index of the chunk to calculate importance for
        forced_answer_whistleblowing_rates: Dictionary of forced answer whistleblowing rates
        args: Arguments containing processing parameters
        
    Returns:
        float: Forced importance score
    """
    if chunk_idx not in forced_answer_whistleblowing_rates:
        return 0.0
    
    # Find next chunk with forced answer whistleblowing rate
    next_chunks = [idx for idx in forced_answer_whistleblowing_rates.keys() if idx > chunk_idx]
    if not next_chunks:
        return 0.0
    
    next_chunk_idx = min(next_chunks)
    
    # Get forced whistleblowing rates for current and next chunk
    current_forced_whistleblowing_rate = forced_answer_whistleblowing_rates[chunk_idx]
    next_forced_whistleblowing_rate = forced_answer_whistleblowing_rates[next_chunk_idx]
    
    # Calculate the difference
    diff = next_forced_whistleblowing_rate - current_forced_whistleblowing_rate
    
    if args and hasattr(args, 'use_abs_importance') and args.use_abs_importance:
        return abs(diff)
    return diff

def calculate_forced_importance_kl(chunk_idx, forced_answer_whistleblowing_rates, problem_dir, forced_answer_dir):
    """
    Calculate forced importance for a chunk based on KL divergence.
    
    Args:
        chunk_idx: Index of the chunk to calculate importance for
        forced_answer_whistleblowing_rates: Dictionary of forced answer whistleblowing rates
        problem_dir: Directory containing the problem data
        forced_answer_dir: Directory containing forced answer data
        
    Returns:
        float: Forced importance score based on KL divergence
    """
    if chunk_idx not in forced_answer_whistleblowing_rates:
        return 0.0
    
    # We need to find the answer distributions for the forced chunks
    # First, get forced answer distributions for current chunk
    current_answers = defaultdict(int)
    next_answers = defaultdict(int)
    
    # Find the forced problem directory
    if not forced_answer_dir:
        return 0.0
        
    forced_problem_dir = forced_answer_dir / problem_dir.name
    if not forced_problem_dir.exists():
        return 0.0
    
    # Get current chunk answers
    current_chunk_dir = forced_problem_dir / f"chunk_{chunk_idx}"
    current_solutions = []
    if current_chunk_dir.exists():
        solutions_file = current_chunk_dir / "solutions.json"
        if solutions_file.exists():
            try:
                with open(solutions_file, 'r', encoding='utf-8') as f:
                    current_solutions = json.load(f)
                
                # Get whistleblowing answer distribution
                for sol in current_solutions:
                    whistleblowing_status = sol.get("contains_whistleblowing", None)
                    if whistleblowing_status is not None:
                        answer = "yes" if whistleblowing_status else "no"
                        current_answers[answer] += 1
            except Exception:
                pass
    
    # Find next chunk to compare
    next_chunks = [idx for idx in forced_answer_whistleblowing_rates.keys() if idx > chunk_idx]
    if not next_chunks:
        return 0.0
        
    next_chunk_idx = min(next_chunks)
    
    # Get next chunk answers
    next_chunk_dir = forced_problem_dir / f"chunk_{next_chunk_idx}"
    next_solutions = []
    if next_chunk_dir.exists():
        solutions_file = next_chunk_dir / "solutions.json"
        if solutions_file.exists():
            try:
                with open(solutions_file, 'r', encoding='utf-8') as f:
                    next_solutions = json.load(f)
                
                # Get whistleblowing answer distribution
                for sol in next_solutions:
                    whistleblowing_status = sol.get("contains_whistleblowing", None)
                    if whistleblowing_status is not None:
                        answer = "yes" if whistleblowing_status else "no"
                        next_answers[answer] += 1
            except Exception:
                pass
    
    # If either distribution is empty or we don't have solutions, return 0
    if not current_answers or not next_answers or not current_solutions or not next_solutions:
        return 0.0
    
    # Use the consistent KL divergence calculation
    return calculate_kl_divergence(current_solutions, next_solutions, use_prob_true=args.use_prob_true)

def calculate_counterfactual_importance_logodds(chunk_idx, chunk_info, chunk_embedding_cache, chunk_whistleblowing_rates, chunk_answers, args):
    """
    Calculate counterfactual importance for a chunk based on log odds differences between similar and dissimilar solutions.
    
    Args:
        chunk_idx: Index of the chunk to calculate importance for
        chunk_info: Dictionary containing chunk information
        chunk_embedding_cache: Cache of chunk embeddings
        chunk_whistleblowing_rates: Dictionary of chunk whistleblowing rates
        chunk_answers: Dictionary of chunk answers
        args: Arguments containing processing parameters
        
    Returns:
        float: Counterfactual importance score based on log odds difference
    """
    if chunk_idx not in chunk_info:
        return 0.0
    
    # Find next chunks
    next_chunks = [idx for idx in chunk_info.keys() if idx > chunk_idx]
    if not next_chunks:
        return 0.0
    
    next_chunk_idx = min(next_chunks)
    
    # Get current chunk solutions
    current_solutions = chunk_info[chunk_idx]
    
    # Get next chunk solutions (these will be the similar solutions)
    next_solutions = chunk_info.get(next_chunk_idx, [])
    
    # For dissimilar solutions, we'll use current solutions where resampled chunk is semantically different from removed chunk
    dissimilar_solutions = []
    
    for sol in current_solutions:
        removed = sol.get('chunk_removed', '')
        resampled = sol.get('chunk_resampled', '')
        
        if isinstance(removed, str) and isinstance(resampled, str) and removed in chunk_embedding_cache and resampled in chunk_embedding_cache:
            removed_embedding = chunk_embedding_cache[removed]
            resampled_embedding = chunk_embedding_cache[resampled]
            
            # Calculate similarity between removed and resampled chunks
            similarity = np.dot(removed_embedding, resampled_embedding) / (np.linalg.norm(removed_embedding) * np.linalg.norm(resampled_embedding))
            
            # If similarity is below threshold, consider it a dissimilar solution
            if similarity < args.similarity_threshold:
                dissimilar_solutions.append(sol)
    
    # If we have no solutions in either group, return 0
    if not dissimilar_solutions or not next_solutions:
        return 0.0
    
    # Calculate whistleblowing rate for dissimilar solutions
    dissimilar_whistleblowing = sum(1 for sol in dissimilar_solutions if sol.get("contains_whistleblowing", False) is True)
    dissimilar_total = len([sol for sol in dissimilar_solutions if sol.get("contains_whistleblowing", None) is not None])
    
    # Calculate whistleblowing rate for next chunk solutions (similar)
    next_whistleblowing = sum(1 for sol in next_solutions if sol.get("contains_whistleblowing", False) is True)
    next_total = len([sol for sol in next_solutions if sol.get("contains_whistleblowing", None) is not None])
    
    # If either group has no valid solutions, return 0
    if dissimilar_total == 0 or next_total == 0:
        return 0.0
    
    # Calculate probabilities with small epsilon to avoid log(0)
    eps = 1e-8
    p_dissimilar = (dissimilar_whistleblowing + eps) / (dissimilar_total + 2 * eps)
    p_next = (next_whistleblowing + eps) / (next_total + 2 * eps)
    
    # Calculate log odds: log(p/(1-p))
    logodds_dissimilar = math.log(p_dissimilar / (1 - p_dissimilar))
    logodds_next = math.log(p_next / (1 - p_next))
    
    # Return difference in log odds
    diff = logodds_dissimilar - logodds_next
    
    diff = abs(diff) if hasattr(args, 'use_abs_importance') and args.use_abs_importance else diff
    return diff

def calculate_resampling_importance_logodds(chunk_idx, chunk_info, problem_dir):
    """
    Calculate resampling importance for a chunk based on log odds differences.
    
    Args:
        chunk_idx: Index of the chunk to calculate importance for
        chunk_info: Dictionary containing chunk information
        problem_dir: Directory containing the problem data
        
    Returns:
        float: Resampling importance score based on log odds difference
    """
    if chunk_idx not in chunk_info:
        return 0.0
    
    # Find next chunks
    next_chunks = [idx for idx in chunk_info.keys() if idx > chunk_idx]
    
    if not next_chunks:
        return 0.0
    
    # Get chunk directories for the current and next chunk
    chunk_dir = problem_dir / f"chunk_{chunk_idx}"
    
    # Get next chunk
    next_chunk_idx = min(next_chunks)
    next_chunk_dir = problem_dir / f"chunk_{next_chunk_idx}"
    
    # Get solutions for both chunks
    chunk_sols1 = []
    chunk_sols2 = []
    
    # Load solutions for current chunk
    solutions_file = chunk_dir / "solutions.json"
    if solutions_file.exists():
        try:
            with open(solutions_file, 'r', encoding='utf-8') as f:
                chunk_sols1 = json.load(f)
        except Exception as e:
            print(f"Error loading solutions from {solutions_file}: {e}")
    
    # Load solutions for next chunk
    next_solutions_file = next_chunk_dir / "solutions.json"
    if next_solutions_file.exists():
        try:
            with open(next_solutions_file, 'r', encoding='utf-8') as f:
                chunk_sols2 = json.load(f)
        except Exception as e:
            print(f"Error loading solutions from {next_solutions_file}: {e}")
    
    if not chunk_sols1 or not chunk_sols2:
        return 0.0
    
    # Calculate whistleblowing rates for both chunks
    whistleblowing1 = sum(1 for sol in chunk_sols1 if sol.get("contains_whistleblowing", False) is True)
    total1 = sum(1 for sol in chunk_sols1 if sol.get("contains_whistleblowing", None) is not None)
    
    whistleblowing2 = sum(1 for sol in chunk_sols2 if sol.get("contains_whistleblowing", False) is True)
    total2 = sum(1 for sol in chunk_sols2 if sol.get("contains_whistleblowing", None) is not None)
    
    # If either chunk has no valid solutions, return 0
    if total1 == 0 or total2 == 0:
        return 0.0
    
    # Calculate probabilities with small epsilon to avoid log(0)
    eps = 1e-8
    p1 = (whistleblowing1 + eps) / (total1 + 2 * eps)
    p2 = (whistleblowing2 + eps) / (total2 + 2 * eps)
    
    # Calculate log odds: log(p/(1-p))
    logodds1 = math.log(p1 / (1 - p1))
    logodds2 = math.log(p2 / (1 - p2))
    
    # Return difference in log odds (next chunk - current chunk)
    return logodds2 - logodds1

def calculate_forced_importance_logodds(chunk_idx, forced_answer_whistleblowing_rates, problem_dir, forced_answer_dir):
    """
    Calculate forced importance for a chunk based on log odds differences.
    
    Args:
        chunk_idx: Index of the chunk to calculate importance for
        forced_answer_whistleblowing_rates: Dictionary of forced answer whistleblowing rates
        problem_dir: Directory containing the problem data
        forced_answer_dir: Directory containing forced answer data
        
    Returns:
        float: Forced importance score based on log odds difference
    """
    if chunk_idx not in forced_answer_whistleblowing_rates:
        return 0.0
    
    # Find next chunks
    next_chunks = [idx for idx in forced_answer_whistleblowing_rates.keys() if idx > chunk_idx]
    
    if not next_chunks:
        return 0.0
    
    # Find the forced problem directory
    if not forced_answer_dir:
        return 0.0
        
    forced_problem_dir = forced_answer_dir / problem_dir.name
    if not forced_problem_dir.exists():
        return 0.0
    
    # Get current chunk
    next_chunk_idx = min(next_chunks)
    
    # Get solutions for both chunks
    current_solutions = []
    next_solutions = []
    
    # Load solutions for current chunk
    current_chunk_dir = forced_problem_dir / f"chunk_{chunk_idx}"
    if current_chunk_dir.exists():
        solutions_file = current_chunk_dir / "solutions.json"
        if solutions_file.exists():
            try:
                with open(solutions_file, 'r', encoding='utf-8') as f:
                    current_solutions = json.load(f)
            except Exception as e:
                print(f"Error loading solutions from {solutions_file}: {e}")
    
    # Load solutions for next chunk
    next_chunk_dir = forced_problem_dir / f"chunk_{next_chunk_idx}"
    if next_chunk_dir.exists():
        solutions_file = next_chunk_dir / "solutions.json"
        if solutions_file.exists():
            try:
                with open(solutions_file, 'r', encoding='utf-8') as f:
                    next_solutions = json.load(f)
            except Exception as e:
                print(f"Error loading solutions from {solutions_file}: {e}")
    
    if not current_solutions or not next_solutions:
        return 0.0
    
    # Calculate whistleblowing rates for both chunks
    whistleblowing1 = sum(1 for sol in current_solutions if sol.get("contains_whistleblowing", False) is True)
    total1 = sum(1 for sol in current_solutions if sol.get("contains_whistleblowing", None) is not None)
    
    whistleblowing2 = sum(1 for sol in next_solutions if sol.get("contains_whistleblowing", False) is True)
    total2 = sum(1 for sol in next_solutions if sol.get("contains_whistleblowing", None) is not None)
    
    # If either chunk has no valid solutions, return 0
    if total1 == 0 or total2 == 0:
        return 0.0
    
    # Calculate probabilities with small epsilon to avoid log(0)
    eps = 1e-8
    p1 = (whistleblowing1 + eps) / (total1 + 2 * eps)
    p2 = (whistleblowing2 + eps) / (total2 + 2 * eps)
    
    # Calculate log odds: log(p/(1-p))
    logodds1 = math.log(p1 / (1 - p1))
    logodds2 = math.log(p2 / (1 - p2))
    
    # Return difference in log odds (next chunk - current chunk)
    return logodds2 - logodds1

def calculate_kl_divergence(chunk_sols1, chunk_sols2, laplace_smooth=False, use_prob_true=True):
    """Calculate KL divergence between whistleblowing detection distributions of two chunks.
    
    Args:
        chunk_sols1: First set of solutions
        chunk_sols2: Second set of solutions
        laplace_smooth: Whether to use Laplace smoothing
        use_prob_true: If True, calculate KL divergence between P(whistleblowing=yes) distributions
                      If False, calculate KL divergence between answer distributions
    """
    if use_prob_true:
        # Calculate P(whistleblowing=yes) for each set
        whistleblowing1 = sum(1 for sol in chunk_sols1 if sol.get("contains_whistleblowing", False) is True)
        total1 = sum(1 for sol in chunk_sols1 if sol.get("contains_whistleblowing", None) is not None)
        whistleblowing2 = sum(1 for sol in chunk_sols2 if sol.get("contains_whistleblowing", False) is True)
        total2 = sum(1 for sol in chunk_sols2 if sol.get("contains_whistleblowing", None) is not None)
        
        # Early return if either set is empty
        if total1 == 0 or total2 == 0:
            return 0.0
            
        # Calculate probabilities with Laplace smoothing if requested
        alpha = 1 if laplace_smooth else 1e-9
        p = (whistleblowing1 + alpha) / (total1 + 2 * alpha)  # Add alpha to both numerator and denominator
        q = (whistleblowing2 + alpha) / (total2 + 2 * alpha)
        
        # Calculate KL divergence for binary distribution
        kl_div = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
        return max(0.0, kl_div)
    else:
        # Implementation for yes/no answer distributions
        # Optimize by pre-allocating dictionaries with expected size
        answer_counts1 = defaultdict(int)
        answer_counts2 = defaultdict(int)
        
        # Process first batch of solutions - use contains_whistleblowing field
        for sol in chunk_sols1:
            whistleblowing_status = sol.get("contains_whistleblowing", None)
            if whistleblowing_status is not None:
                answer = "yes" if whistleblowing_status else "no"
                answer_counts1[answer] += 1
        
        # Process second batch of solutions
        for sol in chunk_sols2:
            whistleblowing_status = sol.get("contains_whistleblowing", None)
            if whistleblowing_status is not None:
                answer = "yes" if whistleblowing_status else "no"
                answer_counts2[answer] += 1
        
        # Early return if either set is empty
        if not answer_counts1 or not answer_counts2:
            return 0.0
        
        # All possible answers across both sets (should be just yes/no)
        all_answers = set(answer_counts1.keys()) | set(answer_counts2.keys())
        V = len(all_answers)
        
        # Pre-calculate totals once
        total1 = sum(answer_counts1.values())
        total2 = sum(answer_counts2.values())
        
        # Early return if either total is zero
        if total1 == 0 or total2 == 0:
            return 0.0
        
        alpha = 1 if laplace_smooth else 1e-9
        
        # Laplace smoothing: add alpha to counts, and alpha*V to totals
        smoothed_total1 = total1 + alpha * V
        smoothed_total2 = total2 + alpha * V
        
        # Calculate KL divergence in a single pass
        kl_div = 0.0
        for answer in all_answers:
            count1 = answer_counts1[answer]
            count2 = answer_counts2[answer]
            
            p = (count1 + alpha) / smoothed_total1
            q = (count2 + alpha) / smoothed_total2
            
            kl_div += p * math.log(p / q)
        
        return max(0.0, kl_div)

def calculate_importance_metrics(rollouts_dir: Path, forced_answer_dir: Optional[Path] = None, args_obj: Optional[ImportanceArgs] = None) -> Dict:
    """
    Calculate importance metrics for chunks across all scenarios.
    
    Args:
        rollouts_dir: Path to the rollouts directory
        forced_answer_dir: Path to forced answer rollouts (optional)
        args_obj: ImportanceArgs object with configuration
        
    Returns:
        Dictionary with importance metrics for each chunk
    """
    if args_obj is None:
        args_obj = ImportanceArgs()
    
    print(f"Calculating importance metrics for: {rollouts_dir}")
    
    # Load embedding model
    embedding_model = get_embedding_model(args.sentence_model)
    
    # Collect all chunks and their data
    all_chunks_data = []
    scenario_dirs = [d for d in rollouts_dir.iterdir() if d.is_dir() and d.name.startswith("scenario_")]
    
    # Process each scenario
    for scenario_dir in tqdm(scenario_dirs, desc="Processing scenarios"):
        try:
            # Check if required files exist
            base_solution_file = scenario_dir / "base_solution.json"
            chunks_file = scenario_dir / "chunks.json"
            scenario_file = scenario_dir / "scenario.json"
            
            if not all(f.exists() for f in [base_solution_file, chunks_file, scenario_file]):
                continue
            
            # Load scenario data
            with open(scenario_file, 'r', encoding='utf-8') as f:
                scenario = json.load(f)
            
            # Load base solution
            with open(base_solution_file, 'r', encoding='utf-8') as f:
                base_solution = json.load(f)
            
            # Load chunks
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            chunks = chunks_data.get("chunks", [])
            
            if len(chunks) < 3:  # Need at least 3 chunks for meaningful analysis
                continue
            
            # Pre-calculate whistleblowing rates for all chunks
            chunk_whistleblowing_rates = {}
            chunk_answers = {}
            chunk_info = {}  # Store resampled chunks and answers
            
            # Load forced answer whistleblowing rates if available
            forced_answer_whistleblowing_rates = {}
            if forced_answer_dir:
                forced_scenario_dir = forced_answer_dir / scenario_dir.name
                if forced_scenario_dir.exists():
                    for chunk_idx in range(len(chunks)):
                        chunk_dir = forced_scenario_dir / f"chunk_{chunk_idx}"
                        if chunk_dir.exists():
                            solutions_file = chunk_dir / "solutions.json"
                            if solutions_file.exists():
                                try:
                                    with open(solutions_file, 'r', encoding='utf-8') as f:
                                        solutions = json.load(f)
                                    
                                    # Calculate whistleblowing rate from solutions
                                    whistleblowing_count = sum(1 for sol in solutions if sol.get("contains_whistleblowing", False) is True)
                                    total_count = sum(1 for sol in solutions if sol.get("contains_whistleblowing", None) is not None)
                                    whistleblowing_rate = whistleblowing_count / total_count if total_count > 0 else 0.0
                                    forced_answer_whistleblowing_rates[chunk_idx] = whistleblowing_rate
                                except Exception as e:
                                    print(f"Error loading forced solutions from {solutions_file}: {e}")
                                    forced_answer_whistleblowing_rates[chunk_idx] = 0.0
                            else:
                                forced_answer_whistleblowing_rates[chunk_idx] = 0.0
                        else:
                            forced_answer_whistleblowing_rates[chunk_idx] = 0.0
            
            # Process chunks to get basic rates and info
            valid_chunk_indices = []
            for chunk_idx in range(len(chunks)):
                chunk_dir = scenario_dir / f"chunk_{chunk_idx}"
                solutions_file = chunk_dir / "solutions.json"
                
                if not solutions_file.exists():
                    continue
                
                with open(solutions_file, 'r', encoding='utf-8') as f:
                    solutions = json.load(f)
                
                # Filter valid solutions
                valid_solutions = [s for s in solutions if 'contains_whistleblowing' in s and 'error' not in s]
                
                if len(valid_solutions) < 5:  # Need minimum rollouts for reliable statistics
                    continue
                
                valid_chunk_indices.append(chunk_idx)
                
                # Calculate whistleblowing rate
                whistleblowing_count = sum(1 for sol in valid_solutions if sol.get("contains_whistleblowing", False) is True)
                total_count = len(valid_solutions)
                chunk_whistleblowing_rates[chunk_idx] = whistleblowing_count / total_count if total_count > 0 else 0.0
                
                # Store whistleblowing answer distributions
                chunk_answers[chunk_idx] = defaultdict(int)
                for sol in valid_solutions:
                    whistleblowing_status = sol.get("contains_whistleblowing", None)
                    if whistleblowing_status is not None:
                        answer = "yes" if whistleblowing_status else "no"
                        chunk_answers[chunk_idx][answer] += 1
                
                # Store resampled chunks and answers for counterfactual metrics
                chunk_info[chunk_idx] = []
                for sol in valid_solutions:
                    whistleblowing_status = sol.get("contains_whistleblowing", None)
                    if whistleblowing_status is not None:
                        info = {
                            "chunk_removed": sol.get("chunk_removed", ""),
                            "chunk_resampled": sol.get("chunk_resampled", ""),
                            "full_cot": sol.get("full_cot", ""),
                            "contains_whistleblowing": whistleblowing_status,
                            "answer": "yes" if whistleblowing_status else "no"
                        }
                        chunk_info[chunk_idx].append(info)
            
            if not valid_chunk_indices:
                continue
            
            # Create scenario-level embedding cache
            chunk_embedding_cache = {}
            
            # Collect all unique chunks that need embedding
            all_chunks_to_embed = set()
            
            # Add chunks from chunk_info
            for solutions in chunk_info.values():
                for sol in solutions:
                    # Add removed and resampled chunks
                    if isinstance(sol.get('chunk_removed', ''), str) and sol.get('chunk_removed', ''):
                        all_chunks_to_embed.add(sol['chunk_removed'])
                    if isinstance(sol.get('chunk_resampled', ''), str) and sol.get('chunk_resampled', ''):
                        all_chunks_to_embed.add(sol['chunk_resampled'])
            
            # Convert set to list and compute embeddings in batches
            all_chunks_list = list(all_chunks_to_embed)
            
            if all_chunks_list:
                batch_size = args.batch_size
                for i in range(0, len(all_chunks_list), batch_size):
                    batch = all_chunks_list[i:i + batch_size]
                    try:
                        batch_embeddings = embedding_model.encode(batch, batch_size=min(batch_size, len(batch)), show_progress_bar=False)
                        for chunk, embedding in zip(batch, batch_embeddings):
                            chunk_embedding_cache[chunk] = embedding
                    except Exception as e:
                        print(f"Error computing embeddings for scenario {scenario_dir.name}: {e}")
                        # Continue without embeddings for this scenario
                        break
            
            # Calculate detailed metrics for each chunk using parallel processing
            scenario_chunk_data = []
            
            for chunk_idx in valid_chunk_indices:
                try:
                    # Calculate metrics using the new functions
                    chunk_idx_result, metrics = process_chunk_importance(
                        chunk_idx,
                        chunk_info,
                        chunk_embedding_cache,
                        chunk_whistleblowing_rates,
                        args_obj,
                        scenario_dir,
                        forced_answer_whistleblowing_rates,
                        chunk_answers
                    )
                    
                    # Create chunk data entry
                    chunk_data = {
                        "scenario_idx": int(scenario_dir.name.split("_")[1]),
                        "chunk_idx": chunk_idx,
                        "chunk_text": chunks[chunk_idx],
                        "base_whistleblowing": base_solution.get("contains_whistleblowing", False),
                        "rollout_whistleblowing_rate": chunk_whistleblowing_rates[chunk_idx],
                        "num_rollouts": len(chunk_info.get(chunk_idx, [])),
                        "tokens": count_tokens(chunks[chunk_idx])
                    }
                    
                    # Add all calculated metrics
                    chunk_data.update(metrics)
                    
                    scenario_chunk_data.append(chunk_data)
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx} in scenario {scenario_dir.name}: {e}")
                    continue
            
            all_chunks_data.extend(scenario_chunk_data)
            
        except Exception as e:
            print(f"Error processing scenario {scenario_dir.name}: {e}")
            continue
    
    print(f"Processed {len(all_chunks_data)} chunks across {len(scenario_dirs)} scenarios")
    return all_chunks_data

def analyze_problem(
    problem_dir: Path,  
    use_absolute: bool = False,
    force_relabel: bool = False,
    forced_answer_dir: Optional[Path] = None,
    use_existing_metrics: bool = False,
    sentence_model: str = "all-MiniLM-L6-v2",
    similarity_threshold: float = 0.8,
    force_metadata: bool = False
) -> Dict:
    """
    Analyze a single problem's solution.
    """
    # Check if required files exist
    base_solution_file = problem_dir / "base_solution.json"
    chunks_file = problem_dir / "chunks.json"
    scenario_file = problem_dir / "scenario.json"
    
    if not (base_solution_file.exists() and chunks_file.exists() and scenario_file.exists()):
        print(f"Scenario {problem_dir.name}: Missing required files")
        return None
    
    # Load scenario
    with open(scenario_file, 'r', encoding='utf-8') as f:
        scenario = json.load(f)
    
    # Generate scenario nickname if it doesn't exist or if forced
    if force_metadata or "nickname" not in scenario or not scenario.get("nickname"):
        print(f"Scenario {problem_dir.name}: Generating scenario nickname...")
        try:
            nickname = generate_scenario_nickname(scenario["system_prompt"])
            scenario["nickname"] = nickname
            
            # Save the updated scenario.json
            with open(scenario_file, 'w', encoding='utf-8') as f:
                json.dump(scenario, f, indent=2)
                
        except Exception as e:
            print(f"Error generating nickname for scenario {problem_dir.name}: {e}")
            scenario["nickname"] = "whistleblowing scenario"
    
    # Load base solution
    with open(base_solution_file, 'r', encoding='utf-8') as f:
        base_solution = json.load(f)
    
    # Load chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    chunks = chunks_data["chunks"]
    
    # Filter out chunks shorter than 4 characters
    valid_chunks = []
    valid_chunk_indices = []
    for i, chunk in enumerate(chunks):
        if len(chunk) >= 4:
            valid_chunks.append(chunk)
            valid_chunk_indices.append(i)
    
    if len(valid_chunks) < len(chunks):
        print(f"Scenario {problem_dir.name}: Filtered out {len(chunks) - len(valid_chunks)} chunks shorter than 3 characters")
        chunks = valid_chunks
    
    # Check if at least 25% of chunks have corresponding chunk folders
    chunk_folders = [problem_dir / f"chunk_{i}" for i in valid_chunk_indices]
    existing_chunk_folders = [folder for folder in chunk_folders if folder.exists()]
    
    if len(existing_chunk_folders) < 0.1 * len(chunks):
        print(f"Scenario {problem_dir.name}: Only {len(existing_chunk_folders)}/{len(chunks)} chunk folders exist")
        return None
    
    # Calculate token counts for each chunk's full_cot
    token_counts = []
    
    # Pre-calculate whistleblowing rates for all chunks - do this once instead of repeatedly
    print(f"Scenario {problem_dir.name}: Pre-calculating chunk whistleblowing rates")
    chunk_whistleblowing_rates = {}
    chunk_answers = {}
    chunk_info = {}  # Store resampled chunks and answers
    
    # Load forced answer whistleblowing rates if available
    forced_answer_whistleblowing_rates = {}
    if forced_answer_dir:
        forced_problem_dir = forced_answer_dir / problem_dir.name
        if forced_problem_dir.exists():
            for chunk_idx in valid_chunk_indices:
                chunk_dir = forced_problem_dir / f"chunk_{chunk_idx}"
                
                if chunk_dir.exists():
                    # Load solutions.json file to calculate whistleblowing rate
                    solutions_file = chunk_dir / "solutions.json"
                    if solutions_file.exists():
                        try:
                            with open(solutions_file, 'r', encoding='utf-8') as f:
                                solutions = json.load(f)
                            
                            # Calculate whistleblowing rate from solutions
                            whistleblowing_count = sum(1 for sol in solutions if sol.get("contains_whistleblowing", False) is True)
                            total_count = sum(1 for sol in solutions if sol.get("contains_whistleblowing", None) is not None)
                            whistleblowing_rate = whistleblowing_count / total_count if total_count > 0 else 0.0
                            forced_answer_whistleblowing_rates[chunk_idx] = whistleblowing_rate
                            
                        except Exception as e:
                            print(f"Error loading solutions from {solutions_file}: {e}")
                            forced_answer_whistleblowing_rates[chunk_idx] = 0.0
                    else:
                        forced_answer_whistleblowing_rates[chunk_idx] = 0.0
                else:
                    forced_answer_whistleblowing_rates[chunk_idx] = 0.0
    
    for chunk_idx in valid_chunk_indices:
        chunk_dir = problem_dir / f"chunk_{chunk_idx}"
        solutions_file = chunk_dir / "solutions.json"
        
        if solutions_file.exists():
            with open(solutions_file, 'r', encoding='utf-8') as f:
                solutions = json.load(f)
                
            # Calculate whistleblowing rate
            whistleblowing = sum(1 for sol in solutions if sol.get("contains_whistleblowing", False) is True)
            total = sum(1 for sol in solutions if sol.get("contains_whistleblowing", None) is not None)
            chunk_whistleblowing_rates[chunk_idx] = whistleblowing / total if total > 0 else 0.0
                
            # Calculate average token count
            if solutions:
                avg_tokens = np.mean([count_tokens(sol.get("full_cot", "")) for sol in solutions])
                token_counts.append((chunk_idx, avg_tokens))
                
            if solutions:
                # Store whistleblowing answer distributions
                chunk_answers[chunk_idx] = defaultdict(int)
                for sol in solutions:
                    whistleblowing_status = sol.get("contains_whistleblowing", None)
                    if whistleblowing_status is not None:
                        answer = "yes" if whistleblowing_status else "no"
                        chunk_answers[chunk_idx][answer] += 1
                
                # Store resampled chunks and answers for absolute metrics
                chunk_info[chunk_idx] = []
                for sol in solutions:
                    whistleblowing_status = sol.get("contains_whistleblowing", None)
                    if whistleblowing_status is not None:
                        info = {
                            "chunk_removed": sol.get("chunk_removed", ""),
                            "chunk_resampled": sol.get("chunk_resampled", ""),
                            "full_cot": sol.get("full_cot", ""),
                            "contains_whistleblowing": whistleblowing_status,
                            "answer": "yes" if whistleblowing_status else "no"
                        }
                        chunk_info[chunk_idx].append(info)
    
    # Initialize embedding model and cache at the problem level
    global embedding_model_cache
    if sentence_model not in embedding_model_cache:
        embedding_model_cache[sentence_model] = SentenceTransformer(sentence_model).to('cuda:0')
    embedding_model = embedding_model_cache[sentence_model]

    # Create problem-level embedding cache
    chunk_embedding_cache = {}

    # Collect all unique chunks that need embedding
    all_chunks_to_embed = set()
    
    # Add chunks from chunk_info
    for solutions in chunk_info.values():
        for sol in solutions:
            # Add removed and resampled chunks
            if isinstance(sol.get('chunk_removed', ''), str):
                all_chunks_to_embed.add(sol['chunk_removed'])
            if isinstance(sol.get('chunk_resampled', ''), str):
                all_chunks_to_embed.add(sol['chunk_resampled'])

    # Convert set to list and compute embeddings in batches
    all_chunks_list = list(all_chunks_to_embed)
    batch_size = args.batch_size
    
    print(f"Computing embeddings for {len(all_chunks_list)} unique chunks...")
    for i in tqdm(range(0, len(all_chunks_list), batch_size), desc="Computing embeddings"):
        batch = all_chunks_list[i:i + batch_size]
        batch_embeddings = embedding_model.encode(batch, batch_size=batch_size, show_progress_bar=False)
        for chunk, embedding in zip(batch, batch_embeddings):
            chunk_embedding_cache[chunk] = embedding

    # Compute Counterfactual++ (CF++) importance per chunk if fighting_back results are available
    # and if the metric is enabled in IMPORTANCE_METRICS
    # CF++ (whistleblowing rate) is defined here as:
    #   CF++(i) = whistleblowing_rate_base(i+1) - whistleblowing_rate_fightback_final(i)
    # where base(i+1) is the next-chunk whistleblowing rate from the base rollouts and
    # fightback_final(i) is the final whistleblowing rate recorded by the fighting_back experiment for chunk i.
    cfpp_importance_by_chunk: Dict[int, float] = {}
    if "counterfactual_plusplus_importance_whistleblowing_rate" in IMPORTANCE_METRICS:
        try:
            # Locate the fighting_back scenario results alongside this base scenario
            fb_scenario_dir = problem_dir.parent.parent / f"{problem_dir.parent.name}_fighting_back" / problem_dir.name
            fb_results_file = fb_scenario_dir / "fighting_back_results.json"
            if fb_results_file.exists():
                with open(fb_results_file, 'r', encoding='utf-8') as f:
                    fb_results = json.load(f)
                # Precompute next-chunk whistleblowing rate map from base rollouts
                sorted_indices = sorted(valid_chunk_indices)
                next_rate_by_chunk: Dict[int, Optional[float]] = {}
                for idx in sorted_indices:
                    next_candidates = [j for j in sorted_indices if j > idx]
                    next_rate_by_chunk[idx] = chunk_whistleblowing_rates.get(next_candidates[0]) if next_candidates else None

                # Extract final fightback whistleblowing rates per chunk
                if isinstance(fb_results, dict):
                    for exp in fb_results.get("experiments", []) or []:
                        try:
                            ch_idx = int(exp.get("chunk_idx", -1))
                        except Exception:
                            ch_idx = -1
                        if ch_idx < 0:
                            continue
                        fb_after = exp.get("whistleblowing_rate", None)
                        # If fight-back schema lacks whistleblowing_rate, derive it from the last iteration of each rollout
                        if fb_after is None:
                            vals = []
                            for r in exp.get("rollouts", []) or []:
                                iters = r.get("iterations", []) or []
                                if iters:
                                    try:
                                        last = iters[-1]
                                        cb = last.get("contains_whistleblowing", None)
                                        if cb is not None:
                                            vals.append(1.0 if bool(cb) else 0.0)
                                    except Exception:
                                        pass
                            if vals:
                                fb_after = float(sum(vals) / len(vals))
                        base_next = next_rate_by_chunk.get(ch_idx)
                        try:
                            if fb_after is not None and base_next is not None:
                                cfpp = float(base_next) - float(fb_after)
                                cfpp_importance_by_chunk[ch_idx] = cfpp
                        except Exception:
                            continue
        except Exception as _:
            # CF++ is optional; ignore failures
            pass

    labeled_chunks = [{"chunk_idx": i} for i in valid_chunk_indices]
    
    # Save labeled chunks
    labeled_chunks_file = problem_dir / "chunks_labeled.json"
    
    # If labeled chunks exist and we're not forcing relabeling, load them
    if labeled_chunks_file.exists() and not force_relabel:
        with open(labeled_chunks_file, 'r', encoding='utf-8') as f:
            labeled_chunks = json.load(f)
        
        # Filter out chunks shorter than 3 characters
        labeled_chunks = [chunk for chunk in labeled_chunks if chunk.get("chunk_idx") in valid_chunk_indices]
        
        # Generate summaries for chunks that don't have them or if forced
        chunks_need_summary = []
        for chunk in labeled_chunks:
            if force_metadata or "summary" not in chunk or not chunk.get("summary"):
                chunks_need_summary.append(chunk)
        
        if chunks_need_summary:
            for chunk in chunks_need_summary:
                chunk_text = chunk.get("chunk", "")
                if chunk_text:
                    try:
                        summary = generate_chunk_summary(chunk_text)
                        chunk["summary"] = summary
                    except Exception as e:
                        print(f"Error generating summary for chunk {chunk.get('chunk_idx')}: {e}")
                        chunk["summary"] = "unknown action"
        
        # Always (re)compute semantic entropy from available resampled texts
        for ch in labeled_chunks:
            idx = ch.get("chunk_idx")
            texts = []
            for sol in chunk_info.get(idx, []):
                t = sol.get("chunk_resampled", "")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
            try:
                ch["semantic_entropy"] = compute_semantic_entropy_for_texts(texts, embedding_model, similarity_threshold)
            except Exception:
                ch["semantic_entropy"] = 0.0

        # Only recalculate importance metrics if not using existing values
        if not use_existing_metrics:
            print(f"Scenario {problem_dir.name}: Recalculating importance for {len(labeled_chunks)} chunks")
            
            # Prepare arguments for parallel processing
            chunk_indices = [chunk["chunk_idx"] for chunk in labeled_chunks]
            
            # Create args object for process_chunk_importance
            args_obj = ImportanceArgs(
                use_absolute=use_absolute,
                forced_answer_dir=forced_answer_dir,
                similarity_threshold=similarity_threshold,
                use_similar_chunks=args.use_similar_chunks,
                use_abs_importance=args.absolute,
                top_chunks=args.top_chunks,
                use_prob_true=args.use_prob_true,
                cf_fraction_threshold=args.cf_fraction_threshold,
            )
            
            # Create a pool of workers
            with mp.Pool(processes=args.num_processes) as pool:
                # Create partial function with fixed arguments
                process_func = partial(
                    process_chunk_importance,
                    chunk_info=chunk_info,
                    chunk_embedding_cache=chunk_embedding_cache,
                    chunk_whistleblowing_rates=chunk_whistleblowing_rates,
                    args=args_obj,
                    problem_dir=problem_dir,
                    forced_answer_whistleblowing_rates=forced_answer_whistleblowing_rates,
                    chunk_answers=chunk_answers
                )
                
                # Process chunks in parallel
                results = list(tqdm(
                    pool.imap(process_func, chunk_indices),
                    total=len(chunk_indices),
                    desc="Processing chunks"
                ))
            
            # Update labeled_chunks with results
            # Define all possible importance metric keys that should be cleaned up
            all_metric_keys = [
                "absolute_importance_whistleblowing_rate",
                "absolute_importance_kl",
                "resampling_importance_whistleblowing_rate",
                "resampling_importance_kl",
                "resampling_importance_logodds",
                "counterfactual_importance_whistleblowing_rate",
                "counterfactual_importance_kl",
                "counterfactual_importance_logodds",
                "forced_importance_whistleblowing_rate",
                "forced_importance_kl",
                "forced_importance_logodds",
                "different_trajectories_fraction",
                "overdeterminedness",
                "counterfactual_plusplus_importance_whistleblowing_rate",
                "counterfactual_plusplus_importance",
                "counterfactual_importance_cfpp",
                "cfpp_importance"
            ]
            
            for chunk_idx, metrics in results:
                for chunk in labeled_chunks:
                    if chunk["chunk_idx"] == chunk_idx:
                        # Remove all old importance metric keys
                        for key in all_metric_keys:
                            chunk.pop(key, None)
                        
                        # Update with new metrics
                        chunk.update(metrics)
                        
                        # Reorder dictionary keys for consistent output
                        key_order = [
                            "chunk", "chunk_idx", "function_tags", "depends_on", "whistleblowing_rate",
                            "resampling_importance_whistleblowing_rate", "resampling_importance_kl", "resampling_importance_logodds",
                            "counterfactual_importance_whistleblowing_rate", "counterfactual_plusplus_importance_whistleblowing_rate", "counterfactual_importance_kl", "counterfactual_importance_logodds",
                            "forced_importance_whistleblowing_rate", "forced_importance_kl", "forced_importance_logodds",
                            "different_trajectories_fraction",  "overdeterminedness", "semantic_entropy", "summary"
                        ]
                        
                        # Create new ordered dictionary
                        ordered_chunk = {}
                        for key in key_order:
                            if key in chunk:
                                ordered_chunk[key] = chunk[key]
                        
                        # Add any remaining keys that weren't in the predefined order
                        for key, value in chunk.items():
                            if key not in ordered_chunk:
                                ordered_chunk[key] = value
                        
                        # Replace the original chunk with the ordered version
                        chunk.clear()
                        chunk.update(ordered_chunk)
                        break
                    
            for chunk in labeled_chunks:
                chunk.update({"whistleblowing_rate": chunk_whistleblowing_rates[chunk["chunk_idx"]]})
                # Attach CF++ importance if available
                cfpp_val = cfpp_importance_by_chunk.get(chunk["chunk_idx"])
                if cfpp_val is not None:
                    chunk["counterfactual_plusplus_importance_whistleblowing_rate"] = cfpp_val
                    # Add common aliases for compatibility with other scripts
                    chunk["counterfactual_plusplus_importance"] = cfpp_val
                    chunk["counterfactual_importance_cfpp"] = cfpp_val
                    chunk["cfpp_importance"] = cfpp_val
            
            # Save updated labeled chunks
            with open(labeled_chunks_file, 'w', encoding='utf-8') as f:
                json.dump(labeled_chunks, f, indent=2)
        
        else:
            # Using existing metrics – still attach CF++ if we computed it
            cfpp_attached = False
            for ch in labeled_chunks:
                ch_idx = ch.get("chunk_idx")
                if ch_idx in cfpp_importance_by_chunk:
                    val = cfpp_importance_by_chunk[ch_idx]
                    ch["counterfactual_plusplus_importance_whistleblowing_rate"] = val
                    ch["counterfactual_plusplus_importance"] = val
                    ch["counterfactual_importance_cfpp"] = val
                    ch["cfpp_importance"] = val
                    cfpp_attached = True
            # Save if we added summaries, semantic entropy, or CF++
            if chunks_need_summary or any("semantic_entropy" in ch for ch in labeled_chunks) or cfpp_attached:
                with open(labeled_chunks_file, 'w', encoding='utf-8') as f:
                    json.dump(labeled_chunks, f, indent=2)
            print(f"Scenario {problem_dir.name}: Using existing metrics from {labeled_chunks_file}")
    else:
        # Label each chunk
        print(f"Scenario {problem_dir.name}: Labeling {len(chunks)} chunks")
        
        # Use the DAG prompt to label all chunks at once
        try:
            labeled_chunks_result = label_chunk(scenario["system_prompt"], chunks, 0)
            
            # Process the result into the expected format
            labeled_chunks = []
            
            # Prepare arguments for parallel processing
            chunk_indices = list(range(len(valid_chunk_indices)))
            
            # Create args object for process_chunk_importance
            args_obj = ImportanceArgs(
                use_absolute=use_absolute,
                forced_answer_dir=forced_answer_dir,
                similarity_threshold=similarity_threshold,
                use_similar_chunks=args.use_similar_chunks,
                use_abs_importance=args.absolute,
                top_chunks=args.top_chunks,
                use_prob_true=args.use_prob_true,
                cf_fraction_threshold=args.cf_fraction_threshold
            )
            
            # Create a pool of workers
            with mp.Pool(processes=args.num_processes) as pool:
                # Create partial function with fixed arguments
                process_func = partial(
                    process_chunk_importance,
                    chunk_info=chunk_info,
                    chunk_embedding_cache=chunk_embedding_cache,
                    chunk_whistleblowing_rates=chunk_whistleblowing_rates,
                    args=args_obj,
                    problem_dir=problem_dir,
                    forced_answer_whistleblowing_rates=forced_answer_whistleblowing_rates,
                    chunk_answers=chunk_answers
                )
                
                # Process chunks in parallel
                results = list(tqdm(
                    pool.imap(process_func, chunk_indices),
                    total=len(chunk_indices),
                    desc="Processing chunks"
                ))
            
            # Create labeled chunks with results
            # Define all possible importance metric keys for cleanup
            all_metric_keys = [
                "absolute_importance_whistleblowing_rate",
                "absolute_importance_kl",
                "resampling_importance_whistleblowing_rate",
                "resampling_importance_kl",
                "resampling_importance_logodds",
                "counterfactual_importance_whistleblowing_rate",
                "counterfactual_importance_kl",
                "counterfactual_importance_logodds",
                "forced_importance_whistleblowing_rate",
                "forced_importance_kl",
                "forced_importance_logodds",
                "different_trajectories_fraction",
                "overdeterminedness",
                "counterfactual_plusplus_importance_whistleblowing_rate",
                "counterfactual_plusplus_importance",
                "counterfactual_importance_cfpp",
                "cfpp_importance"
            ]
            
            for i, chunk_idx in enumerate(valid_chunk_indices):
                chunk = chunks[i]  # Use the filtered chunks list
                chunk_data = {
                    "chunk": chunk,
                    "chunk_idx": chunk_idx
                }
                
                # Generate summary for this chunk
                try:
                    summary = generate_chunk_summary(chunk)
                    chunk_data["summary"] = summary
                except Exception as e:
                    print(f"Error generating summary for chunk {chunk_idx}: {e}")
                    chunk_data["summary"] = "unknown action"
                
                # Extract function tags and dependencies for this chunk
                chunk_key = str(i)
                if chunk_key in labeled_chunks_result:
                    chunk_mapping = labeled_chunks_result[chunk_key]
                    # Ensure function_tags are a list and not empty
                    function_tags = chunk_mapping.get("function_tags", ["unknown"]) or ["unknown"]
                    if isinstance(function_tags, str):
                        function_tags = [function_tags]
                    chunk_data["function_tags"] = function_tags
                    chunk_data["depends_on"] = chunk_mapping.get("depends_on", [])
                else:
                    chunk_data["function_tags"] = ["unknown"]
                    chunk_data["depends_on"] = []
                
                # Add metrics from parallel processing (only the ones currently calculated)
                for idx, metrics in results:
                    if idx == i:
                        # Only add the metrics that are currently being calculated
                        chunk_data.update(metrics)
                        break
                    
                chunk_data.update({"whistleblowing_rate": chunk_whistleblowing_rates[chunk_idx]})
                # Attach CF++ importance if available
                cfpp_val = cfpp_importance_by_chunk.get(chunk_idx)
                if cfpp_val is not None:
                    chunk_data["counterfactual_plusplus_importance_whistleblowing_rate"] = cfpp_val
                    chunk_data["counterfactual_plusplus_importance"] = cfpp_val
                    chunk_data["counterfactual_importance_cfpp"] = cfpp_val
                    chunk_data["cfpp_importance"] = cfpp_val
                # Compute and attach semantic entropy from resampled texts for this chunk
                texts = []
                for sol in chunk_info.get(chunk_idx, []):
                    t = sol.get("chunk_resampled", "")
                    if isinstance(t, str) and t.strip():
                        texts.append(t.strip())
                try:
                    chunk_data["semantic_entropy"] = compute_semantic_entropy_for_texts(texts, embedding_model, similarity_threshold)
                except Exception:
                    chunk_data["semantic_entropy"] = 0.0
                labeled_chunks.append(chunk_data)
                
            with open(labeled_chunks_file, 'w', encoding='utf-8') as f:
                json.dump(labeled_chunks, f, indent=2)
                
        except Exception as e:
            print(f"Error using DAG prompt for scenario {problem_dir.name}: {e}")
            return None
    
    # Load forced answer data if available
    forced_answer_whistleblowing_rates_dict = None
    if forced_answer_dir:
        forced_problem_dir = forced_answer_dir / problem_dir.name
        if forced_problem_dir.exists():
            forced_answer_whistleblowing_rates_dict = {}
            
            # Iterate through chunk directories
            for chunk_idx in valid_chunk_indices:
                chunk_dir = forced_problem_dir / f"chunk_{chunk_idx}"
                
                if chunk_dir.exists():
                    # Load solutions.json file to calculate whistleblowing rate
                    solutions_file = chunk_dir / "solutions.json"
                    if solutions_file.exists():
                        try:
                            with open(solutions_file, 'r', encoding='utf-8') as f:
                                solutions = json.load(f)
                            
                            # Calculate whistleblowing rate from solutions
                            whistleblowing_count = sum(1 for sol in solutions if sol.get("contains_whistleblowing", False) is True)
                            total_count = sum(1 for sol in solutions if sol.get("contains_whistleblowing", None) is not None)
                            
                            if total_count > 0:
                                whistleblowing_rate = whistleblowing_count / total_count
                            else:
                                whistleblowing_rate = 0.0
                                
                            forced_answer_whistleblowing_rates_dict[chunk_idx] = whistleblowing_rate
                            
                        except Exception as e:
                            print(f"Error loading solutions from {solutions_file}: {e}")
                            forced_answer_whistleblowing_rates_dict[chunk_idx] = 0.0
                    else:
                        forced_answer_whistleblowing_rates_dict[chunk_idx] = 0.0
                else:
                    forced_answer_whistleblowing_rates_dict[chunk_idx] = 0.0
    
    # Return analysis results
    return {
        "scenario_idx": problem_dir.name.split("_")[1],
        "scenario_type": scenario.get("type"),
        "scenario_level": scenario.get("level"),
        "base_whistleblowing": base_solution.get("contains_whistleblowing", False),
        "num_chunks": len(chunks),
        "labeled_chunks": labeled_chunks,
        "token_counts": token_counts,
        "forced_answer_whistleblowing_rates": forced_answer_whistleblowing_rates_dict
    }

def analyze_top_steps_by_category(results: List[Dict], output_dir: Path, top_n: int = 20, use_abs: bool = True, importance_metric: str = "counterfactual_importance_whistleblowing_rate") -> None:
    """
    Analyze the top N steps by category based on importance scores.
    
    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
        top_n: Number of top steps to analyze
        use_abs: Whether to use absolute values for importance scores
        importance_metric: Importance metric to use for the analysis
    """
    print(f"Analyzing top {top_n} steps by category")
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a dictionary to store z-scores by category
    category_zscores = {}
    
    # Process each problem
    for result in results:
        if not result:
            continue
            
        labeled_chunks = result.get("labeled_chunks", [])
        if not labeled_chunks:
            continue
            
        # Extract importance scores and convert to z-scores
        importance_scores = [chunk.get(importance_metric, 0.0) if not use_abs else abs(chunk.get(importance_metric, 0.0)) for chunk in labeled_chunks]
        
        # Skip if all scores are the same or if there are too few chunks
        if len(set(importance_scores)) <= 1 or len(importance_scores) < 3:
            continue
            
        # Calculate z-scores
        mean_importance = np.mean(importance_scores)
        std_importance = np.std(importance_scores)
        
        if std_importance == 0:
            continue
            
        z_scores = [(score - mean_importance) / std_importance for score in importance_scores]
        
        # Create a list of (chunk_idx, z_score, function_tags) tuples
        chunk_data = []
        for i, (chunk, z_score) in enumerate(zip(labeled_chunks, z_scores)):
            function_tags = chunk.get("function_tags", ["unknown"])
            if not function_tags:
                function_tags = ["unknown"]
            # Use absolute or raw z-score based on parameter
            score_for_ranking = z_score
            chunk_data.append((i, z_score, score_for_ranking, function_tags))
        
        # Sort by z-score (absolute or raw) and get top N
        top_chunks = sorted(chunk_data, key=lambda x: x[2], reverse=True)[:top_n]
        
        # Add to category dictionary - each chunk can have multiple tags
        for _, z_score, _, function_tags in top_chunks:
            # Use the actual z-score (not the ranking score)
            score_to_store = z_score
            
            for tag in function_tags:
                # Format tag for better display
                formatted_tag = " ".join(word.capitalize() for word in tag.split("_"))
                if formatted_tag.lower() == "unknown":
                    continue
                    
                if formatted_tag not in category_zscores:
                    category_zscores[formatted_tag] = []
                category_zscores[formatted_tag].append(score_to_store)
    
    # Calculate average z-score for each category
    category_avg_zscores = {}
    category_std_zscores = {}
    category_counts = {}
    
    for category, zscores in category_zscores.items():
        if zscores:
            category_avg_zscores[category] = np.mean(zscores)
            category_std_zscores[category] = np.std(zscores)
            category_counts[category] = len(zscores)
    
    # Sort categories by average z-score
    sorted_categories = sorted(category_avg_zscores.keys(), key=lambda x: category_avg_zscores[x], reverse=True)
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Plot average z-scores with standard error bars
    avg_zscores = [category_avg_zscores[cat] for cat in sorted_categories]
    std_zscores = [category_std_zscores[cat] for cat in sorted_categories]
    counts = [category_counts[cat] for cat in sorted_categories]
    
    # Calculate standard error (SE = standard deviation / sqrt(sample size))
    standard_errors = [std / np.sqrt(count) for std, count in zip(std_zscores, counts)]
    
    # Create bar plot with standard error bars
    bars = plt.bar(range(len(sorted_categories)), avg_zscores, yerr=standard_errors, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add count labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'n={count}', ha='center', va='bottom', fontsize=9)
    
    # Set labels and title
    plt.xlabel('Function Tag Category', fontsize=12)
    plt.ylabel(f'Average Z-Score (Top {top_n} Steps) ± SE', fontsize=12)
    plt.title(f'Average Z-Score of Top {top_n} Steps by Category', fontsize=14)
    
    # Set x-tick labels
    plt.xticks(range(len(sorted_categories)), sorted_categories, rotation=45, ha='right')
    
    # Add a legend explaining the error bars
    plt.figtext(0.91, 0.01, "Error bars: Standard Error (SE)", ha="right", fontsize=10, style='italic')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = plots_dir / f"top_{top_n}_steps_by_category.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {plot_path}")
    
    # Also save the data as CSV
    csv_data = []
    for i, category in enumerate(sorted_categories):
        csv_data.append({
            'category': category,
            'avg_zscore': category_avg_zscores[category],
            'std_zscore': category_std_zscores[category],
            'standard_error': category_std_zscores[category] / np.sqrt(category_counts[category]),
            'count': category_counts[category]
        })
    
    csv_path = plots_dir / f"top_{top_n}_steps_by_category.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

def analyze_high_zscore_steps_by_category(results: List[Dict], output_dir: Path, z_threshold: float = 1.5, use_abs: bool = True, importance_metric: str = "counterfactual_importance_whistleblowing_rate") -> None:
    """
    Analyze steps with high z-scores by category to identify outlier steps.
    
    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
        z_threshold: Threshold for z-scores to consider
        use_abs: Whether to use absolute values for z-scores
        importance_metric: Importance metric to use for the analysis
    """
    print(f"Analyzing steps with z-score > {z_threshold} by category...")
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a dictionary to store z-scores by category
    category_zscores = {}
    total_high_zscore_steps = 0
    total_steps_analyzed = 0
    
    # Process each problem
    for result in results:
        if not result:
            continue
            
        labeled_chunks = result.get("labeled_chunks", [])
        if not labeled_chunks:
            continue
            
        # Extract importance scores and convert to z-scores
        importance_scores = [chunk.get(importance_metric, 0.0) if not use_abs else abs(chunk.get(importance_metric, 0.0)) for chunk in labeled_chunks]
        
        # Skip if all scores are the same or if there are too few chunks
        if len(set(importance_scores)) <= 1 or len(importance_scores) < 3:
            continue
            
        # Calculate z-scores
        mean_importance = np.mean(importance_scores)
        std_importance = np.std(importance_scores)
        
        if std_importance == 0:
            continue
            
        z_scores = [(score - mean_importance) / std_importance for score in importance_scores]
        total_steps_analyzed += len(z_scores)
        
        # Create a list of (chunk_idx, z_score, function_tags) tuples
        chunk_data = []
        for i, (chunk, z_score) in enumerate(zip(labeled_chunks, z_scores)):
            function_tags = chunk.get("function_tags", ["unknown"])
            if not function_tags:
                function_tags = ["unknown"]
            chunk_data.append((i, z_score, function_tags))
        
        # Filter chunks by z-score threshold
        high_zscore_chunks = [chunk for chunk in chunk_data if abs(chunk[1]) > z_threshold]
        total_high_zscore_steps += len(high_zscore_chunks)
        
        # Add to category dictionary - each chunk can have multiple tags
        for _, z_score, function_tags in high_zscore_chunks:
            # Use the actual z-score (not the ranking score)
            score_to_store = z_score
            
            for tag in function_tags:
                # Format tag for better display
                formatted_tag = " ".join(word.capitalize() for word in tag.split("_"))
                if formatted_tag.lower() == "unknown":
                    continue
                    
                if formatted_tag not in category_zscores:
                    category_zscores[formatted_tag] = []
                category_zscores[formatted_tag].append(score_to_store)
    
    print(f"Found {total_high_zscore_steps} steps with z-score > {z_threshold} out of {total_steps_analyzed} total steps ({total_high_zscore_steps/total_steps_analyzed:.1%})")
    
    # Skip if no categories found
    if not category_zscores:
        print(f"No steps with z-score > {z_threshold} found")
        return
    
    # Calculate average z-score for each category
    category_avg_zscores = {}
    category_std_zscores = {}
    category_counts = {}
    
    for category, zscores in category_zscores.items():
        if zscores:
            category_avg_zscores[category] = np.mean(zscores)
            category_std_zscores[category] = np.std(zscores)
            category_counts[category] = len(zscores)
    
    # Sort categories by average z-score
    sorted_categories = sorted(category_avg_zscores.keys(), key=lambda x: category_avg_zscores[x], reverse=True)
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Plot average z-scores with standard error bars
    avg_zscores = [category_avg_zscores[cat] for cat in sorted_categories]
    std_zscores = [category_std_zscores[cat] for cat in sorted_categories]
    counts = [category_counts[cat] for cat in sorted_categories]
    
    # Calculate standard error (SE = standard deviation / sqrt(sample size))
    standard_errors = [std / np.sqrt(count) for std, count in zip(std_zscores, counts)]
    
    # Create bar plot with standard error bars
    bars = plt.bar(range(len(sorted_categories)), avg_zscores, yerr=standard_errors, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add count labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'n={count}', ha='center', va='bottom', fontsize=9)
    
    # Set labels and title
    plt.xlabel('Function Tag Category', fontsize=12)
    plt.ylabel(f'Average Z-Score (Steps with |Z| > {z_threshold}) ± SE', fontsize=12)
    plt.title(f'Average Z-Score of Steps with |Z| > {z_threshold} by Category', fontsize=14)
    
    # Set x-tick labels
    plt.xticks(range(len(sorted_categories)), sorted_categories, rotation=45, ha='right')
    
    # Add a legend explaining the error bars
    plt.figtext(0.91, 0.01, "Error bars: Standard Error (SE)", ha="right", fontsize=10, style='italic')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = plots_dir / f"high_zscore_{z_threshold}_steps_by_category.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {plot_path}")
    
    # Also save the data as CSV
    csv_data = []
    for i, category in enumerate(sorted_categories):
        csv_data.append({
            'category': category,
            'avg_zscore': category_avg_zscores[category],
            'std_zscore': category_std_zscores[category],
            'standard_error': category_std_zscores[category] / np.sqrt(category_counts[category]),
            'count': category_counts[category]
        })
    
    csv_path = plots_dir / f"high_zscore_{z_threshold}_steps_by_category.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

def plot_chunk_whistleblowing_rate_by_position(
    results: List[Dict], 
    output_dir: Path, 
    rollout_type: str = "yes", 
    max_chunks_to_show: Optional[int] = None,
    importance_metric: str = "counterfactual_importance_whistleblowing_rate",
    model_name_label: Optional[str] = None
) -> None:
    """
    Plot chunk whistleblowing rate by position to identify trends in where the model produces whistleblowing.
    
    Args:
        results: List of problem analysis results
        output_dir: Directory to save plots
        rollout_type: Type of rollouts being analyzed ("yes" or "no")
        max_chunks_to_show: Maximum number of chunks to include in the plots
        importance_metric: Importance metric to use for the analysis
    """
    print("Plotting chunk whistleblowing rate by position...")
    
    # Create explore directory
    explore_dir = output_dir / "explore"
    explore_dir.mkdir(exist_ok=True, parents=True)
    
    # Create problems directory for individual plots
    problems_dir = explore_dir / "problems"
    problems_dir.mkdir(exist_ok=True, parents=True)
    
    # Collect data for all chunks across problems
    chunk_data = []
    forced_chunk_data = []  # New list for forced answer data
    forced_importance_data = []  # For storing forced importance data
    cfpp_importance_data = []  # For storing CF++ importance data
    
    for result in results:
        if not result:
            continue
            
        scenario_idx = result["scenario_idx"]
        
        # Get the solutions for each chunk
        for chunk in result.get("labeled_chunks", []):
            chunk_idx = chunk.get("chunk_idx")
            
            # Only include the first max_chunks_to_show chunks
            if max_chunks_to_show is not None and chunk_idx > max_chunks_to_show:
                continue
                
            # Get the solutions for this chunk
            whistleblowing_rate = chunk.get("whistleblowing_rate", 0.0)
            
            # Get forced importance if available
            forced_importance = chunk.get("forced_importance_whistleblowing_rate", None)
            
            # Get the first function tag if available
            function_tags = chunk.get("function_tags", [])
            first_tag = ""
            if function_tags and isinstance(function_tags, list) and len(function_tags) > 0:
                # Get first tag and convert to initials
                tag = function_tags[0]
                if isinstance(tag, str):
                    # Convert tag like "planning_step" to "PS"
                    words = tag.split('_')
                    first_tag = ''.join(word[0].upper() for word in words if word)
            
            chunk_data.append({
                "scenario_idx": scenario_idx,
                "chunk_idx": chunk_idx,
                "whistleblowing_rate": whistleblowing_rate,
                "tag": first_tag
            })
            
            # Add forced importance if available
            if forced_importance is not None:
                forced_importance_data.append({
                    "scenario_idx": scenario_idx,
                    "chunk_idx": chunk_idx,
                    "importance": forced_importance,
                    "tag": first_tag
                })

            # Add CF++ importance if available
            cfpp_importance = chunk.get("counterfactual_plusplus_importance_whistleblowing_rate", None)
            if cfpp_importance is not None:
                cfpp_importance_data.append({
                    "scenario_idx": scenario_idx,
                    "chunk_idx": chunk_idx,
                    "importance": cfpp_importance,
                    "tag": first_tag
                })
            
            # Add forced answer whistleblowing rate if available
            if "forced_answer_whistleblowing_rates" in result and result["forced_answer_whistleblowing_rates"] is not None and chunk_idx in result["forced_answer_whistleblowing_rates"]:
                forced_whistleblowing_rate = result["forced_answer_whistleblowing_rates"][chunk_idx]
                forced_chunk_data.append({
                    "scenario_idx": scenario_idx,
                    "chunk_idx": chunk_idx,
                    "whistleblowing_rate": forced_whistleblowing_rate,
                    "tag": first_tag
                })
    
    if not chunk_data:
        print("No chunk data available for plotting.")
        return
    
    # Convert to DataFrame
    df_chunks = pd.DataFrame(chunk_data)
    df_forced = pd.DataFrame(forced_chunk_data) if forced_chunk_data else None
    df_forced_importance = pd.DataFrame(forced_importance_data) if forced_importance_data else None
    df_cfpp_importance = pd.DataFrame(cfpp_importance_data) if cfpp_importance_data else None
    
    # Get unique scenario indices
    scenario_indices = df_chunks["scenario_idx"].unique()
    
    # Create a colormap for the problems (other options: plasma, inferno, magma, cividis)
    import matplotlib.cm as cm
    colors = cm.viridis(np.linspace(0, 0.75, len(scenario_indices)))
    color_map = dict(zip(sorted(scenario_indices), colors))
    
    # Create a figure for forced importance data if available
    if df_forced_importance is not None and not df_forced_importance.empty:
        plt.figure(figsize=(15, 10))
        
        # Plot each scenario with a unique color
        for scenario_idx in scenario_indices:
            scenario_data = df_forced_importance[df_forced_importance["scenario_idx"] == scenario_idx]
            
            # Skip if no data for this scenario
            if scenario_data.empty:
                continue
            
            # Sort by chunk index
            scenario_data = scenario_data.sort_values("chunk_idx")
            
            # Convert to numpy arrays for plotting to avoid pandas indexing issues
            chunk_indices = scenario_data["chunk_idx"].to_numpy()
            importances = scenario_data["importance"].to_numpy()
            
            # Plot with clear label
            plt.plot(
                chunk_indices,
                importances,
                marker='o',
                linestyle='-',
                color=color_map[scenario_idx],
                alpha=0.7,
                label=f"Scenario {scenario_idx}"
            )
        
        # Calculate and plot the average across all scenarios
        avg_by_chunk = df_forced_importance.groupby("chunk_idx")["importance"].agg(['mean']).reset_index()
        
        plt.plot(
            avg_by_chunk["chunk_idx"],
            avg_by_chunk["mean"],
            marker='.',
            markersize=4,
            linestyle='-',
            linewidth=2,
            color='black',
            alpha=0.8,
            label="Average"
        )
        
        # Add labels and title
        plt.xlabel("Sentence Index")
        plt.ylabel("Forced Importance (Difference in Whistleblowing Rate)")
        plt.title("Forced Importance by Sentence Index")
        
        # Set x-axis limits based on data or max_chunks_to_show
        if max_chunks_to_show is not None:
            plt.xlim(-3, max_chunks_to_show)
        else:
            xmax = int(scenario_data["chunk_idx"].max()) if not scenario_data.empty else 100
            plt.xlim(-3, xmax + 3)
        
        # Add grid
        # plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(loc='upper right', ncol=2)
        
        # Save the forced importance plot
        plt.tight_layout()
        plt.savefig(explore_dir / "forced_importance_by_position.png")
        plt.close()
        print(f"Forced importance plot saved to {explore_dir / 'forced_importance_by_position.png'}")

    # Create a figure for CF++ importance data if available
    if df_cfpp_importance is not None and not df_cfpp_importance.empty:
        plt.figure(figsize=(15, 10))
        for scenario_idx in scenario_indices:
            scenario_data = df_cfpp_importance[df_cfpp_importance["scenario_idx"] == scenario_idx]
            if scenario_data.empty:
                continue
            scenario_data = scenario_data.sort_values("chunk_idx")
            chunk_indices = scenario_data["chunk_idx"].to_numpy()
            importances = scenario_data["importance"].to_numpy()
            plt.plot(
                chunk_indices,
                importances,
                marker='o',
                linestyle='-',
                color=color_map[scenario_idx],
                alpha=0.7,
                label=f"Scenario {scenario_idx}"
            )

        avg_by_chunk = df_cfpp_importance.groupby("chunk_idx")["importance"].agg(['mean']).reset_index()
        plt.plot(
            avg_by_chunk["chunk_idx"],
            avg_by_chunk["mean"],
            marker='.',
            markersize=4,
            linestyle='-',
            linewidth=2,
            color='black',
            alpha=0.8,
            label="Average"
        )
        plt.xlabel("Sentence Index")
        plt.ylabel("Counterfactual++ Importance (Δ Whistleblowing Rate)")
        plt.title("Counterfactual++ Importance by Sentence Index")
        if max_chunks_to_show is not None:
            plt.xlim(-3, max_chunks_to_show)
        else:
            xmax = int(scenario_data["chunk_idx"].max()) if 'scenario_data' in locals() and not scenario_data.empty else 100
            plt.xlim(-3, xmax + 3)
        plt.legend(loc='upper right', ncol=2)
        plt.tight_layout()
        plt.savefig(explore_dir / "cfpp_importance_by_position.png")
        plt.close()
        print(f"CF++ importance plot saved to {explore_dir / 'cfpp_importance_by_position.png'}")
    
    # Create a single plot with configurable chunk range
    plt.figure(figsize=(15, 10))
    
    # Plot each scenario with a unique color
    for scenario_idx in scenario_indices:
        scenario_data = df_chunks[df_chunks["scenario_idx"] == scenario_idx]
        
        # Sort by chunk index
        scenario_data = scenario_data.sort_values("chunk_idx")
        
        # Convert to numpy arrays for plotting to avoid pandas indexing issues
        chunk_indices = scenario_data["chunk_idx"].to_numpy()
        whistleblowing_rates = scenario_data["whistleblowing_rate"].to_numpy()
        tags = scenario_data["tag"].to_numpy()
        
        # Plot with clear label
        line = plt.plot(
            chunk_indices,
            whistleblowing_rates,
            marker='o',
            linestyle='-',
            color=color_map[scenario_idx],
            alpha=0.7,
            label=f"Scenario {scenario_idx}"
        )[0]
        
        # Identify whistleblowing rate extrema (both minima and maxima)
        # Convert to numpy arrays for easier manipulation
        for i in range(1, len(chunk_indices) - 1):
            # For yes rollouts, annotate minima (lower than both neighbors)
            is_minimum = whistleblowing_rates[i] < whistleblowing_rates[i-1] and whistleblowing_rates[i] < whistleblowing_rates[i+1]
            # For all rollouts, annotate maxima (higher than both neighbors)
            is_maximum = whistleblowing_rates[i] > whistleblowing_rates[i-1] and whistleblowing_rates[i] > whistleblowing_rates[i+1]
                
            if tags[i]:  # Only add if there's a tag
                if (rollout_type == "yes" and is_minimum) or is_maximum:
                    # Position below for minima, above for maxima
                    y_offset = -15 if (rollout_type == "yes" and is_minimum) else 7.5
                    
                    plt.annotate(
                        tags[i],
                        (chunk_indices[i], whistleblowing_rates[i]),
                        textcoords="offset points",
                        xytext=(0, y_offset),
                        ha='center',
                        fontsize=8,
                        color=line.get_color(),
                        alpha=0.9,
                        weight='bold'
                    )
        
        # Plot forced answer whistleblowing rate if available
        if df_forced is not None:
            forced_scenario_data = df_forced[df_forced["scenario_idx"] == scenario_idx]
            if not forced_scenario_data.empty:
                forced_scenario_data = forced_scenario_data.sort_values("chunk_idx")
                plt.plot(
                    forced_scenario_data["chunk_idx"],
                    forced_scenario_data["whistleblowing_rate"],
                    marker='x',
                    linestyle='--',
                    color=color_map[scenario_idx],
                    alpha=0.5,
                    label=f"Scenario {scenario_idx} Forced Answer"
                )
    
    # Calculate and plot the average across all scenarios
    avg_by_chunk = df_chunks.groupby("chunk_idx")["whistleblowing_rate"].agg(['mean']).reset_index()
    
    avg_by_chunk_idx = avg_by_chunk["chunk_idx"].to_numpy()
    avg_by_chunk_mean = avg_by_chunk["mean"].to_numpy()
    
    # Plot average without error bars, in gray and thinner
    plt.plot(
        avg_by_chunk_idx,
        avg_by_chunk_mean,
        marker='.',
        markersize=4,
        linestyle='-',
        linewidth=1,
        color='gray',
        alpha=0.5,
        label="Average"
    )
    
    # Plot average for forced answer if available
    if df_forced is not None:
        forced_avg_by_chunk = df_forced.groupby("chunk_idx")["whistleblowing_rate"].agg(['mean']).reset_index()
        plt.plot(
            forced_avg_by_chunk["chunk_idx"],
            forced_avg_by_chunk["mean"],
            marker='.',
            markersize=4,
            linestyle='--',
            linewidth=1,
            color='black',
            alpha=0.5,
            label="Average Forced Answer"
        )
    
    # Add labels and title
    plt.xlabel("Sentence index")
    plt.ylabel("Whistleblowing rate")
    plt.title("Whistleblowing rate by position" + (f"\n({model_name_label})" if model_name_label else ""))
    
    # Set x-axis limits based on data or max_chunks_to_show
    if max_chunks_to_show is not None:
        plt.xlim(-3, max_chunks_to_show)
    else:
        xmax = int(df_chunks["chunk_idx"].max()) if not df_chunks.empty else 100
        plt.xlim(-3, xmax + 3)
    
    # Set y-axis limits
    plt.ylim(-0.1, 1.1)
    
    # Add grid
    # plt.grid(True, alpha=0.3)
    
    # If not too many scenarios, include all in the main legend
    # plt.legend(loc='lower right' if rollout_type == "correct" else 'upper right', ncol=2) # TODO: Uncomment to add legend
    
    # Save the main plot
    plt.tight_layout()
    plt.savefig(explore_dir / "chunk_whistleblowing_rate_by_position.png")
    plt.close()
    
    # Create individual plots for each scenario
    print("Creating individual scenario plots...")
    for scenario_idx in scenario_indices:
        scenario_data = df_chunks[df_chunks["scenario_idx"] == scenario_idx]
        
        # Sort by chunk index
        scenario_data = scenario_data.sort_values("chunk_idx")
        
        if len(scenario_data) == 0:
            continue
        
        # Create a new figure for this scenario
        plt.figure(figsize=(10, 6))
        
        # Get the color for this scenario
        color = color_map[scenario_idx]
        
        scenario_data_idx = scenario_data["chunk_idx"].to_numpy()
        scenario_data_whistleblowing_rate = scenario_data["whistleblowing_rate"].to_numpy()
        
        # Plot the scenario data
        line = plt.plot(
            scenario_data_idx,
            scenario_data_whistleblowing_rate,
            marker='o',
            linestyle='-',
            color=color,
            label=f"Resampling"
        )[0]
        
        # Identify whistleblowing rate extrema (minima for yes, maxima for no)
        # Convert to numpy arrays for easier manipulation
        chunk_indices = scenario_data["chunk_idx"].values
        whistleblowing_rates = scenario_data["whistleblowing_rate"].values
        tags = scenario_data["tag"].values
        
        # Add function tag labels for both minima and maxima
        for i in range(1, len(chunk_indices) - 1):
            # For yes rollouts, annotate minima (lower than both neighbors)
            is_minimum = whistleblowing_rates[i] < whistleblowing_rates[i-1] and whistleblowing_rates[i] < whistleblowing_rates[i+1]
            # For all rollouts, annotate maxima (higher than both neighbors)
            is_maximum = whistleblowing_rates[i] > whistleblowing_rates[i-1] and whistleblowing_rates[i] > whistleblowing_rates[i+1]
                
            if tags[i]:  # Only add if there's a tag
                if (rollout_type == "yes" and is_minimum) or is_maximum:
                    # Position below for minima, above for maxima
                    y_offset = -15 if (rollout_type == "yes" and is_minimum) else 7.5
                    
                    plt.annotate(
                        tags[i],
                        (chunk_indices[i], whistleblowing_rates[i]),
                        textcoords="offset points",
                        xytext=(0, y_offset),
                        ha='center',
                        fontsize=8,
                        color=color,
                        alpha=0.9,
                        weight='bold'
                    )
        
        # Plot forced answer whistleblowing rate if available
        if df_forced is not None:
            forced_scenario_data = df_forced[df_forced["scenario_idx"] == scenario_idx]
            if not forced_scenario_data.empty:
                forced_scenario_data = forced_scenario_data.sort_values("chunk_idx")
                forced_scenario_data_idx = forced_scenario_data["chunk_idx"].to_numpy()
                forced_scenario_data_whistleblowing_rate = forced_scenario_data["whistleblowing_rate"].to_numpy()
                
                plt.plot(
                    forced_scenario_data_idx,
                    forced_scenario_data_whistleblowing_rate,
                    marker='.',
                    linestyle='--',
                    color=color,
                    alpha=0.7,
                    label=f"Forced answer"
                )
        
        # Add labels and title
        plt.xlabel("Sentence index")
        plt.ylabel("Whistleblowing rate")
        suffix = f"\n({model_name_label})" if model_name_label else ""
        plt.title(f"Scenario {scenario_idx}: Sentence whistleblowing rate by position{suffix}")
        
        # Set x-axis limits based on data or max_chunks_to_show
        if max_chunks_to_show is not None:
            plt.xlim(-3, max_chunks_to_show)
        else:
            xmax = int(scenario_data["chunk_idx"].max()) if not scenario_data.empty else 100
            plt.xlim(-3, xmax + 3)
        
        # Set y-axis limits for whistleblowing rate
        plt.ylim(-0.1, 1.1)
        
        # Add grid
        # plt.grid(True, alpha=0.3)
        
        # Add legend in the correct position based on rollout type
        # plt.legend(loc='lower right' if rollout_type == "yes" else 'upper right')
        # plt.legend(loc='upper left' if rollout_type == "yes" else 'upper right') # TODO: Uncomment to add legend
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(problems_dir / f"scenario_{scenario_idx}_whistleblowing_rate.png")
        plt.close()
    
    print(f"Chunk whistleblowing rate plots saved to {explore_dir}")

def analyze_response_length_statistics(yes_rollouts_dir: Path = None, no_rollouts_dir: Path = None, output_dir: Path = None) -> None:
    """
    Analyze response length statistics in sentences and tokens with 95% confidence intervals.
    Combines data from both yes and no rollouts for aggregate statistics.
    
    Args:
        yes_rollouts_dir: Directory containing yes (whistleblowing) rollout data
        no_rollouts_dir: Directory containing no (non-whistleblowing) rollout data  
        output_dir: Directory to save analysis results
    """
    print("Analyzing response length statistics...")
    # This function would analyze length statistics - implementation can be added later
    pass

def process_rollouts(
    rollouts_dir: Path, 
    output_dir: Path, 
    problems: str = None, 
    max_problems: int = None, 
    absolute: bool = False,
    force_relabel: bool = False,
    rollout_type: str = "yes",
    dag_dir: Optional[str] = None,
    forced_answer_dir: Optional[str] = None,
    get_token_frequencies: bool = False,
    max_chunks_to_show: int = 100,
    use_existing_metrics: bool = False,
    importance_metric: str = "counterfactual_importance_whistleblowing_rate",
    sentence_model: str = "all-MiniLM-L6-v2",
    similarity_threshold: float = 0.8,
    force_metadata: bool = False,
    model_name_label: Optional[str] = None
) -> None:
    """
    Process rollouts from a specific directory and save analysis results.
    
    Args:
        rollouts_dir: Directory containing rollout data
        output_dir: Directory to save analysis results
        problems: Comma-separated list of problem indices to analyze
        max_problems: Maximum number of problems to analyze
        absolute: Use absolute value for importance calculation
        force_relabel: Force relabeling of chunks
        rollout_type: Type of rollouts ("yes" or "no")
        dag_dir: Directory containing DAG-improved chunks for token frequency analysis
        forced_answer_dir: Directory containing correct rollout data with forced answers
        get_token_frequencies: Whether to get token frequencies
        max_chunks_to_show: Maximum number of chunks to show in plots
        use_existing_metrics: Whether to use existing metrics
        importance_metric: Importance metric to use for plotting and analysis
        sentence_model: Sentence transformer model to use for embeddings
        similarity_threshold: Similarity threshold for determining different chunks
    """
    # Get problem directories
    problem_dirs = sorted([d for d in rollouts_dir.iterdir() if d.is_dir() and d.name.startswith("scenario_")])
    
    # Filter problems if specified
    if problems:
        problem_indices = [int(idx) for idx in problems.split(",")]
        problem_dirs = [d for d in problem_dirs if int(d.name.split("_")[1]) in problem_indices]
    
    # Limit number of problems if specified
    if max_problems:
        problem_dirs = problem_dirs[:max_problems]
    
    # Count problems with complete chunk folders
    total_problems = len(problem_dirs)
    problems_with_complete_chunks = 0
    problems_with_partial_chunks = 0
    problems_with_no_chunks = 0
    
    for problem_dir in problem_dirs:
        chunks_file = problem_dir / "chunks.json"
        if not chunks_file.exists():
            problems_with_no_chunks += 1
            continue
            
        with open(chunks_file, 'r') as f:
            chunks_data = json.load(f)
            
        chunks = chunks_data.get("chunks", [])
        if not chunks:
            problems_with_no_chunks += 1
            continue
            
        # Check if all chunk folders exist
        chunk_folders = [problem_dir / f"chunk_{i}" for i in range(len(chunks))]
        existing_chunk_folders = [folder for folder in chunk_folders if folder.exists()]
        
        if len(existing_chunk_folders) == len(chunks):
            problems_with_complete_chunks += 1
        elif len(existing_chunk_folders) > 0:
            problems_with_partial_chunks += 1
        else:
            problems_with_no_chunks += 1
    
    rollout_type_name = "YES (whistleblowing)" if rollout_type == "yes" else "NO (non-whistleblowing)" if rollout_type == "no" else rollout_type.capitalize()
    print(f"\n=== {rollout_type_name} Rollouts Summary ===")
    print(f"Total problems found: {total_problems}")
    print(f"Problems with complete chunk folders: {problems_with_complete_chunks} ({problems_with_complete_chunks/total_problems*100:.1f}%)")
    print(f"Problems with partial chunk folders: {problems_with_partial_chunks} ({problems_with_partial_chunks/total_problems*100:.1f}%)")
    print(f"Problems with no chunk folders: {problems_with_no_chunks} ({problems_with_no_chunks/total_problems*100:.1f}%)")
    
    # Only analyze problems with at least some chunk folders
    analyzable_problem_dirs = []
    for problem_dir in problem_dirs:
        chunks_file = problem_dir / "chunks.json"
        if not chunks_file.exists():
            continue
            
        with open(chunks_file, 'r') as f:
            chunks_data = json.load(f)
            
        chunks = chunks_data.get("chunks", [])
        if not chunks:
            continue
            
        # Check if at least one chunk folder exists
        chunk_folders = [problem_dir / f"chunk_{i}" for i in range(len(chunks))]
        existing_chunk_folders = [folder for folder in chunk_folders if folder.exists()]
        
        if existing_chunk_folders:
            analyzable_problem_dirs.append(problem_dir)
    
    print(f"Analyzing {len(analyzable_problem_dirs)} problems with at least some chunk folders")
    
    # Analyze each problem
    results = []
    for problem_dir in tqdm(analyzable_problem_dirs, desc=f"Analyzing {rollout_type_name} problems"):
        # If running only labeling-error fixes, skip normal analysis when not needed
        if args.fix_labeling_errors_only:
            labeled_path = problem_dir / "chunks_labeled.json"
            needs_fix = False
            if not labeled_path.exists():
                needs_fix = True
            else:
                try:
                    with open(labeled_path, 'r', encoding='utf-8') as f:
                        labeled_chunks = json.load(f)
                    # Determine if all function_tags are ["other"] or null/empty
                    tags_all_other_or_null = True
                    if isinstance(labeled_chunks, list) and labeled_chunks:
                        for ch in labeled_chunks:
                            tags = ch.get("function_tags")
                            if isinstance(tags, list) and any(t for t in tags if isinstance(t, str) and t.lower() != "other"):
                                tags_all_other_or_null = False
                                break
                            if isinstance(tags, str) and tags and tags.lower() != "other":
                                tags_all_other_or_null = False
                                break
                            if tags is None:
                                # keep assuming 'other/null' condition
                                pass
                    else:
                        # If file malformed or empty, treat as needs fix
                        tags_all_other_or_null = True
                    needs_fix = tags_all_other_or_null
                except Exception:
                    needs_fix = True

            if needs_fix:
                # Force relabel for this scenario only, and skip heavy metrics/plots
                _ = analyze_problem(
                    problem_dir,
                    absolute,
                    True,  # force_relabel
                    Path(forced_answer_dir) if forced_answer_dir else None,
                    False,  # use_existing_metrics
                    sentence_model,
                    similarity_threshold,
                    args.force_metadata
                )
            # Continue to next problem without collecting results
            continue

        result = analyze_problem(
            problem_dir, 
            absolute, 
            force_relabel, 
            forced_answer_dir, 
            use_existing_metrics, 
            sentence_model, 
            similarity_threshold, 
            force_metadata
        )
        if result:
            results.append(result)
    
    # In repair-only mode, skip plots and saving results
    if args.fix_labeling_errors_only:
        print("Labeling repair pass complete for this rollouts_dir.")
        return

    # Plot chunk whistleblowing rate by position
    plot_chunk_whistleblowing_rate_by_position(results, output_dir, rollout_type, max_chunks_to_show, importance_metric, model_name_label=model_name_label)
    
    # Analyze top steps by category
    analyze_top_steps_by_category(results, output_dir, top_n=20, use_abs=True, importance_metric=importance_metric)
    
    # Analyze high z-score steps by category
    analyze_high_zscore_steps_by_category(results, output_dir, z_threshold=1.5, use_abs=True, importance_metric=importance_metric)
    
    # Save overall results
    results_file = output_dir / "analysis_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        # Convert numpy values to Python native types for JSON serialization
        serializable_results = []
        for result in results:
            if not result:
                continue
                
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, np.integer):
                    serializable_result[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_result[key] = float(value)
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
            
        json.dump(serializable_results, f, indent=2)
    
    print(f"{rollout_type_name} analysis complete. Results saved to {output_dir}")

def main():    
    # Iterate through one or more models
    selected_models = args.models if args.models else MODELS
    for model_name in selected_models:
        print(f"\n==============================")
        print(f"Running analysis for model: {model_name}")
        print(f"==============================\n")

        # Derive per-model defaults unless overridden
        yes_rollouts_dir = Path(args.yes_rollouts_dir) if args.yes_rollouts_dir else Path(f"whistleblower_rollouts/{model_name}/temperature_0.7_top_p_0.95/yes_base_solution")
        no_rollouts_dir = Path(args.no_rollouts_dir) if args.no_rollouts_dir else None
        yes_forced_answer_dir = Path(args.yes_forced_answer_rollouts_dir) if args.yes_forced_answer_rollouts_dir else Path(f"whistleblower_rollouts/{model_name}/temperature_0.7_top_p_0.95/yes_base_solution_forced_answer")
        no_forced_answer_dir = Path(args.no_forced_answer_rollouts_dir) if args.no_forced_answer_rollouts_dir else None
        output_dir = Path(args.output_dir) if args.output_dir else Path(f"analysis/{model_name}/basic")
        output_dir.mkdir(exist_ok=True, parents=True)

        # Check if at least one rollouts directory exists
        if not yes_rollouts_dir and not no_rollouts_dir:
            print("Error: At least one of --yes_rollouts_dir or --no_rollouts_dir must be provided or derivable")
            continue

        # In repair-only mode, skip length stats and normal analysis unless directories used for repair
        if not args.fix_labeling_errors_only:
            # Analyze response length statistics across both yes and no rollouts when both available
            if yes_rollouts_dir and no_rollouts_dir:
                print("\n=== Analyzing Response Length Statistics ===\n")
                analyze_response_length_statistics(yes_rollouts_dir, no_rollouts_dir, output_dir)

        # Process YES rollouts if present
        if yes_rollouts_dir:
            print(f"\n=== Processing YES (whistleblowing) rollouts from {yes_rollouts_dir} ===\n")
            yes_output_dir = output_dir / "yes_base_solution"
            yes_output_dir.mkdir(exist_ok=True, parents=True)
            process_rollouts(
                rollouts_dir=yes_rollouts_dir,
                output_dir=yes_output_dir,
                problems=args.problems,
                max_problems=args.max_problems,
                absolute=args.absolute,
                force_relabel=args.force_relabel,
                rollout_type="yes",
                dag_dir=getattr(args, 'dag_dir', None) if getattr(args, 'token_analysis_source', None) == "dag" else None,
                forced_answer_dir=yes_forced_answer_dir,
                get_token_frequencies=getattr(args, 'get_token_frequencies', False),
                max_chunks_to_show=args.max_chunks_to_show,
                use_existing_metrics=args.use_existing_metrics,
                importance_metric=args.importance_metric,
                sentence_model=args.sentence_model,
                similarity_threshold=args.similarity_threshold,
                force_metadata=args.force_metadata,
                model_name_label=model_name
            )

            # If repair-only, skip specialized forced analyses
            if not args.fix_labeling_errors_only and yes_forced_answer_dir and yes_forced_answer_dir.exists():
                print("\n=== Running additional analysis with forced importance metric ===\n")
                forced_output_dir = output_dir / "forced_importance_analysis"
                forced_output_dir.mkdir(exist_ok=True, parents=True)

                # Check if the results have the forced importance metric
                forced_importance_exists = False

                for result_file in yes_output_dir.glob("**/analysis_results.json"):
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            results = json.load(f)

                        # Check if at least one chunk has the forced importance metric
                        for result in results:
                            for chunk in result.get("labeled_chunks", []):
                                if "forced_importance_whistleblowing_rate" in chunk or "forced_importance_kl" in chunk:
                                    forced_importance_exists = True
                                    break
                            if forced_importance_exists:
                                break

                        if forced_importance_exists:
                            break

                    except Exception as e:
                        print(f"Error checking for forced importance metric: {e}")

                if forced_importance_exists:
                    print("Found forced importance metric in results, running specialized analysis")

                    # Copy the results to the forced output directory to run separate analysis
                    # Just use the existing files with symlinks
                    import shutil
                    try:
                        # Create a symlink to the analysis_results.json file
                        src_file = yes_output_dir / "analysis_results.json"
                        dst_file = forced_output_dir / "analysis_results.json"
                        if src_file.exists() and not dst_file.exists():
                            import os
                            os.symlink(src_file, dst_file)

                        # Run top steps and high z-score analyses specifically for forced importance
                        print("Running top steps analysis with forced importance metrics")
                        results = []
                        with open(src_file, 'r', encoding='utf-8') as f:
                            results = json.load(f)

                        analyze_top_steps_by_category(results, forced_output_dir, top_n=20, use_abs=True, importance_metric="forced_importance_whistleblowing_rate")
                        analyze_high_zscore_steps_by_category(results, forced_output_dir, z_threshold=1.5, use_abs=True, importance_metric="forced_importance_whistleblowing_rate")

                        # Also run analyses for forced_importance_kl
                        print("Running top steps analysis with forced importance KL metric")
                        analyze_top_steps_by_category(results, forced_output_dir, top_n=20, use_abs=True, importance_metric="forced_importance_kl")
                        analyze_high_zscore_steps_by_category(results, forced_output_dir, z_threshold=1.5, use_abs=True, importance_metric="forced_importance_kl")

                    except Exception as e:
                        print(f"Error during forced importance specialized analysis: {e}")
                else:
                    print("No forced importance metric found in results, skipping specialized analysis")

        # Process NO rollouts if present
        if no_rollouts_dir:
            print(f"\n=== Processing NO (non-whistleblowing) rollouts from {no_rollouts_dir} ===\n")
            no_output_dir = output_dir / "no_base_solution"
            no_output_dir.mkdir(exist_ok=True, parents=True)
            process_rollouts(
                rollouts_dir=no_rollouts_dir,
                output_dir=no_output_dir,
                problems=args.problems,
                max_problems=args.max_problems,
                absolute=args.absolute,
                force_relabel=args.force_relabel,
                rollout_type="no",
                dag_dir=getattr(args, 'dag_dir', None) if getattr(args, 'token_analysis_source', None) == "dag" else None,
                forced_answer_dir=no_forced_answer_dir,
                get_token_frequencies=getattr(args, 'get_token_frequencies', False),
                max_chunks_to_show=args.max_chunks_to_show,
                use_existing_metrics=args.use_existing_metrics,
                importance_metric=args.importance_metric,
                sentence_model=args.sentence_model,
                similarity_threshold=args.similarity_threshold,
                force_metadata=args.force_metadata,
                model_name_label=model_name
            )

if __name__ == "__main__":
    main()