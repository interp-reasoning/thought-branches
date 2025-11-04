#!/usr/bin/env python3
"""
Analyze where different sentence types land on the probability distribution vs impact.

Objective:
- X-axis: how on-policy/off-policy the sentence is → mean token logprob under the display model
- Y-axis: impact/effect → Δ blackmail rate (after − before)

Sentence types visualized:
- onpolicy_inserted: on-policy (varied) — selected resampled chunk from the varied pipeline
- onpolicy_fixed: on-policy (fixed) — fixed target sentence pipeline
- offpolicy_handwritten: disruption_text from off-policy chain disruption (handwritten target)
- baseline_mode: the original base sentence at that chunk (serves as a proxy for distribution mode; optional via flag)

Data sources:
- On-policy (varied): blackmail_rollouts/<model>/temperature_<t>_top_p_<p>/yes_base_solution_onpolicy_chain_disruption/scenario_*/onpolicy_chain_disruption_*.json
- On-policy (fixed): blackmail_rollouts/<model>/temperature_<t>_top_p_<p>/yes_base_solution_onpolicy_chain_disruption_fixed/scenario_*/onpolicy_chain_disruption_*.json
- Off-policy results: blackmail_rollouts/<model>/temperature_<t>_top_p_<p>/yes_base_solution_chain_disruption/scenario_*/chain_disruption_results.json
- Baseline per-chunk solutions and metadata in: blackmail_rollouts/<model>/temperature_<t>_top_p_<p>/yes_base_solution

Outputs (for qwq-32b only):
- analysis/onpolicy_distribution/qwq-32b/temperature_<t>_top_p_<p>/distribution_points.csv
- analysis/onpolicy_distribution/qwq-32b/temperature_<t>_top_p_<p>/scatter_distribution.png
"""

import os
import re
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

import torch
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # type: ignore
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
except Exception:
    AutoTokenizer, AutoModelForCausalLM = None, None  # type: ignore

# Disruption library base sentences (blackmail domain) - load without importing the module
import ast
def _load_disruption_library_from_source() -> Dict[str, str]:
    path = Path('onpolicy_chain_disruption.py')
    out: Dict[str, str] = {}
    try:
        if not path.exists():
            return out
        src = path.read_text(encoding='utf-8')
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Handle simple assignments only
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == 'DISRUPTION_LIBRARY':
                        try:
                            val = ast.literal_eval(node.value)
                            if isinstance(val, dict):
                                # Ensure str to str
                                for k, v in val.items():
                                    if isinstance(k, str) and isinstance(v, str):
                                        out[k] = v
                        except Exception:
                            continue
        return out
    except Exception:
        return out

DISRUPTION_LIBRARY_BM: Dict[str, str] = _load_disruption_library_from_source()


# ------------------------------
# Config
# ------------------------------

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
MODEL_FOLDER_TO_HF = {
    'qwq-32b': 'Qwen/QwQ-32B',
    'hermes-4-70b': 'NousResearch/Hermes-4-70B',
    'qwen3-235b-a22b': 'Qwen/Qwen3-235B-A22B',
    'llama-3_1-nemotron-ultra-253b-v1': 'nvidia/Llama-3_1-Nemotron-Ultra-253B-v1',
    'deepseek-r1-0528': 'deepseek-ai/DeepSeek-R1-0528',
}

# Provider model IDs for Nebius API (best-effort mapping)
MODEL_FOLDER_TO_NEBIUS = {
    'qwq-32b': 'Qwen/QwQ-32B',
    'hermes-4-70b': 'NousResearch/Hermes-4-70B',
    'qwen3-235b-a22b': 'Qwen/Qwen3-235B-A22B-Instruct-2507',
    'llama-3_1-nemotron-ultra-253b-v1': 'nvidia/Llama-3_1-Nemotron-Ultra-253B-v1',
    'deepseek-r1-0528': 'deepseek-ai/DeepSeek-R1-0528',
}


# ------------------------------
# File utilities
# ------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path: Path, data: Dict[str, Any]) -> None:
    try:
        ensure_dir(path.parent)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception:
        pass


# ------------------------------
# Logprobs caching
# ------------------------------

import hashlib

def _hash_text(text: str) -> str:
    """Generate a short hash for text to use in cache keys."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def get_cache_path(model_folder: str, prefix_text: str, sentence_text: str) -> Path:
    """Generate cache path for a specific logprobs computation."""
    prefix_hash = _hash_text(prefix_text)
    sentence_hash = _hash_text(sentence_text)
    cache_dir = Path('analysis') / 'onpolicy_distribution' / 'cache' / model_folder
    cache_file = cache_dir / f"{prefix_hash}_{sentence_hash}.json"
    return cache_file


def load_cached_logprobs(model_folder: str, prefix_text: str, sentence_text: str) -> Optional[Dict[str, Any]]:
    """Load cached logprobs if available."""
    cache_path = get_cache_path(model_folder, prefix_text, sentence_text)
    return load_json(cache_path)


def save_cached_logprobs(model_folder: str, prefix_text: str, sentence_text: str, logprobs_data: Dict[str, Any]) -> None:
    """Save logprobs to cache."""
    cache_path = get_cache_path(model_folder, prefix_text, sentence_text)
    save_json(cache_path, logprobs_data)


def list_scenarios(input_dir: Path) -> List[Path]:
    return sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith('scenario_')])


def resolve_roots(model_folder: str, temperature: float, top_p: float, random_variant: bool = False) -> Tuple[Path, Path, Path, Path, Path]:
    base = Path('blackmail_rollouts') / model_folder / f"temperature_{str(temperature)}_top_p_{str(top_p)}"
    if bool(random_variant):
        # Random variant folders use the *_fixed_random naming
        baseline = base / 'yes_base_solution'  # baseline unchanged unless specifically provided
        onpolicy_dir = base / 'yes_base_solution_onpolicy_chain_disruption_fixed_random'
        offpolicy_dir = base / 'yes_base_solution_chain_disruption_fixed_random'
        onpolicy_fixed_dir = base / 'yes_base_solution_onpolicy_chain_disruption_fixed_random'
        agent_dir = base / 'yes_base_solution_agent_chain_disruption_fixed_random'
    else:
        baseline = base / 'yes_base_solution'
        onpolicy_dir = base / 'yes_base_solution_onpolicy_chain_disruption'
        offpolicy_dir = base / 'yes_base_solution_chain_disruption'
        onpolicy_fixed_dir = base / 'yes_base_solution_onpolicy_chain_disruption_fixed'
        agent_dir = base / 'yes_base_solution_agent_chain_disruption'
    return baseline, onpolicy_dir, offpolicy_dir, onpolicy_fixed_dir, agent_dir


def build_output_dir(model_folder: str, temperature: float, top_p: float) -> Path:
    out_root = Path('analysis') / 'onpolicy_distribution' / model_folder / f"temperature_{str(temperature)}_top_p_{str(top_p)}"
    ensure_dir(out_root)
    return out_root


# ------------------------------
# Text utilities
# ------------------------------

def load_chunks_and_separators(problem_dir: Path) -> Tuple[List[str], List[str]]:
    """Prefer chunks.json with separators; fallback to chunks_labeled.json."""
    chunks_file = problem_dir / 'chunks.json'
    if chunks_file.exists():
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return (data.get('chunks', []) or [], data.get('separators', []) or [])
        except Exception:
            pass
    chunks_labeled_file = problem_dir / 'chunks_labeled.json'
    if chunks_labeled_file.exists():
        try:
            with open(chunks_labeled_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
                chunks = [c.get('chunk', '') for c in chunks_data]
                seps = [" "] * len(chunks)
                return chunks, seps
        except Exception:
            pass
    return [], []


def make_prefix_text(chunks: List[str], seps: List[str], end_index: int) -> str:
    if not chunks or end_index <= 0:
        return ""
    parts: List[str] = []
    for i in range(end_index):
        parts.append(chunks[i])
        if i < len(seps):
            parts.append(seps[i])
    return "".join(parts)


def build_prefill(system_prompt: str, user_prompt: str, email_content: str, prefix_text: str) -> str:
    return f"{system_prompt}\n\nUser: {user_prompt}\n\n{email_content}\n\nAssistant:\n<think>\n{prefix_text}"


# ------------------------------
# HF model loader and token logprobs (teacher forcing)
# ------------------------------

@dataclass
class HFModelBundle:
    model: Any
    tokenizer: Any
    device: str

logger = logging.getLogger("onpolicy_distribution")


def load_model_and_tokenizer(model_name: str, device: str = 'cuda:0', quantize_4bit: bool = True, device_map_pref: Optional[str] = None) -> HFModelBundle:
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError('transformers is not available in this environment')
    print(f"[dist] Loading HF model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available() and isinstance(device, str) and device.startswith('cuda'):
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
            print('[dist] Using BFloat16')
        else:
            compute_dtype = torch.float16
            print('[dist] Using Float16')
    else:
        compute_dtype = torch.float32
        print('[dist] Using Float32 (CPU)')
    # Try 4-bit GPU load first
    if quantize_4bit and torch.cuda_is_available() if hasattr(torch, 'cuda_is_available') else torch.cuda.is_available() and isinstance(device, str) and device.startswith('cuda'):
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=(device_map_pref or 'auto'),
                quantization_config=quant_config,
                torch_dtype=compute_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model.eval()
            return HFModelBundle(model=model, tokenizer=tokenizer, device=device)
        except Exception as e:
            print(f"[dist] 4-bit GPU load failed: {type(e).__name__}: {e}")

        # Fallback: 8-bit with CPU offload (device_map=auto)
        try:
            offload_dir = Path('analysis') / 'onpolicy_distribution' / 'offload_cache'
            offload_dir.mkdir(parents=True, exist_ok=True)

            # Build conservative max_memory mapping
            max_memory: Dict[str, Any] = {}
            try:
                if torch.cuda.is_available():
                    num_gpu = torch.cuda.device_count()
                    for gi in range(num_gpu):
                        props = torch.cuda.get_device_properties(gi)
                        total_gb = int(props.total_memory / (1024 ** 3))
                        # leave ~15% headroom
                        max_memory[gi] = f"{max(1, int(total_gb * 0.85))}GiB"
            except Exception:
                pass
            try:
                if psutil is not None:
                    vm = psutil.virtual_memory()
                    total_cpu_gb = int(vm.total / (1024 ** 3))
                    max_memory['cpu'] = f"{max(4, int(total_cpu_gb * 0.9))}GiB"
                else:
                    max_memory['cpu'] = '64GiB'
            except Exception:
                max_memory['cpu'] = '64GiB'

            from transformers import BitsAndBytesConfig  # type: ignore
            q8 = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=(device_map_pref or 'auto'),
                quantization_config=q8,
                max_memory=max_memory,
                offload_folder=str(offload_dir),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model.eval()
            print("[dist] Loaded with 8-bit + CPU offload (device_map=auto)")
            return HFModelBundle(model=model, tokenizer=tokenizer, device=(device if torch.cuda.is_available() else 'cpu'))
        except Exception as e2:
            print(f"[dist] 8-bit CPU offload load failed: {type(e2).__name__}: {e2}")

    # Final fallback: non-quantized CPU (slow)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={'': 'cpu'},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = model.to('cpu')
        model.eval()
        print("[dist] Loaded on CPU (no quantization)")
        return HFModelBundle(model=model, tokenizer=tokenizer, device='cpu')
    except Exception as e3:
        print(f"[dist] CPU load failed: {type(e3).__name__}: {e3}")
        raise


@torch.no_grad()
def compute_token_logprobs(bundle: HFModelBundle, prefix_text: str, sentence_text: str, model_folder: Optional[str] = None, use_cache: bool = True) -> Dict[str, Any]:
    """Compute per-token logprobs for sentence_text given prefix_text using teacher forcing.

    Robust to accelerate device_map=auto by keeping inputs on CPU; accelerate handles dispatch.
    Returns tokens, token_ids, per-token logprobs, sum, and mean.
    """
    # Check cache first
    if use_cache and model_folder is not None:
        cached = load_cached_logprobs(model_folder, prefix_text, sentence_text)
        if cached is not None:
            return cached
    
    tokenizer = bundle.tokenizer
    model = bundle.model

    # Tokenize (CPU tensors)
    ids_prefix = tokenizer(prefix_text, add_special_tokens=True, return_tensors='pt')
    ids_target = tokenizer(sentence_text, add_special_tokens=False, return_tensors='pt')

    target_len = int(ids_target['input_ids'].shape[1])
    token_ids: List[int] = ids_target['input_ids'][0].tolist()
    tokens: List[str] = tokenizer.convert_ids_to_tokens(token_ids)
    if target_len <= 0:
        result = {
            'tokens': tokens,
            'token_ids': token_ids,
            'logprobs': [],
            'logprob_sum': 0.0,
            'logprob_mean': 0.0,
        }
        if use_cache and model_folder is not None:
            save_cached_logprobs(model_folder, prefix_text, sentence_text, result)
        return result

    input_ids = torch.cat([ids_prefix['input_ids'], ids_target['input_ids']], dim=1)
    attn = torch.cat([ids_prefix['attention_mask'], ids_target['attention_mask']], dim=1)

    # Ensure inputs are on the same device as the embedding weights (required by torch.embedding)
    try:
        emb_mod = model.get_input_embeddings() if hasattr(model, 'get_input_embeddings') else None
        emb_dev = emb_mod.weight.device if emb_mod is not None and hasattr(emb_mod, 'weight') else None
    except Exception:
        emb_dev = None
    if emb_dev is None:
        try:
            emb_dev = next((p.device for p in model.parameters() if getattr(p, 'device', None) is not None), torch.device('cpu'))
        except Exception:
            emb_dev = torch.device('cpu')
    # Ensure correct dtype for embedding lookups
    input_ids = input_ids.to(device=emb_dev, dtype=torch.long)
    # attention_mask may be bool/long; cast to long to be safe
    attn = attn.to(device=emb_dev, dtype=torch.long)

    # Forward
    try:
        outputs = model(input_ids=input_ids, attention_mask=attn)
    except Exception as e:
        logger.warning(f"forward failed on device {input_ids.device}: {type(e).__name__}: {e}")
        # Last-resort: force CPU
        input_ids = input_ids.to('cpu')
        attn = attn.to('cpu')
        outputs = model(input_ids=input_ids, attention_mask=attn)
    logits = outputs.logits  # [1, T, V]

    prefix_len = int(ids_prefix['input_ids'].shape[1])

    # Use shifted logits for teacher forcing
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    start = max(prefix_len - 1, 0)
    end = min(shift_logits.shape[1], prefix_len + target_len - 1)
    if end <= start:
        logger.debug(f"empty window: logits_T={logits.shape[1]} prefix_len={prefix_len} target_len={target_len} start={start} end={end}")
        # Fallback to per-token stepping
        logprobs = []
        for t in range(target_len):
            pos = prefix_len + t
            idx = pos - 1
            if idx < 0 or idx >= logits.shape[1]:
                continue
            step = torch.log_softmax(logits[:, idx:idx+1, :], dim=-1)
            tok = int(token_ids[t])
            lp = float(step[0, 0, tok].detach().cpu().item())
            logprobs.append(lp)
        arr = np.array(logprobs, dtype=float)
        result = {
            'tokens': tokens,
            'token_ids': token_ids,
            'logprobs': logprobs,
            'logprob_sum': float(arr.sum()) if arr.size > 0 else 0.0,
            'logprob_mean': float(arr.mean()) if arr.size > 0 else 0.0,
        }
        if use_cache and model_folder is not None:
            save_cached_logprobs(model_folder, prefix_text, sentence_text, result)
        return result

    sel_logits = shift_logits[:, start:end, :].to(torch.float32)
    sel_labels = shift_labels[:, start:end]

    log_probs_all = torch.log_softmax(sel_logits, dim=-1)
    gathered = log_probs_all.gather(-1, sel_labels.unsqueeze(-1)).squeeze(-1)  # [1, L]

    logprobs = gathered[0].detach().cpu().numpy().astype(float).tolist()
    arr = np.array(logprobs, dtype=float)
    if arr.size == 0:
        # Fallback: per-token loop, plus diagnostics
        logger.debug(f"vectorized gathered empty: logits_T={logits.shape[1]} prefix_len={prefix_len} target_len={target_len} start={start} end={end}")
        logprobs = []
        for t in range(target_len):
            pos = prefix_len + t
            idx = pos - 1
            if idx < 0 or idx >= logits.shape[1]:
                continue
            step = torch.log_softmax(logits[:, idx:idx+1, :], dim=-1)
            tok = int(token_ids[t])
            lp = float(step[0, 0, tok].detach().cpu().item())
            logprobs.append(lp)
        arr = np.array(logprobs, dtype=float)

    result = {
        'tokens': tokens,
        'token_ids': token_ids,
        'logprobs': logprobs,
        'logprob_sum': float(arr.sum()) if arr.size > 0 else 0.0,
        'logprob_mean': float(arr.mean()) if arr.size > 0 else 0.0,
        'scored_token_count': int(arr.size),
        'total_token_count': int(len(logprobs)),
    }
    if use_cache and model_folder is not None:
        save_cached_logprobs(model_folder, prefix_text, sentence_text, result)
    return result


# ------------------------------
# Remote logprobs via Nebius (per-token, incremental)
# ------------------------------

def _nebius_completion(
    prompt: str,
    *,
    model_id: str,
    api_key: Optional[str],
    timeout: int = 180,
    top_logprobs: int = 20,
    echo: bool = False,
    max_tokens: int = 1,
    retries: int = 3,
    backoff_seconds: float = 0.75,
) -> Optional[Dict[str, Any]]:
    if not api_key:
        logger.warning("NEBIUS_API_KEY not set; cannot fetch logprobs remotely")
        return None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model_id,
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
        "stream": False,
        "logprobs": True,
        "top_logprobs": int(top_logprobs),
    }
    # Many providers (OpenAI-compatible) support echo to return logprobs for prompt tokens.
    # We set it only when requested to avoid changing default behavior.
    if echo:
        payload["echo"] = True
    import httpx  # local import to avoid hard dep when unused
    import time
    attempt = 0
    while True:
        try:
            # Use a connection pool with limited keepalive to avoid lingering sockets
            limits = httpx.Limits(max_keepalive_connections=8, max_connections=64)
            with httpx.Client(timeout=timeout, limits=limits) as client:
                resp = client.post("https://api.studio.nebius.com/v1/completions", headers=headers, json=payload)
                # print(f"Nebius response: {resp.status_code} {resp.text}")
                if resp.status_code == 200:
                    return resp.json()
                # Retry on transient statuses
                if resp.status_code in (429, 500, 502, 503):
                    attempt += 1
                    logger.warning(f"Nebius HTTP {resp.status_code}: {resp.text[:200]}")
                    if attempt > max(0, int(retries)):
                        return None
                    time.sleep(max(0.1, float(backoff_seconds) * (1.6 ** (attempt - 1))))
                    continue
                logger.warning(f"Nebius HTTP {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            attempt += 1
            logger.warning(f"Nebius request failed: {type(e).__name__}: {e}")
            if attempt > max(0, int(retries)):
                return None
            time.sleep(max(0.1, float(backoff_seconds) * (1.6 ** (attempt - 1))))


def compute_token_logprobs_nebius(
    model_folder: str,
    prefix_text: str,
    sentence_text: str,
    tokenizer_cache: Dict[str, Any],
    api_key: Optional[str],
    top_logprobs: int = 20,
    timeout_seconds: int = 180,
    fill_missing: bool = True,
    verbose: bool = False,
    suffix_text: str = "",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Compute per-token logprobs for sentence_text given prefix via Nebius API.

    Uses a sliding window token-matching approach for robust extraction.
    This matches the exact implementation from test_logprob_extraction.py.
    """
    # Check cache first
    if use_cache:
        cached = load_cached_logprobs(model_folder, prefix_text, sentence_text)
        if cached is not None:
            return cached
    
    hf_id = MODEL_FOLDER_TO_HF.get(model_folder, MODEL_FOLDER_TO_HF['qwen3-235b-a22b'])
    tok = tokenizer_cache.get(hf_id)
    if tok is None:
        try:
            from transformers import AutoTokenizer  # type: ignore
            tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            tokenizer_cache[hf_id] = tok
        except Exception:
            result = {'tokens': [], 'token_ids': [], 'logprobs': [], 'logprob_sum': 0.0, 'logprob_mean': 0.0}
            if use_cache:
                save_cached_logprobs(model_folder, prefix_text, sentence_text, result)
            return result

    # Tokenize target locally to return token list for reference (not required by provider)
    ids_target = tok(sentence_text, add_special_tokens=False, return_tensors='pt')
    token_ids: List[int] = ids_target['input_ids'][0].tolist()
    tokens: List[str] = tok.convert_ids_to_tokens(token_ids)

    # Use provider model id for Nebius
    provider_id = MODEL_FOLDER_TO_NEBIUS.get(model_folder, hf_id)

    # Primary strategy: single echo completion with max_tokens=0 and echo=True
    try:
        # Some providers require a non-empty decoder prompt; guard against empty prefix
        safe_prefix = prefix_text if isinstance(prefix_text, str) and len(prefix_text) > 0 else " "
        if verbose:
            print(f"Nebius request: {safe_prefix}")
            print(f"Provider ID: {provider_id}")
            try:
                masked_key = (api_key[:6] + "…" + api_key[-4:]) if (api_key and len(api_key) > 10) else (api_key or "None")
            except Exception:
                masked_key = "***"
            print(f"API Key: {masked_key}")
            print(f"Timeout: {timeout_seconds}")
            print(f"Top Logprobs: {top_logprobs}")
            print(f"Echo: {True}")
            print(f"Max Tokens: {0}")

        res_full = _nebius_completion(
            safe_prefix + sentence_text + (suffix_text or ""),
            model_id=provider_id,
            api_key=api_key,
            top_logprobs=max(1, int(top_logprobs)),
            echo=True,
            max_tokens=0,
            timeout=int(timeout_seconds),
        )
        if verbose:
            print(f"Nebius response (full): {res_full}")

        if res_full and 'choices' in res_full and res_full['choices']:
            lg_full = (res_full['choices'][0] or {}).get('logprobs') or {}
            full_top_lps = lg_full.get('top_logprobs') or []
            full_tokens = lg_full.get('tokens') or []
            
            if not full_top_lps or not full_tokens:
                logger.debug(f"No top_logprobs returned from API for sentence: {sentence_text[:50]}")
                result = {'tokens': tokens, 'token_ids': token_ids, 'logprobs': [], 'logprob_sum': float('nan'), 'logprob_mean': float('nan'), 'scored_token_count': 0, 'total_token_count': 0}
                if use_cache:
                    save_cached_logprobs(model_folder, prefix_text, sentence_text, result)
                return result

            # Sliding window token matching (same as test_logprob_extraction.py)
            def normalize_token(tok: str) -> str:
                """Replace BPE markers: Ġ/\u0120 → space, Ċ/\u010A → newline.
                Also normalize fancy quotes to straight quotes."""
                return (tok.replace('Ġ', ' ').replace('\u0120', ' ')
                           .replace('Ċ', '\n').replace('\u010A', '\n')
                           .replace('\u2019', "'")
                           .replace('\u00e2\u0122\u013b', "'")
                           .replace(''', "'").replace(''', "'")  # Fancy single quotes
                           .replace('"', '"').replace('"', '"'))  # Fancy double quotes
            
            # Prepare token list with logprobs
            token_list = []
            for i, (token, top_lp_dict) in enumerate(zip(full_tokens, full_top_lps)):
                if top_lp_dict is None:
                    continue  # Skip BOS
                
                if isinstance(top_lp_dict, dict) and len(top_lp_dict) > 0:
                    actual_token = list(top_lp_dict.keys())[0]
                    actual_logprob = float(list(top_lp_dict.values())[0])
                    token_list.append((i, actual_token, actual_logprob))
            
            # Normalize sentence for comparison
            normalized_sentence = normalize_token(sentence_text)
            
            # Sliding window: try each position as start
            # Handle case where sentence might start in the middle of a token (e.g., ".ĊĊActually" contains "\n\nActually")
            start_idx = None
            end_idx = None
            sentence_tokens_list = []
            sentence_logprobs_list = []
            
            for start_pos in range(len(token_list)):
                accumulated = ""
                temp_tokens = []
                temp_logprobs = []
                
                for offset in range(len(token_list) - start_pos):
                    i, token, lp = token_list[start_pos + offset]
                    normalized_tok = normalize_token(token)
                    accumulated += normalized_tok
                    temp_tokens.append(token)
                    temp_logprobs.append(lp)
                    
                    # Check if accumulated contains our sentence (handles mid-token matches)
                    if normalized_sentence in accumulated:
                        # Find where in accumulated the sentence starts
                        sentence_start_in_acc = accumulated.find(normalized_sentence)
                        
                        # If sentence doesn't start at position 0, we need to find exact token boundaries
                        if sentence_start_in_acc > 0:
                            # Rebuild to find which tokens overlap with sentence span
                            acc2 = ""
                            final_tokens = []
                            final_logprobs = []
                            for j in range(len(temp_tokens)):
                                tok_j = temp_tokens[j]
                                lp_j = temp_logprobs[j]
                                tok_norm = normalize_token(tok_j)
                                token_start_in_acc = len(acc2)
                                token_end_in_acc = token_start_in_acc + len(tok_norm)
                                acc2 += tok_norm
                                
                                # Include token if it overlaps with sentence span
                                sentence_end_in_acc = sentence_start_in_acc + len(normalized_sentence)
                                if token_end_in_acc > sentence_start_in_acc and token_start_in_acc < sentence_end_in_acc:
                                    final_tokens.append(tok_j)
                                    final_logprobs.append(lp_j)
                            
                            start_idx = token_list[start_pos][0] if final_tokens else None
                            end_idx = (token_list[start_pos + len(temp_tokens) - 1][0] + 1) if final_tokens else None
                            sentence_tokens_list = final_tokens
                            sentence_logprobs_list = final_logprobs
                        else:
                            # Sentence starts at position 0 - exact match from token boundary
                            start_idx = token_list[start_pos][0]
                            end_idx = i + 1
                            sentence_tokens_list = temp_tokens
                            sentence_logprobs_list = temp_logprobs
                        break
                    
                    # If accumulated is much longer than sentence and no match, stop trying this start position
                    if len(accumulated) > len(normalized_sentence) + 10:
                        break
                
                # If we found a match, stop searching
                if start_idx is not None:
                    break
            
            # If no exact match, try with stripped sentence (no leading/trailing whitespace)
            if start_idx is None or not sentence_logprobs_list:
                normalized_sentence_stripped = normalized_sentence.strip()
                
                if normalized_sentence_stripped and normalized_sentence_stripped != normalized_sentence:
                    # Try again with stripped version
                    for start_pos in range(len(token_list)):
                        accumulated = ""
                        temp_tokens = []
                        temp_logprobs = []
                        
                        for offset in range(len(token_list) - start_pos):
                            i, token, lp = token_list[start_pos + offset]
                            normalized_tok = normalize_token(token)
                            accumulated += normalized_tok
                            temp_tokens.append(token)
                            temp_logprobs.append(lp)
                            
                            # Check if accumulated contains stripped sentence
                            if normalized_sentence_stripped in accumulated:
                                sentence_start_in_acc = accumulated.find(normalized_sentence_stripped)
                                
                                if sentence_start_in_acc > 0:
                                    # Find exact token boundaries
                                    acc2 = ""
                                    final_tokens = []
                                    final_logprobs = []
                                    for j in range(len(temp_tokens)):
                                        tok_j = temp_tokens[j]
                                        lp_j = temp_logprobs[j]
                                        tok_norm = normalize_token(tok_j)
                                        token_start_in_acc = len(acc2)
                                        token_end_in_acc = token_start_in_acc + len(tok_norm)
                                        acc2 += tok_norm
                                        
                                        sentence_end_in_acc = sentence_start_in_acc + len(normalized_sentence_stripped)
                                        if token_end_in_acc > sentence_start_in_acc and token_start_in_acc < sentence_end_in_acc:
                                            final_tokens.append(tok_j)
                                            final_logprobs.append(lp_j)
                                    
                                    start_idx = token_list[start_pos][0] if final_tokens else None
                                    end_idx = (token_list[start_pos + len(temp_tokens) - 1][0] + 1) if final_tokens else None
                                    sentence_tokens_list = final_tokens
                                    sentence_logprobs_list = final_logprobs
                                else:
                                    start_idx = token_list[start_pos][0]
                                    end_idx = i + 1
                                    sentence_tokens_list = temp_tokens
                                    sentence_logprobs_list = temp_logprobs
                                break
                            
                            if len(accumulated) > len(normalized_sentence_stripped) + 10:
                                break
                        
                        if start_idx is not None:
                            break
            
            # If we found a match, return it
            if start_idx is not None and sentence_logprobs_list:
                arr = np.array(sentence_logprobs_list, dtype=float)
                result = {
                    'tokens': tokens,
                    'token_ids': token_ids,
                    'logprobs': sentence_logprobs_list,
                    'logprob_sum': float(arr.sum()) if arr.size > 0 else float('nan'),
                    'logprob_mean': float(arr.mean()) if arr.size > 0 else float('nan'),
                    'scored_token_count': int(arr.size),
                    'total_token_count': int(len(sentence_tokens_list)),
                }
                if use_cache:
                    save_cached_logprobs(model_folder, prefix_text, sentence_text, result)
                return result
            else:
                # Could not match - log warning
                logger.warning(f"Token matching failed for sentence: {sentence_text[:60]}")
                logger.warning(f"  Looking for (normalized): {repr(normalized_sentence[:100])}")
                logger.warning(f"  Also tried stripped: {repr(normalized_sentence_stripped[:100])}")
                if token_list:
                    sample_reconstructed = ""
                    for j in range(min(30, len(token_list))):
                        sample_reconstructed += normalize_token(token_list[j][1])
                    logger.warning(f"  Sample token stream: {repr(sample_reconstructed[:150])}")
                
                result = {'tokens': tokens, 'token_ids': token_ids, 'logprobs': [], 'logprob_sum': float('nan'), 'logprob_mean': float('nan'), 'scored_token_count': 0, 'total_token_count': 0}
                if use_cache:
                    save_cached_logprobs(model_folder, prefix_text, sentence_text, result)
                return result
    except Exception as e:
        logger.warning(f"Nebius prompt-logprob scoring (echo) failed: {type(e).__name__}: {e}")

    # Fallback: incremental next-token scoring using top_logprobs to look up the
    # intended token probability. For tokens not present in top-K, fill with the
    # minimum top-K logprob or a conservative floor.
    running = prefix_text
    out_lps: List[float] = []
    floor_lp = -20.0
    req_k = min(200, max(1, int(top_logprobs)))
    for i in range(len(tokens)):
        piece_text = tok.convert_tokens_to_string([tokens[i]])
        res = _nebius_completion(
            running,
            model_id=provider_id,
            api_key=api_key,
            top_logprobs=req_k,
            echo=False,
            max_tokens=1,
            timeout=int(timeout_seconds),
        )
        if not res or 'choices' not in res or not res['choices']:
            # Could not score; use floor
            out_lps.append(float(floor_lp))
            running = running + piece_text
            continue
        choice = res['choices'][0]
        lp_val = None
        min_lp = None
        try:
            lg = choice.get('logprobs') or {}
            # Prefer top_logprobs map for the exact token string
            if 'top_logprobs' in lg and isinstance(lg['top_logprobs'], list) and lg['top_logprobs']:
                tl = lg['top_logprobs'][0]
                if isinstance(tl, dict):
                    if piece_text in tl:
                        lp_val = float(tl[piece_text])
                    try:
                        vals = [float(v) for v in tl.values()]
                        if vals:
                            min_lp = float(min(vals))
                    except Exception:
                        pass
                elif isinstance(tl, list):
                    for entry in tl:
                        try:
                            tok_s = entry.get('token') or entry.get('text')
                            if tok_s == piece_text:
                                lp_val = float(entry.get('logprob'))
                                break
                            lp_e = entry.get('logprob')
                            if lp_e is not None:
                                min_lp = float(min_lp) if min_lp is not None else float(lp_e)
                                if float(lp_e) < float(min_lp):
                                    min_lp = float(lp_e)
                        except Exception:
                            continue
            # Fallback: if the generated token equals our intended piece, use its logprob
            if lp_val is None and 'token_logprobs' in lg and 'tokens' in lg:
                gen_tok = str(lg['tokens'][0]) if lg['tokens'] else None
                gen_lp = float(lg['token_logprobs'][0]) if lg['token_logprobs'] else None
                if gen_tok is not None and piece_text and gen_tok == piece_text:
                    lp_val = gen_lp
        except Exception:
            lp_val = None
        # Include a conservative value for missing tokens
        if lp_val is None:
            lp_val = float(min_lp) if min_lp is not None else float(floor_lp)
        out_lps.append(float(lp_val))
        # Update running with provider token text (not joined string) to reduce whitespace mismatch
        running = running + (tokens[i] if isinstance(tokens[i], str) else piece_text)

    arr = np.array(out_lps, dtype=float)
    result = {
        'tokens': tokens,
        'token_ids': token_ids,
        'logprobs': out_lps,
        'logprob_sum': float(arr.sum()) if arr.size > 0 else float('nan'),
        'logprob_mean': float(arr.mean()) if arr.size > 0 else float('nan'),
        'scored_token_count': int(arr.size),
        'total_token_count': int(len(tokens)),
    }
    if use_cache:
        save_cached_logprobs(model_folder, prefix_text, sentence_text, result)
    return result


# ------------------------------
# Metrics (reuse logic from analyze_onpolicy_chain_disruption)
# ------------------------------

def _read_json(path: Path) -> Any:
    try:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _compute_rate_from_solutions(solutions: Any) -> Optional[float]:
    try:
        if not isinstance(solutions, list) or not solutions:
            return None
        labels = [1 if bool(s.get('contains_blackmail')) else 0 for s in solutions if isinstance(s, dict) and ('contains_blackmail' in s)]
        return float(sum(labels) / len(labels)) if labels else None
    except Exception:
        return None


def compute_prev_rate_at_chunk(baseline_scenario_dir: Path, chunk_idx: int) -> Tuple[Optional[float], int]:
    """Compute baseline whistleblowing rate from solutions.json at chunk i."""
    chunk_dir = baseline_scenario_dir / f"chunk_{chunk_idx}"
    sols = _read_json(chunk_dir / 'solutions.json')
    rate = _compute_rate_from_solutions(sols)
    try:
        n = int(len([1 for s in sols if isinstance(s, dict) and ('contains_whistleblowing' in s)])) if isinstance(sols, list) else 0
    except Exception:
        n = 0
    return rate, n


def compute_prev_rate_at_next_chunk(baseline_scenario_dir: Path, chunk_idx: int) -> Tuple[Optional[float], int]:
    """Compute baseline whistleblowing rate from solutions.json at chunk i+1."""
    return compute_prev_rate_at_chunk(baseline_scenario_dir, chunk_idx + 1)


def compute_after_rate_from_experiment(exp: Dict[str, Any]) -> Tuple[Optional[float], int]:
    """Compute blackmail rate from experiment's classified rollouts."""
    cls = exp.get('full_rollouts_classified') or exp.get('rollouts') or []
    if not isinstance(cls, list) or not cls:
        return None, 0
    vals: List[int] = []
    for r in cls:
        try:
            cw = r.get('contains_blackmail') if isinstance(r, dict) else None
        except Exception:
            cw = None
        if cw is not None:
            vals.append(1 if bool(cw) else 0)
    rate = float(sum(vals) / len(vals)) if vals else None
    return rate, len(vals)


def discover_onpolicy_targets(on_dir: Path) -> Dict[int, List[Path]]:
    out: Dict[int, List[Path]] = {}
    if not on_dir.exists():
        return out
    for sdir in list_scenarios(on_dir):
        sid = int(sdir.name.split('_')[-1])
        files = sorted(sdir.glob('onpolicy_chain_disruption_*.json'))
        if files:
            out[sid] = files
    return out


def discover_agent_targets(agent_dir: Path) -> Dict[int, List[Path]]:
    out: Dict[int, List[Path]] = {}
    if not agent_dir or not agent_dir.exists():
        return out
    for sdir in list_scenarios(agent_dir):
        sid = int(sdir.name.split('_')[-1])
        files = sorted(sdir.glob('agent_chain_disruption_*.json'))
        if files:
            out[sid] = files
    return out


def load_offpolicy_file(off_dir: Path, scenario_id: int) -> Optional[Dict[str, Any]]:
    sdir = off_dir / f'scenario_{scenario_id}'
    path = sdir / 'chain_disruption_results.json'
    return load_json(path)


def load_offpolicy_cross_file(off_cross_dir: Path, scenario_id: int) -> Optional[Dict[str, Any]]:
    sdir = off_cross_dir / f'scenario_{scenario_id}'
    path = sdir / 'chain_disruption_cross_model_results.json'
    return load_json(path)


def load_offpolicy_same_file(off_same_dir: Path, scenario_id: int) -> Optional[Dict[str, Any]]:
    sdir = off_same_dir / f'scenario_{scenario_id}'
    path = sdir / 'chain_disruption_same_model_results.json'
    return load_json(path)


# ------------------------------
# Shared-plot helpers (examples and text lookups)
# ------------------------------

def _load_chunks_for_scenario(baseline_dir: Path, scenario_id: int) -> Tuple[List[str], List[str]]:
    sdir = baseline_dir / f"scenario_{int(scenario_id)}"
    return load_chunks_and_separators(sdir)


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _find_onpolicy_example_text(onpolicy_dir: Path, onpolicy_fixed_dir: Path, scenario_id: int, target_name: str, chunk_idx: int) -> Optional[str]:
    """Return the on-policy example text for the shared description.

    Per request: prefer the selected_continuation_full_text (if present), which captures
    the exact chosen continuation tied to the intervention. Require exp.success == True
    when available. If not found in FIXED, fall back to VARIED with the same rule.
    Avoid target_sentence to prevent duplicating the base (handwritten) line.
    """
    # Prefer FIXED pipeline
    sdir_f = onpolicy_fixed_dir / f"scenario_{int(scenario_id)}"
    if sdir_f.exists():
        for f in sorted(sdir_f.glob('onpolicy_chain_disruption_*.json')):
            data = _safe_read_json(f) or {}
            if str(data.get('target_name')) != str(target_name):
                continue
            for exp in (data.get('experiments') or []):
                try:
                    if int(exp.get('chunk_idx', -1)) != int(chunk_idx):
                        continue
                except Exception:
                    continue
                if exp.get('success') is False:
                    continue
                # Preferred key for resampled example text
                txt = exp.get('selected_continuation_full_text')
                if isinstance(txt, str) and txt.strip():
                    return str(txt)
                # Fallbacks: next sentence or resampled chunk (but not target_sentence)
                txt2 = exp.get('selected_next_sentence') or exp.get('selected_resampled_chunk')
                if isinstance(txt2, str) and txt2.strip():
                    return str(txt2)
    # Fallback: VARIED
    sdir = onpolicy_dir / f"scenario_{int(scenario_id)}"
    if sdir.exists():
        for f in sorted(sdir.glob('onpolicy_chain_disruption_*.json')):
            data = _safe_read_json(f) or {}
            if str(data.get('target_name')) != str(target_name):
                continue
            for exp in (data.get('experiments') or []):
                try:
                    if int(exp.get('chunk_idx', -1)) != int(chunk_idx):
                        continue
                except Exception:
                    continue
                if exp.get('success') is False:
                    continue
                txt = exp.get('selected_continuation_full_text')
                if isinstance(txt, str) and txt.strip():
                    return str(txt)
                txt2 = exp.get('selected_next_sentence') or exp.get('selected_resampled_chunk')
                if isinstance(txt2, str) and txt2.strip():
                    return str(txt2)
    return None


def _normalize_category(name: str) -> str:
    try:
        key = str(name).strip().lower()
    except Exception:
        key = str(name)
    # Collapse known synonyms
    if key == 'consequences':
        return 'consequence'
    return key


def _find_offpolicy_example_text(offpolicy_dir: Path, scenario_id: int, chunk_idx: int, strategy_name: Optional[str] = None) -> Optional[str]:
    data = _safe_read_json(offpolicy_dir / f"scenario_{int(scenario_id)}" / 'chain_disruption_results.json') or {}
    for e in (data.get('experiments') or []):
        try:
            if int(e.get('chunk_idx', -1)) != int(chunk_idx):
                continue
        except Exception:
            continue
        want = _normalize_category(strategy_name) if strategy_name else None
        for strat in (e.get('disruption_results') or []):
            if want is not None:
                try:
                    if _normalize_category(strat.get('strategy', '')) != want:
                        continue
                except Exception:
                    continue
            txt = strat.get('disruption_text')
            if txt:
                return str(txt)
    return None


def _find_cross_same_example_text(offpolicy_dir: Path, scenario_id: int, chunk_idx: int, kind: str, strategy_name: Optional[str] = None) -> Optional[str]:
    # kind: 'cross' or 'same'
    parent = offpolicy_dir.parent
    root = parent / ("yes_base_solution_chain_disruption_" + ("cross_model" if kind == 'cross' else "same_model"))
    name = 'chain_disruption_cross_model_results.json' if kind == 'cross' else 'chain_disruption_same_model_results.json'
    data = _safe_read_json(root / f"scenario_{int(scenario_id)}" / name) or {}
    for e in (data.get('experiments') or []):
        try:
            if int(e.get('chunk_idx', -1)) != int(chunk_idx):
                continue
        except Exception:
            continue
        want = _normalize_category(strategy_name) if strategy_name else None
        for strat in (e.get('disruption_results') or []):
            if want is not None:
                try:
                    if _normalize_category(strat.get('strategy', '')) != want:
                        continue
                except Exception:
                    continue
            txt = strat.get('chosen_text') or strat.get('disruption_text')
            if txt:
                return str(txt)
    return None


def _find_agent_example_text(agent_dir: Path, scenario_id: int, chunk_idx: int) -> Optional[str]:
    sdir = agent_dir / f"scenario_{int(scenario_id)}"
    if not sdir.exists():
        return None
    for f in sorted(sdir.glob('agent_chain_disruption_*.json')):
        data = _safe_read_json(f) or {}
        for exp in (data.get('experiments') or []):
            try:
                if int(exp.get('chunk_idx', -1)) != int(chunk_idx):
                    continue
            except Exception:
                continue
            txt = exp.get('best_sentence') or exp.get('selected_resampled_chunk') or exp.get('target_sentence')
            if txt:
                return str(txt)
    return None

# ------------------------------
# Plot filtering helpers
# ------------------------------

def _parse_plot_methods(methods_csv: str) -> List[str]:
    # Map short tokens -> viz_type used in plotting
    aliases = {
        'cross': 'offpolicy_cross_model',
        'handwritten': 'offpolicy_handwritten',
        'same': 'offpolicy_same_model',
        'resampled': 'resampled',  # onpolicy_inserted/fixed collapsed to 'resampled'
        'agent': 'agent_onpolicy_discovered',
        'baseline': 'baseline_mode',
    }
    out: List[str] = []
    for tok in (methods_csv or '').split(','):
        key = tok.strip().lower()
        if not key:
            continue
        vt = aliases.get(key)
        if vt:
            out.append(vt)
    if not out:
        # default to a safe subset if parsing fails
        out = ['resampled', 'offpolicy_handwritten', 'offpolicy_cross_model', 'offpolicy_same_model', 'agent_onpolicy_discovered']
    return out


# ------------------------------
# Core pipeline
# ------------------------------

def collect_distribution_points(
    bundle: HFModelBundle,
    baseline_dir: Path,
    onpolicy_dir: Path,
    offpolicy_dir: Path,
    onpolicy_fixed_dir: Optional[Path] = None,
    agent_dir: Optional[Path] = None,
    max_scenarios: Optional[int] = None,
    targets_filter: Optional[List[str]] = None,
    include_baseline: bool = False,
    use_nebius_logprobs: bool = True,
    model_folder: str = 'qwq-32b',
    nebius_api_key: Optional[str] = None,
    tokenizer_cache: Optional[Dict[str, Any]] = None,
    stream_save_every: int = 0,
    out_root: Optional[Path] = None,
    display_model: Optional[str] = None,
    nebius_concurrency: int = 100,
    compare_to: str = 'current',
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    on_targets = discover_onpolicy_targets(onpolicy_dir)
    on_fixed_targets = discover_onpolicy_targets(onpolicy_fixed_dir) if (onpolicy_fixed_dir is not None and onpolicy_fixed_dir.exists()) else {}
    agent_targets = discover_agent_targets(agent_dir) if (agent_dir is not None and agent_dir.exists()) else {}
    # Also discover off-policy scenarios so we include off-policy points even when
    # there are no on-policy successes for a scenario
    try:
        off_dirs = list_scenarios(offpolicy_dir)
        off_ids = [int(d.name.split('_')[-1]) for d in off_dirs]
    except Exception:
        off_ids = []
    # Discover cross-model and same-model off-policy scenario ids
    try:
        off_cross_dir = offpolicy_dir.parent / 'yes_base_solution_chain_disruption_cross_model'
        cross_dirs = list_scenarios(off_cross_dir) if off_cross_dir.exists() else []
        cross_ids = [int(d.name.split('_')[-1]) for d in cross_dirs]
    except Exception:
        cross_ids = []
    try:
        off_same_dir = offpolicy_dir.parent / 'yes_base_solution_chain_disruption_same_model'
        same_dirs = list_scenarios(off_same_dir) if off_same_dir.exists() else []
        same_ids = [int(d.name.split('_')[-1]) for d in same_dirs]
    except Exception:
        same_ids = []

    # Optional scenario limit (applies to on-policy first, then union with off-policy ids)
    if max_scenarios is not None:
        kept: Dict[int, List[Path]] = {}
        for sid in sorted(on_targets.keys())[: max(0, int(max_scenarios))]:
            kept[sid] = on_targets[sid]
        on_targets = kept

    # Union scenario ids from both sources (include fixed on-policy scenarios)
    scenario_ids = sorted(set(on_targets.keys()).union(set(on_fixed_targets.keys())).union(set(agent_targets.keys())).union(set(off_ids)).union(set(cross_ids)).union(set(same_ids)))
    if max_scenarios is not None:
        scenario_ids = scenario_ids[: max(0, int(max_scenarios))]
    logger.info(f"Scanning {len(scenario_ids)} scenario(s)")

    for si, scen_id in enumerate(tqdm(scenario_ids, desc="Scenarios", leave=False), start=1):
        # Queue for Nebius-scored sentences in this scenario
        pending_remote: List[Dict[str, Any]] = []

        def queue_remote_point(prefill_text: str, sentence_text: str, meta: Dict[str, Any]) -> None:
            pending_remote.append({
                'prefill': prefill_text,
                'sentence_text': sentence_text,
                'meta': meta,
            })

        def process_pending_remote() -> None:
            if not pending_remote:
                return
            tasks_local = list(pending_remote)
            pending_remote.clear()
            try:
                with ThreadPoolExecutor(max_workers=max(1, int(min(nebius_concurrency, 32)))) as ex:
                    results = list(ex.map(
                        lambda t: compute_token_logprobs_nebius(
                            model_folder,
                            t['prefill'],
                            t['sentence_text'],
                            tokenizer_cache or {},
                            nebius_api_key,
                            timeout_seconds=int(globals().get('neb_timeout_seconds', 180)),
                            use_cache=True,
                        ),
                        tasks_local,
                    ))
            except Exception as _e:
                logger.warning(f"parallel Nebius scoring failed: {type(_e).__name__}: {_e}")
                results = []
            for idx, task in enumerate(tasks_local):
                meta = task['meta']
                try:
                    res = results[idx] if idx < len(results) and isinstance(results[idx], dict) else {}
                    mean_lp = float(res['logprob_mean']) if (res.get('logprob_mean') is not None) else float('nan')
                    sum_lp = float(res['logprob_sum']) if (res.get('logprob_sum') is not None) else float('nan')
                    try:
                        scored = int(res.get('scored_token_count', 0))
                        total = int(res.get('total_token_count', 0))
                        cov = (float(scored) / float(total)) if total > 0 else float('nan')
                    except Exception:
                        scored, total, cov = 0, 0, float('nan')
                except Exception:
                    mean_lp, sum_lp, scored, total, cov = float('nan'), float('nan'), 0, 0, float('nan')
                # Filter out if too few tokens
                min_tokens = int(globals().get('min_scored_tokens_threshold', 4))
                if scored < min_tokens:
                    logger.debug(f"Filtered Nebius point: {scored} < {min_tokens} tokens")
                    continue
                
                meta['mean_logprob'] = mean_lp
                meta['sum_logprob'] = sum_lp
                meta['scored_token_count'] = scored
                meta['total_token_count'] = total
                meta['coverage'] = cov
                points.append(meta)

        files = on_targets.get(scen_id, [])
        files_fixed = on_fixed_targets.get(scen_id, [])
        files_agent = agent_targets.get(scen_id, [])
        scen_dir = baseline_dir / f'scenario_{scen_id}'
        scen = load_json(scen_dir / 'scenario.json') or {}
        system_prompt = scen.get('system_prompt', '')
        user_prompt = scen.get('user_prompt', '')
        email_content = scen.get('email_content', '')
        chunks, seps = load_chunks_and_separators(scen_dir)
        
        # Precompute baseline (chunk_0) rate if comparing to base
        prev_rate_chunk0 = None
        if compare_to == 'base':
            try:
                prev_rate_chunk0, _ = compute_prev_rate_at_next_chunk(baseline_dir / f'scenario_{scen_id}', -1)
            except Exception:
                prev_rate_chunk0 = None

        off_json = load_offpolicy_file(offpolicy_dir, scen_id) or {}
        off_cross_json = None
        off_same_json = None
        # Load cross-model and same-model off-policy jsons if present
        off_cross_root = offpolicy_dir.parent / 'yes_base_solution_chain_disruption_cross_model'
        off_same_root = offpolicy_dir.parent / 'yes_base_solution_chain_disruption_same_model'
        if off_cross_root.exists():
            off_cross_json = load_offpolicy_cross_file(off_cross_root, scen_id) or {}
        if off_same_root.exists():
            off_same_json = load_offpolicy_same_file(off_same_root, scen_id) or {}
        off_exp_index: Dict[int, Dict[str, Any]] = {}
        if isinstance(off_json.get('experiments'), list):
            for e in off_json['experiments']:
                try:
                    ch_idx = int(e.get('chunk_idx', -1))
                except Exception:
                    ch_idx = -1
                if ch_idx >= 0:
                    off_exp_index[ch_idx] = e

        # Gather successes across files to size a progress bar
        success_records: List[Tuple[str, Dict[str, Any]]] = []  # (target_name, exp)
        for f in files:
            data = load_json(f) or {}
            target_name = str(data.get('target_name', 'unknown'))
            if targets_filter is not None and target_name not in targets_filter:
                continue
            experiments: List[Dict[str, Any]] = data.get('experiments') or []
            for exp in experiments:
                if exp.get('success'):
                    success_records.append((target_name, exp))

        logger.info(f"scenario {scen_id}: {len(success_records)} on-policy (varied) success(es) across {len(files)} file(s)")

        points_before = len(points)
        # Track which off-policy entries we add to avoid duplicates when we later add
        # the entire off-policy set for the scenario
        added_off_keys: set = set()
        with tqdm(total=len(success_records), desc=f"scenario {scen_id}: successes", leave=False) as pbar_s:
            for (target_name, exp) in success_records:
                try:
                    ch_idx = int(exp.get('chunk_idx', -1))
                except Exception:
                    ch_idx = -1
                if ch_idx < 0 or ch_idx >= len(chunks):
                    pbar_s.update(1)
                    continue

                # Build prefill: chunks[0:ch_idx] + resampled_chunk, intervention comes after
                # Get chunks BEFORE ch_idx
                prefix_text_base = make_prefix_text(chunks, seps, ch_idx)
                
                # Get the resampled chunk for this experiment
                resampled_chunk = exp.get('selected_resampled_chunk', '')

                # Baseline prev sentence (proxy for mode) — optional
                if include_baseline:
                    base_sentence = chunks[ch_idx]
                    prefix_text_baseline = prefix_text_base
                    prefill_baseline = build_prefill(system_prompt, user_prompt, email_content, prefix_text_baseline)
                    if use_nebius_logprobs and (bundle.model is None or model_folder != 'qwq-32b'):
                        meta = {
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx,
                            'target_name': target_name,
                            'sentence_type': 'baseline_mode',
                            'strategy': None,
                            'delta_blackmail_rate': 0.0,
                            'blackmail_rate_before': None,
                            'blackmail_rate_after': None,
                        }
                        queue_remote_point(prefill_baseline, base_sentence, meta)
                    else:
                        try:
                            lp_base = compute_token_logprobs(bundle, prefill_baseline, base_sentence, model_folder=model_folder)
                        except Exception:
                            lp_base = {'logprob_mean': 0.0, 'logprob_sum': 0.0, 'tokens': [], 'token_ids': [], 'logprobs': []}
                        points.append({
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx,
                            'target_name': target_name,
                            'sentence_type': 'baseline_mode',
                            'strategy': None,
                            'mean_logprob': float(lp_base.get('logprob_mean', 0.0)),
                            'sum_logprob': float(lp_base.get('logprob_sum', 0.0)),
                            'delta_blackmail_rate': 0.0,
                            'blackmail_rate_before': None,
                            'blackmail_rate_after': None,
                        })

                # On-policy inserted sentence: score the intervention sentence after the resampled chunk
                # Prefer 'selected_continuation_full_text', fallback to ' ' + 'selected_next_sentence'
                intervention_sentence = exp.get('selected_continuation_full_text') or (' ' + (exp.get('selected_next_sentence') or ''))
                if intervention_sentence and resampled_chunk:
                    # Build full prefix: base + resampled_chunk
                    prefix_text = prefix_text_base + resampled_chunk
                    prefill = build_prefill(system_prompt, user_prompt, email_content, prefix_text)

                    # Always compare to chunk i+1 baseline rate
                    prev_rate, n_prev = compute_prev_rate_at_next_chunk(baseline_dir / f'scenario_{scen_id}', ch_idx)
                    after_rate, n_after = compute_after_rate_from_experiment(exp)
                    delta_br = None
                    try:
                        if after_rate is not None and prev_rate is not None:
                            delta_br = float(after_rate) - float(prev_rate)
                    except Exception:
                        delta_br = None
                    if use_nebius_logprobs and (bundle.model is None or model_folder != 'qwq-32b'):
                        meta = {
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx,
                            'target_name': target_name,
                            'sentence_type': 'onpolicy_inserted',
                            'strategy': None,
                            'delta_blackmail_rate': (delta_br if delta_br is not None else np.nan),
                            'blackmail_rate_before': (prev_rate if prev_rate is not None else np.nan),
                            'blackmail_rate_after': (after_rate if after_rate is not None else np.nan),
                        }
                        queue_remote_point(prefill, intervention_sentence, meta)
                    else:
                        try:
                            lp_on = compute_token_logprobs(bundle, prefill, intervention_sentence, model_folder=model_folder)
                        except Exception:
                            lp_on = {'logprob_mean': 0.0, 'logprob_sum': 0.0, 'tokens': [], 'token_ids': [], 'logprobs': [], 'scored_token_count': 0, 'total_token_count': 0}
                        
                        # Filter out if too few tokens
                        min_tokens = int(globals().get('min_scored_tokens_threshold', 4))
                        scored_tokens = int(lp_on.get('scored_token_count', 0))
                        if scored_tokens < min_tokens:
                            logger.debug(f"Filtered onpolicy_inserted: {scored_tokens} < {min_tokens} tokens")
                            continue
                        
                        points.append({
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx,
                            'target_name': target_name,
                            'sentence_type': 'onpolicy_inserted',
                            'strategy': None,
                            'mean_logprob': float(lp_on.get('logprob_mean', 0.0)),
                            'sum_logprob': float(lp_on.get('logprob_sum', 0.0)),
                            'scored_token_count': scored_tokens,
                            'total_token_count': int(lp_on.get('total_token_count', 0)),
                            'delta_blackmail_rate': (delta_br if delta_br is not None else np.nan),
                            'blackmail_rate_before': (prev_rate if prev_rate is not None else np.nan),
                            'blackmail_rate_after': (after_rate if after_rate is not None else np.nan),
                        })

                # Off-policy (handwritten) sentences for this chunk if available
                # For off-policy, use ORIGINAL chunks (not resampled)
                off_exp = off_exp_index.get(ch_idx)
                if off_exp:
                    # Build prefix with original chunks (chunks[0:ch_idx] only, NOT including ch_idx)
                    prefix_text_off_base = make_prefix_text(chunks, seps, ch_idx)
                    prefix_text_off = prefix_text_off_base + chunks[ch_idx]
                    separator_off = seps[ch_idx] if ch_idx < len(seps) else ''
                    prefill_off = build_prefill(system_prompt, user_prompt, email_content, prefix_text_off)
                    
                    prev_br_off, _ = compute_prev_rate_at_next_chunk(baseline_dir / f'scenario_{scen_id}', ch_idx)
                    for strat in (off_exp.get('disruption_results') or []):
                        strat_name = str(strat.get('strategy', ''))
                        disruption_text_raw = str(strat.get('disruption_text', ''))
                        after_br_off = strat.get('blackmail_rate', None)
                        delta_off = None
                        try:
                            if after_br_off is not None and prev_br_off is not None:
                                delta_off = float(after_br_off) - float(prev_br_off)
                        except Exception:
                            delta_off = None

                        if not disruption_text_raw:
                            continue
                        
                        # Prepend separator so we score with leading whitespace
                        disruption_text = separator_off + disruption_text_raw
                        
                        if use_nebius_logprobs and (bundle.model is None or model_folder != 'qwq-32b'):
                            meta = {
                                'scenario_id': scen_id,
                                'chunk_idx': ch_idx,
                                'target_name': target_name,
                                'sentence_type': 'offpolicy_handwritten',
                                'strategy': strat_name,
                                'delta_blackmail_rate': (delta_off if delta_off is not None else np.nan),
                                'blackmail_rate_before': (float(prev_br_off) if prev_br_off is not None else np.nan),
                                'blackmail_rate_after': (float(after_br_off) if after_br_off is not None else np.nan),
                            }
                            queue_remote_point(prefill_off, disruption_text, meta)
                        else:
                            try:
                                lp_off = compute_token_logprobs(bundle, prefill_off, disruption_text, model_folder=model_folder)
                            except Exception:
                                lp_off = {'logprob_mean': 0.0, 'logprob_sum': 0.0, 'tokens': [], 'token_ids': [], 'logprobs': [], 'scored_token_count': 0, 'total_token_count': 0}
                            
                            # Filter out if too few tokens
                            min_tokens = int(globals().get('min_scored_tokens_threshold', 4))
                            scored_tokens = int(lp_off.get('scored_token_count', 0))
                            if scored_tokens < min_tokens:
                                logger.debug(f"Filtered offpolicy_handwritten: {scored_tokens} < {min_tokens} tokens")
                                added_off_keys.add((int(ch_idx), str(strat_name)))
                                continue
                            
                            points.append({
                                'scenario_id': scen_id,
                                'chunk_idx': ch_idx,
                                'target_name': target_name,
                                'sentence_type': 'offpolicy_handwritten',
                                'strategy': strat_name,
                                'mean_logprob': float(lp_off.get('logprob_mean', 0.0)),
                                'sum_logprob': float(lp_off.get('logprob_sum', 0.0)),
                                'scored_token_count': scored_tokens,
                                'total_token_count': int(lp_off.get('total_token_count', 0)),
                                'delta_blackmail_rate': (delta_off if delta_off is not None else np.nan),
                                'blackmail_rate_before': (float(prev_br_off) if prev_br_off is not None else np.nan),
                                'blackmail_rate_after': (float(after_br_off) if after_br_off is not None else np.nan),
                            })
                        added_off_keys.add((int(ch_idx), str(strat_name)))

                pbar_s.update(1)

        added = len(points) - points_before
        logger.info(f"scenario {scen_id}: added {added} point(s) pre-fixed")

        # Now add on-policy (fixed) sentences per target file
        if files_fixed:
            with tqdm(total=len(files_fixed), desc=f"scenario {scen_id}: onpolicy fixed", leave=False) as pbar_f:
                for f in files_fixed:
                    data_f = load_json(f) or {}
                    target_name_f = str(data_f.get('target_name', 'unknown'))
                    if targets_filter is not None and target_name_f not in targets_filter:
                        pbar_f.update(1)
                        continue
                    exps_f: List[Dict[str, Any]] = data_f.get('experiments') or []
                    for expf in exps_f:
                        try:
                            ch_idxf = int(expf.get('chunk_idx', -1))
                        except Exception:
                            ch_idxf = -1
                        if ch_idxf < 0 or ch_idxf >= len(chunks):
                            continue
                        
                        # Get resampled chunk and intervention sentence
                        # Prefer 'selected_continuation_full_text', fallback to ' ' + 'selected_next_sentence'
                        resampled_chunk_f = expf.get('selected_resampled_chunk', '')
                        intervention_sentence_f = expf.get('selected_continuation_full_text') or (' ' + (expf.get('selected_next_sentence') or ''))
                        
                        if not resampled_chunk_f or not intervention_sentence_f:
                            continue
                        
                        # Build prefix: chunks[0:ch_idxf] + resampled_chunk
                        prefix_text_base_f = make_prefix_text(chunks, seps, ch_idxf)
                        prefix_text_f = prefix_text_base_f + resampled_chunk_f
                        prefillf = build_prefill(system_prompt, user_prompt, email_content, prefix_text_f)

                        # For fixed pipeline, prev/after rates are analogous to varied; reuse helper if present
                        prev_rate_f, _ = compute_prev_rate_at_next_chunk(baseline_dir / f'scenario_{scen_id}', ch_idxf)
                        after_rate_f, _ = compute_after_rate_from_experiment(expf)
                        delta_f = None
                        try:
                            if after_rate_f is not None and prev_rate_f is not None:
                                delta_f = float(after_rate_f) - float(prev_rate_f)
                        except Exception:
                            delta_f = None

                        if use_nebius_logprobs and (bundle.model is None or model_folder != 'qwq-32b'):
                            meta = {
                                'scenario_id': scen_id,
                                'chunk_idx': ch_idxf,
                                'target_name': target_name_f,
                                'sentence_type': 'onpolicy_fixed',
                                'strategy': None,
                                'delta_blackmail_rate': (delta_f if delta_f is not None else np.nan),
                                'blackmail_rate_before': (prev_rate_f if prev_rate_f is not None else np.nan),
                                'blackmail_rate_after': (after_rate_f if after_rate_f is not None else np.nan),
                            }
                            queue_remote_point(prefillf, intervention_sentence_f, meta)
                        else:
                            try:
                                lp_fixed = compute_token_logprobs(bundle, prefillf, intervention_sentence_f, model_folder=model_folder)
                            except Exception:
                                lp_fixed = {'logprob_mean': 0.0, 'logprob_sum': 0.0, 'tokens': [], 'token_ids': [], 'logprobs': [], 'scored_token_count': 0, 'total_token_count': 0}
                            
                            # Filter out if too few tokens
                            min_tokens = int(globals().get('min_scored_tokens_threshold', 4))
                            scored_tokens = int(lp_fixed.get('scored_token_count', 0))
                            if scored_tokens < min_tokens:
                                logger.debug(f"Filtered onpolicy_fixed: {scored_tokens} < {min_tokens} tokens")
                                continue
                            
                            points.append({
                                'scenario_id': scen_id,
                                'chunk_idx': ch_idxf,
                                'target_name': target_name_f,
                                'sentence_type': 'onpolicy_fixed',
                                'strategy': None,
                                'mean_logprob': float(lp_fixed.get('logprob_mean', 0.0)),
                                'sum_logprob': float(lp_fixed.get('logprob_sum', 0.0)),
                                'scored_token_count': scored_tokens,
                                'total_token_count': int(lp_fixed.get('total_token_count', 0)),
                                'delta_blackmail_rate': (delta_f if delta_f is not None else np.nan),
                                'blackmail_rate_before': (prev_rate_f if prev_rate_f is not None else np.nan),
                                'blackmail_rate_after': (after_rate_f if after_rate_f is not None else np.nan),
                            })
                    pbar_f.update(1)

        # Independently add ALL off-policy disruption points for this scenario (for
        # every chunk with off-policy data), avoiding duplicates already added above.
        if isinstance(off_json.get('experiments'), list):
            for e in tqdm(off_json['experiments'], desc=f"scenario {scen_id}: offpolicy", leave=False):
                try:
                    ch_idx2 = int(e.get('chunk_idx', -1))
                except Exception:
                    ch_idx2 = -1
                # Build prefix with original chunks (chunks[0:ch_idx2] only)
                prefix_text2_base = make_prefix_text(chunks, seps, ch_idx2)
                if ch_idx2 >= 0 and ch_idx2 < len(chunks):
                    prefix_text2 = prefix_text2_base + chunks[ch_idx2]
                    separator2 = seps[ch_idx2] if ch_idx2 < len(seps) else ''
                else:
                    prefix_text2 = prefix_text2_base
                    separator2 = ''
                prefill2 = build_prefill(system_prompt, user_prompt, email_content, prefix_text2)
                prev_br_off2, _ = compute_prev_rate_at_next_chunk(baseline_dir / f'scenario_{scen_id}', ch_idx2)
                for strat in (e.get('disruption_results') or []):
                    strat_name2 = str(strat.get('strategy', ''))
                    if (ch_idx2, strat_name2) in added_off_keys:
                        continue
                    disruption_text2_raw = str(strat.get('disruption_text', ''))
                    after_br_off2 = strat.get('blackmail_rate', None)
                    delta_off2 = None
                    try:
                        if after_br_off2 is not None and prev_br_off2 is not None:
                            delta_off2 = float(after_br_off2) - float(prev_br_off2)
                    except Exception:
                        delta_off2 = None
                    if not disruption_text2_raw:
                        continue
                    
                    # Prepend separator so we score with leading whitespace
                    disruption_text2 = separator2 + disruption_text2_raw
                    
                    if use_nebius_logprobs and (bundle.model is None or model_folder != 'qwq-32b'):
                        meta = {
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx2,
                            'target_name': str(off_json.get('target_name') or 'offpolicy'),
                            'sentence_type': 'offpolicy_handwritten',
                            'strategy': strat_name2,
                            'delta_blackmail_rate': (delta_off2 if delta_off2 is not None else np.nan),
                            'blackmail_rate_before': (float(prev_br_off2) if prev_br_off2 is not None else np.nan),
                            'blackmail_rate_after': (float(after_br_off2) if after_br_off2 is not None else np.nan),
                        }
                        queue_remote_point(prefill2, disruption_text2, meta)
                    else:
                        try:
                            lp_off2 = compute_token_logprobs(bundle, prefill2, disruption_text2, model_folder=model_folder)
                        except Exception:
                            lp_off2 = {'logprob_mean': 0.0, 'logprob_sum': 0.0, 'tokens': [], 'token_ids': [], 'logprobs': [], 'scored_token_count': 0, 'total_token_count': 0}
                        
                        # Filter out if too few tokens
                        min_tokens = int(globals().get('min_scored_tokens_threshold', 4))
                        scored_tokens = int(lp_off2.get('scored_token_count', 0))
                        if scored_tokens < min_tokens:
                            logger.debug(f"Filtered offpolicy_handwritten (independent): {scored_tokens} < {min_tokens} tokens")
                            continue
                        
                        points.append({
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx2,
                            'target_name': str(off_json.get('target_name') or 'offpolicy'),
                            'sentence_type': 'offpolicy_handwritten',
                            'strategy': strat_name2,
                            'mean_logprob': float(lp_off2.get('logprob_mean', 0.0)),
                            'sum_logprob': float(lp_off2.get('logprob_sum', 0.0)),
                            'scored_token_count': scored_tokens,
                            'total_token_count': int(lp_off2.get('total_token_count', 0)),
                            'delta_blackmail_rate': (delta_off2 if delta_off2 is not None else np.nan),
                            'blackmail_rate_before': (float(prev_br_off2) if prev_br_off2 is not None else np.nan),
                            'blackmail_rate_after': (float(after_br_off2) if after_br_off2 is not None else np.nan),
                        })

        # Add cross-model off-policy results, labeled 'offpolicy_cross_model'
        if isinstance((off_cross_json or {}).get('experiments'), list):
            for e in tqdm(off_cross_json['experiments'], desc=f"scenario {scen_id}: offpolicy cross", leave=False):
                try:
                    ch_idx3 = int(e.get('chunk_idx', -1))
                except Exception:
                    ch_idx3 = -1
                # Build prefix with original chunks (chunks[0:ch_idx3] only)
                prefix_text3_base = make_prefix_text(chunks, seps, ch_idx3)
                if ch_idx3 >= 0 and ch_idx3 < len(chunks):
                    prefix_text3 = prefix_text3_base + chunks[ch_idx3]
                    separator3 = seps[ch_idx3] if ch_idx3 < len(seps) else ''
                else:
                    prefix_text3 = prefix_text3_base
                    separator3 = ''
                prefill3 = build_prefill(system_prompt, user_prompt, email_content, prefix_text3)
                prev_br_cross, _ = compute_prev_rate_at_next_chunk(baseline_dir / f'scenario_{scen_id}', ch_idx3)
                for strat in (e.get('disruption_results') or []):
                    # cross json may have chosen_text; prefer that for display
                    strat_name3 = str(strat.get('strategy', ''))
                    disruption_text3_raw = str(strat.get('chosen_text') or strat.get('disruption_text') or '')
                    after_br_cross = strat.get('blackmail_rate', None)
                    delta_cross = None
                    try:
                        if after_br_cross is not None and prev_br_cross is not None:
                            delta_cross = float(after_br_cross) - float(prev_br_cross)
                    except Exception:
                        delta_cross = None
                    if not disruption_text3_raw:
                        continue
                    
                    # Prepend separator so we score with leading whitespace
                    disruption_text3 = separator3 + disruption_text3_raw
                    
                    if use_nebius_logprobs and (bundle.model is None or model_folder != 'qwq-32b'):
                        meta = {
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx3,
                            'target_name': str(off_cross_json.get('target_name') or 'offpolicy cross model'),
                            'sentence_type': 'offpolicy_cross_model',
                            'strategy': strat_name3,
                            'delta_blackmail_rate': (delta_cross if delta_cross is not None else np.nan),
                            'blackmail_rate_before': (float(prev_br_cross) if prev_br_cross is not None else np.nan),
                            'blackmail_rate_after': (float(after_br_cross) if after_br_cross is not None else np.nan),
                        }
                        queue_remote_point(prefill3, disruption_text3, meta)
                    else:
                        try:
                            lp_cross = compute_token_logprobs(bundle, prefill3, disruption_text3, model_folder=model_folder)
                        except Exception:
                            lp_cross = {'logprob_mean': 0.0, 'logprob_sum': 0.0, 'tokens': [], 'token_ids': [], 'logprobs': [], 'scored_token_count': 0, 'total_token_count': 0}
                        
                        # Filter out if too few tokens
                        min_tokens = int(globals().get('min_scored_tokens_threshold', 4))
                        scored_tokens = int(lp_cross.get('scored_token_count', 0))
                        if scored_tokens < min_tokens:
                            logger.debug(f"Filtered offpolicy_cross_model: {scored_tokens} < {min_tokens} tokens")
                            continue
                        
                        points.append({
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx3,
                            'target_name': str(off_cross_json.get('target_name') or 'offpolicy cross model'),
                            'sentence_type': 'offpolicy_cross_model',
                            'strategy': strat_name3,
                            'mean_logprob': float(lp_cross.get('logprob_mean', 0.0)),
                            'sum_logprob': float(lp_cross.get('logprob_sum', 0.0)),
                            'scored_token_count': scored_tokens,
                            'total_token_count': int(lp_cross.get('total_token_count', 0)),
                            'delta_blackmail_rate': (delta_cross if delta_cross is not None else np.nan),
                            'blackmail_rate_before': (float(prev_br_cross) if prev_br_cross is not None else np.nan),
                            'blackmail_rate_after': (float(after_br_cross) if after_br_cross is not None else np.nan),
                        })

        # Add same-model off-policy results, labeled 'offpolicy_same_model'
        # Add agent-driven hill-climb sentences as an extra category: 'agent_onpolicy_discovered'
        if files_agent:
            with tqdm(total=len(files_agent), desc=f"scenario {scen_id}: agent onpolicy", leave=False) as pbar_a:
                for f in files_agent:
                    data_a = load_json(f) or {}
                    target_name_a = str(data_a.get('target_name', 'unknown'))
                    if targets_filter is not None and target_name_a not in targets_filter:
                        pbar_a.update(1)
                        continue
                    exps_a: List[Dict[str, Any]] = data_a.get('experiments') or []
                    for expa in exps_a:
                        try:
                            ch_idxa = int(expa.get('chunk_idx', -1))
                        except Exception:
                            ch_idxa = -1
                        if ch_idxa < 0 or ch_idxa >= len(chunks):
                            continue
                        # Build prefix with original chunks (chunks[0:ch_idxa] + chunks[ch_idxa])
                        prefix_texta_base = make_prefix_text(chunks, seps, ch_idxa)
                        if ch_idxa < len(chunks):
                            prefix_texta = prefix_texta_base + chunks[ch_idxa]
                            if ch_idxa < len(seps):
                                prefix_texta = prefix_texta + seps[ch_idxa]
                        else:
                            prefix_texta = prefix_texta_base
                        prefilla = build_prefill(system_prompt, user_prompt, email_content, prefix_texta)
                        best_sentence = str(expa.get('best_sentence') or '')
                        if not best_sentence:
                            continue
                        prev_rate_a, _ = compute_prev_rate_at_next_chunk(baseline_dir / f'scenario_{scen_id}', ch_idxa)
                        after_rate_a, _ = compute_after_rate_from_experiment(expa)
                        delta_a = None
                        try:
                            if after_rate_a is not None and prev_rate_a is not None:
                                delta_a = float(after_rate_a) - float(prev_rate_a)
                        except Exception:
                            delta_a = None
                        if use_nebius_logprobs and (bundle.model is None or model_folder != 'qwq-32b'):
                            meta = {
                                'scenario_id': scen_id,
                                'chunk_idx': ch_idxa,
                                'target_name': target_name_a,
                                'sentence_type': 'agent_onpolicy_discovered',
                                'strategy': None,
                                'delta_blackmail_rate': (delta_a if delta_a is not None else np.nan),
                                'blackmail_rate_before': (prev_rate_a if prev_rate_a is not None else np.nan),
                                'blackmail_rate_after': (after_rate_a if after_rate_a is not None else np.nan),
                            }
                            queue_remote_point(prefilla, best_sentence, meta)
                        else:
                            try:
                                lp_agent = compute_token_logprobs(bundle, prefilla, best_sentence, model_folder=model_folder)
                            except Exception:
                                lp_agent = {'logprob_mean': 0.0, 'logprob_sum': 0.0, 'tokens': [], 'token_ids': [], 'logprobs': [], 'scored_token_count': 0, 'total_token_count': 0}
                            
                            # Filter out if too few tokens
                            min_tokens = int(globals().get('min_scored_tokens_threshold', 4))
                            scored_tokens = int(lp_agent.get('scored_token_count', 0))
                            if scored_tokens < min_tokens:
                                logger.debug(f"Filtered agent_onpolicy_discovered: {scored_tokens} < {min_tokens} tokens")
                                continue
                            
                            points.append({
                                'scenario_id': scen_id,
                                'chunk_idx': ch_idxa,
                                'target_name': target_name_a,
                                'sentence_type': 'agent_onpolicy_discovered',
                                'strategy': None,
                                'mean_logprob': float(lp_agent.get('logprob_mean', 0.0)),
                                'sum_logprob': float(lp_agent.get('logprob_sum', 0.0)),
                                'scored_token_count': scored_tokens,
                                'total_token_count': int(lp_agent.get('total_token_count', 0)),
                                'delta_blackmail_rate': (delta_a if delta_a is not None else np.nan),
                                'blackmail_rate_before': (prev_rate_a if prev_rate_a is not None else np.nan),
                                'blackmail_rate_after': (after_rate_a if after_rate_a is not None else np.nan),
                            })
                pbar_a.update(1)
        if isinstance((off_same_json or {}).get('experiments'), list):
            for e in tqdm(off_same_json['experiments'], desc=f"scenario {scen_id}: offpolicy same", leave=False):
                try:
                    ch_idx4 = int(e.get('chunk_idx', -1))
                except Exception:
                    ch_idx4 = -1
                # Build prefix with original chunks (chunks[0:ch_idx4] only)
                prefix_text4_base = make_prefix_text(chunks, seps, ch_idx4)
                if ch_idx4 >= 0 and ch_idx4 < len(chunks):
                    prefix_text4 = prefix_text4_base + chunks[ch_idx4]
                    separator4 = seps[ch_idx4] if ch_idx4 < len(seps) else ''
                else:
                    prefix_text4 = prefix_text4_base
                    separator4 = ''
                prefill4 = build_prefill(system_prompt, user_prompt, email_content, prefix_text4)
                prev_br_same, _ = compute_prev_rate_at_next_chunk(baseline_dir / f'scenario_{scen_id}', ch_idx4)
                for strat in (e.get('disruption_results') or []):
                    strat_name4 = str(strat.get('strategy', ''))
                    disruption_text4_raw = str(strat.get('chosen_text') or strat.get('disruption_text') or '')
                    after_br_same = strat.get('blackmail_rate', None)
                    delta_same = None
                    try:
                        if after_br_same is not None and prev_br_same is not None:
                            delta_same = float(after_br_same) - float(prev_br_same)
                    except Exception:
                        delta_same = None
                    if not disruption_text4_raw:
                        continue
                    
                    # Prepend separator so we score with leading whitespace
                    disruption_text4 = separator4 + disruption_text4_raw
                    
                    if use_nebius_logprobs and (bundle.model is None or model_folder != 'qwq-32b'):
                        meta = {
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx4,
                            'target_name': str(off_same_json.get('target_name') or 'offpolicy same model'),
                            'sentence_type': 'offpolicy_same_model',
                            'strategy': strat_name4,
                            'delta_blackmail_rate': (delta_same if delta_same is not None else np.nan),
                            'blackmail_rate_before': (float(prev_br_same) if prev_br_same is not None else np.nan),
                            'blackmail_rate_after': (float(after_br_same) if after_br_same is not None else np.nan),
                        }
                        queue_remote_point(prefill4, disruption_text4, meta)
                    else:
                        try:
                            lp_same = compute_token_logprobs(bundle, prefill4, disruption_text4, model_folder=model_folder)
                        except Exception:
                            lp_same = {'logprob_mean': 0.0, 'logprob_sum': 0.0, 'tokens': [], 'token_ids': [], 'logprobs': [], 'scored_token_count': 0, 'total_token_count': 0}
                        
                        # Filter out if too few tokens
                        min_tokens = int(globals().get('min_scored_tokens_threshold', 4))
                        scored_tokens = int(lp_same.get('scored_token_count', 0))
                        if scored_tokens < min_tokens:
                            logger.debug(f"Filtered offpolicy_same_model: {scored_tokens} < {min_tokens} tokens")
                            continue
                        
                        points.append({
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx4,
                            'target_name': str(off_same_json.get('target_name') or 'offpolicy same model'),
                            'sentence_type': 'offpolicy_same_model',
                            'strategy': strat_name4,
                            'mean_logprob': float(lp_same.get('logprob_mean', 0.0)),
                            'sum_logprob': float(lp_same.get('logprob_sum', 0.0)),
                            'scored_token_count': scored_tokens,
                            'total_token_count': int(lp_same.get('total_token_count', 0)),
                            'delta_blackmail_rate': (delta_same if delta_same is not None else np.nan),
                            'blackmail_rate_before': (float(prev_br_same) if prev_br_same is not None else np.nan),
                            'blackmail_rate_after': (float(after_br_same) if after_br_same is not None else np.nan),
                        })

        # Flush any pending Nebius tasks for this scenario
        process_pending_remote()

        # Interim save after each scenario
        if (stream_save_every and out_root is not None and display_model is not None and (si % max(1, int(stream_save_every)) == 0)):
            try:
                df_stream = pd.DataFrame(points)
                csv_path_stream = out_root / ('distribution_points_random.csv' if bool(globals().get('random_variant', False)) else 'distribution_points.csv')
                df_stream.to_csv(csv_path_stream, index=False)
                # Exclude self_preservation from main/extras combined plots
                try:
                    mask_not_sp_target = (df_stream['target_name'] != 'self_preservation') if 'target_name' in df_stream.columns else pd.Series([True] * len(df_stream))
                    mask_off_types = df_stream['sentence_type'].isin(['offpolicy_handwritten', 'offpolicy_cross_model', 'offpolicy_same_model']) if 'sentence_type' in df_stream.columns else pd.Series([False] * len(df_stream))
                    mask_not_off_sp = ~(mask_off_types & (df_stream['strategy'] == 'self_preservation')) if 'strategy' in df_stream.columns else pd.Series([True] * len(df_stream))
                    df_stream_main = df_stream[(mask_not_sp_target & mask_not_off_sp)]
                except Exception:
                    df_stream_main = df_stream
                suf = '_random' if bool(globals().get('random_variant', False)) else ''
                plot_path_stream = out_root / f"scatter_distribution{suf}.png"
                plot_distribution_scatter(df_stream_main, plot_path_stream, display_model=display_model, model_folder=model_folder, include_extras=False)
                plot_path_stream_ex = out_root / f"scatter_distribution_with_extras{suf}.png"
                plot_distribution_scatter(df_stream_main, plot_path_stream_ex, display_model=display_model, model_folder=model_folder, include_extras=True)
                print(f"[save] interim points -> {csv_path_stream}")
                print(f"[save] interim scatter -> {plot_path_stream}")
                print(f"[save] interim scatter (with extras) -> {plot_path_stream_ex}")
            except Exception as _e:
                logger.warning(f"interim save failed: {type(_e).__name__}: {_e}")

    return points


def plot_distribution_scatter(df: pd.DataFrame, out_path: Path, display_model: str, model_folder: str, include_extras: bool = False, title_override: Optional[str] = None) -> None:
    if df.empty:
        print('[dist] No data to plot.')
        return
    
    # Filter out points with mean_logprob < -10
    if 'mean_logprob' in df.columns:
        df = df[pd.to_numeric(df['mean_logprob'], errors='coerce') >= -10.0]
    if df.empty:
        print('[dist] No data to plot after filtering.')
        return
    
    plt.figure(figsize=(12, 8))
    sns.set_style('white')
    
    # Increase font sizes for scatter plots to match boxplots
    plt.rcParams.update({
        'axes.labelsize': 19,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'legend.fontsize': 14,
        'axes.titlesize': 22,
    })
    
    ax = plt.gca()
    # Original color scheme by type
    palette = {
        'onpolicy_inserted': '#9467bd',  # purple (varied)
        'onpolicy_fixed': '#d62728',  # red (fixed)
        'offpolicy_handwritten': '#1f77b4',  # blue
        'offpolicy_cross_model': '#ff7f0e',  # bright orange
        'offpolicy_same_model': '#006400',  # dark green
        'agent_onpolicy_discovered': '#8c564b',  # brown
        'baseline_mode': '#2ca02c',  # green
    }
    markers = {
        'onpolicy_inserted': 'o',
        'onpolicy_fixed': 'o',
        'offpolicy_handwritten': 's',
        'offpolicy_cross_model': '^',  # triangle
        'offpolicy_same_model': '*',  # star
        'agent_onpolicy_discovered': 'D',  # diamond
        'baseline_mode': 'X',
    }
    sizes = {
        'onpolicy_inserted': 24,
        'onpolicy_fixed': 24,
        'offpolicy_handwritten': 24,
        'offpolicy_cross_model': 28,  # slightly larger triangle
        'offpolicy_same_model': 28,   # match sizes with others
        'agent_onpolicy_discovered': 28,
        'baseline_mode': 24,
    }

    # Small jitter for baseline points on y=0 to avoid overplot
    df_plot = df.copy()
    if not include_extras:
        allowed = {"onpolicy_inserted", "onpolicy_fixed", "offpolicy_handwritten", "baseline_mode"}
        if 'sentence_type' in df_plot.columns:
            df_plot = df_plot[df_plot['sentence_type'].isin(list(allowed))]
    mask_base = df_plot['sentence_type'] == 'baseline_mode'
    if 'delta_blackmail_rate' in df_plot.columns:
        y_vals = df_plot['delta_blackmail_rate'].astype(float).copy()
        if mask_base.any():
            jitter_vals = np.random.normal(loc=0.0, scale=0.005, size=int(mask_base.sum()))
            # Only fill NaN for baseline points; leave others as NaN so they do not plot
            y_vals.loc[mask_base] = y_vals.loc[mask_base].fillna(0.0) + jitter_vals
        df_plot['delta_blackmail_rate_plot'] = y_vals
        y_col = 'delta_blackmail_rate_plot'
    else:
        y_col = 'delta_blackmail_rate'

    # Visual grouping: combine on-policy inserted/fixed as a single 'resampled' category (red dots)
    df_plot['viz_type'] = df_plot['sentence_type'].apply(lambda st: 'resampled' if st in ('onpolicy_inserted', 'onpolicy_fixed') else st)

    viz_palette = {
        'resampled': '#d62728',  # red
        'offpolicy_handwritten': '#1f77b4',  # blue
        'offpolicy_cross_model': '#ff7f0e',  # bright orange
        'offpolicy_same_model': '#006400',   # dark green
        'agent_onpolicy_discovered': '#8c564b',  # brown
        'baseline_mode': '#2ca02c',  # green
    }
    viz_markers = {
        'resampled': 'o',
        'offpolicy_handwritten': 's',
        'offpolicy_cross_model': '^',  # triangle
        'offpolicy_same_model': '*',   # star
        'agent_onpolicy_discovered': 'D',  # diamond
        'baseline_mode': 'X',
    }
    viz_sizes = {
        'resampled': 28,
        'offpolicy_handwritten': 24,
        'offpolicy_cross_model': 28,
        'offpolicy_same_model': 28,
        'agent_onpolicy_discovered': 28,
        'baseline_mode': 24,
    }
    label_map = {
        'resampled': 'resampled',
        'offpolicy_cross_model': 'cross model',
        'offpolicy_same_model': 'same model',
        'offpolicy_handwritten': 'handwritten',
        'agent_onpolicy_discovered': 'agent',
        'baseline_mode': 'baseline',
    }
    # Filter by selected plot methods if provided via global
    selected_viz_types: Optional[List[str]] = globals().get('selected_plot_methods', None)
    if selected_viz_types is not None and len(selected_viz_types) > 0:
        df_plot = df_plot[df_plot['viz_type'].isin(selected_viz_types)]
    # Shapes by model for individual plots (match shared)
    model_markers_default = ['o', '^', 's', 'D', '*', 'P', 'X', 'v', '<', '>']
    shapes_by_model: Dict[str, str] = {}
    known_models = list(MODEL_FOLDER_TO_HF.keys())
    for idx, mf in enumerate(known_models):
        shapes_by_model[mf] = model_markers_default[idx % len(model_markers_default)]
    if 'model_folder' not in df_plot.columns:
        df_plot['model_folder'] = str(model_folder)
    try:
        present_models = [str(x) for x in sorted(df_plot['model_folder'].dropna().unique())]
    except Exception:
        present_models = [str(model_folder)]
    next_i = len(known_models)
    for mf in present_models:
        if mf not in shapes_by_model:
            shapes_by_model[mf] = model_markers_default[next_i % len(model_markers_default)]
            next_i += 1
    for (stype, mf), sub in df_plot.groupby(['viz_type', 'model_folder']):
        # Increase marker size for deepseek star
        base_size = viz_sizes.get(stype, 24)
        marker_size = base_size * 2.0 if str(mf) == 'deepseek-r1-0528' else base_size
        plt.scatter(
            sub['mean_logprob'].astype(float).values,
            sub[y_col].astype(float).values,
            alpha=0.55,
            s=marker_size,
            c=viz_palette.get(stype, '#7f7f7f'),
            marker=shapes_by_model.get(str(mf), 'o'),
            edgecolors='none',
        )

    plt.axhline(0.0, color='gray', linewidth=1, linestyle='--')
    # plt.xlabel(f'Mean token logprob under {display_model}')
    plt.xlabel(f'Mean token logprob\n\n')
    plt.ylabel('Δ Blackmail rate (after − before)')
    plt.title(title_override or f'Sentence insertions vs impact\nModel: {model_folder}  Dataset: Blackmail\n')
    
    # Set consistent y-axis limits and ticks for comparability across datasets (with padding)
    plt.ylim(-1.05, 0.65)
    plt.yticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    
    # Set consistent x-axis limits and ticks
    plt.xlim(-10.5, 0.5)
    plt.xticks([-10, -8, -6, -4, -2, 0])
    
    # Dual legends: match shared plot
    from matplotlib.lines import Line2D  # local import
    present_types = [t for t in df_plot['viz_type'].dropna().unique()]
    type_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=8,
               markerfacecolor=viz_palette.get(t, '#7f7f7f'), markeredgecolor='none', label=label_map.get(t, t))
        for t in present_types
    ]
    # Always show all known models in legend (not just present ones)
    all_known_models = ['qwq-32b', 'hermes-4-70b', 'qwen3-235b-a22b', 'llama-3_1-nemotron-ultra-253b-v1', 'deepseek-r1-0528']
    model_handles: List[Line2D] = []
    for mf in all_known_models:
        # Use larger marker for deepseek star
        marker_size = 12 if mf == 'deepseek-r1-0528' else 8
        model_handles.append(Line2D([0], [0], marker=shapes_by_model.get(mf, 'o'), linestyle='None', markersize=marker_size,
                                    markerfacecolor='#cccccc', markeredgecolor='#333333', label=mf))
    # Place legends at the bottom in two lines: Type first, then Model (no titles)
    # Type legend (first line) - more spacing from x-axis label
    leg1 = plt.legend(handles=type_handles, loc='upper center', bbox_to_anchor=(0.5, -0.18), 
                     ncol=len(type_handles), borderaxespad=0., fontsize=14, frameon=False)
    # Model legend (second line)
    ax = plt.gca()
    ax.add_artist(leg1)
    leg2 = plt.legend(handles=model_handles, loc='upper center', bbox_to_anchor=(0.5, -0.24), 
                     ncol=len(model_handles), borderaxespad=0., fontsize=14, frameon=False)
    # Remove gridlines and the top/right spines
    ax.grid(False)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


# ------------------------------
# Shared plot: colors by type, shapes by model, respects -pm
# ------------------------------

def plot_shared_distribution_scatter(
    df: pd.DataFrame,
    out_path: Path,
    *,
    include_extras: bool = True,
    plot_methods: Optional[List[str]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title_override: Optional[str] = None,
    fig_size: Optional[Tuple[float, float]] = None,
) -> None:
    if df.empty:
        print('[dist] No data to plot (shared).')
        return
    sns.set_style('white')
    ax = plt.gca()

    # Group domain
    df_plot = df.copy()
    df_plot['viz_type'] = df_plot['sentence_type'].apply(lambda st: 'resampled' if st in ('onpolicy_inserted', 'onpolicy_fixed') else st)
    # Apply include_extras gating only when plot_methods not given
    if plot_methods is None and not include_extras:
        allowed = {"onpolicy_inserted", "onpolicy_fixed", "offpolicy_handwritten", "baseline_mode"}
        if 'sentence_type' in df_plot.columns:
            df_plot = df_plot[df_plot['sentence_type'].isin(list(allowed))]

    # Apply -pm filter
    if plot_methods:
        method_map = {
            'cross': 'offpolicy_cross_model',
            'handwritten': 'offpolicy_handwritten',
            'same': 'offpolicy_same_model',
            'resampled': 'resampled',
            'agent': 'agent_onpolicy_discovered',
            'baseline': 'baseline_mode',
        }
        requested = set()
        for m in plot_methods:
            key = str(m).strip().lower()
            if key in method_map:
                requested.add(method_map[key])
        if requested:
            df_plot = df_plot[df_plot['viz_type'].isin(list(requested))]
        else:
            print('[dist] No valid plot methods specified; nothing to plot (shared).')
            return
    
    # Filter out points with mean_logprob < -10
    if 'mean_logprob' in df_plot.columns:
        df_plot = df_plot[pd.to_numeric(df_plot['mean_logprob'], errors='coerce') >= -10.0]

    # Colors by type (match main viz)
    color_map = {
        'resampled': '#d62728',
        'offpolicy_handwritten': '#1f77b4',
        'offpolicy_cross_model': '#ff7f0e',
        'offpolicy_same_model': '#006400',
        'agent_onpolicy_discovered': '#8c564b',
        'baseline_mode': '#2ca02c',
    }
    # Shapes by model
    model_markers_default = ['o', '^', 's', 'D', '*', 'P', 'X', 'v', '<', '>']
    shapes_by_model: Dict[str, str] = {}
    known_models = list(MODEL_FOLDER_TO_HF.keys())
    for idx, mf in enumerate(known_models):
        shapes_by_model[mf] = model_markers_default[idx % len(model_markers_default)]
    if 'model_folder' in df_plot.columns:
        present_models = [str(x) for x in sorted(df_plot['model_folder'].dropna().unique())]
        next_i = len(known_models)
        for mf in present_models:
            if mf not in shapes_by_model:
                shapes_by_model[mf] = model_markers_default[next_i % len(model_markers_default)]
                next_i += 1

    size_map = {
        'resampled': 28,
        'offpolicy_handwritten': 24,
        'offpolicy_cross_model': 28,
        'offpolicy_same_model': 28,
        'agent_onpolicy_discovered': 28,
        'baseline_mode': 24,
    }
    label_map = {
        'resampled': 'resampled',
        'offpolicy_cross_model': 'cross model',
        'offpolicy_same_model': 'same model',
        'offpolicy_handwritten': 'handwritten',
        'agent_onpolicy_discovered': 'agent',
        'baseline_mode': 'baseline',
    }

    plt.figure(figsize=(fig_size if fig_size is not None else (12, 7)))
    
    # Increase font sizes for scatter plots to match boxplots
    plt.rcParams.update({
        'axes.labelsize': 19,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'legend.fontsize': 14,
        'axes.titlesize': 22,
    })
    
    for (stype, mf), sub in df_plot.groupby(['viz_type', 'model_folder']):
        # Increase marker size for deepseek star
        base_size = size_map.get(stype, 24)
        marker_size = base_size * 2.0 if str(mf) == 'deepseek-r1-0528' else base_size
        plt.scatter(
            sub['mean_logprob'].astype(float).values,
            sub['delta_blackmail_rate'].astype(float).values,
            alpha=0.55,
            s=marker_size,
            c=color_map.get(stype, '#7f7f7f'),
            marker=shapes_by_model.get(str(mf), 'o'),
            edgecolors='none',
        )
    plt.axhline(0.0, color='gray', linewidth=1, linestyle='--')
    plt.xlabel('Mean token logprob\n\n')
    plt.ylabel('Δ Blackmail rate (after − before)')
    plt.title(title_override or 'Sentence insertions vs impact\nDataset: Blackmail\n')
    
    # Set consistent y-axis limits and ticks for comparability across datasets (with padding)
    plt.ylim(-1.05, 0.65)
    plt.yticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    
    # Set consistent x-axis limits and ticks
    plt.xlim(-10.5, 0.5)
    plt.xticks([-10, -8, -6, -4, -2, 0])
    
    # Legends - place both at bottom center in two rows
    from matplotlib.lines import Line2D  # local import
    present_types = [t for t in df_plot['viz_type'].dropna().unique()]
    type_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=8,
               markerfacecolor=color_map.get(t, '#7f7f7f'), markeredgecolor='none', label=label_map.get(t, t))
        for t in present_types
    ]
    
    # Always show all known models in legend (not just present ones)
    all_known_models = ['qwq-32b', 'hermes-4-70b', 'qwen3-235b-a22b', 'llama-3_1-nemotron-ultra-253b-v1', 'deepseek-r1-0528']
    model_handles: List[Line2D] = []
    for mf in all_known_models:
        # Use larger marker for deepseek star
        marker_size = 12 if mf == 'deepseek-r1-0528' else 8
        model_handles.append(Line2D([0], [0], marker=shapes_by_model.get(mf, 'o'), linestyle='None', markersize=marker_size,
                                    markerfacecolor='#cccccc', markeredgecolor='#333333', label=mf))
    
    # Place legends at the bottom in two lines: Type first, then Model (no titles)
    # Type legend (first line) - more spacing from x-axis label
    leg1 = plt.legend(handles=type_handles, loc='upper center', bbox_to_anchor=(0.5, -0.18), 
                     ncol=len(type_handles), borderaxespad=0., fontsize=14, frameon=False)
    # Model legend (second line)
    ax = plt.gca()
    ax.add_artist(leg1)
    leg2 = plt.legend(handles=model_handles, loc='upper center', bbox_to_anchor=(0.5, -0.24), 
                     ncol=len(model_handles), borderaxespad=0., fontsize=14, frameon=False)
    ax = plt.gca()
    ax.grid(False)
    sns.despine(ax=ax, top=True, right=True)
    # Apply fixed axis limits when provided for comparability
    try:
        if xlim is not None and len(xlim) == 2:
            plt.xlim(float(xlim[0]), float(xlim[1]))
        if ylim is not None and len(ylim) == 2:
            plt.ylim(float(ylim[0]), float(ylim[1]))
    except Exception:
        pass
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_distribution_boxplot(
    df: pd.DataFrame,
    out_path: Path,
    *,
    display_model: Optional[str] = None,
    model_folder: Optional[str] = None,
    title_override: Optional[str] = None,
    abs_value: bool = False,
) -> None:
    """Create boxplots comparing delta_blackmail_rate across methods."""
    if df.empty:
        print('[dist] No data to plot (boxplot).')
        return
    
    # Apply global min_logprob filter
    try:
        threshold = globals().get('min_logprob_threshold', None)
    except Exception:
        threshold = None
    df_local = df.copy()
    try:
        if threshold is not None and 'mean_logprob' in df_local.columns:
            mlp_series = pd.to_numeric(df_local['mean_logprob'], errors='coerce')
            df_local = df_local[mlp_series > float(threshold)]
    except Exception:
        pass
    if df_local.empty:
        print('[dist] No data to plot after min_logprob filter (boxplot).')
        return
    
    # Create viz_type column
    df_plot = df_local.copy()
    df_plot['viz_type'] = df_plot['sentence_type'].apply(lambda st: 'resampled' if st in ('onpolicy_inserted', 'onpolicy_fixed') else st)
    
    # Filter to only include the methods we want in boxplot
    methods_to_include = ['resampled', 'offpolicy_handwritten', 'offpolicy_cross_model', 'offpolicy_same_model']
    df_plot = df_plot[df_plot['viz_type'].isin(methods_to_include)]
    
    if df_plot.empty:
        print('[dist] No data to plot after method filtering (boxplot).')
        return
    
    # Prepare y-axis data
    if 'delta_blackmail_rate' in df_plot.columns:
        y_data = pd.to_numeric(df_plot['delta_blackmail_rate'], errors='coerce')
        if abs_value:
            y_data = y_data.abs()
        df_plot['y_plot'] = y_data
    else:
        print('[dist] No delta_blackmail_rate column (boxplot).')
        return
    
    # Remove NaN values
    df_plot = df_plot.dropna(subset=['y_plot'])
    if df_plot.empty:
        print('[dist] No valid data after removing NaNs (boxplot).')
        return
    
    # Create figure
    plt.figure(figsize=(10, 6))
    sns.set_style('white')
    
    # Color mapping
    color_map = {
        'resampled': '#d62728',
        'offpolicy_handwritten': '#1f77b4',
        'offpolicy_cross_model': '#ff7f0e',
        'offpolicy_same_model': '#006400',
    }
    
    # Label mapping
    label_map = {
        'resampled': 'resampled',
        'offpolicy_cross_model': 'cross model',
        'offpolicy_same_model': 'same model',
        'offpolicy_handwritten': 'handwritten',
    }
    
    # Order the methods for consistent display
    method_order = ['resampled', 'offpolicy_handwritten', 'offpolicy_cross_model', 'offpolicy_same_model']
    method_order = [m for m in method_order if m in df_plot['viz_type'].unique()]
    
    # Create boxplot
    positions = list(range(len(method_order)))
    box_data = [df_plot[df_plot['viz_type'] == m]['y_plot'].values for m in method_order]
    
    bp = plt.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                     showmeans=True, meanline=True,
                     boxprops=dict(linewidth=1.5),
                     medianprops=dict(color='black', linewidth=2),
                     meanprops=dict(color='red', linewidth=2, linestyle='--'),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    # Color the boxes
    for patch, method in zip(bp['boxes'], method_order):
        patch.set_facecolor(color_map.get(method, '#cccccc'))
        patch.set_alpha(0.7)
    
    # Set x-tick labels
    plt.xticks(positions, [label_map.get(m, m) for m in method_order])
    
    # Labels and title
    ylabel = '| Δ Blackmail rate (after − before) |' if abs_value else 'Δ Blackmail rate (after − before)'
    plt.ylabel(ylabel)
    
    if title_override:
        plt.title(title_override)
    elif model_folder:
        plt.title(f'Comparison of sentence insertion methods\nModel: {model_folder}  Dataset: Blackmail')
    else:
        plt.title('Comparison of sentence insertion methods\nDataset: Blackmail')
    
    # Add horizontal line at y=0 for non-abs plots
    if not abs_value:
        plt.axhline(0.0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    
    # Set consistent y-axis limits for comparability across datasets
    plt.ylim(-1.0, 0.8)
    
    # Clean up
    ax = plt.gca()
    ax.grid(False)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    
    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_shared_distribution_boxplot(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title_override: Optional[str] = None,
    abs_value: bool = False,
    model_style: str = 'transparency',
) -> None:
    """Create grouped boxplots comparing delta_blackmail_rate across methods AND models.
    
    Args:
        model_style: Either 'transparency' (varying alpha) or 'hatch' (line patterns)
    """
    if df.empty:
        print('[dist] No data to plot (shared boxplot).')
        return
    
    # Apply global min_logprob filter
    try:
        threshold = globals().get('min_logprob_threshold', None)
    except Exception:
        threshold = None
    df_local = df.copy()
    try:
        if threshold is not None and 'mean_logprob' in df_local.columns:
            mlp_series = pd.to_numeric(df_local['mean_logprob'], errors='coerce')
            df_local = df_local[mlp_series > float(threshold)]
    except Exception:
        pass
    if df_local.empty:
        print('[dist] No data to plot after min_logprob filter (shared boxplot).')
        return
    
    # Create viz_type column
    df_plot = df_local.copy()
    df_plot['viz_type'] = df_plot['sentence_type'].apply(lambda st: 'resampled' if st in ('onpolicy_inserted', 'onpolicy_fixed') else st)
    
    # Filter to only include the methods we want in boxplot
    methods_to_include = ['resampled', 'offpolicy_handwritten', 'offpolicy_cross_model', 'offpolicy_same_model']
    df_plot = df_plot[df_plot['viz_type'].isin(methods_to_include)]
    
    if df_plot.empty:
        print('[dist] No data to plot after method filtering (shared boxplot).')
        return
    
    # Ensure model_folder column exists
    if 'model_folder' not in df_plot.columns:
        print('[dist] No model_folder column (shared boxplot).')
        return
    
    # Prepare y-axis data
    if 'delta_blackmail_rate' in df_plot.columns:
        y_data = pd.to_numeric(df_plot['delta_blackmail_rate'], errors='coerce')
        if abs_value:
            y_data = y_data.abs()
        df_plot['y_plot'] = y_data
    else:
        print('[dist] No delta_blackmail_rate column (shared boxplot).')
        return
    
    # Remove NaN values
    df_plot = df_plot.dropna(subset=['y_plot'])
    if df_plot.empty:
        print('[dist] No valid data after removing NaNs (shared boxplot).')
        return
    
    # Create figure
    plt.figure(figsize=(9, 6))
    sns.set_style('white')
    
    # Color mapping for methods
    method_color_map = {
        'resampled': '#d62728',
        'offpolicy_handwritten': '#1f77b4',
        'offpolicy_cross_model': '#ff7f0e',
        'offpolicy_same_model': '#006400',
    }
    
    # Label mapping
    label_map = {
        'resampled': 'resampled',
        'offpolicy_cross_model': 'cross model',
        'offpolicy_same_model': 'same model',
        'offpolicy_handwritten': 'handwritten',
    }
    
    # Order the methods for consistent display
    method_order = ['resampled', 'offpolicy_handwritten', 'offpolicy_cross_model', 'offpolicy_same_model']
    method_order = [m for m in method_order if m in df_plot['viz_type'].unique()]
    
    # Get unique models and order by size (smallest to largest)
    # qwq-32b (32B) < hermes-4-70b (70B) < qwen3-235b-a22b (235B) < llama-3_1-nemotron-ultra-253b-v1 (253B) < deepseek-r1-0528 (671B)
    model_size_order = {
        'qwq-32b': 0,
        'hermes-4-70b': 1,
        'qwen3-235b-a22b': 2,
        'llama-3_1-nemotron-ultra-253b-v1': 3,
        'deepseek-r1-0528': 4,
    }
    models_present = list(df_plot['model_folder'].unique())
    models = sorted(models_present, key=lambda m: model_size_order.get(m, 999))
    n_models = len(models)
    
    # Model visual style maps
    # Always include all known models in legend, even if not in data
    all_known_models = ['qwq-32b', 'hermes-4-70b', 'qwen3-235b-a22b', 'llama-3_1-nemotron-ultra-253b-v1', 'deepseek-r1-0528']
    
    if model_style == 'hatch':
        # Hatch patterns matching the existing legend
        model_hatch_map = {}
        model_alpha_map = {}
        for i, model in enumerate(all_known_models):
            if i == 0:  # qwq-32b
                model_hatch_map[model] = None  # Solid/no hatch
            elif i == 1:  # hermes-4-70b
                model_hatch_map[model] = '..'  # Dots pattern
            elif i == 2:  # qwen3-235b-a22b
                model_hatch_map[model] = 'oo'  # Circles pattern
            elif i == 3:  # llama-3_1-nemotron-ultra-253b-v1
                model_hatch_map[model] = 'xx'  # Cross-hatch pattern
            elif i == 4:  # deepseek-r1-0528
                model_hatch_map[model] = '//'  # Diagonal lines
            else:
                model_hatch_map[model] = '||'  # Vertical lines for any additional
            model_alpha_map[model] = 0.75  # Alpha for box color transparency
    else:
        # Transparency style: lighter to darker as size increases
        model_hatch_map = {}
        model_alpha_map = {}
        for i, model in enumerate(all_known_models):
            model_alpha_map[model] = 0.5 + (0.4 * i / max(1, len(all_known_models) - 1))  # Range from 0.5 to 0.9
            model_hatch_map[model] = None
    
    # Create grouped boxplots
    box_width = 0.25
    group_gap = 0.3
    positions_list = []
    box_data_list = []
    colors_list = []
    labels_list = []
    
    for method_idx, method in enumerate(method_order):
        method_base_pos = method_idx * (n_models * box_width + group_gap)
        for model_idx, model in enumerate(models):
            # Filter data for this method and model
            mask = (df_plot['viz_type'] == method) & (df_plot['model_folder'] == model)
            data = df_plot[mask]['y_plot'].values
            
            if len(data) > 0:
                pos = method_base_pos + model_idx * box_width
                positions_list.append(pos)
                box_data_list.append(data)
                colors_list.append(method_color_map.get(method, '#cccccc'))
                labels_list.append(f"{label_map.get(method, method)} - {model}")
    
    # Plot boxplots
    if box_data_list:
        bp = plt.boxplot(box_data_list, positions=positions_list, widths=box_width * 0.8, patch_artist=True,
                         showmeans=True, meanline=True,
                         boxprops=dict(linewidth=1.0, edgecolor='black'),
                         medianprops=dict(color='black', linewidth=1.5),
                         meanprops=dict(color='red', linewidth=1.5, linestyle='--'),
                         whiskerprops=dict(linewidth=1.0),
                         capprops=dict(linewidth=1.0))
        
        # Color the boxes by method, with alpha and/or hatch varying by model
        for patch, color, pos in zip(bp['boxes'], colors_list, positions_list):
            # Find which method this belongs to
            method_idx = int(pos / (n_models * box_width + group_gap))
            if method_idx < len(method_order):
                method = method_order[method_idx]
                # Find which model within the method (use round to avoid floating-point precision issues)
                model_offset = pos - method_idx * (n_models * box_width + group_gap)
                model_idx = int(round(model_offset / box_width))
                if model_idx < len(models):
                    model = models[model_idx]
                    patch.set_facecolor(color)
                    patch.set_alpha(model_alpha_map.get(model, 0.7))
                    hatch = model_hatch_map.get(model)
                    if hatch is not None:
                        patch.set_hatch(hatch)
    
    # Set x-tick labels at the center of each method group
    tick_positions = []
    tick_labels = []
    for method_idx, method in enumerate(method_order):
        method_base_pos = method_idx * (n_models * box_width + group_gap)
        center_pos = method_base_pos + (n_models - 1) * box_width / 2
        tick_positions.append(center_pos)
        tick_labels.append(label_map.get(method, method))
    
    plt.xticks(tick_positions, tick_labels)
    
    # Labels and title
    ylabel = '| Δ Blackmail rate (after − before) |' if abs_value else 'Δ Blackmail rate (after − before)'
    plt.ylabel(ylabel)
    
    if title_override:
        plt.title(title_override)
    else:
        plt.title('Comparison of sentence insertion methods\nDataset: Blackmail')
    
    # Add horizontal line at y=0 for non-abs plots
    if not abs_value:
        plt.axhline(0.0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    
    # Set consistent y-axis limits and ticks for comparability across datasets
    plt.ylim(-1.0, 0.8)
    plt.yticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    
    # Add legend for models - place at bottom center in single row, showing all known models
    # Only show legend if not disabled via --no_boxplot_legend
    try:
        show_legend = not bool(globals().get('no_boxplot_legend', False))
    except Exception:
        show_legend = True
    
    if show_legend:
        from matplotlib.patches import Patch
        if model_style == 'hatch':
            # Show hatch patterns in legend for all known models
            legend_elements = [
                Patch(facecolor='gray', alpha=0.75, hatch=model_hatch_map.get(model), 
                      edgecolor='black', linewidth=1.0, label=model) 
                for model in all_known_models
            ]
        else:
            # Show transparency in legend for all known models
            legend_elements = [Patch(facecolor='gray', alpha=model_alpha_map.get(model, 0.7), edgecolor='black', linewidth=1.0, label=model) for model in all_known_models]
        plt.legend(handles=legend_elements, title='Model', loc='upper center', bbox_to_anchor=(0.5, -0.12), borderaxespad=0., fontsize=14, ncol=len(all_known_models))
    
    # Clean up
    ax = plt.gca()
    ax.grid(False)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    
    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def write_shared_description(
    df_all: pd.DataFrame,
    out_dir: Path,
    *,
    temperature: float,
    top_p: float,
    plot_methods: Optional[List[str]] = None,
) -> None:
    try:
        # Prepare view and apply pm filter
        df_v = df_all.copy()
        df_v['viz_type'] = df_v['sentence_type'].apply(lambda st: 'resampled' if st in ('onpolicy_inserted', 'onpolicy_fixed') else st)
        if plot_methods:
            method_map = {
                'cross': 'offpolicy_cross_model',
                'handwritten': 'offpolicy_handwritten',
                'same': 'offpolicy_same_model',
                'resampled': 'resampled',
                'agent': 'agent_onpolicy_discovered',
                'baseline': 'baseline_mode',
            }
            requested = set()
            for m in plot_methods:
                k = str(m).strip().lower()
                if k in method_map:
                    requested.add(method_map[k])
            if requested:
                df_v = df_v[df_v['viz_type'].isin(list(requested))]

        # 1) Disruption categories (target_name), exclude generic placeholders
        bad_names = set(['offpolicy cross model', 'offpolicy same model', 'offpolicy', 'unknown'])
        name_values: List[str] = []
        try:
            if 'target_name' in df_v.columns:
                name_values = [str(x) for x in df_v['target_name'].dropna().tolist()]
        except Exception:
            name_values = []
        raw_cats = sorted({s for s in name_values if s.strip() and s.strip().lower() not in bad_names})
        # Normalize categories and drop duplicates like 'consequences' -> 'consequence'
        cats_norm: List[str] = []
        seen_norm: set = set()
        for c in raw_cats:
            cn = _normalize_category(c)
            if cn not in seen_norm:
                seen_norm.add(cn)
                cats_norm.append(cn)
        cats = cats_norm

        # 2) Base sentence for each category (from DISRUPTION_LIBRARY in onpolicy_chain_disruption)
        lines_base: List[str] = []
        for cat in cats:
            base_sentence = str(DISRUPTION_LIBRARY_BM.get(cat, '')).strip()
            lines_base.append(f"- {cat}: {base_sentence}")

        # 3) Four examples per category: resampled, same model, handwritten, cross model
        examples_by_cat: Dict[str, Dict[str, str]] = {}
        # Build normalized category column once
        try:
            df_v = df_v.copy()
            df_v['target_name_norm'] = df_v['target_name'].astype(str).str.lower().map(lambda x: _normalize_category(x))
        except Exception:
            pass
        # Progress bar over categories
        for cat in tqdm(cats, desc="shared description: categories", leave=False):
            sub_cat = df_v[df_v.get('target_name_norm') == _normalize_category(cat)] if 'target_name_norm' in df_v.columns else df_v.iloc[0:0]
            if sub_cat.empty:
                continue
            # Collect candidates
            resampled_cands: List[str] = []
            same_cands: List[str] = []
            cross_cands: List[str] = []
            hand_text = str(DISRUPTION_LIBRARY_BM.get(cat, '')).strip()
            # Unique keys and optional cap to first 5 scenarios
            try:
                key_rows = sub_cat[['model_folder','scenario_id','chunk_idx']].dropna().drop_duplicates()
                key_list_all = key_rows.values.tolist()
                # Cap scenarios for speed
                seen_sid: set = set()
                key_list: List[List[Any]] = []
                for mf, sid, ch in key_list_all:
                    sid_i = int(sid)
                    if sid_i not in seen_sid and len(seen_sid) >= 10:
                        break
                    key_list.append([mf, sid, ch])
                    seen_sid.add(sid_i)
            except Exception:
                key_list = []
            seen_r: set = set(); seen_s: set = set(); seen_c: set = set()
            for mf, sid, ch in key_list:
                try:
                    model_folder = str(mf)
                    scenario_id = int(sid)
                    chunk_idx = int(ch)
                except Exception:
                    continue
                baseline_dir, on_dir, off_dir, on_fixed_dir, agent_dir = resolve_roots(model_folder, temperature, top_p)
                ex_r = _find_onpolicy_example_text(on_dir, on_fixed_dir, scenario_id, cat, chunk_idx)
                if isinstance(ex_r, str) and ex_r.strip() and (ex_r not in seen_r):
                    resampled_cands.append(ex_r); seen_r.add(ex_r)
                ex_s = _find_cross_same_example_text(off_dir, scenario_id, chunk_idx, kind='same', strategy_name=cat)
                if isinstance(ex_s, str) and ex_s.strip() and (ex_s not in seen_s):
                    same_cands.append(ex_s); seen_s.add(ex_s)
                ex_c = _find_cross_same_example_text(off_dir, scenario_id, chunk_idx, kind='cross', strategy_name=cat)
                if isinstance(ex_c, str) and ex_c.strip() and (ex_c not in seen_c):
                    cross_cands.append(ex_c); seen_c.add(ex_c)
            # Sample
            cat_examples: Dict[str, str] = {}
            try:
                if resampled_cands:
                    cat_examples['resampled'] = random.choice(resampled_cands)
                if same_cands:
                    cat_examples['same model'] = random.choice(same_cands)
                if hand_text:
                    cat_examples['handwritten'] = hand_text
                if cross_cands:
                    cat_examples['cross model'] = random.choice(cross_cands)
            except Exception:
                if resampled_cands and 'resampled' not in cat_examples:
                    cat_examples['resampled'] = resampled_cands[0]
                if same_cands and 'same model' not in cat_examples:
                    cat_examples['same model'] = same_cands[0]
                if hand_text and 'handwritten' not in cat_examples:
                    cat_examples['handwritten'] = hand_text
                if cross_cands and 'cross model' not in cat_examples:
                    cat_examples['cross model'] = cross_cands[0]
            examples_by_cat[cat] = cat_examples

        # Write file
        out_dir.mkdir(parents=True, exist_ok=True)
        desc_path = out_dir / 'shared_description.txt'
        with open(desc_path, 'w', encoding='utf-8') as f:
            f.write('Disruption categories (from plotted data)\n')
            for c in cats:
                f.write(f"- {c}\n")
            f.write('\nBase sentence per category\n')
            for line in lines_base:
                f.write(line + '\n')
            f.write('\nExamples (per category)\n')
            order_keys = ['resampled', 'same model', 'handwritten', 'cross model']
            for cat in cats:
                f.write(f"- {cat}:\n")
                exs = examples_by_cat.get(cat, {})
                for k in order_keys:
                    if k in exs and exs[k]:
                        f.write(f"    - {k}: {exs[k]}\n")
        print(f"[save] shared description -> {desc_path}")
    except Exception as e:
        logger.warning(f"shared description write failed: {type(e).__name__}: {e}")

# ------------------------------
# CLI
# ------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='On-policy/off-policy distribution visualization (blackmail)')
    parser.add_argument('-m', '--models', type=str, nargs='*', default=list(MODEL_FOLDER_TO_HF.keys()), choices=list(MODEL_FOLDER_TO_HF.keys()), help='Model folders (space-separated; default: all models)')
    parser.add_argument('-t', '--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('-tp', '--top_p', type=float, default=DEFAULT_TOP_P)
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='Device (cuda:0 or cpu)')
    parser.add_argument('-nq', '--no_quantize', action='store_true', default=False, help='Disable 4-bit quantization')
    parser.add_argument('-fc', '--force_cpu', action='store_true', default=False, help='Force full CPU inference even if GPU is available')
    parser.add_argument('-ms', '--max_scenarios', type=int, default=None, help='Limit number of scenarios to scan')
    parser.add_argument('-tg', '--targets', type=str, default=None, help='Comma-separated target names to include (e.g., ethical,question)')
    parser.add_argument('-ib', '--include_baseline', action='store_true', default=False, help='Include baseline (original chunk) points in the plot (default: False)')
    parser.add_argument('-ll', '--log_level', type=str, default='WARNING', choices=['DEBUG','INFO','WARNING','ERROR'], help='Logging level')
    parser.add_argument('-nc', '--nebius_concurrency', type=int, default=16, help='Max concurrent Nebius requests when using remote logprobs (default: 16)')
    parser.add_argument('-mst', '--min_scored_tokens', type=int, default=5, help='Minimum number of scored tokens required for intervention sentence (default: 4, filters out shorter sentences)')
    parser.add_argument('-ct', '--compare_to', type=str, default='current', choices=['base','current'], help='Baseline to compare deltas against: base (chunk_0) or current (chunk_idx)')
    parser.add_argument('-pm', '--plot_methods', type=str, default='cross,handwritten,same,resampled', help='Comma list of methods to plot: cross,handwritten,same,resampled,agent[,baseline]')
    parser.add_argument('-to', '--timeout_seconds', type=int, default=360, help='Nebius API timeout per request in seconds (default: 360)')
    parser.add_argument('-spo', '--shared_plot_only', action='store_true', default=False, help='Only build a single shared plot by reading existing CSVs across models; do not recompute logprobs')
    parser.add_argument('-rnd', '--use_random_variant', action='store_true', default=False, help='Use *_random input folders and save plots with a _random suffix alongside originals')
    parser.add_argument('-wsd', '--write_shared_description', action='store_true', default=False, help='When used with --shared_plot_only, also build shared_description.txt (slower)')
    parser.add_argument('-bms', '--boxplot_model_style', type=str, default='hatch', choices=['transparency', 'hatch'], help='Boxplot model differentiation style: transparency (default) or hatch (line patterns)')
    parser.add_argument('-nbl', '--no_boxplot_legend', action='store_true', default=True, help='Hide legend in boxplots to prevent horizontal stretching (default: False, legend shown)')
    args = parser.parse_args()
    
    # Shared-plot-only mode: aggregate existing CSVs across all models and plot once
    if bool(args.shared_plot_only):
        try:
            # Respect -pm selection for plotting
            globals()['selected_plot_methods'] = _parse_plot_methods(str(args.plot_methods))
        except Exception:
            globals()['selected_plot_methods'] = None  # type: ignore

        frames: List[pd.DataFrame] = []
        for mf in list(MODEL_FOLDER_TO_HF.keys()):
            try:
                base_dir = Path('analysis') / 'onpolicy_distribution' / mf / f"temperature_{str(args.temperature)}_top_p_{str(args.top_p)}"
                # Prefer random CSV if running in random mode; else standard
                prefer = ['distribution_points_random.csv', 'distribution_points.csv'] if bool(args.use_random_variant) else ['distribution_points.csv', 'distribution_points_random.csv']
                csv_path = None
                for name in prefer:
                    p = base_dir / name
                    if p.exists():
                        csv_path = p
                        break
                if csv_path and csv_path.exists():
                    df_m = pd.read_csv(csv_path)
                    df_m['model_folder'] = mf
                    frames.append(df_m)
            except Exception:
                continue
        if not frames:
            print('[dist] No existing CSVs found for shared plot. Nothing to do.')
            return
        df_all = pd.concat(frames, ignore_index=True)

        # Exclude self_preservation from main plot as elsewhere
        try:
            mask_not_sp_target = (df_all['target_name'] != 'self_preservation') if 'target_name' in df_all.columns else pd.Series([True] * len(df_all))
            mask_off_types = df_all['sentence_type'].isin(['offpolicy_handwritten', 'offpolicy_cross_model', 'offpolicy_same_model']) if 'sentence_type' in df_all.columns else pd.Series([False] * len(df_all))
            mask_not_off_sp = ~(mask_off_types & (df_all['strategy'] == 'self_preservation')) if 'strategy' in df_all.columns else pd.Series([True] * len(df_all))
            df_all_main = df_all[(mask_not_sp_target & mask_not_off_sp)]
        except Exception:
            df_all_main = df_all

        shared_root = Path('analysis') / 'onpolicy_distribution' / 'shared' / f"temperature_{str(args.temperature)}_top_p_{str(args.top_p)}"
        out_path_shared = shared_root / ('scatter_distribution_shared_random.png' if bool(args.use_random_variant) else 'scatter_distribution_shared.png')
        # Parse -pm for shared plot
        plot_methods_list: Optional[List[str]] = None
        try:
            if isinstance(args.plot_methods, str) and len(args.plot_methods) > 0:
                plot_methods_list = [s.strip().lower() for s in args.plot_methods.split(',') if s.strip()]
        except Exception:
            plot_methods_list = None
        # First, create the main shared plot and capture axis limits for alignment
        # Bump font sizes globally for shared plots (20% larger)
        try:
            plt.rcParams.update({'font.size': 19, 'axes.titlesize': 22, 'axes.labelsize': 19, 'xtick.labelsize': 17, 'ytick.labelsize': 17, 'legend.fontsize': 17, 'figure.titlesize': 24})
        except Exception:
            pass
        plot_shared_distribution_scatter(df_all_main, out_path_shared, include_extras=True, plot_methods=plot_methods_list, fig_size=(12, 8))
        # Read back axis limits from the saved data for consistency
        try:
            x_min = float(np.nanmin(df_all_main['mean_logprob'].astype(float))) if 'mean_logprob' in df_all_main.columns else None
            x_max = float(np.nanmax(df_all_main['mean_logprob'].astype(float))) if 'mean_logprob' in df_all_main.columns else None
            y_min = float(np.nanmin(df_all_main['delta_blackmail_rate'].astype(float))) if 'delta_blackmail_rate' in df_all_main.columns else None
            y_max = float(np.nanmax(df_all_main['delta_blackmail_rate'].astype(float))) if 'delta_blackmail_rate' in df_all_main.columns else None
            xlim_shared = (x_min, x_max) if (x_min is not None and x_max is not None) else None
            ylim_shared = (y_min, y_max) if (y_min is not None and y_max is not None) else None
        except Exception:
            xlim_shared, ylim_shared = None, None
        # Write description TXT (optional)
        if bool(args.write_shared_description):
            write_shared_description(
                df_all_main,
                shared_root,
                temperature=args.temperature,
                top_p=args.top_p,
                plot_methods=plot_methods_list,
            )
        print(f"[save] shared scatter -> {out_path_shared}")

        # Self-preservation sentence insertion only shared plot
        try:
            df_sp = df_all.copy()
            # Keep only self_preservation rows from all requested types per -pm
            mask_sp = (df_sp['target_name'] == 'self_preservation') if 'target_name' in df_sp.columns else pd.Series([False] * len(df_sp))
            df_sp = df_sp[mask_sp].copy()
            # Apply -pm selection to self-preservation subset (map to viz_type)
            if plot_methods_list is not None:
                method_map = {
                    'cross': 'offpolicy_cross_model',
                    'handwritten': 'offpolicy_handwritten',
                    'same': 'offpolicy_same_model',
                    'resampled': 'resampled',
                    'agent': 'agent_onpolicy_discovered',
                    'baseline': 'baseline_mode',
                }
                requested = set()
                for m in plot_methods_list:
                    k = str(m).strip().lower()
                    if k in method_map:
                        requested.add(method_map[k])
                if requested:
                    df_sp['viz_type'] = df_sp['sentence_type'].apply(lambda st: 'resampled' if st in ('onpolicy_inserted', 'onpolicy_fixed') else st)
                    df_sp = df_sp[df_sp['viz_type'].isin(list(requested))].copy()

            out_sp = shared_root / ('scatter_distribution_self_preservation_shared_random.png' if bool(args.use_random_variant) else 'scatter_distribution_self_preservation_shared.png')
            plot_shared_distribution_scatter(
                df_sp,
                out_sp,
                include_extras=True,
                plot_methods=plot_methods_list,
                xlim=None,
                ylim=ylim_shared,
                title_override='Self-preservation sentence insertions vs impact',
                fig_size=(12, 8)
            )
            print(f"[save] shared scatter (self-preservation) -> {out_sp}")
        except Exception as _e:
            logger.warning(f"shared self-preservation plot failed: {type(_e).__name__}: {_e}")
        
        # Shared boxplots (shared_plot_only mode)
        try:
            boxplot_shared_spo = shared_root / ('boxplot_distribution_shared_random.png' if bool(args.use_random_variant) else 'boxplot_distribution_shared.png')
            plot_shared_distribution_boxplot(df_all_main, boxplot_shared_spo, title_override='Comparison of sentence insertion methods\nDataset: Blackmail\n', abs_value=False, model_style=str(args.boxplot_model_style))
            print(f"[save] shared boxplot -> {boxplot_shared_spo}")
            
            boxplot_shared_abs_spo = shared_root / ('boxplot_distribution_shared_random_abs.png' if bool(args.use_random_variant) else 'boxplot_distribution_shared_abs.png')
            plot_shared_distribution_boxplot(df_all_main, boxplot_shared_abs_spo, title_override='Comparison of sentence insertion methods\nDataset: Blackmail\n', abs_value=True, model_style=str(args.boxplot_model_style))
            print(f"[save] shared boxplot (abs) -> {boxplot_shared_abs_spo}")
        except Exception as _e:
            logger.warning(f"shared boxplot generation failed (shared_plot_only): {type(_e).__name__}: {_e}")
        return

    # Configure logging
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format='[%(asctime)s %(levelname)s] %(message)s', datefmt='%H:%M:%S')

    # Load environment from .env (if present) so NEBIUS_API_KEY is available
    try:
        if load_dotenv is not None:
            load_dotenv()
    except Exception:
        pass

    nebius_api_key = os.getenv('NEBIUS_API_KEY')
    
    # Expose timeout globally for worker calls before any requests
    try:
        globals()['neb_timeout_seconds'] = int(args.timeout_seconds)
    except Exception:
        globals()['neb_timeout_seconds'] = 180

    targets_filter = [t.strip() for t in (args.targets.split(',') if args.targets else [])] if args.targets else None
    # Configure selected plot methods globally BEFORE data collection so interim plots honor the filter
    globals()['selected_plot_methods'] = _parse_plot_methods(str(args.plot_methods))
    # Expose random variant globally for interim save suffixing
    try:
        globals()['random_variant'] = bool(args.use_random_variant)
    except Exception:
        globals()['random_variant'] = False
    # Expose min scored tokens threshold globally
    try:
        globals()['min_scored_tokens_threshold'] = int(args.min_scored_tokens)
    except Exception:
        globals()['min_scored_tokens_threshold'] = 4
    # Expose no boxplot legend flag globally for boxplot functions
    try:
        globals()['no_boxplot_legend'] = bool(args.no_boxplot_legend)
    except Exception:
        globals()['no_boxplot_legend'] = False

    # Loop through all models and run analysis for each
    for model_folder in args.models:
        print(f"\n{'='*80}\nProcessing model: {model_folder}\n{'='*80}")
        
        baseline_dir, onpolicy_dir, offpolicy_dir, onpolicy_fixed_dir, agent_dir = resolve_roots(model_folder, args.temperature, args.top_p, random_variant=bool(args.use_random_variant))
        out_root = build_output_dir(model_folder, args.temperature, args.top_p)

        if not onpolicy_dir.exists():
            print(f"[dist] On-policy dir not found: {onpolicy_dir}")
            continue

        logger.info(f"baseline_dir={baseline_dir}")
        logger.info(f"onpolicy_dir={onpolicy_dir}")
        logger.info(f"offpolicy_dir={offpolicy_dir}")
        logger.info(f"onpolicy_fixed_dir={onpolicy_fixed_dir}")
        logger.info(f"agent_dir={agent_dir}")

        hf_id = MODEL_FOLDER_TO_HF.get(model_folder, MODEL_FOLDER_TO_HF['qwen3-235b-a22b'])
        tokenizer_cache: Dict[str, Any] = {}
        bundle: Optional[HFModelBundle] = None
        
        # Automatically use local model for qwq-32b, Nebius for all others
        use_nebius_logprobs = (model_folder != 'qwq-32b')
        
        if not use_nebius_logprobs:
            # Load local model for qwq-32b
            try:
                if args.force_cpu:
                    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
                device_map_pref = 'balanced_low_0' if (not args.force_cpu) and torch.cuda.is_available() else 'auto'
                print(f'[dist] Loading local model: {hf_id}')
                bundle = load_model_and_tokenizer(hf_id, device=args.device, quantize_4bit=(not args.no_quantize), device_map_pref=device_map_pref)
                if args.force_cpu and bundle is not None:
                    try:
                        bundle.model.to('cpu')
                        bundle = HFModelBundle(model=bundle.model, tokenizer=bundle.tokenizer, device='cpu')
                        print('[dist] Forcing model to CPU for all forwards')
                    except Exception:
                        pass
            except Exception as e:
                print(f"[dist] Failed to load model {hf_id} locally: {type(e).__name__}: {e}")
                continue
        else:
            # Use Nebius API for other models
            print(f'[dist] Using Nebius API for logprobs (model: {model_folder})')

        rows = collect_distribution_points(
            bundle if bundle is not None else HFModelBundle(model=None, tokenizer=None, device='cpu'),
            baseline_dir,
            onpolicy_dir,
            offpolicy_dir,
            onpolicy_fixed_dir=onpolicy_fixed_dir,
            agent_dir=agent_dir,
            max_scenarios=args.max_scenarios,
            targets_filter=targets_filter,
            include_baseline=bool(args.include_baseline),
            use_nebius_logprobs=use_nebius_logprobs,
            model_folder=model_folder,
            nebius_api_key=nebius_api_key,
            tokenizer_cache=tokenizer_cache,
            stream_save_every=1,
            out_root=out_root,
            display_model=hf_id,
            nebius_concurrency=int(args.nebius_concurrency),
            compare_to=str(args.compare_to),
        )

        df = pd.DataFrame(rows)
        logger.info(f"Total points collected: {len(df)}")
        csv_path = out_root / ('distribution_points_random.csv' if bool(args.use_random_variant) else 'distribution_points.csv')
        df.to_csv(csv_path, index=False)
        print(f"[save] points -> {csv_path}")

        # Suffix plots when using random variant
        suffix = '_random' if bool(args.use_random_variant) else ''
        plot_path = out_root / f"scatter_distribution{suffix}.png"
        display_model = hf_id
        # Exclude self_preservation from the two standard plots
        try:
            mask_not_sp_target = (df['target_name'] != 'self_preservation') if 'target_name' in df.columns else pd.Series([True] * len(df))
            mask_off_types = df['sentence_type'].isin(['offpolicy_handwritten', 'offpolicy_cross_model', 'offpolicy_same_model']) if 'sentence_type' in df.columns else pd.Series([False] * len(df))
            mask_not_off_sp = ~(mask_off_types & (df['strategy'] == 'self_preservation')) if 'strategy' in df.columns else pd.Series([True] * len(df))
            df_main = df[(mask_not_sp_target & mask_not_off_sp)]
        except Exception:
            df_main = df
        plot_distribution_scatter(df_main, plot_path, display_model=display_model, model_folder=model_folder, include_extras=False)
        print(f"[save] scatter -> {plot_path}")

        # Second plot including extra off-policy categories explicitly
        extras_plot_path = out_root / f"scatter_distribution_with_extras{suffix}.png"
        plot_distribution_scatter(df_main, extras_plot_path, display_model=display_model, model_folder=model_folder, include_extras=True)
        print(f"[save] scatter (with extras) -> {extras_plot_path}")

        # Boxplots comparing methods
        try:
            boxplot_path = out_root / f"boxplot_distribution{suffix}.png"
            plot_distribution_boxplot(df_main, boxplot_path, display_model=display_model, model_folder=model_folder, abs_value=False)
            print(f"[save] boxplot -> {boxplot_path}")
            
            # Absolute value version of boxplot
            boxplot_path_abs = out_root / f"boxplot_distribution{suffix}_abs.png"
            plot_distribution_boxplot(df_main, boxplot_path_abs, display_model=display_model, model_folder=model_folder, abs_value=True)
            print(f"[save] boxplot (abs) -> {boxplot_path_abs}")
        except Exception as _e:
            logger.warning(f"boxplot generation failed: {type(_e).__name__}: {_e}")

        # New: Self-preservation only plots (on-policy inserted/fixed filtered by target_name == 'self_preservation')
        try:
            if 'target_name' in df.columns and df['target_name'].isin(['self_preservation']).any() or ('strategy' in df.columns and df['strategy'].isin(['self_preservation']).any()):
                # Include:
                # - all rows where target_name == 'self_preservation' (on-policy, baseline, agent if any)
                # - off-policy rows where strategy == 'self_preservation'
                try:
                    mask_sp_target = (df['target_name'] == 'self_preservation') if 'target_name' in df.columns else (pd.Series([False] * len(df)))
                    mask_off_types = df['sentence_type'].isin(['offpolicy_handwritten', 'offpolicy_cross_model', 'offpolicy_same_model']) if 'sentence_type' in df.columns else (pd.Series([False] * len(df)))
                    mask_off_sp = (mask_off_types & (df['strategy'] == 'self_preservation')) if 'strategy' in df.columns else (pd.Series([False] * len(df)))
                    df_sp = df[(mask_sp_target | mask_off_sp)].copy()
                except Exception:
                    df_sp = df[(df.get('target_name') == 'self_preservation')].copy()  # fallback
                if not df_sp.empty:
                    sp_plot_path = out_root / f"scatter_distribution_self_preservation{suffix}.png"
                    plot_distribution_scatter(df_sp, sp_plot_path, display_model=display_model, model_folder=model_folder, include_extras=False, title_override=f'Self-preservation sentence insertions vs impact\nModel: {model_folder}  Dataset: Blackmail')
                    print(f"[save] scatter (self-preservation) -> {sp_plot_path}")
                    # Optionally include baseline_mode for self-preservation-only too (same df_sp already includes baseline if present)
                    sp_plot_path_ex = out_root / f"scatter_distribution_self_preservation_with_extras{suffix}.png"
                    plot_distribution_scatter(df_sp, sp_plot_path_ex, display_model=display_model, model_folder=model_folder, include_extras=True, title_override=f'Self-preservation sentence insertions vs impact\nModel: {model_folder}  Dataset: Blackmail')
                    print(f"[save] scatter (self-preservation with extras) -> {sp_plot_path_ex}")
        except Exception as _e:
            logger.warning(f"self-preservation plots failed: {type(_e).__name__}: {_e}")

    # After all models are processed, generate shared plots
    print(f"\n{'='*80}\nGenerating shared plots across all models\n{'='*80}")
    
    frames: List[pd.DataFrame] = []
    for mf in args.models:
        try:
            base_dir = Path('analysis') / 'onpolicy_distribution' / mf / f"temperature_{str(args.temperature)}_top_p_{str(args.top_p)}"
            # Prefer random CSV if running in random mode; else standard
            prefer = ['distribution_points_random.csv', 'distribution_points.csv'] if bool(args.use_random_variant) else ['distribution_points.csv', 'distribution_points_random.csv']
            csv_path = None
            for name in prefer:
                p = base_dir / name
                if p.exists():
                    csv_path = p
                    break
            if csv_path and csv_path.exists():
                df_m = pd.read_csv(csv_path)
                df_m['model_folder'] = mf
                frames.append(df_m)
        except Exception:
            continue
    
    if not frames:
        print('[dist] No CSVs found for shared plot generation.')
        return
    
    df_all = pd.concat(frames, ignore_index=True)

    # Exclude self_preservation from main plot as elsewhere
    try:
        mask_not_sp_target = (df_all['target_name'] != 'self_preservation') if 'target_name' in df_all.columns else pd.Series([True] * len(df_all))
        mask_off_types = df_all['sentence_type'].isin(['offpolicy_handwritten', 'offpolicy_cross_model', 'offpolicy_same_model']) if 'sentence_type' in df_all.columns else pd.Series([False] * len(df_all))
        mask_not_off_sp = ~(mask_off_types & (df_all['strategy'] == 'self_preservation')) if 'strategy' in df_all.columns else pd.Series([True] * len(df_all))
        df_all_main = df_all[(mask_not_sp_target & mask_not_off_sp)]
    except Exception:
        df_all_main = df_all

    shared_root = Path('analysis') / 'onpolicy_distribution' / 'shared' / f"temperature_{str(args.temperature)}_top_p_{str(args.top_p)}"
    out_path_shared = shared_root / ('scatter_distribution_shared_random.png' if bool(args.use_random_variant) else 'scatter_distribution_shared.png')
    
    # Parse -pm for shared plot
    plot_methods_list: Optional[List[str]] = None
    try:
        if isinstance(args.plot_methods, str) and len(args.plot_methods) > 0:
            plot_methods_list = [s.strip().lower() for s in args.plot_methods.split(',') if s.strip()]
    except Exception:
        plot_methods_list = None
    
    # Bump font sizes globally for shared plots (20% larger)
    try:
        plt.rcParams.update({'font.size': 19, 'axes.titlesize': 22, 'axes.labelsize': 19, 'xtick.labelsize': 17, 'ytick.labelsize': 17, 'legend.fontsize': 17, 'figure.titlesize': 24})
    except Exception:
        pass
    
    plot_shared_distribution_scatter(df_all_main, out_path_shared, include_extras=True, plot_methods=plot_methods_list, fig_size=(12, 8))
    
    # Read back axis limits from the saved data for consistency
    try:
        x_min = float(np.nanmin(df_all_main['mean_logprob'].astype(float))) if 'mean_logprob' in df_all_main.columns else None
        x_max = float(np.nanmax(df_all_main['mean_logprob'].astype(float))) if 'mean_logprob' in df_all_main.columns else None
        y_min = float(np.nanmin(df_all_main['delta_blackmail_rate'].astype(float))) if 'delta_blackmail_rate' in df_all_main.columns else None
        y_max = float(np.nanmax(df_all_main['delta_blackmail_rate'].astype(float))) if 'delta_blackmail_rate' in df_all_main.columns else None
        xlim_shared = (x_min, x_max) if (x_min is not None and x_max is not None) else None
        ylim_shared = (y_min, y_max) if (y_min is not None and y_max is not None) else None
    except Exception:
        xlim_shared, ylim_shared = None, None
    
    # Write description TXT
    write_shared_description(
        df_all_main,
        shared_root,
        temperature=args.temperature,
        top_p=args.top_p,
        plot_methods=plot_methods_list,
    )
    print(f"[save] shared scatter -> {out_path_shared}")

    # Self-preservation sentence insertion only shared plot
    try:
        df_sp = df_all.copy()
        # Keep only self_preservation rows from all requested types per -pm
        mask_sp = (df_sp['target_name'] == 'self_preservation') if 'target_name' in df_sp.columns else pd.Series([False] * len(df_sp))
        df_sp = df_sp[mask_sp].copy()
        # Apply -pm selection to self-preservation subset (map to viz_type)
        if plot_methods_list is not None:
            method_map = {
                'cross': 'offpolicy_cross_model',
                'handwritten': 'offpolicy_handwritten',
                'same': 'offpolicy_same_model',
                'resampled': 'resampled',
                'agent': 'agent_onpolicy_discovered',
                'baseline': 'baseline_mode',
            }
            requested = set()
            for m in plot_methods_list:
                k = str(m).strip().lower()
                if k in method_map:
                    requested.add(method_map[k])
            if requested:
                df_sp['viz_type'] = df_sp['sentence_type'].apply(lambda st: 'resampled' if st in ('onpolicy_inserted', 'onpolicy_fixed') else st)
                df_sp = df_sp[df_sp['viz_type'].isin(list(requested))].copy()
        
        out_sp = shared_root / ('scatter_distribution_self_preservation_shared_random.png' if bool(args.use_random_variant) else 'scatter_distribution_self_preservation_shared.png')
        plot_shared_distribution_scatter(
            df_sp,
            out_sp,
            include_extras=True,
            plot_methods=plot_methods_list,
            xlim=None,
            ylim=ylim_shared,
            title_override='Self-preservation sentence insertions vs impact',
            fig_size=(12, 8)
        )
        print(f"[save] shared scatter (self-preservation) -> {out_sp}")
    except Exception as _e:
        logger.warning(f"shared self-preservation plot failed: {type(_e).__name__}: {_e}")
    
    # Shared boxplots
    try:
        boxplot_shared = shared_root / ('boxplot_distribution_shared_random.png' if bool(args.use_random_variant) else 'boxplot_distribution_shared.png')
        plot_shared_distribution_boxplot(df_all_main, boxplot_shared, title_override='Comparison of sentence insertion methods\nDataset: Blackmail\n', abs_value=False, model_style=str(args.boxplot_model_style))
        print(f"[save] shared boxplot -> {boxplot_shared}")
        
        boxplot_shared_abs = shared_root / ('boxplot_distribution_shared_random_abs.png' if bool(args.use_random_variant) else 'boxplot_distribution_shared_abs.png')
        plot_shared_distribution_boxplot(df_all_main, boxplot_shared_abs, title_override='Comparison of sentence insertion methods\nDataset: Blackmail\n', abs_value=True, model_style=str(args.boxplot_model_style))
        print(f"[save] shared boxplot (abs) -> {boxplot_shared_abs}")
    except Exception as _e:
        logger.warning(f"shared boxplot generation failed: {type(_e).__name__}: {_e}")


if __name__ == '__main__':
    # Basic style consistent with other analysis scripts (20% larger fonts)
    plt.rcParams.update({'font.size': 17, 'axes.titlesize': 19, 'axes.labelsize': 17, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14, 'figure.titlesize': 22})
    main()
