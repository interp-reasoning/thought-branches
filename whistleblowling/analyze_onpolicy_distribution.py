#!/usr/bin/env python3
"""
Analyze where different sentence types land on the probability distribution vs impact (whistleblowing domain).

Objective:
- X-axis: how on-policy/off-policy the sentence is → mean token logprob under the display model
- Y-axis: impact/effect → Δ whistleblowing rate (after − before)

Sentence types visualized:
- onpolicy_inserted: on-policy (varied) — selected resampled chunk from the varied pipeline
- onpolicy_fixed: on-policy (fixed) — fixed target sentence pipeline
- offpolicy_handwritten: disruption_text from off-policy chain disruption (handwritten target)
- baseline_mode: the original base sentence at that chunk (serves as a proxy for distribution mode; optional via flag)

Data sources (whistleblower rollouts):
- On-policy (varied): whistleblower_rollouts/<model>/temperature_<t>_top_p_<p>/yes_base_solution_onpolicy_chain_disruption/{scenario_*|problem_*}/onpolicy_chain_disruption_*.json
- On-policy (fixed): whistleblower_rollouts/<model>/temperature_<t>_top_p_<p>/yes_base_solution_onpolicy_chain_disruption_fixed/{scenario_*|problem_*}/onpolicy_chain_disruption_*.json
- Off-policy results: whistleblower_rollouts/<model>/temperature_<t>_top_p_<p>/yes_base_solution_chain_disruption/scenario_*/chain_disruption_results.json
- Baseline per-chunk solutions and metadata in: whistleblower_rollouts/<model>/temperature_<t>_top_p_<p>/yes_base_solution

Outputs:
- analysis/onpolicy_distribution/<model>/temperature_<t>_top_p_<p>/distribution_points.csv
- analysis/onpolicy_distribution/<model>/temperature_<t>_top_p_<p>/scatter_distribution.png
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

# Disruption library base sentences (whistleblower domain) - load without importing the module
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

DISRUPTION_LIBRARY_WB: Dict[str, str] = _load_disruption_library_from_source()


# ------------------------------
# Config
# ------------------------------

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
MODEL_FOLDER_TO_HF = {
    'llama-3_1-nemotron-ultra-253b-v1': 'nvidia/Llama-3_1-Nemotron-Ultra-253B-v1',
    'hermes-4-70b': 'NousResearch/Hermes-4-70B',
    'deepseek-r1-0528': 'deepseek-ai/DeepSeek-R1-0528',
}

# Provider model IDs for Nebius API (best-effort mapping)
MODEL_FOLDER_TO_NEBIUS = {
    'llama-3_1-nemotron-ultra-253b-v1': 'nvidia/Llama-3_1-Nemotron-Ultra-253B-v1',
    'hermes-4-70b': 'NousResearch/Hermes-4-70B',
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


def list_scenarios(input_dir: Path) -> List[Path]:
    if not input_dir or not input_dir.exists():
        return []
    return sorted([d for d in input_dir.iterdir() if d.is_dir() and (d.name.startswith('scenario_') or d.name.startswith('problem_'))])


def resolve_roots(model_folder: str, temperature: float, top_p: float, random_variant: bool = False) -> Tuple[Path, Path, Path, Path, Path]:
    base = Path('whistleblower_rollouts') / model_folder / f"temperature_{str(temperature)}_top_p_{str(top_p)}"
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
    # Whistleblower scenarios may not always have explicit user/email segments here.
    sys = system_prompt or ""
    return f"{sys}\n\nAssistant:\n<think>\n{prefix_text}"


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
def compute_token_logprobs(bundle: HFModelBundle, prefix_text: str, sentence_text: str) -> Dict[str, Any]:
    """Compute per-token logprobs for sentence_text given prefix_text using teacher forcing.

    Robust to accelerate device_map=auto by keeping inputs on CPU; accelerate handles dispatch.
    Returns tokens, token_ids, per-token logprobs, sum, and mean.
    """
    tokenizer = bundle.tokenizer
    model = bundle.model

    # Tokenize (CPU tensors)
    ids_prefix = tokenizer(prefix_text, add_special_tokens=True, return_tensors='pt')
    ids_target = tokenizer(sentence_text, add_special_tokens=False, return_tensors='pt')

    target_len = int(ids_target['input_ids'].shape[1])
    token_ids: List[int] = ids_target['input_ids'][0].tolist()
    tokens: List[str] = tokenizer.convert_ids_to_tokens(token_ids)
    if target_len <= 0:
        return {
            'tokens': tokens,
            'token_ids': token_ids,
            'logprobs': [],
            'logprob_sum': 0.0,
            'logprob_mean': 0.0,
        }

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
        return {
            'tokens': tokens,
            'token_ids': token_ids,
            'logprobs': logprobs,
            'logprob_sum': float(arr.sum()) if arr.size > 0 else 0.0,
            'logprob_mean': float(arr.mean()) if arr.size > 0 else 0.0,
        }

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

    return {
        'tokens': tokens,
        'token_ids': token_ids,
        'logprobs': logprobs,
        'logprob_sum': float(arr.sum()) if arr.size > 0 else 0.0,
        'logprob_mean': float(arr.mean()) if arr.size > 0 else 0.0,
        'scored_token_count': int(arr.size),
        'total_token_count': int(len(logprobs)),
    }


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
    retries: int = 5,
    backoff_seconds: float = 1.0,
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
    max_attempts = max(1, int(retries) + 1)  # Total attempts = 1 initial + N retries
    while attempt < max_attempts:
        attempt += 1
        try:
            # Use a connection pool with limited keepalive to avoid lingering sockets
            limits = httpx.Limits(max_keepalive_connections=8, max_connections=64)
            with httpx.Client(timeout=timeout, limits=limits) as client:
                resp = client.post("https://api.studio.nebius.com/v1/completions", headers=headers, json=payload)
                # print(f"Nebius response: {resp.status_code} {resp.text}")
                if resp.status_code == 200:
                    if attempt > 1:
                        logger.info(f"Nebius request succeeded on attempt {attempt}/{max_attempts}")
                    return resp.json()
                # Retry on transient statuses
                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt < max_attempts:
                        wait_time = max(0.1, float(backoff_seconds) * (1.6 ** (attempt - 1)))
                        logger.warning(f"Nebius HTTP {resp.status_code} (attempt {attempt}/{max_attempts}, retrying in {wait_time:.1f}s): {resp.text[:200]}")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Nebius HTTP {resp.status_code} (attempt {attempt}/{max_attempts}, giving up): {resp.text[:200]}")
                        return None
                logger.warning(f"Nebius HTTP {resp.status_code} (non-retryable): {resp.text[:200]}")
                return None
        except Exception as e:
            if attempt < max_attempts:
                wait_time = max(0.1, float(backoff_seconds) * (1.6 ** (attempt - 1)))
                logger.warning(f"Nebius request failed (attempt {attempt}/{max_attempts}, retrying in {wait_time:.1f}s): {type(e).__name__}: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Nebius request failed (attempt {attempt}/{max_attempts}, giving up): {type(e).__name__}: {e}")
                return None
    return None


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
    context_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute per-token logprobs for sentence_text given prefix via Nebius API.

    Primary strategy (echo=True, single call):
      - Request prompt token logprobs using completions with echo=True, max_tokens=0
        for the concatenated string: prefix_text + sentence_text.
      - Use the provider-returned text_offset array to slice the token_logprobs
        corresponding exactly to the target sentence span. This avoids BOS/EOS
        mismatches and tokenizer differences, and requires no filling.

    Fallback:
      - If provider does not return prompt token logprobs, fall back to an
        incremental next-token lookup path using top_logprobs to extract the
        intended token's logprob. Ensure we do not silently drop tokens by
        recording a conservative fallback for any missed token.
    """
    # Resolve HF id for display/remote scoring
    first_model = next(iter(MODEL_FOLDER_TO_HF.values()))
    hf_id = MODEL_FOLDER_TO_HF.get(model_folder, first_model)
    tok = tokenizer_cache.get(hf_id)
    if tok is None:
        try:
            from transformers import AutoTokenizer  # type: ignore
            tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            tokenizer_cache[hf_id] = tok
        except Exception:
            return {'tokens': [], 'token_ids': [], 'logprobs': [], 'logprob_sum': 0.0, 'logprob_mean': 0.0}

    # Tokenize target locally to return token list for reference (not required by provider)
    ids_target = tok(sentence_text, add_special_tokens=False, return_tensors='pt')
    token_ids: List[int] = ids_target['input_ids'][0].tolist()
    tokens: List[str] = tok.convert_ids_to_tokens(token_ids)

    # Use provider model id for Nebius
    provider_id = MODEL_FOLDER_TO_NEBIUS.get(model_folder, hf_id)

    # Preferred strategy: single echo completion with max_tokens=0 and echo=True
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
                return {'tokens': tokens, 'token_ids': token_ids, 'logprobs': [], 'logprob_sum': float('nan'), 'logprob_mean': float('nan'), 'scored_token_count': 0, 'total_token_count': 0}

            # Sliding window token matching (same as test_logprob_extraction.py)
            def normalize_token(tok: str) -> str:
                """Replace BPE markers: Ġ/\u0120 → space, Ċ/\u010A → newline.
                Also normalize fancy quotes to straight quotes."""
                return (tok.replace('Ġ', ' ').replace('\u0120', ' ')
                           .replace('Ċ', '\n').replace('\u010A', '\n')
                           .replace('\u2019', "'")
                           .replace('\u00e2\u0122\u013b', "'")
                           .replace('’', "'").replace('’', "'")  # Fancy single quotes
                           .replace('“', '"').replace('”', '"'))  # Fancy double quotes
            
            # Prepare token list with logprobs
            token_list = []
            for i, (tok, top_lp_dict) in enumerate(zip(full_tokens, full_top_lps)):
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
                    i, tok, lp = token_list[start_pos + offset]
                    normalized_tok = normalize_token(tok)
                    accumulated += normalized_tok
                    temp_tokens.append(tok)
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
                            i, tok, lp = token_list[start_pos + offset]
                            normalized_tok = normalize_token(tok)
                            accumulated += normalized_tok
                            temp_tokens.append(tok)
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
                return {
                    'tokens': tokens,
                    'token_ids': token_ids,
                    'logprobs': sentence_logprobs_list,
                    'logprob_sum': float(arr.sum()) if arr.size > 0 else float('nan'),
                    'logprob_mean': float(arr.mean()) if arr.size > 0 else float('nan'),
                    'scored_token_count': int(arr.size),
                    'total_token_count': int(len(sentence_tokens_list)),
                }
            else:
                # Could not match even with stripping - warn with context and save debug data
                ctx = context_info or {}
                ctx_str = f"scenario={ctx.get('scenario_id')}, chunk={ctx.get('chunk_idx')}, type={ctx.get('sentence_type')}"
                if ctx.get('target_name'):
                    ctx_str += f", target={ctx.get('target_name')}"
                if ctx.get('strategy'):
                    ctx_str += f", strategy={ctx.get('strategy')}"
                
                logger.warning(f"Token matching failed [{ctx_str}]")
                logger.warning(f"  Sentence: {sentence_text[:60]}")
                if token_list:
                    sample_reconstructed = ""
                    for j in range(min(30, len(token_list))):
                        sample_reconstructed += normalize_token(token_list[j][1])
                    logger.warning(f"  Looking for (normalized): {repr(normalized_sentence[:100])}")
                    logger.warning(f"  Also tried stripped: {repr(normalized_sentence_stripped[:100])}")
                    logger.warning(f"  Sample token stream: {repr(sample_reconstructed[:150])}")
                
                # Save full API response for debugging
                try:
                    import json as json_module
                    with open('logprobs_data.json', 'w') as f:
                        json_module.dump(lg_full, f, indent=4)
                    logger.warning(f"  Saved full API response to logprobs_data.json for debugging")
                except Exception:
                    pass
                
                return {'tokens': tokens, 'token_ids': token_ids, 'logprobs': [], 'logprob_sum': float('nan'), 'logprob_mean': float('nan'), 'scored_token_count': 0, 'total_token_count': 0}
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
    return {
        'tokens': tokens,
        'token_ids': token_ids,
        'logprobs': out_lps,
        'logprob_sum': float(arr.sum()) if arr.size > 0 else float('nan'),
        'logprob_mean': float(arr.mean()) if arr.size > 0 else float('nan'),
        'scored_token_count': int(arr.size),
        'total_token_count': int(len(tokens)),
    }


# ------------------------------
# Metrics (reuse logic from analyze_onpolicy_chain_disruption)
# ------------------------------

def compute_prev_metrics(baseline_scenario_dir: Path, chunk_idx: int) -> Tuple[Optional[float], int]:
    """Compute baseline whistleblowing rate from solutions.json if present; returns (rate, n)."""
    chunk_dir = baseline_scenario_dir / f"chunk_{chunk_idx}"
    solutions_file = chunk_dir / 'solutions.json'
    if not solutions_file.exists():
        return None, 0
    try:
        with open(solutions_file, 'r', encoding='utf-8') as f:
            sols = json.load(f)
    except Exception:
        return None, 0
    if not isinstance(sols, list) or not sols:
        return None, 0
    try:
        classified_solutions = [
            s for s in sols
            if isinstance(s, dict) and (s.get('contains_whistleblowing') is not None)
        ]
        vals = [1 if bool(s.get('contains_whistleblowing')) else 0 for s in classified_solutions]
        prev_rate = float(sum(vals) / len(vals)) if vals else None
        return prev_rate, int(len(vals))
    except Exception:
        return None, 0


def compute_after_metrics_onpolicy(exp: Dict[str, Any]) -> Tuple[Optional[float], int]:
    """Compute after whistleblowing rate from an on-policy experiment record.
    Supports multiple schemas: contains_whistleblowing or precomputed *_rate fields.
    """
    # Prefer precomputed rate fields if present
    for k in ['whistleblowing_rate']:
        try:
            if exp.get(k) is not None:
                return float(exp.get(k)), -1
        except Exception:
            pass
    cls = exp.get('full_rollouts_classified') or []
    if not isinstance(cls, list) or not cls:
        return None, 0
    vals: List[int] = []
    for r in cls:
        if not isinstance(r, dict):
            continue
        v = r.get('contains_whistleblowing')
        if v is None:
            # Heuristic fallbacks
            if 'whistleblowing' in r:
                v = bool(r.get('whistleblowing'))
            elif 'whistleblowing_score' in r:
                try:
                    v = float(r.get('whistleblowing_score')) > 0.5
                except Exception:
                    v = None
        if v is not None:
            vals.append(1 if bool(v) else 0)
    after_rate = float(sum(vals) / len(vals)) if vals else None
    return after_rate, len(vals)


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
    # Support both scenario_* and problem_*
    sdir = off_dir / f'scenario_{scenario_id}'
    if not sdir.exists():
        alt = off_dir / f'problem_{scenario_id}'
        if alt.exists():
            sdir = alt
    path = sdir / 'chain_disruption_results.json'
    return load_json(path)


def load_offpolicy_cross_file(off_cross_dir: Path, scenario_id: int) -> Optional[Dict[str, Any]]:
    sdir = off_cross_dir / f'scenario_{scenario_id}'
    if not sdir.exists():
        alt = off_cross_dir / f'problem_{scenario_id}'
        if alt.exists():
            sdir = alt
    path = sdir / 'chain_disruption_cross_model_results.json'
    return load_json(path)


def load_offpolicy_same_file(off_same_dir: Path, scenario_id: int) -> Optional[Dict[str, Any]]:
    sdir = off_same_dir / f'scenario_{scenario_id}'
    if not sdir.exists():
        alt = off_same_dir / f'problem_{scenario_id}'
        if alt.exists():
            sdir = alt
    path = sdir / 'chain_disruption_same_model_results.json'
    return load_json(path)


# ------------------------------
# Shared-plot helpers (examples and text lookups)
# ------------------------------

def _load_chunks_for_scenario(baseline_dir: Path, scenario_id: int) -> Tuple[List[str], List[str]]:
    # Support both scenario_* and problem_* naming
    sdir = baseline_dir / f"scenario_{int(scenario_id)}"
    if not sdir.exists():
        alt = baseline_dir / f"problem_{int(scenario_id)}"
        if alt.exists():
            sdir = alt
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
    if not sdir_f.exists():
        altf = onpolicy_fixed_dir / f"problem_{int(scenario_id)}"
        if altf.exists():
            sdir_f = altf
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
    if not sdir.exists():
        alt = onpolicy_dir / f"problem_{int(scenario_id)}"
        if alt.exists():
            sdir = alt
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
        alt = agent_dir / f"problem_{int(scenario_id)}"
        if alt.exists():
            sdir = alt
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
    model_folder: str = list(MODEL_FOLDER_TO_HF.keys())[0],
    nebius_api_key: Optional[str] = None,
    tokenizer_cache: Optional[Dict[str, Any]] = None,
    stream_save_every: int = 0,
    out_root: Optional[Path] = None,
    display_model: Optional[str] = None,
    nebius_concurrency: int = 100,
    compare_to: str = 'current',
    include_niss_onpolicy: bool = True,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    on_targets = discover_onpolicy_targets(onpolicy_dir)
    # Optionally merge in *_niss on-policy outputs (no-include-similar-sentence)
    if bool(include_niss_onpolicy):
        try:
            niss_dir = onpolicy_dir.parent / f"{onpolicy_dir.name}_niss"
            if niss_dir.exists():
                on_targets_niss = discover_onpolicy_targets(niss_dir)
                for sid, files in on_targets_niss.items():
                    if sid in on_targets:
                        have = set(map(str, on_targets[sid]))
                        add = [p for p in files if str(p) not in have]
                        on_targets[sid].extend(add)
                    else:
                        on_targets[sid] = files
        except Exception:
            pass
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

    # Helper to append point with filtering
    filtered_points_count = [0]  # Use list to allow modification in nested function
    def try_append_point(point_dict: Dict[str, Any], sentence_preview: str = "") -> bool:
        """Append point if it meets min_scored_tokens threshold. Returns True if appended."""
        min_tokens = int(globals().get('min_scored_tokens_threshold', 2))
        scored = int(point_dict.get('scored_token_count', 0))
        if scored < min_tokens:
            filtered_points_count[0] += 1
            logger.debug(f"Filtered HF point with {scored} tokens (min: {min_tokens}): {sentence_preview[:50]}")
            return False
        points.append(point_dict)
        return True

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
                            context_info=t.get('meta'),
                        ),
                        tasks_local,
                    ))
            except Exception as _e:
                logger.warning(f"parallel Nebius scoring failed: {type(_e).__name__}: {_e}")
                results = []
            min_tokens_threshold = int(globals().get('min_scored_tokens_threshold', 2))
            filtered_count = 0
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
                
                # Filter out points with too few scored tokens
                if scored < min_tokens_threshold:
                    filtered_count += 1
                    sentence_preview = task.get('sentence_text', '')[:50] if 'sentence_text' in task else ''
                    logger.debug(f"Filtered out point with only {scored} scored tokens (min: {min_tokens_threshold}): {sentence_preview}")
                    continue
                
                meta['mean_logprob'] = mean_lp
                meta['sum_logprob'] = sum_lp
                meta['scored_token_count'] = scored
                meta['total_token_count'] = total
                meta['coverage'] = cov
                points.append(meta)
            
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} point(s) with < {min_tokens_threshold} scored tokens")

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
                prev_rate_chunk0, _ = compute_prev_metrics(baseline_dir / f'scenario_{scen_id}', 0)
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
                    if use_nebius_logprobs and (bundle.model is None):
                        meta = {
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx,
                            'target_name': target_name,
                            'sentence_type': 'baseline_mode',
                            'strategy': None,
                            'delta_whistleblowing_rate': 0.0,
                            'whistleblowing_rate_before': None,
                            'whistleblowing_rate_after': None,
                        }
                        queue_remote_point(prefill, base_sentence, meta)
                    else:
                        try:
                            lp_base = compute_token_logprobs(bundle, prefill, base_sentence)
                        except Exception:
                            lp_base = {'logprob_mean': 0.0, 'logprob_sum': 0.0, 'tokens': [], 'token_ids': [], 'logprobs': [], 'scored_token_count': 0, 'total_token_count': 0}
                        try_append_point({
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx,
                            'target_name': target_name,
                            'sentence_type': 'baseline_mode',
                            'strategy': None,
                            'mean_logprob': float(lp_base.get('logprob_mean', 0.0)),
                            'sum_logprob': float(lp_base.get('logprob_sum', 0.0)),
                            'scored_token_count': int(lp_base.get('scored_token_count', 0)),
                            'total_token_count': int(lp_base.get('total_token_count', 0)),
                            'delta_whistleblowing_rate': 0.0,
                            'whistleblowing_rate_before': None,
                            'whistleblowing_rate_after': None,
                        }, base_sentence)

                # On-policy inserted sentence: score the intervention (selected_continuation_full_text)
                # after the resampled chunk
                intervention_sentence = exp.get('selected_continuation_full_text', '')
                if intervention_sentence and resampled_chunk:
                    # Build full prefix: base + resampled_chunk
                    prefix_text = prefix_text_base + resampled_chunk
                    prefill = build_prefill(system_prompt, user_prompt, email_content, prefix_text)

                    # Compare at chunk i+1 (the chunk immediately following the intervention chunk)
                    prev_rate, n_prev = compute_prev_metrics(baseline_dir / f'scenario_{scen_id}', ch_idx + 1)
                    after_rate, n_after = compute_after_metrics_onpolicy(exp)
                    delta_br = None
                    try:
                        if after_rate is not None and prev_rate is not None:
                            delta_br = float(after_rate) - float(prev_rate)
                    except Exception:
                        delta_br = None
                    if use_nebius_logprobs and (bundle.model is None):
                        meta = {
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx,
                            'target_name': target_name,
                            'sentence_type': 'onpolicy_inserted',
                            'strategy': None,
                            'delta_whistleblowing_rate': (delta_br if delta_br is not None else np.nan),
                            'whistleblowing_rate_before': (prev_rate if prev_rate is not None else np.nan),
                            'whistleblowing_rate_after': (after_rate if after_rate is not None else np.nan),
                        }
                        queue_remote_point(prefill, intervention_sentence, meta)
                    else:
                        try:
                            lp_on = compute_token_logprobs(bundle, prefill, intervention_sentence)
                        except Exception:
                            lp_on = {'logprob_mean': 0.0, 'logprob_sum': 0.0, 'tokens': [], 'token_ids': [], 'logprobs': [], 'scored_token_count': 0, 'total_token_count': 0}
                        try_append_point({
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx,
                            'target_name': target_name,
                            'sentence_type': 'onpolicy_inserted',
                            'strategy': None,
                            'mean_logprob': float(lp_on.get('logprob_mean', 0.0)),
                            'sum_logprob': float(lp_on.get('logprob_sum', 0.0)),
                            'scored_token_count': int(lp_on.get('scored_token_count', 0)),
                            'total_token_count': int(lp_on.get('total_token_count', 0)),
                            'delta_whistleblowing_rate': (delta_br if delta_br is not None else np.nan),
                            'whistleblowing_rate_before': (prev_rate if prev_rate is not None else np.nan),
                            'whistleblowing_rate_after': (after_rate if after_rate is not None else np.nan),
                        }, intervention_sentence)

                # Off-policy (handwritten) sentences for this chunk if available
                # For off-policy, use ORIGINAL chunks (not resampled)
                off_exp = off_exp_index.get(ch_idx)
                if off_exp:
                    # Build prefix with original chunks (chunks[0:ch_idx] only, NOT including ch_idx)
                    prefix_text_off_base = make_prefix_text(chunks, seps, ch_idx)
                    prefix_text_off = prefix_text_off_base + chunks[ch_idx]
                    separator_off = seps[ch_idx] if ch_idx < len(seps) else ''
                    prefill_off = build_prefill(system_prompt, user_prompt, email_content, prefix_text_off)
                    
                    # Compare at chunk i+1 baseline for handwritten off-policy
                    prev_br_off, _ = compute_prev_metrics(baseline_dir / f'scenario_{scen_id}', ch_idx + 1)
                    for strat in (off_exp.get('disruption_results') or []):
                        strat_name = str(strat.get('strategy', ''))
                        disruption_text_raw = str(strat.get('disruption_text', ''))
                        after_br_off = strat.get('whistleblowing_rate', None)
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
                        
                        if use_nebius_logprobs and (bundle.model is None):
                            meta = {
                                'scenario_id': scen_id,
                                'chunk_idx': ch_idx,
                                'target_name': target_name,
                                'sentence_type': 'offpolicy_handwritten',
                                'strategy': strat_name,
                                'delta_whistleblowing_rate': (delta_off if delta_off is not None else np.nan),
                                'whistleblowing_rate_before': (float(prev_br_off) if prev_br_off is not None else np.nan),
                                'whistleblowing_rate_after': (float(after_br_off) if after_br_off is not None else np.nan),
                            }
                            queue_remote_point(prefill_off, disruption_text, meta)
                        else:
                            try:
                                lp_off = compute_token_logprobs(bundle, prefill_off, disruption_text)
                            except Exception:
                                lp_off = {'logprob_mean': 0.0, 'logprob_sum': 0.0, 'tokens': [], 'token_ids': [], 'logprobs': [], 'scored_token_count': 0, 'total_token_count': 0}
                            try_append_point({
                                'scenario_id': scen_id,
                                'chunk_idx': ch_idx,
                                'target_name': target_name,
                                'sentence_type': 'offpolicy_handwritten',
                                'strategy': strat_name,
                                'mean_logprob': float(lp_off.get('logprob_mean', 0.0)),
                                'sum_logprob': float(lp_off.get('logprob_sum', 0.0)),
                                'scored_token_count': int(lp_off.get('scored_token_count', 0)),
                                'total_token_count': int(lp_off.get('total_token_count', 0)),
                                'delta_whistleblowing_rate': (delta_off if delta_off is not None else np.nan),
                                'whistleblowing_rate_before': (float(prev_br_off) if prev_br_off is not None else np.nan),
                                'whistleblowing_rate_after': (float(after_br_off) if after_br_off is not None else np.nan),
                            }, disruption_text)
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
                        resampled_chunk_f = expf.get('selected_resampled_chunk', '')
                        intervention_sentence_f = expf.get('selected_continuation_full_text', '')
                        
                        if not resampled_chunk_f or not intervention_sentence_f:
                            continue
                        
                        # Build prefix: chunks[0:ch_idxf] + resampled_chunk
                        prefix_text_base_f = make_prefix_text(chunks, seps, ch_idxf)
                        prefix_text_f = prefix_text_base_f + resampled_chunk_f
                        prefillf = build_prefill(system_prompt, user_prompt, email_content, prefix_text_f)

                        # For fixed pipeline, prev/after rates are analogous to varied; reuse helper if present
                        # Compare at chunk i+1
                        prev_rate_f, _ = compute_prev_metrics(baseline_dir / f'scenario_{scen_id}', ch_idxf + 1)
                        after_rate_f, _ = compute_after_metrics_onpolicy(expf)
                        delta_f = None
                        try:
                            if after_rate_f is not None and prev_rate_f is not None:
                                delta_f = float(after_rate_f) - float(prev_rate_f)
                        except Exception:
                            delta_f = None

                        if use_nebius_logprobs and (bundle.model is None):
                            meta = {
                                'scenario_id': scen_id,
                                'chunk_idx': ch_idxf,
                                'target_name': target_name_f,
                                'sentence_type': 'onpolicy_fixed',
                                'strategy': None,
                                'delta_whistleblowing_rate': (delta_f if delta_f is not None else np.nan),
                                'whistleblowing_rate_before': (prev_rate_f if prev_rate_f is not None else np.nan),
                                'whistleblowing_rate_after': (after_rate_f if after_rate_f is not None else np.nan),
                            }
                            queue_remote_point(prefillf, intervention_sentence_f, meta)
                        else:
                            try:
                                lp_fixed = compute_token_logprobs(bundle, prefillf, intervention_sentence_f)
                            except Exception:
                                lp_fixed = {'logprob_mean': 0.0, 'logprob_sum': 0.0, 'tokens': [], 'token_ids': [], 'logprobs': [], 'scored_token_count': 0, 'total_token_count': 0}
                            try_append_point({
                                'scenario_id': scen_id,
                                'chunk_idx': ch_idxf,
                                'target_name': target_name_f,
                                'sentence_type': 'onpolicy_fixed',
                                'strategy': None,
                                'mean_logprob': float(lp_fixed.get('logprob_mean', 0.0)),
                                'sum_logprob': float(lp_fixed.get('logprob_sum', 0.0)),
                                'scored_token_count': int(lp_fixed.get('scored_token_count', 0)),
                                'total_token_count': int(lp_fixed.get('total_token_count', 0)),
                                'delta_whistleblowing_rate': (delta_f if delta_f is not None else np.nan),
                                'whistleblowing_rate_before': (prev_rate_f if prev_rate_f is not None else np.nan),
                                'whistleblowing_rate_after': (after_rate_f if after_rate_f is not None else np.nan),
                            }, intervention_sentence_f)
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
                # Compare at chunk i+1 baseline
                prev_br_off2, _ = compute_prev_metrics(baseline_dir / f'scenario_{scen_id}', ch_idx2 + 1)
                for strat in (e.get('disruption_results') or []):
                    strat_name2 = str(strat.get('strategy', ''))
                    if (ch_idx2, strat_name2) in added_off_keys:
                        continue
                    disruption_text2_raw = str(strat.get('disruption_text', ''))
                    after_br_off2 = strat.get('whistleblowing_rate', None)
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
                    
                    if use_nebius_logprobs and (bundle.model is None):
                        meta = {
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx2,
                            'target_name': str(off_json.get('target_name') or 'offpolicy'),
                            'sentence_type': 'offpolicy_handwritten',
                            'strategy': strat_name2,
                            'delta_whistleblowing_rate': (delta_off2 if delta_off2 is not None else np.nan),
                            'whistleblowing_rate_before': (float(prev_br_off2) if prev_br_off2 is not None else np.nan),
                            'whistleblowing_rate_after': (float(after_br_off2) if after_br_off2 is not None else np.nan),
                        }
                        queue_remote_point(prefill2, disruption_text2, meta)
                    else:
                        try:
                            lp_off2 = compute_token_logprobs(bundle, prefill2, disruption_text2)
                        except Exception:
                            lp_off2 = {'logprob_mean': 0.0, 'logprob_sum': 0.0, 'tokens': [], 'token_ids': [], 'logprobs': [], 'scored_token_count': 0, 'total_token_count': 0}
                        try_append_point({
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx2,
                            'target_name': str(off_json.get('target_name') or 'offpolicy'),
                            'sentence_type': 'offpolicy_handwritten',
                            'strategy': strat_name2,
                            'mean_logprob': float(lp_off2.get('logprob_mean', 0.0)),
                            'sum_logprob': float(lp_off2.get('logprob_sum', 0.0)),
                            'scored_token_count': int(lp_off2.get('scored_token_count', 0)),
                            'total_token_count': int(lp_off2.get('total_token_count', 0)),
                            'delta_whistleblowing_rate': (delta_off2 if delta_off2 is not None else np.nan),
                            'whistleblowing_rate_before': (float(prev_br_off2) if prev_br_off2 is not None else np.nan),
                            'whistleblowing_rate_after': (float(after_br_off2) if after_br_off2 is not None else np.nan),
                        }, disruption_text2)

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
                # Compare at chunk i+1 baseline
                prev_br_cross, _ = compute_prev_metrics(baseline_dir / f'scenario_{scen_id}', ch_idx3 + 1)
                for strat in (e.get('disruption_results') or []):
                    # cross json may have chosen_text; prefer that for display
                    strat_name3 = str(strat.get('strategy', ''))
                    disruption_text3_raw = str(strat.get('chosen_text') or strat.get('disruption_text') or '')
                    after_br_cross = strat.get('whistleblowing_rate', None)
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
                    
                    if use_nebius_logprobs and (bundle.model is None):
                        meta = {
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx3,
                            'target_name': str(off_cross_json.get('target_name') or 'offpolicy cross model'),
                            'sentence_type': 'offpolicy_cross_model',
                            'strategy': strat_name3,
                            'delta_whistleblowing_rate': (delta_cross if delta_cross is not None else np.nan),
                            'whistleblowing_rate_before': (float(prev_br_cross) if prev_br_cross is not None else np.nan),
                            'whistleblowing_rate_after': (float(after_br_cross) if after_br_cross is not None else np.nan),
                        }
                        queue_remote_point(prefill3, disruption_text3, meta)
                    else:
                        try:
                            lp_cross = compute_token_logprobs(bundle, prefill3, disruption_text3)
                        except Exception:
                            lp_cross = {'logprob_mean': 0.0, 'logprob_sum': 0.0, 'tokens': [], 'token_ids': [], 'logprobs': [], 'scored_token_count': 0, 'total_token_count': 0}
                        try_append_point({
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx3,
                            'target_name': str(off_cross_json.get('target_name') or 'offpolicy cross model'),
                            'sentence_type': 'offpolicy_cross_model',
                            'strategy': strat_name3,
                            'mean_logprob': float(lp_cross.get('logprob_mean', 0.0)),
                            'sum_logprob': float(lp_cross.get('logprob_sum', 0.0)),
                            'scored_token_count': int(lp_cross.get('scored_token_count', 0)),
                            'total_token_count': int(lp_cross.get('total_token_count', 0)),
                            'delta_whistleblowing_rate': (delta_cross if delta_cross is not None else np.nan),
                            'whistleblowing_rate_before': (float(prev_br_cross) if prev_br_cross is not None else np.nan),
                            'whistleblowing_rate_after': (float(after_br_cross) if after_br_cross is not None else np.nan),
                        }, disruption_text3)

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
                        # Include chunk at ch_idxa in prefix (intervention comes AFTER)
                        prefix_texta = make_prefix_text(chunks, seps, ch_idxa + 1)
                        prefilla = build_prefill(system_prompt, user_prompt, email_content, prefix_texta)
                        best_sentence = str(expa.get('best_sentence') or '')
                        if not best_sentence:
                            continue
                        # Compare at chunk i+1 baseline
                        prev_rate_a, _ = compute_prev_metrics(baseline_dir / f'scenario_{scen_id}', ch_idxa + 1)
                        after_rate_a, _ = compute_after_metrics_onpolicy(expa)  # reuse if they rolled out+classified; else None
                        delta_a = None
                        try:
                            if after_rate_a is not None and prev_rate_a is not None:
                                delta_a = float(after_rate_a) - float(prev_rate_a)
                        except Exception:
                            delta_a = None
                        if use_nebius_logprobs and (bundle.model is None):
                            meta = {
                                'scenario_id': scen_id,
                                'chunk_idx': ch_idxa,
                                'target_name': target_name_a,
                                'sentence_type': 'agent_onpolicy_discovered',
                                'strategy': None,
                                'delta_whistleblowing_rate': (delta_a if delta_a is not None else np.nan),
                                'whistleblowing_rate_before': (prev_rate_a if prev_rate_a is not None else np.nan),
                                'whistleblowing_rate_after': (after_rate_a if after_rate_a is not None else np.nan),
                            }
                            queue_remote_point(prefilla, best_sentence, meta)
                        else:
                            try:
                                lp_agent = compute_token_logprobs(bundle, prefilla, best_sentence)
                            except Exception:
                                lp_agent = {'logprob_mean': 0.0, 'logprob_sum': 0.0, 'tokens': [], 'token_ids': [], 'logprobs': [], 'scored_token_count': 0, 'total_token_count': 0}
                            try_append_point({
                                'scenario_id': scen_id,
                                'chunk_idx': ch_idxa,
                                'target_name': target_name_a,
                                'sentence_type': 'agent_onpolicy_discovered',
                                'strategy': None,
                                'mean_logprob': float(lp_agent.get('logprob_mean', 0.0)),
                                'sum_logprob': float(lp_agent.get('logprob_sum', 0.0)),
                                'scored_token_count': int(lp_agent.get('scored_token_count', 0)),
                                'total_token_count': int(lp_agent.get('total_token_count', 0)),
                                'delta_whistleblowing_rate': (delta_a if delta_a is not None else np.nan),
                                'whistleblowing_rate_before': (prev_rate_a if prev_rate_a is not None else np.nan),
                                'whistleblowing_rate_after': (after_rate_a if after_rate_a is not None else np.nan),
                            }, best_sentence)
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
                # Compare at chunk i+1 baseline
                prev_br_same, _ = compute_prev_metrics(baseline_dir / f'scenario_{scen_id}', ch_idx4 + 1)
                for strat in (e.get('disruption_results') or []):
                    strat_name4 = str(strat.get('strategy', ''))
                    disruption_text4_raw = str(strat.get('chosen_text') or strat.get('disruption_text') or '')
                    after_br_same = strat.get('whistleblowing_rate', None)
                    if after_br_same is None:
                        after_br_same = strat.get('whistleblowing_rate', None)
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
                    
                    if use_nebius_logprobs and (bundle.model is None):
                        meta = {
                            'scenario_id': scen_id,
                            'chunk_idx': ch_idx4,
                            'target_name': str(off_same_json.get('target_name') or 'offpolicy same model'),
                            'sentence_type': 'offpolicy_same_model',
                            'strategy': strat_name4,
                            'delta_whistleblowing_rate': (delta_same if delta_same is not None else np.nan),
                            'whistleblowing_rate_before': (float(prev_br_same) if prev_br_same is not None else np.nan),
                            'whistleblowing_rate_after': (float(after_br_same) if after_br_same is not None else np.nan),
                        }
                        queue_remote_point(prefill4, disruption_text4, meta)
                else:
                    try:
                        lp_same = compute_token_logprobs(bundle, prefill4, disruption_text4)
                    except Exception:
                        lp_same = {'logprob_mean': 0.0, 'logprob_sum': 0.0, 'tokens': [], 'token_ids': [], 'logprobs': [], 'scored_token_count': 0, 'total_token_count': 0}
                    try_append_point({
                        'scenario_id': scen_id,
                        'chunk_idx': ch_idx4,
                        'target_name': str(off_same_json.get('target_name') or 'offpolicy same model'),
                        'sentence_type': 'offpolicy_same_model',
                        'strategy': strat_name4,
                        'mean_logprob': float(lp_same.get('logprob_mean', 0.0)),
                        'sum_logprob': float(lp_same.get('logprob_sum', 0.0)),
                        'scored_token_count': int(lp_same.get('scored_token_count', 0)),
                        'total_token_count': int(lp_same.get('total_token_count', 0)),
                        'delta_whistleblowing_rate': (delta_same if delta_same is not None else np.nan),
                        'whistleblowing_rate_before': (float(prev_br_same) if prev_br_same is not None else np.nan),
                        'whistleblowing_rate_after': (float(after_br_same) if after_br_same is not None else np.nan),
                    }, disruption_text4)

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
                plot_path_stream_abs = out_root / f"scatter_distribution{suf}_abs.png"
                plot_distribution_scatter(df_stream_main, plot_path_stream_abs, display_model=display_model, model_folder=model_folder, include_extras=False, abs_value=True)
                plot_path_stream_ex = out_root / f"scatter_distribution_with_extras{suf}.png"
                plot_distribution_scatter(df_stream_main, plot_path_stream_ex, display_model=display_model, model_folder=model_folder, include_extras=True)
                plot_path_stream_ex_abs = out_root / f"scatter_distribution_with_extras{suf}_abs.png"
                plot_distribution_scatter(df_stream_main, plot_path_stream_ex_abs, display_model=display_model, model_folder=model_folder, include_extras=True, abs_value=True)
                print(f"[save] interim points -> {csv_path_stream}")
                print(f"[save] interim scatter -> {plot_path_stream}")
                print(f"[save] interim scatter (abs) -> {plot_path_stream_abs}")
                print(f"[save] interim scatter (with extras) -> {plot_path_stream_ex}")
                print(f"[save] interim scatter (with extras, abs) -> {plot_path_stream_ex_abs}")
            except Exception as _e:
                logger.warning(f"interim save failed: {type(_e).__name__}: {_e}")

    # Report total filtered count
    if filtered_points_count[0] > 0:
        logger.info(f"Total HF points filtered: {filtered_points_count[0]} (< {int(globals().get('min_scored_tokens_threshold', 2))} scored tokens)")
    
    return points


def plot_distribution_scatter(df: pd.DataFrame, out_path: Path, display_model: str, model_folder: str, include_extras: bool = False, title_override: Optional[str] = None, abs_value: bool = False) -> None:
    if df.empty:
        print('[dist] No data to plot.')
        return
    # Apply global min_logprob filter when configured
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
        print('[dist] No data to plot after min_logprob filter.')
        return
    plt.figure(figsize=(12, 7))
    sns.set_style('white')
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
    df_plot = df_local.copy()
    if not include_extras:
        allowed = {"onpolicy_inserted", "onpolicy_fixed", "offpolicy_handwritten", "baseline_mode"}
        if 'sentence_type' in df_plot.columns:
            df_plot = df_plot[df_plot['sentence_type'].isin(list(allowed))]
    mask_base = df_plot['sentence_type'] == 'baseline_mode'
    if 'delta_whistleblowing_rate' in df_plot.columns:
        y_vals = df_plot['delta_whistleblowing_rate'].astype(float).copy()
        if mask_base.any():
            jitter_vals = np.random.normal(loc=0.0, scale=0.005, size=int(mask_base.sum()))
            # Only fill NaN for baseline points; leave others as NaN so they do not plot
            y_vals.loc[mask_base] = y_vals.loc[mask_base].fillna(0.0) + jitter_vals
        # Apply absolute value if requested
        if abs_value:
            y_vals = y_vals.abs()
        df_plot['delta_whistleblowing_rate_plot'] = y_vals
        y_col = 'delta_whistleblowing_rate_plot'
    else:
        y_col = 'delta_whistleblowing_rate'

    # Visual grouping: combine on-policy inserted/fixed as a single 'resampled' category (red dots)
    df_plot['viz_type'] = df_plot['sentence_type'].apply(lambda st: 'resampled' if st in ('onpolicy_inserted', 'onpolicy_fixed') else st)
    # Optional overlap filter: keep only settings present in both resampled and handwritten
    try:
        overlap_only = bool(globals().get('overlap_red_blue_only_enabled', False))
    except Exception:
        overlap_only = False
    if overlap_only and all(c in df_plot.columns for c in ['viz_type', 'scenario_id', 'chunk_idx', 'target_name']):
        try:
            # Build keys for resampled and handwritten
            resampled_keys = set(zip(
                df_plot.loc[df_plot['viz_type'] == 'resampled', 'scenario_id'],
                df_plot.loc[df_plot['viz_type'] == 'resampled', 'chunk_idx'],
                df_plot.loc[df_plot['viz_type'] == 'resampled', 'target_name']
            ))
            handwritten_keys = set(zip(
                df_plot.loc[df_plot['viz_type'] == 'offpolicy_handwritten', 'scenario_id'],
                df_plot.loc[df_plot['viz_type'] == 'offpolicy_handwritten', 'chunk_idx'],
                df_plot.loc[df_plot['viz_type'] == 'offpolicy_handwritten', 'target_name']
            ))
            keep_keys = resampled_keys.intersection(handwritten_keys)
            if keep_keys:
                mask_keep = list(zip(df_plot.get('scenario_id'), df_plot.get('chunk_idx'), df_plot.get('target_name')))
                df_plot = df_plot[[k in keep_keys for k in mask_keep]]
        except Exception:
            pass

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
    # Optional class balancing: cap each non-resampled class to resampled count
    try:
        balance = bool(globals().get('balance_classes_enabled', False))
    except Exception:
        balance = False
    if balance and 'viz_type' in df_plot.columns:
        try:
            n_r = int(df_plot[df_plot['viz_type'] == 'resampled'].shape[0])
            if n_r > 0:
                def _cap_group(g: pd.DataFrame) -> pd.DataFrame:
                    if getattr(g, 'name', None) == 'resampled' or len(g) <= n_r:
                        return g
                    return g.sample(n=n_r, random_state=0)
                df_plot = df_plot.groupby('viz_type', group_keys=False).apply(_cap_group)
        except Exception:
            pass
    if df_plot.empty:
        print('[dist] No data to plot after class balancing filter.')
        return
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
        plt.scatter(
            sub['mean_logprob'].astype(float).values,
            sub[y_col].astype(float).values,
            alpha=0.55,
            s=viz_sizes.get(stype, 24),
            c=viz_palette.get(stype, '#7f7f7f'),
            marker=shapes_by_model.get(str(mf), 'o'),
            edgecolors='none',
        )

    plt.axhline(0.0, color='gray', linewidth=1, linestyle='--')
    # Optional trendline across all points
    trend_r2 = None
    try:
        if bool(globals().get('trendline_enabled', False)):
            x_vals = pd.to_numeric(df_plot['mean_logprob'], errors='coerce')
            y_vals = pd.to_numeric(df_plot[y_col], errors='coerce')
            mask = (~x_vals.isna()) & (~y_vals.isna())
            x_np = x_vals[mask].values
            y_np = y_vals[mask].values
            if x_np.size >= 2 and float(np.std(x_np)) > 0.0:
                slope, intercept = np.polyfit(x_np, y_np, 1)
                y_pred = slope * x_np + intercept
                ss_res = float(np.sum((y_np - y_pred) ** 2))
                ss_tot = float(np.sum((y_np - float(np.mean(y_np))) ** 2))
                if ss_tot > 0.0:
                    trend_r2 = 1.0 - (ss_res / ss_tot)
                x_line = np.linspace(float(np.min(x_np)), float(np.max(x_np)), 100)
                y_line = slope * x_line + intercept
                plt.plot(x_line, y_line, linestyle='--', color='#444444', linewidth=1.6)
    except Exception:
        trend_r2 = None
    # plt.xlabel(f'Mean token logprob under {display_model}')
    plt.xlabel(f'Mean token logprob')
    ylabel = '| Δ Whistleblowing rate (after − before) |' if abs_value else 'Δ Whistleblowing rate (after − before)'
    plt.ylabel(ylabel)
    plt.title(title_override or f'Sentence insertions vs impact\nModel: {model_folder}  Dataset: Whistleblowing')
    # Dual legends: match shared plot
    from matplotlib.lines import Line2D  # local import
    present_types = [t for t in df_plot['viz_type'].dropna().unique()]
    type_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=8,
               markerfacecolor=viz_palette.get(t, '#7f7f7f'), markeredgecolor='none', label=label_map.get(t, t))
        for t in present_types
    ]
    # Add trendline legend entry if enabled
    if trend_r2 is not None:
        try:
            type_handles.append(Line2D([0], [0], linestyle='--', color='#444444', linewidth=2.0, label=f"trendline (R^2={trend_r2:.2f})"))
        except Exception:
            pass
    model_handles: List[Line2D] = []
    for mf in present_models:
        model_handles.append(Line2D([0], [0], marker=shapes_by_model.get(mf, 'o'), linestyle='None', markersize=8,
                                    markerfacecolor='#cccccc', markeredgecolor='#333333', label=mf))
    leg1 = plt.legend(handles=type_handles, title='Type', loc='lower left')
    if model_handles and len(present_models) > 1:
        leg2 = plt.legend(handles=model_handles, title='Model', loc='upper left')
        ax = plt.gca()
        ax.add_artist(leg1)
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
    abs_value: bool = False,
) -> None:
    if df.empty:
        print('[dist] No data to plot (shared).')
        return
    # Apply global min_logprob filter when configured
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
        print('[dist] No data to plot after min_logprob filter (shared).')
        return
    sns.set_style('white')
    ax = plt.gca()

    # Group domain
    df_plot = df_local.copy()
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
    # Optional overlap filter: keep only settings present in both resampled and handwritten
    try:
        overlap_only = bool(globals().get('overlap_red_blue_only_enabled', False))
    except Exception:
        overlap_only = False
    if overlap_only and all(c in df_plot.columns for c in ['viz_type', 'scenario_id', 'chunk_idx', 'target_name']):
        try:
            resampled_keys = set(zip(
                df_plot.loc[df_plot['viz_type'] == 'resampled', 'scenario_id'],
                df_plot.loc[df_plot['viz_type'] == 'resampled', 'chunk_idx'],
                df_plot.loc[df_plot['viz_type'] == 'resampled', 'target_name']
            ))
            handwritten_keys = set(zip(
                df_plot.loc[df_plot['viz_type'] == 'offpolicy_handwritten', 'scenario_id'],
                df_plot.loc[df_plot['viz_type'] == 'offpolicy_handwritten', 'chunk_idx'],
                df_plot.loc[df_plot['viz_type'] == 'offpolicy_handwritten', 'target_name']
            ))
            keep_keys = resampled_keys.intersection(handwritten_keys)
            if keep_keys:
                mask_keep = list(zip(df_plot.get('scenario_id'), df_plot.get('chunk_idx'), df_plot.get('target_name')))
                df_plot = df_plot[[k in keep_keys for k in mask_keep]]
        except Exception:
            pass
    # Optional class balancing: cap each non-resampled class to resampled count
    try:
        balance = bool(globals().get('balance_classes_enabled', False))
    except Exception:
        balance = False
    if balance and 'viz_type' in df_plot.columns:
        try:
            n_r = int(df_plot[df_plot['viz_type'] == 'resampled'].shape[0])
            if n_r > 0:
                def _cap_group(g: pd.DataFrame) -> pd.DataFrame:
                    if getattr(g, 'name', None) == 'resampled' or len(g) <= n_r:
                        return g
                    return g.sample(n=n_r, random_state=0)
                df_plot = df_plot.groupby('viz_type', group_keys=False).apply(_cap_group)
        except Exception:
            pass
    if df_plot.empty:
        print('[dist] No data to plot after class balancing filter (shared).')
        return

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

    plt.figure(figsize=(12, 5))
    for (stype, mf), sub in df_plot.groupby(['viz_type', 'model_folder']):
        y_data = sub['delta_whistleblowing_rate'].astype(float).values
        if abs_value:
            y_data = np.abs(y_data)
        plt.scatter(
            sub['mean_logprob'].astype(float).values,
            y_data,
            alpha=0.55,
            s=size_map.get(stype, 24),
            c=color_map.get(stype, '#7f7f7f'),
            marker=shapes_by_model.get(str(mf), 'o'),
            edgecolors='none',
        )
    if not abs_value:
        plt.axhline(0.0, color='gray', linewidth=1, linestyle='--')
    # Optional trendline across all points (shared)
    try:
        if bool(globals().get('trendline_enabled', False)):
            x_vals = pd.to_numeric(df_plot['mean_logprob'], errors='coerce')
            y_vals = pd.to_numeric(df_plot['delta_whistleblowing_rate'], errors='coerce')
            if abs_value:
                y_vals = y_vals.abs()
            mask = (~x_vals.isna()) & (~y_vals.isna())
            x_np = x_vals[mask].values
            y_np = y_vals[mask].values
            if x_np.size >= 2 and float(np.std(x_np)) > 0.0:
                slope, intercept = np.polyfit(x_np, y_np, 1)
                y_pred = slope * x_np + intercept
                ss_res = float(np.sum((y_np - y_pred) ** 2))
                ss_tot = float(np.sum((y_np - float(np.mean(y_np))) ** 2))
                trend_r2 = None
                if ss_tot > 0.0:
                    trend_r2 = 1.0 - (ss_res / ss_tot)
                x_line = np.linspace(float(np.min(x_np)), float(np.max(x_np)), 100)
                y_line = slope * x_line + intercept
                plt.plot(x_line, y_line, linestyle='--', color='#444444', linewidth=1.6, label=(f"trendline (R^2={trend_r2:.2f})" if trend_r2 is not None else 'trendline'))
    except Exception:
        pass
    plt.xlabel('Mean token logprob')
    ylabel = '| Δ Whistleblowing rate (after − before) |' if abs_value else 'Δ Whistleblowing rate (after − before)'
    plt.ylabel(ylabel)
    plt.title(title_override or 'Sentence insertions vs impact\nDataset: Whistleblowing')
    # Legends
    from matplotlib.lines import Line2D  # local import
    present_types = [t for t in df_plot['viz_type'].dropna().unique()]
    type_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=8,
               markerfacecolor=color_map.get(t, '#7f7f7f'), markeredgecolor='none', label=label_map.get(t, t))
        for t in present_types
    ]
    model_handles: List[Line2D] = []
    if 'model_folder' in df_plot.columns:
        present_models = [str(x) for x in df_plot['model_folder'].dropna().unique()]
        for mf in present_models:
            model_handles.append(Line2D([0], [0], marker=shapes_by_model.get(mf, 'o'), linestyle='None', markersize=8,
                                        markerfacecolor='#cccccc', markeredgecolor='#333333', label=mf))
    # Place legends outside on the right to avoid overlap; larger fonts for shared plot
    # Merge any existing trendline label from plot into legend entries
    handles_auto, labels_auto = plt.gca().get_legend_handles_labels()
    if handles_auto and labels_auto:
        type_handles = type_handles + handles_auto
    leg1 = plt.legend(handles=type_handles, title='Type', loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0., fontsize=14)
    try:
        leg1.get_title().set_fontsize(14)
    except Exception:
        pass
    if model_handles and len(present_models) > 1:
        leg2 = plt.legend(handles=model_handles, title='Model', loc='upper left', bbox_to_anchor=(1.02, 0.45), borderaxespad=0., fontsize=14)
        try:
            leg2.get_title().set_fontsize(14)
        except Exception:
            pass
        ax = plt.gca()
        ax.add_artist(leg1)
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
            base_sentence = str(DISRUPTION_LIBRARY_WB.get(cat, '')).strip()
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
            hand_text = str(DISRUPTION_LIBRARY_WB.get(cat, '')).strip()
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
    parser = argparse.ArgumentParser(description='On-policy/off-policy distribution visualization (whistleblowing)')
    parser.add_argument('-m', '--models', type=str, nargs='*', default=list(MODEL_FOLDER_TO_HF.keys()), choices=list(MODEL_FOLDER_TO_HF.keys()), help='Model folders (space-separated; default: all models)')
    parser.add_argument('-t', '--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('-tp', '--top_p', type=float, default=DEFAULT_TOP_P)
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='Device (cuda:0 or cpu)')
    parser.add_argument('-nq', '--no_quantize', action='store_true', default=False, help='Disable 4-bit quantization')
    parser.add_argument('-fc', '--force-cpu', action='store_true', default=False, help='Force full CPU inference even if GPU is available')
    parser.add_argument('-unl', '--use-nebius-logprobs', action='store_true', default=True, help='Use Nebius API to fetch per-token logprobs when local model does not fit (default: False)')
    parser.add_argument('-ms', '--max_scenarios', type=int, default=None, help='Limit number of scenarios to scan')
    parser.add_argument('-tg', '--targets', type=str, default=None, help='Comma-separated target names to include (e.g., ethical,question)')
    parser.add_argument('-ib', '--include-baseline', action='store_true', default=False, help='Include baseline (original chunk) points in the plot (default: False)')
    parser.add_argument('-ll', '--log-level', type=str, default='WARNING', choices=['DEBUG','INFO','WARNING','ERROR'], help='Logging level')
    parser.add_argument('-nc', '--nebius-concurrency', type=int, default=16, help='Max concurrent Nebius requests when using remote logprobs (default: 16)')
    parser.add_argument('-ct', '--compare_to', type=str, default='current', choices=['base','current'], help='Baseline to compare deltas against: base (chunk_0) or current (chunk_idx)')
    parser.add_argument('-pm', '--plot_methods', type=str, default='cross,handwritten,same,resampled', help='Comma list of methods to plot: cross,handwritten,same,resampled,agent[,baseline]')
    parser.add_argument('-to', '--timeout_seconds', type=int, default=360, help='Nebius API timeout per request in seconds (default: 180)')
    parser.add_argument('-spo', '--shared_plot_only', action='store_true', default=False, help='Only build a single shared plot by reading existing CSVs across models; do not recompute logprobs')
    parser.add_argument('-niss', '--include_niss_onpolicy', action='store_true', default=True, help='Include *_niss on-policy outputs when aggregating points (default: True)')
    parser.add_argument('-rnd', '--use_random_variant', action='store_true', default=False, help='Use *_random input folders and save plots with a _random suffix alongside originals')
    parser.add_argument('-ml', '--min_mean_logprob', type=float, default=-5, help='Minimum mean_logprob; omit points with mean_logprob <= this from all plots')
    parser.add_argument('-bc', '--balance_classes', action='store_true', default=False, help='When set, downsample each non-resampled class to match count of resampled points')
    parser.add_argument('-oorb', '--only_overlap_resampled_handwritten', action='store_true', default=False, help='When set, keep only settings (scenario, chunk, category) present in BOTH resampled and handwritten')
    parser.add_argument('-tl', '--add_trendline', action='store_true', default=False, help='Add dashed best-fit line with R^2 value to plot legends')
    parser.add_argument('-mst', '--min_scored_tokens', type=int, default=5, help='Minimum number of tokens that must be scored for a point to be included (default: 2; use higher like 5-10 to filter bad API responses)')
    args = parser.parse_args()
    
    # Expose min threshold globally for plotting functions
    try:
        globals()['min_logprob_threshold'] = (float(args.min_mean_logprob) if args.min_mean_logprob is not None else None)
    except Exception:
        globals()['min_logprob_threshold'] = None
    # Expose min scored tokens threshold globally
    try:
        globals()['min_scored_tokens_threshold'] = int(args.min_scored_tokens) if args.min_scored_tokens is not None else 2
    except Exception:
        globals()['min_scored_tokens_threshold'] = 2
    # Expose class balancing flag globally for plotting functions
    try:
        globals()['balance_classes_enabled'] = bool(args.balance_classes)
    except Exception:
        globals()['balance_classes_enabled'] = False
    # Expose overlap filtering flag globally for plotting functions
    try:
        globals()['overlap_red_blue_only_enabled'] = bool(args.only_overlap_resampled_handwritten)
    except Exception:
        globals()['overlap_red_blue_only_enabled'] = False
    # Expose trendline flag globally for plotting functions
    try:
        globals()['trendline_enabled'] = bool(args.add_trendline)
    except Exception:
        globals()['trendline_enabled'] = False

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
            # Shared plotting only: drop extremely off-distribution points with mean_logprob < -10
            if 'mean_logprob' in df_all_main.columns:
                try:
                    mlp = pd.to_numeric(df_all_main['mean_logprob'], errors='coerce')
                    df_all_main = df_all_main[mlp >= -10]
                except Exception:
                    pass
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
        # Bump font sizes globally for shared plots and make figure shorter
        try:
            plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14, 'figure.titlesize': 20})
        except Exception:
            pass
        plot_shared_distribution_scatter(df_all_main, out_path_shared, include_extras=True, plot_methods=plot_methods_list)
        
        # Absolute value version
        out_path_shared_abs_spo = shared_root / ('scatter_distribution_shared_random_abs.png' if bool(args.use_random_variant) else 'scatter_distribution_shared_abs.png')
        plot_shared_distribution_scatter(df_all_main, out_path_shared_abs_spo, include_extras=True, plot_methods=plot_methods_list, abs_value=True)
        
        # Read back axis limits from the saved data for consistency
        try:
            df_axis = df_all_main.copy()
            thr = globals().get('min_logprob_threshold', None)
            if thr is not None and 'mean_logprob' in df_axis.columns:
                mlp_axis = pd.to_numeric(df_axis['mean_logprob'], errors='coerce')
                df_axis = df_axis[mlp_axis > float(thr)]
            # Apply overlap filter for axis estimation as well
            try:
                overlap_only = bool(globals().get('overlap_red_blue_only_enabled', False))
            except Exception:
                overlap_only = False
            if overlap_only and all(c in df_axis.columns for c in ['sentence_type','scenario_id','chunk_idx','target_name']):
                df_axis = df_axis.copy()
                df_axis['viz_type'] = df_axis['sentence_type'].apply(lambda st: 'resampled' if st in ('onpolicy_inserted', 'onpolicy_fixed') else st)
                resampled_keys = set(zip(
                    df_axis.loc[df_axis['viz_type'] == 'resampled', 'scenario_id'],
                    df_axis.loc[df_axis['viz_type'] == 'resampled', 'chunk_idx'],
                    df_axis.loc[df_axis['viz_type'] == 'resampled', 'target_name']
                ))
                handwritten_keys = set(zip(
                    df_axis.loc[df_axis['viz_type'] == 'offpolicy_handwritten', 'scenario_id'],
                    df_axis.loc[df_axis['viz_type'] == 'offpolicy_handwritten', 'chunk_idx'],
                    df_axis.loc[df_axis['viz_type'] == 'offpolicy_handwritten', 'target_name']
                ))
                keep_keys = resampled_keys.intersection(handwritten_keys)
                if keep_keys:
                    mask_keep = list(zip(df_axis.get('scenario_id'), df_axis.get('chunk_idx'), df_axis.get('target_name')))
                    df_axis = df_axis[[k in keep_keys for k in mask_keep]]
            x_min = float(np.nanmin(df_axis['mean_logprob'].astype(float))) if ('mean_logprob' in df_axis.columns and not df_axis.empty) else None
            x_max = float(np.nanmax(df_axis['mean_logprob'].astype(float))) if ('mean_logprob' in df_axis.columns and not df_axis.empty) else None
            y_min = float(np.nanmin(df_axis['delta_whistleblowing_rate'].astype(float))) if ('delta_whistleblowing_rate' in df_axis.columns and not df_axis.empty) else None
            y_max = float(np.nanmax(df_axis['delta_whistleblowing_rate'].astype(float))) if ('delta_whistleblowing_rate' in df_axis.columns and not df_axis.empty) else None
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
        print(f"[save] shared scatter (abs) -> {out_path_shared_abs_spo}")

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
            # Shared plotting only: drop mean_logprob < -10
            if 'mean_logprob' in df_sp.columns:
                try:
                    mlp_sp = pd.to_numeric(df_sp['mean_logprob'], errors='coerce')
                    df_sp = df_sp[mlp_sp >= -10].copy()
                except Exception:
                    pass
            
            out_sp = shared_root / ('scatter_distribution_self_preservation_shared_random.png' if bool(args.use_random_variant) else 'scatter_distribution_self_preservation_shared.png')
            plot_shared_distribution_scatter(
                df_sp,
                out_sp,
                include_extras=True,
                plot_methods=plot_methods_list,
                xlim=None,
                ylim=ylim_shared,
                title_override='Self-preservation sentence insertions vs impact'
            )
            print(f"[save] shared scatter (self-preservation) -> {out_sp}")
            
            # Absolute value version
            out_sp_abs_spo = shared_root / ('scatter_distribution_self_preservation_shared_random_abs.png' if bool(args.use_random_variant) else 'scatter_distribution_self_preservation_shared_abs.png')
            plot_shared_distribution_scatter(
                df_sp,
                out_sp_abs_spo,
                include_extras=True,
                plot_methods=plot_methods_list,
                xlim=None,
                ylim=ylim_shared,
                title_override='Self-preservation sentence insertions vs impact',
                abs_value=True
            )
            print(f"[save] shared scatter (self-preservation, abs) -> {out_sp_abs_spo}")
        except Exception as _e:
            logger.warning(f"shared self-preservation plot failed: {type(_e).__name__}: {_e}")
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

        hf_id = MODEL_FOLDER_TO_HF.get(model_folder, next(iter(MODEL_FOLDER_TO_HF.values())))
        tokenizer_cache: Dict[str, Any] = {}
        bundle: Optional[HFModelBundle] = None
        if not args.use_nebius_logprobs:
            try:
                if args.force_cpu:
                    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
                device_map_pref = 'balanced_low_0' if (not args.force_cpu) and torch.cuda.is_available() else 'auto'
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
            use_nebius_logprobs=bool(args.use_nebius_logprobs),
            model_folder=model_folder,
            nebius_api_key=nebius_api_key,
            tokenizer_cache=tokenizer_cache,
            stream_save_every=1,
            out_root=out_root,
            display_model=hf_id,
            nebius_concurrency=int(args.nebius_concurrency),
            compare_to=str(args.compare_to),
            include_niss_onpolicy=bool(args.include_niss_onpolicy),
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
        
        # Absolute value version
        plot_path_abs = out_root / f"scatter_distribution{suffix}_abs.png"
        plot_distribution_scatter(df_main, plot_path_abs, display_model=display_model, model_folder=model_folder, include_extras=False, abs_value=True)
        print(f"[save] scatter (abs) -> {plot_path_abs}")

        # Second plot including extra off-policy categories explicitly
        extras_plot_path = out_root / f"scatter_distribution_with_extras{suffix}.png"
        plot_distribution_scatter(df_main, extras_plot_path, display_model=display_model, model_folder=model_folder, include_extras=True)
        print(f"[save] scatter (with extras) -> {extras_plot_path}")
        
        # Absolute value version with extras
        extras_plot_path_abs = out_root / f"scatter_distribution_with_extras{suffix}_abs.png"
        plot_distribution_scatter(df_main, extras_plot_path_abs, display_model=display_model, model_folder=model_folder, include_extras=True, abs_value=True)
        print(f"[save] scatter (with extras, abs) -> {extras_plot_path_abs}")

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
                    plot_distribution_scatter(df_sp, sp_plot_path, display_model=display_model, model_folder=model_folder, include_extras=False, title_override=f'Self-preservation sentence insertions vs impact\nModel: {model_folder}  Dataset: Whistleblowing')
                    print(f"[save] scatter (self-preservation) -> {sp_plot_path}")
                    # Absolute value version
                    sp_plot_path_abs = out_root / f"scatter_distribution_self_preservation{suffix}_abs.png"
                    plot_distribution_scatter(df_sp, sp_plot_path_abs, display_model=display_model, model_folder=model_folder, include_extras=False, title_override=f'Self-preservation sentence insertions vs impact\nModel: {model_folder}  Dataset: Whistleblowing', abs_value=True)
                    print(f"[save] scatter (self-preservation, abs) -> {sp_plot_path_abs}")
                    # Optionally include baseline_mode for self-preservation-only too (same df_sp already includes baseline if present)
                    sp_plot_path_ex = out_root / f"scatter_distribution_self_preservation_with_extras{suffix}.png"
                    plot_distribution_scatter(df_sp, sp_plot_path_ex, display_model=display_model, model_folder=model_folder, include_extras=True, title_override=f'Self-preservation sentence insertions vs impact\nModel: {model_folder}  Dataset: Whistleblowing')
                    print(f"[save] scatter (self-preservation with extras) -> {sp_plot_path_ex}")
                    # Absolute value version with extras
                    sp_plot_path_ex_abs = out_root / f"scatter_distribution_self_preservation_with_extras{suffix}_abs.png"
                    plot_distribution_scatter(df_sp, sp_plot_path_ex_abs, display_model=display_model, model_folder=model_folder, include_extras=True, title_override=f'Self-preservation sentence insertions vs impact\nModel: {model_folder}  Dataset: Whistleblowing', abs_value=True)
                    print(f"[save] scatter (self-preservation with extras, abs) -> {sp_plot_path_ex_abs}")
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
        # Shared plotting only: drop extremely off-distribution points with mean_logprob < -10
        if 'mean_logprob' in df_all_main.columns:
            try:
                mlp = pd.to_numeric(df_all_main['mean_logprob'], errors='coerce')
                df_all_main = df_all_main[mlp >= -10]
            except Exception:
                pass
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
    
    # Bump font sizes globally for shared plots
    try:
        plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14, 'figure.titlesize': 20})
    except Exception:
        pass
    
    plot_shared_distribution_scatter(df_all_main, out_path_shared, include_extras=True, plot_methods=plot_methods_list)
    
    # Absolute value version
    out_path_shared_abs = shared_root / ('scatter_distribution_shared_random_abs.png' if bool(args.use_random_variant) else 'scatter_distribution_shared_abs.png')
    plot_shared_distribution_scatter(df_all_main, out_path_shared_abs, include_extras=True, plot_methods=plot_methods_list, abs_value=True)
    
    # Read back axis limits from the saved data for consistency
    try:
        df_axis = df_all_main.copy()
        thr = globals().get('min_logprob_threshold', None)
        if thr is not None and 'mean_logprob' in df_axis.columns:
            mlp_axis = pd.to_numeric(df_axis['mean_logprob'], errors='coerce')
            df_axis = df_axis[mlp_axis > float(thr)]
        # Apply overlap filter for axis estimation as well
        try:
            overlap_only = bool(globals().get('overlap_red_blue_only_enabled', False))
        except Exception:
            overlap_only = False
        if overlap_only and all(c in df_axis.columns for c in ['sentence_type','scenario_id','chunk_idx','target_name']):
            df_axis = df_axis.copy()
            df_axis['viz_type'] = df_axis['sentence_type'].apply(lambda st: 'resampled' if st in ('onpolicy_inserted', 'onpolicy_fixed') else st)
            resampled_keys = set(zip(
                df_axis.loc[df_axis['viz_type'] == 'resampled', 'scenario_id'],
                df_axis.loc[df_axis['viz_type'] == 'resampled', 'chunk_idx'],
                df_axis.loc[df_axis['viz_type'] == 'resampled', 'target_name']
            ))
            handwritten_keys = set(zip(
                df_axis.loc[df_axis['viz_type'] == 'offpolicy_handwritten', 'scenario_id'],
                df_axis.loc[df_axis['viz_type'] == 'offpolicy_handwritten', 'chunk_idx'],
                df_axis.loc[df_axis['viz_type'] == 'offpolicy_handwritten', 'target_name']
            ))
            keep_keys = resampled_keys.intersection(handwritten_keys)
            if keep_keys:
                mask_keep = list(zip(df_axis.get('scenario_id'), df_axis.get('chunk_idx'), df_axis.get('target_name')))
                df_axis = df_axis[[k in keep_keys for k in mask_keep]]
        x_min = float(np.nanmin(df_axis['mean_logprob'].astype(float))) if ('mean_logprob' in df_axis.columns and not df_axis.empty) else None
        x_max = float(np.nanmax(df_axis['mean_logprob'].astype(float))) if ('mean_logprob' in df_axis.columns and not df_axis.empty) else None
        y_min = float(np.nanmin(df_axis['delta_whistleblowing_rate'].astype(float))) if ('delta_whistleblowing_rate' in df_axis.columns and not df_axis.empty) else None
        y_max = float(np.nanmax(df_axis['delta_whistleblowing_rate'].astype(float))) if ('delta_whistleblowing_rate' in df_axis.columns and not df_axis.empty) else None
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
    print(f"[save] shared scatter (abs) -> {out_path_shared_abs}")

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
        # Shared plotting only: drop mean_logprob < -10
        if 'mean_logprob' in df_sp.columns:
            try:
                mlp_sp = pd.to_numeric(df_sp['mean_logprob'], errors='coerce')
                df_sp = df_sp[mlp_sp >= -10].copy()
            except Exception:
                pass
        
        out_sp = shared_root / ('scatter_distribution_self_preservation_shared_random.png' if bool(args.use_random_variant) else 'scatter_distribution_self_preservation_shared.png')
        plot_shared_distribution_scatter(
            df_sp,
            out_sp,
            include_extras=True,
            plot_methods=plot_methods_list,
            xlim=None,
            ylim=ylim_shared,
            title_override='Self-preservation sentence insertions vs impact'
        )
        print(f"[save] shared scatter (self-preservation) -> {out_sp}")
        
        # Absolute value version
        out_sp_abs = shared_root / ('scatter_distribution_self_preservation_shared_random_abs.png' if bool(args.use_random_variant) else 'scatter_distribution_self_preservation_shared_abs.png')
        plot_shared_distribution_scatter(
            df_sp,
            out_sp_abs,
            include_extras=True,
            plot_methods=plot_methods_list,
            xlim=None,
            ylim=ylim_shared,
            title_override='Self-preservation sentence insertions vs impact',
            abs_value=True
        )
        print(f"[save] shared scatter (self-preservation, abs) -> {out_sp_abs}")
    except Exception as _e:
        logger.warning(f"shared self-preservation plot failed: {type(_e).__name__}: {_e}")


if __name__ == '__main__':
    # Basic style consistent with other analysis scripts
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12, 'figure.titlesize': 18})
    main()


