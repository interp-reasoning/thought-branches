#!/usr/bin/env python3
"""
Realistic test for Nebius API logprob extraction using actual whistleblower rollout data.
Tests the character offset method from analyze_onpolicy_distribution.py on real examples.
"""

import os
import json
import httpx
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# Model folder to Nebius API model ID mapping (from analyze_onpolicy_distribution.py)
MODEL_FOLDER_TO_NEBIUS = {
    'llama-3_1-nemotron-ultra-253b-v1': 'nvidia/Llama-3_1-Nemotron-Ultra-253B-v1',
    'hermes-4-70b': 'NousResearch/Hermes-4-70B',
    'deepseek-r1-0528': 'deepseek-ai/DeepSeek-R1-0528',
}


def load_chunks_and_separators(problem_dir: Path) -> Tuple[List[str], List[str]]:
    """Load chunks from chunks.json or chunks_labeled.json."""
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
    """Build prefix from chunks up to end_index."""
    if not chunks or end_index <= 0:
        return ""
    parts: List[str] = []
    for i in range(end_index):
        parts.append(chunks[i])
        if i < len(seps):
            parts.append(seps[i])
    return "".join(parts)


def build_prefill(system_prompt: str, user_prompt: str, email_content: str, prefix_text: str) -> str:
    """Build the prefill text (matches analyze_onpolicy_distribution.py)."""
    sys = system_prompt or ""
    return f"{sys}\n\nAssistant:\n<think>\n{prefix_text}"


def nebius_get_logprobs(
    prefix: str,
    sentence: str,
    model_folder: str,
    api_key: Optional[str] = None,
    print_debug: bool = False,
) -> Dict[str, Any]:
    """
    Get logprobs for sentence given prefix using character offset method.
    Uses a token-matching approach for robustness.
    """
    if not api_key:
        api_key = os.getenv('NEBIUS_API_KEY')
        if not api_key:
            raise ValueError("NEBIUS_API_KEY not set")
    
    # Resolve model_folder to Nebius model ID
    model_id = MODEL_FOLDER_TO_NEBIUS.get(model_folder, MODEL_FOLDER_TO_NEBIUS['llama-3_1-nemotron-ultra-253b-v1'])
    
    full_text = prefix + sentence
    
    # Call API with echo=True to get prompt logprobs
    payload = {
        "model": model_id,
        "prompt": full_text,
        "temperature": 0.0,
        "max_tokens": 0,
        "logprobs": True,
        "echo": True,
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    try:
        with httpx.Client(timeout=180) as client:
            resp = client.post(
                "https://api.studio.nebius.com/v1/completions",
                headers=headers,
                json=payload
            )
            if resp.status_code != 200:
                raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
            result = resp.json()
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}")
    
    # Extract logprobs data
    logprobs_data = result['choices'][0].get('logprobs', {})
    top_logprobs = logprobs_data.get('top_logprobs', [])
    tokens = logprobs_data.get('tokens', [])
    
    if not top_logprobs or not tokens:
        raise RuntimeError("No top_logprobs or tokens returned")
    
    # Sliding window approach: try each token as a potential start position
    # Build normalized token representations and find exact match
    
    def normalize_token(tok: str) -> str:
        """Replace BPE markers with actual characters: Ġ/\u0120 → space, Ċ/\u010A → newline.
        Also normalize fancy quotes to straight quotes."""
        return (tok.replace('Ġ', ' ').replace('\u0120', ' ')
                .replace('Ċ', '\n').replace('\u010A', '\n')
                .replace('\u2019', "'")
                .replace('\u00e2\u0122\u013b', "'")
                .replace('’', "'").replace('’', "'")  # Fancy single quotes
                .replace('“', '"').replace('”', '"'))  # Fancy double quotes
    
    # Prepare token list with logprobs
    token_list = []
    for i, (token, top_lp_dict) in enumerate(zip(tokens, top_logprobs)):
        if top_lp_dict is None:
            # BOS token - skip
            continue
        
        # Get token and logprob (first key-value in dict)
        if isinstance(top_lp_dict, dict) and len(top_lp_dict) > 0:
            actual_token = list(top_lp_dict.keys())[0]
            actual_logprob = float(list(top_lp_dict.values())[0])
            token_list.append((i, actual_token, actual_logprob))
    
    # Normalize sentence for comparison
    normalized_sentence = normalize_token(sentence)
    
    # Sliding window: try each position as start
    # Handle case where sentence might start in the middle of a token (e.g., ".ĊĊActually" contains "\n\nActually")
    start_idx = None
    end_idx = None
    sentence_tokens_list = []
    sentence_logprobs_list = []
    
    for start_pos in range(len(token_list)):
        # Try building from this position
        accumulated = ""
        temp_tokens = []
        temp_logprobs = []
        
        for offset in range(len(token_list) - start_pos):
            i, tok, lp = token_list[start_pos + offset]
            normalized_tok = normalize_token(tok)
            accumulated += normalized_tok
            temp_tokens.append(tok)
            temp_logprobs.append(lp)
            
            # Check if accumulated contains our sentence
            # This handles cases where sentence starts mid-token (e.g., ".ĊĊ" + "Actually" for "\n\nActually")
            if normalized_sentence in accumulated:
                # Find where in accumulated the sentence starts
                sentence_start_in_acc = accumulated.find(normalized_sentence)
                
                # If sentence doesn't start at position 0, we need to trim the first token(s)
                if sentence_start_in_acc > 0:
                    # Rebuild to find exact token boundaries
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
                    # Sentence starts at position 0 - exact match
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
    
    if start_idx is None or not sentence_logprobs_list:
        raise RuntimeError(f"Could not match sentence.\nLooking for: {repr(normalized_sentence[:80])}\nAlso tried stripped: {repr(normalized_sentence.strip()[:80])}")
    
    return {
        'tokens': sentence_tokens_list,
        'logprobs': sentence_logprobs_list,
        'mean': sum(sentence_logprobs_list) / len(sentence_logprobs_list) if sentence_logprobs_list else float('nan'),
        'sum': sum(sentence_logprobs_list) if sentence_logprobs_list else float('nan'),
        'count': len(sentence_logprobs_list),
        'total_tokens': len(sentence_tokens_list),
        # Metadata for verification
        'prefix_end_char': len(prefix),
        'target_end_char': len(prefix) + len(sentence),
        'start_idx': start_idx,
        'end_idx': end_idx,
    }


def load_onpolicy_sentence(
    model_folder: str,
    scenario_id: int,
    chunk_idx: int,
    target_name: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Optional[Tuple[str, str]]:
    """
    Load the actual onpolicy sentence and prefix for a given example.
    Returns (prefix, sentence) or None if not found.
    """
    # Build paths
    base = Path('whistleblower_rollouts') / model_folder / f"temperature_{temperature}_top_p_{top_p}"
    baseline = base / 'yes_base_solution'
    onpolicy_dir = base / 'yes_base_solution_onpolicy_chain_disruption'
    
    # Load scenario metadata
    scenario_dir = baseline / f'scenario_{scenario_id}'
    if not scenario_dir.exists():
        scenario_dir = baseline / f'problem_{scenario_id}'
    if not scenario_dir.exists():
        return None
    
    scenario_json = scenario_dir / 'scenario.json'
    if not scenario_json.exists():
        return None
    
    with open(scenario_json, 'r') as f:
        scenario = json.load(f)
    
    system_prompt = scenario.get('system_prompt', '')
    user_prompt = scenario.get('user_prompt', '')
    email_content = scenario.get('email_content', '')
    
    # Load chunks
    chunks, seps = load_chunks_and_separators(scenario_dir)
    if not chunks or chunk_idx >= len(chunks):
        return None
    
    # Load onpolicy experiment file to get the resampled chunk and intervention sentence
    onpolicy_scenario_dir = onpolicy_dir / f'scenario_{scenario_id}'
    if not onpolicy_scenario_dir.exists():
        onpolicy_scenario_dir = onpolicy_dir / f'problem_{scenario_id}'
    if not onpolicy_scenario_dir.exists():
        return None
    
    # Find the matching experiment file
    for exp_file in onpolicy_scenario_dir.glob('onpolicy_chain_disruption_*.json'):
        with open(exp_file, 'r') as f:
            data = json.load(f)
        
        if str(data.get('target_name', '')) != str(target_name):
            continue
        
        # Find the experiment for this chunk
        for exp in data.get('experiments', []):
            if int(exp.get('chunk_idx', -1)) == chunk_idx and exp.get('success'):
                # Get the resampled chunk (resampled version of chunks[chunk_idx])
                resampled_chunk = exp.get('selected_resampled_chunk', '')
                # Get the intervention sentence that follows the resampled chunk
                intervention_sentence = exp.get('selected_continuation_full_text', '')
                
                if not resampled_chunk or not intervention_sentence:
                    continue
                
                # Build prefix: chunks[0:chunk_idx] + resampled_chunk
                # First get all chunks BEFORE chunk_idx
                prefix_text = make_prefix_text(chunks, seps, chunk_idx)
                # Add the resampled chunk
                prefix_text = prefix_text + resampled_chunk
                # Build full prefill
                prefill = build_prefill(system_prompt, user_prompt, email_content, prefix_text)
                
                return (prefill, intervention_sentence)
    
    return None


def load_offpolicy_sentence(
    model_folder: str,
    scenario_id: int,
    chunk_idx: int,
    strategy: str,
    offpolicy_type: str = 'handwritten',
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Optional[Tuple[str, str]]:
    """
    Load the actual offpolicy sentence and prefix for a given example.
    offpolicy_type: 'handwritten', 'cross_model', or 'same_model'
    Returns (prefix, disruption_text_with_separator) or None if not found.
    """
    # Build paths
    base = Path('whistleblower_rollouts') / model_folder / f"temperature_{temperature}_top_p_{top_p}"
    baseline = base / 'yes_base_solution'
    
    if offpolicy_type == 'cross_model':
        offpolicy_dir = base / 'yes_base_solution_chain_disruption_cross_model'
        results_file = 'chain_disruption_cross_model_results.json'
    elif offpolicy_type == 'same_model':
        offpolicy_dir = base / 'yes_base_solution_chain_disruption_same_model'
        results_file = 'chain_disruption_same_model_results.json'
    else:  # handwritten
        offpolicy_dir = base / 'yes_base_solution_chain_disruption'
        results_file = 'chain_disruption_results.json'
    
    # Load scenario metadata
    scenario_dir = baseline / f'scenario_{scenario_id}'
    if not scenario_dir.exists():
        scenario_dir = baseline / f'problem_{scenario_id}'
    if not scenario_dir.exists():
        return None
    
    scenario_json = scenario_dir / 'scenario.json'
    if not scenario_json.exists():
        return None
    
    with open(scenario_json, 'r') as f:
        scenario = json.load(f)
    
    system_prompt = scenario.get('system_prompt', '')
    user_prompt = scenario.get('user_prompt', '')
    email_content = scenario.get('email_content', '')
    
    # Load chunks
    chunks, seps = load_chunks_and_separators(scenario_dir)
    if not chunks or chunk_idx >= len(chunks):
        return None
    
    # Load offpolicy results file
    offpolicy_scenario_dir = offpolicy_dir / f'scenario_{scenario_id}'
    if not offpolicy_scenario_dir.exists():
        offpolicy_scenario_dir = offpolicy_dir / f'problem_{scenario_id}'
    if not offpolicy_scenario_dir.exists():
        return None
    
    results_path = offpolicy_scenario_dir / results_file
    if not results_path.exists():
        return None
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Find the experiment for this chunk
    for exp in data.get('experiments', []):
        if int(exp.get('chunk_idx', -1)) != chunk_idx:
            continue
        
        # Find the strategy
        for strat in exp.get('disruption_results', []):
            if str(strat.get('strategy', '')).lower() != strategy.lower():
                continue
            
            # Get disruption text (cross/same may use 'chosen_text')
            if offpolicy_type in ['cross_model', 'same_model']:
                disruption_text = strat.get('chosen_text') or strat.get('disruption_text', '')
            else:
                disruption_text = strat.get('disruption_text', '')
            
            if not disruption_text:
                continue
            
            # Build prefix: chunks[0:chunk_idx] (NOT including chunk_idx)
            prefix_text = make_prefix_text(chunks, seps, chunk_idx)
            # Add chunk at chunk_idx
            prefix_text = prefix_text + chunks[chunk_idx]
            # Add separator (usually a space) - this will be included in the sentence we score
            separator = seps[chunk_idx] if chunk_idx < len(seps) else ''
            prefill = build_prefill(system_prompt, user_prompt, email_content, prefix_text)
            
            # Prepend separator to disruption_text so we score it with leading whitespace
            disruption_with_sep = separator + disruption_text
            
            return (prefill, disruption_with_sep)
    
    return None


def test_real_examples():
    """Test logprob extraction on actual whistleblower rollout data."""
    start_time = time.time()
    
    print("\n" + "="*80)
    print("REALISTIC LOGPROB EXTRACTION TESTS")
    print("Using actual whistleblower rollout data")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Track overall results
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    # Load examples from CSVs for each model
    models = ['llama-3_1-nemotron-ultra-253b-v1', 'hermes-4-70b', 'deepseek-r1-0528']
    
    for model_idx, model_folder in enumerate(models, 1):
        print(f"\n[{model_idx}/{len(models)}] Processing model: {model_folder}")
        
        csv_path = Path(f'analysis/onpolicy_distribution/{model_folder}/temperature_0.7_top_p_0.95/distribution_points.csv')
        
        if not csv_path.exists():
            print(f"⚠ Skipping {model_folder}: CSV not found at {csv_path}")
            continue
        
        print(f"  → Loading CSV from {csv_path}")
        
        print(f"\n{'='*80}")
        print(f"MODEL: {model_folder}")
        print(f"{'='*80}\n")
        
        # Load CSV and find examples to test
        print(f"  → Reading CSV...")
        df = pd.read_csv(csv_path)
        print(f"  → Total rows in CSV: {len(df)}")
        
        # Test 1 onpolicy (lowest logprob) + 3 offpolicy (highest logprob each)
        test_examples = []
        
        # 1. Onpolicy: lowest mean_logprob
        onpolicy = df[df['sentence_type'] == 'onpolicy_inserted'].copy()
        if not onpolicy.empty:
            onpolicy_sorted = onpolicy.sort_values('mean_logprob')
            lowest_onpolicy = onpolicy_sorted.iloc[0] if len(onpolicy_sorted) > 0 else None
            if lowest_onpolicy is not None:
                test_examples.append(('onpolicy', lowest_onpolicy))
        
        # 2. Offpolicy handwritten: highest mean_logprob
        offpolicy_hand = df[df['sentence_type'] == 'offpolicy_handwritten'].copy()
        if not offpolicy_hand.empty:
            offpolicy_hand_sorted = offpolicy_hand.sort_values('mean_logprob', ascending=False)
            highest_hand = offpolicy_hand_sorted.iloc[0] if len(offpolicy_hand_sorted) > 0 else None
            if highest_hand is not None:
                test_examples.append(('offpolicy_handwritten', highest_hand))
        
        # 3. Offpolicy same_model: highest mean_logprob
        offpolicy_same = df[df['sentence_type'] == 'offpolicy_same_model'].copy()
        if not offpolicy_same.empty:
            offpolicy_same_sorted = offpolicy_same.sort_values('mean_logprob', ascending=False)
            highest_same = offpolicy_same_sorted.iloc[0] if len(offpolicy_same_sorted) > 0 else None
            if highest_same is not None:
                test_examples.append(('offpolicy_same_model', highest_same))
        
        # 4. Offpolicy cross_model: highest mean_logprob
        offpolicy_cross = df[df['sentence_type'] == 'offpolicy_cross_model'].copy()
        if not offpolicy_cross.empty:
            offpolicy_cross_sorted = offpolicy_cross.sort_values('mean_logprob', ascending=False)
            highest_cross = offpolicy_cross_sorted.iloc[0] if len(offpolicy_cross_sorted) > 0 else None
            if highest_cross is not None:
                test_examples.append(('offpolicy_cross_model', highest_cross))
        
        print(f"  → Selected {len(test_examples)} examples to test (1 onpolicy lowest, up to 3 offpolicy highest)\n")
        
        if not test_examples:
            print(f"⚠ No examples found for {model_folder}")
            continue
        
        for ex_idx, (example_type, row) in enumerate(test_examples, 1):
            scenario_id = int(row['scenario_id'])
            chunk_idx = int(row['chunk_idx'])
            csv_mean_logprob = float(row['mean_logprob'])
            csv_sum_logprob = float(row['sum_logprob'])
            csv_token_count = int(row['scored_token_count'])
            
            # Get type-specific fields
            if example_type == 'onpolicy':
                target_name = str(row['target_name'])
                strategy = None
            else:
                target_name = None
                strategy = str(row['strategy'])
            
            print(f"\n{'-'*80}")
            print(f"[{ex_idx}/{len(test_examples)}] Testing {example_type} example:")
            if example_type == 'onpolicy':
                print(f"  scenario={scenario_id}, chunk={chunk_idx}, target={target_name}")
            else:
                print(f"  scenario={scenario_id}, chunk={chunk_idx}, strategy={strategy}")
            print(f"  CSV values: mean={csv_mean_logprob:.4f}, sum={csv_sum_logprob:.4f}, tokens={csv_token_count}")
            print(f"{'-'*80}")
            
            # Load the actual data
            print(f"\n  → Loading rollout data from JSON files...")
            if example_type == 'onpolicy':
                result = load_onpolicy_sentence(model_folder, scenario_id, chunk_idx, target_name)
            elif example_type == 'offpolicy_handwritten':
                result = load_offpolicy_sentence(model_folder, scenario_id, chunk_idx, strategy, 'handwritten')
            elif example_type == 'offpolicy_same_model':
                result = load_offpolicy_sentence(model_folder, scenario_id, chunk_idx, strategy, 'same_model')
            elif example_type == 'offpolicy_cross_model':
                result = load_offpolicy_sentence(model_folder, scenario_id, chunk_idx, strategy, 'cross_model')
            else:
                result = None
            
            if result is None:
                print(f"  ✗ Could not load data from rollout files")
                continue
            
            print(f"  ✓ Successfully loaded rollout data")
            
            prefix, sentence = result
            
            print(f"\n  → Prefix length: {len(prefix)} chars")
            print(f"  → Sentence: '{sentence[:80]}{'...' if len(sentence) > 80 else ''}'")
            print(f"  → Sentence length: {len(sentence)} chars")
            
            # Get logprobs via API
            print(f"\n  → Calling Nebius API to extract logprobs...")
            total_tests += 1
            try:
                api_result = nebius_get_logprobs(prefix, sentence, model_folder)
                print(f"  ✓ API call successful")
                
                print(f"\n  API Extraction Results:")
                print(f"    - Prefix ends at char {api_result['prefix_end_char']}")
                print(f"    - Target ends at char {api_result['target_end_char']}")
                print(f"    - Token range: [{api_result['start_idx']}, {api_result['end_idx']})")
                print(f"    - Tokens extracted: {len(api_result['tokens'])}")
                print(f"    - First 3 tokens: {api_result['tokens'][:3]}")
                print(f"    - Last 3 tokens: {api_result['tokens'][-3:]}")
                print(f"    - Scored tokens: {api_result['count']}")
                print(f"    - Mean logprob: {api_result['mean']:.4f}")
                print(f"    - Sum logprob: {api_result['sum']:.4f}")
                
                # Show per-token logprobs
                print(f"\n  Per-token logprobs:")
                for i, (tok, lp) in enumerate(zip(api_result['tokens'], api_result['logprobs'])):
                    print(f"    [{i:2d}] {repr(tok):20s} → {lp:7.4f}")
                if api_result['count'] > 0:
                    print(f"    {'':24s}   {'='*7}")
                    print(f"    {'Sum:':24s} {api_result['sum']:7.4f}")
                    print(f"    {'Mean:':24s} {api_result['mean']:7.4f}")
                
                # Compare to CSV
                print(f"\n  Comparison to CSV:")
                print(f"    CSV mean:     {csv_mean_logprob:.4f}")
                print(f"    API mean:     {api_result['mean']:.4f}")
                print(f"    Difference:   {abs(csv_mean_logprob - api_result['mean']):.4f}")
                print(f"    CSV tokens:   {csv_token_count}")
                print(f"    API tokens:   {api_result['count']}")
                
                # Check if close enough (allow small numerical differences)
                mean_close = abs(csv_mean_logprob - api_result['mean']) < 0.1
                token_match = csv_token_count == api_result['count']
                
                print(f"\n  Validation:")
                if mean_close and token_match:
                    print(f"    ✓ PASSED: Values match within tolerance")
                    passed_tests += 1
                else:
                    print(f"    ✗ WARNING: Values differ significantly")
                    failed_tests += 1
                    if not token_match:
                        print(f"      • Token count mismatch: CSV={csv_token_count}, API={api_result['count']}")
                    if not mean_close:
                        print(f"      • Mean logprob difference too large: {abs(csv_mean_logprob - api_result['mean']):.4f}")
                    
            except Exception as e:
                print(f"  ✗ API call failed: {type(e).__name__}: {e}")
                failed_tests += 1
                import traceback
                print("\n  Stack trace:")
                traceback.print_exc()
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE - SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal tests run:     {total_tests}")
    print(f"Passed:              {passed_tests} ({100*passed_tests/total_tests if total_tests > 0 else 0:.1f}%)")
    print(f"Failed/Mismatched:   {failed_tests} ({100*failed_tests/total_tests if total_tests > 0 else 0:.1f}%)")
    print(f"Elapsed time:        {elapsed_time:.2f} seconds")
    
    if passed_tests == total_tests:
        print(f"\n✓ All tests passed! Logprob extraction is working correctly.")
    elif failed_tests > 0:
        print(f"\n⚠ Some tests failed. Review the output above for details.")
    
    print(f"{'='*80}\n")


def main():
    """Run all tests."""
    try:
        test_real_examples()
        return 0
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"TEST FAILED")
        print(f"{'='*80}")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
