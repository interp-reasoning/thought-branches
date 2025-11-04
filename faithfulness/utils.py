import re
from typing import List, Tuple, Optional, Dict
from transformers import AutoTokenizer
import random
from datasets import load_dataset
import torch
from functools import cache
import torch.nn.functional as F
from pkld import pkld


@cache
def get_qwen_tokenizer(base_model):
    from transformers import AutoTokenizer

    if base_model:
        model_name = "Qwen/Qwen2.5-14B"
    else:
        model_name = r"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


@cache
def get_llama_tokenizer():
    from transformers import AutoTokenizer

    model_name = r"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_tokenizer(model_name):
    if "qwen" in model_name:
        return get_qwen_tokenizer(model_name == "qwen-14b-base")
    elif "llama" in model_name.lower():
        return get_llama_tokenizer()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_qwen_raw_tokens_(text, base_model):
    tokenizer = get_qwen_tokenizer(base_model)
    tokens_int = tokenizer.encode(text)
    tokens_words = tokenizer.convert_ids_to_tokens(tokens_int)
    return tokens_words


def get_llama_raw_tokens(text):
    tokenizer = get_llama_tokenizer()
    tokens_int = tokenizer.encode(text)
    tokens_words = tokenizer.convert_ids_to_tokens(tokens_int)
    return tokens_words


def get_raw_tokens(text, model_name):
    if "qwen" in model_name:
        return get_qwen_raw_tokens_(text, base_model=model_name == "qwen-14b-base")
    elif "llama" in model_name.lower():
        return get_llama_raw_tokens(text)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def qwen_tokens_to_clean(tokens, base_model):
    tokenizer = get_qwen_tokenizer(base_model)
    if isinstance(tokens[0], str):
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
    else:
        token_ids = tokens
    # Decode back to text
    clean_text = tokenizer.decode(token_ids)
    return clean_text


def llama_tokens_to_clean(tokens):
    tokenizer = get_llama_tokenizer()
    if isinstance(tokens[0], str):
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
    else:
        token_ids = tokens
    # Decode back to text
    clean_text = tokenizer.decode(token_ids)
    return clean_text


def tokens_to_clean(tokens, model_name):
    if "qwen" in model_name:
        return qwen_tokens_to_clean(tokens, model_name == "qwen-14b-base")
    elif "llama" in model_name.lower():
        return llama_tokens_to_clean(tokens)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_chunk_ranges(
    full_text: str, chunks: List[str], pb_lazy_clean=True, window_size=100
) -> List[Tuple[int, int]]:
    # Get character ranges for each chunk in the full text
    chunk_ranges = []
    current_pos = 0

    for chunk in chunks:
        # Normalize the chunk for comparison (preserve length but standardize whitespace)
        normalized_chunk = re.sub(r"\s+", " ", chunk).strip()
        # normalized_chunk = ""

        # Try to find the chunk in the full text
        chunk_start = -1

        # First try exact match from current position
        exact_match_pos = full_text.find(chunk, current_pos)
        if exact_match_pos != -1:
            chunk_start = exact_match_pos
        else:
            # If exact match fails, try with normalized text
            chunk_words = normalized_chunk.split()

            # Search for the sequence of words, allowing for different whitespace
            for i in range(current_pos, len(full_text) - len(normalized_chunk)):
                # Check if this could be the start of our chunk
                text_window = full_text[i : i + len(normalized_chunk) + 20]  # Add some buffer
                normalized_window = re.sub(r"\s+", " ", text_window).strip()

                if normalized_window.startswith(normalized_chunk):
                    chunk_start = i
                    break

                # If not found with window, try word by word matching
                if (
                    i == current_pos + window_size
                ):  # Limit detailed search to avoid performance issues
                    for j in range(current_pos, len(full_text) - 10):
                        # Try to match first word
                        if re.match(
                            r"\b" + re.escape(chunk_words[0]) + r"\b",
                            full_text[j : j + len(chunk_words[0]) + 5],
                        ):
                            # Check if subsequent words match
                            match_text = full_text[j : j + len(normalized_chunk) + 30]
                            normalized_match = re.sub(r"\s+", " ", match_text).strip()
                            if normalized_match.startswith(normalized_chunk):
                                chunk_start = j
                                break
                    break

        if chunk_start == -1:
            print(f"Warning: Chunk not found in full text: {chunk[:50]}...")
            continue

        # For the end position, find where the content of the chunk ends in the full text
        chunk_content = re.sub(r"\s+", "", chunk)  # Remove all whitespace
        full_text_from_start = full_text[chunk_start:]
        full_text_content = re.sub(
            r"\s+", "", full_text_from_start[: len(chunk) + 50]
        )  # Remove all whitespace

        # Find how many characters of content match
        content_match_len = 0
        for i in range(min(len(chunk_content), len(full_text_content))):
            if chunk_content[i] == full_text_content[i]:
                content_match_len += 1
            else:
                break
        # print(f"Content match length: {content_match_len}")

        # Map content length back to original text with whitespace
        chunk_end = chunk_start
        content_chars_matched = 0
        for i in range(len(full_text_from_start)):
            if chunk_end + i >= len(full_text):
                break
            if not full_text[chunk_start + i].isspace():
                content_chars_matched += 1
            if content_chars_matched > content_match_len:
                break
            chunk_end = chunk_start + i

        chunk_end += 1  # Include the last character
        current_pos = chunk_end

        chunk_ranges.append((chunk_start, chunk_end))

    if pb_lazy_clean:
        chunk_ranges_ = []
        for i in range(len(chunk_ranges)):
            if i < len(chunk_ranges) - 1:
                chunk_range_new = (chunk_ranges[i][0], chunk_ranges[i + 1][0])
                chunk_ranges_.append(chunk_range_new)
            else:
                chunk_ranges_.append((chunk_ranges[i][0], len(full_text)))
        chunk_ranges = chunk_ranges_

    return chunk_ranges


def get_chunk_token_ranges(
    text: str, chunk_ranges: List[Tuple[int, int]], tokenizer: AutoTokenizer
) -> List[Tuple[int, int]]:
    """Convert character positions to token indices"""
    chunk_token_ranges = []

    for chunk_start, chunk_end in chunk_ranges:
        chunk_start_token = tokenizer.encode(text[:chunk_start], add_special_tokens=False)
        chunk_start_token_idx = len(chunk_start_token)
        chunk_end_token = tokenizer.encode(text[:chunk_end], add_special_tokens=False)
        chunk_end_token_idx = len(chunk_end_token)
        chunk_token_ranges.append((chunk_start_token_idx, chunk_end_token_idx))

    return chunk_token_ranges


def extract_boxed_answers(text: str) -> List[str]:
    """
    Extract answers enclosed in \boxed{} from the text with improved handling
    of nested braces and complex LaTeX expressions.

    Args:
        text: The text to extract boxed answers from

    Returns:
        List of extracted boxed answers
    """
    # Find all occurrences of \boxed{
    boxed_starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]

    if not boxed_starts:
        return [""]

    answers = []

    for start_idx in boxed_starts:
        # Start after \boxed{
        idx = start_idx + 7
        brace_count = 1  # We've already opened one brace
        answer = ""

        # Parse until we find the matching closing brace
        while idx < len(text) and brace_count > 0:
            char = text[idx]

            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1

                # Skip the closing brace of \boxed{}
                if brace_count == 0:
                    break

            if brace_count > 0:  # Only add if we're still inside the boxed content
                answer += char

            idx += 1

        if answer:
            answers.append(answer)

    return answers if answers else [""]


def check_answer(answer: str, gt_answer: str) -> bool:
    """
    Check if the generated answer matches the ground truth answer
    after normalizing LaTeX formatting.

    Args:
        answer: The generated answer to check
        gt_answer: The ground truth answer to compare against

    Returns:
        True if the answers match after normalization, False otherwise
    """
    # Normalize both answers
    normalized_answer = normalize_latex(answer)
    normalized_gt_answer = normalize_latex(gt_answer)

    # First check if normalized strings match
    if normalized_answer == normalized_gt_answer:
        return True

    # If string comparison fails, try mathematical equivalence
    try:
        return get_latex_equivalent(answer, gt_answer)
    except Exception as e:
        # If SymPy parsing fails, fall back to string comparison result
        return False


def get_latex_equivalent(answer0, answer1):
    """
    Check if two LaTeX expressions are mathematically equivalent using SymPy.

    Args:
        answer0: First LaTeX expression
        answer1: Second LaTeX expression

    Returns:
        True if expressions are mathematically equivalent, False otherwise
    """
    try:
        from sympy.parsing.latex import parse_latex
        import sympy

        # Clean up the LaTeX expressions for parsing
        answer0 = prepare_latex_for_sympy(answer0)
        answer1 = prepare_latex_for_sympy(answer1)

        # Parse the LaTeX expressions
        expr1 = parse_latex(answer0)
        expr2 = parse_latex(answer1)

        # Check if they are mathematically identical
        equals = expr1.equals(expr2)
        # print(f"First: {answer0}, Second: {answer1}: equals={equals}")
        return equals
    except Exception as e:
        # print(f"Error comparing expressions: {e}")
        return False


def prepare_latex_for_sympy(latex_str):
    """
    Prepare a LaTeX string for SymPy parsing by removing unsupported commands
    and simplifying the expression.
    """
    if not isinstance(latex_str, str):
        return str(latex_str)

    # Remove \boxed{} command
    latex_str = re.sub(r"\\boxed\{(.*?)\}", r"\1", latex_str)

    # Replace common LaTeX commands that SymPy doesn't support
    replacements = {
        r"\\dfrac": r"\\frac",
        r"\\tfrac": r"\\frac",
        r"\\cdot": r"*",
        r"\\times": r"*",
        r"\\div": r"/",
        r"\\left": r"",
        r"\\right": r"",
        r"\\textbf": r"",
        r"\\text": r"",
        r"\\mathrm": r"",
        r"\\!": r"",
        r",": r"",
    }

    for old, new in replacements.items():
        latex_str = re.sub(old, new, latex_str)

    return latex_str


def normalize_latex(latex_str: str) -> str:
    """
    Normalize LaTeX string by applying various transformations.

    Args:
        latex_str: The LaTeX string to normalize

    Returns:
        Normalized LaTeX string
    """
    normalized = latex_str.strip().lower()

    # Replace different fraction notations
    normalized = normalized.replace("dfrac", "frac")
    normalized = normalized.replace("tfrac", "frac")

    # Normalize spaces
    normalized = re.sub(r"\s+", "", normalized)

    # Normalize percentages
    normalized = normalized.replace("\\%", "")

    # Normalize funny commas
    normalized = normalized.replace("{,}", "")

    # Normalize common mathematical notations
    normalized = normalized.replace("\\times", "*")
    normalized = normalized.replace("\\cdot", "*")

    # Normalize decimal representation
    normalized = re.sub(r"(\d+)[\.,](\d+)", r"\1.\2", normalized)

    # Remove unnecessary braces in simple expressions
    normalized = re.sub(r"{([^{}]+)}", r"\1", normalized)

    # Normalize common constants
    normalized = normalized.replace("\\pi", "pi")

    # Remove LaTeX text commands
    normalized = re.sub(r"\\text\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", normalized)

    # Normalize date formats (e.g., "October 30" vs "October\\ 30")
    normalized = re.sub(r"([a-z]+)\\+\s*(\d+)", r"\1\2", normalized)
    normalized = normalized.replace("\\text", "")

    return normalized


@pkld
def split_solution_into_chunks(solution_text: str, drop_think=True) -> List[str]:
    """
    Split a solution into chunks for rollout generation.

    Args:
        solution_text: The full solution text

    Returns:
        List of chunks
    """

    # if drop_think:
    # First, remove the prompt part if present
    if "<think>" in solution_text:
        solution_text = solution_text.split("<think>")[1].strip()

    # Remove the closing tag if present
    if "</think>" in solution_text:
        solution_text = solution_text.split("</think>")[0].strip()

    # Define patterns for chunk boundaries
    sentence_ending_tokens = [".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]

    # Split the text into chunks
    chunks = []
    current_chunk = ""

    # Process the text character by character
    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]

        # Check for paragraph endings
        is_paragraph_end = False
        for pattern in paragraph_ending_patterns:
            if (
                i + len(pattern) <= len(solution_text)
                and solution_text[i : i + len(pattern)] == pattern
            ):
                is_paragraph_end = True
                break

        # Check for sentence endings followed by space or newline
        is_sentence_end = False
        if i < len(solution_text) - 1 and solution_text[i] in sentence_ending_tokens:
            next_char = solution_text[i + 1]
            if next_char == " " or next_char == "\n":
                is_sentence_end = True

        # If we found a boundary, add the chunk and reset
        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

        i += 1

    # Add the last chunk if not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Merge small chunks (less than 10 characters)
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 10:
            # If this is the last chunk, merge with previous chunk if possible
            if i == len(chunks) - 1:
                if i > 0:
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    chunks.pop(i)
            # Otherwise merge with the next chunk
            else:
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)
                # Don't increment i since we need to check the new merged chunk
            # If we're at the beginning and there's only one chunk, just keep it
            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1

    return chunks


def load_math_problems(
    problem_type: Optional[str] = None,
    level: Optional[str] = None,
    num_problems: Optional[int] = None,
    split: str = "train",
    include_problems: Optional[List[int]] = None,
) -> List[Tuple[int, Dict]]:
    """
    Load problems from the MATH dataset with optional filtering.

    Args:
        problem_type: Type of problems to filter by (if None, use all types)
        level: Level of problems to filter by (if None, use all levels)
        num_problems: Number of problems to sample (if None, use all problems)
        split: Dataset split to use ('train' or 'test')

    Returns:
        List of problems with their original indices
    """
    try:
        # Load from Hugging Face dataset
        math_dataset = load_dataset("fdyrd/math")
        dataset_split = math_dataset[split]

        # Add original indices to problems
        indexed_problems = [
            (
                i,
                {
                    "problem": item["problem"],
                    "level": item["level"],
                    "type": item["type"],
                    "gt_solution": item["solution"],
                },
            )
            for i, item in enumerate(dataset_split)
        ]

        # Extract ground truth answers
        for i, problem in indexed_problems:
            gt_boxed_answers = extract_boxed_answers(problem["gt_solution"])
            gt_answer = gt_boxed_answers[0] if gt_boxed_answers else ""
            problem["gt_answer"] = gt_answer

        # Filter by type if specified
        if problem_type is not None:
            indexed_problems = [
                (i, problem)
                for i, problem in indexed_problems
                if problem.get("type") == problem_type
            ]

        # Filter by level if specified
        if level is not None:
            indexed_problems = [
                (i, problem) for i, problem in indexed_problems if problem.get("level") == level
            ]

        # Sample if needed
        if (
            num_problems is not None
            and include_problems is None
            and num_problems < len(indexed_problems)
        ):
            indexed_problems = random.sample(indexed_problems, num_problems)

        if level:
            print(f"Filtered to level: {level}")
        if problem_type:
            print(f"Filtered to type: {problem_type}")

        return indexed_problems
    except Exception as e:
        print(f"Error loading problems: {e}")
        return []


def print_gpu_memory_summary(label="Memory Usage"):
    """Print a concise summary of GPU memory usage"""
    if not torch.cuda.is_available():
        print(f"{label}: CUDA not available")
        return

    # Get statistics
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB

    # Print concise summary
    print(f"=== {label} ===")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Max Used:  {max_allocated:.2f} GB")
    print(f"  Available: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")


@cache
def get_qwen_14b_tokenizer():
    from transformers import AutoTokenizer

    model_name = r"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_qwen_14b_tokens_lower(text, lower=True):
    tokenizer = get_qwen_14b_tokenizer()
    tokens = tokenizer.encode(text)
    raw_words = tokenizer.convert_ids_to_tokens(tokens)
    # Clean the tokens
    cleaned_words = []
    for word in raw_words:
        if word.startswith("Ġ"):
            cleaned_word = word[1:]  # Remove the leading 'Ġ'
        # Add more cleaning rules if needed for other special tokens like 'Ċ' or others
        # elif word == 'Ċ':
        #     cleaned_word = '\\n' # Represent newline explicitly if desired
        else:
            cleaned_word = word
        # Handle potential empty strings after cleaning (if 'Ġ' was the whole token)
        if lower:
            cleaned_word = cleaned_word.lower()
        if cleaned_word:
            cleaned_words.append(cleaned_word)
        elif word == " ":  # Keep actual spaces if they are tokenized separately
            cleaned_words.append(" ")

    return cleaned_words


def get_top_p_logits(logits_tensor, p):
    """
    Returns indices and logits for the top-p nucleus.
    Assumes logits_tensor is a 1D tensor.
    """
    probs = F.softmax(logits_tensor.float(), dim=-1)  # Use float32 for softmax stability
    probs_sorted, indices_sorted = torch.sort(probs, descending=True)
    probs_sum_cumulative = torch.cumsum(probs_sorted, dim=-1)

    # Find the index + 1 of the smallest set >= p
    # Using right=True finds the first index strictly > p
    # We add 1 to include that index itself.
    nucleus_index_plus_1 = torch.searchsorted(probs_sum_cumulative, p, right=False) + 1

    # Ensure at least one token is selected
    nucleus_index_plus_1 = max(1, nucleus_index_plus_1)

    indices_nucleus = indices_sorted[:nucleus_index_plus_1]
    logits_nucleus = logits_tensor[indices_nucleus]  # Get original logits

    return indices_nucleus, logits_nucleus


if __name__ == "__main__":
    token_test = "Get raw tokens! 1 Hello"
    # tokenizer = get_qwen_14b_tokenizer()
    tokenizer = get_llama_tokenizer()
    tokens = tokenizer.encode(token_test)
    raw_words = tokenizer.convert_ids_to_tokens(tokens)
    print(raw_words)
