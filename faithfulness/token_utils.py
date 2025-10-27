from functools import cache


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


def get_qwen_raw_tokens(text, base_model):
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
        return get_qwen_raw_tokens(text, base_model=model_name == "qwen-14b-base")
    elif "llama" in model_name.lower():
        return get_llama_raw_tokens(text)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_qwen_int_tokens(text, base_model):
    tokenizer = get_qwen_tokenizer(base_model)
    tokens_int = tokenizer.encode(text)
    return tokens_int


def get_llama_int_tokens(text):
    tokenizer = get_llama_tokenizer()
    tokens_int = tokenizer.encode(text)
    return tokens_int


def get_int_tokens(text, model_name):
    if "qwen" in model_name:
        return get_qwen_int_tokens(text, base_model=model_name == "qwen-14b-base")
    elif "llama" in model_name.lower():
        return get_llama_int_tokens(text)
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