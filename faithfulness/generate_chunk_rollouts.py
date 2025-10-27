import hashlib
import os
import json
import asyncio
import time
import httpx
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
if not NOVITA_API_KEY:
    print("NOVITA_API_KEY not found in environment variables")

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    print("TOGETHER_API_KEY not found in environment variables")

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
if not FIREWORKS_API_KEY:
    print("FIREWORKS_API_KEY not found in environment variables")



async def make_api_request(
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    provider: str,
    model: str,
    max_retries: int = 3,
    verbose: bool = True,
) -> Dict:
    """Make an API request to either Novita, Together, or Fireworks based on provider setting."""
    # if max_retries == 6:
    max_retries = 50
    if provider == "Novita":
        headers = {
            "Authorization": f"Bearer {NOVITA_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
            "stream": False,
        }

        api_url = "https://api.novita.ai/v3/openai/completions"

    elif provider == "Together":
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        payload = {
            "model": "deepseek-ai/deepseek-r1-distill-qwen-14b",
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }

        api_url = "https://api.together.xyz/v1/completions"

    elif provider == "Fireworks":
        headers = {
            "Authorization": f"Bearer {FIREWORKS_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "accounts/fireworks/models/deepseek-r1-distill-qwen-14b",
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
            "stream": True,
        }

        api_url = "https://api.fireworks.ai/inference/v1/completions"

    # Implement exponential backoff for retries
    retry_delay = 2

    # print(f"{verbose=}")
    # quit()

    for attempt in range(max_retries):
        # print(f"{attempt=}")
        try:
            # Handle streaming responses for Together and Fireworks
            if (
                provider == "Together" or provider == "Fireworks"
            ) and payload.get("stream", False):
                return await handle_streaming_response(
                    api_url, headers, payload, provider, verbose
                )

            # For non-streaming responses
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    api_url, headers=headers, json=payload, timeout=3600
                )

                if response.status_code == 500:
                    if verbose:
                        print(
                            f"Server error (500) on attempt {attempt+1}/{max_retries}. Retrying..."
                        )
                    delay = retry_delay * (2**attempt)
                    if delay > 61:
                        delay = 61
                    await asyncio.sleep(delay)
                    continue

                elif response.status_code == 429:
                    if verbose:
                        print(
                            f"Rate limit (429) on attempt {attempt+1}/{max_retries}. Retrying..."
                        )
                    delay = retry_delay * (2**attempt)
                    if delay > 61:
                        delay = 61
                    await asyncio.sleep(delay)
                    continue

                elif response.status_code != 200:
                    if verbose:
                        print(
                            f"Error from API: {response.status_code} - {response.text}"
                        )
                    if attempt == max_retries - 1:
                        if verbose:
                            print(f"Saving API error: {response.status_code}")
                        return {
                            "error": f"API error: {response.status_code}",
                            "details": response.text,
                        }
                    delay = retry_delay * (2**attempt)
                    if delay > 61:
                        delay = 61
                    await asyncio.sleep(delay)
                    continue

                result = response.json()
                return {
                    "text": result["choices"][0]["text"],
                    "finish_reason": result["choices"][0].get(
                        "finish_reason", ""
                    ),
                    "usage": result.get("usage", {}),
                }

        except ValueError as e:
            if verbose:
                print(
                    f"Exception during API request (attempt {attempt+1}/{max_retries}): {e}"
                )
            if attempt == max_retries - 1:
                return {"error": f"Request exception: {str(e)}"}
            delay = retry_delay * (2**attempt)
            if delay > 61:
                delay = 61
            await asyncio.sleep(delay)

    return {"error": "All API request attempts failed"}


async def handle_streaming_response(
    api_url: str,
    headers: Dict,
    payload: Dict,
    provider: str,
    verbose: bool = True,
) -> Dict:
    """Handle streaming responses from Together or Fireworks API."""
    try:
        collected_text = ""
        finish_reason = None
        usage = None

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", api_url, headers=headers, json=payload, timeout=600
            ) as response:
                if response.status_code != 200:
                    return {
                        "error": f"API error: {response.status_code}",
                        "details": await response.aread(),
                    }

                async for chunk in response.aiter_lines():
                    if not chunk.strip():
                        continue

                    if chunk == "data: [DONE]":
                        break

                    if chunk.startswith("data: "):
                        try:
                            data = json.loads(chunk[6:])

                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]

                                if "text" in choice and choice["text"]:
                                    collected_text += choice["text"]
                                elif (
                                    "delta" in choice
                                    and "content" in choice["delta"]
                                ):
                                    collected_text += choice["delta"]["content"]

                                if choice.get("finish_reason"):
                                    finish_reason = choice["finish_reason"]

                            if "usage" in data and data["usage"]:
                                usage = data["usage"]

                        except json.JSONDecodeError:
                            if verbose:
                                print(f"Failed to parse chunk: {chunk}")

        # For Together API, handle the <think> token
        if provider == "Together":
            if collected_text.startswith("<think>\n"):
                collected_text = collected_text[len("<think>\n") :]
            elif collected_text.startswith("<think>"):
                collected_text = collected_text[len("<think>") :]

        return {
            "text": collected_text,
            "finish_reason": finish_reason or "stop",
            "usage": usage or {},
        }

    except Exception as e:
        if verbose:
            print(f"Exception during streaming: {e}")
        return {"error": f"Streaming exception: {str(e)}"}


async def generate_multiple_responses(
    prompt: str,
    num_responses: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 16384,
    provider: str = "Novita",
    model: str = "deepseek/deepseek-r1-distill-qwen-14b",
    max_retries: int = 6,
    verbose: bool = False,
    check_all_good: bool = False,
    req_exist: bool = False,
) -> List[Dict]:
    """
    Generate multiple responses for a given prompt using the specified provider.

    Args:
        prompt: The prompt to send to the model
        num_responses: Number of responses to generate
        temperature: Sampling temperature (default: 0.6)
        top_p: Top-p sampling parameter (default: 0.95)
        max_tokens: Maximum tokens to generate (default: 16384)
        provider: API provider to use ("Novita", "Together", or "Fireworks")
        model: Model name to use
        max_retries: Maximum number of retries for failed requests (default: 6)
        verbose: Whether to print progress and status messages (default: True)

    Returns:
        Dictionary containing responses and metadata
    """

    model_str = "-".join(model.split("-")[-2:])
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    prompt_hash = prompt_hash[::2]
    if num_responses != 50 and num_responses != 100:
        nr_str = f"_nr{num_responses}"
    else:
        nr_str = ""
    fp_out = f"response_cache/{model_str}/t{temperature}_p{top_p}_tok{max_tokens}_ret{max_retries}{nr_str}_{prompt_hash}.json"

    Path(fp_out).parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(fp_out):
        if verbose:
            print(f"{fp_out} already exists. Loading from cache...")
        with open(fp_out, "r") as f:
            data = json.load(f)
            if not check_all_good:
                return data
            all_good = [
                "text" in data["responses"][i]
                for i in range(len(data["responses"]))
            ]
            if all(all_good):
                return data
            if verbose:
                print("Bad data! Invalid 'text' key in response[0]")
    else:
        if verbose:
            print(f"{fp_out} does not exist. Generating responses...")
    if req_exist:
        return None

    # Validate provider
    if provider not in ["Novita", "Together", "Fireworks"]:
        raise ValueError("Provider must be one of: Novita, Together, Fireworks")

    # Check API keys
    if provider == "Novita":
        assert (
            NOVITA_API_KEY
        ), "NOVITA_API_KEY not found in environment variables"
    elif provider == "Together":
        assert (
            TOGETHER_API_KEY
        ), "TOGETHER_API_KEY not found in environment variables"
    elif provider == "Fireworks":
        assert (
            FIREWORKS_API_KEY
        ), "FIREWORKS_API_KEY not found in environment variables"

    if verbose:
        print(f"Generating {num_responses} responses using {provider}...")

    # Create tasks for all requests
    t_start = time.time()
    tasks = [
        make_api_request(
            prompt,
            temperature,
            top_p,
            max_tokens,
            provider,
            model,
            max_retries,
            verbose,
        )
        for _ in range(num_responses)
    ]

    # Execute all requests concurrently with progress bar
    responses = []
    if verbose:
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=num_responses,
            desc="Generating responses",
        ):
            response = await task
            responses.append(response)
    else:
        for task in asyncio.as_completed(tasks):
            response = await task
            responses.append(response)

    d = {
        "prompt": prompt,
        "num_responses": num_responses,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "provider": provider,
        "model": model,
        "responses": responses,
    }
    with open(fp_out, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

    if verbose:
        print(f"Saved {len(responses)} responses to {fp_out}")
        print(f"Time taken to save responses: {time.time() - t_start} seconds")

    return d


async def call_generate(
    prompt: str,
    num_responses: int,
    temperature: float,
    top_p: float,
    max_tokens: int = 16384,
    provider: str = "Novita",
    model: str = "deepseek/deepseek-r1-distill-qwen-14b",
    max_retries: int = 6,
    verbose: bool = True,
    req_exist: bool = False,
) -> List[Dict]:
    return await generate_multiple_responses(
        prompt=prompt,
        num_responses=num_responses,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        provider=provider,
        model=model,
        max_retries=max_retries,
        verbose=verbose,
        req_exist=req_exist,
    )


# Example usage
if __name__ == "__main__":

    prompt = "Solve this math problem step by step. You MUST put your final answer in \\boxed{}. Problem: What is 2 + 2? Solution: \n<think>\n"
    asyncio.run(
        call_generate(
            prompt,
            5,
            0.7,
            0.95,
            16384,
            "Novita",
            "deepseek/deepseek-r1-distill-qwen-14b",
            3,
        )
    )
