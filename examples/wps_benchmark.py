# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
﻿# wps_benchmark.py
# Copyright RomanAILabs - Daniel Harding
# Real Words Per Second benchmark + NRL metrics

import time
from pathlib import Path
from llama_cpp import Llama
import subprocess

def count_words(text: str) -> int:
    """Count actual words (not tokens)"""
    return len(text.split())

def get_nrl_metrics():
    try:
        result = subprocess.run(
            ["nrl", "bench", "65536", "64", "1", "8", "omega"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except:
        return "NRL metrics unavailable"

def main():
    model_path = Path(r"C:\Users\Asus\Desktop\Documents\RomaPy Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf")
    
    print("Loading Phi-3 mini...")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=8,
        n_batch=512,
        chat_format="chatml",
        verbose=False
    )
    print("Model loaded.\n")

    prompt = "Hello! Tell me a short interesting fact about space."
    print(f"Prompt: {prompt}")
    print("-" * 70)

    start = time.time()
    text = ""
    tokens = 0

    for chunk in llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7,
        stream=True,
    ):
        if "content" in chunk["choices"][0]["delta"]:
            content = chunk["choices"][0]["delta"]["content"]
            print(content, end="", flush=True)
            text += content
            tokens += 1

    total_time = time.time() - start
    words = count_words(text)
    wps = words / total_time if total_time > 0 else 0
    tps = tokens / total_time if total_time > 0 else 0

    print("\n" + "-" * 70)
    print(f"Generated {words} words ({tokens} tokens) in {total_time:.2f}s")
    print(f"WPS (Words Per Second): {wps:.2f}")
    print(f"TPS (Tokens Per Second): {tps:.2f}")

    print("\nNRL Metrics (Omega mode):")
    print(get_nrl_metrics())

if __name__ == "__main__":
    main()
