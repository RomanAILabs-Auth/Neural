# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
﻿# llama_nrl_v01.py
# Llama.Nrl v0.1 — Working GGUF runner with NRL integration

import time
import subprocess
from pathlib import Path
from collections import OrderedDict

MUSCLE_MEMORY: OrderedDict = OrderedDict()
MAX_CACHE = 50

def run_inference(prompt: str, max_tokens: int = 150) -> str:
    """Try RomanAI 4D first, fallback to llama-cpp-python"""
    try:
        romanai_exe = r"C:\Users\Asus\AppData\Roaming\RomanAI-V2\bin\llama-cli.exe"
        cmd = [romanai_exe, "-m", r"C:\Users\Asus\Desktop\Documents\RomaPy Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf",
               "-c", "2048", "-t", "8", "--temp", "0.7", "--top-p", "0.9",
               "-n", str(max_tokens), "-cnv", "--no-display-prompt", "-p",
               "You are a helpful assistant. Answer in ONE short sentence."]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.stdout.strip():
            return result.stdout.strip()
    except:
        pass

    # Fallback
    from llama_cpp import Llama
    llm = Llama(model_path=r"C:\Users\Asus\Desktop\Documents\RomaPy Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf",
                n_ctx=2048, n_threads=8, n_batch=512, chat_format="chatml", verbose=False)
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.7
    )
    return output['choices'][0]['message']['content']

def nrl_lookup(prompt: str):
    return MUSCLE_MEMORY.get(prompt)

def nrl_store(prompt: str, response: str):
    if len(MUSCLE_MEMORY) >= MAX_CACHE:
        MUSCLE_MEMORY.popitem(last=False)
    MUSCLE_MEMORY[prompt] = response

def get_nrl_metrics():
    try:
        result = subprocess.run(["nrl", "bench", "65536", "64", "1", "8", "omega"],
                              capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except:
        return "NRL metrics unavailable"

def main():
    print("\n" + "=" * 72)
    print(" LLAMA.NRL v0.1 — RomanAI + NRL Integration")
    print("=" * 72)

    prompt = "Hello! Tell me a short interesting fact about space."
    print(f"\nPrompt: {prompt}")
    print("-" * 70)

    # Muscle memory check
    cached = nrl_lookup(prompt)
    if cached:
        print(f"[MUSCLE MEMORY HIT] {cached}")
    else:
        start = time.time()
        output = run_inference(prompt)
        print(output)
        nrl_store(prompt, output)
        print(f"\nTime: {time.time() - start:.2f}s")

    print("-" * 70)
    print("NRL Metrics (Omega mode):")
    print(get_nrl_metrics())
    print(f"Cache size: {len(MUSCLE_MEMORY)} patterns")
    print("=" * 72)

if __name__ == "__main__":
    main()
