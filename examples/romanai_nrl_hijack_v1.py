# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
﻿# romanai_nrl_hijack_v1.py
# Copyright RomanAILabs - Daniel Harding
# RomanAI 4D + NRL Muscle Memory (Phase 1)

import time
import subprocess
from pathlib import Path
from collections import OrderedDict

# Simple muscle memory cache (prompt -> response)
MUSCLE_MEMORY: OrderedDict = OrderedDict()
MAX_CACHE_SIZE = 50

def run_romanai(model_path, prompt, max_tokens=150):
    try:
        romanai_exe = r"C:\Users\Asus\AppData\Roaming\RomanAI-V2\bin\llama-cli.exe"
        cmd = [romanai_exe, "-m", str(model_path), "-c", "2048", "-t", "8",
               "--temp", "0.7", "--top-p", "0.9", "-n", str(max_tokens),
               "-cnv", "--no-display-prompt", "-p", 
               "You are a helpful assistant. Answer in ONE short sentence."]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"

def nrl_muscle_memory_lookup(prompt: str) -> str | None:
    """Check if we've seen this pattern before."""
    return MUSCLE_MEMORY.get(prompt)

def nrl_muscle_memory_store(prompt: str, response: str):
    """Store pattern in muscle memory."""
    if len(MUSCLE_MEMORY) >= MAX_CACHE_SIZE:
        MUSCLE_MEMORY.popitem(last=False)  # Remove oldest
    MUSCLE_MEMORY[prompt] = response

def run_nrl_metrics():
    try:
        result = subprocess.run(["nrl", "bench", "65536", "64", "1", "8", "omega"], 
                              capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except:
        return "NRL metrics unavailable"

def main():
    model_path = Path(r"C:\Users\Asus\Desktop\Documents\RomaPy Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf")
    
    print("Starting RomanAI 4D + NRL Hijack v1...")
    print(f"Model: {model_path.name}")
    print("-" * 70)

    prompt = "Hello! Tell me a short interesting fact about space."
    print(f"Prompt: {prompt}")
    print("-" * 70)

    # Check muscle memory first
    cached = nrl_muscle_memory_lookup(prompt)
    if cached:
        print(f"[MUSCLE MEMORY HIT] {cached}")
        print("-" * 70)
        print("NRL Metrics (Omega mode):")
        print(run_nrl_metrics())
        return

    start = time.time()
    output = run_romanai(model_path, prompt)
    total_time = time.time() - start

    print(output if output else "[No output - check romanai path]")
    print("-" * 70)

    # Store in muscle memory
    nrl_muscle_memory_store(prompt, output)

    print("NRL Metrics (Omega mode):")
    print(run_nrl_metrics())

    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Cache size: {len(MUSCLE_MEMORY)} patterns")

if __name__ == "__main__":
    main()
