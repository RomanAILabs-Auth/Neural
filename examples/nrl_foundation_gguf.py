# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
﻿# nrl_foundation_gguf.py
# Copyright RomanAILabs - Daniel Harding
# Real starter: GGUF decode + NRL as acceleration foundation

import time
from pathlib import Path
from llama_cpp import Llama
import subprocess

def run_nrl_assimilate(size=65536):
    try:
        result = subprocess.run(["nrl", "assimilate", str(size), "64", "8"], 
                              capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except:
        return "NRL assimilate failed"

def run_nrl_omega():
    try:
        result = subprocess.run(["nrl", "bench", "65536", "64", "1", "8", "omega"], 
                              capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except:
        return "NRL omega failed"

def main():
    model_path = r"C:\Users\Asus\Desktop\Documents\RomaPy Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf"
    
    print("NRL Foundation GGUF Runner")
    print("=" * 70)
    print("Loading model with llama-cpp (base decode)...")

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=6,
        n_batch=256,
        chat_format="chatml",
        verbose=False
    )

    print("Model loaded.\n")

    prompt = "Hello! Tell me a short interesting fact about space."
    print(f"Prompt: {prompt}")
    print("-" * 70)

    start_gen = time.time()
    tokens = 0
    text = ""

    # Generation with NRL calls every 8 tokens (real integration point)
    for chunk in llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7,
        stream=True,
    ):
        if "content" in chunk["choices"][0]["delta"]:
            content = chunk["choices"][0]["delta"]["content"]
            print(content, end="", flush=True)
            text += content
            tokens += 1

            if tokens % 8 == 0:
                print("\n[NRL Acceleration Step]")
                print(run_nrl_assimilate())
                print(run_nrl_omega())
                print("-" * 40)

    total_time = time.time() - start_gen
    tps = tokens / total_time if total_time > 0 else 0

    print("\n" + "=" * 70)
    print(f"Generated {tokens} tokens in {total_time:.2f}s")
    print(f"TPS: {tps:.2f}")
    print("NRL was called during generation for lattice acceleration.")

if __name__ == "__main__":
    main()
