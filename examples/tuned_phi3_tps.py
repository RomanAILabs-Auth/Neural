# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
﻿# tuned_phi3_tps.py
# Copyright RomanAILabs - Daniel Harding
# Tuned Phi-3 inference for better TPS on laptop

import time
from pathlib import Path
from llama_cpp import Llama

def main():
    model_path = Path(r"C:\Users\Asus\Desktop\Documents\RomaPy Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    print("Loading Phi-3 mini with tuned settings...")
    start = time.time()

    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=8,           # your CPU has 8 logical cores
        n_batch=512,           # larger batch helps TPS
        n_gpu_layers=0,
        verbose=False,
        chat_format="chatml",
    )

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s\n")

    prompt = "Hello! Tell me a short interesting fact about space."
    print(f"Prompt: {prompt}")
    print("-" * 70)

    start_gen = time.time()
    tokens = 0
    text = ""

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

    total_time = time.time() - start_gen
    tps = tokens / total_time if total_time > 0 else 0

    print("\n" + "-" * 70)
    print(f"Generated {tokens} tokens in {total_time:.2f}s")
    print(f"TPS: {tps:.2f} tokens/sec")

    # NRL metrics
    print("\nNRL Metrics (Omega mode):")
    try:
        import subprocess
        result = subprocess.run(["nrl", "bench", "65536", "64", "1", "8", "omega"], 
                              capture_output=True, text=True, timeout=10)
        print(result.stdout.strip())
    except:
        print("NRL bench not available")

if __name__ == "__main__":
    main()
