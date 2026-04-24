# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
﻿# robust_phi3_nrl.py
# Copyright RomanAILabs - Daniel Harding
# Robust Phi-3 inference with real NRL metrics

import time
from pathlib import Path
from llama_cpp import Llama

def main():
    model_path = Path(r"C:\Users\Asus\Desktop\Documents\RomaPy Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf")
    
    if not model_path.exists():
        print(f"ERROR: Model not found:\n{model_path}")
        print("Please check the path to your GGUF file.")
        return

    print("Loading Phi-3 mini 4K Instruct...")
    start = time.time()

    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,          # safer for laptop
            n_threads=6,         # safe default for your CPU
            n_batch=256,
            n_gpu_layers=0,
            verbose=False,
            chat_format="chatml",   # correct for Phi-3
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s\n")

    prompt = "Hello! Tell me a short interesting fact about space."
    print(f"Prompt: {prompt}")
    print("-" * 70)

    start_gen = time.time()
    tokens = 0
    text = ""

    try:
        for chunk in llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
            stream=True,
        ):
            if "content" in chunk["choices"][0]["delta"]:
                content = chunk["choices"][0]["delta"]["content"]
                print(content, end="", flush=True)
                text += content
                tokens += 1
    except Exception as e:
        print(f"\nGeneration error: {e}")

    total_time = time.time() - start_gen
    tps = tokens / total_time if total_time > 0 else 0

    print("\n" + "-" * 70)
    print(f"Generated {tokens} tokens in {total_time:.2f}s")
    print(f"TPS: {tps:.2f} tokens/sec")

    # Real NRL metrics
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
