# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
﻿# real_phi3_tps.py
# Copyright RomanAILabs - Daniel Harding
# Robust high-TPS Phi-3 test with better error handling

import time
import sys
from pathlib import Path
from llama_cpp import Llama

def main():
    model_path = Path(r"C:\Users\Asus\Desktop\Documents\RomaPy Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf")
    
    if not model_path.exists():
        print(f"ERROR: Model file not found at:\n{model_path}")
        print("Please check the path and make sure the GGUF file exists.")
        sys.exit(1)

    print(f"Loading Phi-3 mini 4K Instruct from:\n{model_path}")
    start_load = time.time()

    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=6,           # safe starting point for your laptop
            n_batch=256,           # balanced for TPS
            n_gpu_layers=0,
            verbose=False,
            chat_format="chatml",  # correct for Phi-3
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Common fixes:")
        print("1. Make sure the GGUF file is not corrupted")
        print("2. Try a lower n_batch or n_threads")
        print("3. Reinstall llama-cpp-python with: pip install llama-cpp-python --force-reinstall --no-cache-dir")
        sys.exit(1)

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s\n")

    prompt = "Hello, how are you today? Please respond naturally."
    print(f"Prompt: {prompt}")
    print("-" * 70)

    start_gen = time.time()
    tokens_generated = 0
    output_text = ""

    try:
        stream = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.7,
            stream=True,
        )

        for chunk in stream:
            if "content" in chunk["choices"][0]["delta"]:
                content = chunk["choices"][0]["delta"]["content"]
                print(content, end="", flush=True)
                output_text += content
                tokens_generated += 1

    except Exception as e:
        print(f"\nError during generation: {e}")

    total_time = time.time() - start_gen
    tps = tokens_generated / total_time if total_time > 0 else 0

    print("\n" + "-" * 70)
    print(f"Generated {tokens_generated} tokens in {total_time:.2f}s")
    print(f"Raw TPS: {tps:.2f} tokens/sec")

if __name__ == "__main__":
    main()
