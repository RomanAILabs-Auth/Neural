# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
﻿# simple_phi3_tps.py
# Copyright RomanAILabs - Daniel Harding
# Minimal robust Phi-3 test with good defaults

import time
from pathlib import Path
from llama_cpp import Llama

def main():
    model_path = Path(r"C:\Users\Asus\Desktop\Documents\RomaPy Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please verify the path to your GGUF file.")
        return

    print("Loading Phi-3 mini 4K Instruct...")
    start = time.time()

    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,          # lower context = more stable on laptop
            n_threads=6,         # safe for your CPU
            n_batch=256,         # balanced for TPS
            n_gpu_layers=0,
            verbose=False,
            chat_format="chatml",
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Try lowering n_ctx or n_batch if memory is tight.")
        return

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s\n")

    prompt = "Hello! Tell me a short interesting fact about space."
    print(f"Prompt: {prompt}")
    print("-" * 60)

    start_gen = time.time()
    tokens = 0
    text = ""

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

    total_time = time.time() - start_gen
    tps = tokens / total_time if total_time > 0 else 0

    print("\n" + "-" * 60)
    print(f"Generated {tokens} tokens in {total_time:.2f}s")
    print(f"TPS: {tps:.2f} tokens/sec")

if __name__ == "__main__":
    main()
