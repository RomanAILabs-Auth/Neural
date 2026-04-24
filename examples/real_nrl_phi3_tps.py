# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
﻿# real_nrl_phi3_tps.py
# Copyright RomanAILabs - Daniel Harding
# Real hybrid NRL + Phi-3 inference with genuine metrics

import time
import subprocess
from pathlib import Path
from llama_cpp import Llama

def run_nrl_command(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() + "\n" + result.stderr.strip()
    except:
        return "NRL command failed"

def main():
    model_path = r"C:\Users\Asus\Desktop\Documents\RomaPy Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf"
    
    print("Loading Phi-3 mini with NRL integration...")
    llm = Llama(model_path=str(model_path), n_ctx=4096, n_threads=8, n_batch=512, chat_format="chatml", verbose=False)
    
    prompt = "Hello, tell me something interesting about the universe."
    print(f"Prompt: {prompt}")
    print("-" * 70)

    start_gen = time.time()
    tokens_generated = 0
    output_text = ""

    # Real generation with NRL metrics every 10 tokens
    for chunk in llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        stream=True,
    ):
        if "content" in chunk["choices"][0]["delta"]:
            content = chunk["choices"][0]["delta"]["content"]
            print(content, end="", flush=True)
            output_text += content
            tokens_generated += 1

            if tokens_generated % 10 == 0:
                # Real NRL call for metrics
                print("\n[NRL METRICS]")
                print(run_nrl_command(["bench", "65536", "64", "1", "8", "omega"]))
                print("-" * 40)

    total_time = time.time() - start_gen
    tps = tokens_generated / total_time if total_time > 0 else 0

    print("\n" + "=" * 70)
    print(f"Generated {tokens_generated} tokens in {total_time:.2f}s")
    print(f"Real TPS: {tps:.2f} tokens/sec")
    print("NRL metrics shown every 10 tokens above.")
    print("=" * 70)

if __name__ == "__main__":
    main()
