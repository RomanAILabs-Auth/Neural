# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
﻿# romanai_nrl_phi3.py
# Copyright RomanAILabs - Daniel Harding
# RomanAI 4D GGUF runner + NRL metrics

import time
import subprocess
from pathlib import Path

def run_romanai(model_path, prompt, max_tokens=200):
    try:
        # Correct path from your environment
        romanai_exe = r"C:\Users\Asus\AppData\Roaming\RomanAI-V2\bin\llama-cli.exe"
        
        cmd = [romanai_exe, "-m", str(model_path), "-c", "2048", "-t", "8", 
               "--temp", "0.7", "--top-p", "0.9", "-n", str(max_tokens),
               "-cnv", "--no-display-prompt", "-p", 
               "You are a helpful assistant. Answer in ONE short sentence."]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout.strip()
    except Exception as e:
        return f"RomanAI failed: {e}"

def run_nrl_metrics():
    try:
        result = subprocess.run(["nrl", "bench", "65536", "64", "1", "8", "omega"], 
                              capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except:
        return "NRL metrics unavailable"

def main():
    model_path = Path(r"C:\Users\Asus\Desktop\Documents\RomaPy Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    print("Starting RomanAI 4D + NRL demo...")
    print(f"Model: {model_path.name}")
    print("-" * 70)

    prompt = "Hello! Tell me a short interesting fact about space."
    print(f"Prompt: {prompt}")
    print("-" * 70)

    start = time.time()
    output = run_romanai(model_path, prompt)
    total_time = time.time() - start

    print(output if output else "[No output - check romanai.exe path]")
    print("-" * 70)

    print("NRL Metrics (Omega mode):")
    print(run_nrl_metrics())

    print(f"\nTotal time: {total_time:.2f}s")

if __name__ == "__main__":
    main()
