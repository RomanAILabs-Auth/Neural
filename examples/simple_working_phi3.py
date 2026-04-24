# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
﻿from llama_cpp import Llama
import time

print("Loading Phi-3 mini...")
llm = Llama(
    model_path=r"C:\Users\Asus\Desktop\Documents\RomaPy Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=6,
    n_batch=256,
    chat_format="chatml",
    verbose=False
)

prompt = "Hello! Tell me a short interesting fact about space."

print("\nPrompt:", prompt)
print("-" * 60)

start = time.time()
output = llm.create_chat_completion(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=150,
    temperature=0.7
)

text = output['choices'][0]['message']['content']
total_time = time.time() - start
tps = len(text.split()) / total_time if total_time > 0 else 0

print(text)
print("\n---")
print(f"Generated in {total_time:.2f}s | TPS ≈ {tps:.2f}")
