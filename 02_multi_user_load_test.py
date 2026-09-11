#!/usr/bin/env python3
"""
02_multi_user_load_test.py
Multi-user concurrency benchmark demonstrating vLLM Continuous Batching.
Measures how System Throughput (tok/s) scales vs. Average User Latency (s)
as concurrent users ramp up (1 -> 5 -> 10 -> 20).
"""

import asyncio
import time
import aiohttp

VLLM_URL = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

PROMPTS = [
    "What is machine learning and how does it differ from traditional programming?",
    "Explain neural networks and gradient descent in one concise paragraph.",
    "How does a Transformer model use self-attention during inference?",
    "What is PagedAttention in vLLM and why does it save GPU VRAM?",
    "Describe the difference between the Prefill and Decode phases in LLMs.",
    "What are tokens in the context of Large Language Models?",
    "Why is LLM decoding memory-bandwidth bound on GPUs?",
    "How does continuous batching improve GPU compute utilization?",
    "Explain 4-bit AWQ quantization and its impact on model memory.",
    "What is KV-cache-aware routing in distributed LLM serving?",
]


async def send_request(session: aiohttp.ClientSession, prompt: str, max_tokens: int = 60):
    """Send a single completion request and record latency + tokens generated."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    start = time.perf_counter()
    try:
        async with session.post(f"{VLLM_URL}/v1/completions", json=payload) as resp:
            data = await resp.json()
            elapsed = time.perf_counter() - start
            if resp.status != 200:
                return {"latency": elapsed, "tokens": 0, "success": False}
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            return {"latency": elapsed, "tokens": tokens, "success": True}
    except Exception:
        return {"latency": time.perf_counter() - start, "tokens": 0, "success": False}


async def run_concurrency_level(num_users: int):
    """Fire num_users concurrent requests simultaneously using asyncio.gather."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, PROMPTS[i % len(PROMPTS)])
            for i in range(num_users)
        ]
        start_wall = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_wall = time.perf_counter() - start_wall

    successful = [r for r in results if r["success"]]
    total_tokens = sum(r["tokens"] for r in successful)
    avg_latency = sum(r["latency"] for r in successful) / len(successful) if successful else 0.0
    throughput = total_tokens / total_wall if total_wall > 0 else 0.0

    return {
        "users": num_users,
        "total_tokens": total_tokens,
        "total_time": total_wall,
        "throughput": throughput,
        "avg_latency": avg_latency,
        "success_rate": (len(successful) / len(results)) * 100 if results else 0.0,
    }


async def main():
    print("=" * 72)
    print("AI Infra Lab: Multi-User Concurrency & Continuous Batching Benchmark")
    print("=" * 72)
    print(f"Target Server: {VLLM_URL} | Model: {MODEL_NAME}\n")

    concurrency_steps = [1, 5, 10, 20]
    results = []

    for users in concurrency_steps:
        print(f"  Running load test with {users:>2} concurrent user(s)...", end=" ", flush=True)
        res = await run_concurrency_level(users)
        results.append(res)
        print(f"Done -> {res['throughput']:6.1f} tok/s | Avg Latency: {res['avg_latency']:.2f}s")

    print("\n--- CONTINUOUS BATCHING SCALING TABLE ---")
    print(f"{'Users':>6} {'Total Tokens':>13} {'Wall Time':>10} {'System Throughput':>19} {'Avg Latency':>13}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['users']:>6} "
            f"{r['total_tokens']:>13} "
            f"{r['total_time']:>9.2f}s "
            f"{r['throughput']:>14.1f} tok/s "
            f"{r['avg_latency']:>12.2f}s"
        )

    if len(results) >= 2 and results[0]["throughput"] > 0:
        peak = max(results, key=lambda x: x["throughput"])
        scaling = peak["throughput"] / results[0]["throughput"]
        print(f"\nScaling Factor: {scaling:.1f}x higher system throughput at {peak['users']} concurrent users!")


if __name__ == "__main__":
    asyncio.run(main())
