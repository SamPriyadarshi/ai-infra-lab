#!/usr/bin/env python3
"""
Lab 1 & 2 Benchmark: Measuring Prefill (TTFT) vs. Decode (ITL/TPOT)
and testing KV Cache / Prefix Caching impact on vLLM.
"""

import asyncio
import json
import time
import aiohttp

VLLM_URL = "http://localhost:8000/v1/completions"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Shared system context (~400 tokens) to test Prefix Caching and Prefill cost
SHARED_SYSTEM_PREFIX = (
    "You are an expert AI Infrastructure Architect at Google Cloud. "
    "You specialize in GPU cluster design, Kubernetes (GKE), vLLM continuous batching, "
    "PagedAttention memory management, Prefill-Decode disaggregation (llm-d), and "
    "high-throughput inference optimization. Below is a technical architecture review "
    "document containing system constraints, SLA targets, and memory budgets. "
) * 6


async def stream_request(session: aiohttp.ClientSession, prompt: str, max_tokens: int = 60):
    """Sends a streaming request to measure exact TTFT (Prefill) and ITL (Decode)."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    start_time = time.perf_counter()
    first_token_time = None
    token_timestamps = []

    async with session.post(VLLM_URL, json=payload) as resp:
        async for line in resp.content:
            decoded = line.decode("utf-8").strip()
            if not decoded.startswith("data: ") or decoded == "data: [DONE]":
                continue
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            token_timestamps.append(now)

    total_time = time.perf_counter() - start_time
    ttft = (first_token_time - start_time) if first_token_time else total_time

    # Calculate Inter-Token Latency (ITL) across decode steps
    if len(token_timestamps) > 1:
        decode_intervals = [
            token_timestamps[i] - token_timestamps[i - 1]
            for i in range(1, len(token_timestamps))
        ]
        avg_itl = sum(decode_intervals) / len(decode_intervals)
    else:
        avg_itl = 0.0

    return {
        "ttft_ms": ttft * 1000,
        "avg_itl_ms": avg_itl * 1000,
        "total_tokens": len(token_timestamps),
        "total_time_s": total_time,
    }


async def run_experiments():
    print("=" * 75)
    print("AI INFRA LAB: PREFILL (TTFT) vs. DECODE (ITL) & PREFIX CACHING BENCHMARK")
    print("=" * 75)

    async with aiohttp.ClientSession() as session:
        # Experiment 1: Short Prompt vs. Long Prompt (Prefill Compute Scaling)
        short_prompt = "Explain PagedAttention in two sentences."
        long_prompt = SHARED_SYSTEM_PREFIX + "\nQuestion: Summarize the key bottleneck during the Decode phase."

        print("\n[Experiment 1] Prefill Scaling: Short vs. Long Input Prompt")
        res_short = await stream_request(session, short_prompt, max_tokens=50)
        print(f"  Short Prompt (~8 tokens)   -> TTFT (Prefill): {res_short['ttft_ms']:6.1f} ms | Avg ITL (Decode): {res_short['avg_itl_ms']:5.1f} ms/tok")

        res_long_cold = await stream_request(session, long_prompt, max_tokens=50)
        print(f"  Long Prompt  (~420 tokens) -> TTFT (Prefill): {res_long_cold['ttft_ms']:6.1f} ms | Avg ITL (Decode): {res_long_cold['avg_itl_ms']:5.1f} ms/tok")

        # Experiment 2: Prefix Caching (Warm KV Cache)
        print("\n[Experiment 2] Automatic Prefix Caching (Re-using KV Cache Blocks)")
        follow_up_prompt = SHARED_SYSTEM_PREFIX + "\nQuestion: Why does Prefill-Decode disaggregation prevent Head-of-Line blocking?"
        res_long_warm = await stream_request(session, follow_up_prompt, max_tokens=50)
        print(f"  Warm Prefix  (~420 tokens) -> TTFT (Prefill): {res_long_warm['ttft_ms']:6.1f} ms | Avg ITL (Decode): {res_long_warm['avg_itl_ms']:5.1f} ms/tok")

        speedup = res_long_cold["ttft_ms"] / res_long_warm["ttft_ms"] if res_long_warm["ttft_ms"] > 0 else 1.0
        print(f"  >> Prefix Cache TTFT Speedup: {speedup:.2f}x faster prefill!")


if __name__ == "__main__":
    asyncio.run(run_experiments())
