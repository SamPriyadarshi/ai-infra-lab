#!/usr/bin/env python3
"""
01_paged_attention_simulation.py
Mathematical simulation comparing Contiguous KV Cache Allocation vs.
vLLM's PagedAttention (OS-style virtual memory paging for LLM inference).
"""

import math


def main():
    print("=" * 70)
    print("AI Infra Fundamentals: PagedAttention vs. Contiguous KV Cache")
    print("=" * 70)

    # Sample incoming user requests with varying token lengths
    requests = [
        {"id": 1, "prompt_tokens": 45,  "description": "Short question"},
        {"id": 2, "prompt_tokens": 128, "description": "Medium paragraph"},
        {"id": 3, "prompt_tokens": 23,  "description": "Quick greeting"},
        {"id": 4, "prompt_tokens": 256, "description": "Long document"},
        {"id": 5, "prompt_tokens": 67,  "description": "Code snippet"},
    ]

    max_seq_len = 512  # Worst-case contiguous pre-allocation per request
    page_size = 16     # vLLM default --block-size (16 tokens per page)

    print(f"\nConfiguration:")
    print(f"  - Page size (--block-size): {page_size} tokens per page")
    print(f"  - Contiguous allocation:    {max_seq_len} tokens per request (worst-case)\n")

    total_paged_allocated = 0
    total_contiguous_allocated = 0
    total_used = 0

    print("--- PER-REQUEST KV CACHE ALLOCATION ---")
    for req in requests:
        actual = req["prompt_tokens"]
        total_used += actual

        # Contiguous: reserves max_seq_len (512) slots regardless of actual length
        contiguous_alloc = max_seq_len
        total_contiguous_allocated += contiguous_alloc

        # PagedAttention: allocates fixed-size 16-token blocks on demand
        pages_needed = math.ceil(actual / page_size)
        paged_alloc = pages_needed * page_size
        total_paged_allocated += paged_alloc

        paged_waste = (paged_alloc - actual) / paged_alloc * 100 if paged_alloc > 0 else 0
        page_blocks = "|".join(["##" for _ in range(pages_needed)])

        print(f"  Request {req['id']} ({req['description']:<16}): {actual:>3} tokens -> "
              f"{pages_needed:>2} pages ({paged_alloc:>3} slots) | Internal waste: {paged_waste:>4.1f}%")
        print(f"    Allocated Pages: [{page_blocks}]")

    paged_utilization = (total_used / total_paged_allocated) * 100
    contiguous_utilization = (total_used / total_contiguous_allocated) * 100

    print("\n--- SIDE-BY-SIDE KV CACHE COMPARISON ---")
    print(f"{'Method':<14} {'Total Reserved':>16} {'Actual Used':>14} {'VRAM Utilization':>18}")
    print("-" * 65)
    print(f"{'Contiguous':<14} {total_contiguous_allocated:>12} slots {total_used:>10} slots {contiguous_utilization:>16.1f}%")
    print(f"{'Paged (vLLM)':<14} {total_paged_allocated:>12} slots {total_used:>10} slots {paged_utilization:>16.1f}%")

    savings_ratio = total_contiguous_allocated / total_paged_allocated
    print(f"\nResult: PagedAttention reserves {savings_ratio:.1f}x LESS GPU memory for the exact same workload!")


if __name__ == "__main__":
    main()
