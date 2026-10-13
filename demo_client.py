"""Send sample queries to the running gateway and print a summary table.

    python demo_client.py                 # 3 sample queries via /chat
    python demo_client.py --all           # all 10 benchmark queries
    python demo_client.py "Why is the sky blue?"
    python demo_client.py --compare "Write a Python function to reverse a string."

Start the server first (python run.py) in another terminal.
"""

import argparse
import os
import sys

import requests

BASE_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:8000")

SAMPLES = [
    "Write a Python function to reverse a string.",
    "Summarize the concept of supervised learning in 3 bullet points.",
    "Why is the sky blue?",
]


def post(path, payload):
    resp = requests.post(f"{BASE_URL}{path}", json=payload, timeout=60)
    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code}: {resp.text}")
        return None
    return resp.json()


def show_chat(query):
    print("=" * 70)
    print("QUERY:", query)
    data = post("/chat", {"query": query})
    if not data:
        return
    print(f"  task:     {data['task']}")
    print(f"  model:    {data['model']}" + ("  (fallback used)" if data["fallback_used"] else ""))
    print(f"  latency:  {data['total_latency']:.3f}s (classifier {data['classifier_latency'] * 1000:.3f} ms)")
    cost = data["cost"]
    print(f"  cost:     ${cost:.8f}" if cost is not None else "  cost:     n/a")
    print(f"  tokens:   {data['usage']}")
    print(f"  answer:   {data['answer']}")


def show_compare(query):
    print("=" * 70)
    print("QUERY:", query)
    data = post("/compare", {"query": query})
    if not data:
        return
    b, g = data["baseline"], data["gateway"]
    if "error" in b:
        print("  baseline failed:", b["error"])
    else:
        print(f"  BASELINE  {b['model']}: {b['latency']:.3f}s, cost ${b['cost']:.8f}")
    print(f"  GATEWAY   {g['model']} (task={g['task']}): {g['total_latency']:.3f}s, cost ${g['cost']:.8f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", nargs="*", help="custom queries")
    parser.add_argument("--all", action="store_true", help="run all 10 benchmark queries")
    parser.add_argument("--compare", action="store_true", help="also run the direct baseline")
    args = parser.parse_args()

    try:
        requests.get(f"{BASE_URL}/health", timeout=5).raise_for_status()
    except requests.RequestException:
        print(f"Cannot reach the gateway at {BASE_URL}. Start it first with: python run.py")
        sys.exit(1)

    if args.queries:
        queries = args.queries
    elif args.all:
        from gateway import BENCHMARK_QUERIES

        queries = BENCHMARK_QUERIES
    else:
        queries = SAMPLES

    for q in queries:
        (show_compare if args.compare else show_chat)(q)


if __name__ == "__main__":
    main()
