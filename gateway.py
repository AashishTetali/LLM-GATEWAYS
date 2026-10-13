"""Core logic for the LLM gateway: task classification, model routing,
fallback chains, and latency/cost tracking.

This module has no side effects on import, so it can be used by the FastAPI
app (main.py), the CLI benchmark (benchmark.py), and the tests.
"""

import logging
import os
import time
from statistics import mean

from litellm import completion, completion_cost

logger = logging.getLogger("gateway")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Keep three models in this order.
MODELS = [
    "groq/openai/gpt-oss-20b",
    "groq/openai/gpt-oss-120b",
    "gemini/gemini-3.1-flash-lite",
]

SYSTEM_PROMPT = (
    "Answer concisely in 1-3 sentences. "
    "For coding questions, provide code with minimal explanation."
)

# Standard prices in USD per 1 million tokens: (input price, output price)
PRICING = {
    "gpt-oss-20b": (0.075, 0.30),
    "gpt-oss-120b": (0.15, 0.60),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}

# Model order to try for each task type. The first model is the primary,
# the rest are fallbacks used only if the previous one fails.
ROUTING = {
    "code": [MODELS[0], MODELS[1], MODELS[2]],
    "summary": [MODELS[1], MODELS[0], MODELS[2]],
    "general": [MODELS[0], MODELS[1], MODELS[2]],
}

# The baseline the gateway is compared against (direct call, no routing).
BASELINE_MODEL = MODELS[1]

# gpt-oss models are reasoning models: their hidden "thinking" tokens count
# toward this limit, so a small value can leave no room for the visible answer.
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))

# Optional: "low", "medium" or "high". Set REASONING_EFFORT=low in .env to make
# the Groq gpt-oss models think less (faster and cheaper). Unset by default.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "").strip().lower()

BENCHMARK_QUERIES = [
    "Write a Python function to reverse a string.",
    "Write a Java function to check whether a number is prime.",
    "Explain how to implement binary search in Python.",
    "Summarize the main plot of Harry Potter and the Chamber of Secrets in 5 bullet points.",
    "Summarize the concept of supervised learning in 3 bullet points.",
    "Summarize what REST APIs are and how they work.",
    "What is the difference between RAM and ROM?",
    "Why is the sky blue?",
    "What is the purpose of a database index?",
    "Give me three interesting facts about space.",
]


def short_error(e):
    """One-line error summary (provider errors can include long tracebacks)."""
    lines = str(e).strip().splitlines()
    first = lines[0] if lines else ""
    return f"{type(e).__name__}: {first}"[:300]


class AllModelsFailed(Exception):
    """Raised when every model in the fallback chain failed."""

    def __init__(self, attempts):
        self.attempts = attempts
        last = attempts[-1]["error"] if attempts else "no models configured"
        super().__init__(f"All models failed. Last error: {last}")


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def classify_task(user_query: str) -> str:
    query = user_query.lower()

    code_keywords = [
        "write code", "write a function", "implement",
        "program", "debug", "fix this code",
        "python function", "java function",
        "write a python", "write a java",
    ]

    summary_keywords = [
        "summarize", "summary", "bullet points",
        "summarise", "main plot", "in brief",
    ]

    if any(word in query for word in code_keywords):
        return "code"

    if any(word in query for word in summary_keywords):
        return "summary"

    return "general"


# --------------------------------------------------------------------------
# Cost and usage helpers
# --------------------------------------------------------------------------

def usage_to_dict(usage):
    """Turn a LiteLLM usage object into a plain dict (JSON-serializable)."""
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def get_cost(response, model_name):
    """Estimate cost from token usage, with LiteLLM as fallback."""
    usage = getattr(response, "usage", None)

    if usage:
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)

        logger.info("Token usage: input=%s, output=%s", input_tokens, output_tokens)

        model_lower = model_name.lower()

        for model_key, (input_rate, output_rate) in PRICING.items():
            if model_key in model_lower:
                if input_tokens is not None and output_tokens is not None:
                    return (
                        input_tokens * input_rate
                        + output_tokens * output_rate
                    ) / 1_000_000

    # Fallback: let LiteLLM calculate the cost.
    try:
        return completion_cost(completion_response=response)
    except Exception as e:
        logger.warning("LiteLLM cost calculation failed: %s", e)
        return None


def _answer_text(response):
    return response.choices[0].message.content or ""


def _finish_reason(response):
    """Why generation stopped: 'stop' (finished) or 'length' (hit max_tokens)."""
    return getattr(response.choices[0], "finish_reason", None)


def _reasoning_preview(response):
    """First part of the model's hidden reasoning, if the provider returns it.

    Useful to diagnose empty answers: if the answer is "" but this is filled,
    the model spent its token budget thinking.
    """
    message = response.choices[0].message
    text = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    return text[:200] if isinstance(text, str) and text else None


def _extra_params(model):
    """Extra completion kwargs; reasoning_effort applies to Groq gpt-oss only."""
    if REASONING_EFFORT and model.startswith("groq/openai/gpt-oss"):
        return {"reasoning_effort": REASONING_EFFORT}
    return {}


# --------------------------------------------------------------------------
# Model calls
# --------------------------------------------------------------------------

def call_direct(model, query, max_tokens=None):
    """Baseline: send the query straight to one model, no routing/fallback."""
    start = time.perf_counter()

    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
        **_extra_params(model),
    )

    latency = time.perf_counter() - start
    cost = get_cost(response, model)

    return {
        "model": model,
        "answer": _answer_text(response),
        "latency": latency,
        "cost": cost,
        "finish_reason": _finish_reason(response),
        "reasoning_preview": _reasoning_preview(response),
        "usage": usage_to_dict(response.usage),
    }


def call_with_fallback(model_chain, messages, max_tokens=None):
    """Try each model in order; return the first successful response.

    Returns (response, model_used, attempts). `attempts` records every model
    tried, so a failed-then-recovered request is visible in the API output.
    """
    attempts = []

    for model in model_chain:
        start = time.perf_counter()

        try:
            response = completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
                timeout=REQUEST_TIMEOUT,
                **_extra_params(model),
            )

            elapsed = time.perf_counter() - start
            logger.info("Successful model: %s, request latency: %.3fs", model, elapsed)
            attempts.append({"model": model, "status": "success", "latency": elapsed})
            return response, model, attempts

        except Exception as e:
            elapsed = time.perf_counter() - start
            error = short_error(e)
            logger.warning("Failed: %s - %s", model, error)
            attempts.append(
                {"model": model, "status": "failed", "latency": elapsed, "error": error}
            )

    raise AllModelsFailed(attempts)


def gateway_call(query, max_tokens=None):
    """Classify the query, route it, call with fallbacks, and track metrics."""
    total_start = time.perf_counter()

    classifier_start = time.perf_counter()
    task = classify_task(query)
    classifier_latency = time.perf_counter() - classifier_start

    chain = ROUTING.get(task, ROUTING["general"])

    model_start = time.perf_counter()

    response, used_model, attempts = call_with_fallback(
        model_chain=chain,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_tokens=max_tokens,
    )

    model_latency = time.perf_counter() - model_start
    total_latency = time.perf_counter() - total_start
    cost = get_cost(response, used_model)

    return {
        "task": task,
        "model": used_model,
        "routing_chain": chain,
        "fallback_used": used_model != chain[0],
        "attempts": attempts,
        "answer": _answer_text(response),
        "classifier_latency": classifier_latency,
        "model_latency": model_latency,
        "total_latency": total_latency,
        "cost": cost,
        "finish_reason": _finish_reason(response),
        "reasoning_preview": _reasoning_preview(response),
        "usage": usage_to_dict(response.usage),
    }


# --------------------------------------------------------------------------
# Benchmark (gateway vs. direct baseline)
# --------------------------------------------------------------------------

def _pct_reduction(baseline, new):
    if baseline is None or baseline <= 0:
        return None
    return (baseline - new) / baseline * 100


def run_benchmark(queries=None, on_result=None):
    """Run each query through the baseline and the gateway, then summarize.

    `on_result(index, query, baseline, gateway)` is called after each query
    (either result may be an {"error": ...} dict) so callers can print progress.
    """
    queries = queries if queries is not None else BENCHMARK_QUERIES

    gateway_costs, gateway_latencies = [], []
    direct_costs, direct_latencies = [], []
    results = []

    for i, query in enumerate(queries, 1):
        try:
            baseline = call_direct(BASELINE_MODEL, query)
            if baseline["cost"] is not None:
                direct_costs.append(baseline["cost"])
            direct_latencies.append(baseline["latency"])
        except Exception as e:
            baseline = {"error": short_error(e)}

        try:
            gateway = gateway_call(query)
            if gateway["cost"] is not None:
                gateway_costs.append(gateway["cost"])
            gateway_latencies.append(gateway["total_latency"])
        except Exception as e:
            gateway = {"error": short_error(e)}

        results.append({"query": query, "baseline": baseline, "gateway": gateway})
        if on_result:
            on_result(i, query, baseline, gateway)

    summary = {
        "queries_run": len(queries),
        "baseline_model": BASELINE_MODEL,
        "baseline_successes": len(direct_latencies),
        "gateway_successes": len(gateway_latencies),
    }

    if direct_costs and gateway_costs:
        b, g = mean(direct_costs), mean(gateway_costs)
        summary.update(
            baseline_avg_cost=b,
            gateway_avg_cost=g,
            cost_reduction_pct=_pct_reduction(b, g),
        )

    if direct_latencies and gateway_latencies:
        b, g = mean(direct_latencies), mean(gateway_latencies)
        summary.update(
            baseline_avg_latency=b,
            gateway_avg_latency=g,
            latency_improvement_pct=_pct_reduction(b, g),
        )

    return {"summary": summary, "results": results}