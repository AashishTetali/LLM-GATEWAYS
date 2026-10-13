"""FastAPI app exposing the LLM gateway.

Run:   python run.py            (starts the server and opens Swagger UI)
Docs:  http://127.0.0.1:8000/docs
"""

from dotenv import load_dotenv

load_dotenv()  # read GROQ_API_KEY / GEMINI_API_KEY from .env

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import gateway  # noqa: E402

app = FastAPI(
    title="LLM Gateway with Task-Aware Routing",
    description=(
        "Classifies each query (code / summary / general), routes it to the "
        "model chosen for that task, retries automatically on the next model "
        "if a provider fails (LiteLLM), and tracks per-request latency and "
        "estimated cost."
    ),
    version="1.0.0",
)


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="The user question to route.",
        examples=["Write a Python function to reverse a string."],
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=4096,
        description="Max output tokens (defaults to the MAX_TOKENS env var, 100).",
    )


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/routing")
def routing():
    """Show the routing table: which models are tried, in order, per task."""
    return {
        "models": gateway.MODELS,
        "routing": gateway.ROUTING,
        "baseline_model": gateway.BASELINE_MODEL,
        "pricing_usd_per_1m_tokens": {
            name: {"input": inp, "output": out}
            for name, (inp, out) in gateway.PRICING.items()
        },
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """Route one query through the gateway and return the answer plus metrics."""
    try:
        return gateway.gateway_call(req.query, max_tokens=req.max_tokens)
    except gateway.AllModelsFailed as e:
        raise HTTPException(
            status_code=502,
            detail={"message": str(e), "attempts": e.attempts},
        )


@app.post("/compare")
def compare(req: ChatRequest):
    """Run the same query through the direct baseline and the gateway."""
    try:
        baseline = gateway.call_direct(
            gateway.BASELINE_MODEL, req.query, max_tokens=req.max_tokens
        )
    except Exception as e:
        baseline = {"model": gateway.BASELINE_MODEL, "error": gateway.short_error(e)}

    try:
        routed = gateway.gateway_call(req.query, max_tokens=req.max_tokens)
    except gateway.AllModelsFailed as e:
        raise HTTPException(
            status_code=502,
            detail={"message": str(e), "attempts": e.attempts, "baseline": baseline},
        )

    return {"baseline": baseline, "gateway": routed}


# @app.post("/benchmark")
# def benchmark(
#     limit: int = Query(
#         default=len(gateway.BENCHMARK_QUERIES),
#         ge=1,
#         le=len(gateway.BENCHMARK_QUERIES),
#         description="How many of the 10 benchmark queries to run.",
#     ),
# ):
#     """Run the benchmark queries through baseline and gateway, return averages.

#     Each query makes two model calls, so the full run takes a while.
#     """
#     return gateway.run_benchmark(gateway.BENCHMARK_QUERIES[:limit])
