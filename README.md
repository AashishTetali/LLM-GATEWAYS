# LLM Gateway with Task-Aware Routing

A small FastAPI service that sits in front of several LLMs. It classifies each
incoming query, routes it to the model chosen for that task type, automatically
falls back to the next model if a provider fails, and reports latency and
estimated cost for every request.

## How it works

```
POST /chat  ->  classify_task()  ->  routing table  ->  LiteLLM call
                (code/summary/       (ordered model      with fallback
                 general)             chain per task)    chain + timeout
                                                              |
                                          answer + task + model used + latency + est. cost
```

| Task      | Model order (first = primary, rest = fallbacks)                     |
|-----------|---------------------------------------------------------------------|
| `code`    | gpt-oss-20b → gpt-oss-120b → gemini-3.1-flash-lite                  |
| `summary` | gpt-oss-120b → gpt-oss-20b → gemini-3.1-flash-lite                  |
| `general` | gpt-oss-20b → gpt-oss-120b → gemini-3.1-flash-lite                  |

(gpt-oss models run on Groq; Gemini is called through the Gemini API. All calls
go through [LiteLLM](https://github.com/BerriAI/litellm).)

- **Classifier:** a lightweight rule-based keyword classifier. It adds
  microseconds of latency and has no model call, but it is not a learned model.
- **Fallback:** if a model errors or times out (10 s by default), the next model
  in the chain is tried. The response lists every attempt.
- **Cost:** estimated from token usage and the list prices in `gateway.py`
  (USD per 1M tokens); it is not read from a provider bill.

## Quick start

```bash
git clone https://github.com/AashishTetali/llm-gateway.git
cd llm-gateway

python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # Windows: copy .env.example .env
# then edit .env and add GROQ_API_KEY and GEMINI_API_KEY

python run.py               # starts the server and opens Swagger UI
```

Swagger UI is at http://127.0.0.1:8000/docs. Open `POST /chat`, click
**Try it out**, and send a query.

## API

| Endpoint            | What it does                                                        |
|---------------------|---------------------------------------------------------------------|
| `POST /chat`        | Route one query; returns answer, task, model, latencies, cost, usage, fallback attempts |
| `POST /compare`     | Same query through the direct baseline (gpt-oss-120b) and the gateway |
| `GET /routing`      | Show the routing table, baseline model and pricing                  |
| `GET /health`       | Liveness check                                                      |

Example:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Write a Python function to reverse a string."}'
```

Response (shape):

```json
{
  "task": "code",
  "model": "groq/openai/gpt-oss-20b",
  "routing_chain": ["groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b", "gemini/gemini-3.1-flash-lite"],
  "fallback_used": false,
  "attempts": [{"model": "groq/openai/gpt-oss-20b", "status": "success", "latency": 0.5}],
  "answer": "...",
  "classifier_latency": 0.000004,
  "model_latency": 0.5,
  "total_latency": 0.5,
  "cost": 0.00001,
  "usage": {"prompt_tokens": 40, "completion_tokens": 60, "total_tokens": 100}
}
```

On Windows PowerShell, use `python demo_client.py` instead of curl; it sends
sample queries to the running server and prints a summary table
(`--compare` adds the baseline, `--all` runs all 10 queries).


## Tests

LLM calls are mocked, so no API keys are needed:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Project layout

```
gateway.py      core logic: classifier, routing, fallback, cost, benchmark
main.py         FastAPI app
run.py          one-command launcher (checks keys, starts server, opens /docs)
demo_client.py  sends sample queries to the running server
tests/          pytest suite with mocked LLM calls
```

## Possible next steps

- Replace the keyword classifier with a small learned classifier or embedding similarity
- Larger, labelled benchmark set with answer-quality scoring
- Auth, rate limiting, request logging and a metrics dashboard
- Streaming responses and a Docker image / cloud deployment
