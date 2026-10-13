"""One-command launcher for the demo.

    python run.py

Checks your API keys, starts the FastAPI server, and opens Swagger UI in
the browser, where you can try POST /chat directly.
"""

import os
import sys
import threading
import webbrowser

from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))


def check_keys():
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
        sys.exit(1)
    if not os.getenv("GEMINI_API_KEY"):
        print(
            "WARNING: GEMINI_API_KEY is not set. Groq models will work, "
            "but the Gemini fallback will fail if it is ever needed."
        )


def main():
    check_keys()

    import uvicorn

    url = f"http://{HOST}:{PORT}/docs"
    print(f"\nLLM Gateway starting at {url}\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("main:app", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
