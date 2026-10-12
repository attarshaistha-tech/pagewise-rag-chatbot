"""
test_gemini.py - checks your Gemini key and model WITHOUT the app.
Run:  python test_gemini.py
It prints (1) which models your key can use and (2) whether a simple question works.
"""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
model = os.getenv("LLM_MODEL", "gemini-3.8-flash")

if not key:
    raise SystemExit("No GEMINI_API_KEY found. Check that your .env file is in this folder.")

client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=30_000))

print("Step 1: models your key can use for text answers")
try:
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            names.append(m.name.replace("models/", ""))
    for name in names[:40]:
        print("  -", name)
    if not names:
        print("  (no models listed)")
except Exception as error:
    print("  Could not list models:", type(error).__name__, error)

print(f"\nStep 2: asking the model '{model}' a simple question")
try:
    response = client.models.generate_content(model=model, contents="Say hello in five words.")
    print("  WORKS. Answer:", response.text)
except Exception as error:
    print("  FAILED:", type(error).__name__, error)
    print("\nIf the model name is wrong, copy one name from the Step 1 list into LLM_MODEL in your .env file.")
