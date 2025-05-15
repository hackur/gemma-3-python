web: source .venv/bin/activate && uv run python -m uvicorn gemma3:app --host 0.0.0.0 --port 1337 --reload
proxy: source .venv/bin/activate && uv run python gemma_proxy.py --host 0.0.0.0 --port 1338