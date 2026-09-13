.PHONY: install playground serve

install:
	uv sync

playground:
	uv run adk web appealerAgent

serve:
	uv run uvicorn main:app --host 0.0.0.0 --port 8080
