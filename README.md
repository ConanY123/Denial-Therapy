# Denial Therapy

An ambient AI agent I made for the Kaggle AI agents Google collab course. Built on Google's ADK and Gemini with Antigravity. 

- Detects medical insurance denials with Google Pub/Sub and drafts appeal letters. 
- It classifies each denial, scrubs PII with anti prompt injection measures
- When needed, drafts a letter with an LLM then has human in the loop review. 

## Reqs

- **uv** — Python package manager
- **agents-cli** — `uv tool install google-agents-cli`
- A Gemini API key

## Setup

```bash
# Install dependencies
uv sync

# API key
cp .env.example .env
# then edit .env and set GOOGLE_API_KEY=...
```

## Use

Interactive local playground:

```bash
agents-cli playground
```

Run the ambient service (FastAPI) locally:

```bash
make serve
```

Run the tests:

```bash
uv run pytest tests/unit tests/integration
```