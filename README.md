# Denial Therapy

An ambient AI agent I made for the Kaggle AI agents Google collab course. Built on Google's ADK and Gemini with Antigravity. 

- Detects medical insurance denials with Google Pub/Sub and drafts appeal letters. 
- It classifies each denial, scrubs PII with anti prompt injection measures
- When needed, drafts a letter with an LLM then has human in the loop review. 
∑

- **uv** — Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **Python 3.11–3.13**
- A **Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Setup

```bash
uv sync                 # install deps into .venv
cp .env.example .env     # then set GOOGLE_API_KEY=... in .env
```

## Running it

**Playground:** launches the ADK web UI.

```bash
uv run adk web appealerAgent
```

Open the printed URL, pick `denial_appeal_workflow`, and paste a denial event:

```json
{
  "claim_id": "CLM-88213",
  "payer": "MockPayer",
  "cpt_code": "99213",
  "denial_reason_code": "duplicate claim",
  "billed_amount": 150.00,
  "denial_description": "Previously submitted under claim CLM-88213.",
  "patient_info": "Jane Doe, Member ID M123456789, DOB 01/02/1980"
}
```

**Ambient service:** runs the FastAPI app from `main.py` on port 8080 with
`/` (health), `/pubsub` (ingest), and `/resume` (approve/edit/reject) endpoints.

```bash
make serve
```

## Tests

```bash
uv run pytest tests/unit tests/integration
```

## Limitations for now

Not full production.

- Injection defense is keyword-based so it may miss obfuscated messages
- PII scrubbing is regex not true de-identification.
- In-memory sessions so paused reviews are lost on restart, and there's no `/resume` path for restarted Pub/Sub events.
- Drafts aren't fact-checked(other than just human review) and endpoints have no auth.
