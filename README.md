# Generate more revenue per customer

A recommendation-driven sales engine that converts buying signals into purchases. Built on a production-ready, scalable, agentic architecture.

## Demo

> Coming soon

---

## How it works

**Step 1:** Customer asks the AI agent for music recommendations.

**Step 2:** The agent detects the buying signal, delegates to a specialized recommendations subagent, and returns a personalized list ranked by the customer's purchase history.

**Step 3:** The agent prompts the customer to buy. A human-in-the-loop interrupt pauses execution, awaiting confirmation before any transaction is written. *(Production: completed via Stripe webhook integration.)*

**Step 4:** Customer confirms. The agent converts the buying signal into a completed sales transaction.

---

## Agent Architecture

The system is built as a multi-agent pipeline with a main orchestrator and three specialized subagents.

### Main Agent
- Model: `claude-sonnet-4-6`
- Role: routes customer requests to the right tool or subagent
- Enforces security guardrails at the tool level — not the prompt level

### Subagents

| Subagent | Model | Role |
|---|---|---|
| `recommendations` | `claude-haiku-4-5` | Analyzes purchase history and returns ranked personalized track recommendations |
| `similar_music` | `claude-haiku-4-5` | Fallback when searched artist/album is not in catalog — finds closest matching genre |
| `music_profile` | `claude-haiku-4-5` | Generates a Spotify Wrapped-style summary of the customer's listening history |

Each subagent runs in an **isolated context window** — the main agent only sees the final output, not the intermediate reasoning or tool calls. This keeps the orchestrator lean and the architecture scalable.

---

## Use Cases

### 1. Recommendation-Driven Sales
The core use case. Triggered when a customer asks for recommendations or music discovery.

The recommendations subagent runs a **5-round query sequence**:
- Round 1 (parallel): top genres, owned artist IDs, owned track IDs
- Round 2: playlist IDs containing owned tracks
- Round 3: co-occurrence candidates — unowned tracks sharing playlists with owned tracks, ranked by co-occurrence count
- Round 4 (conditional): genre + owned artist fallback
- Round 5 (conditional): genre + new artist fallback

Candidates are ranked using a **4-tier algorithm**:

| Tier | Signal |
|---|---|
| 1 | Playlist co-occurrence + matching genre + owned artist |
| 2 | Playlist co-occurrence + matching genre + new artist |
| 3 | Matching genre + owned artist (no playlist signal) |
| 4 | Matching genre + new artist (broadest fallback) |

The algorithm stops as soon as 5 recommendations are gathered. If the customer states a specific preference (e.g. "something jazzy"), that overrides their top genres throughout the ranking.

### 2. Purchase with Human-in-the-Loop
When the customer selects a track, `purchase_track()` is called. An `interrupt()` fires before any database write, pausing execution and awaiting customer confirmation.

**Production payment flow:**
1. `interrupt()` sends a Stripe `client_secret` to the frontend
2. Customer completes payment via Stripe's hosted UI
3. Stripe webhook calls `graph.invoke(Command(resume=...))` with the same `thread_id`
4. Agent verifies payment status and writes the invoice to the database

This architecture requires a persistent checkpointer (PostgreSQL or Redis) and a webhook server that maps `payment_intent_id` to `thread_id` — both are standard production infrastructure.

### 3. Catalog Availability
When a customer asks "do you have music by [artist]?" the agent runs two parallel queries:
- **Your Library** — tracks the customer already owns matching the request
- **Available to Purchase** — unowned catalog tracks matching the request

If zero unowned results are found, falls back to the `similar_music` subagent.

### 4. General Inquiry
Handles account details, invoice history, track information, support rep lookups, and purchase patterns via dynamic SQL generation. Guardrailed at the tool level to only ever query data belonging to the authenticated customer.

### 5. Music Profile
Generates a Spotify Wrapped-style narrative: total listening time, favorite artist, genre breakdown, first and most recent purchase, and a haiku grounded in real data.

---

## Production Design Decisions

**Stripe webhook integration for payments**
The HITL purchase flow is designed to plug directly into Stripe in production. The agent pauses at `interrupt()`, the frontend completes the Stripe payment, and the webhook resumes the graph — no polling, no manual approval required. This makes the payment flow fully automated and production-scalable.

**Guardrails at the tool level**
`general_query` rejects any SQL query that does not include `CustomerId = {customer_id}`. This cannot be bypassed by the LLM regardless of what it is instructed to do. `catalog_search` blocks all customer-sensitive tables (`invoice`, `invoiceline`, `customer`, `employee`) at the Python level.

**HITL with `interrupt()` inside the tool**
`interrupt()` is called directly inside `purchase_track()` rather than via `interrupt_on` middleware. This handles both string and dict resume values gracefully — critical for compatibility across Studio and production webhook environments.

**Prompt versioning**
All prompt versions are centralized in `config.py`. Changing a version in one place propagates to the agent, all subagents, and all eval scripts automatically.

**Eval suite**
- `evals/routing.py` — verifies the main agent routes each query type to the correct tool or subagent. Run different system prompt versions via `python evals/routing.py system/system_v2` to compare routing accuracy.
- `evals/recommendations.py` — tests recommendations subagent output quality across prompt versions.

**Authentication**
The demo hardcodes `CustomerId = 5`. The production upgrade path extracts the `CustomerId` from a JWT token on each request and injects it into the agent via a user-scoped `MemoryMiddleware` with `(user_id,)` as the namespace — ensuring every query is strictly scoped to the authenticated customer.

---

## Dataset

Built on the **Chinook database** — an open-source SQLite database representing a digital music store with real customers, invoices, tracks, albums, artists, genres, and playlists. It provides a realistic foundation for demonstrating purchase history analysis, catalog search, and personalized recommendations without requiring proprietary data.

---

## Tech Stack

- `langchain-anthropic` — LLM provider
- `deepagents` — multi-agent harness (`create_deep_agent`)
- `langgraph` — agent graph execution and HITL
- `langsmith` — observability, tracing, and evals
- `python-dotenv` — environment management
- SQLite (Chinook) — data source

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/<your-username>/recommendation-driven-sales-engine.git
cd recommendation-driven-sales-engine
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Configure environment variables**
```bash
cp .env.example .env
```
Fill in your API keys in `.env`:
```
ANTHROPIC_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
```

**4. Run in LangSmith Studio**
Open the project in LangSmith Studio — it will load the graph via `langgraph.json` automatically.

**5. Run evals**
```bash
python evals/routing.py
python evals/recommendations.py
```
