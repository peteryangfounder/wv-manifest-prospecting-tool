# Manifest Prospecting Tool

Manifest Prospecting Tool is an internal VC sourcing dashboard for Wittington Ventures. It turns the public Manifest attendee list into two clear working views: a source list with cleaned attendee rows and a verified prospect list backed by external evidence.

The product uses code for retrieval, storage, filtering, and cost control. It uses Tavily for external search evidence, then uses OpenAI for scoring after that evidence has been retrieved and cached.

Hosted app: https://wv-manifest-prospecting-tool-b8jadagmhgsh8wirbknjb9.streamlit.app/

## Current App

- Guided Streamlit workflow with three steps: prepare source list, verify prospects, review results.
- Responsive source cards for messy attendee data after cleaning, dedupe, and rule screening.
- Verified prospects view for companies that passed evidence enrichment and API scoring.
- Larger full-width charts with horizontal labels for readability.
- Responsive prospect cards so company descriptions and sector tags stay readable across desktop, tablet, and mobile.
- Prominent billing summary that separates provider-billed spend from local token estimates and Tavily credit consumption.
- Model selector and batch cap for controlling how many companies get verified in each run.
- Scoring weight controls for investor preference changes.
- Simple reset button that clears local source rows, cached enrichments, scores, and run history.
- CSV exports for source rows and verified prospects.

## Tech Stack

- Python
- Streamlit
- SQLite
- requests and BeautifulSoup
- Tavily Search API
- OpenAI API
- pandas
- pytest

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=your_openai_key_here
OPENAI_ADMIN_KEY=your_openai_admin_key_here
TAVILY_API_KEY=your_tavily_key_here
OPENAI_MODEL=gpt-4o-mini
MAX_ENRICH=75
MAX_SCORE=75
TAVILY_MAX_RESULTS=3
DATABASE_PATH=data/prospects.db
OPENAI_INPUT_COST_PER_1M_TOKENS=0.15
OPENAI_OUTPUT_COST_PER_1M_TOKENS=0.60
TAVILY_COST_PER_CALL_USD=0.001
OPENAI_BILLING_PROJECT_ID=
OPENAI_BILLING_LOOKBACK_DAYS=31
OPENAI_BILLING_CACHE_TTL_SECONDS=300
TAVILY_PLAN_NAME=Researcher
TAVILY_INCLUDED_MONTHLY_CREDITS=1000
TAVILY_PAY_AS_YOU_GO_ENABLED=false
TAVILY_PAYG_PRICE_PER_CREDIT_USD=0.008
```

The app opens without API keys and can create baseline rule scores. Set `TAVILY_API_KEY` to retrieve external search evidence. Set `OPENAI_API_KEY` to generate API-scored verified prospects. Set `OPENAI_ADMIN_KEY` only when you want the dashboard to read live OpenAI organization costs and usage from the admin billing endpoints; it is separate from the normal inference key.

## Run Locally

```bash
streamlit run app.py
```

Use the dashboard in this order:

1. Click **1. Load and screen source data**.
2. Choose the number of companies to verify and the OpenAI model.
3. Click **2. Verify prospects with APIs**.
4. Review the Overview, Source list, Verified prospects, and Company detail tabs.

For a command-line run:

```bash
python scripts/run_pipeline.py --all
```

You can also run individual stages:

```bash
python scripts/run_pipeline.py --load --classify
python scripts/run_pipeline.py --enrich --max-enrich 25
python scripts/run_pipeline.py --score --max-score 25
```

## How The Pipeline Works

`src/scrape.py` tries to scrape the live Manifest attendee page. If the live scrape returns too little data or fails, it loads `data/manifest_attendees_seed.txt`.

`src/clean.py` preserves the raw company name while creating a normalized key for deduplication. Legal suffixes like `Inc.`, `LLC`, and `Corporation` are removed only for matching.

`src/rules.py` applies cheap classification before any paid API call. It filters obvious non-targets such as large incumbents, investors, associations, universities, consultancies, generic placeholders, and logistics service providers without software or platform signals.

`src/enrich.py` calls Tavily with a compact company-search query and stores titles, URLs, snippets, website hints, and raw JSON in SQLite.

`src/classify.py` sends compact evidence to OpenAI. The model returns structured JSON with company type, startup signal, sector tags, Wittington edge, score components, confidence, rationale, and evidence summary.

`src/db.py` stores companies, enrichments, scores, and run metadata. The dashboard uses that run metadata for API-call counts, tokens, cache hits, and estimated spend.

## Billing And Usage Tracking

The dashboard separates provider-billed spend from internal estimates:

- Tavily search calls
- Tavily plan credits consumed
- Tavily free credits remaining
- Tavily actual billed spend, which is `$0.00` on the Researcher plan while pay-as-you-go is disabled
- OpenAI scoring calls
- OpenAI input tokens
- OpenAI output tokens
- OpenAI total tokens
- Live OpenAI billed cost from `GET /v1/organization/costs` when `OPENAI_ADMIN_KEY` is configured
- Live OpenAI completions usage from `GET /v1/organization/usage/completions` when `OPENAI_ADMIN_KEY` is configured
- Local OpenAI token estimate based on configured token prices
- Tavily shadow value for planning only, not reimbursement
- Streamlit Community Cloud hosting shown as `$0.00`

Provider-billed OpenAI cost is the source of truth for reimbursement when available. If live OpenAI billing is unavailable because `OPENAI_ADMIN_KEY` is missing or the API request fails, the app continues running and clearly labels the local token-based fallback as an estimate rather than an invoice. SQLite stores every run so reruns can show cumulative calls, tokens, cache hits, and local estimates.

`TAVILY_COST_PER_CALL_USD` remains supported as an internal shadow estimate for projections. It is not included in actual provider-billed spend unless Tavily pay-as-you-go is explicitly enabled, in which case overage credits beyond `TAVILY_INCLUDED_MONTHLY_CREDITS` are billed using `TAVILY_PAYG_PRICE_PER_CREDIT_USD`.

## Caching And Reset

- The database is created automatically at `data/prospects.db`.
- Company rows are keyed by normalized name.
- Tavily enrichments are keyed by company and provider.
- OpenAI scores are keyed by company and provider.
- Normal reruns reuse cached rows and continue with the next unprocessed candidate.
- The sidebar reset button clears local source rows, enrichments, scores, and run history when a clean slate is needed.

## Scoring

The weighted score is capped at 100:

- Venture backability: 25
- Wittington sector fit: 25
- Wittington strategic edge: 20
- Stage signal: 10
- Traction signal: 10
- Evidence confidence: 10

Caps prevent non-startups from ranking highly:

- Incumbents without strong startup evidence: max 35
- Investors: max 25
- Media, associations, universities, and government: max 20
- Plain service providers without software or platform evidence: max 45
- Rows without external evidence: max 50

## Deployment

### Railway

The repository includes a `Procfile`:

```bash
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Add the environment variables in Railway and deploy the repo.

### Streamlit Cloud

Create a Streamlit Cloud app from this repository, set `app.py` as the entrypoint, and add the API keys under app secrets or environment variables.

## Known Limitations

- Tavily search can return ambiguous results for short or common company names.
- Deterministic screening is conservative and may miss stealth startups with generic names.
- The seed attendee file is included for reliability, but the live scrape should be rerun before a demo.
- Funding stage is inferred from public snippets unless a richer company-data API is added.
- OpenAI scoring depends on retrieved evidence quality, so thin evidence is marked low confidence.

## Related Notes

- Key decision note: `SUBMISSION_NOTE.md`
- Future capabilities: `FUTURE_CAPABILITIES.md`
