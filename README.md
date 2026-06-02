# Manifest Prospecting Tool

Manifest Prospecting Tool is an internal VC sourcing dashboard for Wittington Ventures. It turns the public Manifest attendee list into two clear working views: a source list with cleaned attendee rows and a verified prospect list backed by external evidence.

The product uses code for retrieval, storage, filtering, concurrency, retry handling, and cost control. It uses Tavily for external search evidence, then uses OpenAI for scoring after that evidence has been retrieved and cached.

Hosted app: https://wv-manifest-prospecting-tool-b8jadagmhgsh8wirbknjb9.streamlit.app/

## Current App

- Guided Streamlit workflow with three steps: prepare source list, verify prospects, review results.
- Responsive source cards for messy attendee data after cleaning, dedupe, and rule screening.
- Verified prospects view for companies that passed evidence enrichment and API scoring.
- Larger full-width charts with horizontal labels for readability.
- Responsive prospect cards so company descriptions and sector tags stay readable across desktop, tablet, and mobile.
- Prominent API usage and cost summary that separates live provider billing, included Tavily credits, Streamlit Cloud hosting, and internal token-rate estimates.
- Wittington project lifetime-to-date OpenAI billing, recent OpenAI billing, last fetch time, cache status, and billing-window metadata.
- Model selector and batch cap for controlling how many companies get verified in each run, with pre-run cost and runtime confirmation before paid provider calls begin.
- Concurrent Tavily enrichment and OpenAI scoring with bounded retry/backoff for rate limits and transient provider errors.
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
TAVILY_CONCURRENCY=12
OPENAI_CONCURRENCY=6
DB_COMMIT_BATCH_SIZE=25
PROVIDER_MAX_RETRIES=4
PROVIDER_BACKOFF_INITIAL_SECONDS=1.0
PROVIDER_BACKOFF_MAX_SECONDS=20.0
TAVILY_MAX_RESULTS=3
DATABASE_PATH=data/prospects.db
OPENAI_INPUT_COST_PER_1M_TOKENS=0.15
OPENAI_OUTPUT_COST_PER_1M_TOKENS=0.60
TAVILY_COST_PER_CALL_USD=0.001
OPENAI_BILLING_PROJECT_ID=proj_ynS2F3GVOCBbgmXvTl9Vl1Ie
OPENAI_BILLING_START_DATE=2026-05-31
OPENAI_BILLING_LOOKBACK_DAYS=30
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
4. Review the projected Tavily calls, OpenAI calls, token-rate estimate, Tavily billed cost, total estimated provider cost, worker counts, and estimated runtime.
5. Click **Confirm and start API run** if the estimate is acceptable.
6. Review the Overview, Source list, Verified prospects, and Company detail tabs.

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

`src/rules.py` applies cheap classification before any paid API call. It filters obvious non-targets such as large incumbents, investors, associations, universities, consultancies, generic placeholders, and logistics service providers without software or platform signals. It also creates a high-priority enrichment queue so paid provider calls start with the rows most likely to produce investor-relevant prospects.

`src/enrich.py` calls Tavily with a compact company-search query and stores titles, URLs, snippets, website hints, and raw JSON in SQLite.

`src/classify.py` sends compact evidence to OpenAI. The model returns structured JSON with company type, startup signal, sector tags, Wittington edge, score components, confidence, rationale, and evidence summary.

`src/db.py` stores companies, enrichments, scores, and run metadata. The dashboard uses that run metadata for API-call counts, tokens, cache hits, and estimated spend.

The enrichment and scoring stages run provider requests concurrently while keeping SQLite writes on the main thread. `TAVILY_CONCURRENCY` and `OPENAI_CONCURRENCY` control the number of simultaneous provider requests. This turns the slowest parts of the workflow from one-company-at-a-time waiting into parallel I/O while preserving deterministic database writes. The worker counts improve throughput but do not change the number of provider calls; the dashboard batch size, high-priority queue, and cache reuse remain the primary cost controls. `DB_COMMIT_BATCH_SIZE` controls how often completed results are committed during a run.

Before a verification run starts, the dashboard shows a confirmation step with projected uncached Tavily calls, projected OpenAI scoring calls, estimated token usage, estimated provider cost, and approximate runtime. No Tavily or OpenAI provider calls are made until the user confirms that estimate.

Provider calls use bounded retries with exponential backoff and jitter for transient errors such as 408, 409, 425, 429, and 5xx responses. When a provider includes `Retry-After`, the app uses it. `PROVIDER_MAX_RETRIES`, `PROVIDER_BACKOFF_INITIAL_SECONDS`, and `PROVIDER_BACKOFF_MAX_SECONDS` control that behavior. Non-retryable provider failures are recorded per company so a single bad row does not stop the full batch.

## Source Funnel

The app does not send all Manifest rows directly to paid APIs. It uses a staged funnel so cheap, deterministic work happens first and paid provider calls are reserved for rows where they are most likely to matter.

The current live scrape produced this first-pass funnel:

- 3,288 raw attendee rows from the Manifest page.
- 3,239 unique companies after normalization and deduplication.
- 2,444 broad candidates after excluding obvious non-targets.
- 281 likely startup or technology rows based on explicit name signals.
- 158 high-signal rows that run first inside the paid enrichment queue.

The 158 high-signal rows are no longer the full paid API boundary. They are the first lane in a ranked queue. The app now enriches high-signal rows first, then continues into other likely-technology rows, then into ambiguous broad candidates until the user-approved batch cap is reached. With a 1,000-company cap on a fresh scrape, the queue contains all 281 likely startup/technology rows plus 719 ambiguous candidates. The broader 2,444-candidate universe remains in SQLite and can be enriched with a larger budget, looser filters, richer data providers, or human-selected batches.

The deterministic rules are intentionally inspectable. They make only hard, explainable decisions before paid enrichment:

1. Normalize the raw name for matching while preserving the displayed company name.
2. Tag obvious sector words such as commerce, healthcare, consumer, food, climate, logistics, supply chain, retail infrastructure, warehouse automation, robotics, AI, and fintech.
3. Exclude generic or incomplete entries such as `startup`, `stealth`, `student`, `none`, `unknown`, and similar placeholders.
4. Exclude Wittington-related rows so the fund is not scored as its own prospect.
5. Exclude known large incumbents such as Amazon, Microsoft, DHL, FedEx, UPS, Walmart, Google, Oracle, IBM, Costco, Loblaw, Target, and similar non-venture prospects.
6. Exclude investor and financial-firm rows using terms such as venture, capital, private equity, investment, asset management, wealth, bank, securities, partners, accelerator, and family office.
7. Exclude media, event, association, university, government, nonprofit, consulting, agency, advisory, legal, and accounting-service rows.
8. Exclude logistics-service rows only when they have logistics words but no software, platform, automation, AI, analytics, visibility, optimization, TMS, WMS, robotics, or similar technology signal.
9. Include rows with explicit technology signals as likely startup or technology candidates.
10. Exclude retailer, brand, CPG, apparel, beauty, foodservice, or beverage rows when they have no technology signal.
11. Keep everything else as `unknown_needs_enrichment`, which means ambiguous rows are preserved for later paid enrichment instead of being thrown away.

Rows with clear technology or Wittington-relevant signals run first. Examples of fast-lane signals include `AI`, `.ai`, robotics, software, SaaS, platform, automation, analytics, visibility, autonomous systems, optimization, TMS, WMS, machine learning, computer vision, warehouse automation, retail infrastructure, healthcare operations, climate, sustainability, carbon, and emissions.

This is why the demo can be cost-effective without relying on a brittle keyword-only boundary. The deterministic name pass is used only for hard exclusions and ordering. It is not the final investment judgment. The app avoids paying Tavily and OpenAI to inspect obvious non-prospects, starts with rows most likely to contain technology companies, then keeps moving into ambiguous candidates within the approved batch cap. Tavily provides low-cost external web evidence, and OpenAI scores the compact evidence rather than the name alone. The app reuses cached provider results and uses compact prompts with a low-cost OpenAI model. This is a ranked cost-control architecture, not an assertion that company names alone are enough to identify every investable startup.

At the current configured prices, a broad pass over all 2,444 API-eligible candidates is still designed to be plausible under a small testing budget when Tavily pay-as-you-go is explicitly enabled: the first 1,000 Tavily credits are included on the Researcher plan, 1,444 additional credits at `$0.008` would be about `$11.55`, and the default OpenAI token-rate estimate for compact `gpt-4o-mini` scoring is typically well below the remaining budget. The confirmation screen computes the actual projected calls, candidate mix, tokens, Tavily overage, pay-as-you-go status, and estimated provider cost before any paid provider calls begin.

## Billing And Usage Tracking

The **API usage and cost** section separates provider-billed spend from internal estimates:

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
- Wittington project lifetime-to-date billing from `OPENAI_BILLING_START_DATE` through the current time
- Recent billing from `OPENAI_BILLING_LOOKBACK_DAYS`
- Internal token-rate estimate based on configured token prices
- Tavily shadow value for planning only
- Streamlit Community Cloud hosting shown as `$0.00`

Provider-billed OpenAI cost for the configured `OPENAI_BILLING_PROJECT_ID` is the source of truth for live billing when available. The primary dashboard total uses `OPENAI_BILLING_START_DATE` through the current time for the Wittington project lifetime-to-date window. The recent-cost card keeps `OPENAI_BILLING_LOOKBACK_DAYS` for a shorter usage view. OpenAI billing responses are paginated and cached for `OPENAI_BILLING_CACHE_TTL_SECONDS` because Streamlit reruns frequently. If project-scoped billing is unavailable and the OpenAI API returns organization-level fallback data, the app labels it as org-wide context and keeps it separate from the project billed-cost total. If live OpenAI billing is unavailable because `OPENAI_ADMIN_KEY` is missing or the API request fails, the app continues running and clearly labels the internal token-rate fallback as an estimate rather than platform billing data. SQLite stores every run so reruns can show cumulative calls, tokens, cache hits, retry counts, and local estimates.

`TAVILY_COST_PER_CALL_USD` remains supported as an internal shadow estimate for projections. It is not included in actual provider-billed spend unless Tavily pay-as-you-go is explicitly enabled, in which case overage credits beyond `TAVILY_INCLUDED_MONTHLY_CREDITS` are billed using `TAVILY_PAYG_PRICE_PER_CREDIT_USD`.

For the current demo configuration, Tavily pay-as-you-go should remain disabled unless Wittington explicitly wants automated overage billing. With pay-as-you-go disabled, the app can still report credits consumed and remaining included credits while keeping Tavily billed spend at `$0.00`.

In the current hosted demo, the visible cost can round to `$0.00` for three separate reasons. First, Tavily pay-as-you-go is disabled, so included Researcher-plan credits are tracked as consumed credits rather than billed spend. Second, the OpenAI scoring work uses `gpt-4o-mini` with compact prompts, so observed local token-rate estimates can be less than one cent for small scored batches. Third, live OpenAI billing can be delayed, cached, or rounded in platform reporting. The dashboard therefore shows both live provider billing and the internal token-rate estimate instead of treating either view as a substitute for the other.

## Performance And Cost Controls

The app is designed to scale from a small demo batch to thousands of attendee rows without making cost or latency invisible:

- Rule screening and baseline scoring run before paid APIs.
- Cached Tavily enrichments and OpenAI scores are reused on reruns.
- The high-priority queue sends likely venture prospects to paid enrichment before lower-fit rows.
- Tavily and OpenAI provider calls run in parallel with configurable worker counts.
- SQLite writes are serialized and batched to avoid thread contention.
- Transient provider failures use bounded retry/backoff with jitter.
- Terminal provider failures fall back to recorded error state or baseline score instead of stopping the run.
- The confirmation step estimates uncached calls, model tokens, OpenAI token-rate cost, Tavily billed cost, total provider cost, worker counts, credits after the run, and approximate runtime before provider calls start.
- Broad recall runs can include ambiguous companies after high-signal rows; the confirmation card shows the candidate mix before execution.
- Live billing reads through `OPENAI_ADMIN_KEY` are administrative reads and are not counted as model/token spend.

The current Streamlit implementation runs verification synchronously after confirmation. A production deployment should move long runs into a resumable background job queue if pause, resume, cancellation, or multi-user scheduling are required.

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
- The first API pass is ranked for cost control. Ambiguous companies are included after high-signal rows, but a full recall pass still requires a larger approved batch cap or richer data sources.
- Long provider runs are synchronous in the current Streamlit app; a background worker architecture would be needed for true pause/resume/cancel across sessions.

## Related Notes

- Key decision note: `SUBMISSION_NOTE.md`
- Future capabilities: `FUTURE_CAPABILITIES.md`
