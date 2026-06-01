# Manifest Prospecting Tool

Manifest Prospecting Tool is a small internal VC sourcing dashboard for Wittington Ventures. It turns the public Manifest attendee list into a cached, sortable, ranked pipeline of companies that may be worth investor attention.

The product thesis is simple: use code for retrieval and storage, use deterministic rules to avoid wasting paid calls on obvious non-prospects, then use an LLM only for compact VC-style judgment once external evidence has been retrieved and cached.

## What This Tool Does

- Scrapes `https://manife.st/who-attends/` and falls back to the included seed list if the live page changes or blocks scraping.
- Parses, cleans, normalizes, and deduplicates messy attendee names.
- Stores all companies, enrichments, scores, and run metadata in SQLite.
- Runs cheap deterministic classification across the full list before paid enrichment.
- Builds a stricter high-priority enrichment queue so paid API calls are not spent on every broad candidate.
- Uses Tavily search as the external API enrichment layer for likely candidates.
- Uses OpenAI for structured scoring only after Tavily evidence exists.
- Separates baseline screening from verified prospects so un-enriched keyword matches are not presented as final investment-quality leads.
- Presents a focused Streamlit dashboard with verified prospects, readable analytics, CSV export, and the full ranked table.

## Why This Architecture

The assignment emphasizes repeatability, cost control, and smart use of AI. The pipeline is therefore staged:

1. **Retrieve in code:** requests and BeautifulSoup fetch the attendee page; Tavily retrieves web evidence.
2. **Cache everything:** SQLite prevents repeated scraping, search, or scoring work on reruns.
3. **Filter before spend:** deterministic rules rank the full list, preserve a broad candidate count, and send paid enrichment only to a stricter high-priority queue.
4. **Use AI for judgment:** OpenAI receives only company name, deterministic tags, and top search snippets, then returns structured JSON.
5. **Keep the UI simple:** Streamlit is enough for a reviewable internal workflow.

## Assignment Checklist

- Real working tool: `streamlit run app.py`
- Scrape and parse Manifest attendee list: `src/scrape.py`
- External API call: Tavily search in `src/enrich.py`
- Queryable data layer: SQLite schema in `src/db.py`
- Rerun cache: enrichments and scores are keyed by company/provider
- Deterministic fit logic: `src/rules.py`
- Paid-call cost control: Tavily/OpenAI default to high-priority companies only
- AI synthesis and scoring: `src/classify.py`
- Wittington-specific scoring: encoded in `src/config.py` and scoring prompt
- Sortable ranked view: Streamlit dataframe in `app.py`
- Cost controls: max enrichment/scoring limits, cache checks, baseline dry mode
- Future capabilities write-up: `FUTURE_CAPABILITIES.md`
- Key decision note: `SUBMISSION_NOTE.md`

## Tech Stack

- Python
- Streamlit
- SQLite
- requests + BeautifulSoup
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
TAVILY_API_KEY=your_tavily_key_here
OPENAI_MODEL=gpt-4o-mini
MAX_ENRICH=75
MAX_SCORE=75
TAVILY_MAX_RESULTS=3
DATABASE_PATH=data/prospects.db
```

The app still opens without API keys and will produce deterministic baseline scores. To satisfy the external enrichment requirement in a real run, set `TAVILY_API_KEY`. To produce LLM-ranked scores, also set `OPENAI_API_KEY`.

## Run Locally

```bash
streamlit run app.py
```

Then use the sidebar's three-step workflow:

1. **Load & classify companies** - scrapes/parses the Manifest list and builds the high-priority queue with no paid API calls.
2. **Generate verified prospects** - runs Tavily enrichment and OpenAI scoring for the high-priority queue using the configured limits.
3. **Verify cache reuse** - proves cached Tavily/OpenAI records are reused without repeat paid calls.

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

`src/scrape.py` tries to scrape the live Manifest attendee page. If the live scrape returns too little data or fails, it loads `data/manifest_attendees_seed.txt`, which was created from the provided assignment attachment.

`src/clean.py` preserves the raw company name while creating a normalized key for deduplication. Legal suffixes like `Inc.`, `LLC`, and `Corporation` are removed only for dedupe.

`src/rules.py` applies cheap classification before any paid API call. It excludes obvious non-targets such as large incumbents, investors, associations, universities, consultancies, pure carriers, ports, trucking firms, and logistics service providers without software/platform signals. It also excludes generic placeholders and attendee-role entries such as `AI Startup`, `Startup`, `Stealth Company`, `TBD`, `Student`, `CEO`, `COO`, `Founder`, `N/A`, and `Unknown`. It still ranks the full attendee list and preserves a broad candidate count, but paid enrichment defaults to a much stricter high-priority queue.

The high-priority queue requires stronger startup or technology evidence: AI, robotics, SaaS, software, automation, platform, analytics, visibility, autonomous systems, optimization, warehouse automation, standalone WMS/TMS signals, retail infrastructure, healthcare operations, climate/sustainability, supply-chain technology, or startup-like evidence such as `.ai` branding. This keeps the workflow cost-aware while still allowing the dashboard to display the whole universe.

`src/enrich.py` calls Tavily with a compact company-search query and stores titles, URLs, snippets, and raw JSON in SQLite.

`src/classify.py` sends only compact evidence to OpenAI. The model returns structured JSON with company type, startup likelihood, sector tags, Wittington edge, score components, confidence, rationale, and evidence summary.

## Baseline Versus Verified Prospects

The app intentionally distinguishes the full-list baseline screen from evidence-backed prospect ranking. Baseline-only rows remain visible in the ranked pipeline as `Baseline only` / `Not cached`, and their displayed score is capped so keyword-only matches do not outrank companies with real evidence.

The `Verified prospects` table only shows companies with cached Tavily enrichment and cached OpenAI scoring. If no such rows exist yet, the app shows an empty state asking the reviewer to run enrichment and scoring rather than presenting low-confidence baseline rows as investment-ready leads.

## Caching And Cost Control

- The database is created automatically at `data/prospects.db`.
- Company rows are keyed by normalized name.
- Tavily enrichments are keyed by company/provider.
- OpenAI scores are keyed by company and marked with provider `openai`.
- Paid Tavily/OpenAI runs default to the high-priority queue, not the broader candidate pool.
- The UI defaults to bounded `MAX_ENRICH` and `MAX_SCORE` values from environment configuration.
- The deterministic baseline gives a full-list screening view even before API keys are configured, but the verified-prospect list is reserved for evidence-backed Tavily/OpenAI results.
- Normal UI reruns use cached rows; force-refresh behavior is reserved for code/CLI use.

### Reviewer Cache Verification

To prove reruns do not redo paid work:

1. Set small `MAX_ENRICH` and `MAX_SCORE` values if you want a tiny run.
2. Click **Generate verified prospects** after loading/classifying companies.
3. Click **Verify cache reuse** in the sidebar.
4. Confirm the success message says the stored enrichment and scoring were reused and that the dashboard shows `Cache hits` increased while `Last API calls` remains `0`.

The normal enrichment and scoring buttons still process the next unprocessed candidate. The verification button intentionally picks one company that already has both cached Tavily and OpenAI records, reruns that exact company's enrichment/scoring path with `force_refresh=False`, and records the result in the `runs` table.

## Scoring

The score is capped at 100:

- Venture-backability: 25
- Wittington sector fit: 25
- Wittington strategic edge: 20
- Stage signal: 10
- Traction signal: 10
- Data confidence: 10

Caps prevent non-startups from floating to the top:

- Incumbents without strong startup evidence: max 35
- Investors: max 25
- Media, associations, universities, government: max 20
- Plain service providers without software/platform evidence: max 45
- No external evidence: max 50

## Deployment

### Railway

The repository includes a `Procfile`:

```bash
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Add the environment variables in Railway and deploy the repo.

### Streamlit Cloud

Create a Streamlit Cloud app from this repository, set `app.py` as the entrypoint, and add the API keys under app secrets/environment variables.

## Known Limitations

- Tavily search can return ambiguous results for short or common company names.
- Deterministic screening is intentionally conservative and may miss stealthy startups with generic names.
- The seed attendee file is included for reliability, but the live scrape should be rerun before an interview demo.
- Funding stage is inferred from public snippets unless a richer company-data API is added.
- OpenAI scoring is only as good as the retrieved evidence; thin evidence is marked low confidence.

## Future Improvements

See `FUTURE_CAPABILITIES.md`.
