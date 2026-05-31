# Manifest Prospecting Tool

Manifest Prospecting Tool is a small internal VC sourcing dashboard for Wittington Ventures. It turns the public Manifest attendee list into a cached, sortable, ranked pipeline of companies that may be worth investor attention.

The product thesis is simple: use code for retrieval and storage, use deterministic rules to avoid wasting paid calls on obvious non-prospects, then use an LLM only for compact VC-style judgment once external evidence has been retrieved and cached.

## What This Tool Does

- Scrapes `https://manife.st/who-attends/` and falls back to the included seed list if the live page changes or blocks scraping.
- Parses, cleans, normalizes, and deduplicates messy attendee names.
- Stores all companies, enrichments, scores, and run metadata in SQLite.
- Runs cheap deterministic classification across the full list before paid enrichment.
- Uses Tavily search as the external API enrichment layer for likely candidates.
- Uses OpenAI for structured scoring only after Tavily evidence exists.
- Presents a Streamlit dashboard with metrics, filters, source evidence, CSV export, and a sortable ranked table.

## Why This Architecture

The assignment emphasizes repeatability, cost control, and smart use of AI. The pipeline is therefore staged:

1. **Retrieve in code:** requests and BeautifulSoup fetch the attendee page; Tavily retrieves web evidence.
2. **Cache everything:** SQLite prevents repeated scraping, search, or scoring work on reruns.
3. **Filter before spend:** deterministic rules screen out incumbents, investors, media, universities, consultants, and plain service providers.
4. **Use AI for judgment:** OpenAI receives only company name, deterministic tags, and top search snippets, then returns structured JSON.
5. **Keep the UI simple:** Streamlit is enough for a reviewable internal workflow.

## Assignment Checklist

- Real working tool: `streamlit run app.py`
- Scrape and parse Manifest attendee list: `src/scrape.py`
- External API call: Tavily search in `src/enrich.py`
- Queryable data layer: SQLite schema in `src/db.py`
- Rerun cache: enrichments and scores are keyed by company/provider
- Deterministic fit logic: `src/rules.py`
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

Then use the sidebar in this order:

1. Scrape/load Manifest list
2. Run deterministic classification
3. Enrich candidates with Tavily
4. Score enriched candidates with OpenAI

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

`src/rules.py` applies cheap classification before any paid API call. It excludes obvious non-targets such as large incumbents, investors, associations, universities, consultancies, and logistics service providers without software/platform signals. It keeps likely startup/tech and ambiguous entries as candidates.

`src/enrich.py` calls Tavily with a compact company-search query and stores titles, URLs, snippets, and raw JSON in SQLite.

`src/classify.py` sends only compact evidence to OpenAI. The model returns structured JSON with company type, startup likelihood, sector tags, Wittington edge, score components, confidence, rationale, and evidence summary.

## Caching And Cost Control

- The database is created automatically at `data/prospects.db`.
- Company rows are keyed by normalized name.
- Tavily enrichments are keyed by company/provider.
- OpenAI scores are keyed by company and marked with provider `openai`.
- The UI defaults to bounded `MAX_ENRICH` and `MAX_SCORE` values.
- The deterministic baseline gives a ranked full-list view even before API keys are configured.
- A force-refresh checkbox exists, but normal reruns reuse cached rows.

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
