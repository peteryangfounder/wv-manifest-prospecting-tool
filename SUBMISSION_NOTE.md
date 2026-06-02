# Key Decisions

The Manifest Prospecting Tool converts the public Manifest attendee list into a ranked venture-prospect workflow for Wittington Ventures. The core decision was to treat the attendee list as a noisy sourcing universe, not as a finished prospect list. The app therefore separates source retrieval, cleaning, deterministic screening, paid enrichment, model scoring, review UI, and usage reporting into inspectable stages.

This staged design keeps the workflow useful for investors and controllable for operators. Source rows remain visible even when they are not investable prospects. Verified prospects are only created after the row has passed rule screening, received external evidence, and been scored against Wittington-specific criteria.

## Data Pipeline

`src/scrape.py` retrieves attendee names from the Manifest attendee page. A seed file is included so the app remains runnable if the live page changes, blocks scraping, or returns too little data. `src/clean.py` preserves the original company name for reviewer visibility and creates a normalized key for deduplication. That normalized key is used for database identity, caching, and reruns.

The deterministic rules layer in `src/rules.py` runs before any paid provider call. It filters clear non-targets such as large incumbents without startup signals, investors, universities, associations, consultancies, brands, placeholder rows, and logistics service providers without software or platform evidence. Ambiguous rows are not discarded too aggressively because a sparse conference record can hide a real early-stage company. Those rows are allowed into the enrichment queue when the cheaper rule pass cannot confidently exclude them.

The rules layer also creates a high-priority enrichment queue. That queue helps the app spend its first Tavily and OpenAI calls on companies most likely to produce useful venture prospects instead of processing the entire source universe in arbitrary list order.

## Funnel Math

The first-pass funnel is deliberately cost-aware. On the current live Manifest scrape, the app retrieved 3,288 raw attendee rows and normalized them to 3,239 unique companies. The deterministic rules marked 2,444 as broad candidates and 158 as high-priority paid-enrichment rows.

That distinction is important. The app does not claim that only 158 companies are possible prospects. It claims that 158 rows have enough explicit technology or Wittington-relevant signal in the attendee name to justify being first in line for paid API enrichment. The rest of the broad candidate universe remains stored, visible, and available for later enrichment. This keeps the demo inexpensive while preserving optionality.

The broad candidate count is high because the rules are conservative about exclusion. If a row is ambiguous and cannot be confidently identified as a non-target, it remains a candidate. The high-priority count is lower because that queue requires stronger positive signals such as `AI`, `.ai`, robotics, software, SaaS, platform, automation, analytics, visibility, autonomous systems, optimization, TMS, WMS, machine learning, computer vision, warehouse automation, retail infrastructure, healthcare operations, climate, sustainability, carbon, or emissions.

The cheap first pass uses deterministic string matching, curated keyword groups, normalized names, and explicit exclusion lists. It does not use an LLM and it does not perform hidden reasoning. That is a feature: the rules are transparent, fast, testable, and easy to challenge in a review meeting. If a partner disagrees with a rule, the rule can be changed and the full source universe can be rescored without re-running paid provider calls.

The risk is recall. A stealth startup with a generic name and no obvious technology signal may not enter the first paid queue. The mitigation is that it is not deleted. It remains in the source list and broad candidate set, with a baseline score and deterministic tags. A production version should add additional cheap recall layers, such as website-domain lookup, embeddings, company database enrichment, partner-selected batches, and first-party CRM signals, before deciding which lower-priority rows deserve paid research.

## Enrichment And Scoring

Tavily is used for public search evidence. The app stores compacted titles, URLs, snippets, website hints, raw JSON, and provider status in SQLite. OpenAI receives the compact evidence and returns structured JSON with company type, startup signal, sector tags, Wittington edge, score components, confidence, rationale, and evidence summary.

The scoring rubric reflects Wittington’s focus on commerce, health, consumer, and climate, with a primary investment range around Series A/B and flexibility outside that range. Each company is evaluated across venture backability, sector fit, strategic edge, stage signal, traction signal, and evidence confidence. Score caps keep non-startups, investors, universities, associations, media organizations, and plain service providers from ranking highly only because they are recognizable or adjacent to supply chain, retail, or logistics.

Baseline deterministic scores are stored for the full attendee list. Paid OpenAI scores improve the ranking where external evidence is available, while baseline scores keep the interface complete when API keys are missing or a provider call fails.

## Performance Architecture

The first implementation processed provider calls serially. That was acceptable for a small demo but not for a 3,000+ row attendee universe. The current pipeline uses `ThreadPoolExecutor` for Tavily enrichment and OpenAI scoring so network-bound provider requests can run concurrently.

`TAVILY_CONCURRENCY` and `OPENAI_CONCURRENCY` control the number of simultaneous provider requests. The defaults are intentionally modest: 12 Tavily workers and 6 OpenAI workers. This improves throughput while leaving room for provider rate limits and keeping local resource use predictable. Concurrency is bounded by the number of companies in the current run, so small batches do not create unnecessary workers.

SQLite writes stay on the main thread and are committed in batches through `DB_COMMIT_BATCH_SIZE`. This is an explicit tradeoff. Provider calls are parallel because they spend most of their time waiting on network I/O. Database writes are serialized because SQLite is a local file-backed database and predictable write ordering is more important than squeezing out marginal write throughput. The result is faster provider processing without introducing avoidable database contention.

Progress updates are throttled through Streamlit callbacks so the UI can show live completion counts, cache hits, errors, retries, and token estimates without spending excessive time rerendering the page.

## Provider Robustness

The provider layer handles common API failure modes directly. Tavily and OpenAI calls use bounded retries for transient conditions, including 408, 409, 425, 429, and 5xx responses. When a provider returns `Retry-After`, the app respects it. Otherwise it uses exponential backoff with jitter, controlled by `PROVIDER_MAX_RETRIES`, `PROVIDER_BACKOFF_INITIAL_SECONDS`, and `PROVIDER_BACKOFF_MAX_SECONDS`.

Terminal provider failures are recorded per company. A failed Tavily enrichment stores an error-status enrichment record. A failed OpenAI score falls back to the deterministic baseline score and records the failure context. This prevents one bad row, one temporary provider issue, or one rate-limit event from stopping the entire batch.

The retry counts, provider attempts, cache hits, API calls, and token usage are stored in run metadata. This gives the operator visibility into whether a run was fast because it was cached, slow because providers were rate-limited, or partially degraded because some provider calls failed.

## Cost And Billing Controls

The app separates live billing data from internal estimates. This is important because an investor-facing billing view should not confuse platform billing with a local approximation based on token rates.

OpenAI inference uses `OPENAI_API_KEY`. OpenAI billing reads use `OPENAI_ADMIN_KEY`. The admin key is only used to query OpenAI organization cost and usage endpoints. Those billing reads are administrative reads and are not counted as model or token spend in the app’s local estimates.

The primary OpenAI billing card reports Wittington project lifetime-to-date billed cost using `OPENAI_BILLING_PROJECT_ID=proj_ynS2F3GVOCBbgmXvTl9Vl1Ie` and `OPENAI_BILLING_START_DATE=2026-05-31`. The recent usage card keeps `OPENAI_BILLING_LOOKBACK_DAYS` for a shorter window, usually the last 30 days. Billing responses are paginated and cached for `OPENAI_BILLING_CACHE_TTL_SECONDS`, defaulting to 300 seconds, because Streamlit reruns frequently.

If live project-scoped OpenAI billing is unavailable, the app does not crash. It shows that live OpenAI billing is unavailable and falls back to the internal token-rate estimate with clear labeling. If OpenAI returns organization-level billing rather than project-scoped billing, the app labels that context separately and does not use organization-wide spend as the project billed-cost total.

Tavily billing is treated separately from OpenAI. On the current Researcher plan with `TAVILY_PAY_AS_YOU_GO_ENABLED=false`, Tavily billed spend is `$0.00` while the app still shows credits consumed and included credits remaining. `TAVILY_COST_PER_CALL_USD` remains as a planning estimate only. Streamlit Community Cloud hosting is shown as `$0.00` unless explicitly configured otherwise.

Before a verification run starts, the dashboard shows a confirmation step with projected uncached Tavily calls, projected OpenAI scoring calls, projected input and output tokens, internal OpenAI token-rate estimate, estimated Tavily billed cost, estimated total provider cost, expected remaining included Tavily credits, configured worker counts, and approximate runtime. No Tavily or OpenAI provider calls begin until the user confirms.

The recommended current Tavily setting is pay-as-you-go disabled. That keeps the demo cost-controlled and avoids automated overage billing. Pay-as-you-go should only be enabled if Wittington explicitly wants larger uncached runs to continue beyond included credits.

The near-zero current cost is explainable rather than mysterious. The app avoids paid calls on obvious non-prospects, begins with a 158-row high-priority queue instead of the full 3,239-company universe, reuses cached provider results, keeps Tavily within included plan credits, and uses compact OpenAI prompts with `gpt-4o-mini`. The dashboard can show `$0.00` live provider billing while the internal OpenAI token-rate estimate shows a fraction of a cent or a few cents because platform billing can round, cache, or lag behind local token accounting.

The right explanation to Wittington is not that the app made 3,000 nuanced investment decisions for free. The accurate explanation is that it performed a transparent, deterministic triage for free, then spent API calls only on the highest-signal rows. That is exactly the point of the architecture: use code for the cheap mechanical narrowing, use search and AI where judgment and evidence synthesis are actually valuable, and keep the full source universe available when more recall is needed.

## Interface Decisions

The Streamlit interface is organized around the work an investor or associate needs to do: prepare the source list, verify a selected batch, then review ranked results. The Overview tab shows source coverage, prospect status, charts, and API usage. The Source list keeps cleaned source rows visible. The Verified prospects tab presents ranked companies with company descriptions, sector tags, score components, confidence, rationale, and evidence. The Company detail tab supports deeper review of one company at a time.

The API usage section was deliberately simplified. It now uses concise labels such as all-time billed cost, OpenAI billed cost for the recent window, internal token-rate estimate, Tavily credits used, last run API calls, and last run token-rate estimate. Slash notation was removed where it could be mistaken for a rate or fraction. Long explanatory billing paragraphs were replaced with compact metadata rows for live billing source, OpenAI project, billing start, last fetched, cache status, Tavily plan, and Streamlit Cloud hosting.

The UI avoids framing local estimates as actual billing. The wording distinguishes live provider billing, platform analytics, included credits, and internal estimates in language that is understandable to investors rather than backend engineers.

## Tradeoffs

The app currently uses Streamlit and SQLite because they are simple, portable, and appropriate for a take-home demo and a lightweight internal tool. Streamlit gives a fast review interface without a separate frontend build. SQLite makes cached provider work durable without adding external infrastructure. The tradeoff is that long-running provider jobs are synchronous in the current app session. A production version should use a resumable background job queue for true pause, resume, cancellation, multi-user scheduling, and durable progress across sessions.

The app prioritizes caching and rule screening before paid APIs. This keeps costs low but means that companies with thin public signals may need manual review or richer data providers. The scoring model is evidence-bound by design: weak evidence lowers confidence rather than allowing the model to invent diligence signals.

The current provider architecture is optimized for network-bound I/O. It should scale materially better than serial processing for thousands of rows, but the correct concurrency settings still depend on provider rate limits, account tier, and acceptable spend. The app therefore exposes worker counts, estimated cost, and estimated runtime before execution instead of hiding those operational choices.

The biggest technical tradeoff is precision versus recall in the first API-backed pass. The current default is precision-first because the assignment asked for a useful ranked prospect list and because paid search/model calls should not be wasted on obvious non-prospects. For a production fund workflow, the next step would be configurable recall modes: a low-cost first pass, a broader partner-reviewed pass, and a full-universe pass with explicit spend approval.

## Validation

The repository includes pytest coverage for cleaning, deterministic rules, database migrations, cache reuse, priority queues, verified prospect view models, billing parsing and pagination, billing cache behavior, Tavily billing math, and concurrent enrichment/scoring persistence. The latest full test run passed with 36 tests.

The final product demonstrates a complete loop: retrieve and clean the source universe, cheaply screen it, enrich likely candidates with external evidence, score those candidates with structured AI output, cache paid work, show ranked investor-facing results, and report live billing and internal estimates without mixing the two.
