# Key Decisions

The Manifest Prospecting Tool converts the public Manifest attendee list into a ranked venture-prospect workflow for Wittington Ventures. The core decision was to treat the attendee list as a noisy sourcing universe, not as a finished prospect list. The app therefore separates source retrieval, raw-row storage, company normalization, deterministic screening, company-page evidence collection, paid search enrichment, model scoring, review UI, and usage reporting into inspectable stages.

This staged design keeps the workflow useful for investors and controllable for operators. Source rows remain visible even when they are not investable prospects. Verified prospects are only created after the row has passed rule screening, received source evidence from company-page metadata or Tavily search, and been scored against Wittington-specific criteria.

## Data Pipeline

`src/scrape.py` retrieves attendee names from the Manifest attendee page. A seed file is included so the app remains runnable if the live page changes, blocks scraping, or returns too little data. The current Streamlit flow stores raw Manifest rows first and normalizes them in a separate operation so reviewers can see the transition from attendee-company text to canonical company names. `src/clean.py` preserves the original company name for reviewer visibility and creates a normalized key for deduplication. That normalized key is used for database identity, caching, and reruns.

The deterministic rules layer in `src/rules.py` runs before any paid provider call. It filters clear non-targets such as large incumbents without startup signals, investors, universities, associations, consultancies, brands, placeholder rows, and logistics service providers without software or platform evidence. Ambiguous rows are not discarded too aggressively because a sparse conference record can hide a real early-stage company. Those rows are allowed into the enrichment queue when the cheaper rule pass cannot confidently exclude them.

The rules layer also creates a high-priority enrichment queue. That queue helps the app spend its first Tavily and OpenAI calls on companies most likely to produce useful venture prospects instead of processing the entire source universe in arbitrary list order.

## Funnel Math

The first-pass funnel is deliberately cost-aware. On the current live Manifest scrape, the app retrieved 3,288 raw attendee rows and normalized them to 3,239 unique companies. The deterministic rules marked 2,444 as broad candidates, 281 as likely startup or technology rows, and 158 as high-signal fast-lane rows.

That distinction is important. The app does not claim that only 158 companies are possible prospects. It claims that 158 rows have enough explicit technology or Wittington-relevant signal in the attendee name to justify running first. The paid API queue is now ranked rather than restricted: high-signal rows run first, other likely startup or technology rows run next, and ambiguous broad candidates run after that until the user-approved batch cap is reached. With a 1,000-company cap on a fresh scrape, the queue contains all 281 likely startup or technology rows plus 719 ambiguous candidates.

The broad candidate count is high because the rules are conservative about exclusion. If a row is ambiguous and cannot be confidently identified as a non-target, it remains a candidate. The high-signal count is lower because that first lane requires stronger positive signals such as `AI`, `.ai`, robotics, software, SaaS, platform, automation, analytics, visibility, autonomous systems, optimization, TMS, WMS, machine learning, computer vision, warehouse automation, retail infrastructure, healthcare operations, climate, sustainability, carbon, or emissions.

The cheap first pass uses deterministic string matching, curated keyword groups, normalized names, and explicit exclusion lists. It does not use an LLM and it does not perform hidden reasoning. That is a feature: the rules are transparent, fast, testable, and easy to challenge in a review meeting. If a partner disagrees with a rule, the rule can be changed and the full source universe can be rescored without re-running paid provider calls.

The exact deterministic decision flow is:

1. Normalize the raw name for matching while preserving the displayed company name.
2. Detect sector tags from name-level terms for commerce, healthcare, consumer, food, climate, logistics, supply chain, retail infrastructure, warehouse automation, robotics, AI, and fintech.
3. Remove generic placeholders and incomplete rows.
4. Remove Wittington-related rows.
5. Remove known large incumbents such as Amazon, Microsoft, DHL, FedEx, UPS, Walmart, Google, Oracle, IBM, Costco, Loblaw, and Target.
6. Remove investor and financial-firm rows based on venture, capital, private equity, investment, asset-management, bank, securities, accelerator, family-office, and related terms.
7. Remove media, event, association, university, government, nonprofit, consulting, agency, advisory, legal, and accounting-service rows.
8. Remove logistics-service rows only when they have logistics words but no software, platform, automation, AI, analytics, visibility, optimization, TMS, WMS, robotics, or similar technology signal.
9. Mark rows with technology signals as likely startup or technology candidates.
10. Remove retailer, brand, CPG, apparel, beauty, foodservice, or beverage rows when they have no technology signal.
11. Preserve all remaining ambiguous rows as broad candidates for later enrichment.

The risk is recall if the batch cap is set too low. A stealth startup with a generic name and no obvious technology signal may not appear in the first few hundred rows. The mitigation is that it is not deleted. It remains in the source list and broad candidate set, with a baseline score and deterministic tags, and it can enter a larger paid batch. A production version should add additional cheap recall layers, such as website-domain lookup, embeddings, company database enrichment, partner-selected batches, and first-party CRM signals, before deciding which lower-priority rows deserve paid research.

## Enrichment And Scoring

The current evidence path has three tiers. First, company-page metadata is collected from conservative domain guesses: titles, descriptions, OpenGraph fields, JSON-LD organization data, headings, and short bounded text snippets. Strong company-page evidence can create a `homepage` enrichment record, which lets the company proceed to OpenAI scoring without Tavily. Second, Tavily is used for public search evidence when the company page is unresolved, thin, ambiguous, blocked, or contradictory. The app stores compacted titles, URLs, snippets, website hints, raw JSON, and provider status in SQLite. Third, OpenAI receives compact source evidence and returns structured JSON with company type, startup signal, sector tags, Wittington edge, score components, confidence, rationale, and evidence summary.

The scoring rubric reflects Wittington’s focus on commerce, health, consumer, and climate, with a primary investment range around Series A/B and flexibility outside that range. Each company is evaluated across venture backability, sector fit, strategic edge, stage signal, traction signal, and evidence confidence. Score caps keep non-startups, investors, universities, associations, media organizations, and plain service providers from ranking highly only because they are recognizable or adjacent to supply chain, retail, or logistics.

Baseline deterministic scores are stored for the full attendee list. Paid OpenAI scores improve the ranking where external evidence is available, while baseline scores keep the interface complete when API keys are missing or a provider call fails.

## Performance Architecture

The first implementation processed provider calls serially. That was acceptable for a small demo but not for a 3,000+ row attendee universe. The current pipeline uses `ThreadPoolExecutor` for company-page checks, Tavily enrichment, and OpenAI scoring so network-bound work can run concurrently.

`HOMEPAGE_CONCURRENCY`, `TAVILY_CONCURRENCY`, and `OPENAI_CONCURRENCY` control the number of simultaneous network/provider requests. The current fast-demo defaults are 96 homepage workers, 48 Tavily workers, and 48 OpenAI workers. Each count is bounded by the number of companies in the current operation, so a 10-row batch does not create 48 unnecessary Tavily or OpenAI workers.

SQLite writes stay on the main thread and are committed in batches through `DB_COMMIT_BATCH_SIZE`. This is an explicit tradeoff. Provider calls are parallel because they spend most of their time waiting on network I/O. Database writes are serialized because SQLite is a local file-backed database and predictable write ordering is more important than squeezing out marginal write throughput. The result is faster provider processing without introducing avoidable database contention.

Progress updates are shown through Streamlit callbacks so the UI can show live completion counts, ETA, worker counts, cache hits, errors, retries, currently processed company names, and token estimates during the operation.

## Provider Robustness

The provider layer handles common API failure modes directly. Tavily and OpenAI calls use bounded retries for transient conditions, including 408, 409, 425, 429, and 5xx responses. When a provider returns `Retry-After`, the app respects it. Otherwise it uses exponential backoff with jitter, controlled by `PROVIDER_MAX_RETRIES`, `PROVIDER_BACKOFF_INITIAL_SECONDS`, and `PROVIDER_BACKOFF_MAX_SECONDS`.

Terminal provider failures are recorded per company. A failed Tavily enrichment stores an error-status enrichment record. A failed OpenAI score falls back to the deterministic baseline score and records the failure context. This prevents one bad row, one temporary provider issue, or one rate-limit event from stopping the entire batch.

The retry counts, provider attempts, cache hits, API calls, and token usage are stored in run metadata. This gives the operator visibility into whether a run was fast because it was cached, slow because providers were rate-limited, or partially degraded because some provider calls failed.

## Cost And Billing Controls

The app separates live billing data from internal estimates. This is important because an investor-facing billing view should not confuse platform billing with a local approximation based on token rates.

OpenAI inference uses `OPENAI_API_KEY`. OpenAI billing reads use `OPENAI_ADMIN_KEY`. The admin key is only used to query OpenAI organization cost and usage endpoints. Those billing reads are administrative reads and are not counted as model or token spend in the app’s local estimates.

The primary OpenAI billing card reports Wittington project lifetime-to-date billed cost using `OPENAI_BILLING_PROJECT_ID=proj_ynS2F3GVOCBbgmXvTl9Vl1Ie` and `OPENAI_BILLING_START_DATE=2026-05-31`. The recent usage card keeps `OPENAI_BILLING_LOOKBACK_DAYS` for a shorter window, usually the last 30 days. Billing responses are paginated and cached for `OPENAI_BILLING_CACHE_TTL_SECONDS`, defaulting to 300 seconds, because Streamlit reruns frequently.

If live project-scoped OpenAI billing is unavailable, the app does not crash. It shows that live OpenAI billing is unavailable and falls back to the internal token-rate estimate with clear labeling. If OpenAI returns organization-level billing rather than project-scoped billing, the app labels that context separately and does not use organization-wide spend as the project billed-cost total.

Tavily billing is treated separately from OpenAI. The app now reads the live Tavily usage API when a Tavily key is configured, caches the snapshot to avoid repeated usage calls on Streamlit reruns, and falls back to local run records when live usage is unavailable. `TAVILY_COST_PER_CALL_USD` remains a pay-as-you-go-equivalent planning estimate. Actual Tavily billed spend is `$0.00` while pay-as-you-go is disabled, and is calculated from overage credits when pay-as-you-go is enabled. Streamlit Community Cloud hosting is shown as `$0.00` unless explicitly configured otherwise.

Before paid work starts, the dashboard shows operation-specific setup metrics with the selected batch size, API-eligible universe, domain-discovery candidates, cached resolved domains, homepage evidence-ready companies, Tavily-needed companies, Tavily calls skipped by homepage evidence, homepage data gaps, projected uncached Tavily calls, projected OpenAI scoring calls, projected input and output tokens, internal OpenAI token-rate estimate, estimated Tavily billed cost, estimated total provider cost, expected remaining included Tavily credits, configured worker counts, and approximate runtime. No Tavily or OpenAI provider calls begin until the operator starts the corresponding web-search or scoring operation.

For a cost-contained demo, Tavily pay-as-you-go can remain disabled. For a full broad recall run, pay-as-you-go can be enabled intentionally; the app then separates live credits, pay-as-you-go overage, and planning value so the operator can explain exactly why billed cost and search-call value may differ.

The near-zero current cost is explainable rather than mysterious. The app avoids paid calls on obvious non-prospects, ranks the 2,444 broad candidates before enrichment, reuses cached provider results, keeps Tavily within included plan credits, and uses compact OpenAI prompts with `gpt-4o-mini`. The dashboard can show `$0.00` live provider billing while the internal OpenAI token-rate estimate shows a fraction of a cent or a few cents because platform billing can round, cache, or lag behind local token accounting.

The right explanation to Wittington is not that the app made 3,000 nuanced investment decisions for free. The accurate explanation is that it performed a transparent, deterministic triage for free, then spent API calls on a ranked queue inside the approved budget. That is exactly the point of the architecture: use code for cheap mechanical exclusion, use search and AI where judgment and evidence synthesis are actually valuable, and keep the full source universe available when more recall is needed.

The reason this is not just "keyword filtering" is that name-level rules only decide two things: obvious non-target exclusion and queue order. They do not decide final investability. A company with no technology word in its name can still remain in `unknown_needs_enrichment` and receive Tavily/OpenAI analysis when the batch cap reaches that part of the queue. The actual evidence-backed assessment happens after Tavily retrieves web snippets and OpenAI scores the company from that compact external evidence.

Under the current pricing assumptions, a broad all-candidate test run is not automatically expensive. The live Manifest scrape has 2,444 API-eligible candidates. If Wittington explicitly enabled Tavily pay-as-you-go for a full broad pass, the first 1,000 Researcher-plan credits are included and the remaining 1,444 credits at `$0.008` would be about `$11.55`. The OpenAI side uses compact `gpt-4o-mini` prompts, so the default cold-start estimate of roughly 1,000 input tokens and 250 output tokens per company implies well under `$1` for 2,444 companies at the configured token rates. Actual token usage can differ, so the app computes and displays the run-specific projection before execution.

## Interface Decisions

The Streamlit interface is organized like a short onboarding deck rather than a dashboard. Each screen teaches one idea: overview, raw source load, company-name normalization, exclusion rules, company-page checks, Tavily web search, OpenAI scoring, cost detail, and scored-company review. There is no sidebar, no tabbed dashboard, and no hidden expanders in the main workflow. The investor moves with Previous and Next, sees progress at the top, and only encounters action buttons when a real workflow action is available.

The latest interface iteration also makes state transitions explicit. The operation slides now show before/after panels so the reviewer can see what data exists before an action, what will happen next, and what changed after the action completes. The global cache reset moved into a compact header control and clears raw rows, company rows, company-page data, Tavily results, and OpenAI scores while preserving run history for cost records.

The company-page, web-search, and scoring steps are intentionally separate. That separation makes the demo easier to explain: page checks are near-free and can avoid Tavily, Tavily only runs for rows still needing source data, and OpenAI scoring only runs after company-page or Tavily evidence is ready. It also lets the UI display a focused ETA and worker count for each operation instead of hiding everything behind one large paid-run button.

The API usage section was deliberately simplified. It now uses literal labels such as OpenAI API + Tavily Search API + Streamlit Cloud hosting billed cost, OpenAI API billed cost for the recent window, OpenAI API token-cost estimate from local token counts, Tavily Search API credits used, last run API calls, and last run token-rate estimate. Slash notation was removed where it could be mistaken for a rate or fraction. Long explanatory billing paragraphs were replaced with compact metadata rows for live billing source, OpenAI project, billing start, last fetched, OpenAI cache status, Tavily usage source, Tavily last fetched, Tavily plan, and Streamlit Cloud hosting.

The UI avoids framing local estimates as actual billing. The wording distinguishes live provider billing, platform analytics, included credits, and internal estimates in language that is understandable to investors rather than backend engineers.

## Tradeoffs

The app currently uses Streamlit and SQLite because they are simple, portable, and appropriate for a take-home demo and a lightweight internal tool. Streamlit gives a fast review interface without a separate frontend build. SQLite makes cached provider work durable without adding external infrastructure. The tradeoff is that long-running provider jobs are synchronous in the current app session. A production version should use a resumable background job queue for true pause, resume, cancellation, multi-user scheduling, and durable progress across sessions.

The app prioritizes caching and rule screening before paid APIs. This keeps costs low but means that companies with thin public signals may need manual review or richer data providers. The scoring model is evidence-bound by design: weak evidence lowers confidence rather than allowing the model to invent diligence signals.

The current provider architecture is optimized for network-bound I/O. It should scale materially better than serial processing for thousands of rows, but the correct concurrency settings still depend on website responsiveness, provider rate limits, account tier, and acceptable spend. The app therefore shows estimated cost, worker count, and runtime before execution instead of hiding the operational impact.

The biggest technical tradeoff is precision versus recall in the first API-backed pass. The current default is ranked rather than exclusive: high-signal rows run first for precision, homepage-positive ambiguous rows can be promoted based on evidence, and unresolved ambiguous candidates remain eligible within broader approved batches for recall. For a production fund workflow, the same pattern can become a low-cost first pass, a broader partner-reviewed pass, and a full-universe pass with explicit spend approval.

The current code supports precision-first, balanced, and recall-first verification postures. The investor-facing demo keeps the default posture selected so the workflow stays focused, but the underlying queue design still makes the precision/recall tradeoff explicit in code.

The strongest engineering step from the follow-up architecture review is now partially implemented: domain discovery and homepage metadata extraction run before Tavily. The app cheaply fetches titles, meta descriptions, OpenGraph descriptions, JSON-LD organization data, headings, and bounded homepage snippets for API-eligible candidates. That creates business-description evidence for companies with non-obvious names, lets the app rank ambiguous companies more intelligently, and reduces how often Tavily/OpenAI need to be used as first evidence sources.

The first bounded version of that evidence cascade is now implemented. It generates conservative domain candidates, fetches homepage metadata with size and timeout limits, extracts business-description evidence, scores domain confidence, and stores route decisions in SQLite. Strong positive homepage evidence can create a `homepage` enrichment record that is eligible for OpenAI scoring without Tavily. Weak, blocked, unresolved, or contradictory homepage evidence routes to Tavily or data-gap handling instead of causing hard exclusion.

The evidence cascade is now investor-visible. The run estimate and main pipeline screen show API-eligible companies, homepage/domain attempts, homepage-positive rows, data gaps, Tavily calls skipped, Tavily calls still required, and estimated paid calls avoided.

Ranked prospect cards show whether the supporting evidence came from homepage metadata, Tavily, or both. They expose source URLs, compact evidence snippets, positive and negative signals, evidence confidence, and uncertainty or data-gap reasons. This prevents the app from presenting only a polished AI summary without showing the evidence behind it.

For the live demo, the app is intentionally focused on the core investor workflow instead of developer controls. The visible surface is a slide-by-slide walkthrough of the pipeline, operation setup, cost and usage, ranked prospects, and evidence-routing counts. The intended narrative is simple: the tool starts with the Manifest list; normalizes attendee-company rows into canonical companies; removes obvious non-prospects; gathers cheap company-page evidence first; escalates to Tavily only when evidence is missing or unclear; sends compact evidence packets to OpenAI for scoring; and reports actual billed cost, tokens, search credits, usage-source status, and paid calls avoided.

## Validation

The repository includes pytest coverage for cleaning, deterministic rules, database migrations, cache reuse, priority queues, homepage evidence extraction and routing, homepage-route summaries, paid-call avoidance counts, route example generation, audit sample generation, ranked-result evidence source selection, OpenAI payload evidence fields, verified prospect view models, billing parsing and pagination, billing cache behavior, live Tavily usage parsing and fallback behavior, Tavily billing math, and concurrent homepage/enrichment/scoring persistence. The latest full test run passed with 55 tests.

The final product demonstrates a complete loop: retrieve and clean the source universe, cheaply screen it, enrich likely candidates with external evidence, score those candidates with structured AI output, cache paid work, show ranked investor-facing results, and report live billing and internal estimates without mixing the two.
