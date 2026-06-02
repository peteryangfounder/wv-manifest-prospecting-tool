# Future Capabilities

The current app turns the Manifest attendee list into a working Wittington Ventures sourcing workflow with raw-row loading, company normalization, deterministic exclusions, company-page checks, Tavily search for unresolved rows, OpenAI scoring, live billing/usage reporting, and scored-company review. The larger opportunity is to make the same workflow apply to any company universe: conference lists, CRM exports, inbound pitch lists, accelerator cohorts, portfolio ecosystem maps, partner referrals, sector scans, and first-party Wittington datasets.

## Generalize Beyond Manifest

A production version should support arbitrary source lists instead of being tied to Manifest. The app should accept CSV uploads, pasted lists, saved database imports, Airtable or Notion tables, CRM exports, partner referral lists, conference attendee files, web-scraped exhibitor lists, and manually curated thesis lists. Each list should become a named sourcing campaign with its own source metadata, filters, scoring weights, billing window, review state, and export history.

The data model should treat Manifest as one source adapter. Additional adapters could cover events, company databases, accelerators, founder communities, job boards, app stores, ecommerce platforms, marketplaces, government registries, news feeds, import-export datasets, and partner-submitted lists. The shared pipeline would remain the same: normalize entities, dedupe companies, screen cheaply, enrich where useful, score against the selected thesis, cache provider work, and present an investor-readable ranking.

For arbitrary datasets, the app should support column mapping. A user should be able to map source columns such as company name, website, country, category, description, employee count, funding stage, source URL, contact name, email, and notes. When a row already includes useful first-party or third-party fields, the pipeline should use them instead of paying to rediscover the same facts.

The source funnel should become configurable by campaign. The current Manifest demo already has precision-first, balanced, and recall-first verification modes in code, plus a ranked candidate queue so obvious technology companies are enriched first and ambiguous candidates remain eligible after them. The interview demo keeps the visible workflow deliberately simple, but a production version should expose those mode settings as reusable campaign policies, including budget caps, recall posture, required evidence depth, default batch sizes, pay-as-you-go policy, and false-negative audit size.

## First-Party Wittington Data

Wittington’s own data would improve ranking quality more than more public search alone. CRM records can show prior contact, ownership, status, pass reasons, round timing, thesis association, partner interest, and relationship history. Meeting notes can capture founder quality, buyer pain, implementation risk, customer references, pricing concerns, technical credibility, and partner reactions.

Email and calendar metadata can add relationship recency and relationship strength without requiring broad summarization of private correspondence. Portfolio data can support lookalike scoring, conflict detection, strategic fit, commercial-introduction potential, and buyer relevance. Past investment outcomes can help calibrate which signals actually mattered historically.

Corporate partner feedback should become a first-class scoring input. For Wittington, feedback from Loblaw, Shoppers Drug Mart, PC Financial, PC Optimum, Choice Properties, Holt Renfrew, Joe Fresh, Lifemark, and related operating businesses could clarify whether a company maps to a real operating need, which buyer would own the problem, whether a pilot path exists, what procurement constraints apply, and whether the product is likely to be usable in practice.

## New Company Discovery

The next version should not only rank lists that users already have. It should help create new lists. A thesis-driven discovery module could search for companies by sector, problem area, geography, stage, customer segment, technology, founder background, recent financing, hiring velocity, regulatory trigger, or strategic buyer need.

Example discovery campaigns could include retail media infrastructure, pharmacy workflow automation, food waste and cold-chain monitoring, loyalty and personalization tools, healthcare access and patient navigation, consumer fintech, SMB commerce operations, climate adaptation for real estate, and logistics software for perishable goods.

The system should continuously gather candidates from public web search, news, accelerators, conference speaker lists, podcast appearances, investor portfolios, grants, job postings, app marketplaces, product directories, and company databases. Candidates should be deduped against Wittington’s CRM and prior reviews before paid enrichment starts.

## Data Quality And Entity Resolution

Scaling beyond one attendee list requires stronger entity resolution. The app should merge duplicate names, alternate spellings, domains, subsidiaries, legal entities, abbreviations, and event-display names. It should preserve source-level provenance so reviewers can see why a company appeared and which records were merged.

A future entity model should include canonical company, aliases, domains, people, locations, source records, relationship records, enrichment records, scores, review decisions, and export history. Confidence should be attached to each match. Low-confidence merges should be reviewable rather than silently collapsed.

The app should also detect stale or conflicting data. If one source says a company is Seed stage and another says Series B, the detail view should show the conflict, the source dates, and the confidence level. Score changes should explain which evidence changed and why the rank moved.

The recall problem should be measured directly. The system should track how many rows were excluded, how many remained broad candidates, how many entered the high-priority queue, and how many later became strong prospects after deeper enrichment. Periodic audits should sample excluded and low-priority rows to estimate false negatives. That would turn the current deterministic funnel into a continuously calibrated sourcing system rather than a fixed keyword screen.

## Enrichment Providers

External integrations should be selected based on whether they improve ranking quality, confidence, or diligence usefulness. Crunchbase or PitchBook could improve funding-stage detection, financing history, investor quality, and venture-backability assessment. People Data Labs, LinkedIn-style headcount data, Clearbit-style enrichment, and website crawling could improve team, location, category, growth, customer-segment, and product-depth signals.

News APIs could surface recent launches, financing events, customer wins, partnerships, executive changes, and strategic shifts. BuiltWith, Similarweb, job postings, app-store data, marketplace reviews, corporate registries, SEC or SEDAR filings, patents, grants, and import-export datasets could add evidence about technology stack, distribution, hiring, regulatory footprint, intellectual property, and commercial activity.

Each provider should be wrapped behind a common adapter interface with cost metadata, rate-limit settings, retry policy, cache TTL, confidence contribution, and field-level provenance. The app should be able to skip providers that are expensive or low-value for a given campaign.

## Performance And Job Architecture

The current implementation parallelizes network-bound company-page checks, Tavily calls, and OpenAI calls while keeping SQLite writes serialized. That is a good demo-scale architecture. A production version should move long runs into durable background jobs with resumable checkpoints.

The production job system should support pause, resume, cancel, retry failed rows, schedule overnight runs, run only uncached rows, cap spend per job, cap spend per source, and stop automatically when provider cost exceeds a configured threshold. It should also support adaptive concurrency: raising or lowering worker counts based on 429 frequency, provider latency, error rate, remaining budget, and account-tier limits.

For very large campaigns, the system should evaluate whether OpenAI Batch API, embeddings, structured extraction batches, or cheaper model tiers can reduce cost. The right approach may vary by step. Lightweight screening can use deterministic rules, embeddings, or smaller models. High-conviction candidates can receive richer synthesis from stronger models. This preserves quality where it matters without spending heavily on low-priority rows.

Future cheap-recall layers could build on the current website-domain and homepage-metadata pass with embeddings over company names and descriptions, low-cost classifier models, company database lookups, and stratified sampling of ambiguous rows. These layers would reduce the chance that a promising company with a generic name is missed by the first deterministic pass.

The first bounded web-metadata pass, conservative domain attempts, company-page scoring route, Tavily escalation route, data-gap route, and lightweight false-negative audit sample now exist, but all of them should become richer. Future versions should add better domain discovery, bounded `/about`, `/product`, `/platform`, and `/solutions` fetches, more robust wrong-entity detection, semantic scoring over extracted text, persistent reviewer labels, and measured audit outcomes for homepage-routed soft exclusions. That would add recall without sending every row immediately to an LLM.

The target production cascade should be: conservative deterministic exclusion, domain discovery, homepage metadata extraction, local semantic triage, Tavily Basic Search for unresolved or uncertain rows, evidence-gated OpenAI scoring, dual ranking by investment fit and review priority, and false-negative audits. The app should store each stage as a versioned artifact so the team can measure whether the extra stage improved recall, precision, cost per useful lead, and review burden.

## Cost Governance

The current app already separates live OpenAI project billing, recent billing, internal token-rate estimates, live Tavily usage snapshots when available, local Tavily run records when live usage is unavailable, Tavily included credits, Tavily pay-as-you-go billed spend, and Streamlit Cloud hosting. Production cost governance should add account-level budgets, campaign-level budgets, per-provider budgets, approval thresholds, and audit history.

Before a user launches a large uncached run, the app should continue to show projected page checks, Tavily calls, OpenAI calls, token usage, expected provider cost, credits remaining, pay-as-you-go overage exposure, worker counts, estimated runtime, and cache assumptions. For bigger deployments, it should also show confidence bands based on prior observed tokens per company, website timeout rates, provider error rates, and expected retry overhead.

Pay-as-you-go providers should default to disabled for cost-contained demos unless a fund operator explicitly enables them. If enabled, the app should make overage exposure clear before execution and should stop automatically at a configured spend ceiling. The current app reports the exposure; the production version should enforce it.

## Review Workflow

A production version should add persistent review state beyond the current scored-company cards. Saved shortlists, owner assignment, partner comments, duplicate resolution, review status, thesis-specific views, and audit history would make the tool useful for recurring sourcing work. Relevant Wittington views could include commerce infrastructure, retail operations, health services, consumer fintech, loyalty, climate, logistics software, real estate operations, and pharmacy technology.

Useful bulk actions would include CRM export, reviewer assignment, deeper diligence requests, partner-introduction notes, meeting-request drafts, memo generation, and follow-up reminders. The company detail page should show source records, enrichment evidence, score components, confidence level, score history, rank changes, and the reason for each material score change.

The app should support human correction. If a reviewer changes a stage, sector, exclusion reason, or Wittington strategic-edge assessment, that correction should be stored as first-party signal. Future scores should respect the correction and show when model output differs from human-reviewed truth.

## Investor-Facing Intelligence

The ranking should answer whether a company is worth Wittington’s attention in the current sourcing cycle. A lower-profile company with a clear operating use case, relevant relationship signals, recent hiring, and strong sector fit may rank above a better-known company with weaker strategic relevance.

Future versions should provide thesis summaries, market maps, ranked shortlists, key diligence questions, suggested operating-company introductions, competitive context, buyer relevance, and a short memo for each high-priority company. Every claim should remain tied to evidence so the tool supports investment judgment instead of replacing it with opaque scoring.

## Demo-To-Production Handoff

The current interview demo is intentionally narrow: one source list, one investor narrative, a guided Streamlit walkthrough, SQLite persistence, synchronous operations, and transparent provider-cost reporting. That makes it explainable in a live meeting. The production version should preserve that explainability while adding multi-campaign persistence, background jobs, durable approvals, richer evidence adapters, reviewer workflow, and CRM integration.

The highest-leverage next build would be a campaign model. Each campaign should store source adapter, uploaded rows, source mapping, scoring thesis, budget policy, verification mode, provider settings, review state, cost history, and exports. Manifest would become the first campaign adapter rather than the hard-coded product boundary.

The second priority should be persistent human review. The current app can rank and explain scored companies, but it does not yet remember partner decisions as durable first-party signal. Review decisions, pass reasons, corrected sectors, corrected stages, ownership, and partner comments should feed future scoring and recall audits.

The third priority should be production job control. The current split operation slides already make page checks, Tavily search, and OpenAI scoring operationally separate. Production should turn those operations into resumable jobs with spend ceilings, cancellation, retry failed rows, adaptive concurrency, and after-action reports that compare projected cost, actual billed cost, actual credits, runtime, errors, and useful prospects found.

The long-term goal is a repeatable sourcing intelligence system: start with any company universe, enrich it with public and first-party data, score it against Wittington’s current theses, control provider spend, explain uncertainty, preserve evidence, learn from reviewer feedback, and turn noisy lists into an actionable venture pipeline.
