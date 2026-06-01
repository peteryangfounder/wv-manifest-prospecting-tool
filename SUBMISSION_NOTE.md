# Submission Note

I built this as a staged sourcing workflow rather than a one-shot AI pass. The Manifest attendee list is scraped and parsed in code, then stored in SQLite with normalized company keys so reruns are repeatable and do not redo paid work.

The current product model has two main data views. The Source list is the cleaned version of the messy attendee file after dedupe and deterministic screening. The Verified prospects list is the smaller set of companies that received external evidence enrichment and OpenAI scoring.

The cost-control design is deliberate. The tool first runs deterministic rules over the full list to identify obvious non-prospects such as large incumbents, investors, associations, universities, consultancies, retailers, brands, generic placeholders, and logistics service providers without software or platform signals. Only likely startup or technology companies and ambiguous candidates are sent to Tavily for external search enrichment. Only cached Tavily evidence is sent to OpenAI for structured scoring.

The dashboard now treats cost and resource usage as a first-class investor-facing metric. It tracks Tavily search calls, OpenAI scoring calls, input tokens, output tokens, total tokens, estimated Tavily spend, estimated OpenAI spend, total tracked API spend, and average spend per API-scored company.

The scoring is grounded in Wittington Ventures' focus areas and edge. Scores combine venture backability, Wittington sector fit, Wittington strategic edge, stage signal, traction signal, and evidence confidence. Caps prevent non-startups from ranking highly just because they are large or loosely relevant to logistics.

The Streamlit app is meant to be reviewable and operational. It exposes a guided three-step workflow, adjustable batch size, model selection, scoring weights, readable full-width charts, source and verified prospect exports, company detail, and a simple reset button for returning to a clean slate.
