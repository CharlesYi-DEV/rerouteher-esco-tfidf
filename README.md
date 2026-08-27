# ReRouteHer ESCO Matcher — TF-IDF + Logistic Regression

CPU-friendly service that receives a job title, skills, and optional work length, then returns ranked ESCO occupation codes plus clearly labelled MASCO candidates. It is designed for misspellings, partial wording, and common aliases without requiring a GPU.

## What is included

- Responsive Next.js/Vinext website
- FastAPI prediction API and OpenAPI docs
- Character `char_wb` 3–5 gram + word 1–2 gram TF-IDF
- Multiclass Logistic Regression for well-supported JobHop labels
- Full-catalog character/skill retrieval fallback when classifier confidence is low
- Official ESCO–O*NET preferred titles and ESCO v1.2.1 occupation–skill matrix features
- Quarter-derived role duration, resume totals, and duration-band features
- Docker Compose deployment for the web and API services

## Data coverage

| Measure | Result |
| --- | ---: |
| JobHop rows | 47,224 |
| Resumes | 35,568 |
| Distinct historical ESCO codes | 2,278 |
| Rows linked to an official ESCO skill profile | 46,621 (98.7231%) |
| Exact-linked historical ESCO codes | 2,242 |
| Full ESCO retrieval catalog | 2,980 occupations |
| Logistic Regression classes (`>=20` JobHop rows) | 438 |
| Historical row coverage of classifier classes | 79.3697% |

Work length is an estimate because JobHop dates are quarter-granular. A same-quarter role is treated as 0.25 years. See [`data_quality_report.json`](data/processed/data_quality_report.json), [`DATA_SOURCES.md`](DATA_SOURCES.md), and the [`MODEL_CARD.md`](MODEL_CARD.md).

## Run with Docker

```bash
cp .env.example .env
docker compose up --build
```

- Website: <http://localhost:3000>
- API docs: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>

Set `NEXT_PUBLIC_API_BASE_URL` to the browser-accessible API URL on your server. Set `CORS_ORIGINS` to the deployed web origin. Do not use a Docker-internal hostname for `NEXT_PUBLIC_API_BASE_URL`; the request is made by the user's browser.

## API contract

```bash
curl -X POST http://localhost:8000/match \
  -H 'Content-Type: application/json' \
  -d '{
    "job_title": "software craftsperson",
    "skills": ["Python", "API design", "testing"],
    "work_length_years": 3.5,
    "top_k": 3
  }'
```

The response returns `esco_code`, `esco_title`, `esco_uri`, ranking score, method, skill overlaps, duration fit, and an optional MASCO candidate. `masco_mapping_status=isco4_candidate_requires_validation` means exactly that: it is not an official crosswalk.

## Local development

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm run dev
```

Run the API separately:

```bash
python -m venv .venv
.venv/bin/pip install -r api/requirements.txt
.venv/bin/uvicorn api.app:app --reload --port 8000
```

## Rebuild data and model

Processed data and the trained artifact are committed for reproducible deployment. To regenerate them from the archived project sources:

```bash
python scripts/build_dataset.py \
  --jobhop ../JobHop_v2_2019plus_corrected_2026-08-27/jobhop_v2_confirmed_active_2019plus.csv \
  --esco-crosswalk ../ReRouteHer_DataTeam_Fresh_HighStandard_2026-08-27/00_SOURCE_ARCHIVE/ESCO_to_ONET-SOC_official.xlsx \
  --esco-matrix ../ReRouteHer_DataTeam_Fresh_HighStandard_2026-08-27/00_SOURCE_ARCHIVE/ESCO_v1.2.1_skills_occupations_matrix.xlsx \
  --masco-catalog ../MASCO_remote_work/output/masco_2020_individual_occupations_en.csv \
  --output-dir data/processed
python scripts/train.py
```

The reported synthetic typo top-1/top-3 score checks robustness of the generated taxonomy examples only. It is not independent resume accuracy. Before production decisions, evaluate on a manually reviewed resume/title holdout and calibrate a reject/analyst-review threshold.

## Repository map

```text
api/              FastAPI service and matcher
app/              Interactive website
data/processed/   Reproducible features, ESCO profiles, quality report
model/            Deployable TF-IDF/Logistic Regression artifact
scripts/          Source-to-feature and model training pipelines
```

Code is MIT-licensed. Data terms are separate; keep the repository private until JobHop redistribution rights are confirmed.
