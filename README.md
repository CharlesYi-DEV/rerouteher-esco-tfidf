# ReRouteHer CV → ESCO technical feasibility test

Internal comparison harness for the ReRouteHer CV-upload flow. A tester uploads one CV; the backend extracts the latest job title, skills, and role length, then runs the same features through both model paths:

1. TF-IDF character/word features + Logistic Regression, with full-catalog retrieval fallback.
2. `sentence-transformers/all-MiniLM-L6-v2` semantic retrieval.

The page is intentionally plain. It exposes parser output, warnings, timing, and both top-three result tables for technical review.

## Current flow

```text
PDF / DOCX / TXT upload
        ↓
in-memory text extraction
        ↓
latest title + skills + latest-role duration
        ↓
┌────────────────────────────┬───────────────────────────┐
│ TF-IDF + Logistic Regression│ all-MiniLM-L6-v2          │
└────────────────────────────┴───────────────────────────┘
        ↓
two ranked ESCO result tables
```

The CV is not written to disk or a database. Uploads are limited to 10 MB and 50 PDF pages. Image-only/scanned PDFs require OCR before testing.

## Run with Docker

```bash
cp .env.example .env
docker compose up --build
```

Local Docker Compose automatically applies `docker-compose.override.yml` and publishes the gateway on `127.0.0.1:3000`. Coolify explicitly loads only `docker-compose.yml`, which intentionally contains no host-port bindings.

- Internal test page: <http://localhost:3000>
- API documentation through the web gateway: <http://localhost:3000/api/docs>
- API health check through the web gateway: <http://localhost:3000/api/health>

The Compose deployment does not bind host ports. Coolify should route the application domain to the `web` service on container port `3000`; that service forwards `/api/*` to the private `api` service. This avoids collisions with ports already used on the Docker host and keeps the API on the same browser origin.

## API

```bash
curl -X POST http://localhost:3000/api/match-cv \
  -F 'cv_file=@/path/to/resume.pdf'
```

The response contains:

- file metadata and `stored: false`;
- extracted title, skills, latest-role length, total non-overlapping experience, parser provenance, warnings, and a short text preview;
- parsing and inference timings;
- TF-IDF top-three results;
- MiniLM top-three results;
- tentative MASCO four-digit candidates labelled `isco4_candidate_requires_validation`.

## CV feature extraction

- Text: `pypdf` for text PDFs, `python-docx` for DOCX, and UTF-8/Latin-1 decoding for TXT.
- Latest job title: experience-heading detection plus the most recent employment date range and nearby role-like line.
- Skills: explicit skills sections/labelled lines plus exact phrases from official ESCO level-3 skill groups.
- Latest-role length: months in the date range associated with the selected latest title. This is the duration sent to both models.
- Total experience: union of non-overlapping employment ranges, reported for diagnosis but not used for ranking.

CV layouts vary. The extracted-feature panel must be reviewed before treating model comparison results as meaningful. A parser warning is not a model result.

## Data coverage

| Measure | Result |
| --- | ---: |
| JobHop rows | 47,224 |
| Resumes | 35,568 |
| Historical JobHop ESCO codes | 2,278 |
| Rows linked to official ESCO skill profiles | 46,621 (98.7231%) |
| Searchable ESCO occupation profiles | 2,980 |
| Logistic Regression classes (`>=20` JobHop rows) | 438 |
| MiniLM embedding dimensions | 384 |

Data construction uses the official ESCO–O*NET titles and ESCO v1.2.1 Skill–Occupation Matrix 3.0. See [`DATA_SOURCES.md`](DATA_SOURCES.md), [`MODEL_CARD.md`](MODEL_CARD.md), and [`data_quality_report.json`](data/processed/data_quality_report.json).

## Local development

```bash
corepack enable
pnpm install --frozen-lockfile
python -m venv .venv
.venv/bin/pip install -r api/requirements.txt
.venv/bin/uvicorn api.app:app --reload --port 8000
pnpm run dev
```

Run checks:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q api scripts
pnpm run lint
pnpm run build
```

## Repository map

```text
api/cv_parser.py      CV extraction and feature derivation
api/matcher.py        TF-IDF/Logistic Regression model path
api/minilm_matcher.py MiniLM semantic model path
app/                  Plain internal comparison page
model/                Both deployable model artifacts
data/processed/       ESCO-linked JobHop features and quality report
tests/                CV parser checks
```

The separate [`rerouteher-esco-minilm`](https://github.com/CharlesYi-DEV/rerouteher-esco-minilm) repository remains the standalone semantic component. This repository is the single website and combined deployment target.

Code is MIT-licensed. Keep the repository private until JobHop redistribution rights and CV-handling controls are formally approved.
