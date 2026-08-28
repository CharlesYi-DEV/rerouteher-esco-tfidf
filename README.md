# ReRouteHer CV → six-digit MASCO matching test

Internal comparison harness for the ReRouteHer CV-upload flow. A tester uploads
one CV; the backend extracts the latest job title, skills, and role length, then
runs the same features through two JobHop-trained six-digit MASCO model paths:

1. word + character `char_wb` TF-IDF with balanced Logistic Regression;
2. `sentence-transformers/all-MiniLM-L6-v2` JobHop class-centroid retrieval.

Every returned `masco_code` contains exactly six digits, for example `251201`.
The official printed form (`2512-01`) is display-only. Four-digit codes are
retained only as parent-group lineage and are never predictions.

ESCO codes and titles are also retained beside each MASCO result for
comparison. They come from the project ESCO-to-MASCO crosswalk, are marked as
pending domain-owner review, and are never presented as the predicted label.

## Current flow

```text
PDF / DOCX / TXT upload
        ↓
in-memory text extraction
        ↓
latest title + skills + latest-role duration
        ↓
JobHop-compatible structured feature text
        ↓
┌──────────────────────────────┬─────────────────────────────┐
│ TF-IDF + Logistic Regression │ MiniLM class-centroid model │
└──────────────────────────────┴─────────────────────────────┘
        ↓
two ranked six-digit MASCO suggestion tables
        ↓
linked ESCO codes/titles shown as comparison metadata
```

The CV is not written to disk or a database. Uploads are limited to 10 MB and
50 PDF pages. Image-only/scanned PDFs require OCR before testing. All model
outputs require user confirmation and must not be used as automatic employment
decisions.

## Resume-data scope

JobHop v2 confirmed-active 2019+ is the only resume/career-history dataset used
for training. The exact approved raw file is pinned in the artifact with
SHA-256:

```text
423bb1410db68feec2c5196297ee13277781dce1754de6e2d7c0daac1f4f53d4
```

MASCO 2020 supplies six-digit occupation codes, titles, descriptions, and task
text. ESCO supplies occupational reference titles and skill groups used while
constructing the JobHop examples. Neither taxonomy is an additional resume
dataset.

## Model coverage and evaluation

| Measure | Result |
| --- | ---: |
| JobHop-derived transition examples | 562 |
| Preserved train / validation / test split | 455 / 62 / 45 |
| Pre-merge granular labels | 41 |
| Trainable six-digit classes after sparse-label policy | 19 |
| Full six-digit MASCO catalog | 258 |
| TF-IDF validation accuracy / macro-F1 | 19.35% / 15.25% |
| TF-IDF test accuracy / macro-F1 | 20.00% / 11.88% |
| MiniLM validation accuracy / macro-F1 | 25.81% / 15.67% |
| MiniLM test accuracy / macro-F1 | 24.44% / 14.45% |

MiniLM remains the research benchmark winner on validation macro-F1. These are
prototype results, not production approval. The JobHop inputs contain
structured Belgian/Flemish career histories rather than raw Malaysian CV text,
and the project ESCO-to-MASCO crosswalk still requires domain-owner review.

## Run with Docker

```bash
cp .env.example .env
docker compose up --build
```

Local Docker Compose publishes the gateway on `127.0.0.1:3000`. Coolify should
route the application domain to the `web` service on container port `3000`;
that service forwards `/api/*` to the private `api` service.

- Test page: <http://localhost:3000>
- API documentation: <http://localhost:3000/api/docs>
- API health: <http://localhost:3000/api/health>

## API

```bash
curl -X POST http://localhost:3000/api/match-cv \
  -F 'cv_file=@/path/to/resume.pdf'
```

The response includes file metadata, extracted CV features, inference timings,
the JobHop-only model policy, and TF-IDF/MiniLM top-three MASCO suggestions.
Each match contains the stored six-digit code, printed hyphen form, MASCO title,
four-digit parent lineage, linked ESCO comparison entries, score, ranking scope,
and confirmation requirement.

## Retrain

The deployable artifact is rebuilt with the service's own scikit-learn and
MiniLM runtime:

```bash
.venv/bin/python scripts/train_masco.py \
  --jobhop-source /path/to/jobhop_v2_confirmed_active_2019plus.csv \
  --local-files-only
```

Training fails if the raw JobHop SHA-256 changes, any label is not six digits,
the D11 catalog is not 258 unique codes, or a model class is absent from the
catalog. The project ESCO comparison crosswalk is also validated and embedded
in the artifact.

## Checks

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q api scripts
pnpm run lint
pnpm run build
```

## Repository map

```text
api/cv_parser.py          CV extraction and feature derivation
api/masco_matcher.py      Six-digit TF-IDF and MiniLM inference
app/                      Internal comparison page
scripts/train_masco.py    JobHop-only reproducible retraining
data/masco/               D12 examples, 258-role catalog, and ESCO comparison crosswalk
model/masco/              Deployable artifact and metrics
tests/                    Parser and six-digit contract checks
```

See [`DATA_SOURCES.md`](DATA_SOURCES.md) and [`MODEL_CARD.md`](MODEL_CARD.md)
for provenance, limits, and safe-use guidance.

Code is MIT-licensed. Keep the repository private until JobHop redistribution
rights and CV-handling controls are formally approved.
