# Model card

## Intended use

Internal technical-feasibility comparison of two ESCO occupation-matching approaches from an uploaded CV. The system is taxonomy-coding decision support, not a hiring, eligibility, job-fit, or candidate-quality system.

## Input pipeline

The upload endpoint accepts PDF, DOCX, or TXT. It extracts text in memory, finds the latest experience entry, parses an explicit skills section when present, matches official ESCO skill-group phrases, and estimates the latest role's duration from dates. It also reports total non-overlapping experience for diagnosis, but only latest-role duration enters model ranking.

Parser output is heuristic and must be inspected. Multi-column documents, unusual headings, image-only PDFs, decorative timelines, and ambiguous company/title ordering can produce an incorrect title or duration. OCR is not included.

## Model path 1

- Character `char_wb` 3–5 grams plus word 1–2 grams.
- Multiclass Logistic Regression for 438 exact-linked ESCO classes with at least 20 JobHop rows.
- Low-confidence fallback across 2,980 ESCO occupation profiles using character-title and official skill-group similarity.

## Model path 2

- `sentence-transformers/all-MiniLM-L6-v2`, 384-dimensional normalized embeddings.
- 2,980 exact-linked ESCO occupation profiles.
- 90% semantic cosine and 10% normalized-title similarity, with an optional 3% latest-role-duration adjustment.

Both paths receive the same extracted title, skills, and role length.

## Evaluation status

The repositories currently provide pipeline smoke tests for CV parsing, spelling variation, semantic alias matching, and web/API builds. These are not independent resume-accuracy measurements. A production decision requires a manually reviewed CV holdout, top-1/top-3 and coverage metrics, parser-versus-model error separation, subgroup analysis, confidence calibration, and an analyst-review threshold.

## Risks and safeguards

- An incorrect parser result makes the model comparison invalid; always inspect extracted features first.
- Historical occupation data and the pretrained encoder can reproduce occupational and language bias.
- Role length can proxy age, caregiving interruptions, and other protected or sensitive circumstances. Its ranking weight is intentionally small and should be removed unless its incremental value and subgroup impact are justified.
- Never add gender, name, photo, nationality, disability, family status, or other protected attributes as model features.
- Do not interpret score or rank as employability, seniority, fit, or merit.
- MASCO values are tentative four-digit candidates, not verified mappings.

## Privacy

The API processes each CV in memory and does not persist it. Production deployment still requires transport encryption, access controls, request/log review, retention rules, incident handling, and an approved privacy notice. Do not enable request-body logging.
