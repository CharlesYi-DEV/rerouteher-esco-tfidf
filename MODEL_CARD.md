# Model card

## Intended use

Internal technical-feasibility comparison of two six-digit MASCO occupation
suggestion approaches from an uploaded CV. The system supports taxonomy coding;
it is not a hiring, eligibility, job-fit, candidate-quality, or employability
system. Every suggestion requires confirmation by the user.

## Label contract

- Stored prediction: exactly six digits, such as `251201`.
- Printed MASCO form: `2512-01`, display only.
- Four-digit code: parent-group lineage only, never a role prediction.
- ESCO code/title: retained project-crosswalk comparison metadata, never the
  predicted label; pending domain-owner review.
- Catalog: 258 unique granular MASCO 2020 occupations across ten curated parent
  groups.

The API validates the MASCO contract when loading the artifact and again before
formatting each prediction. It returns the primary and all linked ESCO
comparison rows separately, including their authority and review status.

## Training scope

JobHop v2 confirmed-active 2019+ is the sole resume/career-history dataset. The
artifact pins raw source SHA-256
`423bb1410db68feec2c5196297ee13277781dce1754de6e2d7c0daac1f4f53d4`.
MASCO and ESCO are occupational reference sources, not resume datasets.

The supervised set contains 562 JobHop transition examples with the preserved
455/62/45 train/validation/test split. Forty-one granular labels appear before
the sparsity policy; labels with fewer than five training examples merge to a
curated six-digit anchor in the same parent group, leaving 19 trainable classes.

## Input pipeline

The upload endpoint accepts PDF, DOCX, or TXT. It extracts text in memory, finds
the latest experience entry, parses an explicit skills section when present,
and estimates the latest role's duration. Those fields are adapted into the
structured career-history text format used by D12.

This is an important transfer limitation: D12 was trained from structured
JobHop career histories, not raw Malaysian CV text. Parser output must be
inspected before reviewing model suggestions. Multi-column layouts, unusual
headings, image-only PDFs, and ambiguous company/title ordering can produce
incorrect features. OCR is not included.

## Model path 1

- word 1–2 grams plus character `char_wb` 3–5 grams;
- balanced multiclass Logistic Regression over 19 six-digit classes;
- validation-calibrated low-confidence fallback using character TF-IDF over all
  258 catalog roles.

Held-out results: validation accuracy 19.35%, macro-F1 15.25%, top-3 41.94%;
test accuracy 20.00%, macro-F1 11.88%, top-3 37.78%.

## Model path 2

- `sentence-transformers/all-MiniLM-L6-v2`;
- 384-dimensional normalized embeddings;
- cosine retrieval over 19 JobHop training-class centroids.

Held-out results: validation accuracy 25.81%, macro-F1 15.67%; test accuracy
24.44%, macro-F1 14.45%. MiniLM is the research benchmark winner on validation
macro-F1, but it is not approved for automatic production use.

## Risks and safeguards

- Low held-out accuracy means results are suggestions, not authoritative codes.
- The project ESCO-to-six-digit-MASCO crosswalk requires domain-owner review.
- Sparse labels are merged, so some valid granular occupations cannot be direct
  supervised predictions; only the fallback can search all 258 roles.
- The historical dataset and pretrained encoder may reproduce occupational,
  geographic, and language bias.
- Role length may proxy age, caregiving interruptions, or other sensitive
  circumstances; do not use rank or score as merit or employability.
- Never add gender, name, photo, nationality, disability, family status, or
  other protected attributes as model features.

## Privacy

The API processes each CV in memory and does not persist it. Production use
still requires transport encryption, access controls, request/log review,
retention rules, incident handling, and an approved privacy notice. Request-body
logging must remain disabled.
