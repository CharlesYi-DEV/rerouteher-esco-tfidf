# Model card

## Intended use

Suggest up to five ESCO occupation codes from a user-supplied job title, skills, and optional role length. The output is decision support for taxonomy coding and should allow analyst review; it is not a hiring, eligibility, or candidate-quality score.

## Architecture

- Primary path: character `char_wb` 3–5 grams plus word 1–2 grams, then multiclass Logistic Regression.
- Fallback path: 82% character-title similarity and 18% skill-profile similarity, with an optional 3% duration adjustment.
- Training labels: 438 exact-linked ESCO classes with at least 20 JobHop rows.
- Retrieval coverage: 2,980 official-title exact-linked ESCO occupations.
- Normalization: common abbreviations and aliases, plus character n-grams for misspellings and partial words.

## Features

`job_title`, normalized `skills`, `work_length_years`, duration band, and the official ESCO v1.2.1 level-3 skill-group profile. JobHop work length is estimated from quarter fields, with 0.25 years as the same-quarter minimum.

## Evaluation

The artifact reports 100% top-1/top-3 on generated typo variants of the taxonomy profiles. This is a pipeline robustness check, not independent real-resume accuracy. Production release requires a manually labelled holdout, per-class/error analysis, confidence calibration, and review of low-confidence or high-impact cases.

## Risks and safeguards

- Historical occupation data can reproduce occupational segregation and under-representation.
- Work length can proxy age, caregiving interruptions, and other protected or sensitive circumstances. Keep its weight low, measure its incremental value, and remove it if subgroup impact is not justified.
- Do not use gender, name, photo, nationality, disability, family status, or other protected attributes.
- Do not interpret rank or score as employability, seniority, fit, or merit.
- MASCO values are tentative four-digit candidates and require validation.
- Log model version and inputs, expose top-k alternatives, and retain an analyst override.

## Privacy

The committed processed table contains pseudonymous resume IDs and job-history-derived fields. Keep the repository private, minimize access, and confirm JobHop redistribution and retention rules before any public release.
