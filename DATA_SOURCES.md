# Data sources and reuse notes

The code is MIT-licensed. The data retains the terms of its original sources.

| Asset | Use in this project | Source / status |
| --- | --- | --- |
| JobHop v2 confirmed-active 2019+ | Sole resume/career-history source for 562 D12 transition examples | Project-provided processed dataset. Raw SHA-256 is pinned in the model. Confirm redistribution rights before making the repository public. |
| MASCO 2020 official catalog | Exact six-digit codes, printed code forms, occupation titles, descriptions, and task text for the 258-role D11 catalog | Department of Statistics Malaysia official PDF archived by the project. Occupational reference data only. |
| ESCO–O*NET official crosswalk | ESCO occupation titles used to interpret JobHop codes while constructing structured examples | European Commission ESCO crosswalk and [technical report](https://esco.ec.europa.eu/system/files/2022-12/ONET%20ESCO%20Technical%20Report.pdf). Occupational reference data only. |
| ESCO v1.2.1 Skill–Occupation Matrix 3.0 | Top skill groups used to describe prior JobHop occupation records | European Commission [matrix publication](https://esco.ec.europa.eu/en/about-esco/publications/publication/skills-occupations-matrix-tables). Occupational reference data only. |

No other resume dataset is used. ESCO and MASCO are taxonomies/reference
sources and must not be counted as resume datasets.

The ESCO-to-six-digit-MASCO label crosswalk is a curated project mapping, not an
official published crosswalk. It requires domain-owner confirmation before any
production decision. Four-digit MASCO group codes appear only for source
lineage; all deployed role labels and predictions match `^\d{6}$`.
