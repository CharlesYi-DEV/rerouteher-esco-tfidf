# Data sources and reuse notes

The code is MIT-licensed. The data keeps the terms of its original sources.

| Asset | Use in this project | Source / status |
| --- | --- | --- |
| JobHop v2 2019+ | Historical occupation codes and quarter-level work history | Project-provided processed resume dataset. Confirm redistribution rights before making this repository public. |
| ESCO–O*NET official crosswalk | ESCO codes and preferred occupation titles | European Commission ESCO crosswalk and [technical report](https://esco.ec.europa.eu/system/files/2022-12/ONET%20ESCO%20Technical%20Report.pdf). |
| ESCO v1.2.1 Skill–Occupation Matrix 3.0 | Weighted level-3 skill groups linked to occupation URIs | European Commission [matrix publication](https://esco.ec.europa.eu/en/about-esco/publications/publication/skills-occupations-matrix-tables). |
| MASCO 2020 occupation catalog | Label lookup for a tentative four-digit candidate | Department of Statistics Malaysia source archived by the project. No official ESCO-to-MASCO crosswalk was available. |

Unless a source says otherwise, EU-owned ESCO content is reusable under CC BY 4.0; see the [ESCO copyright notice](https://esco.ec.europa.eu/uk/node/456).

The `masco_candidate_code` is only the first four digits of the ISCO/ESCO code when that value exists in the local MASCO catalog. Every such record is labelled `isco4_candidate_requires_validation`. Do not present it as a verified MASCO mapping.
