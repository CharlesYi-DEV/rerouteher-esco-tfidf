# D12 deployable training inputs

`D12_training_examples_jobhop_v2_2019plus.csv` contains 562 structured career
transition examples derived only from the approved JobHop v2 confirmed-active
2019+ release. Its preserved split contains 455 train, 62 validation, and 45
test examples.

`D12_granular_catalog.csv` contains 258 unique MASCO 2020 occupations. Every
`masco_code` is exactly six digits; `masco_code_printed` retains the official
hyphen form for display, and `source_parent_group_code` is lineage only.

The retraining script requires the approved raw JobHop file and verifies its
SHA-256 before fitting:

```text
423bb1410db68feec2c5196297ee13277781dce1754de6e2d7c0daac1f4f53d4
```

MASCO and ESCO are occupational reference sources and are not additional
resume datasets. The ESCO-to-MASCO mapping is project-curated and requires
domain-owner review before production use.
