#!/usr/bin/env python3
"""Build normalized JobHop v2 + ESCO occupation-skill feature tables.

The official ESCO-to-O*NET workbook supplies ESCO/ISCO codes and preferred
titles.  The official ESCO v1.2.1 matrix supplies occupation URIs and weighted
level-3 skill groups.  They are joined by a strict normalized preferred-title
match; unmatched records are retained and explicitly flagged.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_title(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def quarter_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.extract(r"Q([1-4])", expand=False))


def duration_band(years: float) -> str:
    if pd.isna(years):
        return "unknown"
    if years < 1:
        return "under_1_year"
    if years < 3:
        return "1_to_3_years"
    if years < 5:
        return "3_to_5_years"
    if years < 10:
        return "5_to_10_years"
    return "10_plus_years"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobhop", type=Path, required=True)
    parser.add_argument("--esco-crosswalk", type=Path, required=True)
    parser.add_argument("--esco-matrix", type=Path, required=True)
    parser.add_argument("--masco-catalog", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-skills", type=int, default=24)
    return parser.parse_args()


def load_esco_profiles(matrix_path: Path, top_skills: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_excel(matrix_path, sheet_name="Matrix 3.0", header=None)
    skill_uris = matrix.iloc[0, 2:].astype("string")
    skill_labels = matrix.iloc[1, 2:].astype("string")

    long_rows: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    for _, row in matrix.iloc[2:].iterrows():
        occupation_uri = row.iloc[0]
        occupation_title = row.iloc[1]
        if pd.isna(occupation_uri) or pd.isna(occupation_title):
            continue
        weights = pd.to_numeric(row.iloc[2:], errors="coerce").fillna(0.0).to_numpy(float)
        positive = np.flatnonzero(weights > 0)
        ranked = positive[np.argsort(weights[positive])[::-1]] if len(positive) else positive
        ranked = ranked[:top_skills]
        skills: list[str] = []
        weighted: list[str] = []
        for rank, idx in enumerate(ranked, start=1):
            label = str(skill_labels.iloc[idx]).strip()
            uri = str(skill_uris.iloc[idx]).strip()
            weight = float(weights[idx])
            skills.append(label)
            weighted.append(f"{label}::{weight:.6f}")
            long_rows.append(
                {
                    "esco_occupation_uri": str(occupation_uri),
                    "esco_title": str(occupation_title),
                    "skill_group_uri": uri,
                    "skill_group_label": label,
                    "matrix_weight": round(weight, 8),
                    "skill_rank": rank,
                    "source_version": "ESCO v1.2.1 Matrix 3.0",
                }
            )
        profiles.append(
            {
                "esco_occupation_uri": str(occupation_uri),
                "esco_title_matrix": str(occupation_title),
                "normalized_title": normalize_title(occupation_title),
                "top_skill_groups": " | ".join(skills),
                "top_skill_groups_weighted": " | ".join(weighted),
                "top_skill_count": len(skills),
            }
        )
    return pd.DataFrame(profiles), pd.DataFrame(long_rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobhop = pd.read_csv(args.jobhop, dtype={"matched_code": "string"})
    crosswalk = pd.read_excel(args.esco_crosswalk, sheet_name=0, header=3, dtype=str)
    crosswalk = crosswalk.iloc[:, :4]
    crosswalk.columns = ["esco_code", "esco_title", "onet_soc_code", "onet_soc_title"]
    code_titles = crosswalk[["esco_code", "esco_title"]].dropna().drop_duplicates()
    code_titles["normalized_title"] = code_titles["esco_title"].map(normalize_title)

    matrix_profiles, matrix_skill_rows = load_esco_profiles(args.esco_matrix, args.top_skills)
    catalog = code_titles.merge(matrix_profiles, on="normalized_title", how="left", validate="one_to_one")
    catalog["mapping_status"] = np.where(
        catalog["esco_occupation_uri"].notna(), "official_title_exact", "no_matrix_title_match"
    )

    if args.masco_catalog and args.masco_catalog.exists():
        masco = pd.read_csv(args.masco_catalog, dtype=str)
        masco_units = (
            masco[["masco_4d_code", "masco_unit_group_title_en"]]
            .dropna()
            .drop_duplicates("masco_4d_code")
        )
        catalog["masco_candidate_code"] = catalog["esco_code"].str.extract(r"^(\d{4})", expand=False)
        catalog = catalog.merge(masco_units, left_on="masco_candidate_code", right_on="masco_4d_code", how="left")
        catalog["masco_mapping_status"] = np.where(
            catalog["masco_unit_group_title_en"].notna(),
            "isco4_candidate_requires_validation",
            "no_candidate",
        )
        catalog = catalog.drop(columns=["masco_4d_code"])
    else:
        catalog["masco_candidate_code"] = pd.NA
        catalog["masco_unit_group_title_en"] = pd.NA
        catalog["masco_mapping_status"] = "not_evaluated"

    start_q = quarter_number(jobhop["start_date"])
    end_q = quarter_number(jobhop["end_date"])
    quarter_delta = (jobhop["end_year"] - jobhop["start_year"]) * 4 + (end_q - start_q)
    jobhop["work_length_years_est"] = (quarter_delta.clip(lower=1) / 4).round(2)
    jobhop["work_length_months_est"] = (jobhop["work_length_years_est"] * 12).round().astype("Int64")
    jobhop["work_length_band"] = jobhop["work_length_years_est"].map(duration_band)

    by_resume = jobhop.groupby("resume_id", observed=True)
    resume_features = by_resume.agg(
        resume_job_count=("matched_code", "size"),
        resume_total_role_years_est=("work_length_years_est", "sum"),
        resume_first_start_year=("start_year", "min"),
        resume_last_end_year=("end_year", "max"),
    ).reset_index()
    resume_features["resume_span_years_est"] = (
        resume_features["resume_last_end_year"] - resume_features["resume_first_start_year"]
    ).clip(lower=0)
    jobhop = jobhop.merge(resume_features, on="resume_id", how="left", validate="many_to_one")

    enriched = jobhop.merge(
        catalog[
            [
                "esco_code",
                "esco_title",
                "esco_occupation_uri",
                "top_skill_groups",
                "top_skill_groups_weighted",
                "top_skill_count",
                "mapping_status",
                "masco_candidate_code",
                "masco_unit_group_title_en",
                "masco_mapping_status",
            ]
        ],
        left_on="matched_code",
        right_on="esco_code",
        how="left",
        validate="many_to_one",
    )
    enriched["mapping_status"] = enriched["mapping_status"].fillna("no_official_code_match")
    enriched["skill_feature_text"] = enriched["top_skill_groups"].fillna("")

    counts = jobhop["matched_code"].value_counts().rename("jobhop_record_count")
    duration_stats = jobhop.groupby("matched_code")["work_length_years_est"].agg(
        work_length_median_years="median",
        work_length_mean_years="mean",
        work_length_p25_years=lambda s: s.quantile(0.25),
        work_length_p75_years=lambda s: s.quantile(0.75),
    )
    catalog = catalog.merge(counts, left_on="esco_code", right_index=True, how="left")
    catalog = catalog.merge(duration_stats, left_on="esco_code", right_index=True, how="left")
    catalog["jobhop_record_count"] = catalog["jobhop_record_count"].fillna(0).astype(int)

    mapped_uris = set(catalog.loc[catalog["mapping_status"].eq("official_title_exact"), "esco_occupation_uri"])
    skill_links = matrix_skill_rows[matrix_skill_rows["esco_occupation_uri"].isin(mapped_uris)].copy()
    uri_codes = catalog[["esco_code", "esco_occupation_uri"]].dropna(subset=["esco_occupation_uri"])
    skill_links = skill_links.merge(
        uri_codes, on="esco_occupation_uri", how="inner", validate="many_to_one"
    )

    enriched_path = args.output_dir / "jobhop_esco_features.csv.gz"
    catalog_path = args.output_dir / "esco_occupation_profiles.csv"
    skills_path = args.output_dir / "esco_occupation_skill_weights.csv"
    report_path = args.output_dir / "data_quality_report.json"
    enriched.to_csv(enriched_path, index=False, compression="gzip")
    catalog.to_csv(catalog_path, index=False)
    skill_links.to_csv(skills_path, index=False)

    exact = enriched["mapping_status"].eq("official_title_exact")
    report = {
        "jobhop_rows": int(len(enriched)),
        "jobhop_resumes": int(enriched["resume_id"].nunique()),
        "jobhop_distinct_codes": int(enriched["matched_code"].nunique()),
        "exact_esco_skill_link_rows": int(exact.sum()),
        "exact_esco_skill_link_rate": round(float(exact.mean()), 6),
        "exact_esco_codes": int(enriched.loc[exact, "matched_code"].nunique()),
        "catalog_codes": int(len(catalog)),
        "catalog_codes_present_in_jobhop": int((catalog["jobhop_record_count"] > 0).sum()),
        "skill_link_rows": int(len(skill_links)),
        "duration_method": "Quarter-granularity elapsed estimate; minimum 0.25 years for same-quarter records.",
        "esco_join_method": "Strict normalized preferred-title match between official crosswalk and ESCO v1.2.1 matrix.",
        "masco_note": "Four-digit candidates are not an official ESCO-to-MASCO crosswalk and require validation.",
        "source_files": {
            "jobhop": args.jobhop.name,
            "esco_crosswalk": args.esco_crosswalk.name,
            "esco_matrix": args.esco_matrix.name,
            "masco_catalog": args.masco_catalog.name if args.masco_catalog else None,
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
