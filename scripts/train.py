#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression


ALIASES = {
    "human resources": "hr",
    "administrative": "admin",
    "software developer": "software engineer",
    "user interface": "ui",
    "user experience": "ux",
    "customer service representative": "csr",
    "sales representative": "sales rep",
}


def duration_band(years: float | None) -> str:
    if years is None or pd.isna(years): return "work_length_unknown"
    if years < 1: return "work_length_under_1_year"
    if years < 3: return "work_length_1_to_3_years"
    if years < 5: return "work_length_3_to_5_years"
    if years < 10: return "work_length_5_to_10_years"
    return "work_length_10_plus_years"


def compose(title: str, skills: str, years: float | None) -> str:
    title = title.lower()
    return f"title {title}. job {title}. occupation {title}. skills {skills.lower()}. {duration_band(years)}"


def alias_variant(title: str) -> str:
    result = title.lower()
    for source, target in ALIASES.items():
        result = result.replace(source, target)
    return result


def typo_variant(title: str) -> str:
    words = title.lower().split()
    if not words:
        return title.lower()
    idx = max(range(len(words)), key=lambda i: len(words[i]))
    word = words[idx]
    if len(word) > 5:
        cut = max(1, len(word) // 2)
        words[idx] = word[:cut] + word[cut + 1 :]
    return " ".join(words)


def partial_variant(title: str) -> str:
    words = [word for word in re.findall(r"[a-z0-9]+", title.lower()) if len(word) > 2]
    return " ".join(words[: max(1, (len(words) + 1) // 2)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, default=Path("data/processed/esco_occupation_profiles.csv"))
    parser.add_argument("--output", type=Path, default=Path("model/tfidf_logreg.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("model/metrics.json"))
    parser.add_argument("--min-jobhop-records", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = pd.read_csv(args.profiles, dtype={"esco_code": str, "masco_candidate_code": str})
    profiles = profiles[
        profiles["mapping_status"].eq("official_title_exact")
        & profiles["top_skill_groups"].notna()
        & profiles["jobhop_record_count"].ge(args.min_jobhop_records)
    ].copy()

    train_texts: list[str] = []
    train_labels: list[str] = []
    test_texts: list[str] = []
    test_labels: list[str] = []
    for row in profiles.itertuples(index=False):
        title = str(row.esco_title)
        skills = str(row.top_skill_groups)
        years = None if pd.isna(row.work_length_median_years) else float(row.work_length_median_years)
        for variant in [title, alias_variant(title), partial_variant(title)]:
            train_texts.append(compose(variant, skills, years))
            train_labels.append(str(row.esco_code))
        test_texts.append(compose(typo_variant(title), skills, None))
        test_labels.append(str(row.esco_code))

    features = FeatureUnion(
        [
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=8000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=3000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
        ],
        transformer_weights={"char": 1.0, "word": 0.55},
    )
    pipeline = Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    solver="saga",
                    C=4.0,
                    max_iter=40,
                    tol=0.01,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(train_texts, train_labels)
    probabilities = pipeline.predict_proba(test_texts)
    predictions = pipeline.named_steps["classifier"].classes_[np.argmax(probabilities, axis=1)]
    classes = pipeline.named_steps["classifier"].classes_

    full_catalog = pd.read_csv(args.profiles, dtype={"esco_code": str, "masco_candidate_code": str})
    full_catalog = full_catalog[
        full_catalog["mapping_status"].eq("official_title_exact") & full_catalog["top_skill_groups"].notna()
    ].copy()
    title_texts = [str(row.esco_title).lower() for row in full_catalog.itertuples(index=False)]
    skill_texts = [str(row.top_skill_groups).lower() for row in full_catalog.itertuples(index=False)]
    title_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=40000, sublinear_tf=True
    )
    title_matrix = title_vectorizer.fit_transform(title_texts)
    skill_vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1, max_features=20000, sublinear_tf=True
    )
    skill_matrix = skill_vectorizer.fit_transform(skill_texts)

    metrics = {
        "model": "TF-IDF char_wb(3,5) + word(1,2) + LogisticRegression",
        "trained_exact_esco_classes": int(len(classes)),
        "full_fallback_catalog_classes": int(len(full_catalog)),
        "synthetic_typo_top1_accuracy": round(float(accuracy_score(test_labels, predictions)), 6),
        "synthetic_typo_top3_accuracy": round(
            float(top_k_accuracy_score(test_labels, probabilities, k=3, labels=classes)), 6
        ),
        "evaluation_warning": "Synthetic taxonomy robustness only; not independent real-resume accuracy.",
        "minimum_jobhop_records_per_logreg_class": args.min_jobhop_records,
        "logreg_historical_row_coverage": round(
            float(profiles["jobhop_record_count"].sum() / pd.read_csv(args.profiles)["jobhop_record_count"].sum()), 6
        ),
    }
    artifact = {
        "pipeline": pipeline,
        "title_vectorizer": title_vectorizer,
        "title_matrix": title_matrix,
        "skill_vectorizer": skill_vectorizer,
        "skill_matrix": skill_matrix,
        "catalog": full_catalog.where(pd.notna(full_catalog), None).to_dict(orient="records"),
        "low_confidence_threshold": 0.45,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
