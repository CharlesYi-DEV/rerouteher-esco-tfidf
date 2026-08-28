#!/usr/bin/env python3
"""Train the deployable D12 six-digit MASCO matchers.

The supervised examples are derived only from the approved JobHop v2 2019+
career-history release. MASCO 2020 supplies the six-digit taxonomy catalogue;
it is reference data, not a second resume dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline


LABEL_PATTERN = re.compile(r"^\d{6}$")
ESCO_PATTERN = re.compile(r"^\d{4}(?:\.\d+)+$")
EXPECTED_JOBHOP_SHA256 = "423bb1410db68feec2c5196297ee13277781dce1754de6e2d7c0daac1f4f53d4"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("data/masco/D12_training_examples_jobhop_v2_2019plus.csv"),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/masco/D12_granular_catalog.csv"),
    )
    parser.add_argument(
        "--esco-crosswalk",
        type=Path,
        default=Path("data/masco/D12_esco_to_masco_label_crosswalk.csv"),
        help=(
            "Project ESCO-to-MASCO comparison crosswalk. ESCO is retained as "
            "comparison metadata and is never used as the prediction label."
        ),
    )
    parser.add_argument("--jobhop-source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model/masco/d12_granular_masco_classifier.joblib"),
    )
    parser.add_argument(
        "--metrics", type=Path, default=Path("model/masco/metrics.json")
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the MiniLM weights to already exist in the local cache.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def validate_inputs(
    examples: list[dict[str, str]], catalog: list[dict[str, str]], source: Path
) -> str:
    source_hash = sha256(source)
    if source_hash != EXPECTED_JOBHOP_SHA256:
        raise ValueError(
            "The raw resume source is not the approved JobHop v2 2019+ file: "
            f"expected {EXPECTED_JOBHOP_SHA256}, got {source_hash}."
        )
    if not examples or not catalog:
        raise ValueError("Training examples and the MASCO catalog must be non-empty.")
    if {row["split"] for row in examples} != {"train", "val", "test"}:
        raise ValueError("Expected the preserved JobHop train/val/test split contract.")
    labels = [row["masco_code"] for row in examples]
    codes = [row["masco_code"] for row in catalog]
    if not all(LABEL_PATTERN.fullmatch(code) for code in labels + codes):
        raise ValueError("Every class, example label, and catalog key must be six digits.")
    if len(codes) != len(set(codes)):
        raise ValueError("The MASCO catalog contains duplicate six-digit codes.")
    if len(catalog) != 258:
        raise ValueError(f"Expected 258 D11 roles, found {len(catalog)}.")
    if not set(labels).issubset(codes):
        raise ValueError("A training label is absent from the D11 MASCO catalog.")
    return source_hash


def build_pipeline(c_value: float) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=40000,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=60000,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=4000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def split_rows(
    rows: list[dict[str, str]], split: str
) -> tuple[list[str], list[str]]:
    selected = [row for row in rows if row["split"] == split]
    return [row["text"] for row in selected], [row["masco_code"] for row in selected]


def classifier_metrics(
    pipeline: Pipeline, texts: list[str], labels: list[str]
) -> dict[str, Any]:
    probabilities = pipeline.predict_proba(texts)
    classes = list(pipeline.named_steps["classifier"].classes_)
    predictions = np.asarray(classes)[probabilities.argmax(axis=1)]
    top_k = min(3, len(classes))
    return {
        "n_examples": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)
        ),
        "top3_accuracy": float(
            top_k_accuracy_score(labels, probabilities, k=top_k, labels=classes)
        ),
    }


def retrieval_metrics(
    vectors: np.ndarray,
    labels: list[str],
    classes: list[str],
    centroids: dict[str, np.ndarray],
) -> dict[str, Any]:
    centroid_matrix = np.vstack([centroids[code] for code in classes])
    predictions = np.asarray(classes)[(vectors @ centroid_matrix.T).argmax(axis=1)]
    return {
        "n_examples": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)
        ),
    }


def main() -> None:
    args = parse_args()
    examples = read_rows(args.examples)
    catalog_rows = read_rows(args.catalog)
    crosswalk_rows = read_rows(args.esco_crosswalk)
    source_hash = validate_inputs(examples, catalog_rows, args.jobhop_source)

    train_texts, train_labels = split_rows(examples, "train")
    val_texts, val_labels = split_rows(examples, "val")
    test_texts, test_labels = split_rows(examples, "test")

    grid_results: list[dict[str, float]] = []
    selected: tuple[tuple[float, float], float, Pipeline] | None = None
    for c_value in (0.5, 1.0, 2.0, 4.0):
        candidate = build_pipeline(c_value)
        candidate.fit(train_texts, train_labels)
        result = classifier_metrics(candidate, val_texts, val_labels)
        grid_results.append(
            {
                "C": c_value,
                "validation_accuracy": result["accuracy"],
                "validation_macro_f1": result["macro_f1"],
                "validation_top3_accuracy": result["top3_accuracy"],
            }
        )
        key = (result["macro_f1"], result["accuracy"])
        if selected is None or key > selected[0]:
            selected = (key, c_value, candidate)
    assert selected is not None
    _, selected_c, pipeline = selected
    classes = [str(code) for code in pipeline.named_steps["classifier"].classes_]
    if not all(LABEL_PATTERN.fullmatch(code) for code in classes):
        raise AssertionError("The fitted classifier emitted a non-six-digit class.")

    catalog: list[dict[str, Any]] = []
    for row in catalog_rows:
        catalog.append(
            {
                "masco_code": row["masco_code"],
                "masco_code_printed": row["masco_code_printed"],
                "role_title": row["role_title"],
                "source_parent_group_code": row["source_parent_group_code"],
                "flexible_role": row["flexible_role"].strip().lower() == "true",
                "profile_text": row["profile_text"],
            }
        )

    catalog_codes = {row["masco_code"] for row in catalog}
    comparison_index: dict[str, dict[str, dict[str, Any]]] = {}
    for row in crosswalk_rows:
        esco_code = row["target_esco_code"].strip()
        if not ESCO_PATTERN.fullmatch(esco_code):
            raise ValueError(f"Invalid ESCO comparison code: {esco_code!r}")
        related_masco_codes = {
            row["premerge_masco_code"].strip(),
            row["masco_code"].strip(),
        }
        if not all(LABEL_PATTERN.fullmatch(code) for code in related_masco_codes):
            raise ValueError("Every MASCO code in the ESCO comparison crosswalk must be six digits.")
        if not related_masco_codes.issubset(catalog_codes):
            raise ValueError("An ESCO comparison references a MASCO role outside the D11 catalog.")
        for masco_code in related_masco_codes:
            comparisons = comparison_index.setdefault(masco_code, {})
            comparison = comparisons.setdefault(
                esco_code,
                {
                    "esco_code": esco_code,
                    "esco_title": row["target_esco_title"].strip(),
                    "jobhop_examples": 0,
                    "crosswalk_method": row["crosswalk_method"].strip(),
                    "crosswalk_authority": row["crosswalk_authority"].strip(),
                    "review_status": row["review_status"].strip(),
                },
            )
            comparison["jobhop_examples"] += int(row["total_examples"])

    esco_comparisons_by_masco = {
        masco_code: sorted(
            comparisons.values(),
            key=lambda item: (-int(item["jobhop_examples"]), item["esco_code"]),
        )
        for masco_code, comparisons in comparison_index.items()
    }
    fallback_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        max_features=80000,
    )
    fallback_matrix = fallback_vectorizer.fit_transform(
        [row["profile_text"] for row in catalog]
    )

    val_probabilities = pipeline.predict_proba(val_texts)
    val_predictions = np.asarray(classes)[val_probabilities.argmax(axis=1)]
    val_confidence = val_probabilities.max(axis=1)
    val_fallback_scores = fallback_vectorizer.transform(val_texts) @ fallback_matrix.T
    val_fallback_indexes = np.asarray(val_fallback_scores.argmax(axis=1)).ravel()
    val_fallback = np.asarray([catalog[index]["masco_code"] for index in val_fallback_indexes])
    threshold_results: list[dict[str, Any]] = []
    for threshold in np.arange(0.0, 0.91, 0.05):
        final = np.where(val_confidence >= threshold, val_predictions, val_fallback)
        threshold_results.append(
            {
                "threshold": round(float(threshold), 2),
                "classifier_coverage": float(np.mean(val_confidence >= threshold)),
                "two_tier_accuracy": float(accuracy_score(val_labels, final)),
            }
        )
    threshold_results.sort(
        key=lambda row: (row["two_tier_accuracy"], row["threshold"]), reverse=True
    )
    low_confidence_threshold = float(threshold_results[0]["threshold"])

    encoder = SentenceTransformer(MODEL_NAME, local_files_only=args.local_files_only)
    all_texts = [row["text"] for row in examples]
    all_vectors = encoder.encode(
        all_texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    all_labels = np.asarray([row["masco_code"] for row in examples])
    all_splits = np.asarray([row["split"] for row in examples])
    centroids: dict[str, np.ndarray] = {}
    for code in classes:
        centroid = all_vectors[(all_splits == "train") & (all_labels == code)].mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[code] = np.asarray(centroid / norm if norm else centroid, dtype=float)

    minilm_metrics: dict[str, Any] = {}
    for split, labels in (("val", val_labels), ("test", test_labels)):
        indexes = np.flatnonzero(all_splits == split)
        minilm_metrics[split] = retrieval_metrics(
            all_vectors[indexes], labels, classes, centroids
        )

    validation = classifier_metrics(pipeline, val_texts, val_labels)
    test = classifier_metrics(pipeline, test_texts, test_labels)
    selected_research_model = (
        "minilm_centroid_retrieval"
        if minilm_metrics["val"]["macro_f1"] > validation["macro_f1"]
        else "word_char_tfidf_balanced_logistic_regression"
    )
    metrics = {
        "task": "D12 six-digit MASCO occupation classifier",
        "resume_dataset_policy": "JobHop v2 confirmed-active 2019+ is the sole resume/career-history dataset.",
        "jobhop_source_sha256": source_hash,
        "label_rule": "Every model class, catalog key, and prediction is exactly six digits.",
        "training_examples": len(examples),
        "split_counts": {
            split: int(np.sum(all_splits == split)) for split in ("train", "val", "test")
        },
        "class_count": len(classes),
        "catalog_size": len(catalog),
        "esco_comparison_crosswalk_rows": len(crosswalk_rows),
        "masco_roles_with_esco_comparisons": len(esco_comparisons_by_masco),
        "esco_comparison_policy": (
            "ESCO codes and titles are retained as project-crosswalk comparison metadata; "
            "the predicted label remains an exact six-digit MASCO code."
        ),
        "selected_C": selected_c,
        "grid_results": grid_results,
        "low_confidence_threshold": low_confidence_threshold,
        "validation": validation,
        "test": test,
        "minilm_centroid_benchmark": minilm_metrics,
        "selected_research_model": selected_research_model,
        "selection_metric": "validation macro-F1",
        "scikit_learn_version": sklearn.__version__,
        "embedding_model": MODEL_NAME,
        "raw_resume_text_available": False,
        "deployment_input_adapter": (
            "The API converts extracted title, skills, and duration into the D12 "
            "structured JobHop history-text format."
        ),
        "production_approved": False,
        "human_confirmation_required": True,
        "production_blockers": [
            "JobHop contains structured Belgian/Flemish career histories rather than raw Malaysian CV text.",
            "The project ESCO-to-six-digit-MASCO crosswalk requires domain-owner review.",
            "Sparse granular labels are merged to six-digit curated anchors.",
        ],
    }
    artifact = {
        "format_version": 4,
        "task": metrics["task"],
        "label_regex": r"^\d{6}$",
        "resume_dataset": "JobHop v2 confirmed active 2019+ only",
        "resume_dataset_sha256": source_hash,
        "pipeline": pipeline,
        "fallback_vectorizer": fallback_vectorizer,
        "fallback_catalog_matrix": fallback_matrix,
        "catalog": catalog,
        "esco_comparisons_by_masco": esco_comparisons_by_masco,
        "classes": classes,
        "low_confidence_threshold": low_confidence_threshold,
        "embedding_model": MODEL_NAME,
        "embedding_class_centroids": centroids,
        "selected_research_model": selected_research_model,
        "selection_metric": "validation macro-F1",
        "compose_text_format": "structured prior career history; no raw CV text",
        "sparsity_policy": "Premerge classes with fewer than 5 train examples merge to a curated six-digit anchor in the same MASCO parent group.",
        "random_state": RANDOM_STATE,
        "scikit_learn_version": sklearn.__version__,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
