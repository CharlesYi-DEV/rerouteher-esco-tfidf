from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


LABEL_PATTERN = re.compile(r"^\d{6}$")
EXPECTED_RESUME_DATASET = "JobHop v2 confirmed active 2019+ only"

ALIASES = {
    r"\bhr\b": "human resources",
    r"\bhuman resource\b": "human resources",
    r"\badmin\b": "administrative",
    r"\bdev\b": "developer",
    r"\bsoftware engineer\b": "software developer",
    r"\bprogram+m?er\b": "software developer",
    r"\bcoder\b": "software developer",
    r"\bui[ /-]*ux\b": "user interface user experience",
    r"\bcsr\b": "customer service representative",
    r"\bsales rep\b": "sales representative",
    r"\bdata scientist\b": "data analyst",
    r"\baccounts assistant\b": "accounting assistant",
    r"\bwarehous(?:e)? operat(?:o)?r\b": "warehouse worker",
}


def normalize_text(value: str) -> str:
    text = value.lower().strip()
    for pattern, replacement in ALIASES.items():
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text)


def compose_text(
    job_title: str, skills: list[str], work_length_years: float | None
) -> str:
    """Adapt extracted CV features to the D12 JobHop history-text format."""
    parts = [f"title {normalize_text(job_title)}"]
    normalized_skills = [normalize_text(skill) for skill in skills if skill.strip()]
    if normalized_skills:
        parts.append("skills " + ", ".join(normalized_skills))
    if work_length_years is not None:
        parts.append(f"duration {max(0.0, float(work_length_years)):.2f} years")
    return " ; ".join(parts)


class GranularMascoMatcher:
    def __init__(
        self,
        artifact_path: str | Path | None = None,
        encoder: Any | None = None,
        load_encoder: bool = True,
    ) -> None:
        path = Path(
            artifact_path
            or os.environ.get(
                "MASCO_MODEL_PATH",
                Path(__file__).parents[1]
                / "model"
                / "masco"
                / "d12_granular_masco_classifier.joblib",
            )
        )
        self.artifact: dict[str, Any] = joblib.load(path)
        self._validate_artifact()
        self.pipeline = self.artifact["pipeline"]
        self.fallback_vectorizer = self.artifact["fallback_vectorizer"]
        self.fallback_catalog_matrix = self.artifact["fallback_catalog_matrix"]
        self.catalog: list[dict[str, Any]] = self.artifact["catalog"]
        self.catalog_by_code = {row["masco_code"]: row for row in self.catalog}
        self.classes = [str(code) for code in self.artifact["classes"]]
        self.low_confidence_threshold = float(
            self.artifact["low_confidence_threshold"]
        )
        self.centroid_matrix = np.vstack(
            [self.artifact["embedding_class_centroids"][code] for code in self.classes]
        )
        self.model_name = str(self.artifact["embedding_model"])
        self.encoder = encoder
        if self.encoder is None and load_encoder:
            local_only = os.getenv("SENTENCE_MODEL_LOCAL_ONLY", "true").lower() not in {
                "0",
                "false",
                "no",
            }
            self.encoder = SentenceTransformer(
                self.model_name, local_files_only=local_only
            )

    def _validate_artifact(self) -> None:
        required = {
            "pipeline",
            "fallback_vectorizer",
            "fallback_catalog_matrix",
            "catalog",
            "classes",
            "embedding_model",
            "embedding_class_centroids",
            "resume_dataset",
            "resume_dataset_sha256",
            "label_regex",
        }
        missing = required - set(self.artifact)
        if missing:
            raise ValueError(f"MASCO artifact is missing keys: {sorted(missing)}")
        if self.artifact["resume_dataset"] != EXPECTED_RESUME_DATASET:
            raise ValueError("The model was not trained from the approved JobHop-only scope.")
        if self.artifact["label_regex"] != r"^\d{6}$":
            raise ValueError("The model does not declare the six-digit MASCO label rule.")
        codes = [str(row.get("masco_code", "")) for row in self.artifact["catalog"]]
        classes = [str(code) for code in self.artifact["classes"]]
        if len(codes) != 258 or len(set(codes)) != 258:
            raise ValueError("The D11 catalog must contain 258 unique MASCO roles.")
        if not all(LABEL_PATTERN.fullmatch(code) for code in codes + classes):
            raise ValueError("Every MASCO catalog key and model class must be six digits.")
        if not set(classes).issubset(codes):
            raise ValueError("A trained six-digit class is missing from the MASCO catalog.")

    @staticmethod
    def _matched_skills(
        query_skills: list[str], profile: dict[str, Any]
    ) -> list[str]:
        profile_text = normalize_text(str(profile.get("profile_text", "")))
        matches: list[str] = []
        for skill in query_skills:
            normalized = normalize_text(skill)
            tokens = set(re.findall(r"[a-z0-9]+", normalized))
            if normalized in profile_text or len(tokens & set(profile_text.split())) >= 2:
                matches.append(skill)
        return matches[:6]

    def _format_match(
        self,
        code: str,
        score: float,
        method: str,
        query_skills: list[str],
        rank_scope: str,
        semantic_score: float | None = None,
    ) -> dict[str, Any]:
        if not LABEL_PATTERN.fullmatch(code):
            raise ValueError(f"Model emitted a non-six-digit MASCO code: {code!r}")
        profile = self.catalog_by_code[code]
        result = {
            "masco_code": code,
            "masco_code_printed": profile["masco_code_printed"],
            "masco_title": profile["role_title"],
            "source_parent_group_code": profile["source_parent_group_code"],
            "score": round(float(score), 4),
            "method": method,
            "rank_scope": rank_scope,
            "matched_skills": self._matched_skills(query_skills, profile),
            "requires_user_confirmation": True,
        }
        if semantic_score is not None:
            result["semantic_score"] = round(float(semantic_score), 4)
        return result

    def predict_tfidf(
        self,
        job_title: str,
        skills: list[str],
        work_length_years: float | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        text = compose_text(job_title, skills, work_length_years)
        probabilities = self.pipeline.predict_proba([text])[0]
        classes = [str(code) for code in self.pipeline.named_steps["classifier"].classes_]
        order = np.argsort(probabilities)[::-1]

        if float(probabilities[order[0]]) >= self.low_confidence_threshold:
            matches = [
                self._format_match(
                    classes[index],
                    probabilities[index],
                    "word_char_tfidf_logistic_regression",
                    skills,
                    "19 trainable six-digit classes",
                )
                for index in order[:top_k]
            ]
            method = "word_char_tfidf_logistic_regression"
        else:
            query = self.fallback_vectorizer.transform([text])
            scores = (query @ self.fallback_catalog_matrix.T).toarray()[0]
            fallback_order = np.argsort(scores)[::-1][:top_k]
            matches = [
                self._format_match(
                    str(self.catalog[index]["masco_code"]),
                    scores[index],
                    "character_tfidf_catalog_fallback",
                    skills,
                    "all 258 six-digit catalog roles",
                )
                for index in fallback_order
            ]
            method = "character_tfidf_catalog_fallback"

        return {
            "model": "word_char_tfidf_balanced_logistic_regression",
            "method_used": method,
            "label_format": "six-digit MASCO",
            "low_confidence_threshold": self.low_confidence_threshold,
            "input": {
                "job_title": job_title,
                "skills": skills,
                "work_length_years": work_length_years,
                "structured_text": text,
            },
            "matches": matches,
        }

    def predict_minilm(
        self,
        job_title: str,
        skills: list[str],
        work_length_years: float | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        if self.encoder is None:
            raise RuntimeError("MiniLM encoder was not loaded.")
        text = compose_text(job_title, skills, work_length_years)
        vector = self.encoder.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        scores = self.centroid_matrix @ vector
        order = np.argsort(scores)[::-1][:top_k]
        matches = [
            self._format_match(
                self.classes[index],
                scores[index],
                "minilm_jobhop_class_centroid_retrieval",
                skills,
                "19 trainable six-digit classes",
                semantic_score=scores[index],
            )
            for index in order
        ]
        return {
            "model": self.model_name,
            "method_used": "minilm_jobhop_class_centroid_retrieval",
            "label_format": "six-digit MASCO",
            "input": {
                "job_title": job_title,
                "skills": skills,
                "work_length_years": work_length_years,
                "structured_text": text,
            },
            "matches": matches,
        }
