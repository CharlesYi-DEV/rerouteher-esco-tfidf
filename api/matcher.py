from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


ALIASES = {
    r"\bhr\b": "human resources",
    r"\bhuman resource\b": "human resources",
    r"\badmin\b": "administrative",
    r"\bdev\b": "developer",
    r"\bsoftware engineer\b": "software developer",
    r"\bprogram+m?er\b": "software developer",
    r"\bcoder\b": "software developer",
    r"\bsoftware craft(?:s)?person\b": "software developer",
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


def duration_band(years: float | None) -> str:
    if years is None:
        return "work_length_unknown"
    if years < 1:
        return "work_length_under_1_year"
    if years < 3:
        return "work_length_1_to_3_years"
    if years < 5:
        return "work_length_3_to_5_years"
    if years < 10:
        return "work_length_5_to_10_years"
    return "work_length_10_plus_years"


def compose_text(job_title: str, skills: list[str], work_length_years: float | None) -> str:
    title = normalize_text(job_title)
    skill_text = ", ".join(normalize_text(skill) for skill in skills if skill.strip())
    return f"title {title}. job {title}. occupation {title}. skills {skill_text}. {duration_band(work_length_years)}"


class TfidfEscoMatcher:
    def __init__(self, artifact_path: str | Path | None = None) -> None:
        path = Path(
            artifact_path
            or os.environ.get("MODEL_PATH", Path(__file__).parents[1] / "model" / "tfidf_logreg.joblib")
        )
        self.artifact: dict[str, Any] = joblib.load(path)
        self.pipeline = self.artifact["pipeline"]
        self.title_vectorizer = self.artifact["title_vectorizer"]
        self.title_matrix = self.artifact["title_matrix"]
        self.skill_vectorizer = self.artifact["skill_vectorizer"]
        self.skill_matrix = self.artifact["skill_matrix"]
        self.catalog = self.artifact["catalog"]
        self.catalog_by_code = {row["esco_code"]: row for row in self.catalog}
        self.low_confidence_threshold = float(self.artifact.get("low_confidence_threshold", 0.45))

    @staticmethod
    def _matched_skills(query_skills: list[str], profile: dict[str, Any]) -> list[str]:
        available = [s.strip() for s in str(profile.get("top_skill_groups", "")).split("|") if s.strip()]
        query_tokens = set(normalize_text(" ".join(query_skills)).split())
        scored = []
        for skill in available:
            overlap = len(query_tokens & set(normalize_text(skill).split()))
            if overlap:
                scored.append((overlap, skill))
        return [skill for _, skill in sorted(scored, reverse=True)[:6]]

    @staticmethod
    def _duration_fit(work_length_years: float | None, profile: dict[str, Any]) -> float | None:
        if work_length_years is None:
            return None
        median = profile.get("work_length_median_years")
        if median is None or (isinstance(median, float) and math.isnan(median)):
            return None
        return round(math.exp(-abs(float(work_length_years) - float(median)) / 5), 4)

    def _format_match(
        self,
        profile: dict[str, Any],
        score: float,
        method: str,
        query_skills: list[str],
        work_length_years: float | None,
    ) -> dict[str, Any]:
        masco_code = profile.get("masco_candidate_code")
        if isinstance(masco_code, float) and math.isnan(masco_code):
            masco_code = None
        return {
            "esco_code": profile["esco_code"],
            "esco_title": profile["esco_title"],
            "esco_uri": profile.get("esco_occupation_uri"),
            "masco_candidate_code": masco_code,
            "masco_mapping_status": profile.get("masco_mapping_status", "not_evaluated"),
            "score": round(float(score), 4),
            "method": method,
            "matched_skills": self._matched_skills(query_skills, profile),
            "reference_work_length_median_years": profile.get("work_length_median_years"),
            "work_length_fit": self._duration_fit(work_length_years, profile),
        }

    def predict(
        self,
        job_title: str,
        skills: list[str],
        work_length_years: float | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        text = compose_text(job_title, skills, work_length_years)
        probabilities = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.named_steps["classifier"].classes_
        order = np.argsort(probabilities)[::-1]
        primary_score = float(probabilities[order[0]])

        if primary_score >= self.low_confidence_threshold:
            matches = []
            for idx in order[:top_k]:
                code = str(classes[idx])
                profile = self.catalog_by_code.get(code)
                if profile:
                    matches.append(
                        self._format_match(profile, probabilities[idx], "tfidf_logreg", skills, work_length_years)
                    )
            method = "tfidf_logreg"
        else:
            title_query = self.title_vectorizer.transform([normalize_text(job_title)])
            skill_query = self.skill_vectorizer.transform([" ".join(normalize_text(s) for s in skills)])
            title_scores = cosine_similarity(title_query, self.title_matrix)[0]
            skill_scores = cosine_similarity(skill_query, self.skill_matrix)[0] if skills else np.zeros_like(title_scores)
            scores = 0.82 * title_scores + 0.18 * skill_scores
            if work_length_years is not None:
                duration_scores = np.array([
                    self._duration_fit(work_length_years, profile) or 0.0 for profile in self.catalog
                ])
                scores = 0.97 * scores + 0.03 * duration_scores
            retrieval_order = np.argsort(scores)[::-1][:top_k]
            matches = [
                self._format_match(
                    self.catalog[idx], scores[idx], "char_tfidf_retrieval_fallback", skills, work_length_years
                )
                for idx in retrieval_order
            ]
            method = "char_tfidf_retrieval_fallback"

        return {
            "model": "tfidf_char_word_logistic_regression",
            "method_used": method,
            "low_confidence_threshold": self.low_confidence_threshold,
            "input": {
                "job_title": job_title,
                "skills": skills,
                "work_length_years": work_length_years,
            },
            "matches": matches,
        }
