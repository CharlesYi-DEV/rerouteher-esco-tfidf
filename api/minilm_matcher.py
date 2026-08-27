from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz.fuzz import ratio
from sentence_transformers import SentenceTransformer


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


def compose_text(job_title: str, skills: list[str]) -> str:
    title = normalize_text(job_title)
    skill_text = ", ".join(normalize_text(skill) for skill in skills if skill.strip())
    return f"Occupation title: {title}. Relevant skills and work areas: {skill_text}."


class MiniLmEscoMatcher:
    def __init__(self, model_dir: str | Path | None = None) -> None:
        root = Path(model_dir or os.getenv("MINILM_MODEL_DIR", Path(__file__).parents[1] / "model" / "minilm"))
        self.catalog: list[dict[str, Any]] = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
        self.embeddings = np.load(root / "profile_embeddings.npy")
        self.model_name = os.getenv("SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name)

    @staticmethod
    def _duration_fit(years: float | None, profile: dict[str, Any]) -> float | None:
        if years is None:
            return None
        median = profile.get("work_length_median_years")
        if median is None or not math.isfinite(float(median)):
            return None
        return round(math.exp(-abs(years - float(median)) / 5), 4)

    @staticmethod
    def _matched_skills(query_skills: list[str], profile: dict[str, Any]) -> list[str]:
        query = normalize_text(" ".join(query_skills))
        available = [item.strip() for item in str(profile.get("top_skill_groups", "")).split("|") if item.strip()]
        ranked = sorted(((ratio(query, normalize_text(item)), item) for item in available), reverse=True)
        return [item for score, item in ranked[:6] if score >= 20]

    def predict(
        self,
        job_title: str,
        skills: list[str],
        work_length_years: float | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        normalized_title = normalize_text(job_title)
        query = compose_text(job_title, skills)
        embedding = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        semantic_scores = self.embeddings @ embedding
        title_scores = np.array([
            ratio(normalized_title, normalize_text(str(profile["esco_title"]))) / 100 for profile in self.catalog
        ])
        scores = 0.90 * semantic_scores + 0.10 * title_scores
        if work_length_years is not None:
            duration_scores = np.array([
                self._duration_fit(work_length_years, profile) or 0.0 for profile in self.catalog
            ])
            scores = 0.97 * scores + 0.03 * duration_scores
        order = np.argsort(scores)[::-1][:top_k]

        matches = []
        for idx in order:
            profile = self.catalog[int(idx)]
            matches.append(
                {
                    "esco_code": profile["esco_code"],
                    "esco_title": profile["esco_title"],
                    "esco_uri": profile.get("esco_occupation_uri"),
                    "masco_candidate_code": profile.get("masco_candidate_code"),
                    "masco_mapping_status": profile.get("masco_mapping_status", "not_evaluated"),
                    "score": round(float(scores[idx]), 4),
                    "semantic_score": round(float(semantic_scores[idx]), 4),
                    "title_similarity": round(float(title_scores[idx]), 4),
                    "method": "all_minilm_l6_v2_semantic_retrieval",
                    "matched_skills": self._matched_skills(skills, profile),
                    "reference_work_length_median_years": profile.get("work_length_median_years"),
                    "work_length_fit": self._duration_fit(work_length_years, profile),
                }
            )
        return {
            "model": self.model_name,
            "method_used": "all_minilm_l6_v2_semantic_retrieval",
            "input": {"job_title": job_title, "skills": skills, "work_length_years": work_length_years},
            "matches": matches,
        }
