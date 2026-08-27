from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from api.matcher import TfidfEscoMatcher


class MatchRequest(BaseModel):
    job_title: str = Field(min_length=2, max_length=160)
    skills: list[str] = Field(default_factory=list, max_length=60)
    work_length_years: float | None = Field(default=None, ge=0, le=80)
    top_k: int = Field(default=3, ge=1, le=5)

    @field_validator("skills")
    @classmethod
    def clean_skills(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


@lru_cache(maxsize=1)
def get_matcher() -> TfidfEscoMatcher:
    return TfidfEscoMatcher()


app = FastAPI(
    title="ReRouteHer ESCO Matcher — TF-IDF",
    version="1.0.0",
    description="Job title and skills to ESCO occupation using character/word TF-IDF and Logistic Regression.",
)

origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    try:
        get_matcher()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc
    return {"status": "ok", "model": "tfidf_logreg"}


@app.post("/match")
def match(payload: MatchRequest) -> dict:
    return get_matcher().predict(
        job_title=payload.job_title,
        skills=payload.skills,
        work_length_years=payload.work_length_years,
        top_k=payload.top_k,
    )
