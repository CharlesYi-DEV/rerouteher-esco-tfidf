from __future__ import annotations

import os
import re
import time
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.cv_parser import MAX_FILE_BYTES, CvParseError, parse_cv
from api.matcher import TfidfEscoMatcher
from api.minilm_matcher import MiniLmEscoMatcher


@lru_cache(maxsize=1)
def get_tfidf_matcher() -> TfidfEscoMatcher:
    return TfidfEscoMatcher()


@lru_cache(maxsize=1)
def get_minilm_matcher() -> MiniLmEscoMatcher:
    return MiniLmEscoMatcher()


@lru_cache(maxsize=1)
def get_official_skill_groups() -> tuple[str, ...]:
    groups: set[str] = set()
    for profile in get_tfidf_matcher().catalog:
        groups.update(
            item.strip()
            for item in str(profile.get("top_skill_groups", "")).split("|")
            if item.strip()
        )
    return tuple(sorted(groups))


app = FastAPI(
    title="ReRouteHer CV → ESCO Feasibility API",
    version="2.0.0",
    description="Upload one CV and compare TF-IDF/Logistic Regression with all-MiniLM-L6-v2 ESCO matching.",
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
def health() -> dict:
    try:
        get_tfidf_matcher()
        get_minilm_matcher()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc
    return {
        "status": "ok",
        "models": ["tfidf_char_word_logistic_regression", "sentence-transformers/all-MiniLM-L6-v2"],
    }


@app.post("/match-cv")
async def match_cv(cv_file: UploadFile = File(...)) -> dict:
    started = time.perf_counter()
    filename = re.split(r"[/\\]", cv_file.filename or "uploaded_cv")[-1][:255]
    content_type = cv_file.content_type
    content = await cv_file.read(MAX_FILE_BYTES + 1)
    await cv_file.close()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded CV exceeds the 10 MB test limit.")

    parse_started = time.perf_counter()
    try:
        parsed = parse_cv(filename, content, get_official_skill_groups())
    except CvParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    parse_ms = (time.perf_counter() - parse_started) * 1000

    tfidf_started = time.perf_counter()
    tfidf = get_tfidf_matcher().predict(
        job_title=parsed.job_title,
        skills=parsed.skills,
        work_length_years=parsed.work_length_years,
        top_k=3,
    )
    tfidf_ms = (time.perf_counter() - tfidf_started) * 1000

    minilm_started = time.perf_counter()
    minilm = get_minilm_matcher().predict(
        job_title=parsed.job_title,
        skills=parsed.skills,
        work_length_years=parsed.work_length_years,
        top_k=3,
    )
    minilm_ms = (time.perf_counter() - minilm_started) * 1000

    return {
        "file": {
            "name": filename,
            "content_type": content_type,
            "size_bytes": len(content),
            "stored": False,
        },
        "extracted_features": parsed.to_dict(),
        "timings_ms": {
            "parse": round(parse_ms, 2),
            "tfidf": round(tfidf_ms, 2),
            "minilm": round(minilm_ms, 2),
            "total": round((time.perf_counter() - started) * 1000, 2),
        },
        "tfidf": tfidf,
        "minilm": minilm,
    }
