from __future__ import annotations

import os
import re
import time
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.cv_parser import MAX_FILE_BYTES, CvParseError, parse_cv
from api.masco_matcher import GranularMascoMatcher


@lru_cache(maxsize=1)
def get_matcher() -> GranularMascoMatcher:
    return GranularMascoMatcher()


app = FastAPI(
    title="ReRouteHer CV → six-digit MASCO Feasibility API",
    version="3.0.0",
    description="Upload one CV and compare JobHop-trained TF-IDF and MiniLM six-digit MASCO suggestions.",
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
        matcher = get_matcher()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc
    return {
        "status": "ok",
        "taxonomy": "MASCO 2020 granular occupations",
        "label_format": "six digits",
        "label_regex": r"^\d{6}$",
        "catalog_roles": len(matcher.catalog),
        "trainable_classes": len(matcher.classes),
        "resume_dataset": matcher.artifact["resume_dataset"],
        "resume_dataset_sha256": matcher.artifact["resume_dataset_sha256"],
        "models": [
            "word_char_tfidf_balanced_logistic_regression",
            "minilm_jobhop_class_centroid_retrieval",
        ],
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
        parsed = parse_cv(filename, content)
    except CvParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    parse_ms = (time.perf_counter() - parse_started) * 1000

    tfidf_started = time.perf_counter()
    tfidf = get_matcher().predict_tfidf(
        job_title=parsed.job_title,
        skills=parsed.skills,
        work_length_years=parsed.work_length_years,
        top_k=3,
    )
    tfidf_ms = (time.perf_counter() - tfidf_started) * 1000

    minilm_started = time.perf_counter()
    minilm = get_matcher().predict_minilm(
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
        "model_policy": {
            "taxonomy": "MASCO 2020 granular occupations",
            "label_format": "exactly six digits; printed hyphen form is display-only",
            "resume_dataset": get_matcher().artifact["resume_dataset"],
            "human_confirmation_required": True,
            "automatic_employment_decision_use": False,
        },
        "tfidf": tfidf,
        "minilm": minilm,
    }
