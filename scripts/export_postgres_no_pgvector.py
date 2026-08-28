#!/usr/bin/env python3
"""Build a self-contained PostgreSQL data import without pgvector.

The generated dump uses only core PostgreSQL types. MiniLM embeddings are stored
as 384-dimensional ``real[]`` values, preserving them without requiring the
``vector`` extension.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TextIO

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "database" / "rerouteher_no_pgvector.sql.gz"
NULL = r"\N"


def open_csv(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def nullable(value: str | None, convert: Callable[[str], Any] | None = None) -> Any:
    if value is None or value == "":
        return NULL
    return convert(value) if convert else value


def integer_from_decimal(value: str) -> int:
    return int(float(value))


def write_copy(
    handle: TextIO,
    table: str,
    columns: list[str],
    rows: Iterable[Iterable[Any]],
) -> int:
    handle.write(
        f"COPY rerouteher.{table} ({', '.join(columns)}) FROM STDIN "
        "WITH (FORMAT csv, NULL '\\N', QUOTE '\"', ESCAPE '\"');\n"
    )
    writer = csv.writer(handle, lineterminator="\n")
    count = 0
    for row in rows:
        writer.writerow(row)
        count += 1
    handle.write("\\.\n\n")
    return count


def profile_rows(path: Path) -> Iterator[list[Any]]:
    for row in open_csv(path):
        yield [
            row["esco_code"],
            row["esco_title"],
            row["normalized_title"],
            nullable(row["esco_occupation_uri"]),
            nullable(row["esco_title_matrix"]),
            nullable(row["top_skill_groups"]),
            nullable(row["top_skill_groups_weighted"]),
            nullable(row["top_skill_count"], integer_from_decimal),
            row["mapping_status"],
            nullable(row["masco_candidate_code"]),
            nullable(row["masco_unit_group_title_en"]),
            row["masco_mapping_status"],
            int(row["jobhop_record_count"]),
            nullable(row["work_length_median_years"], float),
            nullable(row["work_length_mean_years"], float),
            nullable(row["work_length_p25_years"], float),
            nullable(row["work_length_p75_years"], float),
        ]


def skill_rows(path: Path) -> Iterator[list[Any]]:
    for row in open_csv(path):
        yield [
            row["esco_code"],
            row["esco_occupation_uri"],
            row["esco_title"],
            row["skill_group_uri"],
            row["skill_group_label"],
            float(row["matrix_weight"]),
            int(row["skill_rank"]),
            row["source_version"],
        ]


def jobhop_rows(path: Path) -> Iterator[list[Any]]:
    for feature_id, row in enumerate(open_csv(path), start=1):
        yield [
            feature_id,
            int(row["resume_id"]),
            row["matched_code"],
            row["start_date"],
            row["end_date"],
            row["university_level"],
            row["split"],
            int(row["start_year"]),
            int(row["end_year"]),
            float(row["work_length_years_est"]),
            int(row["work_length_months_est"]),
            row["work_length_band"],
            int(row["resume_job_count"]),
            float(row["resume_total_role_years_est"]),
            int(row["resume_first_start_year"]),
            int(row["resume_last_end_year"]),
            float(row["resume_span_years_est"]),
            nullable(row["esco_code"]),
            nullable(row["esco_title"]),
            nullable(row["esco_occupation_uri"]),
            nullable(row["top_skill_groups"]),
            nullable(row["top_skill_groups_weighted"]),
            nullable(row["top_skill_count"], integer_from_decimal),
            row["mapping_status"],
            nullable(row["masco_candidate_code"]),
            nullable(row["masco_unit_group_title_en"]),
            nullable(row["masco_mapping_status"]),
            nullable(row["skill_feature_text"]),
        ]


def embedding_rows(catalog_path: Path, embeddings_path: Path) -> Iterator[list[Any]]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.shape != (len(catalog), 384):
        raise ValueError(
            f"Expected {len(catalog)} x 384 embeddings, received {embeddings.shape}."
        )
    codes = [item["esco_code"] for item in catalog]
    if len(codes) != len(set(codes)):
        raise ValueError("MiniLM catalog contains duplicate ESCO codes.")

    for index, (profile, vector) in enumerate(zip(catalog, embeddings, strict=True)):
        array_literal = "{" + ",".join(format(float(value), ".9g") for value in vector) + "}"
        yield [
            profile["esco_code"],
            index,
            "sentence-transformers/all-MiniLM-L6-v2",
            384,
            True,
            array_literal,
        ]


SCHEMA_SQL = r"""\set ON_ERROR_STOP on

BEGIN;

CREATE SCHEMA IF NOT EXISTS rerouteher;

CREATE TABLE rerouteher.dataset_metadata (
    metadata_key text PRIMARY KEY,
    metadata_value jsonb NOT NULL
);

CREATE TABLE rerouteher.occupation_profiles (
    esco_code text PRIMARY KEY,
    esco_title text NOT NULL,
    normalized_title text NOT NULL,
    esco_occupation_uri text UNIQUE,
    esco_title_matrix text,
    top_skill_groups text,
    top_skill_groups_weighted text,
    top_skill_count integer CHECK (top_skill_count >= 0),
    mapping_status text NOT NULL,
    masco_candidate_code text,
    masco_unit_group_title_en text,
    masco_mapping_status text NOT NULL,
    jobhop_record_count integer NOT NULL CHECK (jobhop_record_count >= 0),
    work_length_median_years double precision CHECK (work_length_median_years >= 0),
    work_length_mean_years double precision CHECK (work_length_mean_years >= 0),
    work_length_p25_years double precision CHECK (work_length_p25_years >= 0),
    work_length_p75_years double precision CHECK (work_length_p75_years >= 0)
);

CREATE TABLE rerouteher.occupation_skill_weights (
    esco_code text NOT NULL REFERENCES rerouteher.occupation_profiles(esco_code),
    esco_occupation_uri text NOT NULL,
    esco_title text NOT NULL,
    skill_group_uri text NOT NULL,
    skill_group_label text NOT NULL,
    matrix_weight real NOT NULL CHECK (matrix_weight >= 0),
    skill_rank integer NOT NULL CHECK (skill_rank > 0),
    source_version text NOT NULL,
    PRIMARY KEY (esco_code, skill_group_uri)
);

CREATE TABLE rerouteher.jobhop_esco_features (
    jobhop_feature_id bigint PRIMARY KEY,
    resume_id bigint NOT NULL,
    matched_code text NOT NULL,
    start_date text NOT NULL,
    end_date text NOT NULL,
    university_level text NOT NULL,
    split text NOT NULL CHECK (split IN ('train', 'val', 'test')),
    start_year integer NOT NULL,
    end_year integer NOT NULL,
    work_length_years_est double precision NOT NULL CHECK (work_length_years_est >= 0),
    work_length_months_est integer NOT NULL CHECK (work_length_months_est >= 0),
    work_length_band text NOT NULL,
    resume_job_count integer NOT NULL CHECK (resume_job_count > 0),
    resume_total_role_years_est double precision NOT NULL CHECK (resume_total_role_years_est >= 0),
    resume_first_start_year integer NOT NULL,
    resume_last_end_year integer NOT NULL,
    resume_span_years_est double precision NOT NULL CHECK (resume_span_years_est >= 0),
    esco_code text REFERENCES rerouteher.occupation_profiles(esco_code),
    esco_title text,
    esco_occupation_uri text,
    top_skill_groups text,
    top_skill_groups_weighted text,
    top_skill_count integer CHECK (top_skill_count >= 0),
    mapping_status text NOT NULL,
    masco_candidate_code text,
    masco_unit_group_title_en text,
    masco_mapping_status text,
    skill_feature_text text,
    CHECK (end_year >= start_year),
    CHECK (resume_last_end_year >= resume_first_start_year)
);

CREATE TABLE rerouteher.minilm_embeddings (
    esco_code text PRIMARY KEY REFERENCES rerouteher.occupation_profiles(esco_code),
    embedding_index integer NOT NULL UNIQUE CHECK (embedding_index >= 0),
    model_name text NOT NULL,
    dimensions integer NOT NULL CHECK (dimensions = 384),
    l2_normalized boolean NOT NULL,
    embedding real[] NOT NULL,
    CHECK (cardinality(embedding) = dimensions)
);

"""


POST_DATA_SQL = r"""CREATE INDEX occupation_profiles_normalized_title_idx
    ON rerouteher.occupation_profiles (normalized_title);
CREATE INDEX occupation_profiles_masco_code_idx
    ON rerouteher.occupation_profiles (masco_candidate_code);
CREATE INDEX occupation_profiles_search_idx
    ON rerouteher.occupation_profiles USING gin (
        to_tsvector('simple', coalesce(esco_title, '') || ' ' || coalesce(top_skill_groups, ''))
    );
CREATE INDEX occupation_skill_weights_rank_idx
    ON rerouteher.occupation_skill_weights (esco_code, skill_rank);
CREATE INDEX occupation_skill_weights_skill_uri_idx
    ON rerouteher.occupation_skill_weights (skill_group_uri);
CREATE INDEX jobhop_esco_features_resume_idx
    ON rerouteher.jobhop_esco_features (resume_id);
CREATE INDEX jobhop_esco_features_esco_idx
    ON rerouteher.jobhop_esco_features (esco_code);
CREATE INDEX jobhop_esco_features_matched_code_idx
    ON rerouteher.jobhop_esco_features (matched_code);
CREATE INDEX jobhop_esco_features_split_idx
    ON rerouteher.jobhop_esco_features (split);

CREATE FUNCTION rerouteher.cosine_similarity(left_vector real[], right_vector real[])
RETURNS double precision
LANGUAGE sql
IMMUTABLE
STRICT
PARALLEL SAFE
AS $$
    SELECT sum(left_value::double precision * right_value::double precision)
           / NULLIF(
               sqrt(sum(left_value::double precision * left_value::double precision))
               * sqrt(sum(right_value::double precision * right_value::double precision)),
               0
           )
    FROM unnest(left_vector, right_vector) AS pairs(left_value, right_value);
$$;

CREATE VIEW rerouteher.database_counts AS
SELECT 'occupation_profiles'::text AS relation_name, count(*)::bigint AS row_count
FROM rerouteher.occupation_profiles
UNION ALL
SELECT 'occupation_skill_weights', count(*) FROM rerouteher.occupation_skill_weights
UNION ALL
SELECT 'jobhop_esco_features', count(*) FROM rerouteher.jobhop_esco_features
UNION ALL
SELECT 'minilm_embeddings', count(*) FROM rerouteher.minilm_embeddings;

ANALYZE rerouteher.occupation_profiles;
ANALYZE rerouteher.occupation_skill_weights;
ANALYZE rerouteher.jobhop_esco_features;
ANALYZE rerouteher.minilm_embeddings;

COMMIT;
"""


PROFILE_COLUMNS = [
    "esco_code",
    "esco_title",
    "normalized_title",
    "esco_occupation_uri",
    "esco_title_matrix",
    "top_skill_groups",
    "top_skill_groups_weighted",
    "top_skill_count",
    "mapping_status",
    "masco_candidate_code",
    "masco_unit_group_title_en",
    "masco_mapping_status",
    "jobhop_record_count",
    "work_length_median_years",
    "work_length_mean_years",
    "work_length_p25_years",
    "work_length_p75_years",
]

SKILL_COLUMNS = [
    "esco_code",
    "esco_occupation_uri",
    "esco_title",
    "skill_group_uri",
    "skill_group_label",
    "matrix_weight",
    "skill_rank",
    "source_version",
]

JOBHOP_COLUMNS = [
    "jobhop_feature_id",
    "resume_id",
    "matched_code",
    "start_date",
    "end_date",
    "university_level",
    "split",
    "start_year",
    "end_year",
    "work_length_years_est",
    "work_length_months_est",
    "work_length_band",
    "resume_job_count",
    "resume_total_role_years_est",
    "resume_first_start_year",
    "resume_last_end_year",
    "resume_span_years_est",
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
    "skill_feature_text",
]

EMBEDDING_COLUMNS = [
    "esco_code",
    "embedding_index",
    "model_name",
    "dimensions",
    "l2_normalized",
    "embedding",
]


def build(output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    profile_path = ROOT / "data" / "processed" / "esco_occupation_profiles.csv"
    skill_path = ROOT / "data" / "processed" / "esco_occupation_skill_weights.csv"
    jobhop_path = ROOT / "data" / "processed" / "jobhop_esco_features.csv.gz"
    catalog_path = ROOT / "model" / "minilm" / "catalog.json"
    embeddings_path = ROOT / "model" / "minilm" / "profile_embeddings.npy"

    metadata = {
        "data_quality_report": json.loads(
            (ROOT / "data" / "processed" / "data_quality_report.json").read_text(encoding="utf-8")
        ),
        "tfidf_model_metrics": json.loads((ROOT / "model" / "metrics.json").read_text(encoding="utf-8")),
        "minilm_model_metrics": json.loads(
            (ROOT / "model" / "minilm" / "metrics.json").read_text(encoding="utf-8")
        ),
        "import_format": {
            "schema": "rerouteher",
            "postgres_extensions_required": [],
            "embedding_storage": "real[384]",
            "source_repository": "CharlesYi-DEV/rerouteher-esco-tfidf",
        },
    }

    counts: dict[str, int] = {}
    with output.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="") as handle:
                handle.write("-- ReRouteHer complete PostgreSQL import (no pgvector required)\n")
                handle.write("-- Generated from the repository's processed data and model artifacts.\n\n")
                handle.write(SCHEMA_SQL)
                counts["dataset_metadata"] = write_copy(
                    handle,
                    "dataset_metadata",
                    ["metadata_key", "metadata_value"],
                    (
                        [key, json.dumps(value, ensure_ascii=False, separators=(",", ":"))]
                        for key, value in metadata.items()
                    ),
                )
                counts["occupation_profiles"] = write_copy(
                    handle, "occupation_profiles", PROFILE_COLUMNS, profile_rows(profile_path)
                )
                counts["occupation_skill_weights"] = write_copy(
                    handle, "occupation_skill_weights", SKILL_COLUMNS, skill_rows(skill_path)
                )
                counts["jobhop_esco_features"] = write_copy(
                    handle, "jobhop_esco_features", JOBHOP_COLUMNS, jobhop_rows(jobhop_path)
                )
                counts["minilm_embeddings"] = write_copy(
                    handle,
                    "minilm_embeddings",
                    EMBEDDING_COLUMNS,
                    embedding_rows(catalog_path, embeddings_path),
                )
                handle.write(POST_DATA_SQL)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    try:
        checksum_target = output.relative_to(ROOT)
    except ValueError:
        checksum_target = Path(output.name)
    checksum_path.write_text(f"{digest}  {checksum_target}\n", encoding="utf-8")
    return {
        "output": str(output),
        "compressed_bytes": output.stat().st_size,
        "sha256": digest,
        "counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
