# PostgreSQL import without pgvector

`rerouteher_no_pgvector.sql.gz` is a complete, self-contained PostgreSQL import
for the processed ReRouteHer data. It requires no PostgreSQL extensions.

It creates the `rerouteher` schema with:

- `dataset_metadata`: data-quality and model-metric JSON;
- `occupation_profiles`: 3,349 ESCO occupation profiles;
- `occupation_skill_weights`: 55,265 official ESCO Matrix links;
- `jobhop_esco_features`: 47,224 processed JobHop role rows;
- `minilm_embeddings`: 2,980 normalized 384-dimensional MiniLM embeddings,
  stored as core PostgreSQL `real[]` arrays rather than pgvector values;
- `database_counts`: a verification view;
- `cosine_similarity(real[], real[])`: an exact, extension-free similarity
  function for feasibility checks.

## Import

Use a new database, or first remove an existing `rerouteher` schema yourself.
The import deliberately does not drop or overwrite existing tables.

```bash
gunzip -c database/rerouteher_no_pgvector.sql.gz | psql "$DATABASE_URL"
```

Verify the expected row counts:

```sql
SELECT * FROM rerouteher.database_counts ORDER BY relation_name;
```

Expected results:

| relation | rows |
| --- | ---: |
| `jobhop_esco_features` | 47,224 |
| `minilm_embeddings` | 2,980 |
| `occupation_profiles` | 3,349 |
| `occupation_skill_weights` | 55,265 |

Validate the downloaded/generated archive before import:

```bash
shasum -a 256 -c database/rerouteher_no_pgvector.sql.gz.sha256
```

## Rebuild

The export is deterministic for unchanged source artifacts:

```bash
.venv/bin/python scripts/export_postgres_no_pgvector.py
```

The application itself still reads the deployed TF-IDF model and MiniLM `.npy`
file directly. The database copy is for integration, audit, SQL access, and
future backend work; PostgreSQL array similarity is exact but has no approximate
nearest-neighbour index and should not replace the current NumPy inference path
for production traffic.
