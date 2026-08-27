'use client';

import { FormEvent, useState } from 'react';

type Match = {
  esco_code: string;
  esco_title: string;
  masco_candidate_code?: string | null;
  masco_mapping_status: string;
  score: number;
  semantic_score?: number;
  title_similarity?: number;
  method: string;
  matched_skills: string[];
  reference_work_length_median_years?: number | null;
};

type ModelResult = {
  model: string;
  method_used: string;
  matches: Match[];
};

type CvResult = {
  file: { name: string; content_type?: string | null; size_bytes: number; stored: boolean };
  extracted_features: {
    job_title: string;
    skills: string[];
    work_length_years?: number | null;
    total_experience_years?: number | null;
    job_title_source: string;
    skills_source: string;
    work_length_method: string;
    employment_date_ranges_found: number;
    text_characters: number;
    text_preview: string;
    warnings: string[];
  };
  timings_ms: { parse: number; tfidf: number; minilm: number; total: number };
  tfidf: ModelResult;
  minilm: ModelResult;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';

function ResultsTable({ title, result, elapsed }: { title: string; result: ModelResult; elapsed: number }) {
  return (
    <section className="model-section">
      <div className="section-heading">
        <h2>{title}</h2>
        <span>{elapsed.toFixed(1)} ms</span>
      </div>
      <p className="method">Method: {result.method_used}</p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>ESCO code</th>
              <th>Occupation</th>
              <th>Score</th>
              <th>MASCO candidate</th>
              <th>Matched skill evidence</th>
            </tr>
          </thead>
          <tbody>
            {result.matches.map((match, index) => (
              <tr key={`${title}-${match.esco_code}`}>
                <td>{index + 1}</td>
                <td><code>{match.esco_code}</code></td>
                <td>{match.esco_title}</td>
                <td>{match.score.toFixed(4)}</td>
                <td>
                  {match.masco_candidate_code
                    ? <>{match.masco_candidate_code}<br /><small>requires validation</small></>
                    : '—'}
                </td>
                <td>{match.matched_skills.length ? match.matched_skills.join('; ') : 'No direct phrase overlap'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<CvResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    setLoading(true);
    setError('');
    setResult(null);
    const body = new FormData();
    body.append('cv_file', file);
    try {
      const response = await fetch(`${API_BASE}/match-cv`, { method: 'POST', body });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? `API returned HTTP ${response.status}`);
      }
      setResult(await response.json());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to run the CV comparison.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>ReRouteHer CV → ESCO technical feasibility test</h1>
      <p className="intro">
        Upload one CV. The server extracts the latest job title, skills, and employment length, then runs both models using the same extracted features.
      </p>

      <form onSubmit={submit}>
        <label htmlFor="cv-file">CV file (PDF, DOCX, or TXT; maximum 10 MB)</label>
        <input
          id="cv-file"
          type="file"
          accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          required
        />
        <button type="submit" disabled={!file || loading}>{loading ? 'Running both models…' : 'Upload and compare'}</button>
      </form>

      <p className="privacy-note">Internal test only. The API processes the upload in memory and does not save the CV.</p>
      {error && <div className="error" role="alert"><strong>Test failed:</strong> {error}</div>}

      {result && (
        <div aria-live="polite">
          <section className="parser-section">
            <div className="section-heading">
              <h2>Extracted CV features</h2>
              <span>Total request: {result.timings_ms.total.toFixed(1)} ms</span>
            </div>
            <dl>
              <div><dt>File</dt><dd>{result.file.name} ({Math.round(result.file.size_bytes / 1024)} KB)</dd></div>
              <div><dt>Latest job title</dt><dd>{result.extracted_features.job_title}</dd></div>
              <div><dt>Title extraction</dt><dd>{result.extracted_features.job_title_source}</dd></div>
              <div><dt>Skills</dt><dd>{result.extracted_features.skills.length ? result.extracted_features.skills.join(', ') : 'None extracted'}</dd></div>
              <div><dt>Skills extraction</dt><dd>{result.extracted_features.skills_source}</dd></div>
              <div><dt>Latest-role length</dt><dd>{result.extracted_features.work_length_years != null ? `${result.extracted_features.work_length_years} years` : 'Not extracted'}</dd></div>
              <div><dt>Total non-overlapping experience</dt><dd>{result.extracted_features.total_experience_years != null ? `${result.extracted_features.total_experience_years} years` : 'Not extracted'}</dd></div>
              <div><dt>Date ranges found</dt><dd>{result.extracted_features.employment_date_ranges_found}</dd></div>
              <div><dt>Parser time</dt><dd>{result.timings_ms.parse.toFixed(1)} ms</dd></div>
            </dl>
            {result.extracted_features.warnings.length > 0 && (
              <div className="warnings">
                <strong>Parser warnings</strong>
                <ul>{result.extracted_features.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
              </div>
            )}
            <details>
              <summary>Extracted text preview</summary>
              <pre>{result.extracted_features.text_preview}</pre>
            </details>
          </section>

          <ResultsTable title="1. TF-IDF + Logistic Regression" result={result.tfidf} elapsed={result.timings_ms.tfidf} />
          <ResultsTable title="2. all-MiniLM-L6-v2 embeddings" result={result.minilm} elapsed={result.timings_ms.minilm} />
        </div>
      )}
    </main>
  );
}
