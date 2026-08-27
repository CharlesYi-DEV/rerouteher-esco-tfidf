'use client';

import { FormEvent, useMemo, useState } from 'react';

type Match = {
  esco_code: string;
  esco_title: string;
  esco_uri?: string | null;
  masco_candidate_code?: string | null;
  masco_mapping_status: string;
  score: number;
  method: string;
  matched_skills: string[];
  reference_work_length_median_years?: number | null;
};

type MatchResponse = { method_used: string; matches: Match[] };
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';

export default function Home() {
  const [jobTitle, setJobTitle] = useState('Software Engineer');
  const [skillText, setSkillText] = useState('programming, databases, software testing, problem solving');
  const [workLength, setWorkLength] = useState('3');
  const [result, setResult] = useState<MatchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const skills = useMemo(() => skillText.split(/[,\n]/).map((skill) => skill.trim()).filter(Boolean), [skillText]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${API_BASE}/match`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_title: jobTitle,
          skills,
          work_length_years: workLength === '' ? null : Number(workLength),
          top_k: 3,
        }),
      });
      if (!response.ok) throw new Error(`Matcher returned ${response.status}`);
      setResult(await response.json());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to reach the matcher.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="ReRouteHer occupation matcher">
          <span className="brandMark">R</span>
          <span>ReRouteHer <b>Occupation Lab</b></span>
        </a>
        <span className="methodPill">Path 01 · TF-IDF + Logistic Regression</span>
      </header>

      <section className="hero" id="top">
        <div className="heroCopy">
          <p className="eyebrow">ESCO OCCUPATION CODING</p>
          <h1>Turn varied job wording into a standard occupation.</h1>
          <p className="lede">
            Character n-grams recognise partial words, spelling variation and light typos. Skills and optional work length add context before the model returns an ESCO match.
          </p>
          <div className="proofRow" aria-label="Dataset summary">
            <div><strong>47,224</strong><span>JobHop records</span></div>
            <div><strong>98.72%</strong><span>official skill-link coverage</span></div>
            <div><strong>2,242</strong><span>exact ESCO codes linked</span></div>
          </div>
        </div>

        <form className="matcherCard" onSubmit={submit}>
          <div className="cardHeading">
            <div><span className="stepTag">TRY THE MODEL</span><h2>Describe the previous job</h2></div>
            <span className="localBadge">CPU-ready</span>
          </div>
          <label>
            Job title
            <input value={jobTitle} onChange={(event) => setJobTitle(event.target.value)} required minLength={2} />
            <small>Try “programer”, “HR admin”, or a partial title.</small>
          </label>
          <label>
            Skills
            <textarea value={skillText} onChange={(event) => setSkillText(event.target.value)} rows={4} />
            <small>Comma-separated skills help resolve ambiguous titles.</small>
          </label>
          <label>
            Work length in years <span className="optional">optional</span>
            <input type="number" min="0" max="80" step="0.25" value={workLength} onChange={(event) => setWorkLength(event.target.value)} />
          </label>
          <button type="submit" disabled={loading}>{loading ? 'Matching…' : 'Find ESCO occupation'}</button>
          {error && <p className="error" role="alert">{error}</p>}
        </form>
      </section>

      <section className="resultsSection" aria-live="polite">
        <div className="sectionTitle">
          <div><p className="eyebrow">MODEL RESPONSE</p><h2>{result ? 'Best occupation matches' : 'Results appear here'}</h2></div>
          {result && <span className="methodNote">Method used: {result.method_used.replaceAll('_', ' ')}</span>}
        </div>
        {result ? (
          <div className="resultGrid">
            {result.matches.map((match, index) => (
              <article className={`resultCard ${index === 0 ? 'primary' : ''}`} key={match.esco_code}>
                <div className="rank">{String(index + 1).padStart(2, '0')}</div>
                <div className="score">{Math.round(match.score * 100)}%</div>
                <p className="resultCode">ESCO {match.esco_code}</p>
                <h3>{match.esco_title}</h3>
                {match.masco_candidate_code && <p className="masco">MASCO candidate {match.masco_candidate_code} · validation required</p>}
                <div className="skillChips">
                  {(match.matched_skills.length ? match.matched_skills : ['No direct skill-word overlap']).map((skill) => <span key={skill}>{skill}</span>)}
                </div>
                {match.reference_work_length_median_years != null && <p className="durationNote">Historical median: {match.reference_work_length_median_years} years</p>}
              </article>
            ))}
          </div>
        ) : (
          <div className="emptyState"><div className="emptyGlyph">Aa</div><p>The model compares sub-word patterns, skill evidence and the optional duration feature.</p></div>
        )}
      </section>

      <section className="methodSection">
        <div><p className="eyebrow">WHY THIS PATH</p><h2>Designed for imperfect wording.</h2></div>
        <div className="methodSteps">
          <article><span>01</span><h3>Normalise aliases</h3><p>Common shorthand such as HR, admin and UI/UX is expanded before scoring.</p></article>
          <article><span>02</span><h3>Read character fragments</h3><p>3–5 character n-grams retain evidence when only part of a word is correct.</p></article>
          <article><span>03</span><h3>Return evidence</h3><p>The API returns top matches, scores, matched skills and clearly marked MASCO candidates.</p></article>
        </div>
      </section>
    </main>
  );
}
