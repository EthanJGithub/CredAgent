import { useState } from "react";
import ShapBar from "./ShapBar.jsx";

const TABS = ["Decision", "SHAP", "Compliance", "Adverse Notice", "Audit"];

function Metric({ k, v }) {
  return (
    <div className="metric">
      <div className="k">{k}</div>
      <div className="v">{v}</div>
    </div>
  );
}

export default function DecisionResult({ result, onHumanReview, reviewing }) {
  const [tab, setTab] = useState("Decision");
  const [hrDecision, setHrDecision] = useState("APPROVE");
  const [hrNotes, setHrNotes] = useState("");
  if (!result) return null;

  const prob = result.risk_probability;
  const badge = result.final_decision || (result.requires_human_review ? "PENDING" : "PENDING");

  return (
    <div className="card">
      <h2 style={{ marginTop: 0 }}>Decision Result</h2>
      <span className={`badge ${badge}`}>
        {{ APPROVE: "✅", DECLINE: "❌", REFER: "⚠️", PENDING: "⏳" }[badge] || "⏳"} {badge === "PENDING" ? "AWAITING REVIEW" : badge}
      </span>

      <div className="grid4" style={{ marginTop: 18 }}>
        <Metric k="Default Probability" v={prob != null ? `${(prob * 100).toFixed(1)}%` : "N/A"} />
        <Metric k="Risk Tier" v={result.risk_tier || "N/A"} />
        <Metric k="Credit Limit" v={result.credit_limit ? `$${result.credit_limit.toLocaleString()}` : "—"} />
        <Metric k="Processing" v={result.processing_time_ms ? `${Math.round(result.processing_time_ms)} ms` : "N/A"} />
      </div>

      <div className="tabs">
        {TABS.map((t) => (
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>{t}</button>
        ))}
      </div>

      {tab === "Decision" && (
        <div>
          {result.decision_reasoning && <p>{result.decision_reasoning}</p>}
          <ol>
            {(result.top_risk_factors || []).map((f, i) => <li key={i}>{f}</li>)}
          </ol>
        </div>
      )}
      {tab === "SHAP" && (
        <div>
          <ShapBar shapValues={result.shap_values} />
          <p className="note">SHAP contributions to the model's log-odds (margin); higher log-odds = higher default probability. Red increases risk · green decreases it.</p>
        </div>
      )}
      {tab === "Compliance" && (
        <div>
          {(result.compliance_flags || []).length > 0 ? (
            <div className="flag-banner bad">⚠️ {result.compliance_flags.length} compliance flag(s): {result.compliance_flags.join(", ")}</div>
          ) : (
            <div className="flag-banner ok">✅ No compliance issues detected.</div>
          )}
          {(result.retrieved_policy_excerpts || []).map((ex, i) => (
            <pre key={i} className="notice">{ex.slice(0, 600)}{ex.length > 600 ? "…" : ""}</pre>
          ))}
        </div>
      )}
      {tab === "Adverse Notice" && (
        result.adverse_action_notice
          ? <pre className="notice">{result.adverse_action_notice}</pre>
          : <p className="note">{result.final_decision === "APPROVE" ? "No adverse action notice required — approved." : "Adverse action notice appears here for declined applications."}</p>
      )}
      {tab === "Audit" && (
        <div>{(result.audit_trail || []).map((e, i) => <div key={i} className="audit-line">{e}</div>)}</div>
      )}

      {result.requires_human_review && (
        <div className="card" style={{ marginTop: 18 }}>
          <h3 style={{ marginTop: 0 }}>👤 Human Review Required</h3>
          <p className="note warn">Borderline risk score or compliance flag — a reviewer's decision is required.</p>
          <label>Your Decision</label>
          <select value={hrDecision} onChange={(e) => setHrDecision(e.target.value)}>
            <option>APPROVE</option>
            <option>DECLINE</option>
          </select>
          <label>Reviewer Notes</label>
          <input value={hrNotes} onChange={(e) => setHrNotes(e.target.value)} placeholder="Justification…" />
          <div style={{ marginTop: 12 }}>
            <button className="btn" disabled={reviewing} onClick={() => onHumanReview(hrDecision, hrNotes)}>
              {reviewing ? "Submitting…" : "Submit Review"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
