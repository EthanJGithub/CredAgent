import { useEffect, useState } from "react";
import { api } from "./api.js";
import ApplicantForm from "./components/ApplicantForm.jsx";
import DecisionResult from "./components/DecisionResult.jsx";
import Monitoring from "./components/Monitoring.jsx";

export default function App() {
  const [view, setView] = useState("score");
  const [health, setHealth] = useState(null);
  const [result, setResult] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [reviewing, setReviewing] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: "error" }));
  }, []);

  const submit = async (payload) => {
    setSubmitting(true);
    setError(null);
    setResult(null); // clear stale result when a new application is sent
    try {
      setResult(await api.submitDecision(payload));
    } catch (e) {
      setError(e.message);
    } finally {
      setSubmitting(false);
    }
  };

  const review = async (decision, notes) => {
    setReviewing(true);
    setError(null);
    try {
      setResult(await api.humanReview(result.applicant_id, decision, notes));
    } catch (e) {
      setError(e.message);
    } finally {
      setReviewing(false);
    }
  };

  const dot = (ok) => (ok ? "🟢" : "🔴");

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">Cred<span>Agent</span></div>
        <div className="tagline">Agentic Credit Decisioning</div>

        <nav className="nav">
          <button className={view === "score" ? "active" : ""} onClick={() => setView("score")}>📝 Score an Applicant</button>
          <button className={view === "monitor" ? "active" : ""} onClick={() => setView("monitor")}>📈 Portfolio Monitoring</button>
        </nav>

        <div className="legend">
          <strong>System Status</strong>
          <table>
            <tbody>
              <tr><td>{dot(health?.status === "ok")} API</td></tr>
              <tr><td>{dot(health?.model_loaded)} ML Model</td></tr>
              <tr><td>{dot(health?.vectorstore_ready)} Vector Store</td></tr>
            </tbody>
          </table>
          <strong>Decision Tiers</strong>
          <table>
            <tbody>
              <tr><td>🟢 LOW &lt; 30%</td><td>Auto-Approve</td></tr>
              <tr><td>🟡 MED 30–55%</td><td>Human Review</td></tr>
              <tr><td>🟠 HIGH 55–75%</td><td>Auto-Decline</td></tr>
              <tr><td>🔴 DECLINE &gt; 75%</td><td>Auto-Decline</td></tr>
            </tbody>
          </table>
        </div>
        <div className="sidebar-foot">React · Nivo · FastAPI · LangGraph · XGBoost · SHAP · ChromaDB</div>
      </aside>

      <main className="main">
        {view === "score" ? (
          <>
            <h1>Applicant Credit Assessment</h1>
            <p className="sub">Real-time agentic risk decisioning for employer-sponsored installment lending.</p>
            {error && <div className="error">{error}</div>}
            <ApplicantForm onSubmit={submit} submitting={submitting} />
            <DecisionResult result={result} onHumanReview={review} reviewing={reviewing} />
          </>
        ) : (
          <>
            <h1>Portfolio Monitoring</h1>
            <p className="sub">Decision volume, risk distribution, and fair-lending disparate-impact analysis across all recorded decisions.</p>
            <Monitoring />
          </>
        )}
      </main>
    </div>
  );
}
