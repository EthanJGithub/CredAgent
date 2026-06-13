import { useEffect, useState } from "react";
import { ResponsivePie } from "@nivo/pie";
import { ResponsiveBar } from "@nivo/bar";
import { api } from "../api.js";

const nivoTheme = {
  text: { fill: "#93a0b8", fontSize: 11 },
  axis: { ticks: { text: { fill: "#93a0b8" } }, legend: { text: { fill: "#93a0b8" } } },
  grid: { line: { stroke: "#2a3550" } },
  tooltip: { container: { background: "#1f2940", color: "#e7ecf5" } },
  legends: { text: { fill: "#93a0b8" } },
};
const DECISION_COLORS = { APPROVE: "#28a745", DECLINE: "#dc3545", REFER: "#ffc107" };

function Metric({ k, v }) {
  return <div className="metric"><div className="k">{k}</div><div className="v">{v}</div></div>;
}

export default function Monitoring() {
  const [summary, setSummary] = useState(null);
  const [rows, setRows] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([api.monitoringSummary(), api.monitoringDecisions(25)])
      .then(([s, d]) => { setSummary(s); setRows(d.decisions); })
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="error">Could not load monitoring data: {error}</div>;
  if (!summary) return <p className="spinner">Loading portfolio analytics…</p>;

  const dc = summary.decision_counts;
  const pieData = Object.keys(dc).map((k) => ({ id: k, label: k, value: dc[k], color: DECISION_COLORS[k] }));
  const tc = summary.tier_counts;
  const tierData = Object.keys(tc).map((k) => ({ tier: k, count: tc[k] }));
  const fl = summary.fair_lending;
  const genderData = Object.entries(fl.by_group || {}).map(([g, v]) => ({
    group: g === "F" ? "Female" : g === "M" ? "Male" : g,
    "approval rate": Number((v.approval_rate * 100).toFixed(1)),
  }));

  return (
    <div>
      <div className="grid4">
        <Metric k="Total Decisions" v={summary.total.toLocaleString()} />
        <Metric k="Approval Rate" v={`${(summary.approval_rate * 100).toFixed(1)}%`} />
        <Metric k="Avg Default Prob" v={`${(summary.avg_default_probability * 100).toFixed(1)}%`} />
        <Metric k="Declines" v={dc.DECLINE.toLocaleString()} />
      </div>

      <div className="grid2" style={{ marginTop: 20 }}>
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Decision Mix</h3>
          <div className="chart-sm">
            <ResponsivePie
              data={pieData}
              margin={{ top: 20, right: 20, bottom: 40, left: 20 }}
              innerRadius={0.55} padAngle={1} cornerRadius={3}
              colors={(d) => d.data.color}
              borderWidth={1} borderColor={{ from: "color", modifiers: [["darker", 0.4]] }}
              arcLabelsTextColor="#0f1420"
              arcLinkLabelsColor={{ from: "color" }}
              arcLinkLabelsTextColor="#93a0b8"
              theme={nivoTheme}
            />
          </div>
        </div>
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Risk Tier Distribution</h3>
          <div className="chart-sm">
            <ResponsiveBar
              data={tierData} keys={["count"]} indexBy="tier"
              margin={{ top: 20, right: 20, bottom: 40, left: 50 }} padding={0.3}
              colors={["#5b8cff"]} theme={nivoTheme}
              axisBottom={{ tickSize: 0 }} axisLeft={{ tickSize: 0 }}
            />
          </div>
        </div>
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>⚖️ Fair-Lending Monitoring — Disparate Impact</h3>
        {fl.adverse_impact_ratio != null ? (
          <div className={`flag-banner ${fl.flag ? "bad" : "ok"}`}>
            {fl.flag ? "⚠️ Potential disparate impact" : "✅ Within tolerance"} — adverse-impact ratio{" "}
            <strong>{fl.adverse_impact_ratio.toFixed(2)}</strong> ({fl.rule}; flag if &lt; 0.80).
            Gender is <em>not</em> a model input; this compares outcomes of permissible features across groups.
          </div>
        ) : <p className="note">Not enough data across groups yet.</p>}
        <div className="chart-sm">
          <ResponsiveBar
            data={genderData} keys={["approval rate"]} indexBy="group"
            margin={{ top: 20, right: 20, bottom: 50, left: 55 }} padding={0.4}
            colors={["#5b8cff"]} theme={nivoTheme}
            axisLeft={{ legend: "approval rate (%)", legendPosition: "middle", legendOffset: -45, tickSize: 0 }}
            axisBottom={{ tickSize: 0 }}
            labelTextColor="#0f1420"
          />
        </div>
      </div>

      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h3 style={{ margin: 0 }}>Recent Decisions</h3>
          <a className="btn-ghost" href={api.exportUrl()} target="_blank" rel="noreferrer">⬇️ Export CSV</a>
        </div>
        <table className="data" style={{ marginTop: 12 }}>
          <thead>
            <tr><th>Applicant</th><th>Gender</th><th>Tier</th><th>Prob</th><th>Decision</th><th>Source</th></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.applicant_id}>
                <td>{r.applicant_id}</td>
                <td>{r.code_gender}</td>
                <td>{r.risk_tier}</td>
                <td>{r.risk_probability != null ? `${(r.risk_probability * 100).toFixed(1)}%` : "—"}</td>
                <td><span className={`pill ${r.final_decision}`}>{r.final_decision}</span></td>
                <td className="note">{r.source}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
