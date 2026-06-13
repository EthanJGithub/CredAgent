// Thin API client for the CredAgent FastAPI backend.
const BASE = import.meta.env.VITE_API_BASE || "/api/v1";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  base: BASE,
  health: () => request("/health"),
  submitDecision: (payload) =>
    request("/decisions", { method: "POST", body: JSON.stringify(payload) }),
  humanReview: (applicantId, decision, notes) =>
    request(`/decisions/${encodeURIComponent(applicantId)}/human-review`, {
      method: "POST",
      body: JSON.stringify({ human_decision: decision, human_notes: notes }),
    }),
  monitoringSummary: () => request("/monitoring/summary"),
  monitoringDecisions: (limit = 50) => request(`/monitoring/decisions?limit=${limit}`),
  exportUrl: () => `${BASE}/monitoring/export.csv`,
};
