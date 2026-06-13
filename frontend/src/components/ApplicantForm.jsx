import { useState } from "react";
import { PRESETS, INCOME_TYPES, EDU_TYPES } from "../presets.js";

const DEFAULTS = {
  applicant_id: "", amt_income_total: 150000, amt_credit: 250000, amt_annuity: 20000,
  cnt_children: 0, days_birth: -12000, days_employed: -2000,
  ext_source_1: 0.6, ext_source_2: 0.58, ext_source_3: 0.55,
  code_gender: "F", flag_own_car: 0, flag_own_realty: 0,
  name_income_type: "Working", name_education_type: "Secondary / secondary special",
};

export default function ApplicantForm({ onSubmit, submitting }) {
  const [form, setForm] = useState(DEFAULTS);
  const set = (k) => (e) => {
    const val = e.target.type === "number" ? Number(e.target.value) : e.target.value;
    setForm((f) => ({ ...f, [k]: val }));
  };
  const loadPreset = (key) => setForm({ ...DEFAULTS, ...PRESETS[key] });

  const submit = (e) => {
    e.preventDefault();
    if (!form.applicant_id.trim()) return alert("Applicant ID is required.");
    onSubmit({
      ...form,
      flag_own_car: Number(form.flag_own_car),
      flag_own_realty: Number(form.flag_own_realty),
    });
  };

  return (
    <form className="card" onSubmit={submit}>
      <div className="btn-row">
        <span className="note" style={{ alignSelf: "center", marginRight: 6 }}>Load real applicant:</span>
        <button type="button" className="btn-ghost" onClick={() => loadPreset("low")}>📗 Low Risk</button>
        <button type="button" className="btn-ghost" onClick={() => loadPreset("medium")}>📙 Medium Risk</button>
        <button type="button" className="btn-ghost" onClick={() => loadPreset("high")}>📕 High Risk</button>
      </div>

      <div className="grid2">
        <div>
          <label>Applicant ID</label>
          <input value={form.applicant_id} onChange={set("applicant_id")} placeholder="APP-001" />
          <label>Annual Income ($)</label>
          <input type="number" value={form.amt_income_total} onChange={set("amt_income_total")} />
          <label>Requested Credit ($)</label>
          <input type="number" value={form.amt_credit} onChange={set("amt_credit")} />
          <label>Monthly Payment ($)</label>
          <input type="number" value={form.amt_annuity} onChange={set("amt_annuity")} />
          <label>Number of Children</label>
          <input type="number" value={form.cnt_children} onChange={set("cnt_children")} />
        </div>
        <div>
          <label>Days Since Birth (negative)</label>
          <input type="number" value={form.days_birth} onChange={set("days_birth")} />
          <label>Days Employed (negative = employed)</label>
          <input type="number" value={form.days_employed} onChange={set("days_employed")} />
          <label>External Credit Score 1 ({form.ext_source_1})</label>
          <input type="range" min="0" max="1" step="0.01" value={form.ext_source_1} onChange={set("ext_source_1")} />
          <label>External Credit Score 2 ({form.ext_source_2})</label>
          <input type="range" min="0" max="1" step="0.01" value={form.ext_source_2} onChange={set("ext_source_2")} />
          <label>External Credit Score 3 ({form.ext_source_3})</label>
          <input type="range" min="0" max="1" step="0.01" value={form.ext_source_3} onChange={set("ext_source_3")} />
        </div>
      </div>

      <div className="grid3">
        <div>
          <label>Gender</label>
          <select value={form.code_gender} onChange={set("code_gender")}>
            <option>F</option><option>M</option><option>X</option>
          </select>
          <div className="note warn">Collected for fair-lending monitoring only — not a model input.</div>
        </div>
        <div>
          <label>Income Type</label>
          <select value={form.name_income_type} onChange={set("name_income_type")}>
            {INCOME_TYPES.map((t) => <option key={t}>{t}</option>)}
          </select>
        </div>
        <div>
          <label>Education Level</label>
          <select value={form.name_education_type} onChange={set("name_education_type")}>
            {EDU_TYPES.map((t) => <option key={t}>{t}</option>)}
          </select>
        </div>
      </div>

      <div className="grid2" style={{ marginTop: 6 }}>
        <div>
          <label><input type="checkbox" style={{ width: "auto", marginRight: 8 }} checked={!!Number(form.flag_own_car)} onChange={(e) => setForm((f) => ({ ...f, flag_own_car: e.target.checked ? 1 : 0 }))} /> Owns Car</label>
        </div>
        <div>
          <label><input type="checkbox" style={{ width: "auto", marginRight: 8 }} checked={!!Number(form.flag_own_realty)} onChange={(e) => setForm((f) => ({ ...f, flag_own_realty: e.target.checked ? 1 : 0 }))} /> Owns Realty</label>
        </div>
      </div>

      <div style={{ marginTop: 18 }}>
        <button className="btn" disabled={submitting}>{submitting ? "Running pipeline…" : "🚀 Submit for Decisioning"}</button>
      </div>
    </form>
  );
}
