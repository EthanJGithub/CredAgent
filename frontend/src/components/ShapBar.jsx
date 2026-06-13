import { ResponsiveBar } from "@nivo/bar";

const DISPLAY = {
  EXT_SOURCE_1: "Ext Credit Score 1", EXT_SOURCE_2: "Ext Credit Score 2",
  EXT_SOURCE_3: "Ext Credit Score 3", debt_to_income: "Debt-to-Income",
  credit_to_income_ratio: "Credit/Income Ratio", annuity_to_credit_ratio: "Annuity/Credit Ratio",
  employment_months: "Employment Length", age_years: "Applicant Age",
  AMT_CREDIT: "Credit Amount", AMT_INCOME_TOTAL: "Annual Income",
  AMT_ANNUITY: "Monthly Payment", CNT_CHILDREN: "Number of Children",
  has_income_stability: "Income Stability",
  NAME_INCOME_TYPE_Working: "Income: Working",
  NAME_EDUCATION_TYPE_Higher_education: "Education: Higher",
  DAYS_BIRTH: "Age (days)", DAYS_EMPLOYED: "Employment (days)",
  FLAG_OWN_CAR: "Owns Car", FLAG_OWN_REALTY: "Owns Realty",
  NAME_INCOME_TYPE_Commercial_associate: "Income: Self-employed",
};

const label = (f) => DISPLAY[f] || f.replace(/_/g, " ");

// Nivo diverging horizontal bar of SHAP contributions (log-odds / model margin).
export default function ShapBar({ shapValues }) {
  if (!shapValues || Object.keys(shapValues).length === 0) {
    return <p className="spinner">SHAP values not available.</p>;
  }
  const data = Object.entries(shapValues)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 10)
    .map(([feature, value]) => ({
      feature: label(feature),
      value: Number(value.toFixed(4)),
      color: value > 0 ? "#dc3545" : "#28a745",
    }))
    .reverse(); // largest at top after horizontal layout

  return (
    <div className="chart">
      <ResponsiveBar
        data={data}
        keys={["value"]}
        indexBy="feature"
        layout="horizontal"
        margin={{ top: 10, right: 40, bottom: 50, left: 150 }}
        padding={0.3}
        colors={(bar) => bar.data.color}
        enableGridX
        enableGridY={false}
        axisBottom={{ legend: "SHAP value — impact on log-odds of default", legendPosition: "middle", legendOffset: 38 }}
        axisLeft={{ tickSize: 0, tickPadding: 8 }}
        labelSkipWidth={9999} // values shown via tooltip; keep bars clean
        theme={{
          background: "transparent",
          text: { fill: "#93a0b8", fontSize: 11 },
          axis: { legend: { text: { fill: "#93a0b8" } }, ticks: { text: { fill: "#93a0b8" } } },
          grid: { line: { stroke: "#2a3550" } },
          tooltip: { container: { background: "#1f2940", color: "#e7ecf5" } },
        }}
        tooltip={({ data }) => (
          <div style={{ padding: "6px 10px", background: "#1f2940", border: "1px solid #2a3550", borderRadius: 6 }}>
            <strong>{data.feature}</strong>
            <div style={{ color: data.value > 0 ? "#ff7785" : "#5fd07a" }}>
              {data.value > 0 ? "+" : ""}{data.value} {data.value > 0 ? "(increases risk)" : "(decreases risk)"}
            </div>
          </div>
        )}
      />
    </div>
  );
}
