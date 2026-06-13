// Real Home Credit applicants that land cleanly in each risk tier.
export const PRESETS = {
  low: {
    applicant_id: "low-risk-demo",
    amt_credit: 239850, amt_income_total: 157500, amt_annuity: 23494.5,
    days_birth: -12967, days_employed: -1996,
    ext_source_1: 0.838, ext_source_2: 0.356, ext_source_3: 0.608,
    code_gender: "F", flag_own_car: 0, flag_own_realty: 1, cnt_children: 0,
    name_income_type: "Working", name_education_type: "Secondary / secondary special",
  },
  medium: {
    applicant_id: "medium-risk-demo",
    amt_credit: 495000, amt_income_total: 180000, amt_annuity: 24750,
    days_birth: -11623, days_employed: -1809,
    ext_source_1: 0.641, ext_source_2: 0.482, ext_source_3: 0.161,
    code_gender: "F", flag_own_car: 1, flag_own_realty: 1, cnt_children: 0,
    name_income_type: "Working", name_education_type: "Incomplete higher",
  },
  high: {
    applicant_id: "high-risk-demo",
    amt_credit: 545040, amt_income_total: 112500, amt_annuity: 26640,
    days_birth: -10948, days_employed: -1721,
    ext_source_1: 0.348, ext_source_2: 0.463, ext_source_3: 0.10,
    code_gender: "F", flag_own_car: 1, flag_own_realty: 1, cnt_children: 0,
    name_income_type: "Working", name_education_type: "Secondary / secondary special",
  },
};

export const INCOME_TYPES = [
  "Working", "Commercial associate", "Pensioner", "State servant",
  "Unemployed", "Student", "Businessman", "Maternity leave",
];
export const EDU_TYPES = [
  "Higher education", "Secondary / secondary special", "Incomplete higher",
  "Lower secondary", "Academic degree",
];
