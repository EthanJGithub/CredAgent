"""Demo applicant presets — drawn from REAL Home Credit applicants.

Each preset is an actual row from the Home Credit dataset whose predicted
default probability lands cleanly in one tier under the trained model:

    low-risk-demo     pred ~0.09  -> LOW    -> APPROVE
    medium-risk-demo  pred ~0.44  -> MEDIUM -> human review
    high-risk-demo    pred ~0.86  -> DECLINE-> declined + adverse action notice
                                     (this applicant actually defaulted; TARGET=1)

Using real records keeps the demo honest: these are genuine applications, not
hand-tuned inputs. Amounts are on the dataset's native scale.
"""

PRESETS = {
    "low": {
        "applicant_id": "low-risk-demo",
        "amt_credit": 239850.0, "amt_income_total": 157500.0, "amt_annuity": 23494.5,
        "days_birth": -12967, "days_employed": -1996,
        "ext_source_1": 0.838, "ext_source_2": 0.356, "ext_source_3": 0.608,
        "code_gender": "F", "flag_own_car": 0, "flag_own_realty": 1,
        "cnt_children": 0, "name_income_type": "Working",
        "name_education_type": "Secondary / secondary special",
    },
    "medium": {
        "applicant_id": "medium-risk-demo",
        "amt_credit": 495000.0, "amt_income_total": 180000.0, "amt_annuity": 24750.0,
        "days_birth": -11623, "days_employed": -1809,
        "ext_source_1": 0.641, "ext_source_2": 0.482, "ext_source_3": 0.161,
        "code_gender": "F", "flag_own_car": 1, "flag_own_realty": 1,
        "cnt_children": 0, "name_income_type": "Working",
        "name_education_type": "Incomplete higher",
    },
    "high": {
        "applicant_id": "high-risk-demo",
        "amt_credit": 545040.0, "amt_income_total": 112500.0, "amt_annuity": 26640.0,
        "days_birth": -10948, "days_employed": -1721,
        "ext_source_1": 0.348, "ext_source_2": 0.463, "ext_source_3": 0.10,
        "code_gender": "F", "flag_own_car": 1, "flag_own_realty": 1,
        "cnt_children": 0, "name_income_type": "Working",
        "name_education_type": "Secondary / secondary special",
    },
}
