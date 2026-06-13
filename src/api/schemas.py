from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class ApplicationRequest(BaseModel):
    applicant_id: str = Field(..., description="Unique applicant identifier (UUID)")
    amt_credit: float = Field(..., gt=0, description="Requested credit amount in USD")
    amt_income_total: float = Field(..., gt=0, description="Annual income in USD")
    amt_annuity: float = Field(..., gt=0, description="Monthly loan annuity payment")
    days_birth: int = Field(..., lt=0, description="Days since birth (negative integer)")
    days_employed: int = Field(..., description="Days employed (negative = currently employed)")
    ext_source_1: Optional[float] = Field(None, ge=0, le=1)
    ext_source_2: Optional[float] = Field(None, ge=0, le=1)
    ext_source_3: Optional[float] = Field(None, ge=0, le=1)
    code_gender: str = Field(..., pattern="^[MFX]$")
    flag_own_car: int = Field(0, ge=0, le=1)
    flag_own_realty: int = Field(0, ge=0, le=1)
    cnt_children: int = Field(0, ge=0)
    name_income_type: str
    name_education_type: str

    model_config = {"json_schema_extra": {
        "example": {
            "applicant_id": "a1b2c3d4-0000-0000-0000-000000000001",
            "amt_credit": 15000.0,
            "amt_income_total": 60000.0,
            "amt_annuity": 450.0,
            "days_birth": -12000,
            "days_employed": -3650,
            "ext_source_1": 0.72,
            "ext_source_2": 0.61,
            "ext_source_3": 0.55,
            "code_gender": "M",
            "flag_own_car": 1,
            "flag_own_realty": 0,
            "cnt_children": 1,
            "name_income_type": "Working",
            "name_education_type": "Higher education"
        }
    }}


class DecisionResponse(BaseModel):
    applicant_id: str
    final_decision: Optional[str]
    credit_limit: Optional[float]
    risk_probability: Optional[float]
    risk_tier: Optional[str]
    top_risk_factors: Optional[List[str]]
    shap_values: Optional[Dict[str, float]] = None
    decision_reasoning: Optional[str]
    decision_confidence: Optional[float] = None
    compliance_flags: List[str]
    retrieved_policy_excerpts: List[str] = []
    adverse_action_notice: Optional[str]
    processing_time_ms: Optional[float]
    requires_human_review: bool
    audit_trail: List[str]


class HumanReviewRequest(BaseModel):
    human_decision: str = Field(..., pattern="^(APPROVE|DECLINE)$")
    human_notes: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorstore_ready: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    features: List[str]
    training_auc: float
    decision_thresholds: dict
