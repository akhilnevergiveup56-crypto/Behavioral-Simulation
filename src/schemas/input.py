from pydantic import BaseModel, Field
from typing import Optional


EMOTIONS = [
    "anger",
    "anxiety",
    "fear",
    "sadness",
    "joy",
    "empathy",
    "patience",
    "confidence",
]

TENDENCIES = [
    "impulsiveness",
    "assertiveness",
    "conflict_avoidance",
    "risk_tolerance",
    "self_control",
    "need_for_approval",
    "trust_tendency",
    "adaptability",
]


class ProfileValues(BaseModel):
    anger: float = Field(50, ge=0, le=100)
    anxiety: float = Field(50, ge=0, le=100)
    fear: float = Field(50, ge=0, le=100)
    sadness: float = Field(50, ge=0, le=100)
    joy: float = Field(50, ge=0, le=100)
    empathy: float = Field(50, ge=0, le=100)
    patience: float = Field(50, ge=0, le=100)
    confidence: float = Field(50, ge=0, le=100)

    impulsiveness: float = Field(50, ge=0, le=100)
    assertiveness: float = Field(50, ge=0, le=100)
    conflict_avoidance: float = Field(50, ge=0, le=100)
    risk_tolerance: float = Field(50, ge=0, le=100)
    self_control: float = Field(50, ge=0, le=100)
    need_for_approval: float = Field(50, ge=0, le=100)
    trust_tendency: float = Field(50, ge=0, le=100)
    adaptability: float = Field(50, ge=0, le=100)


class AdvancedContext(BaseModel):
    """Optional structured context. Free-form stories will be added later."""
    family_communication: Optional[float] = Field(None, ge=0, le=100)
    social_support: Optional[float] = Field(None, ge=0, le=100)
    trust_difficulty: Optional[float] = Field(None, ge=0, le=100)
    conflict_exposure: Optional[float] = Field(None, ge=0, le=100)
    relationship_stability: Optional[float] = Field(None, ge=0, le=100)
    authority_sensitivity: Optional[float] = Field(None, ge=0, le=100)
    rejection_sensitivity: Optional[float] = Field(None, ge=0, le=100)
    environmental_stress: Optional[float] = Field(None, ge=0, le=100)
    free_form_story: Optional[str] = None


class SimulationRequest(BaseModel):
    profile: ProfileValues
    context: Optional[AdvancedContext] = None
    scenario: str = Field(..., min_length=20, max_length=10000)
