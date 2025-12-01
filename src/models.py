from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class DialogueTurn(BaseModel):
    role: Literal["user", "assistant"]
    utterance: str
    action: Optional[Dict[str, Any]] = None


class AnalyzeRequest(BaseModel):
    session_id: str
    user_input: Optional[str] = None
    ocr_texts: List[str] = Field(default_factory=list)
    dialogue_history: List[DialogueTurn] = Field(default_factory=list)
    last_btn: Optional[str] = None

    @field_validator("ocr_texts", mode="before")
    def _ensure_list(cls, value: Any) -> Any:
        return value or []


class Action(BaseModel):
    type: Literal["click_text", "speak_only", "ask_clarification"]
    params: Dict[str, Any] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    status: Literal["success", "ambiguous", "fail"]
    confidence: float = Field(ge=0.0, le=1.0)
    response_message: str
    action: Action


class ScreenDetectRequest(BaseModel):
    previous_texts: List[str] = Field(default_factory=list)
    current_texts: List[str] = Field(default_factory=list)
    session_id: str
    user_input: Optional[str] = None
    dialogue_history: List[DialogueTurn] = Field(default_factory=list)
    last_btn: Optional[str] = None

    @field_validator("previous_texts", "current_texts", mode="before")
    def _ensure_list(cls, value: Any) -> Any:
        return value or []


class ScreenDetectResponse(BaseModel):
    is_changed: bool
    similarity_score: float = Field(ge=0.0, le=1.0)
    ai_analysis: Optional[AnalyzeResponse] = None
