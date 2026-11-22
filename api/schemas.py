from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Decision = Literal["allow", "confirm", "block"]
RiskLevel = Literal["none", "low", "medium", "high", "critical"]
Name = Annotated[str, Field(min_length=1, max_length=128)]


class InputModel(BaseModel):

    model_config = ConfigDict(strict=True, extra="ignore")


class Message(InputModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=200_000)


class Attachment(InputModel):
    type: str = Field(default="document", max_length=128)
    name: str = Field(default="", max_length=512)
    content: str = Field(max_length=200_000)


class Context(InputModel):
    target_agent: Name
    user_role: Name
    user_id: str | None = Field(default=None, max_length=128)
    user_name: str | None = Field(default=None, max_length=256)
    system_prompt: str | None = Field(default=None, max_length=200_000)
    history: list[Message] = Field(default_factory=list, max_length=1000)
    attachments: list[Attachment] = Field(default_factory=list, max_length=100)


class CheckRequest(InputModel):
    user_prompt: str = Field(min_length=1, max_length=200_000)
    context: Context


class CheckResult(BaseModel):
    decision: Decision
    risk_level: RiskLevel
    risk_category: str
    explanation: str
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    policy_version: int | str
    attack_probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    intent: list[dict] = Field(default_factory=list)
    checks: list[dict] = Field(default_factory=list)
    backend: str = "layered"
    intent_total: int = 0
    details_truncated: bool = False

    @field_validator("intent", mode="after")
    @classmethod
    def bound_intent(cls, value):
        return value[:32]


class CheckResponse(CheckResult):
    request_id: str


class DemoRequest(InputModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    user_prompt: str = Field(min_length=1, max_length=200_000)
    history: list[Message] = Field(default_factory=list, max_length=1000)
    attachments: list[Attachment] = Field(default_factory=list, max_length=100)


class Confirmation(InputModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    confirmed: Literal[True]

    @field_validator("confirmed", mode="before")
    @classmethod
    def explicit_boolean(cls, value):
        if value is not True:
            raise ValueError("Требуется confirmed: true")
        return value


class AgentReceipt(BaseModel):
    receipt_id: str
    request_id: str
    target_agent: str
    received_at: str
    message: str = "Демо-агент получил запрос."


class DemoResponse(BaseModel):
    request_id: str
    check: CheckResult
    status: Literal["blocked", "pending_confirmation", "delivered"]
    confirmation_token: str | None = None
    expires_at: str | None = None
    receipt: AgentReceipt | None = None
