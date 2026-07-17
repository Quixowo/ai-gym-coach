"""Pydantic request/response models for the auth endpoints.

Email is validated with a plain ``str`` + lightweight regex rather than
``pydantic.EmailStr``. ``EmailStr`` requires the ``email-validator`` package,
which is **not** an installed (or transitive) dependency here; adding a new dep
is out of scope for this phase, and a simple structural check is sufficient for
a portfolio project's registration form. See the auth report / LESSONS entry.

``UserResponse`` deliberately omits ``hashed_password`` — it is never serialized
back to a client.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

ExperienceLevel = Literal["beginner", "intermediate", "advanced"]
PrimaryGoal = Literal["hypertrophy", "strength", "fat_loss", "general"]

# Deliberately permissive structural check (one @, a dot in the domain, no
# whitespace) — not RFC-5322-complete, just enough to reject obvious garbage
# without pulling in email-validator.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validate_email(value: str) -> str:
    value = value.strip().lower()
    if not _EMAIL_RE.match(value):
        raise ValueError("invalid email address")
    return value


class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str
    experience_level: ExperienceLevel
    primary_goal: PrimaryGoal
    injury_notes: str | None = None

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return _validate_email(v)

    @field_validator("password")
    @classmethod
    def _password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("password must be at least 8 characters")
        return v

    @field_validator("display_name")
    @classmethod
    def _display_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("display_name must not be empty")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return _validate_email(v)


class UserResponse(BaseModel):
    # ``from_attributes`` lets us build this straight from the ORM User row.
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    display_name: str
    experience_level: str
    primary_goal: str
    injury_notes: str | None
    created_at: datetime
