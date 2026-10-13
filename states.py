from typing import TypedDict
from pydantic import BaseModel, field_validator


# Example TypedDict State schema
class StateTypedDict(TypedDict):
    topic: str
    summary: str
    score: str


# Example Pydantic BaseModel State with validation
class State(BaseModel):
    topic: str
    summary: str
    score: int

    @field_validator("score")
    @classmethod
    def positive_score(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Score must be positive")
        return v
