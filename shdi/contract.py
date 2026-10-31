"""Data contract: the explicit rules the agent validates and repairs against.

Field names follow a small subset of the Open Data Contract Standard where they
fit (type, required, nullable, unique, primary_key). Each check kind has a
default severity and on_violation policy that a column can override under
`checks:`.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ColumnType = Literal["integer", "number", "string", "date", "timestamp", "boolean"]
Severity = Literal["error", "warn"]
OnViolation = Literal["propose_fix", "quarantine", "report_only"]

CHECK_KINDS = (
    "required",
    "type",
    "format",
    "not_null",
    "unique",
    "min",
    "max",
    "accepted_values",
    "pattern",
    "max_null_ratio",
)

# Defaults per check kind. A column overrides any of these under `checks:`.
DEFAULT_POLICY: Dict[str, Dict[str, str]] = {
    "required": {"severity": "error", "on_violation": "propose_fix"},
    "type": {"severity": "error", "on_violation": "quarantine"},
    "format": {"severity": "warn", "on_violation": "propose_fix"},
    "not_null": {"severity": "error", "on_violation": "propose_fix"},
    "unique": {"severity": "error", "on_violation": "quarantine"},
    "min": {"severity": "error", "on_violation": "propose_fix"},
    "max": {"severity": "error", "on_violation": "propose_fix"},
    "accepted_values": {"severity": "error", "on_violation": "propose_fix"},
    "pattern": {"severity": "error", "on_violation": "propose_fix"},
    "max_null_ratio": {"severity": "warn", "on_violation": "report_only"},
    "primary_key": {"severity": "error", "on_violation": "quarantine"},
    "duplicate_rows": {"severity": "error", "on_violation": "propose_fix"},
}


class CheckPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[Severity] = None
    on_violation: Optional[OnViolation] = None


class Column(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ColumnType
    description: Optional[str] = None
    required: bool = True
    nullable: bool = True
    unique: bool = False
    min: Optional[Union[float, int, str]] = None
    max: Optional[Union[float, int, str]] = None
    accepted_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    default: Optional[Any] = None
    measure: bool = False
    max_null_ratio: Optional[float] = Field(default=None, ge=0, le=1)
    unit_factors: List[float] = Field(default_factory=list)
    checks: Dict[str, CheckPolicy] = Field(default_factory=dict)

    @field_validator("checks")
    @classmethod
    def _known_checks(cls, v: Dict[str, CheckPolicy]) -> Dict[str, CheckPolicy]:
        unknown = sorted(set(v) - set(CHECK_KINDS))
        if unknown:
            raise ValueError(f"unknown check kinds {unknown}; known: {list(CHECK_KINDS)}")
        return v

    @field_validator("min", "max", mode="before")
    @classmethod
    def _yaml_dates(cls, v: Any) -> Any:
        # An unquoted 1970-01-01 in YAML arrives as a date object.
        if isinstance(v, (dt.date, dt.datetime)):
            return v.isoformat()
        return v

    @field_validator("unit_factors")
    @classmethod
    def _positive_factors(cls, v: List[float]) -> List[float]:
        bad = [f for f in v if not f > 0 or f == 1]
        if bad:
            raise ValueError(f"unit_factors must be positive and not 1, got {bad}")
        return v

    @model_validator(mode="after")
    def _bounds_match_type(self) -> "Column":
        from .values import INVALID, parse

        for name in ("min", "max"):
            bound = getattr(self, name)
            if bound is None:
                continue
            if self.type in ("string", "boolean"):
                raise ValueError(f"{name} does not apply to a {self.type} column")
            if self.type in ("integer", "number") and (isinstance(bound, bool) or not isinstance(bound, (int, float))):
                raise ValueError(f"{name} must be a number for a {self.type} column")
            if self.type == "integer" and not float(bound).is_integer():
                raise ValueError(f"{name} must be a whole number for an integer column")
            if self.type in ("date", "timestamp"):
                if bound != "today":
                    try:
                        dt.date.fromisoformat(str(bound))
                    except ValueError as e:
                        raise ValueError(f"{name} must be an ISO date or 'today'") from e
        if self.measure and self.type not in ("integer", "number"):
            raise ValueError("measure: true only applies to integer or number columns")
        if self.unit_factors and self.type not in ("integer", "number"):
            raise ValueError("unit_factors only applies to integer or number columns")
        if self.default is not None:
            if parse(self.default, self.type).value is INVALID:
                raise ValueError(f"default {self.default!r} is not a valid {self.type}")
            if self.accepted_values is not None and self.default not in self.accepted_values:
                raise ValueError(f"default {self.default!r} is not in accepted_values")
        return self


class DatasetPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[Severity] = None
    on_violation: Optional[OnViolation] = None


class AnomalySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mad_threshold: float = Field(default=3.5, gt=0)
    min_rows: int = Field(default=5, ge=3)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    apiVersion: Literal["shdi/v1"] = "shdi/v1"
    name: str
    version: str = "0.1.0"
    description: Optional[str] = None
    primary_key: List[str] = Field(default_factory=list)
    max_null_ratio: Optional[float] = Field(default=None, ge=0, le=1)
    allow_duplicate_rows: bool = False
    checks: Dict[str, DatasetPolicy] = Field(default_factory=dict)
    anomaly: AnomalySettings = Field(default_factory=AnomalySettings)
    columns: Dict[str, Column]

    @field_validator("checks")
    @classmethod
    def _dataset_checks(cls, v: Dict[str, DatasetPolicy]) -> Dict[str, DatasetPolicy]:
        unknown = sorted(set(v) - {"primary_key", "duplicate_rows", "max_null_ratio"})
        if unknown:
            raise ValueError(f"unknown dataset checks {unknown}")
        return v

    @model_validator(mode="after")
    def _pk_columns_exist(self) -> "Contract":
        missing = [c for c in self.primary_key if c not in self.columns]
        if missing:
            raise ValueError(f"primary_key columns not declared in columns: {missing}")
        for c in self.primary_key:
            if self.columns[c].nullable:
                raise ValueError(f"primary key column '{c}' must be nullable: false")
        return self

    def policy(self, kind: str, column: Optional[str] = None) -> Dict[str, str]:
        """Severity and on_violation for one check, after overrides."""
        out = dict(DEFAULT_POLICY[kind])
        override: Optional[BaseModel] = None
        if column is not None and column in self.columns:
            override = self.columns[column].checks.get(kind)
        if override is None:
            # Dataset-level policy (primary_key, duplicate_rows, max_null_ratio)
            # also acts as the default for that check on every column.
            override = self.checks.get(kind)
        if override is not None:
            for k in ("severity", "on_violation"):
                val = getattr(override, k)
                if val is not None:
                    out[k] = val
        return out

    def measures(self) -> List[str]:
        return [n for n, c in self.columns.items() if c.measure]

    def fingerprint(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_contract(path: Union[str, Path]) -> Contract:
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"contract {p} is not valid YAML: {e}") from None
    if not isinstance(data, dict):
        raise ValueError(f"contract {p} is not a YAML mapping")
    return Contract.model_validate(data)


def contract_from_dict(data: Dict[str, Any]) -> Contract:
    return Contract.model_validate(data)


def resolve_bound(bound: Any, type_: str, as_of: dt.date) -> Any:
    """Turn a contract bound into a comparable value for the column type."""
    if bound is None:
        return None
    if type_ == "date":
        return as_of if bound == "today" else dt.date.fromisoformat(str(bound))
    if type_ == "timestamp":
        if bound == "today":
            return dt.datetime.combine(as_of, dt.time.max)
        return dt.datetime.fromisoformat(str(bound))
    return bound
