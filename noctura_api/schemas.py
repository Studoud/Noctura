from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WeatherImport(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "source": "weather_api",
                    "timestamp": "2026-10-17T12:00:00+02:00",
                    "measurements": {
                        "cloud_cover_percent": 60,
                        "natural_light_factor_percent": 42,
                    },
                }
            ]
        },
    )

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: Literal["weather_api", "manual"] = "weather_api"
    measurements: dict[str, Any] = Field(
        description="Mesures météo/astronomiques observées, sans valeur par défaut inventée."
    )

    @field_validator("measurements")
    @classmethod
    def validate_measurements(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not values:
            raise ValueError("Au moins une mesure météo est requise.")
        for key, value in values.items():
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError(f"La mesure {key} doit être une valeur simple.")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"La mesure {key} doit être finie.")
        for key in ("cloud_cover_percent", "natural_light_factor_percent", "couverture_nuageuse"):
            value = values.get(key)
            if value is not None and (
                isinstance(value, bool) or not 0 <= float(value) <= 100
            ):
                raise ValueError(f"{key} doit être compris entre 0 et 100.")
        return values

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp doit inclure un fuseau horaire.")
        return value


class AttendancePrediction(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "hour_start": 14,
                    "zone_group_id": "G1",
                    "zone_specifique": "quai_est",
                    "is_daylight": 1,
                    "is_winter": 0,
                    "is_storm": 0,
                    "indice_global": 11,
                }
            ]
        },
    )

    hour_start: int = Field(ge=0, le=22, description="Début du créneau de deux heures.")
    zone_group_id: str = Field(min_length=1)
    zone_specifique: str = Field(min_length=1)
    is_daylight: int = Field(ge=0, le=1)
    is_winter: int = Field(ge=0, le=1)
    is_storm: int = Field(ge=0, le=1)
    indice_global: int = Field(ge=0)

    @field_validator("hour_start")
    @classmethod
    def validate_two_hour_slot(cls, value: int) -> int:
        if value % 2:
            raise ValueError("hour_start doit correspondre au début d'un créneau de deux heures.")
        return value

    def model_features(self) -> dict[str, Any]:
        from math import cos, pi, sin

        return {
            "hour_start": self.hour_start,
            "sin_hour": sin(2 * pi * self.hour_start / 24),
            "cos_hour": cos(2 * pi * self.hour_start / 24),
            "zone_group_id": self.zone_group_id,
            "zone_specifique": self.zone_specifique,
            "is_daylight": self.is_daylight,
            "is_winter": self.is_winter,
            "is_storm": self.is_storm,
            "indice_global": self.indice_global,
        }


class LightingRecommendationRequest(AttendancePrediction):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("timestamp")
    @classmethod
    def require_timestamp_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp doit inclure un fuseau horaire.")
        return value


class DecisionReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["approve", "reject"]
    reviewer: str = Field(min_length=1, max_length=100)
    note: str | None = Field(default=None, max_length=500)
