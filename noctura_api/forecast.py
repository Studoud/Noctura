from __future__ import annotations

import json
import math
from datetime import date, datetime, time, timedelta
from typing import Any, Literal
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from noctura_api.data_manager import GROUPS, ZONES, DataManager
from noctura_api.ml_service import ModelService
from noctura_api.port_config import (
    PORT_LATITUDE,
    PORT_LONGITUDE,
    PORT_NAME,
    PORT_TIMEZONE,
)

LATITUDE = PORT_LATITUDE
LONGITUDE = PORT_LONGITUDE
TIMEZONE = PORT_TIMEZONE
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
THUNDERSTORM_CODES = {95, 96, 99}


class WeatherForecastError(RuntimeError):
    pass


class HistoricalReferenceError(RuntimeError):
    pass


def fetch_hourly_weather() -> dict[str, Any]:
    params = urlencode(
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": "is_day,weather_code,cloud_cover",
            "timezone": TIMEZONE,
            "forecast_days": 2,
        }
    )
    try:
        with urlopen(f"{OPEN_METEO_URL}?{params}", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeatherForecastError(
            "Impossible de récupérer les prévisions horaires depuis Open-Meteo."
        ) from exc

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise WeatherForecastError("La réponse Open-Meteo ne contient pas les heures prévues.")
    return hourly


def available_zones() -> list[dict[str, str]]:
    return [
        {
            "zone_specifique": zone,
            "zone_group_id": group,
            "zone_group_name": GROUPS[group],
        }
        for zone, group in ZONES.items()
    ]


def predict_remaining_today(
    data_manager: DataManager,
    model_service: ModelService,
    *,
    day: Literal["today", "tomorrow"] = "today",
    zone: str | None = None,
    now: datetime | None = None,
    hourly_weather: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = model_service.active_model()
    if model is None:
        raise HistoricalReferenceError("Le modèle n'est pas encore entraîné.")

    local_tz = ZoneInfo(TIMEZONE)
    current_time = (now or datetime.now(local_tz)).astimezone(local_tz)
    target_date = current_time.date() + timedelta(days=day == "tomorrow")
    if zone is not None and zone not in ZONES:
        raise KeyError(zone)
    selected_zones = [zone] if zone else list(ZONES)
    future_slots = []
    for hour_start in range(0, 24, 2):
        slot_start = datetime.combine(
            target_date, time(hour=hour_start), tzinfo=local_tz
        )
        if day == "today" and slot_start <= current_time:
            continue
        future_slots.append((hour_start, slot_start))

    if hourly_weather is not None:
        hourly = hourly_weather
    elif future_slots:
        hourly = fetch_hourly_weather()
    else:
        hourly = {"time": [], "is_day": [], "weather_code": []}
    times = hourly.get("time")
    daylight_values = hourly.get("is_day")
    weather_codes = hourly.get("weather_code")
    cloud_values = hourly.get("cloud_cover")
    if not all(isinstance(values, list) for values in (times, daylight_values, weather_codes)):
        raise WeatherForecastError(
            "Les prévisions Open-Meteo ne contiennent pas les variables nécessaires."
        )

    forecast_by_time: dict[str, tuple[Any, Any, Any]] = {}
    for index, timestamp in enumerate(times):
        if index >= len(daylight_values) or index >= len(weather_codes):
            break
        cloud_cover = (
            cloud_values[index]
            if isinstance(cloud_values, list) and index < len(cloud_values)
            else None
        )
        forecast_by_time[timestamp] = (
            daylight_values[index],
            weather_codes[index],
            cloud_cover,
        )

    if future_slots:
        attendance = data_manager.read_attendance()
        training_hours = set(
            attendance.loc[
                attendance["is_synthetic_daytime"] == 0, "hour_start"
            ].astype(int)
        )
    else:
        attendance = None
        training_hours = set()
    items: list[dict[str, Any]] = []
    for hour_start, slot_start in future_slots:
        assert attendance is not None
        timestamp_key = slot_start.strftime("%Y-%m-%dT%H:%M")
        forecast = forecast_by_time.get(timestamp_key)
        if forecast is None:
            raise WeatherForecastError(
                f"La prévision météo est indisponible pour {timestamp_key}."
            )
        is_day, weather_code, cloud_cover = forecast
        if is_day not in (0, 1) or weather_code is None:
            raise WeatherForecastError(
                f"Les variables météo nécessaires sont manquantes pour {timestamp_key}."
            )
        is_storm = int(int(weather_code) in THUNDERSTORM_CODES)
        is_winter = int(current_time.month in (12, 1, 2))

        for selected_zone in selected_zones:
            group = ZONES[selected_zone]
            zone_hour_rows = attendance.loc[
                (attendance["zone_specifique"] == selected_zone)
                & (attendance["hour_start"] == hour_start),
                "indice_global",
            ]
            observed_rows = attendance.loc[
                (attendance["zone_specifique"] == selected_zone)
                & (attendance["hour_start"] == hour_start)
                & (attendance["is_synthetic_daytime"] == 0),
                "indice_global",
            ]
            if observed_rows.empty and zone_hour_rows.empty:
                raise HistoricalReferenceError(
                    f"Aucune référence historique pour {selected_zone}, "
                    f"créneau {hour_start:02d}h."
                )
            uses_synthetic_reference = observed_rows.empty
            reference_rows = zone_hour_rows if uses_synthetic_reference else observed_rows
            indice_global = int(round(float(reference_rows.median())))
            features = {
                "hour_start": hour_start,
                "zone_group_id": group,
                "zone_specifique": selected_zone,
                "is_daylight": int(is_day),
                "is_winter": is_winter,
                "is_storm": is_storm,
                "indice_global": indice_global,
            }
            items.append(
                {
                    "zone": selected_zone,
                    "zone_group_id": group,
                    "time_slot": f"{hour_start:02d}h-{(hour_start + 2) % 24:02d}h",
                    "starts_at": slot_start.isoformat(),
                    "predicted": model_service.predict(
                        {
                            **features,
                            "sin_hour": math.sin(2 * math.pi * hour_start / 24),
                            "cos_hour": math.cos(2 * math.pi * hour_start / 24),
                        }
                    ),
                    "weather": {
                        "source": "Open-Meteo",
                        "is_daylight": int(is_day),
                        "weather_code": int(weather_code),
                        "cloud_cover_percent": cloud_cover,
                    },
                    "features": {
                        "is_winter": is_winter,
                        "is_storm": is_storm,
                        "indice_global": indice_global,
                        "indice_global_source": (
                            "dataset_median_including_synthetic_rows_for_zone_and_time_slot"
                            if uses_synthetic_reference
                            else "non_synthetic_historical_median_for_zone_and_time_slot"
                        ),
                    },
                    "training_coverage": (
                        "observed_training_rows_for_hour"
                        if hour_start in training_hours
                        else "no_observed_training_rows_for_hour"
                    ),
                    "unit": "individuals_per_two_hour_slot",
                }
            )

    warnings = [
        "indice_global utilise la médiane du jeu existant par zone et créneau; cette variable n'est pas fournie par Open-Meteo.",
        "Les prédictions sont des estimations de fréquentation, pas des mesures réelles.",
        "Aucune commande ni intensité d'éclairage opérationnelle n'est générée.",
    ]
    if any(
        item["features"]["indice_global_source"].startswith(
            "dataset_median_including_synthetic_rows"
        )
        for item in items
    ):
        warnings.append(
            "Pour certains créneaux, faute de lignes historiques observées, "
            "la médiane de indice_global inclut des lignes marquées synthétiques."
        )
    if any(
        item["training_coverage"] == "no_observed_training_rows_for_hour"
        for item in items
    ):
        warnings.append(
            "Le jeu d'entraînement ne contient pas de lignes observées pour certains "
            "créneaux; ces prédictions horaires sont particulièrement incertaines."
        )
    if not items:
        warnings.append(
            "Aucun créneau futur ne reste aujourd'hui; demandez les prévisions de demain."
        )

    return {
        "status": "predicted" if items else "no_future_slots",
        "port": PORT_NAME,
        "location": {"latitude": LATITUDE, "longitude": LONGITUDE},
        "date": target_date.isoformat(),
        "day": day,
        "timezone": TIMEZONE,
        "weather_source": "Open-Meteo" if items else None,
        "model_version": model["version"],
        "zone_filter": zone,
        "prediction_count": len(items),
        "items": items,
        "warnings": warnings,
    }
