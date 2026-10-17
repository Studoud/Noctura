"""Données météo, marines et astro en temps réel (sans clé API) — port de Toulon."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

import requests
from astral import Depression, LocationInfo
from astral.moon import azimuth as moon_azimuth
from astral.moon import elevation as moon_elevation
from astral.moon import moonrise, moonset
from astral.moon import phase as moon_phase_days
from astral.sun import azimuth as sun_azimuth
from astral.sun import elevation as sun_elevation
from astral.sun import sun

# Port de Toulon (France)
LATITUDE = 43.12
LONGITUDE = 5.93
PORT = LocationInfo(
    name="Toulon",
    region="France",
    timezone="Europe/Paris",
    latitude=LATITUDE,
    longitude=LONGITUDE,
)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
REQUEST_TIMEOUT_S = 8

DEFAULT_METEO_ASTRO: dict[str, Any] = {
    "port": "Toulon",
    "couverture_nuageuse": 50,
    "luminosite_lune": 50,
    "phase_lune": "Premier quartier",
}

# Cycle lunaire astral : 0 → 27.99
_PHASES_FR = (
    (1.75, "Nouvelle lune"),
    (5.25, "Premier croissant"),
    (8.75, "Premier quartier"),
    (12.25, "Lune gibbeuse croissante"),
    (15.75, "Pleine lune"),
    (19.25, "Lune gibbeuse décroissante"),
    (22.75, "Dernier quartier"),
    (26.25, "Dernier croissant"),
)

_WMO_FR = {
    0: "Ciel clair",
    1: "Principalement clair",
    2: "Partiellement nuageux",
    3: "Couvert",
    45: "Brouillard",
    48: "Brouillard givrant",
    51: "Bruine légère",
    53: "Bruine",
    55: "Bruine dense",
    56: "Bruine verglaçante légère",
    57: "Bruine verglaçante",
    61: "Pluie faible",
    63: "Pluie",
    65: "Pluie forte",
    66: "Pluie verglaçante légère",
    67: "Pluie verglaçante",
    71: "Neige faible",
    73: "Neige",
    75: "Neige forte",
    77: "Grains de neige",
    80: "Averses faibles",
    81: "Averses",
    82: "Averses violentes",
    85: "Averses de neige faibles",
    86: "Averses de neige",
    95: "Orage",
    96: "Orage avec grêle faible",
    99: "Orage avec grêle",
}

_VENT_CARDINAL = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO",
)


def _now() -> datetime:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Europe/Paris"))
    except Exception:
        return datetime.now().astimezone()


def _hhmm(value: datetime | None) -> str | None:
    return value.strftime("%H:%M") if value else None


def _num(value: Any, ndigits: int = 1) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _int(value: Any) -> int | None:
    if value is None:
        return None
    return int(round(float(value)))


def _phase_lune_fr(age: float) -> str:
    for seuil, nom in _PHASES_FR:
        if age < seuil:
            return nom
    return "Nouvelle lune"


def _luminosite_lune_pct(age: float) -> int:
    fraction = (1.0 - math.cos(2.0 * math.pi * age / 28.0)) / 2.0
    return int(round(max(0.0, min(1.0, fraction)) * 100))


def _cardinal(degres: float | None) -> str | None:
    if degres is None:
        return None
    index = int((float(degres) % 360) / 22.5 + 0.5) % 16
    return _VENT_CARDINAL[index]


def _description_meteo(code: int | None) -> str | None:
    if code is None:
        return None
    return _WMO_FR.get(int(code), f"Code météo {code}")


def _periode(altitude_soleil: float | None, est_jour_api: int | None) -> str:
    if altitude_soleil is not None:
        if altitude_soleil > 0:
            return "jour"
        if altitude_soleil > -6:
            return "crepuscule_civil"
        if altitude_soleil > -12:
            return "crepuscule_nautique"
        if altitude_soleil > -18:
            return "crepuscule_astronomique"
        return "nuit"
    if est_jour_api == 1:
        return "jour"
    if est_jour_api == 0:
        return "nuit"
    return "inconnu"


def _risque_brouillard(
    visibilite_m: float | None,
    humidite: float | None,
    temperature: float | None,
    point_rosee: float | None,
    vent_kmh: float | None,
    code_meteo: int | None,
) -> int:
    score = 0
    if code_meteo in {45, 48}:
        score += 70
    if visibilite_m is not None:
        if visibilite_m < 200:
            score += 70
        elif visibilite_m < 1000:
            score += 50
        elif visibilite_m < 4000:
            score += 20
    if humidite is not None:
        if humidite >= 95:
            score += 25
        elif humidite >= 90:
            score += 15
        elif humidite >= 80:
            score += 5
    if temperature is not None and point_rosee is not None:
        delta = abs(temperature - point_rosee)
        if delta <= 1:
            score += 20
        elif delta <= 2:
            score += 10
    if vent_kmh is not None and vent_kmh < 8:
        score += 10
    return min(100, score)


def _facteur_lumiere_naturelle(
    altitude_soleil: float | None,
    altitude_lune: float | None,
    luminosite_lune: int,
    couverture: int | None,
) -> int:
    """0 = noir complet, 100 = plein jour. Sert à doser l'éclairage artificiel."""
    nuages = (couverture if couverture is not None else 50) / 100.0
    if altitude_soleil is not None and altitude_soleil > 0:
        ciel = min(100.0, 70.0 + altitude_soleil * 2.0)
        return int(round(ciel * (1.0 - 0.35 * nuages)))

    crepuscule = 0.0
    if altitude_soleil is not None and altitude_soleil > -18:
        crepuscule = max(0.0, (altitude_soleil + 18.0) / 18.0) * 45.0

    clair_de_lune = 0.0
    if altitude_lune is not None and altitude_lune > 0:
        hauteur = min(altitude_lune, 45.0) / 45.0
        clair_de_lune = luminosite_lune * 0.28 * hauteur * (1.0 - 0.75 * nuages)

    return int(round(max(0.0, min(100.0, max(crepuscule, clair_de_lune)))))


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for essai in range(3):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(0.5 * (essai + 1))
    raise last_error if last_error else RuntimeError(f"{url} indisponible")


def _index_heure(times: list[str], now: datetime) -> int:
    cible = now.strftime("%Y-%m-%dT%H:00")
    index = 0
    for i, stamp in enumerate(times):
        if stamp <= cible:
            index = i
        else:
            break
    return index


def _hourly_at(hourly: dict[str, Any], key: str, index: int) -> Any:
    values = hourly.get(key)
    if not values or index >= len(values):
        return None
    return values[index]


def _meteo(now: datetime) -> dict[str, Any]:
    payload = _get_json(
        OPEN_METEO_URL,
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "current": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "is_day",
                    "precipitation",
                    "rain",
                    "showers",
                    "weather_code",
                    "cloud_cover",
                    "pressure_msl",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "wind_gusts_10m",
                ]
            ),
            "hourly": ",".join(
                [
                    "visibility",
                    "dew_point_2m",
                    "cloud_cover_low",
                    "cloud_cover_mid",
                    "cloud_cover_high",
                    "shortwave_radiation",
                    "precipitation_probability",
                ]
            ),
            "timezone": "Europe/Paris",
            "forecast_days": 1,
            "wind_speed_unit": "kmh",
        },
    )
    current = payload["current"]
    hourly = payload.get("hourly", {})
    index = _index_heure(hourly.get("time", []), now)

    code = _int(current.get("weather_code"))
    couverture = _int(current.get("cloud_cover"))
    temperature = _num(current.get("temperature_2m"))
    humidite = _int(current.get("relative_humidity_2m"))
    vent = _num(current.get("wind_speed_10m"))
    direction = _num(current.get("wind_direction_10m"), 0)
    visibilite = _num(_hourly_at(hourly, "visibility", index), 0)
    point_rosee = _num(_hourly_at(hourly, "dew_point_2m", index))

    return {
        "heure_meteo": current.get("time"),
        "temperature_c": temperature,
        "temperature_ressentie_c": _num(current.get("apparent_temperature")),
        "humidite": humidite,
        "precipitation_mm": _num(current.get("precipitation"), 2),
        "pluie_mm": _num(current.get("rain"), 2),
        "averses_mm": _num(current.get("showers"), 2),
        "probabilite_pluie": _int(_hourly_at(hourly, "precipitation_probability", index)),
        "code_meteo": code,
        "description_meteo": _description_meteo(code),
        "couverture_nuageuse": couverture,
        "nuages_bas": _int(_hourly_at(hourly, "cloud_cover_low", index)),
        "nuages_moyens": _int(_hourly_at(hourly, "cloud_cover_mid", index)),
        "nuages_hauts": _int(_hourly_at(hourly, "cloud_cover_high", index)),
        "pression_hpa": _num(current.get("pressure_msl"), 0),
        "vent_kmh": vent,
        "rafales_kmh": _num(current.get("wind_gusts_10m")),
        "direction_vent_deg": direction,
        "direction_vent": _cardinal(direction),
        "visibilite_m": visibilite,
        "point_rosee_c": point_rosee,
        "rayonnement_solaire_wm2": _num(_hourly_at(hourly, "shortwave_radiation", index), 0),
        "est_jour": bool(current.get("is_day")),
        "risque_brouillard": _risque_brouillard(
            visibilite, humidite, temperature, point_rosee, vent, code
        ),
    }


def _marine() -> dict[str, Any]:
    payload = _get_json(
        MARINE_URL,
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "current": ",".join(
                [
                    "wave_height",
                    "wave_direction",
                    "wave_period",
                    "wind_wave_height",
                    "swell_wave_height",
                    "sea_surface_temperature",
                ]
            ),
            "timezone": "Europe/Paris",
        },
    )
    current = payload["current"]
    direction = _num(current.get("wave_direction"), 0)
    return {
        "hauteur_vagues_m": _num(current.get("wave_height"), 2),
        "periode_vagues_s": _num(current.get("wave_period")),
        "direction_vagues_deg": direction,
        "direction_vagues": _cardinal(direction),
        "hauteur_vagues_vent_m": _num(current.get("wind_wave_height"), 2),
        "hauteur_houle_m": _num(current.get("swell_wave_height"), 2),
        "temperature_mer_c": _num(current.get("sea_surface_temperature")),
    }


def _lune_et_soleil(now: datetime) -> dict[str, Any]:
    observer = PORT.observer
    tz = now.tzinfo or timezone.utc
    age = float(moon_phase_days(now))
    alt_soleil = _num(sun_elevation(observer, now))
    alt_lune = _num(moon_elevation(observer, now))
    lum_lune = _luminosite_lune_pct(age)

    civil = sun(observer, date=now.date(), tzinfo=tz, dawn_dusk_depression=Depression.CIVIL)
    nautique = sun(observer, date=now.date(), tzinfo=tz, dawn_dusk_depression=Depression.NAUTICAL)
    astro = sun(observer, date=now.date(), tzinfo=tz, dawn_dusk_depression=Depression.ASTRONOMICAL)

    lever_lune = moonrise(observer, date=now.date(), tzinfo=tz)
    coucher_lune = moonset(observer, date=now.date(), tzinfo=tz)

    return {
        "luminosite_lune": lum_lune,
        "phase_lune": _phase_lune_fr(age),
        "age_lune_jours": _num(age, 2),
        "altitude_lune": alt_lune,
        "azimuth_lune": _num(moon_azimuth(observer, now), 0),
        "lune_visible": alt_lune is not None and alt_lune > 0,
        "lever_lune": _hhmm(lever_lune),
        "coucher_lune": _hhmm(coucher_lune),
        "altitude_soleil": alt_soleil,
        "azimuth_soleil": _num(sun_azimuth(observer, now), 0),
        "aube": _hhmm(civil["dawn"]),
        "lever_soleil": _hhmm(civil["sunrise"]),
        "coucher_soleil": _hhmm(civil["sunset"]),
        "crepuscule": _hhmm(civil["dusk"]),
        "aube_nautique": _hhmm(nautique["dawn"]),
        "crepuscule_nautique": _hhmm(nautique["dusk"]),
        "nuit_astronomique": _hhmm(astro["dusk"]),
        "est_nuit": alt_soleil is None or alt_soleil <= 0,
        "periode": _periode(alt_soleil, None),
    }


def get_meteo_astro() -> dict[str, Any]:
    """Retourne météo + mer + lune/soleil pour Toulon. Jamais d'exception vers l'appelant."""
    result = dict(DEFAULT_METEO_ASTRO)
    now = _now()
    result["port"] = PORT.name
    result["latitude"] = LATITUDE
    result["longitude"] = LONGITUDE
    result["heure_locale"] = now.strftime("%Y-%m-%d %H:%M")

    try:
        astro = _lune_et_soleil(now)
        result.update(astro)
    except Exception as exc:
        print(f"[meteo_astro] calcul lune/soleil indisponible ({exc}), valeurs par défaut.")
        astro = {}

    try:
        meteo = _meteo(now)
        result.update(meteo)
        result["periode"] = _periode(result.get("altitude_soleil"), int(meteo.get("est_jour") or 0))
        result["est_nuit"] = not bool(meteo.get("est_jour")) if meteo.get("est_jour") is not None else result.get("est_nuit")
    except Exception as exc:
        print(f"[meteo_astro] Open-Meteo indisponible ({exc}), météo par défaut.")
        meteo = {}

    try:
        result.update(_marine())
    except Exception as exc:
        print(f"[meteo_astro] API marine indisponible ({exc}), mer ignorée.")

    result["facteur_lumiere_naturelle"] = _facteur_lumiere_naturelle(
        result.get("altitude_soleil"),
        result.get("altitude_lune"),
        int(result.get("luminosite_lune") or 0),
        result.get("couverture_nuageuse"),
    )
    return result


if __name__ == "__main__":
    data = get_meteo_astro()
    print(f"Port : {data.get('port')} ({LATITUDE}, {LONGITUDE})  {data.get('heure_locale')}")
    unites = {
        "couverture_nuageuse": " %",
        "luminosite_lune": " %",
        "humidite": " %",
        "nuages_bas": " %",
        "nuages_moyens": " %",
        "nuages_hauts": " %",
        "probabilite_pluie": " %",
        "risque_brouillard": " %",
        "facteur_lumiere_naturelle": " %",
        "temperature_c": " °C",
        "temperature_ressentie_c": " °C",
        "point_rosee_c": " °C",
        "temperature_mer_c": " °C",
        "vent_kmh": " km/h",
        "rafales_kmh": " km/h",
        "visibilite_m": " m",
        "pression_hpa": " hPa",
        "precipitation_mm": " mm",
        "pluie_mm": " mm",
        "averses_mm": " mm",
        "hauteur_vagues_m": " m",
        "hauteur_vagues_vent_m": " m",
        "hauteur_houle_m": " m",
        "periode_vagues_s": " s",
        "altitude_soleil": " °",
        "altitude_lune": " °",
        "azimuth_soleil": " °",
        "azimuth_lune": " °",
        "rayonnement_solaire_wm2": " W/m²",
    }
    for cle, valeur in data.items():
        print(f"  {cle}: {valeur}{unites.get(cle, '')}")
