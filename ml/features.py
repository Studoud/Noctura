"""Fonction unique de features : utilisée à l'entraînement (ml/train.py) ET en service
(cerveau/prediction.py). Toute feature doit être calculable en temps réel à partir de
l'heure, de la date et de la météo live.

Règles de calcul des drapeaux (version SEUILS_VERSION)
------------------------------------------------------
- is_daylight : 1 si l'altitude du soleil au MILIEU du créneau de 2 h est > 0° au port de
  Porquerolles. Règle astronomique, indépendante de la météo. Elle reproduit le dataset
  actuel (06h-08h et 20h-22h = jour) pour une date de fin de printemps / début d'été ;
  ml/train.py vérifie cette compatibilité à chaque entraînement.
- is_winter : 1 si le mois est décembre, janvier ou février (hiver météorologique).
- saison (sin_doy, cos_doy) et is_weekend : tirés de la date seule (samedi, dimanche).
- is_storm : 1 si rafales >= 62 km/h (Beaufort 8, coup de vent) OU code WMO d'orage
  (95, 96, 99) OU hauteur de vagues >= 2,5 m. Manquant si ni rafales ni code météo ne
  sont connus.

HYPOTHÈSES : les seuils réels qui ont servi à construire le dataset ne sont pas connus
(is_winter et is_storm valent 0 sur toutes les lignes). Ces seuils sont donc des
hypothèses documentées, à remplacer par les seuils du générateur dès qu'ils seront
fournis ; changer un seuil impose de changer SEUILS_VERSION et de réentraîner.

MÉTÉO : la météo live vient d'Open-Meteo aux coordonnées du port de Porquerolles. C'est
une sortie de modèle numérique (maille de quelques km), pas une mesure sur place.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any, Mapping

SEUILS_VERSION = "seuils-v0-hypothese"

# Coordonnées du port de Porquerolles (calcul astronomique uniquement).
LAT_PORT = 43.0003
LON_PORT = 6.2006
FUSEAU = "Europe/Paris"

SEUILS: dict[str, Any] = {
    "storm_rafales_kmh": 62.0,
    "storm_codes_wmo": (95, 96, 99),
    "storm_vagues_m": 2.5,
    "winter_mois": (12, 1, 2),
    "daylight_altitude_soleil_deg": 0.0,
}

ZONES: tuple[str, ...] = (
    "capitainerie",
    "entree_du_port",
    "pontons_centraux",
    "quai_est",
    "quai_nord",
    "quai_ouest",
    "acces_village",
    "quai_sud_commerces",
)

GROUPE_DE_ZONE: dict[str, str] = {
    "capitainerie": "G1",
    "entree_du_port": "G1",
    "pontons_centraux": "G1",
    "quai_est": "G1",
    "quai_nord": "G1",
    "quai_ouest": "G1",
    "acces_village": "G2",
    "quai_sud_commerces": "G2",
}

NOMS_FEATURES: tuple[str, ...] = (
    "sin_hour",
    "cos_hour",
    "sin_2hour",
    "cos_2hour",
    *(f"zone={z}" for z in ZONES),
    "sin_doy",
    "cos_doy",
    "is_weekend",
    "is_daylight",
    "is_winter",
    "is_storm",
)

# Groupes de colonnes, pour l'importance par permutation.
GROUPES_FEATURES: dict[str, tuple[str, ...]] = {
    "heure": ("sin_hour", "cos_hour", "sin_2hour", "cos_2hour"),
    "zone": tuple(f"zone={z}" for z in ZONES),
    "saison": ("sin_doy", "cos_doy"),
    "is_weekend": ("is_weekend",),
    "is_daylight": ("is_daylight",),
    "is_winter": ("is_winter",),
    "is_storm": ("is_storm",),
}


class FeatureManquante(ValueError):
    """Une feature ne peut pas être calculée : l'appelant doit se replier sur le profil."""


def _localiser(t: datetime) -> datetime:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(FUSEAU)
    return t.replace(tzinfo=tz) if t.tzinfo is None else t.astimezone(tz)


def creneau_de(t: datetime) -> int:
    """Heure de début du créneau de 2 h contenant t (0, 2, ..., 22), en heure locale."""
    return (_localiser(t).hour // 2) * 2


def calculer_is_daylight(t: datetime) -> int:
    from astral import Observer
    from astral.sun import elevation

    t = _localiser(t)
    milieu = t.replace(hour=creneau_de(t), minute=0, second=0, microsecond=0) + timedelta(hours=1)
    altitude = elevation(Observer(latitude=LAT_PORT, longitude=LON_PORT), milieu)
    return int(altitude > SEUILS["daylight_altitude_soleil_deg"])


def calculer_is_winter(t: datetime) -> int:
    return int(_localiser(t).month in SEUILS["winter_mois"])


def calculer_is_storm(meteo: Mapping[str, Any] | None) -> int:
    if not meteo:
        raise FeatureManquante("météo absente : is_storm incalculable")
    rafales = meteo.get("rafales_kmh")
    code = meteo.get("code_meteo")
    vagues = meteo.get("hauteur_vagues_m")
    if rafales is None and code is None:
        raise FeatureManquante("ni rafales ni code météo : is_storm incalculable")
    tempete = (
        (rafales is not None and float(rafales) >= SEUILS["storm_rafales_kmh"])
        or (code is not None and int(code) in SEUILS["storm_codes_wmo"])
        or (vagues is not None and float(vagues) >= SEUILS["storm_vagues_m"])
    )
    return int(tempete)


def vecteur(
    zone: str,
    hour_start: int,
    is_daylight: int,
    is_winter: int,
    is_storm: int,
    jour: date,
) -> list[float]:
    """Encodage commun entraînement / service. Ordre des colonnes = NOMS_FEATURES."""
    if jour is None:
        raise FeatureManquante("date manquante")
    if zone not in ZONES:
        raise FeatureManquante(f"zone inconnue : {zone!r}")
    for nom, valeur in (("is_daylight", is_daylight), ("is_winter", is_winter), ("is_storm", is_storm)):
        if valeur is None:
            raise FeatureManquante(f"{nom} manquant")
    angle = 2.0 * math.pi * int(hour_start) / 24.0
    angle_an = 2.0 * math.pi * (jour.timetuple().tm_yday - 1) / 365.25
    return [
        math.sin(angle),
        math.cos(angle),
        math.sin(2.0 * angle),
        math.cos(2.0 * angle),
        *(1.0 if zone == z else 0.0 for z in ZONES),
        math.sin(angle_an),
        math.cos(angle_an),
        float(jour.weekday() >= 5),
        float(int(is_daylight)),
        float(int(is_winter)),
        float(int(is_storm)),
    ]


def drapeaux_depuis_contexte(t: datetime, meteo: Mapping[str, Any] | None) -> dict[str, int]:
    """is_daylight, is_winter, is_storm à partir de l'heure, de la date et de la météo live."""
    try:
        is_daylight = calculer_is_daylight(t)
    except FeatureManquante:
        raise
    except Exception as exc:  # astral absent ou calcul impossible
        raise FeatureManquante(f"is_daylight incalculable ({exc})") from exc
    return {
        "is_daylight": is_daylight,
        "is_winter": calculer_is_winter(t),
        "is_storm": calculer_is_storm(meteo),
    }


def features_service(zone: str, t: datetime, meteo: Mapping[str, Any] | None) -> list[float]:
    """Features d'une zone à l'instant t. Lève FeatureManquante si l'une manque."""
    drapeaux = drapeaux_depuis_contexte(t, meteo)
    return vecteur(zone, creneau_de(t), jour=_localiser(t).date(), **drapeaux)
