"""Simulateur de fréquentation multi-journées, pour la DÉMONSTRATION (hackathon).

Les données produites sont INVENTÉES. Un modèle entraîné dessus apprend ce générateur,
pas le vrai port : il sert à montrer la chaîne complète (features, entraînement,
service, moteur de décision), jamais à piloter l'éclairage réel. Chaque ligne porte
source=simule, et ml/train.py --demo produit un modèle marqué donnees_simulees,
refusé par PREDICTION_MODE=ml.

Point de départ : la journée réelle de data/dataset_ml_frequentation_24h.csv (profil
par zone, jour de semaine estival). Effets simulés, tous multiplicatifs :
- saison touristique (pic fin juillet, creux en hiver) ;
- week-end (+20 %) ;
- obscurité : un créneau de jour en été qui tombe de nuit en hiver perd 40 % ;
- tempête (rafales, orage, vagues) : -55 %, sauf capitainerie (-10 %, veille des amarrages) ;
- aléa journalier non expliqué (log-normal, sigma 0,12) puis tirage de Poisson.
Les drapeaux is_daylight / is_winter / is_storm sont calculés par ml/features.py, avec
exactement les mêmes règles qu'en service.

    python -m ml.simulation            -> data/dataset_simule.csv
"""
from __future__ import annotations

import csv
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from cerveau.prediction import PROFIL_CRENEAUX  # noqa: E402
from ml import features as F  # noqa: E402

SEED = 2025
DEBUT, FIN = date(2024, 1, 1), date(2025, 12, 31)
SORTIE = RACINE / "data" / "dataset_simule.csv"

JOUR_REFERENCE = 176          # ~25 juin : date de la journée réelle (déduite de is_daylight)
PIC_SAISON, LARGEUR_SAISON, PLANCHER_SAISON = 205, 50.0, 0.25
EFFET_WEEKEND = 1.20
EFFET_NUIT_PRECOCE = 0.60
EFFET_TEMPETE = 0.45
EFFET_TEMPETE_ZONE = {"capitainerie": 0.90}
SIGMA_ALEA_JOUR = 0.12
PROBA_TEMPETE = {"hiver": 0.10, "intersaison": 0.05, "ete": 0.02}
PROBA_ORAGE = {"hiver": 0.0, "intersaison": 0.03, "ete": 0.03}

HEURES = tuple(range(0, 24, 2))
JOUR_ETE = {h: int(6 <= h < 22) for h in HEURES}  # is_daylight de la journée réelle
NOMS_GROUPES = {"G1": "Port et débarcadère", "G2": "Village et commerces"}
COLONNES = (
    "date", "creneau", "hour_start", "hour_end", "sin_hour", "cos_hour", "zone_group_id",
    "zone_group_name", "zone_specifique", "is_daylight", "is_winter", "is_storm",
    "is_synthetic_daytime", "indice_global", "frequentation_target", "source",
    "rafales_kmh", "code_meteo", "hauteur_vagues_m",
)


def facteur_saison(jour: date) -> float:
    def brut(doy: int) -> float:
        return PLANCHER_SAISON + (1 - PLANCHER_SAISON) * math.exp(-(((doy - PIC_SAISON) / LARGEUR_SAISON) ** 2))

    return brut(jour.timetuple().tm_yday) / brut(JOUR_REFERENCE)


def saison(jour: date) -> str:
    if jour.month in (12, 1, 2):
        return "hiver"
    return "ete" if jour.month in (6, 7, 8) else "intersaison"


def meteo_journee(jour: date, rng: np.random.Generator) -> list[dict[str, float]]:
    """Météo par créneau : rafales de fond, épisode de tempête éventuel, orage éventuel."""
    s = saison(jour)
    fond = rng.gamma(3.0, (12.0 if s == "hiver" else 8.0))
    rafales = fond + rng.normal(0.0, 5.0, len(HEURES))
    codes = np.full(len(HEURES), 1)
    if rng.random() < PROBA_TEMPETE[s]:
        debut, duree = rng.integers(0, len(HEURES)), rng.integers(3, 9)
        for i in range(debut, min(len(HEURES), debut + duree)):
            rafales[i] += rng.uniform(35.0, 60.0)
    if rng.random() < PROBA_ORAGE[s]:
        debut = rng.integers(7, 11)
        codes[debut:debut + 2] = 95
    rafales = np.clip(rafales, 0.0, None)
    vagues = np.clip(0.3 + rafales / 30.0 + rng.normal(0.0, 0.2, len(HEURES)), 0.0, None)
    return [{"rafales_kmh": round(float(r), 1), "code_meteo": int(c), "hauteur_vagues_m": round(float(v), 2)}
            for r, c, v in zip(rafales, codes, vagues)]


def simuler(debut: date = DEBUT, fin: date = FIN, seed: int = SEED) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    tz = ZoneInfo(F.FUSEAU)
    lignes = []
    jour = debut
    while jour <= fin:
        meteo = meteo_journee(jour, rng)
        alea = rng.lognormal(0.0, SIGMA_ALEA_JOUR)
        base = facteur_saison(jour) * (EFFET_WEEKEND if jour.weekday() >= 5 else 1.0) * alea
        for i, h in enumerate(HEURES):
            t = datetime(jour.year, jour.month, jour.day, h, tzinfo=tz)
            is_daylight = F.calculer_is_daylight(t)
            is_storm = F.calculer_is_storm(meteo[i])
            creneau = []
            for zone in F.ZONES:
                mu = PROFIL_CRENEAUX[zone][i] * base
                if JOUR_ETE[h] and not is_daylight:
                    mu *= EFFET_NUIT_PRECOCE
                if is_storm:
                    mu *= EFFET_TEMPETE_ZONE.get(zone, EFFET_TEMPETE)
                creneau.append((zone, min(100, int(rng.poisson(mu)))))
            indice = max(0, round(1.082 * np.mean([y for _, y in creneau]) - 0.367))
            angle = 2 * math.pi * h / 24
            for zone, y in creneau:
                groupe = F.GROUPE_DE_ZONE[zone]
                lignes.append({
                    "date": jour.isoformat(), "creneau": f"{h:02d}h-{(h + 2) % 24:02d}h", "hour_start": h,
                    "hour_end": h + 2, "sin_hour": math.sin(angle), "cos_hour": math.cos(angle),
                    "zone_group_id": groupe, "zone_group_name": NOMS_GROUPES[groupe], "zone_specifique": zone,
                    "is_daylight": is_daylight, "is_winter": F.calculer_is_winter(t), "is_storm": is_storm,
                    "is_synthetic_daytime": True, "indice_global": indice, "frequentation_target": y,
                    "source": "simule", **meteo[i],
                })
        jour += timedelta(days=1)
    return lignes


def main(sortie: Path = SORTIE) -> Path:
    lignes = simuler()
    with open(sortie, "w", encoding="utf-8", newline="") as f:
        ecrivain = csv.DictWriter(f, fieldnames=COLONNES)
        ecrivain.writeheader()
        ecrivain.writerows(lignes)
    jours = len({l["date"] for l in lignes})
    tempetes = sum(l["is_storm"] for l in lignes) / len(lignes)
    print(f"{len(lignes)} lignes SIMULÉES, {jours} journées, {100 * tempetes:.1f} % de créneaux en tempête -> {sortie}")
    return sortie


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else SORTIE)
