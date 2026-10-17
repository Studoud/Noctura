"""Règles d'éclairage : transforment la demande prédite en niveau d'éclairage.

Les règles décident ; le modèle ne fait que prédire la demande. Chaque décision garde
la liste de ses raisons, pour être explicable. Le validateur de sécurité
(cerveau/securite.py) passe toujours après et peut relever le niveau.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

# Lumière naturelle (0-100, data/meteo_astro.py) à partir de laquelle on n'éclaire plus.
SEUIL_LUMIERE_NATURELLE = 60
# Atténuation maximale apportée par le crépuscule ou la lune (proportionnelle à la lumière).
ATTENUATION_LUMIERE_MAX = 0.5
# Paliers : (P90 de demande strictement inférieur à, niveau d'éclairage %, nom).
PALIERS_DEMANDE = (
    (5.0, 10, "veille"),
    (12.0, 30, "faible"),
    (25.0, 60, "moyenne"),
    (math.inf, 100, "forte"),
)
# Prédiction de repli (modèle indisponible) : prudence.
PLANCHER_ESTIME = 30
# Sobriété renforcée selon la sensibilité écologique (ml/biodiversite.py).
PLAFOND_BIODIVERSITE = {"alerte": 60, "vigilance": 80}
TEMPERATURE_K = {"alerte": 2200, "vigilance": 2400}
TEMPERATURE_K_DEFAUT = 2700


@dataclass
class Decision:
    zone: str
    niveau: int
    temperature_k: int
    p50: float
    p90: float
    estime: bool
    raisons: list[str] = field(default_factory=list)
    securite: list[str] = field(default_factory=list)

    def en_dict(self) -> dict[str, Any]:
        return asdict(self)


def palier(p90: float) -> tuple[int, str]:
    for seuil, niveau, nom in PALIERS_DEMANDE:
        if p90 < seuil:
            return niveau, nom
    return PALIERS_DEMANDE[-1][1], PALIERS_DEMANDE[-1][2]


def decider(zone: str, p50: float, p90: float, estime: bool, contexte: Mapping[str, Any]) -> Decision:
    """Niveau d'éclairage (0-100 %) d'une zone. On raisonne sur le P90 : sous-estimer est dangereux."""
    lumiere = float(contexte.get("facteur_lumiere_naturelle") or 0)
    bio = contexte.get("niveau_biodiversite", "inconnu")
    raisons: list[str] = []

    if lumiere >= SEUIL_LUMIERE_NATURELLE:
        niveau = 0.0
        raisons.append(f"lumière naturelle {lumiere:.0f} % ≥ {SEUIL_LUMIERE_NATURELLE} % : éclairage inutile")
    else:
        base, nom = palier(p90)
        niveau = float(base)
        raisons.append(f"demande P90 {p90:.1f} : palier {nom} ({base} %)")
        if lumiere > 0:
            attenuation = ATTENUATION_LUMIERE_MAX * lumiere / SEUIL_LUMIERE_NATURELLE
            niveau *= 1 - attenuation
            raisons.append(f"lumière naturelle {lumiere:.0f} % (crépuscule ou lune) : -{100 * attenuation:.0f} %")
        if estime and niveau < PLANCHER_ESTIME:
            niveau = PLANCHER_ESTIME
            raisons.append(f"prédiction de repli : plancher de prudence {PLANCHER_ESTIME} %")
        plafond = PLAFOND_BIODIVERSITE.get(bio)
        if plafond is not None and niveau > plafond:
            niveau = plafond
            raisons.append(f"sensibilité écologique « {bio} » : plafond {plafond} %")

    return Decision(
        zone=zone,
        niveau=int(round(max(0.0, min(100.0, niveau)))),
        temperature_k=TEMPERATURE_K.get(bio, TEMPERATURE_K_DEFAUT),
        p50=round(p50, 1),
        p90=round(p90, 1),
        estime=estime,
        raisons=raisons,
    )
