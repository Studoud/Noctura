"""Validateur de sécurité : dernier mot sur chaque décision d'éclairage.

Il ne fait que RELEVER des niveaux (planchers) ; il ne baisse jamais une décision. Il
s'applique après les règles, y compris après le plafond de sobriété écologique : la
sécurité des personnes et de la navigation passe avant.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from cerveau.prediction import CHEMIN_ZONES, zones_dataset
from cerveau.regles import SEUIL_LUMIERE_NATURELLE, Decision

ZONES_NAVIGATION = ("entree_du_port", "capitainerie")
ZONES_PORT = zones_dataset("A")

SEUIL_PRESENCE_P90 = 5.0          # au-delà, des personnes sont probables : jamais éteint
PLANCHER_PRESENCE = 20
PLANCHER_NAVIGATION_NUIT = 30     # repérage depuis la mer
PLANCHER_TEMPETE = 50             # secours, amarrages
SEUIL_BROUILLARD = 50             # risque_brouillard (0-100)
PLANCHER_BROUILLARD_PORT = 60
PLANCHER_BROUILLARD = 40


def _charger_zones_critiques() -> dict[str, tuple[int, str]]:
    """zone -> (plancher de nuit, motif), depuis cerveau/port_porquerolles.json."""
    try:
        config = json.loads(CHEMIN_ZONES.read_text(encoding="utf-8")).get("zones_critiques", {})
        return {z: (int(c["plancher_nuit"]), c.get("motif", "zone critique")) for z, c in config.items()}
    except Exception:
        # Fichier illisible : on garde la zone critique connue plutôt que de la perdre.
        return {"quai_sud_commerces": (50, "zone critique")}


ZONES_CRITIQUES = _charger_zones_critiques()


def valider(decision: Decision, contexte: Mapping[str, Any]) -> Decision:
    nuit = float(contexte.get("facteur_lumiere_naturelle") or 0) < SEUIL_LUMIERE_NATURELLE
    brouillard = float(contexte.get("risque_brouillard") or 0)
    zone = decision.zone
    planchers: list[tuple[int, str]] = []

    if nuit and decision.p90 >= SEUIL_PRESENCE_P90:
        planchers.append((PLANCHER_PRESENCE, f"présence probable (P90 ≥ {SEUIL_PRESENCE_P90:.0f}) : jamais éteint"))
    if nuit and zone in ZONES_CRITIQUES:
        plancher, motif = ZONES_CRITIQUES[zone]
        planchers.append((plancher, motif))
    if nuit and zone in ZONES_NAVIGATION:
        planchers.append((PLANCHER_NAVIGATION_NUIT, "zone de navigation de nuit"))
    if nuit and contexte.get("is_storm"):
        planchers.append((PLANCHER_TEMPETE, "tempête"))
    if brouillard >= SEUIL_BROUILLARD:
        plancher = PLANCHER_BROUILLARD_PORT if zone in ZONES_PORT else PLANCHER_BROUILLARD
        planchers.append((plancher, f"brouillard (risque {brouillard:.0f} %)"))
    if nuit and contexte.get("is_storm") is None:
        planchers.append((PLANCHER_PRESENCE, "météo indisponible : tempête non exclue"))

    for plancher, motif in sorted(planchers, reverse=True):
        if decision.niveau < plancher:
            decision.securite.append(f"{motif} : relevé de {decision.niveau} % à {plancher} %")
            decision.niveau = plancher
    decision.niveau = max(0, min(100, decision.niveau))
    return decision
