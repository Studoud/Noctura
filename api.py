"""API web NOCTURA : expose le moteur de décision d'éclairage à l'interface.

    uvicorn api:app --reload --port 8000
    -> documentation interactive : http://localhost:8000/docs

Variables d'environnement :
- PREDICTION_MODE = profil | ml | demo   mode par défaut si la requête n'en précise pas
- NOCTURA_CORS    = origines autorisées, séparées par des virgules (défaut : *)
"""
from __future__ import annotations

import json
import os
import threading
from datetime import date as Date
from datetime import datetime
from datetime import time as Time
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cerveau.prediction import CHEMIN_MODELE, CHEMIN_MODELE_DEMO, CHEMIN_ZONES
from cerveau.securite import ZONES_CRITIQUES
from engine import PARIS, SCENARIOS, Noctura
from ml import features as F
from ml.biodiversite import CHEMIN_SORTIE as CHEMIN_BIODIVERSITE

VERSION_API = "1.0.0"
Mode = Literal["profil", "ml", "demo"]
Scenario = Literal["live", "calme", "couvert", "tempete", "orage", "brouillard"]
NiveauBio = Literal["inconnu", "stable", "vigilance", "alerte"]

app = FastAPI(
    title="NOCTURA API",
    version=VERSION_API,
    description="Décisions d'éclairage adaptatif du port de Porquerolles : demande prédite, "
    "règles de sobriété et validateur de sécurité. Le modèle prédit la demande ; il ne décide jamais seul.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("NOCTURA_CORS", "*").split(",") if o.strip()],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- schémas


class DecisionZone(BaseModel):
    zone: str
    niveau: int
    temperature_k: int
    p50: float
    p90: float
    estime: bool
    raisons: list[str]
    securite: list[str]


class Decision(BaseModel):
    instant: str
    mode: Mode
    prediction: str
    meteo_source: str
    contexte: dict[str, Any]
    zones: list[DecisionZone]
    puissance_moyenne: float


# --------------------------------------------------------------------------- moteurs

_moteurs: dict[tuple[str, Optional[str]], Noctura] = {}
_verrou = threading.Lock()  # le moteur garde la météo de l'instant décidé : un appel à la fois


def _mode(mode: Optional[str]) -> str:
    return mode or os.environ.get("PREDICTION_MODE", "profil").strip().lower()


def _moteur(mode: str, bio: Optional[str]) -> Noctura:
    cle = (mode, bio)
    if cle not in _moteurs:
        _moteurs[cle] = Noctura(mode=mode, niveau_biodiversite=bio)
    return _moteurs[cle]


def _instant(jour: Optional[Date], heure: Optional[Time]) -> datetime:
    maintenant = datetime.now(PARIS)
    return datetime.combine(jour or maintenant.date(), heure or maintenant.time().replace(microsecond=0), tzinfo=PARIS)


def _serialiser(resultat: dict[str, Any], mode: str, scenario: str) -> dict[str, Any]:
    return dict(
        resultat,
        mode=mode,
        meteo_source="live (Open-Meteo, port de Porquerolles)" if scenario == "live" else f"scénario « {scenario} »",
        zones=[d.en_dict() for d in resultat["zones"]],
    )


def _lire_artefact(chemin: Path) -> dict[str, Any]:
    if not chemin.exists():
        return {"present": False}
    try:
        import joblib

        a = joblib.load(chemin)
        return {
            "present": True,
            "accepte": a.get("accepte"),
            "donnees_simulees": a.get("donnees_simulees", False),
            "nom_modele": a.get("nom_modele"),
            "version": a.get("version"),
            "date_entrainement": a.get("date_entrainement"),
            "metriques_test": a.get("metriques_test") or a.get("metriques_test_reel"),
            "raisons_refus": a.get("raisons_refus", []),
            "dataset": a.get("dataset"),
        }
    except Exception as exc:
        return {"present": True, "erreur": f"illisible : {exc}"}


# --------------------------------------------------------------------------- routes


@app.get("/api/sante", tags=["système"])
def sante() -> dict[str, Any]:
    """État du service et du modèle utilisé par défaut."""
    mode = _mode(None)
    moteur = _moteur(mode, None)
    return {
        "statut": "ok",
        "version_api": VERSION_API,
        "mode_par_defaut": mode,
        "prediction": moteur.source_prediction,
        "heure_serveur": datetime.now(PARIS).isoformat(timespec="seconds"),
    }


@app.get("/api/decision", response_model=Decision, tags=["décisions"])
def decision(
    date: Optional[Date] = Query(None, description="AAAA-MM-JJ (défaut : aujourd'hui)"),
    heure: Optional[Time] = Query(None, description="HH:MM (défaut : maintenant)"),
    scenario: Scenario = Query("live", description="météo réelle (live) ou scénario de démonstration"),
    mode: Optional[Mode] = Query(None, description="prédiction : profil, ml ou demo (défaut : PREDICTION_MODE)"),
    bio: Optional[NiveauBio] = Query(None, description="force le niveau de biodiversité (défaut : calculé)"),
) -> dict[str, Any]:
    """Décision d'éclairage des 8 zones à un instant donné."""
    m = _mode(mode)
    meteo = None if scenario == "live" else dict(SCENARIOS[scenario])
    with _verrou:
        resultat = _moteur(m, bio).decider(_instant(date, heure), meteo)
    return _serialiser(resultat, m, scenario)


@app.get("/api/decision/journee", response_model=list[Decision], tags=["décisions"])
def decision_journee(
    date: Optional[Date] = Query(None, description="AAAA-MM-JJ (défaut : aujourd'hui)"),
    scenario: Scenario = Query("live"),
    mode: Optional[Mode] = Query(None),
    bio: Optional[NiveauBio] = Query(None),
) -> list[dict[str, Any]]:
    """12 décisions (une par créneau de 2 h, au milieu du créneau) pour tracer une journée.
    La même météo (live actuelle ou scénario) est appliquée à tous les créneaux."""
    m = _mode(mode)
    meteo = None if scenario == "live" else dict(SCENARIOS[scenario])
    with _verrou:
        resultats = _moteur(m, bio).decider_journee(date or datetime.now(PARIS).date(), meteo)
    return [_serialiser(r, m, scenario) for r in resultats]


@app.get("/api/meteo", tags=["données"])
def meteo() -> dict[str, Any]:
    """Météo, mer, soleil et lune en direct au port (Open-Meteo, cache 10 min)."""
    from cerveau.prediction import _meteo_live_en_cache

    donnees = _meteo_live_en_cache()
    if not donnees:
        raise HTTPException(503, "météo indisponible")
    return dict(donnees)


@app.get("/api/zones", tags=["données"])
def zones() -> dict[str, Any]:
    """Zones du port, groupes A/B, alias et zones critiques."""
    config = json.loads(CHEMIN_ZONES.read_text(encoding="utf-8"))
    return {
        "zones": [{"zone": z, "groupe": F.GROUPE_DE_ZONE[z]} for z in F.ZONES],
        "groupes": config.get("zones", {}),
        "zones_critiques": {z: {"plancher_nuit": p, "motif": m} for z, (p, m) in ZONES_CRITIQUES.items()},
        "port": {k: config.get(k) for k in ("port", "latitude", "longitude")},
    }


@app.get("/api/scenarios", tags=["données"])
def scenarios() -> dict[str, Any]:
    """Scénarios météo de démonstration disponibles (en plus de « live »)."""
    return SCENARIOS


@app.get("/api/biodiversite", tags=["données"])
def biodiversite() -> dict[str, Any]:
    """Indicateur de sensibilité écologique (tendance par espèce, niveau global)."""
    if not CHEMIN_BIODIVERSITE.exists():
        raise HTTPException(404, "indicateur absent : lancer python -m ml.biodiversite")
    return json.loads(CHEMIN_BIODIVERSITE.read_text(encoding="utf-8"))


@app.get("/api/modele", tags=["système"])
def modele() -> dict[str, Any]:
    """Statut et métriques des modèles ML (réel et démo)."""
    return {"ml": _lire_artefact(CHEMIN_MODELE), "demo": _lire_artefact(CHEMIN_MODELE_DEMO)}
