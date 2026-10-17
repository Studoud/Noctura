"""Prédiction de la DEMANDE (fréquentation 0-100) par zone et par instant.

Ce module ne décide jamais de l'éclairage : il fournit une estimation de fréquentation
aux règles et au validateur de sécurité, seuls décideurs.

Interface commune :
    indice(zone, t)     -> (valeur, estime)
    indice_p90(zone, t) -> (valeur, estime)
`estime` vaut True quand la valeur est une estimation de repli (zone inconnue, modèle
ML indisponible, feature manquante) plutôt que la réponse normale de la source
configurée : les règles doivent alors rester prudentes.

Sélection par la variable d'environnement PREDICTION_MODE (profil par défaut) :
- profil : profil horaire interpolé ;
- ml     : modèle entraîné sur données réelles (ml/model.joblib), refuse tout modèle simulé ;
- demo   : modèle de démonstration entraîné sur données SIMULÉES (ml/model_demo.joblib).
"""
from __future__ import annotations

import json
import logging
import os
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

log = logging.getLogger(__name__)

RACINE = Path(__file__).resolve().parent.parent
CHEMIN_ZONES = Path(__file__).resolve().parent / "port_porquerolles.json"
CHEMIN_MODELE = RACINE / "ml" / "model.joblib"
CHEMIN_MODELE_DEMO = RACINE / "ml" / "model_demo.joblib"
BORNE_MIN, BORNE_MAX = 0.0, 100.0

# Profil de référence : fréquentation par zone au début de chaque créneau de 2 h,
# issue de data/dataset_ml_frequentation_24h.csv (journée unique, 06h-20h synthétique).
_HEURES = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
PROFIL_CRENEAUX: dict[str, tuple[float, ...]] = {
    "acces_village": (10, 6, 8, 11, 19, 28, 32, 26, 24, 22, 18, 15),
    "capitainerie": (5, 3, 5, 8, 14, 18, 16, 15, 14, 12, 10, 7),
    "entree_du_port": (7, 5, 7, 10, 16, 22, 24, 20, 18, 17, 14, 10),
    "pontons_centraux": (8, 5, 8, 12, 20, 26, 28, 25, 22, 21, 18, 12),
    "quai_est": (6, 4, 7, 9, 15, 20, 22, 19, 17, 16, 14, 9),
    "quai_nord": (5, 4, 6, 8, 14, 18, 20, 18, 16, 15, 12, 8),
    "quai_ouest": (9, 6, 9, 11, 18, 25, 28, 24, 22, 21, 20, 14),
    "quai_sud_commerces": (13, 7, 8, 10, 22, 32, 38, 30, 28, 27, 24, 20),
}
# Marge du P90 du profil : heuristique conservatrice NON calibrée (pas de données pour le faire).
MARGE_P90_PROFIL = (1.25, 2.0)


def _borner(valeur: float) -> float:
    return float(min(BORNE_MAX, max(BORNE_MIN, valeur)))


def interpolation_circulaire(x: float, xs: list[float], ys: list[float], periode: float = 24.0) -> float:
    """Interpolation linéaire sur un axe circulaire (heures de la journée)."""
    points = sorted(zip(xs, ys))
    if not points:
        raise ValueError("aucun point d'interpolation")
    if len(points) == 1:
        return float(points[0][1])
    x = x % periode
    for xi, yi in points:
        if xi == x:
            return float(yi)
    avant = max((p for p in points if p[0] < x), default=None)
    apres = min((p for p in points if p[0] > x), default=None)
    if avant is None:
        avant = (points[-1][0] - periode, points[-1][1])
    if apres is None:
        apres = (points[0][0] + periode, points[0][1])
    poids = (x - avant[0]) / (apres[0] - avant[0])
    return float(avant[1] + (apres[1] - avant[1]) * poids)


# --------------------------------------------------------------------------- zones

def _normaliser(texte: str) -> str:
    sans_accents = unicodedata.normalize("NFKD", str(texte)).encode("ascii", "ignore").decode()
    return " ".join(sans_accents.lower().replace("_", " ").replace("-", " ").split())


def _charger_correspondance(chemin: Path = CHEMIN_ZONES) -> dict[str, tuple[str, ...]]:
    """Nom normalisé (zone A/B, alias, zone du dataset) -> zones du dataset."""
    table: dict[str, tuple[str, ...]] = {_normaliser(z): (z,) for z in PROFIL_CRENEAUX}
    try:
        config = json.loads(chemin.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Correspondance des zones illisible (%s) : noms du dataset seuls.", exc)
        return table
    for zone, alias in config.get("alias_zones_dataset", {}).items():
        for nom in (zone, *alias):
            table[_normaliser(nom)] = (zone,)
    for code, zone in config.get("zones", {}).items():
        membres = tuple(z for z in zone.get("zones_dataset", []) if z in PROFIL_CRENEAUX)
        if membres:
            for nom in (code, zone.get("nom", ""), *zone.get("alias", [])):
                if nom:
                    table[_normaliser(nom)] = membres
    return table


_CORRESPONDANCE = _charger_correspondance()


def zones_dataset(zone: str) -> tuple[str, ...]:
    """Zones du dataset couvertes par `zone` (code A/B, alias ou nom du dataset). Vide si inconnue."""
    return _CORRESPONDANCE.get(_normaliser(zone), ())


# --------------------------------------------------------------------------- profil

class ProfilFrequentation:
    """Profil horaire interpolé par zone (comportement par défaut et repli du ML)."""

    def _valeur_zone(self, zone: str, t: datetime) -> float:
        heure = t.hour + t.minute / 60.0
        milieux = [h + 1.0 for h in _HEURES]  # valeur du créneau placée en son milieu
        return interpolation_circulaire(heure, milieux, list(PROFIL_CRENEAUX[zone]))

    def _agreger(self, zone: str, t: datetime, p90: bool) -> tuple[float, bool]:
        membres = zones_dataset(zone)
        estime = not membres
        if not membres:  # zone inconnue : on suppose la zone la plus fréquentée (sécurité)
            membres = tuple(PROFIL_CRENEAUX)
        valeur = max(self._valeur_zone(z, t) for z in membres)
        if p90:
            facteur, ajout = MARGE_P90_PROFIL
            valeur = valeur * facteur + ajout
        return _borner(valeur), estime

    def indice(self, zone: str, t: datetime) -> tuple[float, bool]:
        return self._agreger(zone, t, p90=False)

    def indice_p90(self, zone: str, t: datetime) -> tuple[float, bool]:
        return self._agreger(zone, t, p90=True)


# --------------------------------------------------------------------------- ML

_CACHE_METEO: dict[str, Any] = {"instant": 0.0, "valeur": None}
DUREE_CACHE_METEO_S = 600


def _meteo_live_en_cache() -> Mapping[str, Any] | None:
    maintenant = time.monotonic()
    if _CACHE_METEO["valeur"] is None or maintenant - _CACHE_METEO["instant"] > DUREE_CACHE_METEO_S:
        from data.meteo_astro import get_meteo_astro

        _CACHE_METEO["valeur"] = get_meteo_astro()
        _CACHE_METEO["instant"] = maintenant
    return _CACHE_METEO["valeur"]


class PredicteurML:
    """Modèle ML de fréquentation, avec repli automatique sur le profil.

    Repli si : modèle absent, illisible, incompatible avec ml/features.py, non accepté
    par le critère d'acceptation (sauf exiger_acceptation=False), feature manquante
    (météo indisponible...) ou erreur de prédiction. Un modèle entraîné sur données
    simulées est refusé sauf autoriser_simule=True (mode demo).
    """

    def __init__(
        self,
        chemin_modele: str | os.PathLike | None = None,
        fournisseur_meteo: Callable[[], Mapping[str, Any] | None] | None = None,
        profil: ProfilFrequentation | None = None,
        exiger_acceptation: bool = True,
        autoriser_simule: bool = False,
    ) -> None:
        self.profil = profil or ProfilFrequentation()
        self._fournisseur_meteo = fournisseur_meteo or _meteo_live_en_cache
        self.raison_repli: str | None = None
        self.derniere_source = "profil"
        self._artefact = self._charger(Path(chemin_modele or CHEMIN_MODELE), exiger_acceptation, autoriser_simule)

    @property
    def actif(self) -> bool:
        return self._artefact is not None

    @property
    def simule(self) -> bool:
        return bool(self._artefact and self._artefact.get("donnees_simulees"))

    def _charger(self, chemin: Path, exiger_acceptation: bool, autoriser_simule: bool) -> dict[str, Any] | None:
        try:
            import joblib

            from ml import features as F

            artefact = joblib.load(chemin)
            if list(artefact["noms_features"]) != list(F.NOMS_FEATURES):
                raise ValueError("features du modèle différentes de ml/features.py")
            if artefact["seuils_version"] != F.SEUILS_VERSION:
                raise ValueError("seuils du modèle différents de ml/features.py")
            if artefact.get("donnees_simulees") and not autoriser_simule:
                raise ValueError("modèle entraîné sur données simulées (utiliser PREDICTION_MODE=demo)")
            if exiger_acceptation and not artefact.get("accepte", False):
                raise ValueError("modèle non accepté : " + "; ".join(artefact.get("raisons_refus", [])))
            return artefact
        except Exception as exc:
            self.raison_repli = f"modèle indisponible ({exc})"
            log.warning("PredicteurML : %s, repli sur le profil.", self.raison_repli)
            return None

    def _predire_zones(self, membres: tuple[str, ...], t: datetime) -> tuple[float, float]:
        import numpy as np

        from ml import features as F

        meteo = self._fournisseur_meteo()
        drapeaux = F.drapeaux_depuis_contexte(t, meteo)
        creneau = F.creneau_de(t)
        jour = F._localiser(t).date()
        X = np.array([F.vecteur(z, creneau, jour=jour, **drapeaux) for z in membres])
        mu = np.clip(self._artefact["modele"].predict(X), BORNE_MIN, BORNE_MAX)
        p90 = np.clip(mu + self._artefact["q_conformal"] * np.sqrt(np.maximum(mu, 1.0)), mu, BORNE_MAX)
        return float(mu.max()), float(p90.max())

    def _repondre(self, zone: str, t: datetime, p90: bool) -> tuple[float, bool]:
        membres = zones_dataset(zone)
        if self._artefact is not None and membres:
            try:
                p50, haut = self._predire_zones(membres, t)
                self.derniere_source = "ml"
                return _borner(haut if p90 else p50), False
            except Exception as exc:
                self.raison_repli = f"prédiction ML impossible ({exc})"
                log.warning("PredicteurML : %s, repli sur le profil.", self.raison_repli)
        self.derniere_source = "profil"
        valeur, _ = (self.profil.indice_p90 if p90 else self.profil.indice)(zone, t)
        return valeur, True

    def indice(self, zone: str, t: datetime) -> tuple[float, bool]:
        return self._repondre(zone, t, p90=False)

    def indice_p90(self, zone: str, t: datetime) -> tuple[float, bool]:
        return self._repondre(zone, t, p90=True)


def creer_predicteur(mode: str | None = None, **options: Any) -> ProfilFrequentation | PredicteurML:
    """Prédicteur selon `mode`, sinon PREDICTION_MODE (profil par défaut)."""
    mode = (mode or os.environ.get("PREDICTION_MODE", "profil")).strip().lower()
    if mode == "ml":
        return PredicteurML(**options)
    if mode == "demo":
        options.setdefault("chemin_modele", CHEMIN_MODELE_DEMO)
        return PredicteurML(autoriser_simule=True, **options)
    if mode != "profil":
        log.warning("PREDICTION_MODE=%r inconnu, profil utilisé.", mode)
    return ProfilFrequentation()
