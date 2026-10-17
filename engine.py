"""Noyau NOCTURA : météo -> prédiction de la demande -> règles -> sécurité -> décision.

    python engine.py                                   # maintenant, météo live
    python engine.py --heure 22:30 --scenario tempete  # scénario de démonstration
    python engine.py --mode demo --date 2025-08-15 --heure 21:00 --bio alerte
    python engine.py --json

Rôles :
- cerveau/prediction.py : prédit la DEMANDE (profil, ou modèle ML selon PREDICTION_MODE) ;
- cerveau/regles.py     : décide du niveau d'éclairage, avec ses raisons ;
- cerveau/securite.py   : relève les niveaux si la sécurité l'exige (dernier mot).
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from cerveau import regles, securite
from cerveau.prediction import PredicteurML, creer_predicteur
from ml import features as F
from ml.biodiversite import niveau_sensibilite

PARIS = ZoneInfo("Europe/Paris")
# Champs météo recopiés dans le contexte de chaque décision (affichage côté interface).
METEO_EXPOSEE = (
    "heure_meteo", "temperature_c", "vent_kmh", "rafales_kmh", "code_meteo", "description_meteo",
    "couverture_nuageuse", "visibilite_m", "hauteur_vagues_m",
)

# Météos de démonstration (mêmes clés que data/meteo_astro.get_meteo_astro).
SCENARIOS: dict[str, dict[str, Any]] = {
    "calme": {"rafales_kmh": 15.0, "code_meteo": 1, "hauteur_vagues_m": 0.4, "couverture_nuageuse": 20, "risque_brouillard": 0},
    "couvert": {"rafales_kmh": 25.0, "code_meteo": 3, "hauteur_vagues_m": 0.8, "couverture_nuageuse": 100, "risque_brouillard": 10},
    "tempete": {"rafales_kmh": 85.0, "code_meteo": 63, "hauteur_vagues_m": 3.2, "couverture_nuageuse": 100, "risque_brouillard": 10},
    "orage": {"rafales_kmh": 45.0, "code_meteo": 95, "hauteur_vagues_m": 1.5, "couverture_nuageuse": 100, "risque_brouillard": 5},
    "brouillard": {"rafales_kmh": 5.0, "code_meteo": 45, "hauteur_vagues_m": 0.2, "couverture_nuageuse": 100, "risque_brouillard": 90},
}


def _meteo_live() -> Mapping[str, Any]:
    """Météo réelle Open-Meteo du port, mise en cache 10 minutes."""
    from cerveau.prediction import _meteo_live_en_cache

    return _meteo_live_en_cache()


class Noctura:
    def __init__(
        self,
        predicteur: Any = None,
        fournisseur_meteo: Callable[[], Mapping[str, Any] | None] | None = None,
        niveau_biodiversite: str | None = None,
        mode: str | None = None,
    ) -> None:
        self._meteo: Mapping[str, Any] | None = None
        self.fournisseur_meteo = fournisseur_meteo or _meteo_live
        # Le prédicteur ML lit la même météo que le moteur (celle de l'instant décidé).
        self.predicteur = predicteur or creer_predicteur(mode, fournisseur_meteo=lambda: self._meteo)
        self.niveau_biodiversite = niveau_biodiversite or niveau_sensibilite()

    @property
    def source_prediction(self) -> str:
        if isinstance(self.predicteur, PredicteurML):
            if not self.predicteur.actif:
                return "profil (repli : modèle ML indisponible)"
            return "ML DÉMO (données simulées)" if self.predicteur.simule else "ML"
        return "profil"

    def contexte(self, t: datetime, meteo: Mapping[str, Any] | None) -> dict[str, Any]:
        """Lumière naturelle à l'instant t (soleil, lune, nuages), brouillard, tempête, biodiversité."""
        from data.meteo_astro import _facteur_lumiere_naturelle, _lune_et_soleil

        meteo = meteo or {}
        try:
            astro = _lune_et_soleil(t)
        except Exception:
            astro = {}
        facteur = _facteur_lumiere_naturelle(
            astro.get("altitude_soleil"),
            astro.get("altitude_lune"),
            int(astro.get("luminosite_lune") or 0),
            meteo.get("couverture_nuageuse"),
        )
        try:
            tempete: int | None = F.calculer_is_storm(meteo)
        except F.FeatureManquante:
            tempete = None
        return {
            "facteur_lumiere_naturelle": facteur,
            "altitude_soleil": astro.get("altitude_soleil"),
            "phase_lune": astro.get("phase_lune"),
            "risque_brouillard": meteo.get("risque_brouillard"),
            "is_storm": tempete,
            "niveau_biodiversite": self.niveau_biodiversite,
            "meteo": {cle: meteo.get(cle) for cle in METEO_EXPOSEE},
        }

    def decider(self, t: datetime | None = None, meteo: Mapping[str, Any] | None = None) -> dict[str, Any]:
        t = (t or datetime.now(PARIS)).astimezone(PARIS) if (t is None or t.tzinfo) else t.replace(tzinfo=PARIS)
        self._meteo = meteo if meteo is not None else self.fournisseur_meteo()
        ctx = self.contexte(t, self._meteo)
        decisions = []
        for zone in F.ZONES:
            p50, estime = self.predicteur.indice(zone, t)
            p90, estime_p90 = self.predicteur.indice_p90(zone, t)
            decision = regles.decider(zone, p50, p90, estime or estime_p90, ctx)
            decisions.append(securite.valider(decision, ctx))
        return {
            "instant": t.isoformat(timespec="minutes"),
            "prediction": self.source_prediction,
            "contexte": ctx,
            "zones": decisions,
            "puissance_moyenne": round(sum(d.niveau for d in decisions) / len(decisions), 1),
        }

    def decider_journee(self, jour: date, meteo: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Une décision par créneau de 2 h (au milieu du créneau), de 01h00 à 23h00.
        La même météo est appliquée à toute la journée (météo live actuelle ou scénario)."""
        meteo = meteo if meteo is not None else self.fournisseur_meteo()
        return [self.decider(datetime.combine(jour, time(h + 1, 0), tzinfo=PARIS), meteo) for h in range(0, 24, 2)]


def afficher(resultat: dict[str, Any], scenario: str) -> None:
    ctx = resultat["contexte"]
    tempete = {None: "inconnue", 0: "non", 1: "oui"}[ctx["is_storm"]]
    print(f"NOCTURA · {resultat['instant']} · prédiction : {resultat['prediction']} · météo : {scenario}")
    print(f"Lumière naturelle {ctx['facteur_lumiere_naturelle']} % · lune : {ctx['phase_lune']} · "
          f"brouillard {ctx['risque_brouillard'] if ctx['risque_brouillard'] is not None else '?'} % · "
          f"tempête : {tempete} · biodiversité : {ctx['niveau_biodiversite']}")
    print()
    print(f"{'zone':20s} {'P50':>5s} {'P90':>5s} {'éclairage':>9s} {'teinte':>7s}  raisons")
    for d in resultat["zones"]:
        motifs = "; ".join(d.raisons + [f"SÉCURITÉ : {s}" for s in d.securite])
        marque = " (repli)" if d.estime else ""
        print(f"{d.zone:20s} {d.p50:5.1f} {d.p90:5.1f} {d.niveau:8d} % {d.temperature_k:6d}K  {motifs}{marque}")
    print()
    print(f"Puissance moyenne : {resultat['puissance_moyenne']} % de la puissance maximale")


def main() -> None:
    parser = argparse.ArgumentParser(description="Décision d'éclairage NOCTURA")
    parser.add_argument("--date", help="AAAA-MM-JJ (défaut : aujourd'hui)")
    parser.add_argument("--heure", help="HH:MM (défaut : maintenant)")
    parser.add_argument("--scenario", choices=["live", *SCENARIOS], default="live", help="météo live ou scénario de démo")
    parser.add_argument("--mode", choices=["profil", "ml", "demo"], help="force PREDICTION_MODE")
    parser.add_argument("--bio", choices=["inconnu", "stable", "vigilance", "alerte"], help="force le niveau de biodiversité")
    parser.add_argument("--json", action="store_true", help="sortie JSON")
    args = parser.parse_args()

    maintenant = datetime.now(PARIS)
    jour = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else maintenant.date()
    heure = datetime.strptime(args.heure, "%H:%M").time() if args.heure else maintenant.time()
    t = datetime.combine(jour, heure, tzinfo=PARIS)
    meteo = None if args.scenario == "live" else dict(SCENARIOS[args.scenario])

    moteur = Noctura(niveau_biodiversite=args.bio, mode=args.mode)
    resultat = moteur.decider(t, meteo)
    if args.json:
        sortie = dict(resultat, zones=[d.en_dict() for d in resultat["zones"]])
        print(json.dumps(sortie, ensure_ascii=False, indent=2))
    else:
        afficher(resultat, args.scenario)


if __name__ == "__main__":
    main()
