"""Indicateur de sensibilité écologique à partir des comptages annuels d'espèces.

4 années (2022-2025) ne permettent pas d'apprendre un effet : la biodiversité n'entre
PAS dans le modèle de fréquentation. On produit un indicateur DESCRIPTIF (tendance par
espèce, niveau d'alerte) que le noyau peut utiliser pour durcir la politique de
sobriété. Aucune conclusion causale : une baisse de comptage n'est pas attribuée à
l'éclairage.

Formats CSV acceptés (séparateur virgule, point-virgule ou tabulation) :
- long  : annee,espece,effectif[,taxon_indetermine]  (alias : year / species / comptage, count, nombre)
- large : espece,2022,2023,2024,2025
Source par défaut : data/biodev_1_tidy.csv (format long). Les taxons indéterminés
(taxon_indetermine = 1, ex. « Sparidé indéterminé ») sont exclus : un regroupement
d'espèces n'a pas de tendance interprétable.

Tendance : moyenne des 2 dernières années comparée à celle des 2 premières (plus robuste
qu'un simple premier/dernier point sur 4 valeurs), confirmée par le signe de la pente.
Une espèce dont l'effectif total est inférieur à MIN_EFFECTIF_TOTAL reste « inconnu » :
quelques individus de plus ou de moins ne disent rien.

    python -m ml.biodiversite [chemin_csv]
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

RACINE = Path(__file__).resolve().parent.parent
CHEMIN_CSV = RACINE / "data" / "biodev_1_tidy.csv"
CHEMIN_SORTIE = RACINE / "ml" / "sensibilite_ecologique.json"

MIN_ANNEES = 3
MIN_EFFECTIF_TOTAL = 20
SEUIL_ALERTE = -0.30      # baisse de 30 % ou plus (2 dernières années vs 2 premières)
SEUIL_VIGILANCE = -0.10   # baisse de 10 % à 30 %
ORDRE_NIVEAUX = ("inconnu", "stable", "vigilance", "alerte")

_COL_ANNEE = ("annee", "année", "year")
_COL_ESPECE = ("espece", "espèce", "species", "nom")
_COL_COMPTAGE = ("comptage", "effectif", "nombre", "count", "valeur")
_COL_INDETERMINE = ("taxon_indetermine",)


def _colonne(entetes: list[str], candidats: tuple[str, ...]) -> str | None:
    bas = {e.strip().lower(): e for e in entetes}
    return next((bas[c] for c in candidats if c in bas), None)


def charger(chemin: Path = CHEMIN_CSV, exclus: list[str] | None = None) -> dict[str, dict[int, float]]:
    """espece -> {annee: comptage}. Vide si le fichier est absent ou vide.
    Les taxons indéterminés sont ajoutés à `exclus` au lieu d'être retournés."""
    if not chemin.exists() or chemin.stat().st_size == 0:
        return {}
    with open(chemin, encoding="utf-8-sig", newline="") as f:
        texte = f.read().splitlines()
    try:
        dialecte = csv.Sniffer().sniff(texte[0], delimiters=",;\t")
    except csv.Error:
        dialecte = csv.excel
    lecteur = csv.DictReader(texte, dialect=dialecte)
    entetes = lecteur.fieldnames or []
    lignes = list(lecteur)
    series: dict[str, dict[int, float]] = defaultdict(dict)
    c_annee, c_espece, c_comptage, c_indet = (
        _colonne(entetes, c) for c in (_COL_ANNEE, _COL_ESPECE, _COL_COMPTAGE, _COL_INDETERMINE)
    )
    if c_indet and c_espece:
        indetermines = {l[c_espece].strip() for l in lignes if (l.get(c_indet) or "").strip() == "1"}
        if exclus is not None:
            exclus.extend(sorted(indetermines))
        lignes = [l for l in lignes if l[c_espece].strip() not in indetermines]
    if c_annee and c_espece and c_comptage:
        for l in lignes:
            try:
                series[l[c_espece].strip()][int(l[c_annee])] = float(l[c_comptage])
            except (ValueError, TypeError, AttributeError):
                continue
    elif c_espece:
        annees = [e for e in entetes if e.strip().isdigit()]
        for l in lignes:
            for a in annees:
                try:
                    series[l[c_espece].strip()][int(a)] = float(l[a])
                except (ValueError, TypeError):
                    continue
    return {e: dict(sorted(v.items())) for e, v in series.items() if e}


def tendance(serie: dict[int, float]) -> dict[str, Any]:
    annees = sorted(serie)
    if len(annees) < MIN_ANNEES:
        return {"n_annees": len(annees), "niveau": "inconnu", "motif": f"moins de {MIN_ANNEES} années"}
    valeurs = [serie[a] for a in annees]
    n = len(annees)
    if sum(valeurs) < MIN_EFFECTIF_TOTAL:
        return {"n_annees": n, "effectif_total": sum(valeurs), "niveau": "inconnu",
                "motif": f"effectif total < {MIN_EFFECTIF_TOTAL} individus"}
    moy_a, moy_v = sum(annees) / n, sum(valeurs) / n
    pente = sum((a - moy_a) * (v - moy_v) for a, v in zip(annees, valeurs)) / sum((a - moy_a) ** 2 for a in annees)
    debut, fin = sum(valeurs[:2]) / 2, sum(valeurs[-2:]) / 2
    variation = (fin - debut) / debut if debut > 0 else None
    # Les deux signaux doivent aller dans le même sens : sur 4 points, une seule mesure est trop fragile.
    pente_relative = pente / moy_v if moy_v > 0 else None
    if variation is None or pente_relative is None:
        niveau = "inconnu"
    elif variation <= SEUIL_ALERTE and pente < 0:
        niveau = "alerte"
    elif variation <= SEUIL_VIGILANCE and pente < 0:
        niveau = "vigilance"
    else:
        niveau = "stable"
    return {
        "n_annees": n,
        "annees": [annees[0], annees[-1]],
        "effectif_total": sum(valeurs),
        "variation_premiere_derniere": variation,
        "pente_par_an": pente,
        "pente_relative_par_an": pente_relative,
        "niveau": niveau,
    }


def sensibilite(chemin: Path = CHEMIN_CSV) -> dict[str, Any]:
    exclus: list[str] = []
    especes = {e: tendance(s) for e, s in sorted(charger(chemin, exclus).items())}
    niveaux = [t["niveau"] for t in especes.values()]
    global_ = max(niveaux, key=ORDRE_NIVEAUX.index) if niveaux else "inconnu"
    return {
        "niveau_global": global_,
        "especes_en_alerte": [e for e, t in especes.items() if t["niveau"] == "alerte"],
        "especes_en_vigilance": [e for e, t in especes.items() if t["niveau"] == "vigilance"],
        "especes": especes,
        "taxons_exclus": exclus,
        "seuils": {"alerte": SEUIL_ALERTE, "vigilance": SEUIL_VIGILANCE, "min_annees": MIN_ANNEES,
                   "min_effectif_total": MIN_EFFECTIF_TOTAL},
        "avertissement": "Tendance descriptive sur peu d'années ; aucune conclusion causale (l'éclairage n'est pas désigné comme cause).",
    }


def niveau_sensibilite(chemin: Path = CHEMIN_SORTIE) -> str:
    """Niveau global (inconnu | stable | vigilance | alerte) pour le noyau. Jamais d'exception."""
    try:
        return json.loads(chemin.read_text(encoding="utf-8"))["niveau_global"]
    except Exception:
        return "inconnu"


def main(chemin: Path = CHEMIN_CSV, sortie: Path = CHEMIN_SORTIE) -> dict[str, Any]:
    resultat = sensibilite(chemin)
    sortie.write_text(json.dumps(resultat, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sensibilité écologique : {resultat['niveau_global']} ({len(resultat['especes'])} espèces) -> {sortie}")
    return resultat


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CHEMIN_CSV)
