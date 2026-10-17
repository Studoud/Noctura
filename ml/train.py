"""Entraînement du modèle de fréquentation NOCTURA.

    python -m ml.train [chemin_csv]          # données réelles -> ml/model.joblib
    python -m ml.train --demo                 # données SIMULÉES -> ml/model_demo.joblib

Étapes : chargement, contrôles qualité, vérification de la fuite d'indice_global,
features (ml/features.py), validation par groupes, comparaison baseline / GLM Poisson /
LightGBM Poisson, P90 conformal, critère d'acceptation, entraînement final, écriture de
ml/model.joblib, ml/model_card.md et ml/rapport_evaluation.md.

Choix de validation
-------------------
- Le dataset contient une colonne `date` (plusieurs journées) : mode "jours". Le test réel
  est formé des dernières journées réelles, jamais vues ; la calibration se fait par
  GroupKFold par journée sur le reste.
- Sinon (une seule journée, cas actuel) : mode "creneaux". Chaque créneau est retiré à
  tour de rôle (leave-one-créneau-out) ; les 8 zones d'un même créneau, quasi dupliquées,
  ne sont jamais à la fois en entraînement et en test.
Dans les deux modes, le score de référence est calculé sur les lignes RÉELLES du test ;
le synthétique est rapporté à part. Avec --demo seulement, la population évaluée est la
population SIMULÉE (ml/simulation.py) et le modèle produit est marqué donnees_simulees.
"""
from __future__ import annotations

import csv
import hashlib
import math
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np

RACINE = Path(__file__).resolve().parent.parent
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from cerveau.prediction import interpolation_circulaire  # noqa: E402
from ml import features as F  # noqa: E402

SEED = 42
VERSION_MODELE = "1.0.0"
DATASET = RACINE / "data" / "dataset_ml_frequentation_24h_clean.csv"
DATASET_DEMO = RACINE / "data" / "dataset_simule.csv"
DOSSIER = RACINE / "ml"

MIN_JOURS_EVALUES = 5        # en dessous, aucune métrique n'est assez stable pour activer le ML
GAIN_MAE_MIN = 0.10          # le modèle doit battre la baseline d'au moins 10 % de MAE (test réel)
COUVERTURE_P90 = (0.85, 0.95)
NIVEAU_P90 = 0.90
BORNES = (0.0, 100.0)
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 30
MAX_LIGNES_PERMUTATION = 20000

COLONNES = (
    "creneau", "hour_start", "hour_end", "sin_hour", "cos_hour", "zone_group_id",
    "zone_group_name", "zone_specifique", "is_daylight", "is_winter", "is_storm",
    "is_synthetic_daytime", "indice_global", "frequentation_target",
)


# =========================================================================== données

def _bool(texte: str) -> bool:
    valeur = texte.strip().lower()
    if valeur in ("true", "1"):
        return True
    if valeur in ("false", "0"):
        return False
    raise ValueError(f"booléen invalide : {texte!r}")


def charger(chemin: Path, demo: bool = False) -> tuple[list[dict[str, str]], list[dict[str, Any]], int]:
    """Lignes typées. nature : reel | synthetique | simule (colonne `source` si présente).
    `evalue` marque la population du score : réelle, ou simulée en mode démo."""
    with open(chemin, encoding="utf-8", newline="") as f:
        brut = list(csv.DictReader(f))
    lignes, rejetees = [], 0
    for r in brut:
        try:
            synthetique = _bool(r["is_synthetic_daytime"])
            source = (r.get("source") or "").strip().lower()
            nature = source if source in ("reel", "synthetique", "simule") else ("synthetique" if synthetique else "reel")
            texte_date = (r.get("date") or "").strip() or None
            lignes.append({
                "date": texte_date,
                "date_iso": bool(texte_date),
                "jour": date.fromisoformat(texte_date) if texte_date else None,
                "date_inferee": False,
                "nature": nature,
                "evalue": nature == ("simule" if demo else "reel"),
                "creneau": r["creneau"],
                "hour_start": int(r["hour_start"]),
                "zone": r["zone_specifique"],
                "groupe": r["zone_group_id"],
                "is_daylight": int(r["is_daylight"]),
                "is_winter": int(r["is_winter"]),
                "is_storm": int(r["is_storm"]),
                "synthetique": nature != "reel",
                "indice_global": float(r["indice_global"]),
                "y": float(r["frequentation_target"]),
            })
        except (KeyError, ValueError, TypeError):
            rejetees += 1
    if any(l["date"] is None for l in lignes):
        # Pas de colonne date : une journée = une suite de créneaux de 00h à 22h ; un retour
        # en arrière de l'heure marque le début de la journée suivante (libellés J001, J002...).
        numero, precedente = 1, -1
        for l in lignes:
            if l["hour_start"] < precedente:
                numero += 1
            precedente = l["hour_start"]
            l["date"] = f"J{numero:03d}"
        if numero == 1:  # une seule journée : pas de groupe par jour
            for l in lignes:
                l["date"] = None
    if any(l["jour"] is None for l in lignes):
        # Date calendaire inconnue (sert seulement aux features saison / week-end) :
        # journée unique -> milieu de la plage compatible avec is_daylight ; sinon date fixe.
        compat = verifier_regle_daylight(lignes) if mode_validation(lignes) == "creneaux" else {}
        if compat.get("premiere"):
            debut, fin = date.fromisoformat(compat["premiere"]), date.fromisoformat(compat["derniere"])
            reference = debut + (fin - debut) / 2
        else:
            reference = date(2025, 6, 25)
        for l in lignes:
            if l["jour"] is None:
                l["jour"], l["date_inferee"] = reference, True
    return brut, lignes, rejetees


def mode_validation(lignes: list[dict[str, Any]]) -> str:
    return "jours" if len({l["date"] for l in lignes if l["date"]}) >= 3 else "creneaux"


def groupe(l: dict[str, Any], mode: str) -> str:
    return l["date"] if mode == "jours" else l["creneau"]


def nb_jours_evalues(lignes: list[dict[str, Any]]) -> int:
    dates = {l["date"] for l in lignes if l["evalue"]}
    if not dates:
        return 0
    return len(dates - {None}) or 1


def ajouter_indice_decale(lignes: list[dict[str, Any]]) -> bool:
    """indice_global du créneau précédent (même journée ou veille). Seulement en multi-journées."""
    if mode_validation(lignes) != "jours":
        return False
    table = {(l["date"], l["hour_start"]): l["indice_global"] for l in lignes}
    ordre = list(dict.fromkeys(sorted(l["date"] for l in lignes)))
    veille = {d: (ordre[i - 1] if i else None) for i, d in enumerate(ordre)}
    for l in lignes:
        if l["hour_start"] >= 2:
            cle = (l["date"], l["hour_start"] - 2)
        else:
            cle = (veille[l["date"]], 22)
        l["indice_decale"] = table.get(cle)
    return True


# =========================================================================== contrôles

def _stats(valeurs: Iterable[float]) -> dict[str, float]:
    v = np.asarray(list(valeurs), dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "moyenne": float(v.mean()), "ecart_type": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}


def controles_qualite(brut: list[dict[str, str]], lignes: list[dict[str, Any]], rejetees: int) -> dict[str, Any]:
    colonnes = list(brut[0].keys()) if brut else []
    manquantes = {c: sum(1 for r in brut if (r.get(c) or "").strip() == "") for c in colonnes}
    cle = Counter((l["date"], l["creneau"], l["zone"]) for l in lignes)
    constantes = [c for c in colonnes if len({r[c] for r in brut}) == 1]
    reel = [l for l in lignes if l["nature"] == "reel"]
    synth = [l for l in lignes if l["nature"] == "synthetique"]
    simu = [l for l in lignes if l["nature"] == "simule"]

    def part_zone(sous: list[dict[str, Any]]) -> dict[str, float]:
        moy = defaultdict(list)
        for l in sous:
            moy[(l["date"], l["creneau"])].append(l["y"])
        rapports = defaultdict(list)
        for l in sous:
            m = np.mean(moy[(l["date"], l["creneau"])])
            if m > 0:
                rapports[l["zone"]].append(l["y"] / m)
        return {z: float(np.mean(v)) for z, v in sorted(rapports.items())}

    return {
        "n_lignes": len(brut),
        "n_lignes_valides": len(lignes),
        "n_lignes_rejetees": rejetees,
        "colonnes_absentes": [c for c in COLONNES if c not in colonnes],
        "valeurs_manquantes": {c: n for c, n in manquantes.items() if n},
        "doublons_exacts": len(brut) - len({tuple(r.values()) for r in brut}),
        "doublons_cle": sum(n - 1 for n in cle.values() if n > 1),
        "colonnes_constantes": constantes,
        "n_reelles": len(reel),
        "n_synthetiques": len(synth),
        "n_simulees": len(simu),
        "date_inferee": any(l["date_inferee"] for l in lignes),
        "jours_inferes": not all(l["date_iso"] for l in lignes) and len({l["date"] for l in lignes}) > 1,
        "n_jours_total": len({l["date"] for l in lignes}),
        "creneaux_reels": sorted({l["creneau"] for l in reel}),
        "creneaux_synthetiques": sorted({l["creneau"] for l in synth}),
        "cible_reel": _stats(l["y"] for l in reel),
        "cible_synthetique": _stats(l["y"] for l in synth),
        "part_zone_reel": part_zone(reel),
        "part_zone_synthetique": part_zone(synth),
        "n_jours_evalues": nb_jours_evalues(lignes),
        "domaine": {k: sorted({l[k] for l in lignes}) for k in ("is_daylight", "is_winter", "is_storm")},
    }


def verifier_fuite(lignes: list[dict[str, Any]]) -> dict[str, Any]:
    """indice_global est-il l'agrégat des cibles du même créneau ?"""
    par_creneau = defaultdict(list)
    indices = defaultdict(set)
    for l in lignes:
        par_creneau[(l["date"], l["creneau"])].append(l["y"])
        indices[(l["date"], l["creneau"])].add(l["indice_global"])
    cles = sorted(par_creneau)
    ig = np.array([next(iter(indices[k])) for k in cles])
    moy = np.array([np.mean(par_creneau[k]) for k in cles])
    corr = float(np.corrcoef(ig, moy)[0, 1])
    pente, ordonnee = np.polyfit(moy, ig, 1)
    residu = float(np.abs(ig - (pente * moy + ordonnee)).max())
    rapports = defaultdict(list)
    for l in lignes:
        if l["indice_global"] > 0:
            rapports[l["zone"]].append(l["y"] / l["indice_global"])
    return {
        "unique_par_creneau": all(len(v) == 1 for v in indices.values()),
        "correlation_avec_moyenne_cibles": corr,
        "reconstruction": {"pente": float(pente), "ordonnee": float(ordonnee), "residu_max": residu},
        "rapport_cible_sur_indice_par_zone": {
            z: {"moyenne": float(np.mean(v)), "ecart_type": float(np.std(v, ddof=1))} for z, v in sorted(rapports.items())
        },
        "fuite": corr > 0.95,
    }


def verifier_regle_daylight(lignes: list[dict[str, Any]], annee: int = 2025) -> dict[str, Any]:
    """La règle is_daylight de features.py reproduit-elle le dataset ? (cohérence entraînement / service)"""
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(F.FUSEAU)
    if mode_validation(lignes) == "jours" and not all(l["date_iso"] for l in lignes):
        return {"mode": "jours_sans_date", "n_jours": len({l["date"] for l in lignes})}
    if mode_validation(lignes) == "jours":
        ecarts = sum(
            F.calculer_is_daylight(datetime.fromisoformat(l["date"]).replace(hour=l["hour_start"], tzinfo=tz))
            != l["is_daylight"]
            for l in lignes
        )
        return {"mode": "par_ligne", "taux_ecart": ecarts / len(lignes)}
    attendu = {l["hour_start"]: l["is_daylight"] for l in lignes}
    compatibles = []
    jour = date(annee, 1, 1)
    while jour.year == annee:
        if all(
            F.calculer_is_daylight(datetime(jour.year, jour.month, jour.day, h, tzinfo=tz)) == v
            for h, v in attendu.items()
        ):
            compatibles.append(jour)
        jour += timedelta(days=1)
    return {
        "mode": "date_inconnue",
        "n_dates_compatibles": len(compatibles),
        "premiere": compatibles[0].isoformat() if compatibles else None,
        "derniere": compatibles[-1].isoformat() if compatibles else None,
    }


# =========================================================================== modèles

def matrice(lignes: list[dict[str, Any]], extra: str | None = None) -> np.ndarray:
    X = []
    for l in lignes:
        x = F.vecteur(l["zone"], l["hour_start"], l["is_daylight"], l["is_winter"], l["is_storm"], jour=l["jour"])
        if extra:
            x.append(math.log1p(l[extra]))
        X.append(x)
    return np.asarray(X, dtype=float)


def cible(lignes: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([l["y"] for l in lignes], dtype=float)


class Baseline:
    """Moyenne zone × créneau ; créneau jamais vu : interpolation circulaire entre voisins."""

    nom = "baseline_zone_x_creneau"
    servable = True

    def fit(self, lignes: list[dict[str, Any]]) -> "Baseline":
        acc = defaultdict(list)
        for l in lignes:
            acc[(l["zone"], l["hour_start"])].append(l["y"])
        self.table = {k: float(np.mean(v)) for k, v in acc.items()}
        self.moyenne = float(np.mean([l["y"] for l in lignes]))
        return self

    def _valeur(self, zone: str, heure: int) -> float:
        if (zone, heure) in self.table:
            return self.table[(zone, heure)]
        points = sorted((h, v) for (z, h), v in self.table.items() if z == zone)
        if not points:
            return self.moyenne
        return interpolation_circulaire(heure, [h for h, _ in points], [v for _, v in points])

    def predict(self, lignes: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([self._valeur(l["zone"], l["hour_start"]) for l in lignes])


class GLMPoisson:
    nom = "glm_poisson"

    def __init__(self, alpha: float = 1e-3, extra: str | None = None) -> None:
        self.alpha, self.extra = alpha, extra
        self.servable = extra is None

    def fit(self, lignes: list[dict[str, Any]]) -> "GLMPoisson":
        from sklearn.linear_model import PoissonRegressor

        self.estimateur = PoissonRegressor(alpha=self.alpha, max_iter=5000).fit(matrice(lignes, self.extra), cible(lignes))
        return self

    def predict(self, lignes: list[dict[str, Any]]) -> np.ndarray:
        return self.estimateur.predict(matrice(lignes, self.extra))


class LGBMPoisson:
    nom = "lightgbm_poisson"

    def __init__(self, n_estimators: int = 200, extra: str | None = None) -> None:
        self.n_estimators, self.extra = n_estimators, extra
        self.servable = extra is None

    def fit(self, lignes: list[dict[str, Any]]) -> "LGBMPoisson":
        import lightgbm as lgb

        self.estimateur = lgb.LGBMRegressor(
            objective="poisson", n_estimators=self.n_estimators, learning_rate=0.05,
            max_depth=2, num_leaves=4, min_child_samples=5, reg_lambda=1.0,
            random_state=SEED, deterministic=True, force_col_wise=True, n_jobs=1, verbose=-1,
        ).fit(matrice(lignes, self.extra), cible(lignes))
        return self

    def predict(self, lignes: list[dict[str, Any]]) -> np.ndarray:
        return self.estimateur.predict(matrice(lignes, self.extra))


# Candidats dans l'ordre de simplicité ; chaque candidat a une petite grille d'hyperparamètres.
def candidats(avec_decale: bool) -> list[tuple[str, Callable[..., Any], list[dict[str, Any]]]]:
    liste = [
        ("baseline", Baseline, [{}]),
        ("glm_poisson", GLMPoisson, [{"alpha": a} for a in (1e-4, 1e-3, 1e-2, 1e-1)]),
        ("lightgbm_poisson", LGBMPoisson, [{"n_estimators": n} for n in (100, 300)]),
        ("glm_poisson+indice_global (FUITE)", GLMPoisson, [{"alpha": 1e-3, "extra": "indice_global"}]),
        ("lightgbm_poisson+indice_global (FUITE)", LGBMPoisson, [{"n_estimators": 300, "extra": "indice_global"}]),
    ]
    if avec_decale:
        liste += [
            ("glm_poisson+indice_decale", GLMPoisson, [{"alpha": 1e-3, "extra": "indice_decale"}]),
            ("lightgbm_poisson+indice_decale", LGBMPoisson, [{"n_estimators": 300, "extra": "indice_decale"}]),
        ]
    return liste


# =========================================================================== validation

def _plis(lignes: list[dict[str, Any]], mode: str, k_max: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    groupes = [groupe(l, mode) for l in lignes]
    k = min(k_max, len(set(groupes)))
    if k < 2:
        return []
    return list(GroupKFold(n_splits=k).split(np.zeros(len(lignes)), groups=groupes))


def quantile_conformal(residus: np.ndarray, niveau: float = NIVEAU_P90) -> float:
    n = residus.size
    if n == 0:
        return 1.2816  # repli : quantile 90 % d'une loi normale réduite
    rang = min(1.0, math.ceil((n + 1) * niveau) / n)
    return float(np.quantile(residus, rang, method="higher"))


def residus_normalises(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (y - mu) / np.sqrt(np.maximum(mu, 1.0))


def ajuster_et_calibrer(fabrique: Callable[..., Any], grille: list[dict[str, Any]], train: list[dict[str, Any]], mode: str):
    """Choix des hyperparamètres et quantile conformal par validation croisée interne groupée."""
    y = cible(train)
    plis = _plis(train, mode)
    meilleur, meilleur_mae, meilleur_oof = grille[0], math.inf, None
    for params in grille:
        if not plis:
            break
        oof = np.zeros(len(train))
        for idx_a, idx_v in plis:
            modele = fabrique(**params).fit([train[i] for i in idx_a])
            oof[idx_v] = np.clip(modele.predict([train[i] for i in idx_v]), *BORNES)
        mae = float(np.mean(np.abs(y - oof)))
        if mae < meilleur_mae:
            meilleur, meilleur_mae, meilleur_oof = params, mae, oof
    q = quantile_conformal(residus_normalises(y, meilleur_oof)) if meilleur_oof is not None else quantile_conformal(np.array([]))
    return fabrique(**meilleur).fit(train), q, meilleur


def predire(modele: Any, q: float, lignes: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    mu = np.clip(modele.predict(lignes), *BORNES)
    p90 = np.minimum(np.maximum(mu + q * np.sqrt(np.maximum(mu, 1.0)), mu), BORNES[1])
    return mu, p90


def decoupages_externes(lignes: list[dict[str, Any]], mode: str) -> list[tuple[list[int], list[int]]]:
    if mode == "jours":
        dates_reelles = sorted({l["date"] for l in lignes if l["evalue"]})
        n_test = max(1, round(0.2 * len(dates_reelles)))
        test = set(dates_reelles[-n_test:])
        return [([i for i, l in enumerate(lignes) if l["date"] not in test],
                 [i for i, l in enumerate(lignes) if l["date"] in test])]
    creneaux = sorted({l["creneau"] for l in lignes})
    return [([i for i, l in enumerate(lignes) if l["creneau"] != c],
             [i for i, l in enumerate(lignes) if l["creneau"] == c]) for c in creneaux]


def evaluer(lignes: list[dict[str, Any]], mode: str, avec_decale: bool) -> tuple[list[int], dict[str, dict[str, Any]]]:
    """Prédictions hors échantillon (P50 et P90) de chaque candidat sur les découpages externes."""
    decoupages = decoupages_externes(lignes, mode)
    ordre = [i for _, test in decoupages for i in test]
    resultats = {}
    for nom, fabrique, grille in candidats(avec_decale):
        extra = grille[0].get("extra")
        utilisable = [extra is None or l.get(extra) is not None for l in lignes]
        mu = np.full(len(lignes), np.nan)
        p90 = np.full(len(lignes), np.nan)
        params_vus = []
        for idx_train, idx_test in decoupages:
            train = [lignes[i] for i in idx_train if utilisable[i]]
            test_idx = [i for i in idx_test if utilisable[i]]
            modele, q, params = ajuster_et_calibrer(fabrique, grille, train, mode)
            params_vus.append(params)
            m, h = predire(modele, q, [lignes[i] for i in test_idx])
            mu[test_idx], p90[test_idx] = m, h
        resultats[nom] = {"mu": mu, "p90": p90, "servable": extra is None, "params": params_vus}
    return ordre, resultats


# =========================================================================== métriques

def metriques(y: np.ndarray, mu: np.ndarray, p90: np.ndarray) -> dict[str, float]:
    ok = ~np.isnan(mu)
    y, mu, p90 = y[ok], mu[ok], p90[ok]
    if y.size == 0:
        return {"n": 0}
    return {
        "n": int(y.size),
        "mae": float(np.mean(np.abs(y - mu))),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "couverture_p90": float(np.mean(y <= p90)),
        "taux_sous_estimation_p50": float(np.mean(y > mu)),
        "largeur_moyenne_p90": float(np.mean(p90 - mu)),
    }


def segments(lignes: list[dict[str, Any]], ordre: list[int]) -> dict[str, list[int]]:
    seg: dict[str, list[int]] = defaultdict(list)
    for i in ordre:
        l = lignes[i]
        nature = l["nature"]
        moment = "jour" if l["is_daylight"] else "nuit"
        seg[nature].append(i)
        seg[f"{nature} · {moment}"].append(i)
        seg[f"tout · {moment}"].append(i)
        if l["evalue"]:
            seg[f"{nature} · zone {l['zone']}"].append(i)
    return dict(seg)


def gain_bootstrap(lignes, idx_reels, y, mu_modele, mu_base, mode) -> dict[str, float]:
    """IC 95 % du gain relatif de MAE, rééchantillonnage des groupes indépendants (créneaux ou jours)."""
    par_groupe = defaultdict(list)
    for i in idx_reels:
        par_groupe[groupe(lignes[i], mode)].append(i)
    cles = sorted(par_groupe)
    erreur_m = np.array([np.abs(y[par_groupe[c]] - mu_modele[par_groupe[c]]).sum() for c in cles])
    erreur_b = np.array([np.abs(y[par_groupe[c]] - mu_base[par_groupe[c]]).sum() for c in cles])
    tirages = np.random.default_rng(SEED).integers(0, len(cles), size=(N_BOOTSTRAP, len(cles)))
    somme_m, somme_b = erreur_m[tirages].sum(axis=1), erreur_b[tirages].sum(axis=1)
    gains = 1.0 - somme_m[somme_b > 0] / somme_b[somme_b > 0]
    return {"bas": float(np.percentile(gains, 2.5)), "haut": float(np.percentile(gains, 97.5)), "n_groupes": len(cles)}


def importance_permutation(modele: Any, lignes: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Hausse de MAE quand on permute un groupe de colonnes (sur la population évaluée, en échantillon)."""
    rng = np.random.default_rng(SEED)
    if len(lignes) > MAX_LIGNES_PERMUTATION:
        lignes = [lignes[i] for i in rng.choice(len(lignes), MAX_LIGNES_PERMUTATION, replace=False)]
    X = matrice(lignes)
    y = cible(lignes)
    estimateur = modele.estimateur
    reference = np.mean(np.abs(y - np.clip(estimateur.predict(X), *BORNES)))
    resultat = {}
    for nom, colonnes in F.GROUPES_FEATURES.items():
        idx = [F.NOMS_FEATURES.index(c) for c in colonnes]
        hausses = []
        for _ in range(N_PERMUTATIONS):
            Xp = X.copy()
            Xp[:, idx] = X[rng.permutation(len(X))][:, idx]
            hausses.append(np.mean(np.abs(y - np.clip(estimateur.predict(Xp), *BORNES))) - reference)
        resultat[nom] = {"moyenne": float(np.mean(hausses)), "ecart_type": float(np.std(hausses))}
    return resultat


# =========================================================================== rapports

def _f(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _pct(x: Any) -> str:
    return "—" if x is None else f"{100 * x:.1f} %"


def ecrire_rapport(chemin: Path, ctx: dict[str, Any]) -> None:
    qc, fuite, tab, dec, pop, lib = ctx["qc"], ctx["fuite"], ctx["tableaux"], ctx["decision"], ctx["pop"], ctx["lib"]
    noms = list(tab)
    autre = "synthetique" if pop == "reel" else "reel"
    L = [f"# Rapport d'évaluation : fréquentation NOCTURA{' (DÉMO, DONNÉES SIMULÉES)' if ctx['demo'] else ''}", ""]
    if ctx["demo"]:
        L += [f"> **Données simulées par `ml/simulation.py`.** Ces scores mesurent la capacité du modèle à retrouver le générateur, "
              "pas le comportement réel du port. Le modèle produit est réservé à `PREDICTION_MODE=demo`.", ""]
    L += [
        f"Généré par `ml/train.py` le {ctx['date']} (graine {SEED}). Tous les chiffres ci-dessous sont calculés par le script ; aucun n'est saisi à la main.",
        "",
        "## Protocole",
        "",
        f"- Dataset : `{ctx['dataset']}` (sha256 `{ctx['sha256'][:12]}…`), {qc['n_lignes']} lignes : "
        f"{qc['n_reelles']} réelles, {qc['n_synthetiques']} synthétiques, {qc['n_simulees']} simulées.",
        f"- Population évaluée : lignes **{lib}** ({qc['n_jours_evalues']} journée(s)).",
        f"- Mode de validation : **{ctx['mode']}**. " + (
            f"Chaque créneau est retiré à tour de rôle (leave-one-créneau-out) : chaque ligne est prédite par un modèle qui n'a jamais vu son créneau. Le test {lib} est l'union des créneaux {lib}."
            if ctx["mode"] == "creneaux" else
            f"Les {ctx['n_jours_test']} dernières journées ({ctx['periode_test']}) forment le test, jamais vues ; calibration par GroupKFold par journée sur les précédentes."
        ),
        "- Hyperparamètres et quantile P90 (conformal sur résidus normalisés `(y-μ)/√μ`) choisis par validation croisée interne groupée, dans chaque pli externe.",
        "- `indice_global` est exclu des modèles servables (fuite, voir plus bas) ; les variantes « (FUITE) » ne sont là que pour mesurer son effet.",
        "",
        "## Comparaison des modèles (test hors échantillon)",
        "",
        f"| Modèle | Servable | MAE {lib} | RMSE {lib} | Couv. P90 {lib} | MAE {autre} | RMSE {autre} | Gain MAE vs baseline |",
        "|---|---|---|---|---|---|---|---|",
    ]
    base_mae = tab["baseline"][pop]["mae"]
    for nom in noms:
        r, s = tab[nom][pop], tab[nom].get(autre, {})
        gain = 1 - r["mae"] / base_mae if base_mae else None
        L.append(f"| {nom} | {'oui' if ctx['resultats'][nom]['servable'] else 'non'} | {_f(r.get('mae'))} | {_f(r.get('rmse'))} | {_pct(r.get('couverture_p90'))} | {_f(s.get('mae'))} | {_f(s.get('rmse'))} | {'—' if nom == 'baseline' else _pct(gain)} |")
    L += ["", f"IC 95 % du gain de MAE {lib} (bootstrap sur les groupes indépendants) :", ""]
    for nom, ic in ctx["ic"].items():
        L.append(f"- {nom} : [{_pct(ic['bas'])} ; {_pct(ic['haut'])}] sur {ic['n_groupes']} groupes")
    L += ["", "## Détail par segment", ""]
    seg_noms = sorted({s for nom in noms for s in tab[nom]})
    for nom in noms:
        L += [f"### {nom}", "", "| Segment | n | MAE | RMSE | Couv. P90 | Sous-estimation P50 |", "|---|---|---|---|---|---|"]
        for s in seg_noms:
            m = tab[nom].get(s)
            if m and m.get("n"):
                L.append(f"| {s} | {m['n']} | {_f(m['mae'])} | {_f(m['rmse'])} | {_pct(m['couverture_p90'])} | {_pct(m['taux_sous_estimation_p50'])} |")
        L.append("")
    n_pop = len(ctx["idx_pop"])
    L += [
        "## Fuite de `indice_global`",
        "",
        f"- Une seule valeur par créneau : {fuite['unique_par_creneau']}.",
        f"- Corrélation avec la moyenne des cibles du même créneau : **r = {fuite['correlation_avec_moyenne_cibles']:.4f}**.",
        f"- Reconstruction : `indice_global ≈ {fuite['reconstruction']['pente']:.3f} × moyenne(cible) {fuite['reconstruction']['ordonnee']:+.3f}`, résidu max {fuite['reconstruction']['residu_max']:.2f}.",
        f"- Verdict : {'**fuite confirmée**, variable exclue du modèle servi' if fuite['fuite'] else 'pas de fuite détectée'}.",
        "- Version décalée (créneau précédent) : " + ("comparée ci-dessus." if ctx["avec_decale"] else "non calculable sur une seule journée (le décalage de 00h-02h retomberait sur le 22h-00h du même jour)."),
        "",
        "## Couverture du P90",
        "",
        f"Cible : entre {_pct(COUVERTURE_P90[0])} et {_pct(COUVERTURE_P90[1])} sur le test {lib}. Voir la colonne « Couv. P90 {lib} ». "
        f"Sur {n_pop} points testés, chaque point couvert ou non déplace la couverture de {100 / max(1, n_pop):.2f} points.",
        "",
        "## Importance des features",
        "",
    ]
    if ctx["importance"]:
        L += [f"Modèle `{ctx['candidat']}` final, permutation par groupe de colonnes ({N_PERMUTATIONS} répétitions), "
              f"hausse de MAE sur les lignes {lib}. **Calculé en échantillon** (le modèle a vu ces lignes) : indicatif.", "",
              "| Groupe | Hausse de MAE | Écart-type |", "|---|---|---|"]
        for g, v in sorted(ctx["importance"].items(), key=lambda kv: -kv[1]["moyenne"]):
            L.append(f"| {g} | {_f(v['moyenne'], 3)} | {_f(v['ecart_type'], 3)} |")
        constants = [k for k, v in qc["domaine"].items() if len(v) < 2]
        if constants:
            L += ["", f"{', '.join(f'`{c}`' for c in constants)} constant(s) dans le dataset ({qc['domaine']}) : une importance nulle signifie « jamais observé », pas « sans effet »."]
    L += ["", "## Décision", "", f"**{'Modèle accepté' if dec['accepte'] else 'Modèle refusé : le profil reste actif'}** (candidat évalué : `{ctx['candidat']}`).", ""]
    L += [f"- {r}" for r in dec["raisons"]]
    chemin.write_text("\n".join(L) + "\n", encoding="utf-8")


def ecrire_model_card(chemin: Path, ctx: dict[str, Any]) -> None:
    qc, dec, pop, lib = ctx["qc"], ctx["decision"], ctx["pop"], ctx["lib"]
    r = ctx["tableaux"][ctx["candidat"]][pop]
    b = ctx["tableaux"]["baseline"][pop]
    dl = ctx["daylight"]
    if ctx["demo"]:
        statut = ("ACCEPTÉ SUR DONNÉES SIMULÉES : activable uniquement avec PREDICTION_MODE=demo (refusé par PREDICTION_MODE=ml)"
                  if dec["accepte"] else "REFUSÉ, même sur données simulées")
    else:
        statut = "ACCEPTÉ, activable avec PREDICTION_MODE=ml" if dec["accepte"] else "REFUSÉ : le prédicteur ML se replie sur le profil"
    constants = [k for k, v in qc["domaine"].items() if len(v) < 2]
    L = [f"# Fiche modèle : fréquentation NOCTURA v{VERSION_MODELE}{' (DÉMO)' if ctx['demo'] else ''}", ""]
    if ctx["demo"]:
        L += ["> **Modèle de démonstration entraîné sur des données SIMULÉES** (`ml/simulation.py`). Il apprend le générateur, "
              "pas le vrai port. Ne jamais l'utiliser pour piloter l'éclairage réel.", ""]
    L += [
        f"- **Date d'entraînement** : {ctx['date']}",
        f"- **Version** : {VERSION_MODELE} ; features `{F.SEUILS_VERSION}` ({len(F.NOMS_FEATURES)} colonnes) ; graine {SEED}",
        f"- **Modèle** : `{ctx['candidat']}` (paramètres {ctx['params_final']}), P90 conformal (q = {ctx['q_final']:.3f} sur résidus normalisés)",
        f"- **Statut** : {statut}",
        "",
        "## Usage prévu",
        "",
        "Prédire la **demande** (fréquentation 0-100) par zone et par créneau de 2 h. Le modèle ne décide jamais de l'éclairage : les règles (`cerveau/regles.py`) et le validateur de sécurité (`cerveau/securite.py`) restent les seuls décideurs. Utiliser `indice_p90` pour toute décision où sous-estimer la fréquentation éteindrait la lumière sur des personnes.",
        "",
        "## Données",
        "",
        f"- `{ctx['dataset']}` (sha256 `{ctx['sha256']}`)",
        f"- {qc['n_lignes']} lignes : {qc['n_reelles']} réelles, {qc['n_synthetiques']} synthétiques, {qc['n_simulees']} simulées "
        f"(**{100 * (qc['n_lignes'] - qc['n_reelles']) / max(1, qc['n_lignes']):.0f} % non réel**)",
        f"- Journées évaluées ({lib}) : {qc['n_jours_evalues']} (minimum requis : {MIN_JOURS_EVALUES})",
        f"- Domaine observé : {qc['domaine']}",
        *(["- Pas de colonne `date` : journées reconstituées par la suite des créneaux "
           f"({qc['n_jours_total']} journées) ; date calendaire inconnue, donc les features saison / week-end sont constantes à l'entraînement et sans effet."]
          if qc["jours_inferes"] else
          ["- Date absente du CSV : date de référence déduite de `is_daylight` (milieu de la plage compatible)."] if qc["date_inferee"] else []),
        "- Variable exclue : `indice_global` (agrégat des cibles du même créneau, fuite confirmée).",
        "",
        f"## Métriques (test {lib} hors échantillon)",
        "",
        "| | MAE | RMSE | Couverture P90 |",
        "|---|---|---|---|",
        f"| Baseline zone × créneau | {_f(b['mae'])} | {_f(b['rmse'])} | {_pct(b['couverture_p90'])} |",
        f"| {ctx['candidat']} | {_f(r['mae'])} | {_f(r['rmse'])} | {_pct(r['couverture_p90'])} |",
        "",
        f"Détail : `ml/{ctx['nom_rapport']}`.",
        "",
        "## Décision",
        "",
        *[f"- {x}" for x in dec["raisons"]],
        "",
        "## Limites",
        "",
        "- **Météo** : Open-Meteo aux coordonnées du port de Porquerolles ; sortie de modèle numérique (maille de quelques km), pas une mesure sur place.",
        f"- **Seuils `is_storm` / `is_winter` hypothétiques** ({F.SEUILS_VERSION}) : ceux du générateur du dataset d'origine ne sont pas connus.",
        *([f"- {', '.join(f'`{c}`' for c in constants)} constant(s) dans les données : aucun effet appris pour ces situations."] if constants else []),
        f"- **`is_daylight`** : règle « soleil au-dessus de l'horizon au milieu du créneau ». "
        + (f"Compatible avec le dataset pour {dl['n_dates_compatibles']} dates de l'année (du {dl['premiere']} au {dl['derniere']})."
           if dl.get("mode") == "date_inconnue" else
           "Non vérifiable ligne à ligne (pas de date calendaire dans le dataset)." if dl.get("mode") == "jours_sans_date" else
           f"Taux d'écart avec le dataset : {_pct(dl.get('taux_ecart'))}."),
        "- La biodiversité n'est pas une feature (4 années ne permettent pas d'apprendre un effet) : voir `ml/biodiversite.py`.",
        "",
        "## Réentraîner",
        "",
        ("`python -m ml.simulation` puis `python -m ml.train --demo`." if ctx["demo"] else
         "Ajouter une colonne `date` (AAAA-MM-JJ) et les nouvelles journées réelles au CSV, puis `python -m ml.train`. "
         "Le script passe automatiquement en validation par journées et réévalue le critère d'acceptation."),
    ]
    chemin.write_text("\n".join(L) + "\n", encoding="utf-8")


# =========================================================================== principal

def main(chemin: Path | None = None, dossier: Path = DOSSIER, demo: bool = False) -> dict[str, Any]:
    np.random.seed(SEED)
    chemin = Path(chemin or (DATASET_DEMO if demo else DATASET))
    suffixe = "_demo" if demo else ""
    pop, lib = ("simule", "SIMULÉ") if demo else ("reel", "réel")

    brut, lignes, rejetees = charger(chemin, demo=demo)
    qc = controles_qualite(brut, lignes, rejetees)
    if not any(l["evalue"] for l in lignes):
        raise SystemExit(f"Aucune ligne {lib} à évaluer dans {chemin}" + ("" if demo else " (données simulées : utiliser --demo)"))
    fuite = verifier_fuite(lignes)
    daylight = verifier_regle_daylight(lignes)
    mode = mode_validation(lignes)
    avec_decale = ajouter_indice_decale(lignes)

    ordre, resultats = evaluer(lignes, mode, avec_decale)
    y = cible(lignes)
    seg = segments(lignes, ordre)
    tableaux = {nom: {s: metriques(y[idx], r["mu"][idx], r["p90"][idx]) for s, idx in seg.items()} for nom, r in resultats.items()}

    idx_pop = seg.get(pop, [])
    base = tableaux["baseline"][pop]
    ic = {nom: gain_bootstrap(lignes, idx_pop, y, resultats[nom]["mu"], resultats["baseline"]["mu"], mode)
          for nom in resultats if nom != "baseline" and resultats[nom]["servable"]}

    # Décision : le plus simple des candidats servables qui satisfait tout le critère.
    servables = [n for n in resultats if n != "baseline" and resultats[n]["servable"]]
    verdicts = {}
    for nom in servables:
        r = tableaux[nom][pop]
        gain = 1 - r["mae"] / base["mae"] if base["mae"] else 0.0
        raisons = []
        if gain < GAIN_MAE_MIN:
            raisons.append(f"gain de MAE {lib} {_pct(gain)} < {_pct(GAIN_MAE_MIN)} (MAE {r['mae']:.2f} contre {base['mae']:.2f} pour la baseline)")
        if ic[nom]["bas"] <= 0:
            raisons.append(f"gain non significativement positif : IC 95 % [{_pct(ic[nom]['bas'])} ; {_pct(ic[nom]['haut'])}]")
        if not COUVERTURE_P90[0] <= r["couverture_p90"] <= COUVERTURE_P90[1]:
            raisons.append(f"couverture P90 {lib} {_pct(r['couverture_p90'])} hors de [{_pct(COUVERTURE_P90[0])} ; {_pct(COUVERTURE_P90[1])}]")
        if qc["n_jours_evalues"] < MIN_JOURS_EVALUES:
            raisons.append(f"{qc['n_jours_evalues']} journée(s) évaluée(s) ({lib}) < {MIN_JOURS_EVALUES} : test trop petit ({r['n']} points, {ic[nom]['n_groupes']} groupes indépendants) pour conclure")
        verdicts[nom] = raisons
    acceptes = [n for n in servables if not verdicts[n]]
    candidat = acceptes[0] if acceptes else min(servables, key=lambda n: tableaux[n][pop]["mae"])
    if acceptes:
        r, g = tableaux[candidat][pop], ic[candidat]
        raisons_ok = [f"`{candidat}` : gain de MAE {lib} {_pct(1 - r['mae'] / base['mae'])} (IC 95 % [{_pct(g['bas'])} ; {_pct(g['haut'])}]), "
                      f"couverture P90 {_pct(r['couverture_p90'])}, {qc['n_jours_evalues']} journées : critère satisfait."]
        raisons_ok += [f"`{n}` (non retenu, plus complexe ou refusé) : " + ("; ".join(verdicts[n]) or "critère satisfait aussi")
                       for n in servables if n != candidat]
    decision = {
        "accepte": bool(acceptes),
        "raisons": raisons_ok if acceptes else [f"`{n}` : " + " ; ".join(verdicts[n]) for n in servables],
    }

    # Entraînement final sur toutes les données.
    nom_fabrique = {n: (f, g) for n, f, g in candidats(False)}
    fabrique, grille = nom_fabrique[candidat]
    modele, q, params = ajuster_et_calibrer(fabrique, grille, lignes, mode)
    importance = importance_permutation(modele, [l for l in lignes if l["evalue"]] or lignes)

    sha = hashlib.sha256(chemin.read_bytes()).hexdigest()
    horodatage = datetime.now().astimezone().isoformat(timespec="seconds")
    dates_test = sorted({lignes[i]["date"] for i in idx_pop if lignes[i]["date"]})
    artefact = {
        "version": VERSION_MODELE,
        "date_entrainement": horodatage,
        "seuils_version": F.SEUILS_VERSION,
        "seuils": F.SEUILS,
        "noms_features": list(F.NOMS_FEATURES),
        "nom_modele": candidat,
        "params": params,
        "modele": modele.estimateur,
        "q_conformal": q,
        "niveau_p90": NIVEAU_P90,
        "bornes": BORNES,
        "accepte": decision["accepte"],
        "raisons_refus": [] if decision["accepte"] else decision["raisons"],
        "donnees_simulees": demo,
        "metriques_test": {"population": pop, "modele": tableaux[candidat][pop], "baseline": base},
        "domaine": qc["domaine"],
        "dataset": {"chemin": chemin.relative_to(RACINE).as_posix() if chemin.is_relative_to(RACINE) else str(chemin), "sha256": sha,
                    "n_lignes": qc["n_lignes"], "n_reelles": qc["n_reelles"], "n_synthetiques": qc["n_synthetiques"],
                    "n_simulees": qc["n_simulees"], "n_jours_evalues": qc["n_jours_evalues"]},
        "seed": SEED,
    }
    dossier.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, dossier / f"model{suffixe}.joblib")

    ctx = {
        "date": horodatage, "dataset": artefact["dataset"]["chemin"], "sha256": sha, "qc": qc, "fuite": fuite,
        "daylight": daylight, "mode": mode, "avec_decale": avec_decale, "resultats": resultats, "tableaux": tableaux,
        "ic": ic, "decision": decision, "candidat": candidat, "params_final": params, "q_final": q, "importance": importance,
        "demo": demo, "pop": pop, "lib": lib, "idx_pop": idx_pop, "nom_rapport": f"rapport_evaluation{suffixe}.md",
        "n_jours_test": len(dates_test), "periode_test": f"{dates_test[0]} → {dates_test[-1]}" if dates_test else "—",
    }
    ecrire_rapport(dossier / f"rapport_evaluation{suffixe}.md", ctx)
    ecrire_model_card(dossier / f"model_card{suffixe}.md", ctx)

    print(f"Contrôles : {qc['n_lignes']} lignes ({qc['n_reelles']} réelles, {qc['n_simulees']} simulées), doublons {qc['doublons_exacts']}, "
          f"manquantes {qc['valeurs_manquantes'] or 0}, constantes {qc['colonnes_constantes']}")
    print(f"Fuite indice_global : r = {fuite['correlation_avec_moyenne_cibles']:.4f} -> {'FUITE' if fuite['fuite'] else 'ok'}")
    print(f"is_daylight : {daylight}")
    for nom in resultats:
        r = tableaux[nom][pop]
        print(f"  {nom:42s} MAE {lib} {r['mae']:6.2f}  RMSE {r['rmse']:6.2f}  couv. P90 {100 * r['couverture_p90']:5.1f} %")
    print(f"Décision : {'ACCEPTÉ' if decision['accepte'] else 'REFUSÉ'} ({candidat}){' [DÉMO, données simulées]' if demo else ''}")
    for raison in decision["raisons"]:
        print(f"  - {raison}")
    return ctx


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraînement du modèle de fréquentation NOCTURA")
    parser.add_argument("csv", nargs="?", type=Path, help="dataset (défaut : réel, ou simulé avec --demo)")
    parser.add_argument("--demo", action="store_true", help="entraîne sur données SIMULÉES -> ml/model_demo.joblib")
    arguments = parser.parse_args()
    main(arguments.csv, demo=arguments.demo)
