#!/usr/bin/env python3
"""Validation de schéma + nettoyage du dataset de fréquentation 24h.

Usage:
    python validate_clean.py INPUT.csv [--out-dir DIR] [--chunksize 500000]

Sorties (dans --out-dir, par défaut ./out) :
    <nom>_clean.csv     lignes valides, types normalisés, colonnes ordonnées
    <nom>_rejects.csv   lignes invalides + colonne reject_reason
    <nom>_report.json   rapport (comptes par règle, avertissements)

Code retour : 0 si aucune ligne rejetée, 1 sinon, 2 si le schéma est cassé.
Lecture par chunks : fonctionne sur des fichiers massifs, même structure.
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- schéma ---
COLUMNS = [  # ordre canonique de sortie
    "creneau", "hour_start", "hour_end", "sin_hour", "cos_hour",
    "zone_group_id", "zone_group_name", "zone_specifique",
    "is_daylight", "is_winter", "is_storm", "is_synthetic_daytime",
    "indice_global", "frequentation_target",
]
STR_COLS = ["creneau", "zone_group_id", "zone_group_name", "zone_specifique"]
INT_COLS = ["hour_start", "hour_end", "indice_global", "frequentation_target"]
FLOAT_COLS = ["sin_hour", "cos_hour"]
FLAG_COLS = ["is_daylight", "is_winter", "is_storm", "is_synthetic_daytime"]

# Référentiel observé dans le dataset actuel. Une valeur inconnue = avertissement
# (de nouvelles zones peuvent apparaître) ; une valeur connue mal rattachée = rejet.
GROUPS = {"G1": "Port et débarcadère", "G2": "Village et commerces"}
ZONES = {
    "capitainerie": "G1", "entree_du_port": "G1", "pontons_centraux": "G1",
    "quai_est": "G1", "quai_nord": "G1", "quai_ouest": "G1",
    "acces_village": "G2", "quai_sud_commerces": "G2",
}
SLOT_HOURS = 2
TRIG_TOL = 1e-6
BOOL_MAP = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}


def parse(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les types ; ce qui n'est pas convertible devient NaN/<NA>."""
    out = pd.DataFrame(index=df.index)
    for c in STR_COLS:
        out[c] = df[c].astype("string").str.strip()
    for c in INT_COLS:
        num = pd.to_numeric(df[c], errors="coerce")
        # un décimal (ex. 3.5) n'est pas un entier valide -> NaN
        num = num.where(num == num.round())
        out[c] = num.astype("Int64")
    for c in FLOAT_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    for c in FLAG_COLS:
        s = df[c].astype("string").str.strip().str.lower().map(BOOL_MAP)
        out[c] = s.astype("Int64")
    return out


def row_rules(d: pd.DataFrame) -> dict:
    """Retourne {règle: masque booléen True = ligne OK}."""
    parsed_ok = d[COLUMNS].notna().all(axis=1)
    hs, he = d["hour_start"], d["hour_end"]

    hour_ok = hs.between(0, 23) & (he == (hs + SLOT_HOURS) % 24)
    label = hs.astype("string").str.zfill(2) + "h-" + he.astype("string").str.zfill(2) + "h"
    label_ok = d["creneau"] == label
    trig_ok = (
        (d["sin_hour"] - np.sin(2 * np.pi * hs.astype("float64") / 24)).abs().lt(TRIG_TOL)
        & (d["cos_hour"] - np.cos(2 * np.pi * hs.astype("float64") / 24)).abs().lt(TRIG_TOL)
    )
    flags_ok = pd.concat([d[c].isin([0, 1]) for c in FLAG_COLS], axis=1).all(axis=1)

    exp_group = d["zone_specifique"].map(ZONES)  # NaN si zone inconnue
    zone_ok = exp_group.isna() | (exp_group == d["zone_group_id"])
    exp_name = d["zone_group_id"].map(GROUPS)
    group_ok = exp_name.isna() | (exp_name == d["zone_group_name"])
    target_ok = d["frequentation_target"] >= 0

    rules = {
        "type_ou_valeur_manquante": parsed_ok,
        "heure_incoherente": hour_ok,
        "creneau_incoherent": label_ok,
        "sin_cos_incoherents": trig_ok,
        "flag_non_binaire": flags_ok,
        "zone_mal_rattachee_au_groupe": zone_ok,
        "groupe_id_nom_incoherent": group_ok,
        "target_negative": target_ok,
    }
    # un NaN dans une comparaison = règle non satisfaite ; la 1re règle l'explique
    return {k: v.fillna(False).astype(bool) for k, v in rules.items()}


def duplicate_mask(d: pd.DataFrame, seen_rows: sqlite3.Connection) -> pd.Series:
    """Repère les doublons normalisés sur tout le fichier sans garder l'historique en RAM."""
    row_keys = [
        json.dumps(
            [
                None if pd.isna(value) else value.item() if hasattr(value, "item") else value
                for value in row
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for row in d.itertuples(index=False, name=None)
    ]
    local_duplicates = pd.Series(row_keys, index=d.index).duplicated()
    distinct_keys = list(dict.fromkeys(row_keys))
    previously_seen = set()
    for offset in range(0, len(distinct_keys), 500):
        batch = distinct_keys[offset : offset + 500]
        placeholders = ",".join("?" for _ in batch)
        previously_seen.update(
            row[0]
            for row in seen_rows.execute(
                f"SELECT row_key FROM seen_rows WHERE row_key IN ({placeholders})",
                batch,
            )
        )
    seen_rows.executemany(
        "INSERT OR IGNORE INTO seen_rows (row_key) VALUES (?)",
        ((key,) for key in distinct_keys),
    )
    return local_duplicates | pd.Series(row_keys, index=d.index).isin(previously_seen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--drop-duplicates", action="store_true",
                    help="supprime les doublons exacts dans tout le fichier (désactivé par défaut)")
    a = ap.parse_args()

    src = Path(a.input)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f_clean, f_rej, f_rep = (out_dir / f"{src.stem}{s}" for s in ("_clean.csv", "_rejects.csv", "_report.json"))
    for f in (f_clean, f_rej):
        f.unlink(missing_ok=True)

    rep = {"fichier": src.name, "lignes_lues": 0, "lignes_propres": 0, "lignes_rejetees": 0,
           "rejets_par_regle": {}, "doublons_exacts": 0, "avertissements": []}
    uniques = {c: set() for c in FLAG_COLS}
    unknown_zones, unknown_groups = set(), set()
    wrote_clean = wrote_rej = False
    seen_rows = sqlite3.connect("")
    seen_rows.execute("CREATE TABLE seen_rows (row_key TEXT PRIMARY KEY)")

    for i, raw in enumerate(pd.read_csv(src, chunksize=a.chunksize, dtype=str, keep_default_na=False,
                                        na_values=[""])):
        raw.columns = raw.columns.str.strip()
        if i == 0:  # contrôle structurel : bloquant
            missing = [c for c in COLUMNS if c not in raw.columns]
            extra = [c for c in raw.columns if c not in COLUMNS]
            if missing:
                print(f"ERREUR schéma : colonnes manquantes {missing}", file=sys.stderr)
                seen_rows.close()
                return 2
            if extra:
                rep["avertissements"].append(f"colonnes supplémentaires ignorées : {extra}")
        raw = raw[COLUMNS]
        rep["lignes_lues"] += len(raw)

        d = parse(raw)
        dup = duplicate_mask(d, seen_rows)
        rep["doublons_exacts"] += int(dup.sum())

        rules = row_rules(d)
        ok = pd.concat(rules.values(), axis=1).all(axis=1)
        for name, mask in rules.items():
            n = int((~mask).sum())
            if n:
                rep["rejets_par_regle"][name] = rep["rejets_par_regle"].get(name, 0) + n

        good = d[ok].copy()
        if a.drop_duplicates:
            good = good.loc[~dup.loc[ok]]
        for c in FLAG_COLS:
            uniques[c] |= set(good[c].dropna().unique().tolist())
        unknown_zones |= set(good.loc[~good["zone_specifique"].isin(ZONES), "zone_specifique"])
        unknown_groups |= set(good.loc[~good["zone_group_id"].isin(GROUPS), "zone_group_id"])

        # nettoyage : sin/cos recalculés (supprime le bruit flottant type 6e-17)
        h = good["hour_start"].astype("float64")
        good["sin_hour"] = np.sin(2 * np.pi * h / 24).round(10) + 0.0
        good["cos_hour"] = np.cos(2 * np.pi * h / 24).round(10) + 0.0
        for c in INT_COLS + FLAG_COLS:
            good[c] = good[c].astype("int64")
        good[COLUMNS].to_csv(f_clean, mode="a", header=not wrote_clean, index=False)
        wrote_clean = True
        rep["lignes_propres"] += len(good)

        bad = raw[~ok].copy()
        if len(bad):
            reasons = pd.concat([(~m).map({True: n, False: ""}) for n, m in rules.items()], axis=1)
            bad["reject_reason"] = reasons.loc[bad.index].apply(lambda r: ";".join(x for x in r if x), axis=1)
            bad.to_csv(f_rej, mode="a", header=not wrote_rej, index=False)
            wrote_rej = True
            rep["lignes_rejetees"] += len(bad)

    seen_rows.close()

    # avertissements globaux (non bloquants)
    for c, vals in uniques.items():
        if len(vals) == 1:
            rep["avertissements"].append(
                f"{c} est constante ({vals.pop()}) : aucune information pour l'entraînement sur ces données")
    if unknown_zones:
        rep["avertissements"].append(f"zones hors référentiel : {sorted(unknown_zones)}")
    if unknown_groups:
        rep["avertissements"].append(f"groupes hors référentiel : {sorted(unknown_groups)}")
    if rep["doublons_exacts"]:
        rep["avertissements"].append(
            f"{rep['doublons_exacts']} doublons exacts (conservés ; --drop-duplicates pour les retirer)")

    f_rep.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 1 if rep["lignes_rejetees"] else 0


if __name__ == "__main__":
    sys.exit(main())
