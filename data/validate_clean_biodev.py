#!/usr/bin/env python3
"""Validation + nettoyage du classeur biodev (effectifs de poissons par espèce et par année).

Usage:
    python validate_clean_biodev.py INPUT.xlsx [--out-dir DIR]

Structure attendue (les années sont détectées automatiquement, donc de nouvelles
années ajoutées plus tard sont acceptées) :
    Feuille « Espèces et évolution » : Espèce | <années...> | Total <a>-<b> | Évolution <a>-<b>
    Feuille « Synthèse annuelle »    : Indicateur | <années...>
                                       (Nombre total d'individus ; Nombre d'espèces recensées)
    Feuille « Source et méthode »    : Point | Détail   (conservée telle quelle, non validée)

Sorties : <nom>_tidy.csv (format long, prêt pour un ML), <nom>_report.json
Code retour : 0 = OK, 1 = incohérences de données, 2 = structure cassée.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import NoReturn, Optional

import pandas as pd

# ---------------------------------------------------------------- schéma ---
S_ESP, S_SYN = "Espèces et évolution", "Synthèse annuelle"
IND_TOTAL, IND_NSP = "Nombre total d'individus", "Nombre d'espèces recensées"


def fail(msg: str) -> NoReturn:
    print(f"ERREUR structure : {msg}", file=sys.stderr)
    sys.exit(2)


def evolution_label(first: int, last: int, y_last: str) -> str:
    if first > 0 and last == 0:
        return f"Non observée en {y_last}"
    if first == 0 and last > 0:
        return f"Nouvelle observation en {y_last}"
    if last > first:
        return "En hausse"
    if last < first:
        return "En baisse"
    return "Stable"


def split_name(raw: str) -> tuple[Optional[str], str]:
    """'Saupe (Sarpa salpa)' -> ('Saupe', 'Sarpa salpa') ; 'Symphodus sp.' -> (None, 'Symphodus sp.')"""
    m = re.match(r"^(.*?)\s*\((.+)\)$", raw)
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, raw.strip())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--out-dir", default="out")
    return ap.parse_args()


def load_sheets(src: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sheets = pd.read_excel(src, sheet_name=None)
    for s in (S_ESP, S_SYN):
        if s not in sheets:
            fail(f"feuille manquante : {s!r}")
    esp, syn = sheets[S_ESP].copy(), sheets[S_SYN].copy()
    esp.columns = [str(c).strip() for c in esp.columns]
    syn.columns = [str(c).strip() for c in syn.columns]
    return esp, syn


def find_schema(
    esp: pd.DataFrame, syn: pd.DataFrame
) -> tuple[list[str], str, str]:
    years = [c for c in esp.columns if re.fullmatch(r"\d{4}", c)]
    c_tot = next((c for c in esp.columns if c.startswith("Total")), None)
    c_evo = next((c for c in esp.columns if c.startswith("Évolution")), None)
    if "Espèce" not in esp.columns or not years or not c_tot or not c_evo:
        fail("colonnes attendues : Espèce, années (YYYY), Total ..., Évolution ...")
    if "Indicateur" not in syn.columns or not set(years) <= set(syn.columns):
        fail("feuille de synthèse : colonne Indicateur + mêmes années requises")
    return years, c_tot, c_evo


def validate_values(
    esp: pd.DataFrame, years: list[str], c_tot: str
) -> list[str]:
    errors = []

    # ---- types et valeurs
    esp["Espèce"] = esp["Espèce"].astype("string").str.strip()
    if esp["Espèce"].isna().any() or (esp["Espèce"] == "").any():
        errors.append("nom d'espèce manquant")
    for c in years + [c_tot]:
        num = pd.to_numeric(esp[c], errors="coerce")
        if num.isna().any() or (num != num.round()).any() or (num < 0).any():
            errors.append(f"colonne {c} : valeurs manquantes, non entières ou négatives")
        esp[c] = num
    if esp["Espèce"].str.lower().duplicated().any():
        errors.append(f"espèces en double : {esp.loc[esp['Espèce'].str.lower().duplicated(), 'Espèce'].tolist()}")
    if not errors:
        esp[years + [c_tot]] = esp[years + [c_tot]].astype("int64")
    return errors


def validate_consistency(
    esp: pd.DataFrame,
    syn: pd.DataFrame,
    years: list[str],
    c_tot: str,
    c_evo: str,
) -> list[str]:
    errors = []
    # ---- cohérences croisées
    bad = esp[esp[years].sum(axis=1) != esp[c_tot]]
    if len(bad):
        errors.append(f"total par espèce ≠ somme des années : {bad['Espèce'].tolist()}")

    s = syn.set_index("Indicateur")[years].apply(pd.to_numeric, errors="coerce")
    for ind, calc in ((IND_TOTAL, esp[years].sum()), (IND_NSP, (esp[years] > 0).sum())):
        if ind not in s.index:
            errors.append(f"indicateur absent de la synthèse : {ind!r}")
            continue
        diff = [y for y in years if s.loc[ind, y] != calc[y]]
        if diff:
            errors.append(f"{ind} : synthèse ≠ recalcul pour {diff} "
                          f"(synthèse {s.loc[ind, diff].tolist()}, recalcul {calc[diff].tolist()})")

    y0, y1 = years[0], years[-1]
    exp = esp.apply(lambda r: evolution_label(r[y0], r[y1], y1), axis=1)
    bad = esp[exp != esp[c_evo].astype("string").str.strip()]
    if len(bad):
        errors.append(f"libellé d'évolution incohérent : {bad['Espèce'].tolist()}")
    return errors


def add_species_metadata(
    esp: pd.DataFrame, years: list[str], c_evo: str, warnings: list[str]
) -> None:
    y0, y1 = years[0], years[-1]
    # ---- avertissements (non bloquants)
    stable_mid = esp[(esp[c_evo] == "Stable") & (esp[[y0, y1]].sum(axis=1) == 0)]
    if len(stable_mid):
        warnings.append(f"{len(stable_mid)} espèces « Stable » = absentes en {y0} et {y1}, vues seulement "
                        f"entre-temps (le libellé compare uniquement {y0} et {y1}) : {stable_mid['Espèce'].tolist()}")
    names = esp["Espèce"].map(split_name)
    esp["nom_commun"] = names.str[0]
    esp["nom_scientifique"] = names.str[1]
    esp["taxon_indetermine"] = esp["nom_scientifique"].str.contains(r"\bsp\.?$", regex=True).astype(int)
    if esp["taxon_indetermine"].any():
        warnings.append(f"{int(esp['taxon_indetermine'].sum())} taxons indéterminés (sp.) : "
                        f"{esp.loc[esp['taxon_indetermine'] == 1, 'Espèce'].tolist()}")
    warnings.append("un 0 = « espèce non mentionnée cette année » (cf. feuille Source et méthode), "
                    "pas forcément une absence mesurée")


def build_tidy(esp: pd.DataFrame, years: list[str]) -> pd.DataFrame:
    tidy = esp.melt(id_vars=["Espèce", "nom_commun", "nom_scientifique", "taxon_indetermine"],
                    value_vars=years, var_name="annee", value_name="effectif")
    tidy = tidy.rename(columns={"Espèce": "espece"})
    tidy["annee"] = tidy["annee"].astype(int)
    tidy["effectif"] = tidy["effectif"].astype(int)
    tidy = tidy.sort_values(["espece", "annee"]).reset_index(drop=True)
    assert len(tidy) == len(esp) * len(years)
    return tidy


def write_report(report: dict[str, object], path: Path) -> None:
    content = json.dumps(report, indent=2, ensure_ascii=False)
    path.write_text(content, encoding="utf-8")
    print(content)


def main() -> int:
    args = parse_args()
    src, out = Path(args.input), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    esp, syn = load_sheets(src)
    years, c_tot, c_evo = find_schema(esp, syn)
    errors = validate_values(esp, years, c_tot)
    warnings = []
    if errors:  # inutile de recouper des chiffres invalides
        report = {"statut": "ERREUR", "erreurs": errors, "avertissements": warnings}
        write_report(report, out / f"{src.stem}_report.json")
        return 1

    errors.extend(validate_consistency(esp, syn, years, c_tot, c_evo))
    add_species_metadata(esp, years, c_evo, warnings)

    # ---- format long (tidy)
    tidy = build_tidy(esp, years)
    tidy.to_csv(out / f"{src.stem}_tidy.csv", index=False)

    report = {
        "fichier": src.name, "statut": "ERREUR" if errors else "OK",
        "annees": [int(y) for y in years], "nb_especes": int(len(esp)), "lignes_tidy": int(len(tidy)),
        "erreurs": errors, "avertissements": warnings,
    }
    write_report(report, out / f"{src.stem}_report.json")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
