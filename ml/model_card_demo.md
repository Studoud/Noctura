# Fiche modèle : fréquentation NOCTURA v1.0.0 (DÉMO)

> **Modèle de démonstration entraîné sur des données SIMULÉES** (`ml/simulation.py`). Il apprend le générateur, pas le vrai port. Ne jamais l'utiliser pour piloter l'éclairage réel.

- **Date d'entraînement** : 2026-10-17T15:51:41+02:00
- **Version** : 1.0.0 ; features `seuils-v0-hypothese` (18 colonnes) ; graine 42
- **Modèle** : `glm_poisson` (paramètres {'alpha': 0.0001}), P90 conformal (q = 1.631 sur résidus normalisés)
- **Statut** : ACCEPTÉ SUR DONNÉES SIMULÉES : activable uniquement avec PREDICTION_MODE=demo (refusé par PREDICTION_MODE=ml)

## Usage prévu

Prédire la **demande** (fréquentation 0-100) par zone et par créneau de 2 h. Le modèle ne décide jamais de l'éclairage : les règles (`cerveau/regles.py`) et le validateur de sécurité (`cerveau/securite.py`) restent les seuls décideurs. Utiliser `indice_p90` pour toute décision où sous-estimer la fréquentation éteindrait la lumière sur des personnes.

## Données

- `data/dataset_simule.csv` (sha256 `311cee0de9d30ffe4f9a8e4c77315e10d54eb510a5a78c7107ad6ec9f24c7dbb`)
- 70176 lignes : 0 réelles, 0 synthétiques, 70176 simulées (**100 % non réel**)
- Journées évaluées (SIMULÉ) : 731 (minimum requis : 5)
- Domaine observé : {'is_daylight': [0, 1], 'is_winter': [0, 1], 'is_storm': [0, 1]}
- Variable exclue : `indice_global` (agrégat des cibles du même créneau, fuite confirmée).

## Métriques (test SIMULÉ hors échantillon)

| | MAE | RMSE | Couverture P90 |
|---|---|---|---|
| Baseline zone × créneau | 4.51 | 6.05 | 94.9 % |
| glm_poisson | 2.63 | 3.59 | 94.5 % |

Détail : `ml/rapport_evaluation_demo.md`.

## Décision

- `glm_poisson` : gain de MAE SIMULÉ 41.6 % (IC 95 % [36.3 % ; 46.4 %]), couverture P90 94.5 %, 731 journées : critère satisfait.
- `lightgbm_poisson` (non retenu, plus complexe ou refusé) : critère satisfait aussi

## Limites

- **Météo** : Open-Meteo aux coordonnées du port de Porquerolles ; sortie de modèle numérique (maille de quelques km), pas une mesure sur place.
- **Seuils `is_storm` / `is_winter` hypothétiques** (seuils-v0-hypothese) : ceux du générateur du dataset d'origine ne sont pas connus.
- **`is_daylight`** : règle « soleil au-dessus de l'horizon au milieu du créneau ». Taux d'écart avec le dataset : 0.0 %.
- La biodiversité n'est pas une feature (4 années ne permettent pas d'apprendre un effet) : voir `ml/biodiversite.py`.

## Réentraîner

`python -m ml.simulation` puis `python -m ml.train --demo`.
