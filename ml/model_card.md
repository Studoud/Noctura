# Fiche modèle : fréquentation NOCTURA v1.0.0

- **Date d'entraînement** : 2026-10-17T16:30:20+02:00
- **Version** : 1.0.0 ; features `seuils-v0-hypothese` (18 colonnes) ; graine 42
- **Modèle** : `glm_poisson` (paramètres {'alpha': 0.0001}), P90 conformal (q = 0.788 sur résidus normalisés)
- **Statut** : ACCEPTÉ, activable avec PREDICTION_MODE=ml

## Usage prévu

Prédire la **demande** (fréquentation 0-100) par zone et par créneau de 2 h. Le modèle ne décide jamais de l'éclairage : les règles (`cerveau/regles.py`) et le validateur de sécurité (`cerveau/securite.py`) restent les seuls décideurs. Utiliser `indice_p90` pour toute décision où sous-estimer la fréquentation éteindrait la lumière sur des personnes.

## Données

- `data/dataset_ml_frequentation_24h_clean.csv` (sha256 `9dcc01314104e9deff2c970d1cae036b82ab8e293c5f706a0d249be157f5c562`)
- 10000 lignes : 4176 réelles, 5824 synthétiques, 0 simulées (**58 % non réel**)
- Journées évaluées (réel) : 105 (minimum requis : 5)
- Domaine observé : {'is_daylight': [0, 1], 'is_winter': [0, 1], 'is_storm': [0, 1]}
- Pas de colonne `date` : journées reconstituées par la suite des créneaux (105 journées) ; date calendaire inconnue, donc les features saison / week-end sont constantes à l'entraînement et sans effet.
- Variable exclue : `indice_global` (agrégat des cibles du même créneau, fuite confirmée).

## Métriques (test réel hors échantillon)

| | MAE | RMSE | Couverture P90 |
|---|---|---|---|
| Baseline zone × créneau | 2.13 | 2.68 | 98.8 % |
| glm_poisson | 1.16 | 1.57 | 90.0 % |

Détail : `ml/rapport_evaluation.md`.

## Décision

- `glm_poisson` : gain de MAE réel 45.5 % (IC 95 % [28.4 % ; 58.4 %]), couverture P90 90.0 %, 105 journées : critère satisfait.
- `lightgbm_poisson` (non retenu, plus complexe ou refusé) : critère satisfait aussi

## Limites

- **Météo** : Open-Meteo aux coordonnées du port de Porquerolles ; sortie de modèle numérique (maille de quelques km), pas une mesure sur place.
- **Seuils `is_storm` / `is_winter` hypothétiques** (seuils-v0-hypothese) : ceux du générateur du dataset d'origine ne sont pas connus.
- **`is_daylight`** : règle « soleil au-dessus de l'horizon au milieu du créneau ». Non vérifiable ligne à ligne (pas de date calendaire dans le dataset).
- La biodiversité n'est pas une feature (4 années ne permettent pas d'apprendre un effet) : voir `ml/biodiversite.py`.

## Réentraîner

Ajouter une colonne `date` (AAAA-MM-JJ) et les nouvelles journées réelles au CSV, puis `python -m ml.train`. Le script passe automatiquement en validation par journées et réévalue le critère d'acceptation.
