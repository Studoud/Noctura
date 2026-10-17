# Rapport d'évaluation : fréquentation NOCTURA

Généré par `ml/train.py` le 2026-10-17T16:30:20+02:00 (graine 42). Tous les chiffres ci-dessous sont calculés par le script ; aucun n'est saisi à la main.

## Protocole

- Dataset : `data/dataset_ml_frequentation_24h_clean.csv` (sha256 `9dcc01314104…`), 10000 lignes : 4176 réelles, 5824 synthétiques, 0 simulées.
- Population évaluée : lignes **réel** (105 journée(s)).
- Mode de validation : **jours**. Les 21 dernières journées (J085 → J105) forment le test, jamais vues ; calibration par GroupKFold par journée sur les précédentes.
- Hyperparamètres et quantile P90 (conformal sur résidus normalisés `(y-μ)/√μ`) choisis par validation croisée interne groupée, dans chaque pli externe.
- `indice_global` est exclu des modèles servables (fuite, voir plus bas) ; les variantes « (FUITE) » ne sont là que pour mesurer son effet.

## Comparaison des modèles (test hors échantillon)

| Modèle | Servable | MAE réel | RMSE réel | Couv. P90 réel | MAE synthetique | RMSE synthetique | Gain MAE vs baseline |
|---|---|---|---|---|---|---|---|
| baseline | oui | 2.13 | 2.68 | 98.8 % | 4.84 | 5.81 | — |
| glm_poisson | oui | 1.16 | 1.57 | 90.0 % | 1.68 | 2.30 | 45.5 % |
| lightgbm_poisson | oui | 1.26 | 1.61 | 94.6 % | 1.74 | 2.40 | 40.9 % |
| glm_poisson+indice_global (FUITE) | non | 0.80 | 1.12 | 92.6 % | 1.34 | 1.86 | 62.4 % |
| lightgbm_poisson+indice_global (FUITE) | non | 0.94 | 1.28 | 92.3 % | 1.42 | 1.92 | 55.9 % |
| glm_poisson+indice_decale | non | 1.18 | 1.58 | 90.7 % | 1.68 | 2.32 | 44.4 % |
| lightgbm_poisson+indice_decale | non | 1.26 | 1.59 | 94.2 % | 1.85 | 2.58 | 40.9 % |

IC 95 % du gain de MAE réel (bootstrap sur les groupes indépendants) :

- glm_poisson : [28.4 % ; 58.4 %] sur 21 groupes
- lightgbm_poisson : [25.2 % ; 52.5 %] sur 21 groupes

## Détail par segment

### baseline

| Segment | n | MAE | RMSE | Couv. P90 | Sous-estimation P50 |
|---|---|---|---|---|---|
| reel | 816 | 2.13 | 2.68 | 98.8 % | 30.5 % |
| reel · nuit | 816 | 2.13 | 2.68 | 98.8 % | 30.5 % |
| reel · zone acces_village | 102 | 2.49 | 3.00 | 99.0 % | 32.4 % |
| reel · zone capitainerie | 102 | 1.28 | 1.56 | 100.0 % | 34.3 % |
| reel · zone entree_du_port | 102 | 1.95 | 2.35 | 99.0 % | 31.4 % |
| reel · zone pontons_centraux | 102 | 2.33 | 2.86 | 96.1 % | 30.4 % |
| reel · zone quai_est | 102 | 1.78 | 2.16 | 100.0 % | 32.4 % |
| reel · zone quai_nord | 102 | 1.54 | 1.82 | 100.0 % | 30.4 % |
| reel · zone quai_ouest | 102 | 2.46 | 3.00 | 99.0 % | 26.5 % |
| reel · zone quai_sud_commerces | 102 | 3.18 | 3.91 | 97.1 % | 26.5 % |
| synthetique | 1120 | 4.84 | 5.81 | 95.2 % | 24.9 % |
| synthetique · jour | 768 | 4.90 | 5.81 | 93.0 % | 33.2 % |
| synthetique · nuit | 352 | 4.72 | 5.81 | 100.0 % | 6.8 % |
| tout · jour | 768 | 4.90 | 5.81 | 93.0 % | 33.2 % |
| tout · nuit | 1168 | 2.91 | 3.90 | 99.1 % | 23.4 % |

### glm_poisson

| Segment | n | MAE | RMSE | Couv. P90 | Sous-estimation P50 |
|---|---|---|---|---|---|
| reel | 816 | 1.16 | 1.57 | 90.0 % | 42.4 % |
| reel · nuit | 816 | 1.16 | 1.57 | 90.0 % | 42.4 % |
| reel · zone acces_village | 102 | 1.30 | 1.70 | 89.2 % | 52.0 % |
| reel · zone capitainerie | 102 | 0.83 | 1.00 | 97.1 % | 28.4 % |
| reel · zone entree_du_port | 102 | 0.98 | 1.33 | 93.1 % | 47.1 % |
| reel · zone pontons_centraux | 102 | 1.18 | 1.48 | 91.2 % | 34.3 % |
| reel · zone quai_est | 102 | 1.05 | 1.27 | 90.2 % | 35.3 % |
| reel · zone quai_nord | 102 | 0.93 | 1.15 | 96.1 % | 31.4 % |
| reel · zone quai_ouest | 102 | 1.14 | 1.46 | 85.3 % | 59.8 % |
| reel · zone quai_sud_commerces | 102 | 1.86 | 2.61 | 77.5 % | 51.0 % |
| synthetique | 1120 | 1.68 | 2.30 | 93.8 % | 43.8 % |
| synthetique · jour | 768 | 1.94 | 2.61 | 91.5 % | 47.3 % |
| synthetique · nuit | 352 | 1.10 | 1.43 | 98.6 % | 36.4 % |
| tout · jour | 768 | 1.94 | 2.61 | 91.5 % | 47.3 % |
| tout · nuit | 1168 | 1.14 | 1.53 | 92.6 % | 40.6 % |

### lightgbm_poisson

| Segment | n | MAE | RMSE | Couv. P90 | Sous-estimation P50 |
|---|---|---|---|---|---|
| reel | 816 | 1.26 | 1.61 | 94.6 % | 31.0 % |
| reel · nuit | 816 | 1.26 | 1.61 | 94.6 % | 31.0 % |
| reel · zone acces_village | 102 | 1.19 | 1.52 | 95.1 % | 52.9 % |
| reel · zone capitainerie | 102 | 0.98 | 1.14 | 100.0 % | 8.8 % |
| reel · zone entree_du_port | 102 | 1.28 | 1.53 | 99.0 % | 13.7 % |
| reel · zone pontons_centraux | 102 | 1.16 | 1.39 | 98.0 % | 29.4 % |
| reel · zone quai_est | 102 | 1.27 | 1.47 | 100.0 % | 16.7 % |
| reel · zone quai_nord | 102 | 1.18 | 1.37 | 100.0 % | 10.8 % |
| reel · zone quai_ouest | 102 | 1.05 | 1.38 | 91.2 % | 62.7 % |
| reel · zone quai_sud_commerces | 102 | 1.94 | 2.62 | 73.5 % | 52.9 % |
| synthetique | 1120 | 1.74 | 2.40 | 92.8 % | 39.7 % |
| synthetique · jour | 768 | 2.02 | 2.72 | 91.1 % | 41.3 % |
| synthetique · nuit | 352 | 1.13 | 1.48 | 96.3 % | 36.4 % |
| tout · jour | 768 | 2.02 | 2.72 | 91.1 % | 41.3 % |
| tout · nuit | 1168 | 1.22 | 1.57 | 95.1 % | 32.6 % |

### glm_poisson+indice_global (FUITE)

| Segment | n | MAE | RMSE | Couv. P90 | Sous-estimation P50 |
|---|---|---|---|---|---|
| reel | 816 | 0.80 | 1.12 | 92.6 % | 43.6 % |
| reel · nuit | 816 | 0.80 | 1.12 | 92.6 % | 43.6 % |
| reel · zone acces_village | 102 | 0.71 | 0.95 | 97.1 % | 50.0 % |
| reel · zone capitainerie | 102 | 0.50 | 0.66 | 100.0 % | 21.6 % |
| reel · zone entree_du_port | 102 | 0.69 | 0.90 | 91.2 % | 48.0 % |
| reel · zone pontons_centraux | 102 | 0.73 | 0.92 | 98.0 % | 29.4 % |
| reel · zone quai_est | 102 | 0.72 | 0.92 | 98.0 % | 36.3 % |
| reel · zone quai_nord | 102 | 0.74 | 0.89 | 98.0 % | 32.4 % |
| reel · zone quai_ouest | 102 | 0.71 | 0.99 | 89.2 % | 72.5 % |
| reel · zone quai_sud_commerces | 102 | 1.59 | 2.11 | 69.6 % | 58.8 % |
| synthetique | 1120 | 1.34 | 1.86 | 90.8 % | 47.9 % |
| synthetique · jour | 768 | 1.57 | 2.12 | 87.8 % | 50.1 % |
| synthetique · nuit | 352 | 0.83 | 1.10 | 97.4 % | 42.9 % |
| tout · jour | 768 | 1.57 | 2.12 | 87.8 % | 50.1 % |
| tout · nuit | 1168 | 0.81 | 1.11 | 94.1 % | 43.4 % |

### lightgbm_poisson+indice_global (FUITE)

| Segment | n | MAE | RMSE | Couv. P90 | Sous-estimation P50 |
|---|---|---|---|---|---|
| reel | 816 | 0.94 | 1.28 | 92.3 % | 36.2 % |
| reel · nuit | 816 | 0.94 | 1.28 | 92.3 % | 36.2 % |
| reel · zone acces_village | 102 | 0.83 | 1.09 | 95.1 % | 64.7 % |
| reel · zone capitainerie | 102 | 0.68 | 0.79 | 100.0 % | 2.9 % |
| reel · zone entree_du_port | 102 | 0.86 | 1.06 | 99.0 % | 16.7 % |
| reel · zone pontons_centraux | 102 | 0.76 | 0.91 | 99.0 % | 30.4 % |
| reel · zone quai_est | 102 | 0.83 | 1.02 | 100.0 % | 18.6 % |
| reel · zone quai_nord | 102 | 0.89 | 1.04 | 99.0 % | 12.7 % |
| reel · zone quai_ouest | 102 | 0.78 | 1.09 | 89.2 % | 76.5 % |
| reel · zone quai_sud_commerces | 102 | 1.89 | 2.45 | 56.9 % | 66.7 % |
| synthetique | 1120 | 1.42 | 1.92 | 92.4 % | 40.2 % |
| synthetique · jour | 768 | 1.66 | 2.18 | 90.2 % | 42.4 % |
| synthetique · nuit | 352 | 0.91 | 1.15 | 97.2 % | 35.2 % |
| tout · jour | 768 | 1.66 | 2.18 | 90.2 % | 42.4 % |
| tout · nuit | 1168 | 0.93 | 1.24 | 93.8 % | 35.9 % |

### glm_poisson+indice_decale

| Segment | n | MAE | RMSE | Couv. P90 | Sous-estimation P50 |
|---|---|---|---|---|---|
| reel | 816 | 1.18 | 1.58 | 90.7 % | 41.5 % |
| reel · nuit | 816 | 1.18 | 1.58 | 90.7 % | 41.5 % |
| reel · zone acces_village | 102 | 1.30 | 1.71 | 91.2 % | 46.1 % |
| reel · zone capitainerie | 102 | 0.87 | 1.07 | 97.1 % | 29.4 % |
| reel · zone entree_du_port | 102 | 1.02 | 1.38 | 93.1 % | 36.3 % |
| reel · zone pontons_centraux | 102 | 1.23 | 1.53 | 93.1 % | 33.3 % |
| reel · zone quai_est | 102 | 1.06 | 1.29 | 90.2 % | 42.2 % |
| reel · zone quai_nord | 102 | 0.96 | 1.21 | 96.1 % | 26.5 % |
| reel · zone quai_ouest | 102 | 1.13 | 1.44 | 86.3 % | 65.7 % |
| reel · zone quai_sud_commerces | 102 | 1.89 | 2.56 | 78.4 % | 52.9 % |
| synthetique | 1120 | 1.68 | 2.32 | 93.5 % | 43.5 % |
| synthetique · jour | 768 | 1.97 | 2.64 | 91.4 % | 44.5 % |
| synthetique · nuit | 352 | 1.05 | 1.37 | 98.0 % | 41.2 % |
| tout · jour | 768 | 1.97 | 2.64 | 91.4 % | 44.5 % |
| tout · nuit | 1168 | 1.14 | 1.52 | 92.9 % | 41.4 % |

### lightgbm_poisson+indice_decale

| Segment | n | MAE | RMSE | Couv. P90 | Sous-estimation P50 |
|---|---|---|---|---|---|
| reel | 816 | 1.26 | 1.59 | 94.2 % | 32.6 % |
| reel · nuit | 816 | 1.26 | 1.59 | 94.2 % | 32.6 % |
| reel · zone acces_village | 102 | 1.18 | 1.52 | 95.1 % | 54.9 % |
| reel · zone capitainerie | 102 | 1.00 | 1.15 | 100.0 % | 7.8 % |
| reel · zone entree_du_port | 102 | 1.18 | 1.46 | 98.0 % | 18.6 % |
| reel · zone pontons_centraux | 102 | 1.15 | 1.34 | 98.0 % | 27.5 % |
| reel · zone quai_est | 102 | 1.28 | 1.48 | 100.0 % | 17.6 % |
| reel · zone quai_nord | 102 | 1.18 | 1.37 | 100.0 % | 16.7 % |
| reel · zone quai_ouest | 102 | 1.09 | 1.40 | 91.2 % | 57.8 % |
| reel · zone quai_sud_commerces | 102 | 2.00 | 2.60 | 71.6 % | 59.8 % |
| synthetique | 1120 | 1.85 | 2.58 | 89.7 % | 45.0 % |
| synthetique · jour | 768 | 2.13 | 2.91 | 88.9 % | 48.2 % |
| synthetique · nuit | 352 | 1.25 | 1.64 | 91.5 % | 38.1 % |
| tout · jour | 768 | 2.13 | 2.91 | 88.9 % | 48.2 % |
| tout · nuit | 1168 | 1.25 | 1.61 | 93.4 % | 34.2 % |

## Fuite de `indice_global`

- Une seule valeur par créneau : True.
- Corrélation avec la moyenne des cibles du même créneau : **r = 0.9955**.
- Reconstruction : `indice_global ≈ 1.075 × moyenne(cible) -0.227`, résidu max 3.15.
- Verdict : **fuite confirmée**, variable exclue du modèle servi.
- Version décalée (créneau précédent) : comparée ci-dessus.

## Couverture du P90

Cible : entre 85.0 % et 95.0 % sur le test réel. Voir la colonne « Couv. P90 réel ». Sur 816 points testés, chaque point couvert ou non déplace la couverture de 0.12 points.

## Importance des features

Modèle `glm_poisson` final, permutation par groupe de colonnes (30 répétitions), hausse de MAE sur les lignes réel. **Calculé en échantillon** (le modèle a vu ces lignes) : indicatif.

| Groupe | Hausse de MAE | Écart-type |
|---|---|---|
| heure | 1.565 | 0.024 |
| zone | 1.186 | 0.026 |
| is_winter | 1.044 | 0.019 |
| is_storm | 0.232 | 0.008 |
| is_daylight | 0.122 | 0.006 |
| saison | 0.000 | 0.000 |
| is_weekend | 0.000 | 0.000 |

## Décision

**Modèle accepté** (candidat évalué : `glm_poisson`).

- `glm_poisson` : gain de MAE réel 45.5 % (IC 95 % [28.4 % ; 58.4 %]), couverture P90 90.0 %, 105 journées : critère satisfait.
- `lightgbm_poisson` (non retenu, plus complexe ou refusé) : critère satisfait aussi
