# Rapport d'évaluation : fréquentation NOCTURA (DÉMO, DONNÉES SIMULÉES)

> **Données simulées par `ml/simulation.py`.** Ces scores mesurent la capacité du modèle à retrouver le générateur, pas le comportement réel du port. Le modèle produit est réservé à `PREDICTION_MODE=demo`.

Généré par `ml/train.py` le 2026-10-17T15:51:41+02:00 (graine 42). Tous les chiffres ci-dessous sont calculés par le script ; aucun n'est saisi à la main.

## Protocole

- Dataset : `data/dataset_simule.csv` (sha256 `311cee0de9d3…`), 70176 lignes : 0 réelles, 0 synthétiques, 70176 simulées.
- Population évaluée : lignes **SIMULÉ** (731 journée(s)).
- Mode de validation : **jours**. Les 146 dernières journées (2025-08-08 → 2025-12-31) forment le test, jamais vues ; calibration par GroupKFold par journée sur les précédentes.
- Hyperparamètres et quantile P90 (conformal sur résidus normalisés `(y-μ)/√μ`) choisis par validation croisée interne groupée, dans chaque pli externe.
- `indice_global` est exclu des modèles servables (fuite, voir plus bas) ; les variantes « (FUITE) » ne sont là que pour mesurer son effet.

## Comparaison des modèles (test hors échantillon)

| Modèle | Servable | MAE SIMULÉ | RMSE SIMULÉ | Couv. P90 SIMULÉ | MAE reel | RMSE reel | Gain MAE vs baseline |
|---|---|---|---|---|---|---|---|
| baseline | oui | 4.51 | 6.05 | 94.9 % | — | — | — |
| glm_poisson | oui | 2.63 | 3.59 | 94.5 % | — | — | 41.6 % |
| lightgbm_poisson | oui | 2.46 | 3.45 | 91.2 % | — | — | 45.5 % |
| glm_poisson+indice_global (FUITE) | non | 1.96 | 2.66 | 90.2 % | — | — | 56.5 % |
| lightgbm_poisson+indice_global (FUITE) | non | 1.99 | 2.69 | 90.3 % | — | — | 55.9 % |
| glm_poisson+indice_decale | non | 2.44 | 3.29 | 92.2 % | — | — | 46.0 % |
| lightgbm_poisson+indice_decale | non | 2.42 | 3.35 | 89.8 % | — | — | 46.3 % |

IC 95 % du gain de MAE SIMULÉ (bootstrap sur les groupes indépendants) :

- glm_poisson : [36.3 % ; 46.4 %] sur 146 groupes
- lightgbm_poisson : [42.0 % ; 48.8 %] sur 146 groupes

## Détail par segment

### baseline

| Segment | n | MAE | RMSE | Couv. P90 | Sous-estimation P50 |
|---|---|---|---|---|---|
| simule | 14016 | 4.51 | 6.05 | 94.9 % | 31.4 % |
| simule · jour | 6400 | 6.13 | 7.82 | 91.5 % | 37.5 % |
| simule · nuit | 7616 | 3.14 | 3.99 | 97.8 % | 26.3 % |
| simule · zone acces_village | 1752 | 5.24 | 6.89 | 94.3 % | 32.1 % |
| simule · zone capitainerie | 1752 | 3.26 | 4.34 | 96.2 % | 33.3 % |
| simule · zone entree_du_port | 1752 | 4.19 | 5.66 | 94.9 % | 31.8 % |
| simule · zone pontons_centraux | 1752 | 5.03 | 6.63 | 94.3 % | 32.1 % |
| simule · zone quai_est | 1752 | 3.97 | 5.18 | 95.3 % | 30.3 % |
| simule · zone quai_nord | 1752 | 3.51 | 4.61 | 96.3 % | 30.1 % |
| simule · zone quai_ouest | 1752 | 4.83 | 6.18 | 94.8 % | 30.2 % |
| simule · zone quai_sud_commerces | 1752 | 6.05 | 8.03 | 93.3 % | 31.6 % |
| tout · jour | 6400 | 6.13 | 7.82 | 91.5 % | 37.5 % |
| tout · nuit | 7616 | 3.14 | 3.99 | 97.8 % | 26.3 % |

### glm_poisson

| Segment | n | MAE | RMSE | Couv. P90 | Sous-estimation P50 |
|---|---|---|---|---|---|
| simule | 14016 | 2.63 | 3.59 | 94.5 % | 35.4 % |
| simule · jour | 6400 | 3.49 | 4.55 | 94.4 % | 35.9 % |
| simule · nuit | 7616 | 1.91 | 2.51 | 94.7 % | 35.0 % |
| simule · zone acces_village | 1752 | 2.97 | 3.99 | 94.0 % | 34.9 % |
| simule · zone capitainerie | 1752 | 2.14 | 2.88 | 93.7 % | 37.7 % |
| simule · zone entree_du_port | 1752 | 2.47 | 3.35 | 94.7 % | 35.6 % |
| simule · zone pontons_centraux | 1752 | 2.79 | 3.79 | 95.1 % | 34.0 % |
| simule · zone quai_est | 1752 | 2.31 | 3.13 | 95.3 % | 35.8 % |
| simule · zone quai_nord | 1752 | 2.23 | 2.99 | 95.5 % | 34.5 % |
| simule · zone quai_ouest | 1752 | 2.75 | 3.67 | 94.3 % | 35.9 % |
| simule · zone quai_sud_commerces | 1752 | 3.39 | 4.59 | 93.7 % | 35.3 % |
| tout · jour | 6400 | 3.49 | 4.55 | 94.4 % | 35.9 % |
| tout · nuit | 7616 | 1.91 | 2.51 | 94.7 % | 35.0 % |

### lightgbm_poisson

| Segment | n | MAE | RMSE | Couv. P90 | Sous-estimation P50 |
|---|---|---|---|---|---|
| simule | 14016 | 2.46 | 3.45 | 91.2 % | 40.1 % |
| simule · jour | 6400 | 3.27 | 4.41 | 89.3 % | 42.4 % |
| simule · nuit | 7616 | 1.78 | 2.35 | 92.7 % | 38.2 % |
| simule · zone acces_village | 1752 | 2.81 | 4.06 | 87.0 % | 47.1 % |
| simule · zone capitainerie | 1752 | 2.05 | 2.68 | 95.3 % | 32.9 % |
| simule · zone entree_du_port | 1752 | 2.38 | 3.22 | 95.1 % | 32.5 % |
| simule · zone pontons_centraux | 1752 | 2.58 | 3.73 | 88.2 % | 45.5 % |
| simule · zone quai_est | 1752 | 2.26 | 3.03 | 95.7 % | 32.5 % |
| simule · zone quai_nord | 1752 | 2.16 | 2.88 | 95.6 % | 30.1 % |
| simule · zone quai_ouest | 1752 | 2.51 | 3.43 | 86.4 % | 50.5 % |
| simule · zone quai_sud_commerces | 1752 | 2.92 | 4.24 | 86.0 % | 49.7 % |
| tout · jour | 6400 | 3.27 | 4.41 | 89.3 % | 42.4 % |
| tout · nuit | 7616 | 1.78 | 2.35 | 92.7 % | 38.2 % |

### glm_poisson+indice_global (FUITE)

| Segment | n | MAE | RMSE | Couv. P90 | Sous-estimation P50 |
|---|---|---|---|---|---|
| simule | 14016 | 1.96 | 2.66 | 90.2 % | 46.9 % |
| simule · jour | 6400 | 2.52 | 3.30 | 89.9 % | 48.4 % |
| simule · nuit | 7616 | 1.49 | 1.97 | 90.4 % | 45.6 % |
| simule · zone acces_village | 1752 | 2.15 | 2.92 | 90.4 % | 51.8 % |
| simule · zone capitainerie | 1752 | 1.76 | 2.36 | 89.2 % | 46.6 % |
| simule · zone entree_du_port | 1752 | 1.86 | 2.51 | 91.2 % | 46.3 % |
| simule · zone pontons_centraux | 1752 | 2.04 | 2.75 | 91.1 % | 44.7 % |
| simule · zone quai_est | 1752 | 1.78 | 2.39 | 89.0 % | 46.7 % |
| simule · zone quai_nord | 1752 | 1.72 | 2.33 | 90.7 % | 43.2 % |
| simule · zone quai_ouest | 1752 | 2.08 | 2.75 | 89.7 % | 48.6 % |
| simule · zone quai_sud_commerces | 1752 | 2.31 | 3.15 | 90.3 % | 47.1 % |
| tout · jour | 6400 | 2.52 | 3.30 | 89.9 % | 48.4 % |
| tout · nuit | 7616 | 1.49 | 1.97 | 90.4 % | 45.6 % |

### lightgbm_poisson+indice_global (FUITE)

| Segment | n | MAE | RMSE | Couv. P90 | Sous-estimation P50 |
|---|---|---|---|---|---|
| simule | 14016 | 1.99 | 2.69 | 90.3 % | 45.0 % |
| simule · jour | 6400 | 2.55 | 3.34 | 89.8 % | 46.8 % |
| simule · nuit | 7616 | 1.52 | 1.99 | 90.8 % | 43.4 % |
| simule · zone acces_village | 1752 | 2.21 | 3.03 | 87.6 % | 54.7 % |
| simule · zone capitainerie | 1752 | 1.77 | 2.34 | 91.6 % | 39.2 % |
| simule · zone entree_du_port | 1752 | 1.91 | 2.56 | 93.6 % | 40.6 % |
| simule · zone pontons_centraux | 1752 | 2.07 | 2.83 | 88.1 % | 45.6 % |
| simule · zone quai_est | 1752 | 1.82 | 2.42 | 94.2 % | 38.2 % |
| simule · zone quai_nord | 1752 | 1.77 | 2.37 | 92.4 % | 38.1 % |
| simule · zone quai_ouest | 1752 | 2.04 | 2.71 | 87.8 % | 49.9 % |
| simule · zone quai_sud_commerces | 1752 | 2.31 | 3.17 | 87.3 % | 53.6 % |
| tout · jour | 6400 | 2.55 | 3.34 | 89.8 % | 46.8 % |
| tout · nuit | 7616 | 1.52 | 1.99 | 90.8 % | 43.4 % |

### glm_poisson+indice_decale

| Segment | n | MAE | RMSE | Couv. P90 | Sous-estimation P50 |
|---|---|---|---|---|---|
| simule | 14016 | 2.44 | 3.29 | 92.2 % | 40.5 % |
| simule · jour | 6400 | 3.04 | 3.96 | 92.4 % | 42.2 % |
| simule · nuit | 7616 | 1.92 | 2.60 | 91.9 % | 39.1 % |
| simule · zone acces_village | 1752 | 2.73 | 3.64 | 90.9 % | 39.7 % |
| simule · zone capitainerie | 1752 | 2.00 | 2.68 | 91.4 % | 42.7 % |
| simule · zone entree_du_port | 1752 | 2.31 | 3.10 | 92.2 % | 41.4 % |
| simule · zone pontons_centraux | 1752 | 2.59 | 3.45 | 93.1 % | 39.0 % |
| simule · zone quai_est | 1752 | 2.13 | 2.88 | 93.5 % | 41.2 % |
| simule · zone quai_nord | 1752 | 2.06 | 2.76 | 93.6 % | 39.9 % |
| simule · zone quai_ouest | 1752 | 2.57 | 3.36 | 92.0 % | 40.7 % |
| simule · zone quai_sud_commerces | 1752 | 3.11 | 4.20 | 90.6 % | 39.6 % |
| tout · jour | 6400 | 3.04 | 3.96 | 92.4 % | 42.2 % |
| tout · nuit | 7616 | 1.92 | 2.60 | 91.9 % | 39.1 % |

### lightgbm_poisson+indice_decale

| Segment | n | MAE | RMSE | Couv. P90 | Sous-estimation P50 |
|---|---|---|---|---|---|
| simule | 14016 | 2.42 | 3.35 | 89.8 % | 43.8 % |
| simule · jour | 6400 | 3.07 | 4.08 | 89.0 % | 46.0 % |
| simule · nuit | 7616 | 1.87 | 2.59 | 90.3 % | 42.0 % |
| simule · zone acces_village | 1752 | 2.72 | 3.83 | 85.2 % | 50.7 % |
| simule · zone capitainerie | 1752 | 2.07 | 2.75 | 94.1 % | 35.6 % |
| simule · zone entree_du_port | 1752 | 2.34 | 3.15 | 93.3 % | 38.1 % |
| simule · zone pontons_centraux | 1752 | 2.56 | 3.60 | 87.2 % | 48.8 % |
| simule · zone quai_est | 1752 | 2.16 | 2.92 | 93.7 % | 36.6 % |
| simule · zone quai_nord | 1752 | 2.07 | 2.79 | 93.9 % | 35.9 % |
| simule · zone quai_ouest | 1752 | 2.49 | 3.34 | 85.8 % | 52.1 % |
| simule · zone quai_sud_commerces | 1752 | 2.96 | 4.17 | 84.8 % | 52.6 % |
| tout · jour | 6400 | 3.07 | 4.08 | 89.0 % | 46.0 % |
| tout · nuit | 7616 | 1.87 | 2.59 | 90.3 % | 42.0 % |

## Fuite de `indice_global`

- Une seule valeur par créneau : True.
- Corrélation avec la moyenne des cibles du même créneau : **r = 0.9993**.
- Reconstruction : `indice_global ≈ 1.082 × moyenne(cible) -0.361`, résidu max 0.50.
- Verdict : **fuite confirmée**, variable exclue du modèle servi.
- Version décalée (créneau précédent) : comparée ci-dessus.

## Couverture du P90

Cible : entre 85.0 % et 95.0 % sur le test SIMULÉ. Voir la colonne « Couv. P90 SIMULÉ ». Sur 14016 points testés, chaque point couvert ou non déplace la couverture de 0.01 points.

## Importance des features

Modèle `glm_poisson` final, permutation par groupe de colonnes (30 répétitions), hausse de MAE sur les lignes SIMULÉ. **Calculé en échantillon** (le modèle a vu ces lignes) : indicatif.

| Groupe | Hausse de MAE | Écart-type |
|---|---|---|
| saison | 4.560 | 0.037 |
| heure | 1.317 | 0.019 |
| is_winter | 1.153 | 0.022 |
| zone | 0.752 | 0.014 |
| is_daylight | 0.647 | 0.015 |
| is_storm | 0.260 | 0.008 |
| is_weekend | 0.130 | 0.005 |

## Décision

**Modèle accepté** (candidat évalué : `glm_poisson`).

- `glm_poisson` : gain de MAE SIMULÉ 41.6 % (IC 95 % [36.3 % ; 46.4 %]), couverture P90 94.5 %, 731 journées : critère satisfait.
- `lightgbm_poisson` (non retenu, plus complexe ou refusé) : critère satisfait aussi
