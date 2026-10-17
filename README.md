# Noctura

Backend REST indépendant pour préparer les données, entraîner explicitement un
modèle de fréquentation, produire des prédictions et expliquer les
recommandations d'éclairage. L'API ne commande aucun équipement.

## Installation et démarrage

Python 3.10 ou supérieur est requis.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn noctura_api.api:app --reload
```

La documentation interactive est disponible sur <http://127.0.0.1:8000/docs> et
le schéma OpenAPI JSON sur <http://127.0.0.1:8000/openapi.json>.

Le stockage par défaut est `storage/` (ignoré par Git). Il peut être déplacé en
définissant `NOCTURA_STORAGE_DIR`. Les origines CORS par défaut sont
`http://localhost:3000` et `http://localhost:5173`; configurez
`NOCTURA_CORS_ORIGINS` avec une liste séparée par des virgules pour le frontend.

L'application cible un seul site : le port de Porquerolles, à Hyères
(43.0024397, 6.2006567; fuseau `Europe/Paris`). Les zones de fréquentation et
d'énergie sont des sous-zones de ce même port, pas des ports séparés. Le dépôt
inclut des jeux de fréquentation, biodiversité et énergie simulée. Les imports
valides sont conservés dans `storage/datasets/` et ne remplacent jamais le
modèle déjà actif.

## Flux d'utilisation

1. Consultez `GET /api/v1/datasets/status`.
2. Importez les nouveaux CSV avec les endpoints dataset ci-dessous, si besoin.
3. Envoyez des mesures observées à `POST /api/v1/datasets/weather`. Les clés de
   mesure sont conservées telles quelles; aucune mesure n'est fabriquée.
4. Lancez explicitement `POST /api/v1/ml/train`. Le traitement est synchrone car
   le modèle de départ est léger. Consultez le job, le modèle et ses évaluations.
5. Demandez une prédiction ou une recommandation. Une prédiction de
   fréquentation n'est disponible qu'après un entraînement réussi; sans modèle,
   l'API répond `409 model_not_trained`.

L'import des CSV et les prédictions ne déclenchent jamais l'entraînement. Le
modèle est une régression par arbre de décision; l'entraînement exclut les lignes
marquées `is_synthetic_daytime=1`. Il refuse les jeux de moins de 20 lignes
non synthétiques. L'évaluation réserve des groupes complets de caractéristiques
au holdout pour éviter de répartir la même combinaison entre entraînement et
test. Le CSV fréquentation utilisé pour Noctora n'a pas de dates; cette
séparation ne valide donc pas les performances sur des journées futures.
Le CSV `data/eclairage_historique_simule.csv` contient bien des dates pour
chaque jour de 2025, mais c'est un autre jeu : ses valeurs de fréquentation
sont simulées et ne sont pas les cibles observées `frequentation_target` du
modèle Noctora. Ces dates ne peuvent donc pas servir à mesurer la précision de
Noctora sur de futures fréquentations observées. Elles servent à l'analyse
temporelle de la simulation énergétique.

Guide autonome à transmettre à l'équipe frontend : [README_FRONTEND.md](./README_FRONTEND.md).

## Endpoints

Tous les endpoints sont préfixés par `/api/v1`. Les erreurs ont la forme
`{"error":{"code":"...","message":"...","details":[]}}`.
Les succès renvoient HTTP 200. Les imports de fichiers peuvent renvoyer 413
(fichier trop volumineux), 415 (extension incorrecte) ou 422 (CSV invalide);
une requête JSON invalide renvoie 422 `invalid_request`. Un entraînement avec
des données insuffisantes renvoie 409 `training_rejected` et un `job_id`.
Un identifiant de job inconnu renvoie 404 `job_not_found`. Les erreurs
inattendues renvoient 500 `internal_error` et sont consignées côté serveur.

| Méthode | Chemin | Description |
|---|---|---|
| GET | `/health` | Vérification de disponibilité |
| GET | `/port` | Nom, position unique du port et zones pour le frontend |
| POST | `/datasets/weather` | Enregistrement de mesures météo/astro JSON |
| POST | `/datasets/biodiversity` | Import CSV multipart |
| POST | `/datasets/attendance` | Import CSV multipart |
| GET | `/datasets/status` | Disponibilité, qualité et date des sources |
| GET | `/datasets/zones` | Liste des zones reconnues du port |
| POST | `/ml/train` | Entraînement explicitement demandé (200, job terminé ou 409 si rejeté) |
| GET | `/ml/train/{job_id}` | Statut et métriques du job |
| GET | `/ml/model` | Version et métriques du modèle actif |
| GET | `/ml/evaluations` | Historique des entraînements réussis |
| POST | `/predictions/attendance` | Prédiction de fréquentation |
| GET | `/predictions/attendance/today?zone=quai_est&day=today` | Prévisions météo et prédictions des créneaux restants; `day=today` ou `day=tomorrow`, zone facultative |
| GET | `/energy/forecast?day=today` | Estimation simulée de l'énergie, économie potentielle et classement des zones |
| GET | `/energy/forecast?day=tomorrow` | Même résultat pour demain |
| POST | `/lighting/recommendations` | Recommandation simulée et justifiée |
| GET | `/lighting/recommendations` | Historique filtrable par date, zone et créneau |
| POST | `/lighting/decisions/generate?zone=quai_est&day=today` | Génère les propositions simulées pour aujourd'hui ou demain (zone facultative) |
| GET | `/lighting/decisions?date=YYYY-MM-DD&zone=quai_est&status=pending_human_review` | Liste filtrable pour le frontend |
| POST | `/lighting/decisions/{decision_id}/review` | Accepte/refuse une proposition en simulation |
| GET | `/reports/decisions` | Rapport des décisions, mêmes filtres |
| GET | `/reports/energy` | État des mesures réelles et lien vers l'estimation simulée |
| GET | `/reports/biodiversity` | Indicateurs descriptifs et état des contraintes |

## Guide frontend : catalogue complet des API

Adresse locale de base :

```js
const API = "http://127.0.0.1:8000/api/v1";
```

En production, remplacez cette adresse par celle du serveur. Le tableau
d'introduction présente des chemins relatifs; le catalogue ci-dessous donne les
routes complètes. Les réponses sont en JSON, sauf les imports CSV. Les routes
sont actuellement sans authentification :
ne les exposez pas directement sur Internet sans ajouter une authentification
et des autorisations.

### Toutes les routes

| Méthode et route complète | Paramètres / corps envoyés par le frontend | Ce que renvoie l'API et quand l'utiliser |
|---|---|---|
| `GET /api/v1/health` | Aucun | `{status:"ok"}`. Vérifier que le serveur répond. |
| `GET /api/v1/port` | Aucun | Nom, latitude, longitude, fuseau et liste des zones du port de Porquerolles. Pour initialiser l'en-tête et le sélecteur de zones. |
| `GET /api/v1/datasets/zones` | Aucun | `{count, items}` avec l'identifiant et le groupe de chaque zone. Pour remplir un menu de zones. |
| `GET /api/v1/datasets/status` | Aucun | Disponibilité, qualité, origine et date de mise à jour des données météo, fréquentation et biodiversité. |
| `POST /api/v1/datasets/weather` | JSON : `measurements` obligatoire et non vide; `source` (`weather_api` ou `manual`) et `timestamp` optionnels. Si fourni, `timestamp` doit avoir un fuseau horaire. | Enregistre des mesures météo fournies par l'appelant. Ce n'est pas requis par la prévision automatique Open-Meteo. |
| `POST /api/v1/datasets/attendance` | `multipart/form-data`, champ fichier nommé `file`, CSV de fréquentation. | Statut de validation et nombre de lignes/synthétiques. Utilisé pour fournir un jeu de données, pas pour lancer l'entraînement. |
| `POST /api/v1/datasets/biodiversity` | `multipart/form-data`, champ fichier nommé `file`, CSV biodiversité. | Statut de validation, nombre de lignes et d'espèces. |
| `POST /api/v1/ml/train` | Aucun | Lance explicitement l'entraînement; renvoie le job, sa version, ses métriques et ses nombres de lignes. Ne pas appeler à chaque affichage. |
| `GET /api/v1/ml/train/{job_id}` | Remplacer `{job_id}` par l'identifiant reçu à l'entraînement. | État et résultat d'un entraînement. HTTP 404 si le job n'existe pas. |
| `GET /api/v1/ml/model` | Aucun | Modèle actif `ready`, ou `not_trained` si aucun modèle n'a encore été entraîné. Contient les métriques quand le modèle existe. |
| `GET /api/v1/ml/evaluations` | Aucun | `{count, items}` : historique des entraînements réussis, métriques et limites de validation. |
| `POST /api/v1/predictions/attendance` | JSON ci-dessous. | Une estimation de fréquentation pour une zone et un créneau de deux heures. |
| `GET /api/v1/predictions/attendance/today` | `zone` facultatif; `day=today` (défaut) ou `day=tomorrow`. | Prévision météo Open-Meteo à Porquerolles et fréquentation estimée des créneaux futurs de la date choisie. Sans zone, toutes les zones. Requiert un modèle entraîné et Internet. |
| `POST /api/v1/lighting/recommendations` | JSON de prédiction ci-dessous; `timestamp` optionnel. | Enregistre une recommandation explicative, toujours avec intensité `null` et `pending_validation`. Pas une commande d'éclairage. |
| `GET /api/v1/lighting/recommendations` | Filtres facultatifs `date=YYYY-MM-DD`, `zone`, `time_slot=14h-16h`. | Historique des recommandations enregistrées : `{count, items}`. |
| `POST /api/v1/lighting/decisions/generate` | `zone` facultatif et `day=today` (défaut) ou `day=tomorrow`. | Propositions de démonstration : hausse/baisse/maintien, luminosité de référence/proposée, raisons et statut en attente de revue humaine. Requiert modèle entraîné et accès météo. Une requête aujourd'hui sans créneau restant renvoie `409 no_future_slots`; utilisez `day=tomorrow`. |
| `GET /api/v1/lighting/decisions` | Filtres facultatifs `date=YYYY-MM-DD`, `zone`, `status`. | Propositions sauvegardées : `{count, items}`. |
| `POST /api/v1/lighting/decisions/{decision_id}/review` | Remplacer `{decision_id}`; corps JSON de revue ci-dessous. | Enregistre l'acceptation ou le refus. L'approbation reste simulée; `hardware_command_sent` vaut toujours `false`. |
| `GET /api/v1/reports/decisions` | Filtres facultatifs `date`, `zone`, `time_slot`. | Rapport des recommandations enregistrées via `/lighting/recommendations` : `{count, items}`. |
| `GET /api/v1/energy/forecast` | `day=today` (défaut) ou `day=tomorrow`. | Prévision simulée d'énergie, intervalles Monte-Carlo, économies potentielles, zones les plus consommatrices, métriques Noctora et rapport des recommandations enregistrées pour cette date. |
| `GET /api/v1/reports/energy` | Aucun | État des compteurs réels et lien vers la prévision simulée. Les mesures réelles sont actuellement `null`. |
| `GET /api/v1/reports/biodiversity` | Aucun | Indicateurs descriptifs de biodiversité et état des contraintes écologiques. |

Les prédictions manuelles et recommandations vérifient que la zone et son groupe
correspondent au port. Le modèle doit être entraîné (sinon `409 model_not_trained`)
et une recommandation manuelle doit concerner un créneau futur à Porquerolles
(sinon `422 slot_in_past`). Les erreurs sont dans `error.code` et
`error.message`, pour permettre au frontend d'afficher un message clair.

### Corps JSON à envoyer

Prédiction ou recommandation de créneau (l'heure doit être paire, de 0 à 22;
les indicateurs sont 0 ou 1) :

```json
{
  "hour_start": 20,
  "zone_group_id": "G1",
  "zone_specifique": "quai_est",
  "is_daylight": 0,
  "is_winter": 0,
  "is_storm": 0,
  "indice_global": 6
}
```

Revue d'une proposition :

```json
{
  "action": "approve",
  "reviewer": "responsable",
  "note": "Vérification de démonstration"
}
```

`action` doit être `approve` ou `reject`; `reviewer` est obligatoire et la note
est optionnelle. Le champ `reviewer` est un nom déclaré, pas une authentification.

### Appels JavaScript prêts à adapter

Fonction utilitaire pour envoyer les appels JSON et afficher les erreurs API :

```js
async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(`${response.status}: ${data.error?.message ?? "Erreur API"}`);
  }
  return data;
}
```

Charger les infos du port et afficher le résumé énergie pour demain :

```js
const [port, forecast] = await Promise.all([
  api("/port"),
  api("/energy/forecast?day=tomorrow"),
]);
console.log(port.zones); // Options du menu zone
console.log(forecast.summary); // kWh, économie et intervalles
console.log(forecast.top_consuming_zones); // Zones à afficher en priorité
console.log(forecast.monte_carlo.simulations); // Nombre de tirages : 10000
console.log(forecast.model_performance); // Modèle Noctora de fréquentation
console.log(forecast.decision_report); // Recommandations enregistrées pour demain
```

Générer et examiner les propositions de démonstration :

```js
const proposalSet = await api(
  "/lighting/decisions/generate?zone=quai_est",
  { method: "POST" }
);
const proposal = proposalSet.items[0];

const decisionList = await api(
  "/lighting/decisions?date=2026-10-17&zone=quai_est&status=pending_human_review"
);

const reviewed = await api(
  `/lighting/decisions/${proposal.decision_id}/review`,
  {
    method: "POST",
    body: JSON.stringify({
      action: "approve",
      reviewer: "responsable",
      note: "Validation de démonstration",
    }),
  }
);
console.log(reviewed.hardware_command_sent); // Toujours false
```

Importer un CSV avec `fetch` et `FormData` (ne pas fixer manuellement
`Content-Type`, le navigateur ajoute la frontière multipart) :

```js
const form = new FormData();
form.append("file", fileInput.files[0]);
const response = await fetch(`${API}/datasets/attendance`, {
  method: "POST",
  body: form,
});
const result = await response.json();
```

### Comment lire les statuts et les erreurs

- `estimated_simulation` : chiffre énergie simulé, **pas** une mesure de compteur.
- `model_performance.status: "ready"` : les métriques MAE/RMSE/R² concernent
  la fréquentation, pas la précision énergétique. `not_trained` signifie
  qu'aucun modèle actif n'est disponible.
- `pending_human_review`, `approved_simulation`, `rejected` : cycle de revue
  des propositions. Même `approved_simulation` ne commande aucun matériel.
- `decision_report.status: "no_decisions_recorded"` : aucune recommandation
  enregistrée via `/lighting/recommendations` pour la date; ce n'est pas une
  erreur et ce rapport est distinct des propositions `/lighting/decisions`.
- Les erreurs ont la forme `{"error":{"code":"...","message":"..."}}`.
  Codes HTTP utiles : `422` requête invalide, zone inconnue, groupe incompatible
  ou créneau passé,
  `404` identifiant introuvable, `409` modèle non entraîné, absence de créneau
  futur ou proposition déjà revue, `413` fichier trop gros, `415` format fichier
  incorrect, `502` météo indisponible, `503` données de simulation
  indisponibles, `500` erreur serveur.
- En développement, le serveur autorise CORS depuis `http://localhost:3000` et
  `http://localhost:5173`. Configurez `NOCTURA_CORS_ORIGINS` pour les origines
  du frontend déployé.

Swagger interactif, qui contient aussi les schémas JSON et les réponses :
<http://127.0.0.1:8000/docs>. Le schéma brut destiné aux outils frontend est :
<http://127.0.0.1:8000/openapi.json>.

### Import météo

`POST /datasets/weather` accepte du JSON, par exemple :

```json
{
  "source": "weather_api",
  "timestamp": "2026-10-17T12:00:00+02:00",
  "measurements": {
    "cloud_cover_percent": 60,
    "natural_light_factor_percent": 42,
    "visibility_m": 8000
  }
}
```

`source` vaut `weather_api` ou `manual`. Les pourcentages de couverture
nuageuse et de lumière naturelle sont bornés entre 0 et 100.

### Import CSV

Les deux imports attendent un champ multipart nommé `file`, avec une extension
`.csv`, encodage UTF-8 et séparateur virgule.

- Fréquentation : les colonnes documentées dans `data/validate_clean.py`.
  Toutes les lignes sont vérifiées (types, créneaux, sin/cos, indicateurs,
  rattachements de zone et valeurs non négatives). Les doublons exacts sont
  comptés sur tout le fichier, même entre chunks; ils sont conservés par défaut
  et signalés comme avertissement. `--drop-duplicates` les retire globalement.
  Les lignes synthétiques restent
  identifiées par `is_synthetic_daytime` et sont exclues de l'entraînement.
- Biodiversité : `espece,nom_commun,nom_scientifique,taxon_indetermine,annee,effectif`.
  Les années/effectifs doivent être entiers, les effectifs non négatifs, le flag
  binaire, et chaque couple espèce/année unique.

Exemple PowerShell :

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/datasets/attendance `
  -F "file=@data/dataset_ml_frequentation_24h_clean.csv"
```

Un CSV invalide renvoie `422 dataset_validation_failed` avec les contrôles
échoués. Un fichier non-CSV renvoie `415`; un fichier de plus de 20 Mio renvoie
`413`.

### Entraînement manuel

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/ml/train
curl.exe http://127.0.0.1:8000/api/v1/ml/model
curl.exe http://127.0.0.1:8000/api/v1/ml/evaluations
```

La réponse du job donne son `job_id`, son état, les métriques holdout (MAE,
RMSE, R²), le nombre de lignes utilisées et le nombre de lignes synthétiques
écartées. Si un entraînement échoue, le dernier modèle actif n'est pas remplacé.
Un jeu insuffisant retourne HTTP `409` et un identifiant de job consultable.

L'évaluation réserve des **groupes entiers de caractéristiques** au holdout :
la même combinaison d'heure, zone, météo et indice ne peut pas se retrouver à la
fois en entraînement et en test. Le résultat inclut les métriques du modèle et
celles d'une référence simple qui prédit la moyenne de fréquentation du jeu
d'entraînement (`baseline_mae`, `baseline_rmse`, `baseline_r2`). MAE est
l'erreur absolue moyenne (personnes par créneau), RMSE
pénalise davantage les grandes erreurs et R² compare la variance expliquée à la
référence moyenne (il peut être négatif). Sur le jeu inclus, l'évaluation
reproductible donne MAE `1.6079`, RMSE `2.2171`, R² `0.6462`, contre MAE
`3.0835` et RMSE `3.8961` pour la référence moyenne. Ce résultat est meilleur
sur ce holdout seulement; il ne garantit pas une précision réelle. Le CSV de
fréquentation d'entraînement ne contient pas de date. Même si le CSV énergétique
contient des dates, sa fréquentation est simulée et ne fournit pas des cibles
indépendantes correspondantes; aucune validation temporelle de Noctora ou preuve
de performance en production n'est donc actuellement possible. Consultez
`model.validation` pour ces limites.

### Prédiction et recommandation

Les corps de prédiction et de recommandation utilisent `hour_start` (début d'un
créneau de deux heures, pair entre 0 et 22), `zone_group_id`, `zone_specifique`,
`is_daylight`, `is_winter`, `is_storm` (0/1) et `indice_global` :

```json
{
  "hour_start": 14,
  "zone_group_id": "G1",
  "zone_specifique": "quai_est",
  "is_daylight": 1,
  "is_winter": 0,
  "is_storm": 0,
  "indice_global": 11
}
```

Sans modèle actif, les routes de prédiction et de recommandation renvoient HTTP
`409 model_not_trained`, au lieu d'une réponse de succès contenant une valeur
nulle. Une requête pour une zone inconnue ou un groupe incompatible renvoie
`422`. Une recommandation manuelle concernant un créneau déjà commencé renvoie
`422 slot_in_past`. La recommandation ne fournit **jamais** d'intensité
opérationnelle (`recommended_intensity_percent: null`) : les seuils réglementaires,
les minima de sécurité et les contraintes écologiques doivent d'abord être
définis et validés. L'éclairage reste en attente de validation humaine et de
simulation.

`GET /lighting/recommendations?date=2026-10-17&zone=quai_est&time_slot=14h-16h`
filtre l'historique; chaque paramètre est facultatif.

Le rapport énergie renvoie `measured_energy_kwh: null` et
`estimated_savings_percent: null`, car aucune mesure électrique n'est fournie.
Le rapport biodiversité décrit les observations, sans les convertir en seuils
écologiques non validés.

## Exemples de tous les endpoints

Exécutez depuis la racine du dépôt dans PowerShell. Les imports de fichiers
utilisent ceux déjà inclus dans `data/`.

```powershell
$base = "http://127.0.0.1:8000/api/v1"

# Santé
curl.exe "$base/health"

# Sources : les CSV du dépôt sont disponibles comme jeux de départ
curl.exe "$base/datasets/status"

# Météo, importée depuis une source observée
curl.exe -X POST "$base/datasets/weather" -H "Content-Type: application/json" `
  -d '{"source":"weather_api","timestamp":"2026-10-17T12:00:00+02:00","measurements":{"cloud_cover_percent":60,"natural_light_factor_percent":42}}'

# Imports de fréquentation et de biodiversité (multipart/form-data)
curl.exe -X POST "$base/datasets/attendance" -F "file=@data/dataset_ml_frequentation_24h_clean.csv"
curl.exe -X POST "$base/datasets/biodiversity" -F "file=@data/biodev_1_tidy.csv"

# Contrôle/normalisation en streaming; le rapport compte les doublons
# globalement, même lorsqu'ils sont répartis entre plusieurs chunks.
python data/validate_clean.py data/dataset_ml_frequentation_24h.csv --chunksize 1000

# Entraînement explicite puis consultation du job, du modèle et des évaluations
$job = curl.exe -s -X POST "$base/ml/train" | ConvertFrom-Json
curl.exe "$base/ml/train/$($job.job_id)"
curl.exe "$base/ml/model"
curl.exe "$base/ml/evaluations"

# Prédiction manuelle; aucune commande d'éclairage n'est émise
$features = @{
  hour_start = 14; zone_group_id = "G1"; zone_specifique = "quai_est"
  is_daylight = 1; is_winter = 0; is_storm = 0; indice_global = 11
}
Invoke-RestMethod -Method Post -Uri "$base/predictions/attendance" `
  -ContentType "application/json" -Body ($features | ConvertTo-Json -Compress)

# Recommandation pour demain (créneau futur), distincte d'une décision simulée
$recommendation = $features.Clone()
$recommendation.timestamp = (Get-Date).AddDays(1).ToString("o")
Invoke-RestMethod -Method Post -Uri "$base/lighting/recommendations" `
  -ContentType "application/json" -Body ($recommendation | ConvertTo-Json -Compress)
Invoke-RestMethod "$base/lighting/recommendations?zone=quai_est"

# Zones et prévisions automatiques pour aujourd'hui.
# Sans zone, renvoie toutes les zones; avec zone, renvoie uniquement cette zone.
curl.exe "$base/datasets/zones"
curl.exe "$base/predictions/attendance/today"
curl.exe "$base/predictions/attendance/today?zone=quai_est"
curl.exe "$base/predictions/attendance/today?zone=quai_est&day=tomorrow"

# Rapports
curl.exe "$base/reports/decisions?zone=quai_est"
curl.exe "$base/reports/energy"
curl.exe "$base/reports/biodiversity"
```

La prédiction quotidienne récupère les prévisions horaires d'Open-Meteo pour
Porquerolles (aucune clé API nécessaire), puis prédit chaque créneau de deux heures
qui n'a pas encore commencé. `day=today` (défaut) ou `day=tomorrow` choisit la
date; `zone` est facultatif, omettez-le pour toutes les zones. La route garde
son ancien chemin `/today` pour compatibilité. Si aucun créneau ne reste
aujourd'hui, elle renvoie `status: "no_future_slots"` et une liste vide plutôt
qu'une prévision trompeuse. Les champs météo alimentent `is_daylight` et l'indicateur d'orage
(codes WMO 95/96/99). Comme Open-Meteo ne fournit pas `indice_global`, celui-ci
est renseigné avec la médiane du jeu existant pour la zone et le créneau; la
réponse indique si cette référence inclut des lignes synthétiques. Elle signale
aussi les heures sans données observées d'entraînement. La route échoue
explicitement si la météo ou une référence nécessaire manque.
Ces résultats restent des estimations de fréquentation, sans action automatique
ni intensité d'éclairage.

### Prévision énergétique pour le frontend

`GET /api/v1/energy/forecast?day=today` et `?day=tomorrow` renvoient un objet
compact avec `summary.predicted_consumption_kwh`,
`summary.full_power_reference_kwh`, `summary.potential_savings_kwh`,
`summary.potential_savings_percent`, leurs intervalles P05/P50/P95, la
probabilité de réaliser une économie, `top_consuming_zones` et le détail `zones`.
Le bloc `monte_carlo` indique le nombre de tirages, les dates historiques
distinctes utilisées et la méthode. Par défaut, le calcul réalise **10 000
tirages** bootstrap avec remise de profils journaliers comparables; ils ne
constituent pas 10 000 jours d'observations indépendants.
`GET /api/v1/port` expose les coordonnées uniques et les sous-zones.

**Important :** aucun compteur réel n'est encore connecté. Les prévisions sont
des simulations Monte-Carlo par bootstrap de profils journaliers du jeu
`data/eclairage_historique_simule.csv` (année 2025), appariés par mois et jour
de semaine. L'API fournit des quantiles P05/P50/P95 et la probabilité empirique
d'économie. L'économie est comparée à un scénario théorique où la puissance
installée fonctionne à 100 % pendant 24 heures; ce n'est ni une économie
mesurée ni une consigne d'exploitation. Le nombre de tirages est affiché dans
`monte_carlo.simulations`; le nombre de journées historiques distinctes est
dans `monte_carlo.historical_profiles_used`. Les tirages ne créent pas de
nouvelles observations : les intervalles restent indicatifs lorsque peu de
profils de référence sont disponibles.
`GET /api/v1/reports/energy` continue donc de renvoyer `measured_energy_kwh:
null` et distingue explicitement les mesures réelles de cette simulation.
Chaque réponse énergie inclut également `model_performance` pour le modèle de
fréquentation **Noctora** (MAE, RMSE, R², version et nombre de lignes), ainsi
que `decision_report` avec les recommandations enregistrées à cette date et leur
état. Ces métriques mesurent la tâche fréquentation, pas la précision de la
simulation énergétique; en l'absence de modèle entraîné, elles valent `null`.
Le rapport de décisions complet reste disponible séparément à
`GET /api/v1/reports/decisions?date=YYYY-MM-DD`.

Exemples PowerShell :

```powershell
$base = "http://127.0.0.1:8000/api/v1"
Invoke-RestMethod "$base/port" | ConvertTo-Json -Depth 5
Invoke-RestMethod "$base/energy/forecast?day=today" | ConvertTo-Json -Depth 6
Invoke-RestMethod "$base/energy/forecast?day=tomorrow" | ConvertTo-Json -Depth 6
```

Les clients frontend peuvent afficher directement `summary` pour les cartes
principales, puis `top_consuming_zones` pour le classement. Toujours afficher
`status: "estimated_simulation"` et les `warnings` avec les valeurs.

### Propositions de décision d'éclairage (mode démonstration)

Le frontend peut demander les propositions du jour :

```powershell
$decisions = Invoke-RestMethod `
  -Method Post "$base/lighting/decisions/generate?zone=quai_est"
$decisions | ConvertTo-Json -Depth 8
```

Sans `zone`, l'API génère des propositions pour toutes les zones où une prévision
est disponible. Chaque proposition donne `action` (`increase`, `decrease`,
`maintain`), les luminosités historique et proposée, `change_percentage_points`,
`reason`, `reason_codes` et `status: "pending_human_review"`. Les règles de
démonstration sont visibles dans `rules` : au moins 15 personnes prévoit une
hausse de 10 points, au plus 5 une baisse de 10 points, et entre les deux le
maintien. Les niveaux sont bornés entre 0 et 100 %.

Lister ou filtrer les propositions sauvegardées :

```powershell
Invoke-RestMethod "$base/lighting/decisions?date=2026-10-17&zone=quai_est&status=pending_human_review" |
  ConvertTo-Json -Depth 8
```

Le frontend peut demander une revue humaine en envoyant `reviewer` et, si
souhaité, une note :

```powershell
$decisionId = $decisions.items[0].decision_id
$review = @{ action = "approve"; reviewer = "responsable"; note = "Validation de démonstration" }
Invoke-RestMethod -Method Post `
  -Uri "$base/lighting/decisions/$decisionId/review" `
  -ContentType "application/json" `
  -Body ($review | ConvertTo-Json -Compress)
```

`action` peut valoir `approve` ou `reject`; une décision examinée une deuxième
fois retourne HTTP 409. Une approbation donne `approved_simulation` uniquement.
**Aucune de ces routes n'envoie de commande à un luminaire.** Les seuils,
variations et niveaux de cette démo ne sont pas validés pour l'exploitation :
un pilotage réel devra attendre des règles de sécurité et d'écologie approuvées,
une authentification des utilisateurs et l'intégration sécurisée au contrôleur.

Réponses types :

```json
{"status":"not_trained","model":null}
```

Avant l'entraînement, une prédiction ou une recommandation renvoie HTTP `409`
avec `error.code: "model_not_trained"`. Une recommandation reste en simulation avec
`recommended_intensity_percent: null` et `status: "pending_validation"`.
Un import refusé renvoie HTTP 422 avec un code d'erreur et les validations
échouées. Une tentative d'entraînement avec trop peu de données non synthétiques
renvoie HTTP 409 et un `job_id` dont le statut est `failed`.

Principales formes de réponses JSON :

Les valeurs entre chevrons ci-dessous sont des placeholders documentaires;
elles sont calculées ou attribuées à l'exécution et ne sont pas des littéraux
à envoyer à l'API.

```json
{"status":"ok"}
```

```json
{"status":"validated","dataset":{"rows":96,"synthetic_rows":56,"warnings":[]}}
```

```json
{"sources":{"weather":{"available":false,"validation_status":"not_imported"},"attendance":{"available":true,"validation_status":"valid","quality":{"valid_rows":96,"invalid_rows":0,"synthetic_rows":56}}}}
```

```json
{"job_id":"<uuid>","status":"completed","metrics":{"mae":"<calculée>","rmse":"<calculée>","r2":"<calculé>","holdout_rows":"<calculé>"},"model_version":"<version>","training_rows":"<calculé>","excluded_synthetic_rows":"<calculé>"}
```

```json
{"status":"ready","model":{"version":"<version>","trained_at":"<ISO-8601>","target":"frequentation_target","metrics":{"mae":"<calculée>","rmse":"<calculée>","r2":"<calculé>","holdout_rows":"<calculé>"}}}
```

```json
{"count":1,"items":[{"job_id":"<uuid>","model_version":"<version>","metrics":{"mae":"<calculée>","rmse":"<calculée>","r2":"<calculé>","holdout_rows":"<calculé>"}}]}
```

```json
{"zone":"quai_est","time_slot":"14h-16h","predicted":"<prédite>","unit":"individuals_per_two_hour_slot","model_version":"<version>","status":"predicted","warnings":[]}
```

```json
{"zone":"quai_est","timestamp":"<ISO-8601>","recommended_intensity_percent":null,"status":"pending_validation","attendance":{"predicted":"<prédite>","time_slot":"14h-16h"},"environment":{"weather_source":"weather_api","natural_light_factor_percent":42,"cloud_cover_percent":60},"constraints":{"safety_check":"not_configured","ecological_check":"pending_validation","constraints_applied":[]},"explanation":["Les règles doivent être validées avant toute intensité opérationnelle."]}
```

Les endpoints de recommandations et de rapport de décisions renvoient
`{"count":0,"items":[]}` lorsqu'aucun résultat ne correspond aux filtres.
Le rapport énergie renvoie `status: "not_measured"` pour les relevés réels; la
simulation se récupère via `/api/v1/energy/forecast`. Le rapport biodiversité renvoie le nombre d'espèces,
les années disponibles et `constraints: "pending_validation"`.

## Tests

```powershell
python -m pytest -q
```
