# Guide frontend Noctura

Ce document explique comment connecter le frontend du hackathon au backend
Noctura. Il décrit les routes disponibles, les réponses attendues et les limites
à afficher dans l'interface.

## 1. Démarrer et vérifier le backend

Depuis la racine du dépôt, dans un terminal PowerShell :

```powershell
python -m uvicorn noctura_api.api:app --reload
```

Le serveur local est accessible à `http://127.0.0.1:8000`.

- Documentation interactive pour essayer les appels : <http://127.0.0.1:8000/docs>
- Schéma OpenAPI : <http://127.0.0.1:8000/openapi.json>
- Vérification rapide : <http://127.0.0.1:8000/api/v1/health>

L'adresse de base utilisée par le frontend est :

```js
const API_BASE = "http://127.0.0.1:8000/api/v1";
```

Le backend autorise par défaut les origines de développement
`http://localhost:3000` et `http://localhost:5173`. Si le serveur du frontend
utilise une autre origine, configurez `NOCTURA_CORS_ORIGINS` côté backend.

## 2. Parcours conseillé pour la démo

1. Charger les informations du port, les zones et l'état des jeux de données.
2. Entraîner le modèle une fois au démarrage de la démo si aucun modèle actif
   n'est disponible. L'entraînement est manuel, pas automatique à chaque appel.
3. Charger les prévisions de fréquentation et l'estimation énergétique du jour
   ou de demain.
4. Générer des propositions de décision d'éclairage, puis les afficher et
   éventuellement les approuver/refuser en mode simulation.
5. Afficher les avertissements et les états de simulation avec les résultats.

Le dépôt contient des données de départ. Il n'est normalement pas nécessaire
d'importer les CSV pour exécuter la démo.

## 3. Routes disponibles

Toutes les routes sont sous `/api/v1`. Le tableau contient les **22 routes**
actuellement exposées.

### Port et données

| # | Méthode et chemin | Usage frontend |
|---|---|---|
| 1 | `GET /health` | Vérifier que le backend répond; résultat `{ "status": "ok" }`. |
| 2 | `GET /port` | Charger le nom, les coordonnées et les zones du seul port supporté : Porquerolles. |
| 3 | `GET /datasets/zones` | Charger la liste des zones et leur groupe pour un menu de sélection. |
| 4 | `GET /datasets/status` | Afficher la disponibilité et la qualité des données fréquentation, météo et biodiversité. |
| 5 | `POST /datasets/weather` | Importer une mesure météo/astro JSON facultative; ne déclenche pas les prévisions automatiques. |
| 6 | `POST /datasets/attendance` | Importer un CSV de fréquentation via `multipart/form-data`, champ `file`. |
| 7 | `POST /datasets/biodiversity` | Importer un CSV biodiversité via `multipart/form-data`, champ `file`. |

### Modèle et fréquentation

| # | Méthode et chemin | Usage frontend |
|---|---|---|
| 8 | `POST /ml/train` | Lancer explicitement l'entraînement. Il est synchrone; ne pas l'appeler à chaque chargement d'écran. |
| 9 | `GET /ml/train/{job_id}` | Consulter le résultat du job obtenu lors de l'entraînement. |
| 10 | `GET /ml/model` | Consulter le modèle actif et son état (`ready` ou `not_trained`). |
| 11 | `GET /ml/evaluations` | Afficher les entraînements réussis, métriques et limites de validation. |
| 12 | `POST /predictions/attendance` | Demander une estimation manuelle pour une zone et un créneau de deux heures. |
| 13 | `GET /predictions/attendance/today` | Récupérer les estimations météo et de fréquentation des créneaux futurs. Paramètres : `day=today` (défaut) ou `day=tomorrow`, et `zone` optionnelle. Le chemin `/today` est conservé pour compatibilité, y compris avec `day=tomorrow`. |

### Éclairage et rapports

| # | Méthode et chemin | Usage frontend |
|---|---|---|
| 14 | `POST /lighting/recommendations` | Enregistrer une recommandation descriptive à partir des données fournies par l'appelant. |
| 15 | `GET /lighting/recommendations` | Lister ces recommandations. Filtres optionnels : `date`, `zone`, `time_slot`. |
| 16 | `POST /lighting/decisions/generate` | Générer des propositions d'éclairage simulées. Paramètres optionnels : `zone`, `day=today` ou `day=tomorrow`. |
| 17 | `GET /lighting/decisions` | Lister les propositions simulées. Filtres optionnels : `date`, `zone`, `status`. |
| 18 | `POST /lighting/decisions/{decision_id}/review` | Approuver ou refuser une proposition; cela ne commande jamais un luminaire. |
| 19 | `GET /reports/decisions` | Consulter le rapport des recommandations enregistrées, avec filtres `date`, `zone`, `time_slot`. |
| 20 | `GET /energy/forecast` | Charger l'estimation énergétique simulée et son détail par zone. Paramètre : `day=today` (défaut) ou `day=tomorrow`. |
| 21 | `GET /reports/energy` | Vérifier la disponibilité des mesures d'énergie réelles. Le jeu actuel ne contient pas de mesures de compteur réelles. |
| 22 | `GET /reports/biodiversity` | Consulter le résumé descriptif de biodiversité. Les contraintes écologiques restent à valider. |

## 4. Utilitaire JavaScript pour les appels

L'utilitaire ci-dessous parse le JSON et transforme les réponses HTTP d'erreur
en exceptions avec un code exploitable par l'interface.

```js
async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
  });

  const data = await response.json();
  if (!response.ok) {
    const error = new Error(
      data.error?.message ?? `Erreur API HTTP ${response.status}`
    );
    error.status = response.status;
    error.code = data.error?.code ?? "http_error";
    error.details = data.error?.details ?? [];
    throw error;
  }
  return data;
}
```

Exemple d'affichage d'erreur :

```js
try {
  const result = await api("/predictions/attendance/today?day=tomorrow");
  renderForecast(result);
} catch (error) {
  showError(`${error.code}: ${error.message}`);
}
```

## 5. Chargement initial de l'application

Ces appels peuvent être faits en parallèle :

```js
const [health, port, zones, datasets, model] = await Promise.all([
  api("/health"),
  api("/port"),
  api("/datasets/zones"),
  api("/datasets/status"),
  api("/ml/model"),
]);

console.log(port.name);            // Port de Porquerolles
console.log(port.location);        // Coordonnées et fuseau Europe/Paris
console.log(zones.items);          // Zones sélectionnables
console.log(datasets.sources);     // État des jeux de données
console.log(model.status);         // "ready" ou "not_trained"
```

Si `model.status` vaut `not_trained`, proposer à l'utilisateur autorisé de
lancer l'entraînement, ou le faire une fois pendant la préparation de la démo :

```js
const training = await api("/ml/train", { method: "POST" });
console.log(training.status, training.metrics, training.validation);
```

Les routes `POST /predictions/attendance`, `POST /lighting/recommendations`
et les prévisions du jour nécessitent un modèle entraîné. Sans modèle, elles
renvoient HTTP `409` avec `error.code: "model_not_trained"`.

## 6. Prévisions de fréquentation

### Prévision automatique aujourd'hui ou demain

Les prévisions automatiques récupèrent la météo horaire Open-Meteo pour
Porquerolles. Aucun appel frontend à Open-Meteo n'est requis. Le serveur backend
doit toutefois avoir accès à Internet.

```js
const today = await api("/predictions/attendance/today?zone=quai_est");
const tomorrow = await api(
  "/predictions/attendance/today?day=tomorrow&zone=quai_est"
);

console.log(today.date, today.timezone, today.items);
console.log(tomorrow.date, tomorrow.items);
```

Sans paramètre `zone`, la réponse inclut toutes les zones. Chaque élément de
`items` comprend notamment `zone`, `time_slot`, `predicted`, les variables
météo utilisées et des informations sur la couverture des données
d'entraînement. `predicted` est une estimation de personnes par créneau de deux
heures, pas un comptage réel.

Le modèle utilise `indice_global`, qui n'est pas fourni par Open-Meteo. L'API le
remplace par la médiane historique disponible et le signale dans
`features.indice_global_source` et dans `warnings`. Certains créneaux peuvent
s'appuyer sur des lignes synthétiques ou avoir peu de données observées.

Si aucun créneau futur ne reste aujourd'hui, la réponse utilise
`status: "no_future_slots"` et renvoie `items: []`; demander `day=tomorrow`.

### Prédiction manuelle

La route manuelle nécessite les champs suivants :

```js
const prediction = await api("/predictions/attendance", {
  method: "POST",
  body: JSON.stringify({
    hour_start: 20,
    zone_group_id: "G1",
    zone_specifique: "quai_est",
    is_daylight: 0,
    is_winter: 0,
    is_storm: 0,
    indice_global: 6,
  }),
});
```

`hour_start` est l'heure de début d'un créneau de deux heures, paire entre 0
et 22. Les trois indicateurs météo/saison sont des entiers `0` ou `1`;
`indice_global` est un entier non négatif. La zone et le groupe doivent
correspondre; par exemple `quai_est` appartient à `G1`. La réponse comprend
`predicted`, `unit`, `model_version` et `warnings`.

## 7. Énergie : simulation seulement

```js
const energyToday = await api("/energy/forecast?day=today");
const energyTomorrow = await api("/energy/forecast?day=tomorrow");

console.log(energyToday.status);                 // "estimated_simulation"
console.log(energyToday.summary);                // Résumé kWh et économie potentielle
console.log(energyToday.top_consuming_zones);    // Trois plus fortes consommations simulées
console.log(energyToday.zones);                  // Détail de toutes les zones
console.log(energyToday.monte_carlo);            // Tirages et journées historiques distinctes
console.log(energyToday.model_performance);      // Métriques de fréquentation Noctora
```

L'énergie est estimée par bootstrap de profils journaliers du fichier
`data/eclairage_historique_simule.csv`, qui couvre 2025. Les 10 000 tirages
Monte-Carlo rééchantillonnent ces profils; ils ne correspondent pas à 10 000
journées mesurées. Les économies sont comparées à une référence théorique de
puissance installée à 100 % pendant 24 heures. Ce ne sont ni des économies
mesurées ni garanties.

La réponse contient les estimations et intervalles par zone directement dans
`zones`; l'API ne propose pas de filtre `zone` pour cette route. Pour afficher
une zone sélectionnée, filtrez côté frontend, par exemple :

```js
const selectedZone = energyToday.zones.find((item) => item.zone === "quai_est");
```

`model_performance` décrit le modèle de fréquentation, pas la précision de la
simulation énergétique.

## 8. Recommandations et propositions d'éclairage : deux choses distinctes

### Recommandation descriptive enregistrée

`POST /lighting/recommendations` enregistre une recommandation à partir des
caractéristiques fournies, mais ne calcule pas une intensité d'éclairage. Son
corps accepte les champs de prédiction et un `timestamp` optionnel. Pour éviter
de viser un créneau passé, fournir un timestamp ISO-8601 du jour visé et choisir
un `hour_start` qui n'a pas encore commencé dans le fuseau `Europe/Paris`.

```js
const tomorrowAt20 = new Date();
tomorrowAt20.setDate(tomorrowAt20.getDate() + 1);

const recommendation = await api("/lighting/recommendations", {
  method: "POST",
  body: JSON.stringify({
    hour_start: 20,
    zone_group_id: "G1",
    zone_specifique: "quai_est",
    is_daylight: 0,
    is_winter: 0,
    is_storm: 0,
    indice_global: 6,
    timestamp: tomorrowAt20.toISOString(),
  }),
});
```

Le serveur renvoie l'état `pending_validation`,
`recommended_intensity_percent: null`, les contrôles de sécurité/écologie, les
champs météo dernièrement importés (éventuellement `null`) et des explications.
Cette route n'appelle pas elle-même Open-Meteo. Les recommandations enregistrées
se consultent avec :

```text
GET /api/v1/lighting/recommendations?zone=quai_est
GET /api/v1/lighting/recommendations?date=2026-10-18&zone=quai_est&time_slot=20h-22h
```

### Proposition simulée à examiner

Les propositions calculées par le système utilisent la prévision automatique
et les profils d'éclairage historiques simulés :

```js
const generated = await api(
  "/lighting/decisions/generate?day=tomorrow&zone=quai_est",
  { method: "POST" }
);
console.log(generated.items);
```

Omettre `zone` pour générer des propositions pour toutes les zones. Une
proposition peut contenir `action` (`increase`, `decrease`, `maintain`),
`current_brightness_percent`, `proposed_brightness_percent`,
`change_percentage_points`, `reason`, `status` et `warnings`. Les seuils de
fréquentation (15 et 5 personnes) et le pas de 10 points sont des règles de
démonstration, pas des règles validées pour un vrai port.

Lister les propositions :

```text
GET /api/v1/lighting/decisions?date=2026-10-18&zone=quai_est&status=pending_human_review
```

Le frontend peut envoyer une revue humaine :

```js
const decision = generated.items[0];
const review = await api(
  `/lighting/decisions/${decision.decision_id}/review`,
  {
    method: "POST",
    body: JSON.stringify({
      action: "approve", // ou "reject"
      reviewer: "responsable",
      note: "Revue de démonstration",
    }),
  }
);

console.log(review.status);                // "approved_simulation" ou "rejected"
console.log(review.hardware_command_sent); // false
```

Une proposition déjà examinée ne peut pas être revue une deuxième fois et
renvoie HTTP `409`. L'approbation est uniquement enregistrée comme
`approved_simulation`. **Aucune commande de matériel n'est envoyée.**

## 9. Rapports et import de fichiers

Rapports :

```js
const energyStatus = await api("/reports/energy");
const biodiversity = await api("/reports/biodiversity");
const recommendationReport = await api("/reports/decisions?zone=quai_est");
```

`/reports/energy` indique l'état des mesures réelles; actuellement
`measured_energy_kwh` vaut `null`. Pour l'estimation, appeler `/energy/forecast`.
Le rapport biodiversité décrit les observations, mais ne définit pas de seuils
écologiques opérationnels.

Pour importer un CSV du navigateur, utiliser `FormData`; ne pas définir
manuellement l'en-tête `Content-Type`, car le navigateur doit ajouter la
frontière multipart :

```js
const form = new FormData();
form.append("file", fileInput.files[0]);
const response = await fetch(`${API_BASE}/datasets/attendance`, {
  method: "POST",
  body: form,
});
const result = await response.json();
if (!response.ok) {
  throw new Error(result.error?.message ?? "Échec de l'import CSV");
}
```

L'import valide un fichier `.csv` de 20 Mio maximum. Pour cette démo, utilisez
de préférence les données fournies avec le dépôt plutôt que d'importer depuis
l'interface.

## 10. Erreurs à gérer côté frontend

Les erreurs JSON ont la forme générale suivante :

```json
{
  "error": {
    "code": "model_not_trained",
    "message": "Entraînez explicitement le modèle avant de demander une prédiction."
  }
}
```

| HTTP | Codes fréquents | Réaction frontend conseillée |
|---|---|---|
| `409` | `model_not_trained`, `training_rejected`, `no_future_slots`, `decision_already_reviewed` | Afficher le message; entraîner le modèle, demander demain si aucun créneau ne reste, ou actualiser la proposition. |
| `422` | `invalid_request`, `unknown_zone`, `zone_group_mismatch`, `slot_in_past`, `dataset_validation_failed` | Montrer les détails et demander de corriger les champs ou le fichier. |
| `404` | `job_not_found`, `decision_not_found` | Indiquer que l'identifiant n'existe plus ou est incorrect. |
| `413` / `415` | `file_too_large`, `unsupported_file` | Demander un CSV de moins de 20 Mio. |
| `502` | `weather_unavailable` | Indiquer que la météo distante est temporairement indisponible et proposer de réessayer. |
| `503` | `energy_simulation_unavailable`, `decision_simulation_unavailable` | Signaler que les données locales nécessaires à la simulation sont indisponibles. |
| `500` | `internal_error` | Afficher un message générique; consulter les journaux du backend. |

## 11. Limites à afficher pendant la démo

- Le projet est un prototype de hackathon pour le seul port de Porquerolles.
- Les nombres de fréquentation sont des estimations de Noctora, pas des mesures
  réelles. Les données synthétiques sont exclues de l'entraînement, mais peuvent
  être utilisées comme référence `indice_global` lorsque les observations
  manquent; la réponse le signale.
- Les métriques MAE, RMSE et R² sont calculées sur un holdout groupé. Le CSV de
  fréquentation pour Noctora n'a pas de dates. Le CSV d'éclairage, lui, a des
  dates de 2025, mais sa fréquentation est simulée et ne valide pas la précision
  de Noctora sur de futures fréquentations observées.
- Les chiffres énergétiques et économies potentielles reposent sur des données
  simulées; aucune mesure réelle de compteur n'est actuellement fournie.
- Les recommandations gardent leur intensité à `null` tant que la sécurité et
  l'écologie ne sont pas validées. Les propositions sont de démonstration et
  toute revue reste sans effet matériel.
- L'API n'a pas d'authentification. Utilisez-la pour la démo locale ou un
  environnement hackathon contrôlé, pas comme système de production.

Pour plus de détails, de schémas et d'exemples, consulter aussi le
[README principal](./README.md) et la documentation interactive
<http://127.0.0.1:8000/docs>.
