# NOCTURA : moteur d'éclairage adaptatif du port de Porquerolles

Ce document explique le backend : ce qu'il calcule, avec quelles données, comment le lancer, comment le vérifier, et comment l'interface web s'y connecte.

**En une phrase :** toutes les 2 heures, pour chacune des 8 zones du port, NOCTURA estime combien de personnes seront présentes, puis décide du niveau d'éclairage (0-100 %) et de sa teinte. La décision tient compte de la lumière naturelle, de la météo, de la sécurité et de la biodiversité.

---

## Sommaire

1. [Architecture](#1-architecture)
2. [Installation et lancement](#2-installation-et-lancement)
3. [Les données](#3-les-données)
4. [La prédiction de fréquentation (ML)](#4-la-prédiction-de-fréquentation-ml)
5. [Le moteur de décision](#5-le-moteur-de-décision)
6. [L'API](#6-lapi)
7. [Connecter l'interface web](#7-connecter-linterface-web)
8. [Méthodes de vérification](#8-méthodes-de-vérification)
9. [Limites connues et suite](#9-limites-connues-et-suite)
10. [Carte des fichiers](#10-carte-des-fichiers)

---

## 1. Architecture

```
            Open-Meteo (météo + mer, coordonnées du port)      astral (soleil, lune)
                              │                                        │
                              ▼                                        ▼
                     data/meteo_astro.py ──────────────► contexte : lumière naturelle,
                              │                          brouillard, tempête
                              ▼
   ml/features.py ──► cerveau/prediction.py ──► P50 / P90 de fréquentation par zone
   (une seule fonction       │  profil | ml | demo
    de features)             ▼
                     cerveau/regles.py ──────► niveau d'éclairage + teinte + raisons
                              │        ▲
                              │        └── ml/biodiversite.py (niveau d'alerte écologique)
                              ▼
                     cerveau/securite.py ────► planchers de sécurité (dernier mot)
                              │
                              ▼
                         engine.py  ◄──── api.py (HTTP/JSON) ◄──── interface web
```

**Principe clé : le modèle prédit la demande, il ne décide jamais.** Les règles décident du niveau d'éclairage, puis le validateur de sécurité peut seulement l'augmenter, jamais le baisser.

---

## 2. Installation et lancement

Prérequis : Python 3.11 ou plus récent (testé avec 3.14).

```powershell
pip install -r requirements.txt
$env:PYTHONIOENCODING = "utf-8"      # Windows : affiche correctement les accents
```

| Je veux… | Commande |
|---|---|
| Lancer l'API | `uvicorn api:app --reload --port 8000` |
| Voir la doc interactive de l'API | ouvrir http://localhost:8000/docs |
| Une décision dans le terminal | `python engine.py --mode ml` |
| Un scénario de démo | `python engine.py --mode ml --date 2025-01-15 --heure 03:00 --scenario tempete` |
| La sortie JSON | `python engine.py --mode ml --json` |
| La météo réelle du port | `python data/meteo_astro.py` |
| Réentraîner le modèle réel | `python -m ml.train` |
| Régénérer la démo simulée | `python -m ml.simulation` puis `python -m ml.train --demo` |
| Recalculer l'indicateur biodiversité | `python -m ml.biodiversite` |
| Lancer tous les tests | `python -m unittest discover -s tests -t .` |

Le mode de prédiction par défaut se règle par variable d'environnement : `$env:PREDICTION_MODE = "ml"` (valeurs possibles : `profil`, `ml`, `demo`). Sans réglage, le mode est `profil`.

---

## 3. Les données

| Fichier | Contenu | Statut |
|---|---|---|
| `data/dataset_ml_frequentation_24h_clean.csv` | 10 000 lignes, soit 104 journées complètes + 1 partielle, 8 zones × 12 créneaux de 2 h. 35 jours d'hiver, 41 créneaux de tempête. | **Dataset d'entraînement du modèle réel.** Nuits marquées réelles (4 176 lignes), journées synthétiques (5 824 lignes). |
| `data/dataset_ml_frequentation_24h.csv` | Une seule journée d'été (96 lignes). | Ancien dataset, conservé pour la référence et les tests. |
| `data/dataset_simule.csv` | 70 176 lignes, 731 journées (2024-2025). | **Inventé** par `ml/simulation.py`, pour la démo uniquement. |
| `data/biodev_1_tidy.csv` | Comptages annuels d'espèces 2022-2025 (format long). | Source de l'indicateur biodiversité. |
| `data/biodiversite.csv` | Même contenu en tableau (séparé par tabulations). | Lisible aussi, mais non utilisé par défaut. |
| Météo live | Open-Meteo (prévision + API marine) aux coordonnées du port (43,0003 N ; 6,2006 E). | Données réelles et à jour, mais **issues d'un modèle météo** (maille de quelques km), pas d'un capteur posé sur le port. |

### Colonnes du dataset de fréquentation

`creneau, hour_start, hour_end, sin_hour, cos_hour, zone_group_id, zone_group_name, zone_specifique, is_daylight, is_winter, is_storm, is_synthetic_daytime, indice_global, frequentation_target`

- **Cible :** `frequentation_target` (0-100 ; observé de 1 à 50).
- **`indice_global` est exclu du modèle.** C'est presque la moyenne des cibles du même créneau (corrélation 0,9955), donc une fuite de la réponse, et il n'est pas disponible en direct.
- **Pas de colonne `date` :** les journées sont reconstituées à partir de la suite des créneaux (J001 à J105). Un retour à 00h-02h marque le début d'une nouvelle journée.

### Les 8 zones

| Groupe | Code | Zones |
|---|---|---|
| G1 Port et débarcadère | **A** | capitainerie, entree_du_port, pontons_centraux, quai_est, quai_nord, quai_ouest |
| G2 Village et commerces | **B** | acces_village, quai_sud_commerces (**zone critique**) |

La correspondance des noms (codes A/B, alias comme « quai sud » ou « passe ») se trouve dans `cerveau/port_porquerolles.json`.

---

## 4. La prédiction de fréquentation (ML)

### Ce que le modèle reçoit (`ml/features.py`)

C'est une seule fonction, partagée entre l'entraînement et le service. Toutes les entrées sont calculables en direct :

| Feature | Calcul |
|---|---|
| Heure | sinus et cosinus de l'heure (2 harmoniques), pour que 23 h soit proche de 0 h |
| Zone | une colonne par zone (8) |
| Saison, week-end | jour de l'année (sinus et cosinus) et samedi/dimanche |
| `is_daylight` | soleil au-dessus de l'horizon au **milieu** du créneau (astral, coordonnées du port) |
| `is_winter` | mois de décembre, janvier ou février |
| `is_storm` | rafales ≥ 62 km/h **ou** orage (code WMO 95, 96, 99) **ou** vagues ≥ 2,5 m |

Les seuils de `is_winter` et `is_storm` sont des **hypothèses** (`SEUILS_VERSION = "seuils-v0-hypothese"`) : ceux qui ont servi à construire le dataset ne sont pas connus. Si la météo ne permet pas de calculer `is_storm`, la prédiction **se replie sur le profil**.

### Entraînement et validation (`ml/train.py`)

1. **Contrôles qualité :** doublons, valeurs manquantes, colonnes constantes, distributions réel / synthétique.
2. **Vérification de la fuite** d'`indice_global` : corrélation et reconstruction.
3. **Validation sans tricherie :** les **21 dernières journées** forment le test et ne sont jamais vues pendant l'entraînement. Le choix des réglages se fait par validation croisée groupée par journée.
4. **Trois modèles comparés :** baseline (moyenne zone × créneau), GLM Poisson, LightGBM Poisson peu profond.
5. **P90 conformal :** une valeur haute, ne devant être dépassée qu'environ 1 fois sur 10, calculée à partir des erreurs passées. Sous-estimer est dangereux, car on éteindrait la lumière sur des gens.
6. **Critère d'acceptation :** le modèle n'est activé que si toutes ces conditions sont remplies :
   - au moins 10 % de MAE en moins que la baseline sur le test réel ;
   - un gain statistiquement positif (IC 95 % bootstrap) ;
   - une couverture du P90 entre 85 et 95 % ;
   - au moins 5 journées réelles.
7. **Sorties :** `ml/model.joblib`, `ml/model_card.md`, `ml/rapport_evaluation.md`. Tous les chiffres sont calculés par le script, aucun n'est saisi à la main.

### Résultats actuels (test réel : 21 journées jamais vues, dont 12 d'hiver et 2 avec tempête)

| Modèle | MAE réel | Couverture P90 | Gain vs baseline |
|---|---|---|---|
| Baseline zone × créneau | 2,13 | 98,8 % | — |
| **GLM Poisson (retenu, accepté)** | **1,16** | **90,0 %** | **−45,5 %** (IC 95 % [28 % ; 58 %]) |
| LightGBM Poisson | 1,26 | 94,6 % | aussi accepté, mais plus complexe |

Le GLM est retenu parce que c'est le modèle le plus simple qui remplit le critère.

### Les trois modes de prédiction

| Mode | Source | Usage |
|---|---|---|
| `profil` | Profil horaire fixe par zone | Défaut et repli de sécurité |
| `ml` | `ml/model.joblib`, entraîné sur le dataset réel | Usage réel. **Refuse** tout modèle simulé ou non accepté |
| `demo` | `ml/model_demo.joblib`, entraîné sur données **simulées** | Démonstration uniquement |

**Repli automatique :** si le modèle est absent, illisible, incompatible ou refusé, ou si une feature manque, la prédiction utilise le profil. Le champ `estime` vaut alors `true`.

---

## 5. Le moteur de décision

Pour chaque zone, le calcul suit toujours le même ordre :

```
P90 de demande ─► palier ─► atténuation (crépuscule/lune) ─► plancher « repli » ─► plafond biodiversité ─► SÉCURITÉ
```

### Règles (`cerveau/regles.py`)

| Règle | Valeur |
|---|---|
| Lumière naturelle ≥ 60 % | éclairage à 0 % (il fait jour) |
| P90 < 5 | palier **veille** 10 % |
| 5 ≤ P90 < 12 | palier **faible** 30 % |
| 12 ≤ P90 < 25 | palier **moyenne** 60 % |
| P90 ≥ 25 | palier **forte** 100 % |
| Crépuscule ou clair de lune | jusqu'à −50 %, proportionnellement à la lumière naturelle |
| Prédiction de repli (`estime`) | minimum 30 % |
| Biodiversité « vigilance » | plafond 80 %, teinte 2400 K |
| Biodiversité « alerte » | plafond 60 %, teinte 2200 K (ambrée) |
| Sinon | teinte 2700 K |

### Sécurité (`cerveau/securite.py`) : le dernier mot, jamais à la baisse

| Situation | Plancher |
|---|---|
| **quai_sud_commerces, la nuit (zone critique)** | **50 %**, la baisse est limitée à 50 % maximum |
| Présence probable la nuit (P90 ≥ 5) | 20 % (jamais éteint) |
| entree_du_port et capitainerie, la nuit | 30 % (repérage depuis la mer) |
| Tempête, la nuit | 50 % partout |
| Brouillard (risque ≥ 50 %) | 60 % pour les zones du port, 40 % ailleurs |
| Météo inconnue, la nuit | 20 % |

La zone critique se règle dans `cerveau/port_porquerolles.json` :

```json
"zones_critiques": {
  "quai_sud_commerces": { "plancher_nuit": 50, "motif": "zone critique de Porquerolles : baisse limitée à 50 % maximum la nuit" }
}
```

### Biodiversité (`ml/biodiversite.py`)

Elle n'entre **pas** dans le modèle : 4 années ne suffisent pas pour apprendre un effet. Le module produit seulement un indicateur descriptif :

- Les taxons indéterminés sont exclus (athérine sp., mulet, sparidé, Symphodus sp., poisson non identifié).
- Les espèces avec moins de 20 individus sur 4 ans restent « inconnu ».
- La tendance compare la moyenne des 2 dernières années à celle des 2 premières, confirmée par le signe de la pente :
  - baisse ≥ 30 % : **alerte** ;
  - baisse de 10 à 30 % : **vigilance** ;
  - sinon : **stable**.
- **Résultat actuel : alerte**, avec l'oblade à −53 % et le sar à museau pointu à −55 %. Le sparaillon (−4 %) et la saupe (+64 %) sont stables.
- Aucune conclusion causale : une baisse n'est pas attribuée à l'éclairage.

---

## 6. L'API

Lancement : `uvicorn api:app --reload --port 8000`. Toutes les routes sont en **GET** et renvoient du **JSON**. La documentation interactive se trouve sur **/docs** : elle permet d'essayer chaque route depuis le navigateur.

| Route | Rôle | Paramètres |
|---|---|---|
| `GET /api/sante` | État du service, mode par défaut, modèle actif | — |
| `GET /api/decision` | Décision des 8 zones à un instant | `date`, `heure`, `scenario`, `mode`, `bio` |
| `GET /api/decision/journee` | 12 décisions (01h, 03h… 23h) pour tracer une journée | `date`, `scenario`, `mode`, `bio` |
| `GET /api/meteo` | Météo, mer, soleil et lune en direct (cache 10 min) | — |
| `GET /api/zones` | Zones, groupes A/B, zones critiques, coordonnées | — |
| `GET /api/scenarios` | Scénarios météo de démonstration | — |
| `GET /api/biodiversite` | Indicateur écologique détaillé par espèce | — |
| `GET /api/modele` | Statut et métriques des modèles `ml` et `demo` | — |

### Paramètres

| Paramètre | Format | Défaut | Valeurs |
|---|---|---|---|
| `date` | `AAAA-MM-JJ` | aujourd'hui | — |
| `heure` | `HH:MM` | maintenant | — |
| `scenario` | texte | `live` | `live`, `calme`, `couvert`, `tempete`, `orage`, `brouillard` |
| `mode` | texte | `PREDICTION_MODE` | `profil`, `ml`, `demo` |
| `bio` | texte | calculé | `inconnu`, `stable`, `vigilance`, `alerte` |

Un paramètre invalide renvoie l'erreur **422** avec le détail du champ fautif. Si Open-Meteo est injoignable, l'API ne tombe pas en panne :
- `/api/meteo` renvoie seulement les valeurs par défaut et l'astronomie, **sans `heure_meteo`** ;
- `/api/decision` continue de répondre, avec `contexte.is_storm = null` ; la sécurité applique alors le plancher « météo inconnue ».

### Exemple : `GET /api/decision?date=2025-07-26&heure=23:00&scenario=calme&mode=ml&bio=alerte`

Réponse réelle (2 zones sur 8 montrées) :

```json
{
  "instant": "2025-07-26T23:00+02:00",
  "mode": "ml",
  "prediction": "ML",
  "meteo_source": "scénario « calme »",
  "contexte": {
    "facteur_lumiere_naturelle": 1,
    "altitude_soleil": -17.5,
    "phase_lune": "Premier croissant",
    "risque_brouillard": 0,
    "is_storm": 0,
    "niveau_biodiversite": "alerte",
    "meteo": { "rafales_kmh": 15.0, "code_meteo": 1, "couverture_nuageuse": 20, "hauteur_vagues_m": 0.4, "temperature_c": null }
  },
  "zones": [
    {
      "zone": "capitainerie",
      "niveau": 30,
      "temperature_k": 2200,
      "p50": 7.5,
      "p90": 9.7,
      "estime": false,
      "raisons": ["demande P90 9.7 : palier faible (30 %)", "lumière naturelle 1 % (crépuscule ou lune) : -1 %"],
      "securite": []
    },
    {
      "zone": "quai_sud_commerces",
      "niveau": 60,
      "temperature_k": 2200,
      "p50": 15.3,
      "p90": 18.4,
      "estime": false,
      "raisons": ["demande P90 18.4 : palier moyenne (60 %)", "lumière naturelle 1 % (crépuscule ou lune) : -1 %"],
      "securite": []
    }
  ],
  "puissance_moyenne": 48.8
}
```

En scénario `tempete` une nuit d'hiver, quai_sud_commerces reçoit par exemple :
`"niveau": 50, "securite": ["zone critique de Porquerolles : baisse limitée à 50 % maximum la nuit : relevé de 9 % à 50 %"]`.

### Signification des champs

| Champ | Type | Sens |
|---|---|---|
| `instant` | texte ISO | Instant décidé (heure de Paris) |
| `mode` / `prediction` | texte | Mode demandé ; source réellement utilisée (`ML`, `profil`, `profil (repli…)`, `ML DÉMO…`) |
| `meteo_source` | texte | Météo réelle (`live…`) ou scénario |
| `contexte.facteur_lumiere_naturelle` | 0-100 | 0 = noir complet, 100 = plein jour |
| `contexte.is_storm` | 0, 1 ou `null` | `null` = météo inconnue |
| `contexte.meteo` | objet | Température, vent, rafales, vagues… (`null` en scénario si non défini) |
| `zones[].niveau` | 0-100 entier | **Puissance d'éclairage à appliquer, en %** |
| `zones[].temperature_k` | entier | Teinte : 2200 (ambrée), 2400 ou 2700 K |
| `zones[].p50` / `p90` | nombre | Fréquentation probable / haute estimée (0-100) |
| `zones[].estime` | booléen | `true` = valeur de repli : à signaler dans l'interface |
| `zones[].raisons` | liste de textes | Explication de la décision |
| `zones[].securite` | liste de textes | Règles de sécurité qui ont relevé le niveau (vide sinon) |
| `puissance_moyenne` | nombre | Moyenne des 8 niveaux (indicateur de sobriété) |

---

## 7. Connecter l'interface web

### En développement

1. Lancer l'API : `uvicorn api:app --reload --port 8000`.
2. Le CORS est ouvert par défaut (`*`), donc le front peut appeler `http://localhost:8000` directement.
3. Option plus propre avec Vite : un proxy évite d'écrire l'URL complète.

```js
// vite.config.js
export default {
  server: { proxy: { "/api": "http://localhost:8000" } },
};
```

### En ligne

Restreindre le CORS à l'adresse du front :

```powershell
$env:NOCTURA_CORS = "https://mon-front.exemple.fr"
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Appeler l'API (JavaScript)

```js
const API = "http://localhost:8000"; // ou "" avec le proxy Vite

export async function getDecision({ date, heure, scenario = "live", mode } = {}) {
  const params = new URLSearchParams({ scenario });
  if (date) params.set("date", date);     // "2025-07-26"
  if (heure) params.set("heure", heure);  // "23:00"
  if (mode) params.set("mode", mode);     // "ml" | "profil" | "demo"
  const r = await fetch(`${API}/api/decision?${params}`);
  if (!r.ok) throw new Error(`API ${r.status} : ${await r.text()}`);
  return r.json();
}
```

### Exemple React : rafraîchissement automatique

```jsx
import { useEffect, useState } from "react";
import { getDecision } from "./api";

export function useDecision(options) {
  const [data, setData] = useState(null);
  const [erreur, setErreur] = useState(null);
  useEffect(() => {
    let actif = true;
    const charger = () => getDecision(options).then(d => actif && setData(d)).catch(e => actif && setErreur(e));
    charger();
    const id = setInterval(charger, 5 * 60 * 1000); // toutes les 5 min (la météo est en cache 10 min)
    return () => { actif = false; clearInterval(id); };
  }, [JSON.stringify(options)]);
  return { data, erreur };
}

export function TableauZones() {
  const { data, erreur } = useDecision({ mode: "ml" });
  if (erreur) return <p>API injoignable</p>;
  if (!data) return <p>Chargement…</p>;
  return (
    <ul>
      {data.zones.map(z => (
        <li key={z.zone} title={[...z.raisons, ...z.securite].join("\n")}>
          <span style={{ background: couleurTeinte(z.temperature_k), opacity: 0.2 + 0.8 * z.niveau / 100 }}>●</span>
          {" "}{z.zone} : {z.niveau} %{z.securite.length > 0 && " 🛡"}{z.estime && " (estimation de repli)"}
        </li>
      ))}
    </ul>
  );
}

// Teinte approximative pour l'affichage
const couleurTeinte = k => ({ 2200: "#FFB46B", 2400: "#FFBE7E", 2700: "#FFC58F" }[k] ?? "#FFC58F");
```

### Conseils d'affichage

| Donnée | Proposition d'affichage |
|---|---|
| `niveau` | Intensité ou opacité de la zone sur la carte du port |
| `temperature_k` | Couleur de la lumière (ambrée = mode biodiversité) |
| `raisons` + `securite` | Infobulle « pourquoi ce niveau ? » : c'est l'argument d'explicabilité |
| `securite` non vide | Pictogramme bouclier |
| `estime = true` | Badge « estimation de repli » |
| `/api/decision/journee` | Courbe de la journée : `puissance_moyenne` par créneau, ou une courbe par zone |
| `/api/meteo` | Bandeau météo : température, vent, vagues, phase de lune |
| `/api/scenarios` | Liste déroulante « simulation » pour la démo (tempête, brouillard…) |
| `/api/biodiversite` | Panneau espèces : niveau global et tendances |

---

## 8. Méthodes de vérification

### Tests automatiques : 48 tests

```powershell
python -m unittest discover -s tests -t .
```

| Fichier | Ce qui est vérifié |
|---|---|
| `tests/test_ml.py` | Features déterministes ; `indice_global` exclu ; seuils jour/hiver/tempête ; détection de la fuite ; journées reconstituées ; sorties bornées 0-100 ; P90 ≥ P50 ; repli si modèle absent, illisible, incompatible, refusé ou météo manquante ; modèle simulé refusé en mode `ml` ; tendances biodiversité, taxons indéterminés et petits effectifs, fichier tabulé |
| `tests/test_engine.py` | Paliers ; jour = éteint ; plancher de repli ; plafond biodiversité ; jamais éteint sur des personnes ; tempête prioritaire sur la sobriété ; brouillard ; navigation ; **zone critique à 50 % la nuit** ; la sécurité ne baisse jamais ; simulateur déterministe |
| `tests/test_api.py` | Toutes les routes ; scénarios ; mode `ml` ; journée de 12 créneaux ; erreurs 422 sur paramètres invalides ; météo simulée hors réseau |

### Vérifications à la main

| Commande | Résultat attendu |
|---|---|
| `python engine.py --mode ml --date 2025-07-26 --heure 13:00 --scenario calme` | tout à **0 %** (jour) |
| `python engine.py --mode ml --date 2025-01-15 --heure 03:00 --scenario tempete` | tout ≥ **50 %**, quai_sud_commerces marqué « zone critique » |
| `python engine.py --mode ml --heure 23:00 --scenario brouillard` | zones du port ≥ **60 %** |
| `python engine.py --mode ml` puis `python data/meteo_astro.py` | `heure_meteo` du jour ; valeurs proches d'un site météo pour Porquerolles |
| `python -m ml.train` | `Décision : ACCEPTÉ (glm_poisson)` |
| http://localhost:8000/docs | toutes les routes testables depuis le navigateur |

### Vérifier que la météo est réelle

- La route `/api/meteo` ou le script `data/meteo_astro.py` donne `heure_meteo`, l'heure de la mesure, qui doit être celle d'aujourd'hui.
- Les valeurs changent au fil de la journée.
- Hors connexion, le message `Open-Meteo indisponible… météo par défaut` apparaît dans le terminal, `heure_meteo` est absent et `contexte.is_storm` vaut `null`.
- Dans `/api/decision`, `meteo_source` indique `live…` ou `scénario…`.

---

## 9. Limites connues et suite

- **Origine des nuits « réelles » non vérifiée :** elles suivent le même motif régulier que le générateur du dataset. À confirmer avec la personne qui a fourni le fichier.
- **Pas de date dans le dataset :** le modèle n'a rien appris sur la saison fine ni sur le week-end. Ajouter une colonne `date` améliorerait le modèle sans changer le code.
- **Seuils `is_storm` / `is_winter` hypothétiques :** les remplacer par ceux du générateur, puis réentraîner.
- **P90 sous-couvrant sur quai_sud_commerces** (77,5 % au lieu de 90 %) : compensé la nuit par le plancher de 50 %. Une amélioration possible serait un P90 calculé zone par zone.
- **Météo modélisée, pas mesurée :** idéalement, brancher une station sur place.
- **Seuils de décision** (paliers, planchers) à faire valider par la capitainerie.
- **Zone C** du port : non définie.
- **`/api/decision/journee` en mode `live`** applique la météo actuelle à toute la journée : ce n'est pas une prévision horaire.

---

## 10. Carte des fichiers

```
api.py                       API web (FastAPI)
engine.py                    moteur : météo -> prédiction -> règles -> sécurité ; CLI
cerveau/
  prediction.py              profil, PredicteurML, modes profil/ml/demo, correspondance des zones
  regles.py                  paliers, lumière naturelle, plafond biodiversité, teinte
  securite.py                planchers de sécurité, zone critique
  port_porquerolles.json     zones A/B, alias, zones critiques, coordonnées
data/
  meteo_astro.py             météo + mer Open-Meteo, soleil et lune (astral)
  dataset_ml_frequentation_24h_clean.csv   dataset d'entraînement (105 journées)
  dataset_ml_frequentation_24h.csv         ancienne journée unique
  dataset_simule.csv         données simulées (démo)
  biodev_1_tidy.csv          comptages d'espèces 2022-2025
ml/
  features.py                fonction unique de features + seuils
  train.py                   entraînement, validation, critère d'acceptation, rapports
  simulation.py              simulateur de démo
  biodiversite.py            indicateur de sensibilité écologique
  model.joblib, model_card.md, rapport_evaluation.md            modèle réel
  model_demo.joblib, model_card_demo.md, rapport_evaluation_demo.md   modèle démo
  sensibilite_ecologique.json
tests/                       48 tests (ML, moteur, API)
requirements.txt             dépendances épinglées
```
