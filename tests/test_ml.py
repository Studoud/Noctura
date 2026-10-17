import json
import os
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import joblib
import numpy as np

from cerveau import prediction as P
from ml import biodiversite as B
from ml import features as F
from ml import train as T

PARIS = ZoneInfo("Europe/Paris")
METEO_CALME = {"rafales_kmh": 20.0, "code_meteo": 1, "hauteur_vagues_m": 0.4}
NUIT_ETE = datetime(2025, 6, 21, 23, 30, tzinfo=PARIS)


class EstimateurExcessif:
    """Estimateur factice qui sort de l'échelle 0-100."""

    def predict(self, X):
        return np.full(len(X), 500.0)


def _artefact(modele, **surcharges):
    artefact = {
        "noms_features": list(F.NOMS_FEATURES),
        "seuils_version": F.SEUILS_VERSION,
        "modele": modele,
        "q_conformal": 1.5,
        "accepte": True,
        "raisons_refus": [],
    }
    artefact.update(surcharges)
    return artefact


class TestFeatures(unittest.TestCase):
    def test_deterministes(self):
        a = F.features_service("quai_est", NUIT_ETE, METEO_CALME)
        b = F.features_service("quai_est", NUIT_ETE, METEO_CALME)
        self.assertEqual(a, b)
        self.assertEqual(len(a), len(F.NOMS_FEATURES))

    def test_encodage_heure_identique_au_dataset(self):
        x = F.vecteur("capitainerie", 2, 0, 0, 0, jour=date(2025, 6, 21))
        self.assertAlmostEqual(x[0], 0.49999999999999994)
        self.assertAlmostEqual(x[1], 0.8660254037844387)

    def test_indice_global_exclu(self):
        self.assertFalse(any("indice" in nom for nom in F.NOMS_FEATURES))

    def test_daylight_milieu_de_creneau(self):
        self.assertEqual(F.calculer_is_daylight(datetime(2025, 6, 21, 7, 0, tzinfo=PARIS)), 1)
        self.assertEqual(F.calculer_is_daylight(datetime(2025, 6, 21, 21, 0, tzinfo=PARIS)), 1)
        self.assertEqual(F.calculer_is_daylight(datetime(2025, 6, 21, 5, 0, tzinfo=PARIS)), 0)
        self.assertEqual(F.calculer_is_daylight(datetime(2025, 12, 21, 7, 0, tzinfo=PARIS)), 0)

    def test_hiver_et_tempete(self):
        self.assertEqual(F.calculer_is_winter(datetime(2025, 1, 15, tzinfo=PARIS)), 1)
        self.assertEqual(F.calculer_is_winter(datetime(2025, 7, 15, tzinfo=PARIS)), 0)
        self.assertEqual(F.calculer_is_storm(METEO_CALME), 0)
        self.assertEqual(F.calculer_is_storm({"rafales_kmh": 70.0}), 1)
        self.assertEqual(F.calculer_is_storm({"code_meteo": 95}), 1)
        self.assertEqual(F.calculer_is_storm({"rafales_kmh": 10.0, "hauteur_vagues_m": 3.0}), 1)

    def test_features_manquantes(self):
        with self.assertRaises(F.FeatureManquante):
            F.calculer_is_storm(None)
        with self.assertRaises(F.FeatureManquante):
            F.calculer_is_storm({"couverture_nuageuse": 50})
        with self.assertRaises(F.FeatureManquante):
            F.vecteur("zone_inconnue", 0, 0, 0, 0, jour=date(2025, 6, 21))

    def test_regle_daylight_compatible_avec_dataset(self):
        _, lignes, _ = T.charger(T.RACINE / "data" / "dataset_ml_frequentation_24h.csv")
        self.assertGreater(T.verifier_regle_daylight(lignes)["n_dates_compatibles"], 0)

    def test_journees_reconstituees_sans_colonne_date(self):
        _, lignes, _ = T.charger(T.DATASET)
        jours = {l["date"] for l in lignes}
        self.assertEqual(T.mode_validation(lignes), "jours")
        self.assertGreater(len(jours), 100)
        premier_jour = [l for l in lignes if l["date"] == "J001"]
        self.assertEqual(len(premier_jour), 12 * len(F.ZONES))


class TestFuite(unittest.TestCase):
    def test_indice_global_detecte(self):
        _, lignes, _ = T.charger(T.DATASET)
        self.assertTrue(T.verifier_fuite(lignes)["fuite"])


class TestProfil(unittest.TestCase):
    def test_bornes_et_p90(self):
        profil = P.ProfilFrequentation()
        for zone in (*P.PROFIL_CRENEAUX, "A", "B"):
            for heure in range(24):
                t = datetime(2025, 6, 21, heure, 30, tzinfo=PARIS)
                p50, estime = profil.indice(zone, t)
                p90, _ = profil.indice_p90(zone, t)
                self.assertFalse(estime)
                self.assertTrue(0.0 <= p50 <= 100.0)
                self.assertTrue(0.0 <= p90 <= 100.0)
                self.assertGreaterEqual(p90, p50)

    def test_correspondance_zones(self):
        self.assertEqual(len(P.zones_dataset("A")), 6)
        self.assertEqual(set(P.zones_dataset("B")), {"acces_village", "quai_sud_commerces"})
        self.assertEqual(P.zones_dataset("Quai Sud"), ("quai_sud_commerces",))
        self.assertEqual(P.zones_dataset("pontons_centraux"), ("pontons_centraux",))
        self.assertEqual(P.zones_dataset("inconnue"), ())

    def test_zone_inconnue_prudente(self):
        profil = P.ProfilFrequentation()
        valeur, estime = profil.indice("inconnue", NUIT_ETE)
        self.assertTrue(estime)
        self.assertEqual(valeur, max(profil.indice(z, NUIT_ETE)[0] for z in P.PROFIL_CRENEAUX))


class TestPredicteurML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sklearn.linear_model import PoissonRegressor

        _, lignes, _ = T.charger(T.DATASET)
        modele = PoissonRegressor(alpha=1e-3, max_iter=5000).fit(T.matrice(lignes), T.cible(lignes))
        cls.dossier = tempfile.TemporaryDirectory()
        cls.chemin = Path(cls.dossier.name) / "model.joblib"
        joblib.dump(_artefact(modele), cls.chemin)

    @classmethod
    def tearDownClass(cls):
        cls.dossier.cleanup()

    def _predicteur(self, chemin=None, meteo=METEO_CALME, **options):
        return P.PredicteurML(chemin or self.chemin, fournisseur_meteo=lambda: meteo, **options)

    def test_sorties_bornees_et_p90(self):
        predicteur = self._predicteur()
        self.assertTrue(predicteur.actif)
        for zone in (*P.PROFIL_CRENEAUX, "A", "B"):
            for heure in range(0, 24, 2):
                t = datetime(2025, 6, 21, heure, 15, tzinfo=PARIS)
                p50, estime = predicteur.indice(zone, t)
                p90, _ = predicteur.indice_p90(zone, t)
                self.assertFalse(estime)
                self.assertEqual(predicteur.derniere_source, "ml")
                self.assertTrue(0.0 <= p50 <= 100.0)
                self.assertTrue(0.0 <= p90 <= 100.0)
                self.assertGreaterEqual(p90, p50)

    def test_bornage_sortie_excessive(self):
        chemin = Path(self.dossier.name) / "excessif.joblib"
        joblib.dump(_artefact(EstimateurExcessif()), chemin)
        predicteur = self._predicteur(chemin)
        self.assertEqual(predicteur.indice("quai_est", NUIT_ETE), (100.0, False))
        self.assertEqual(predicteur.indice_p90("quai_est", NUIT_ETE), (100.0, False))

    def test_repli_modele_absent(self):
        predicteur = self._predicteur(Path(self.dossier.name) / "absent.joblib")
        self.assertFalse(predicteur.actif)
        valeur, estime = predicteur.indice("quai_est", NUIT_ETE)
        self.assertTrue(estime)
        self.assertEqual(valeur, P.ProfilFrequentation().indice("quai_est", NUIT_ETE)[0])
        self.assertEqual(predicteur.derniere_source, "profil")

    def test_repli_modele_illisible(self):
        chemin = Path(self.dossier.name) / "corrompu.joblib"
        chemin.write_bytes(b"pas un modele")
        self.assertFalse(self._predicteur(chemin).actif)

    def test_repli_features_incompatibles(self):
        chemin = Path(self.dossier.name) / "incompatible.joblib"
        joblib.dump(_artefact(EstimateurExcessif(), noms_features=["autre"]), chemin)
        self.assertFalse(self._predicteur(chemin).actif)

    def test_repli_modele_refuse(self):
        chemin = Path(self.dossier.name) / "refuse.joblib"
        joblib.dump(_artefact(EstimateurExcessif(), accepte=False, raisons_refus=["test"]), chemin)
        self.assertFalse(self._predicteur(chemin).actif)
        self.assertTrue(self._predicteur(chemin, exiger_acceptation=False).actif)

    def test_repli_meteo_manquante(self):
        predicteur = self._predicteur(meteo=None)
        valeur, estime = predicteur.indice_p90("B", NUIT_ETE)
        self.assertTrue(estime)
        self.assertEqual(predicteur.derniere_source, "profil")
        self.assertEqual(valeur, P.ProfilFrequentation().indice_p90("B", NUIT_ETE)[0])

    def test_modele_simule_reserve_au_mode_demo(self):
        chemin = Path(self.dossier.name) / "simule.joblib"
        joblib.dump(_artefact(EstimateurExcessif(), donnees_simulees=True), chemin)
        self.assertFalse(self._predicteur(chemin).actif)
        predicteur = self._predicteur(chemin, autoriser_simule=True)
        self.assertTrue(predicteur.actif)
        self.assertTrue(predicteur.simule)
        with mock.patch.dict(os.environ, {"PREDICTION_MODE": "demo"}):
            self.assertTrue(P.creer_predicteur(chemin_modele=chemin, fournisseur_meteo=lambda: METEO_CALME).simule)

    def test_mode_par_variable_environnement(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PREDICTION_MODE", None)
            self.assertIsInstance(P.creer_predicteur(), P.ProfilFrequentation)
        with mock.patch.dict(os.environ, {"PREDICTION_MODE": "ml"}):
            predicteur = P.creer_predicteur(chemin_modele=self.chemin, fournisseur_meteo=lambda: METEO_CALME)
            self.assertIsInstance(predicteur, P.PredicteurML)


class TestBiodiversite(unittest.TestCase):
    def test_tendances(self):
        self.assertEqual(B.tendance({2022: 100, 2023: 90, 2024: 70, 2025: 60})["niveau"], "alerte")
        self.assertEqual(B.tendance({2022: 100, 2023: 95, 2024: 90, 2025: 85})["niveau"], "vigilance")
        self.assertEqual(B.tendance({2022: 100, 2023: 110, 2024: 105, 2025: 120})["niveau"], "stable")
        self.assertEqual(B.tendance({2024: 100, 2025: 10})["niveau"], "inconnu")

    def test_format_tidy_exclut_indetermines_et_petits_effectifs(self):
        with tempfile.TemporaryDirectory() as d:
            tidy = Path(d) / "tidy.csv"
            tidy.write_text(
                "espece,nom_commun,nom_scientifique,taxon_indetermine,annee,effectif\n"
                + "".join(f"Sar,Sar,Diplodus,0,{a},{n}\n" for a, n in zip(range(2022, 2026), (16, 6, 8, 2)))
                + "".join(f"Sparidé sp.,,Sparidae sp.,1,{a},{n}\n" for a, n in zip(range(2022, 2026), (80, 0, 0, 0)))
                + "".join(f"Mérou,Mérou,Epinephelus,0,{a},{n}\n" for a, n in zip(range(2022, 2026), (0, 0, 0, 1))),
                encoding="utf-8",
            )
            resultat = B.sensibilite(tidy)
            self.assertEqual(resultat["taxons_exclus"], ["Sparidé sp."])
            self.assertEqual(resultat["especes"]["Sar"]["niveau"], "alerte")
            self.assertEqual(resultat["especes"]["Mérou"]["niveau"], "inconnu")

    def test_fichier_tabule(self):
        with tempfile.TemporaryDirectory() as d:
            tab = Path(d) / "tab.csv"
            tab.write_text("Espèce\t2022\t2023\t2024\t2025\tTotal\nSaupe\t77\t174\t188\t224\t663\n", encoding="utf-8")
            self.assertEqual(B.sensibilite(tab)["especes"]["Saupe"]["niveau"], "stable")

    def test_formats_et_fichier_vide(self):
        with tempfile.TemporaryDirectory() as d:
            vide = Path(d) / "vide.csv"
            vide.write_text("", encoding="utf-8")
            self.assertEqual(B.sensibilite(vide)["niveau_global"], "inconnu")

            large = Path(d) / "large.csv"
            large.write_text("espece,2022,2023,2024,2025\natherine,100,90,70,60\nsparaillon,50,52,55,51\n", encoding="utf-8")
            resultat = B.sensibilite(large)
            self.assertEqual(resultat["niveau_global"], "alerte")
            self.assertEqual(resultat["especes_en_alerte"], ["atherine"])

            long_ = Path(d) / "long.csv"
            long_.write_text("annee,espece,comptage\n2022,sar,10\n2023,sar,11\n2024,sar,12\n2025,sar,12\n", encoding="utf-8")
            self.assertEqual(B.sensibilite(long_)["niveau_global"], "stable")

            sortie = Path(d) / "sortie.json"
            sortie.write_text(json.dumps({"niveau_global": "vigilance"}), encoding="utf-8")
            self.assertEqual(B.niveau_sensibilite(sortie), "vigilance")
            self.assertEqual(B.niveau_sensibilite(Path(d) / "absent.json"), "inconnu")


if __name__ == "__main__":
    unittest.main()
