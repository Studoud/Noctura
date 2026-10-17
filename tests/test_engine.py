import unittest
from datetime import date, datetime
from zoneinfo import ZoneInfo

from cerveau import regles, securite
from cerveau.prediction import ProfilFrequentation
from engine import SCENARIOS, Noctura
from ml import features as F
from ml import simulation as S

PARIS = ZoneInfo("Europe/Paris")
NUIT = {"facteur_lumiere_naturelle": 0, "risque_brouillard": 0, "is_storm": 0, "niveau_biodiversite": "inconnu"}


def _decision(zone="quai_est", p90=3.0, estime=False, **contexte):
    ctx = dict(NUIT, **contexte)
    return securite.valider(regles.decider(zone, p90 / 2, p90, estime, ctx), ctx)


class TestRegles(unittest.TestCase):
    def test_paliers_de_demande(self):
        self.assertEqual(_decision(p90=3).niveau, 10)    # veille : P90 < 5, aucun plancher de présence
        self.assertEqual(_decision(p90=8).niveau, 30)
        self.assertEqual(_decision(p90=20).niveau, 60)
        self.assertEqual(_decision(p90=40).niveau, 100)

    def test_jour_eteint(self):
        self.assertEqual(_decision(p90=40, facteur_lumiere_naturelle=80).niveau, 0)

    def test_repli_prudent(self):
        self.assertGreaterEqual(_decision(p90=1, estime=True).niveau, regles.PLANCHER_ESTIME)

    def test_biodiversite_plafonne_et_ambre(self):
        d = _decision(p90=40, niveau_biodiversite="alerte")
        self.assertEqual(d.niveau, 60)
        self.assertEqual(d.temperature_k, 2200)


class TestSecurite(unittest.TestCase):
    def test_jamais_eteint_sur_des_personnes(self):
        for p90 in (5, 6, 10):
            self.assertGreaterEqual(_decision(p90=p90, niveau_biodiversite="alerte").niveau, securite.PLANCHER_PRESENCE)

    def test_tempete_prime_sur_la_sobriete(self):
        self.assertGreaterEqual(_decision(p90=1, is_storm=1, niveau_biodiversite="alerte").niveau, securite.PLANCHER_TEMPETE)

    def test_brouillard_port_et_village(self):
        self.assertEqual(_decision("quai_nord", p90=1, risque_brouillard=90).niveau, securite.PLANCHER_BROUILLARD_PORT)
        self.assertEqual(_decision("acces_village", p90=1, risque_brouillard=90).niveau, securite.PLANCHER_BROUILLARD)

    def test_zone_critique_quai_sud(self):
        # Nuit, presque personne, biodiversité en alerte : la zone critique ne descend pas sous 50 %.
        nuit = _decision("quai_sud_commerces", p90=1, niveau_biodiversite="alerte")
        self.assertEqual(nuit.niveau, 50)
        self.assertTrue(any("zone critique" in s for s in nuit.securite))
        # Les autres zones ne sont pas concernées.
        self.assertLess(_decision("acces_village", p90=1).niveau, 50)
        # En plein jour, pas de plancher.
        self.assertEqual(_decision("quai_sud_commerces", p90=40, facteur_lumiere_naturelle=80).niveau, 0)

    def test_navigation_de_nuit(self):
        self.assertGreaterEqual(_decision("entree_du_port", p90=0).niveau, securite.PLANCHER_NAVIGATION_NUIT)

    def test_meteo_inconnue(self):
        self.assertGreaterEqual(_decision(p90=0, is_storm=None).niveau, securite.PLANCHER_PRESENCE)

    def test_securite_ne_baisse_jamais(self):
        ctx = dict(NUIT)
        brute = regles.decider("quai_est", 30, 40, False, ctx)
        niveau = brute.niveau
        self.assertEqual(securite.valider(brute, ctx).niveau, niveau)


class TestMoteur(unittest.TestCase):
    def setUp(self):
        self.moteur = Noctura(predicteur=ProfilFrequentation(), fournisseur_meteo=lambda: SCENARIOS["calme"],
                              niveau_biodiversite="inconnu")

    def test_toutes_les_zones_bornees(self):
        for heure in range(24):
            resultat = self.moteur.decider(datetime(2025, 7, 26, heure, 30, tzinfo=PARIS))
            self.assertEqual(len(resultat["zones"]), len(F.ZONES))
            for d in resultat["zones"]:
                self.assertTrue(0 <= d.niveau <= 100)

    def test_midi_eteint_minuit_allume(self):
        midi = self.moteur.decider(datetime(2025, 7, 26, 13, 0, tzinfo=PARIS))
        nuit = self.moteur.decider(datetime(2025, 7, 26, 23, 0, tzinfo=PARIS))
        self.assertEqual(midi["puissance_moyenne"], 0)
        self.assertGreater(nuit["puissance_moyenne"], 0)

    def test_scenario_tempete(self):
        resultat = self.moteur.decider(datetime(2025, 1, 15, 3, 0, tzinfo=PARIS), SCENARIOS["tempete"])
        self.assertEqual(resultat["contexte"]["is_storm"], 1)
        self.assertTrue(all(d.niveau >= securite.PLANCHER_TEMPETE for d in resultat["zones"]))


class TestSimulation(unittest.TestCase):
    def test_deterministe_et_coherente(self):
        a = S.simuler(date(2025, 1, 1), date(2025, 1, 3))
        b = S.simuler(date(2025, 1, 1), date(2025, 1, 3))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3 * 12 * len(F.ZONES))
        for ligne in a:
            self.assertEqual(ligne["source"], "simule")
            self.assertTrue(0 <= ligne["frequentation_target"] <= 100)
            self.assertEqual(ligne["is_storm"], F.calculer_is_storm(ligne))

    def test_saison(self):
        self.assertGreater(S.facteur_saison(date(2025, 7, 24)), S.facteur_saison(date(2025, 1, 15)))


if __name__ == "__main__":
    unittest.main()
