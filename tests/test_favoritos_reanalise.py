"""Refazer a análise de um episódio não pode apagar os favoritos dele.

Na 1.38.0 a "Análise rápida" rodando por cima de episódios prontos apagou os
favoritos do Ajk: as cenas eram recriadas com ids novos e o favorito, preso
ao id, ia junto em cascata.

    python -m unittest discover -s tests -p "test_favoritos_reanalise.py"
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.storage.db import Database


class TestFavoritosNaReanalise(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "index.db")
        anime = self.db.upsert_anime(anilist_id=None, title="Teste")
        self.ep = self.db.upsert_episode(anime, 1, 1, "x.mkv", "", root=self.tmp.name)
        self.lucy = self.db.upsert_character(anime_id=anime, name="Lucy", anilist_id=None)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _cortar(self, limites: list[float]) -> list[int]:
        ids = []
        for i, (a, b) in enumerate(zip(limites, limites[1:])):
            ids.append(self.db.insert_shot(self.ep, i, f"shots/{i:04d}.mp4", None, a, b))
        return ids

    def _favoritas(self) -> dict[int, int]:
        with self.db.connect() as c:
            return {
                r["shot_id"]: r["character_id"]
                for r in c.execute("SELECT shot_id, character_id FROM favorite")
            }

    def test_mesmo_corte(self) -> None:
        ids = self._cortar([0, 2, 5, 9, 12])
        self.db.toggle_favorite(ids[1], self.lucy)  # 2–5 s
        self.db.toggle_favorite(ids[3], 0)  # 9–12 s, em "Todas as cenas"

        self.db.clear_episode_shots(self.ep)
        novos = self._cortar([0, 2, 5, 9, 12])

        self.assertEqual(self._favoritas(), {novos[1]: self.lucy, novos[3]: 0})

    def test_corte_diferente_vai_pelo_momento(self) -> None:
        ids = self._cortar([0, 2, 5, 9, 12])
        self.db.toggle_favorite(ids[2], self.lucy)  # 5–9 s, meio 7 s

        self.db.clear_episode_shots(self.ep)
        # O corte novo partiu a cena em duas: o favorito vai pra que tem o 7 s.
        novos = self._cortar([0, 2, 5, 6.5, 9, 12])

        self.assertEqual(self._favoritas(), {novos[3]: self.lucy})
        with self.db.connect() as c:
            self.assertEqual(c.execute("SELECT count(*) FROM favorito_guardado").fetchone()[0], 0)

    def test_trecho_que_sumiu_fica_guardado(self) -> None:
        ids = self._cortar([0, 2, 5])
        self.db.toggle_favorite(ids[1], self.lucy)  # meio 3,5 s

        self.db.clear_episode_shots(self.ep)
        self._cortar([0, 2])  # o corte novo pulou o fim
        self.assertEqual(self._favoritas(), {})
        # ...mas não se perde: um corte que volte a cobrir o trecho o devolve.
        self.db.clear_episode_shots(self.ep)
        novos = self._cortar([0, 2, 5])
        self.assertEqual(self._favoritas(), {novos[1]: self.lucy})


if __name__ == "__main__":
    unittest.main()
