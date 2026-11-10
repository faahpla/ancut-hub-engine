"""O mesmo anime não vira duas pastas por causa do nome do arquivo.

Caso do Levi (11/2026): "Avatar - Os Sete Refúgios S01E10.mkv" e
"Avatar.Seven.Havens.S01E07.mkv" viraram dois animes, com os mesmos 8
personagens duplicados. O pacote importado respondia pelos dois nomes.

    python -m unittest discover -s tests -p "test_mesmo_anime.py"
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.pipeline import Pipeline, _bundle_do_importado
from app.storage.anime_folders import AnimeFolderStore
from app.storage.db import Database
from app.storage.elencos_importados import PASTA, anime_ligado, animes_do_pacote, listar
from app.storage.juntar_animes import fundir_animes_locais, juntar
from app.video_ingest import EpisodeInfo

PT = "Avatar - Os Sete Refúgios"
EN = "Avatar Seven Havens"
ELENCO = ["Agam", "Daemon", "Datu", "Geet", "Jae", "Karthik", "Nisha", "Pavi"]


class TestMesmoAnime(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        raiz = Path(self.tmp.name)
        self.cache, self.saida = raiz / "cache", raiz / "saida"
        self.db = Database(self.cache / "index.db")
        self.animes = {}
        for titulo, eps in ((PT, [10, 11, 12, 13]), (EN, [7, 8, 9])):
            aid = self.db.upsert_anime(anilist_id=None, title=titulo)
            self.animes[titulo] = aid
            chars = {n: self.db.upsert_character(anime_id=aid, name=n, anilist_id=None) for n in ELENCO}
            for n in eps:
                root = self.saida / titulo / f"S01E{n:02d}"
                (root / "shots").mkdir(parents=True)
                eid = self.db.upsert_episode(aid, 1, n, f"{titulo} {n}.mkv", "", root=str(root))
                self.db.set_episode_root(eid, str(root))
                sid = self.db.insert_shot(eid, 0, "shots/0000.mp4", None, 0, 2)
                if titulo == EN:  # as marcações estão do lado "errado"
                    self.db.assign_character(sid, chars["Agam"], 0.9)
                    self.db.toggle_favorite(sid, chars["Agam"])
        pasta = self.cache / PASTA / "avatar-seven-havens"
        pasta.mkdir(parents=True)
        (pasta / "elenco.json").write_text(json.dumps({
            "titulo": EN, "chaves_titulo": ["avatar seven havens", "avatar os sete refugios"],
            "anilist_ids": [], "personagens": [{"nome": n} for n in ELENCO],
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _pipeline(self) -> Pipeline:
        p = Pipeline.__new__(Pipeline)
        p.db = self.db
        p.cfg = SimpleNamespace(cache_path=self.cache)
        return p

    def test_pacote_sabe_que_sao_o_mesmo(self) -> None:
        el = listar(self.cache)[0]
        self.assertEqual(set(animes_do_pacote(self.db, el)), {(PT, PT), (EN, EN)})
        self.assertEqual(anime_ligado(self.db, el), (PT, PT))  # o de mais episódios

    def test_arquivo_em_ingles_cai_no_anime_que_existe(self) -> None:
        pastas = AnimeFolderStore(self.cache)
        info = EpisodeInfo(EN, 1, 14, Path("Avatar.Seven.Havens.S01E14.mkv"))
        titulo = self._pipeline()._anime_local_do_pacote(info, "", pastas)
        self.assertEqual(titulo, PT)
        self.assertEqual(pastas.decide(typed=EN), PT)
        b = _bundle_do_importado(self.cache, info, self.db)
        self.assertEqual(b.title, PT)

    def test_juntar_funde_anime_e_personagens(self) -> None:
        plano = juntar(self.saida, EN, PT, self.db)
        self.assertTrue(plano.pode, plano.payload())
        r = fundir_animes_locais(self.db, self.saida / PT)
        self.assertEqual(r, {"fundidos": 1, "personagens": 8})
        with self.db.connect() as c:
            avatar = c.execute("SELECT count(*) FROM anime WHERE title LIKE '%Avatar%'").fetchone()[0]
            agams = c.execute("SELECT id, anime_id FROM character WHERE name='Agam'").fetchall()
            marcas = c.execute(
                "SELECT count(*) FROM shot_character sc JOIN character ch ON ch.id = sc.character_id "
                "WHERE ch.name='Agam'").fetchone()[0]
            favs = c.execute("SELECT count(*) FROM favorite WHERE character_id=?", (agams[0][0],)).fetchone()[0]
            eps = c.execute("SELECT count(*) FROM episode WHERE anime_id=?", (self.animes[PT],)).fetchone()[0]
        self.assertEqual(avatar, 1)
        self.assertEqual(len(agams), 1)
        self.assertEqual(agams[0][1], self.animes[PT])
        self.assertEqual(marcas, 3)  # as 3 marcações do lado inglês vieram junto
        self.assertEqual(favs, 3)
        self.assertEqual(eps, 7)


if __name__ == "__main__":
    unittest.main()
