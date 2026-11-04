"""Marcar pelo Explorer: clipe que aparece numa pasta de personagem vira marcação.

    python -m unittest discover -s tests -p "test_explorer_marcar.py"
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from app.storage.db import Database
from app.storage.explorer_sync import aplicar, ler
from app.storage.metadata_writer import build_shot_payload, write_shots_json
from app.storage.organizer import organize_by_character


class MarcarPeloExplorer(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.db = db = Database(tmp / "index.db")
        self.raiz = raiz = tmp / "Cyberpunk" / "S01E01"
        (raiz / "shots").mkdir(parents=True)
        anime = db.upsert_anime(anilist_id=None, title="Cyberpunk")
        self.ep = ep = db.upsert_episode(anime, 1, 1, str(tmp / "ep.mkv"), "", root=str(raiz))
        db.set_episode_root(ep, str(raiz))
        self.ids, payload = [], []
        for i in range(3):
            f = raiz / "shots" / f"{i:04d}.mp4"
            f.write_bytes(b"clipe %d" % i * (i + 1))
            self.ids.append(db.insert_shot(episode_id=ep, idx=i, file=f"shots/{f.name}", keyframe=None,
                                           start=float(i), end=float(i + 1)))
            payload.append(build_shot_payload({"idx": i, "file": f"shots/{f.name}", "keyframe": None,
                                               "start": float(i), "end": float(i + 1)}, "Cyberpunk", 1, 1, []))
        write_shots_json(raiz / "metadata" / "shots.json", payload)
        self.lucy = db.upsert_character(anime_id=anime, name="Lucy", anilist_id=None, mal_id=None, role="Main")
        self.david = db.upsert_character(anime_id=anime, name="David", anilist_id=None, mal_id=None, role="Main")
        db.assign_character(self.ids[0], self.lucy, 0.9)
        db.assign_character(self.ids[1], self.lucy, 0.9)
        organize_by_character(raiz / "shots" / "0000.mp4", raiz, ["Lucy"])
        organize_by_character(raiz / "shots" / "0001.mp4", raiz, ["Lucy"])
        (raiz / "by_character" / "David").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_copiar_e_arrastar(self):
        bc = self.raiz / "by_character"
        # arrastar (mesmo disco = o mesmo arquivo): cena 1 sai da Lucy e vai pro David
        os.replace(bc / "Lucy" / "0001.mp4", bc / "David" / "0001.mp4")
        # copiar (outro arquivo, mesmo nome e tamanho): cena 2 numa pasta que o anime não conhece
        (bc / "Adam Smasher").mkdir()
        shutil.copyfile(self.raiz / "shots" / "0002.mp4", bc / "Adam Smasher" / "0002.mp4")

        m = ler(self.raiz, self.db, self.ep)
        p = m.payload()
        self.assertEqual(sorted((x["shotIdx"], x["character"], x["isNew"]) for x in p["newPairs"]),
                         [(1, "David", False), (2, "Adam Smasher", True)])
        self.assertEqual([(x["shotIdx"], x["character"]) for x in p["unlinkedPairs"]], [(1, "Lucy")])

        r = aplicar(self.db, self.ep, m)
        self.assertEqual(r["added"], 2)
        self.assertEqual([c["name"] for c in self.db.characters_in_shot(self.ids[1])], ["David"])
        self.assertEqual([c["name"] for c in self.db.characters_in_shot(self.ids[2])], ["Adam Smasher"])
        # lembrado pra reanálise
        acoes = {(o["shot_idx"], o["action"]) for o in self.db.manual_overrides(self.ep)}
        self.assertIn((1, "add"), acoes)
        self.assertIn((1, "block"), acoes)
        self.assertIn((2, "add"), acoes)
        # e chega no Dangai
        linhas = json.loads((self.raiz / "metadata" / "shots.json").read_text(encoding="utf-8"))
        self.assertEqual([c["name"] for c in linhas[2]["characters"]], ["Adam Smasher"])

        # depois de aplicar, nada mais a fazer
        self.assertTrue(ler(self.raiz, self.db, self.ep).vazio)

    def test_hardlink_que_o_app_fez_nao_conta_como_novo(self):
        self.assertEqual(ler(self.raiz, self.db, self.ep).payload()["newPairs"], [])


if __name__ == "__main__":
    unittest.main()
