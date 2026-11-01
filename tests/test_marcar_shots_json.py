"""Marcação feita depois da análise tem que chegar no shots.json.

O Dangai tira quem está em cada cena do `metadata/shots.json`, não do banco.
Antes, só a análise escrevia esse arquivo: marcar à mão, criar personagem
novo ou tirar um personagem ficava só no AnCut.
"""

import json
import tempfile
import unittest
from pathlib import Path

from app.storage.db import Database
from app.storage.marcar import marcar_por_nome, marcar_varias, remover_personagem
from app.storage.metadata_writer import build_shot_payload, write_shots_json


def _montar(tmp: Path):
    db = Database(tmp / "index.db")
    raiz = tmp / "Cyberpunk" / "S01E01"
    (raiz / "shots").mkdir(parents=True)
    anime = db.upsert_anime(anilist_id=None, title="Cyberpunk Edgerunners")
    ep = db.upsert_episode(anime, 1, 1, str(tmp / "ep.mkv"), "", root=str(raiz))
    db.set_episode_root(ep, str(raiz))
    payload, ids = [], []
    for i in range(3):
        f = raiz / "shots" / f"shot_{i:04d}.mp4"
        f.write_bytes(b"x")
        ids.append(db.insert_shot(
            episode_id=ep, idx=i, file=f"shots/{f.name}", keyframe=None,
            start=float(i), end=float(i + 1),
        ))
        payload.append(build_shot_payload(
            {"idx": i, "file": f"shots/{f.name}", "keyframe": None,
             "start": float(i), "end": float(i + 1)},
            "Cyberpunk Edgerunners", 1, 1, [],
        ))
    write_shots_json(raiz / "metadata" / "shots.json", payload)
    return db, raiz, ep, ids


def _json(raiz: Path) -> list[dict]:
    return json.loads((raiz / "metadata" / "shots.json").read_text(encoding="utf-8"))


class MarcarChegaNoJson(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_personagem_novo_chega_no_json(self):
        db, raiz, ep, ids = _montar(self.tmp)
        marcar_por_nome(db, ids[:2], "Adam Smasher")
        linhas = _json(raiz)
        assert [c["name"] for c in linhas[0]["characters"]] == ["Adam Smasher"]
        assert [c["name"] for c in linhas[1]["characters"]] == ["Adam Smasher"]
        assert linhas[2]["characters"] == []
        # e a pasta do personagem, que é onde o FAAH trabalha
        assert (raiz / "by_character" / "Adam Smasher" / "shot_0000.mp4").exists()


    def test_mesmo_nome_escrito_diferente_reusa(self):
        db, raiz, ep, ids = _montar(self.tmp)
        a = marcar_por_nome(db, [ids[0]], "Adam Smasher")
        b = marcar_por_nome(db, [ids[1]], "smasher  adam")
        assert a["characterId"] == b["characterId"]

    def test_par_e_remocao_chegam_no_json(self):
        db, raiz, ep, ids = _montar(self.tmp)
        lucy = marcar_por_nome(db, [ids[0]], "Lucy")["characterId"]
        marcar_por_nome(db, [ids[0]], "David Martinez")
        assert _json(raiz)[0]["pairs"] == ["David Martinez+Lucy"]

        marcar_varias(db, [ids[0]], lucy, remover=True)
        assert [c["name"] for c in _json(raiz)[0]["characters"]] == ["David Martinez"]
        assert _json(raiz)[0]["pairs"] == []

        david = db.characters_in_shot(ids[0])[0]["id"]
        remover_personagem(db, ep, int(david))
        assert _json(raiz)[0]["characters"] == []


if __name__ == "__main__":
    unittest.main()
