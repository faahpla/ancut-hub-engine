"""Juntar cenas: os personagens ficam, e a junção sobrevive à reanálise.

    python -m unittest discover -s tests -p "test_juncoes.py"
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.shot_detection import ShotBounds
from app.storage import juncoes


class Janelas(unittest.TestCase):
    def test_colapsa_pelo_meio_da_cena(self):
        shots = [ShotBounds(0, 0.0, 2.0), ShotBounds(1, 2.0, 3.0), ShotBounds(2, 3.0, 5.0), ShotBounds(3, 5.0, 6.0)]
        saida = juncoes.aplicar(shots, [{"inicio": 2.0, "fim": 5.0}])
        self.assertEqual([(s.idx, s.start, s.end) for s in saida], [(0, 0.0, 2.0), (1, 2.0, 5.0), (3, 5.0, 6.0)])

    def test_fronteira_que_andou_um_pouco_continua_dentro(self):
        # a redetecção moveu o corte 2 quadros: o meio da cena ainda cai na janela
        shots = [ShotBounds(0, 0.0, 2.08), ShotBounds(1, 2.08, 3.0), ShotBounds(2, 3.0, 5.0)]
        saida = juncoes.aplicar(shots, [{"inicio": 2.0, "fim": 5.0}])
        self.assertEqual([s.idx for s in saida], [0, 1])
        self.assertEqual(saida[1].end, 5.0)

    def test_juntar_de_novo_funde_as_janelas(self):
        with tempfile.TemporaryDirectory() as t:
            raiz = Path(t)
            juncoes.registrar(raiz, 2.0, 4.0, 1, 2, "0001-0002.mp4")
            juncoes.registrar(raiz, 2.0, 6.0, 1, 3, "0001-0003.mp4")
            j = juncoes.ler(raiz)
            self.assertEqual(len(j), 1)
            self.assertEqual((j[0]["inicio"], j[0]["fim"], j[0]["de"], j[0]["ate"], j[0]["arquivo"]),
                             (2.0, 6.0, 1, 3, "0001-0003.mp4"))

    def test_clipe_juntado_vira_o_da_cena_antes_do_corte(self):
        with tempfile.TemporaryDirectory() as t:
            shots = Path(t)
            (shots / "0001-0003.mp4").write_bytes(b"x")
            juncoes.preparar_arquivos(shots, [{"inicio": 2.0, "fim": 6.0, "de": 1, "ate": 3, "arquivo": "0001-0003.mp4"}])
            self.assertTrue((shots / "0001.mp4").exists())
            self.assertFalse((shots / "0001-0003.mp4").exists())


def _ffmpeg() -> str:
    from app.ffmpeg_locate import ffmpeg_binary

    return str(ffmpeg_binary())


class JuntarDeVerdade(unittest.TestCase):
    """O comando `merge` com clipes de verdade (vídeo sintético) e banco temporário."""

    def test_personagens_ficam_e_a_juncao_e_anotada(self):
        from app import headless
        from app.storage.db import Database
        from app.storage.metadata_writer import build_shot_payload, write_shots_json

        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            db = Database(tmp / "index.db")
            raiz = tmp / "Anime" / "S01E01"
            (raiz / "shots").mkdir(parents=True)
            (raiz / "keyframes").mkdir()
            anime = db.upsert_anime(anilist_id=None, title="Anime")
            ep = db.upsert_episode(anime, 1, 1, str(tmp / "ep.mkv"), "", root=str(raiz))
            db.set_episode_root(ep, str(raiz))
            ids, payload = [], []
            for i in range(3):
                f = raiz / "shots" / f"{i:04d}.mp4"
                subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                                f"testsrc=duration=1:size=160x90:rate=24", "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", str(f)], check=True)
                ids.append(db.insert_shot(episode_id=ep, idx=i, file=f"shots/{f.name}",
                                          keyframe=f"keyframes/{i:04d}_1.jpg", start=float(i), end=float(i + 1)))
                payload.append(build_shot_payload({"idx": i, "file": f"shots/{f.name}", "keyframe": None,
                                                   "start": float(i), "end": float(i + 1)}, "Anime", 1, 1, []))
            write_shots_json(raiz / "metadata" / "shots.json", payload)
            lucy = db.upsert_character(anime_id=anime, name="Lucy", anilist_id=None, mal_id=None, role="Main")
            david = db.upsert_character(anime_id=anime, name="David", anilist_id=None, mal_id=None, role="Main")
            db.assign_character(ids[1], lucy, 0.91)
            db.assign_character(ids[2], david, 0.88)

            cfg = mock.Mock(cache_path=tmp, keyframes_per_shot=3)
            eventos: list[dict] = []
            with mock.patch("app.config.Config.load", return_value=cfg),                     mock.patch.object(headless, "_emit", eventos.append):
                headless._merge_shots(ep, ids[1:])
            evento = eventos[-1]
            self.assertEqual(evento["type"], "merged", eventos)

            novo = evento["shotId"]
            self.assertEqual(sorted(c["name"] for c in db.characters_in_shot(novo)), ["David", "Lucy"])
            self.assertTrue((raiz / "by_character" / "Lucy" / "0001-0002.mp4").exists())
            self.assertTrue((raiz / "by_pair" / "David+Lucy" / "0001-0002.mp4").exists())
            self.assertEqual(len(list((raiz / "keyframes").glob("0001_*.jpg"))), 3)
            j = juncoes.ler(raiz)
            self.assertEqual((j[0]["inicio"], j[0]["fim"], j[0]["de"], j[0]["ate"]), (1.0, 3.0, 1, 2))
            linha = next(x for x in json.loads((raiz / "metadata" / "shots.json").read_text(encoding="utf-8"))
                         if x["shot_id"] == "0001")
            self.assertEqual(sorted(c["name"] for c in linha["characters"]), ["David", "Lucy"])


if __name__ == "__main__":
    unittest.main()
