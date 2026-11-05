"""B-roll: nome da pasta, título tirado do arquivo, número livre e conversão.

    python -m unittest discover -s tests -p "test_broll.py"
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.storage.broll import converter, numero_para, planejar_conversao, titulo_do_arquivo
from app.storage.db import Database
from app.storage.temporadas import planejar as planejar_temporada
from app.storage.temporadas import slug
from app.video_ingest import EpisodeInfo, pasta_do_broll


class TestNomes(unittest.TestCase):
    def test_pasta(self) -> None:
        self.assertEqual(pasta_do_broll(3, "Launch Trailer"), "B-roll 03 - Launch Trailer")
        self.assertEqual(pasta_do_broll(12), "B-roll 12")
        # dois-pontos e barra não podem ir pro nome de pasta no Windows
        self.assertEqual(
            pasta_do_broll(1, "Cyberpunk 2077: Phantom Liberty | Trailer?"),
            "B-roll 01 - Cyberpunk 2077 - Phantom Liberty Trailer",
        )
        self.assertEqual(pasta_do_broll(2, "Teaser..."), "B-roll 02 - Teaser")

    def test_slug_igual_nos_dois_lugares(self) -> None:
        info = EpisodeInfo("X", 1, 4, Path("a.mp4"), kind="BROLL", titulo="Gameplay")
        self.assertEqual(info.slug, slug(1, 4, "BROLL", "Gameplay"))
        self.assertEqual(slug(1, 2019, "MOVIE"), "Filme (2019)")

    def test_titulo_do_arquivo(self) -> None:
        t, o = titulo_do_arquivo(
            r"C:\x\Cyberpunk_2077_-_Phantom_Liberty_Official_Trailer [reABCMNGM3w].h264nv.mp4",
            "Cyberpunk 2077",
        )
        self.assertEqual(t, "Phantom Liberty Official Trailer")
        self.assertEqual(o, "https://www.youtube.com/watch?v=reABCMNGM3w")
        t, o = titulo_do_arquivo(r"C:\x\meu video.mkv")
        self.assertEqual((t, o), ("meu video", ""))


class TestBanco(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        raiz = Path(self.tmp.name)
        self.db = Database(raiz / "index.db")
        self.saida = raiz / "saida"
        self.pasta = self.saida / "Cyberpunk 2077"
        self.anime = self.db.upsert_anime(anilist_id=None, title="Cyberpunk 2077")
        self.ids = []
        nomes = [
            "Cyberpunk_2077_Official_Gameplay_Trailer [BO8lX3hDU30].h264nv.mp4",
            "Cyberpunk_2077_Official_E3_2018_Trailer [8X2kIfS6fb8].h264nv.mp4",
        ]
        for n, arquivo in enumerate(nomes, start=1):
            root = self.pasta / f"S01E{n:02d}"
            (root / "shots").mkdir(parents=True)
            eid = self.db.upsert_episode(self.anime, 1, n, rf"C:\raw\{arquivo}", "", root=str(root))
            self.db.set_episode_root(eid, str(root))
            self.ids.append(eid)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_converte_e_numera(self) -> None:
        plano = planejar_conversao(self.db, self.ids)
        self.assertTrue(plano.pode, plano.payload())
        self.assertEqual(
            [m.para_pasta.split("\\")[-1].split("/")[-1] for m in plano.mudancas],
            ["B-roll 01 - Official Gameplay Trailer", "B-roll 02 - Official E3 2018 Trailer"],
        )
        converter(self.db, self.ids)
        self.assertTrue((self.pasta / "B-roll 01 - Official Gameplay Trailer" / "shots").is_dir())
        self.assertFalse((self.pasta / "S01E01").exists())
        linhas = {r["id"]: r for r in self.db.episodes_by_ids(self.ids)}
        r = linhas[self.ids[0]]
        self.assertEqual((r["kind"], r["episode"], r["titulo"]), ("BROLL", 1, "Official Gameplay Trailer"))
        self.assertEqual(r["origem"], "https://www.youtube.com/watch?v=BO8lX3hDU30")

        # o próximo número livre é o 3; o mesmo link volta pra vaga dele
        self.assertEqual(numero_para(self.db, self.pasta), 3)
        self.assertEqual(
            numero_para(self.db, self.pasta, "https://www.youtube.com/watch?v=8X2kIfS6fb8"), 2
        )
        # pasta "B-roll 07" deixada por fora também ocupa o número
        (self.pasta / "B-roll 07 - solta").mkdir()
        self.assertEqual(numero_para(self.db, self.pasta), 8)

        # converter de novo não faz nada
        self.assertEqual(planejar_conversao(self.db, self.ids).erro, "todos já são B-roll")

    def test_mudar_temporada_nao_mexe_em_broll(self) -> None:
        converter(self.db, self.ids[:1])
        plano = planejar_temporada(self.ids, 2, self.db)
        movidos = {m.episode_id for m in plano.mudancas}
        self.assertNotIn(self.ids[0], movidos)
        self.assertIn(self.ids[1], movidos)


if __name__ == "__main__":
    unittest.main()
