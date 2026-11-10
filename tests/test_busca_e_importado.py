"""A busca não aceita casamento lixo, e o pacote importado vale quando o anime
não está online. Caso do Levi (11/2026, "Avatar - Os Sete Refúgios").

    python -m unittest discover -s tests -p "test_busca_e_importado.py"
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.pipeline import _bundle_do_importado
from app.providers.anime_provider import _search_variants, _truncado_plausivel
from app.storage.elencos_importados import PASTA
from app.video_ingest import EpisodeInfo


class TestBusca(unittest.TestCase):
    def test_avatar_nao_vira_kings_avatar(self) -> None:
        nome = "Avatar - Os Sete Refúgios"
        candidato = ["Quanzhi Gaoshou", "The King's Avatar"]
        for q in _search_variants(nome):
            if q != nome:
                self.assertFalse(_truncado_plausivel(nome, q, candidato), q)

    def test_nome_inteiro_passa_direto(self) -> None:
        self.assertTrue(_truncado_plausivel("Kusuriya no Hitorigoto", "Kusuriya no Hitorigoto", ["X"]))

    def test_encurtado_que_faz_sentido(self) -> None:
        self.assertTrue(_truncado_plausivel(
            "Black Clover Mahou Tei no Ken Extra", "Black Clover Mahou", ["Black Clover 2nd Season"]))
        self.assertTrue(_truncado_plausivel(
            "Tensei Shitara Slime Datta Ken 4th Season Extra", "Tensei Shitara Slime Datta",
            ["Tensei Shitara Slime Datta Ken 4th Season", ""]))


class TestImportado(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name)
        pasta = self.cache / PASTA / "avatar-seven-havens"
        pasta.mkdir(parents=True)
        (pasta / "elenco.json").write_text(json.dumps({
            "titulo": "Avatar Seven Havens",
            "chaves_titulo": ["avatar os sete refugios", "avatar seven havens"],
            "anilist_ids": [],
            "personagens": [{"nome": "Agam"}, {"nome": "Nisha"}, {"nome": ""}],
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_vira_o_elenco_quando_nao_esta_online(self) -> None:
        info = EpisodeInfo("Avatar - Os Sete Refúgios", 1, 10, Path("x.mkv"))
        b = _bundle_do_importado(self.cache, info)
        self.assertIsNotNone(b)
        self.assertEqual([c.name for c in b.characters], ["Agam", "Nisha"])
        self.assertEqual(b.title, "Avatar - Os Sete Refúgios")
        self.assertEqual(b.cache_id_override, "local-avatar-os-sete-ref-gios")

    def test_vale_pela_pasta_escolhida(self) -> None:
        info = EpisodeInfo("Avatar 7 Havens Ep", 1, 10, Path("x.mkv"), output_folder="Avatar Seven Havens")
        self.assertIsNotNone(_bundle_do_importado(self.cache, info))

    def test_anime_sem_pacote(self) -> None:
        info = EpisodeInfo("Outro Anime", 1, 1, Path("x.mkv"))
        self.assertIsNone(_bundle_do_importado(self.cache, info))


if __name__ == "__main__":
    unittest.main()
