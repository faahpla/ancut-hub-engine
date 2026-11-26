"""Episódio com 4 dígitos no nome do arquivo.

Caso do Faah (25/11/2026): "[Anime Time] One Piece - 1053 - Sanji's Mutation...
.mkv" virou o anime "One Piece - 1053 - Sanji's Mutation..." com S01E01 — o
número não casava porque as regras só liam até 3 dígitos. Cada episódio virou um
anime próprio, numa pasta própria.

    python -m unittest discover -s tests -p "test_nome_quatro_digitos.py"
"""

from __future__ import annotations

import unittest

from app.video_ingest import parse_filename


class TestNomeQuatroDigitos(unittest.TestCase):
    def confere(self, arquivo: str, anime: str, temporada: int, episodio: int, tipo: str = "") -> None:
        info = parse_filename(arquivo)
        self.assertEqual(
            (info.anime, info.season, info.episode, info.kind),
            (anime, temporada, episodio, tipo),
            arquivo,
        )

    def test_one_piece_com_titulo_depois(self) -> None:
        self.confere("[Anime Time] One Piece - 1053 - Sanji's Mutation - The Two Arms in Crisis!.mkv", "One Piece", 1, 1053)
        self.confere("[Anime Time] One Piece - 1057 - For Luffy Sanji and Zoro's Oath.mkv", "One Piece", 1, 1057)

    def test_zero_na_frente(self) -> None:
        self.confere("[Anime Time] One Piece - 0925 - Dashing! The Righteous Soba Mask!.mkv", "One Piece", 1, 925)

    def test_release_comum(self) -> None:
        self.confere("[Erai-raws] One Piece - 1178 [1080p CR WEBRip HEVC AAC][MultiSub][8265FD3E].mkv", "One Piece", 1, 1178)

    def test_sxxexxxx(self) -> None:
        self.confere("One Piece S01E1100.mkv", "One Piece", 1, 1100)
        self.confere("Detective Conan - Episode 1150.mkv", "Detective Conan", 1, 1150)

    def test_o_que_ja_funcionava(self) -> None:
        self.confere("[SubsPlease] Sousou no Frieren - 05 (1080p) [ABC12345].mkv", "Sousou no Frieren", 1, 5)
        self.confere("Chainsaw Man S01E04.mkv", "Chainsaw Man", 1, 4)
        self.confere("Mushoku Tensei S2 - 13.mkv", "Mushoku Tensei", 2, 13)

    def test_ano_continua_filme(self) -> None:
        self.confere("Matrix (1999).mkv", "Matrix", 1, 1999, "MOVIE")
        self.confere("Your Name - 2016.mkv", "Your Name", 1, 2016, "MOVIE")


if __name__ == "__main__":
    unittest.main()
