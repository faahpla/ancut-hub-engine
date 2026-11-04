"""Quem fala em cada cena (legenda) e a busca de cena por descrição.

    python -m unittest discover -s tests -p "test_falas_e_busca.py"
"""

import unittest

from app.falas import falantes_por_cena, ler_ass
from app.visao import para_ingles, separar_nomes

ASS = """[Script Info]
Title: teste

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.50,Default,MAGNA,0,0,0,,Ei, Asta!
Dialogue: 0,0:00:04.00,0:00:05.00,Default,,0,0,0,,(placa sem nome)
Dialogue: 0,0:00:06.00,0:00:09.00,Default,ASTA; ZORA,0,0,0,,Juntos, ao mesmo tempo
Dialogue: 0,0:00:10.00,0:00:10.20,Default,DANTE,0,0,0,,Hm.
Dialogue: 0,0:00:12.00,0:00:13.00,Default,Narrator,0,0,0,,Enquanto isso...
"""


class Legenda(unittest.TestCase):
    def test_le_quem_fala_e_quando(self):
        falas = ler_ass(ASS)
        self.assertEqual(len(falas), 5)
        self.assertEqual((falas[0]["inicio"], falas[0]["fim"], falas[0]["quem"]), (1.0, 3.5, "MAGNA"))
        self.assertEqual(falas[1]["quem"], "")

    def test_quem_fala_em_cada_cena(self):
        falas = [f for f in ler_ass(ASS) if f["quem"]]
        elenco = {1: "Magna", 2: "Asta", 3: "Zora", 4: "Dante"}
        cenas = [(10, 0.0, 3.0), (11, 3.0, 5.5), (12, 5.5, 9.5), (13, 9.9, 11.0), (14, 11.5, 14.0)]
        r = falantes_por_cena(falas, cenas, elenco)
        self.assertEqual(set(r[10]), {1})          # MAGNA, em maiúscula, é a Magna
        self.assertNotIn(11, r)                     # meio segundo da Magna + placa sem nome
        self.assertEqual(set(r[12]), {2, 3})       # "ASTA; ZORA" são dois
        self.assertNotIn(13, r)                     # 0,2 s de fala numa cena de 1,1 s não conta
        self.assertNotIn(14, r)                     # narrador não é do elenco


class Busca(unittest.TestCase):
    def test_traduz_o_que_se_ve(self):
        self.assertEqual(para_ingles("luta na chuva"), "fight in the rain")
        self.assertEqual(para_ingles("pôr do sol na praia"), "sunset in the beach")
        self.assertEqual(para_ingles("explosões"), "explosions")
        self.assertEqual(para_ingles("fight in the rain"), "fight in the rain")

    def test_nome_vira_filtro(self):
        nomes = ["Lucy", "David Martinez", "Rudeus Greyrat", "Rosa"]
        self.assertEqual(separar_nomes("Lucy chorando", nomes), (["Lucy"], "chorando"))
        self.assertEqual(separar_nomes("David Martinez correndo", nomes), (["David Martinez"], "correndo"))
        # palavra do glossário não é tomada por nome: "rosa" aqui é a cor
        self.assertEqual(separar_nomes("cabelo rosa", nomes), ([], "cabelo rosa"))


if __name__ == "__main__":
    unittest.main()
