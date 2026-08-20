"""Favoritos, agrupados por anime → personagem.

O pedido do FAAH: "Favs → Mushoku Tensei → Rudeus, e lá estariam todos os
clipes favoritados dele".

Por que o favorito guarda o PERSONAGEM junto: sem isso não há como montar o
segundo nível. Uma cena costuma ter mais de um personagem, então "os favoritos
do Rudeus" não é dedutível de "as cenas favoritas" — a informação de qual
deles motivou o favorito só existe no momento do clique, e é ali que ela é
gravada.

O anime sai da PASTA, não do título, pelo mesmo motivo da Biblioteca: o título
varia com a fonte e a mesma franquia responde nomes diferentes. Ver
`storage/anime_folders.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..naming import find_token_match
from .db import Database

SEM_PERSONAGEM = "Sem personagem"


@dataclass
class GrupoFavorito:
    """Um personagem dentro de um anime, com os favoritos dele."""

    personagem: str
    cenas: list[dict] = field(default_factory=list)

    def payload(self) -> dict:
        return {"character": self.personagem, "shots": self.cenas}


@dataclass
class AnimeFavorito:
    pasta: str
    personagens: list[GrupoFavorito] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(len(g.cenas) for g in self.personagens)

    def payload(self) -> dict:
        return {
            "anime": self.pasta,
            "total": self.total,
            "characters": [g.payload() for g in self.personagens],
        }


def _rel(caminho: Path, base: Path) -> str:
    try:
        return str(caminho.resolve().relative_to(base))
    except (ValueError, OSError):
        return ""


def listar(db: Database, output_dir: Path | str) -> list[AnimeFavorito]:
    """Favoritos do acervo inteiro, em anime → personagem → cenas."""
    base = Path(output_dir).resolve()
    animes: dict[str, AnimeFavorito] = {}

    for r in db.favorites():
        raiz = Path(str(r["output_root"] or ""))
        if not raiz.name:
            continue
        arquivo = raiz / str(r["file"] or "")
        kf = raiz / str(r["keyframe"]) if r["keyframe"] else None
        pasta = raiz.parent.name
        nome = str(r["character_name"] or "").strip() or SEM_PERSONAGEM

        anime = animes.get(pasta.lower())
        if anime is None:
            anime = AnimeFavorito(pasta=pasta)
            animes[pasta.lower()] = anime

        # As grafias do mesmo personagem são unidas aqui também, pela mesma
        # regra da busca por personagem: "Greyrat, Rudeus" e "Rudeus" são um
        # só, e favoritos feitos em temporadas diferentes têm que cair juntos.
        alvo = (
            find_token_match(nome, [g.personagem for g in anime.personagens])
            if nome != SEM_PERSONAGEM
            else (SEM_PERSONAGEM if any(g.personagem == SEM_PERSONAGEM for g in anime.personagens) else None)
        )
        grupo = next((g for g in anime.personagens if g.personagem == alvo), None)
        if grupo is None:
            grupo = GrupoFavorito(personagem=nome)
            anime.personagens.append(grupo)

        grupo.cenas.append({
            "id": r["shot_id"],
            "characterId": r["character_id"],
            "idx": r["idx"],
            "file": _rel(arquivo, base),
            "keyframe": _rel(kf, base) if kf else "",
            "absolute": str(arquivo),
            "duration": r["duration"],
            "confidence": r["confidence"],
            "season": r["season"],
            "episode": r["episode"],
            "kind": r["kind"] or "",
            "episodeId": r["episode_id"],
            "favoritedAt": r["created_at"],
        })

    saida = list(animes.values())
    for a in saida:
        # "Sem personagem" por último: é o balaio, não um personagem.
        a.personagens.sort(
            key=lambda g: (g.personagem == SEM_PERSONAGEM, -len(g.cenas), g.personagem.lower())
        )
    saida.sort(key=lambda a: (-a.total, a.pasta.lower()))
    return saida
