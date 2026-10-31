"""Pacote de personagens: levar o "quem é quem" de um anime pra outro AnCut.

Ideia do Levi. O FAAH batizou o Apothecary Diaries inteiro no Modo
Descoberta; quem nunca analisou o anime começa do zero — e, sem a API do
MyAnimeList no ar, sem nem a lista de nomes. A análise em si não viaja (os
clipes de cada um saem de arquivos e cortes diferentes), mas o conhecimento
sim: os personagens com nome, os rostos aprovados e o centroide de cada um.

Importado, é como se a fonte online respondesse, e já com as referências
que o FAAH curou: a análise normal acha as fotos na pasta de refs, e o Modo
Descoberta reconhece pelo centroide do banco.

Formato (um .zip):

    elenco.json                      o manifesto (ver `exportar`)
    animes/<cache_id>/metadata.json  o cache do catálogo (elenco + franquia)
    animes/<cache_id>/characters/<personagem>/*.jpg   as referências

O manifesto leva o modelo de embedding: centroide de outro modelo não é
importado (as fotos sim — o outro lado recalcula).
"""

from __future__ import annotations

import base64
import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..references.image_filters import PLACEHOLDERS
from ..references.reference_store import resolve_anime_dir
from .db import Database

FORMATO = "ancut-elenco"
VERSAO = 1
_IMAGENS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class Resumo:
    animes: int = 0
    personagens: int = 0
    fotos: int = 0
    centroides: int = 0
    titulos: list[str] = field(default_factory=list)
    erro: str = ""

    def payload(self) -> dict:
        return {
            "animes": self.animes,
            "characters": self.personagens,
            "photos": self.fotos,
            "centroids": self.centroides,
            "titles": self.titulos,
            "erro": self.erro,
        }


def _cache_id_do_anime(cache_path: Path, anime: dict) -> str | None:
    """A pasta de refs da franquia deste anime (al<raiz>, mal<id> ou local-…)."""
    from ..providers.anime_provider import local_cache_id

    root_dir = cache_path / "anime_db"
    if anime.get("anilist_id"):
        for p in root_dir.glob("*/metadata.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            aid = anime["anilist_id"]
            if aid == d.get("anilist_id") or aid in (d.get("franchise_ids") or []):
                raiz = d.get("franchise_root_id") or d.get("anilist_id")
                if raiz:
                    return f"al{raiz}"
        return f"al{anime['anilist_id']}"
    if anime.get("mal_id"):
        return f"mal{anime['mal_id']}"
    return local_cache_id(anime.get("title") or "")


def exportar(
    db: Database,
    cache_path: Path,
    episode_ids: list[int],
    destino: Path,
    modelo: dict,
) -> Resumo:
    """Grava o pacote com os animes (e franquias) dos episódios dados."""
    resumo = Resumo()
    with db.connect() as c:
        marcas = ",".join("?" * len(episode_ids)) or "NULL"
        animes = [
            dict(r) for r in c.execute(
                "SELECT DISTINCT a.id, a.title, a.title_english, a.anilist_id, a.mal_id "
                f"FROM anime a JOIN episode e ON e.anime_id = a.id WHERE e.id IN ({marcas})",
                tuple(episode_ids),
            )
        ]
    if not animes:
        resumo.erro = "nenhum anime nesses episódios"
        return resumo

    manifesto: dict = {
        "formato": FORMATO,
        "versao": VERSAO,
        "criado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": modelo,
        "animes": [],
    }
    pastas: dict[str, Path] = {}
    for a in animes:
        cache_id = _cache_id_do_anime(cache_path, a)
        personagens = []
        for ch in db.get_characters_for_anime(a["id"]):
            if not (ch.get("name") or "").strip():
                continue
            personagens.append({
                "nome": ch["name"],
                "papel": ch.get("role"),
                "anilist_id": ch.get("anilist_id"),
                "mal_id": ch.get("mal_id"),
                "centroide": (
                    base64.b64encode(ch["embedding"]).decode("ascii")
                    if ch.get("embedding") else None
                ),
            })
        if not personagens:
            continue  # linha sem ninguém batizado (análise que só cortou)
        pasta = resolve_anime_dir(cache_path, cache_id) if cache_id else None
        if pasta is not None and pasta.is_dir():
            pastas[cache_id] = pasta
        manifesto["animes"].append({
            "titulo": a["title"],
            "titulo_ingles": a.get("title_english"),
            "anilist_id": a.get("anilist_id"),
            "mal_id": a.get("mal_id"),
            "cache_id": cache_id,
            "personagens": personagens,
        })
        resumo.animes += 1
        resumo.personagens += len(personagens)
        resumo.centroides += sum(1 for p in personagens if p["centroide"])
        resumo.titulos.append(a["title"])

    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + ".part")
    with zipfile.ZipFile(parcial, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("elenco.json", json.dumps(manifesto, ensure_ascii=False, indent=2))
        for cache_id, pasta in pastas.items():
            meta = pasta / "metadata.json"
            if meta.is_file():
                z.write(meta, f"animes/{cache_id}/metadata.json")
            chars = pasta / "characters"
            if not chars.is_dir():
                continue
            for d in chars.iterdir():
                if not d.is_dir() or d.name.startswith("_"):
                    continue
                for f in d.iterdir():
                    # Só a imagem em si: nada do `_filtered` (preto e branco
                    # que o catálogo mandou) nem o "NO IMAGE" da AniList.
                    if (f.is_file() and f.suffix.lower() in _IMAGENS
                            and f.name.lower() not in PLACEHOLDERS):
                        z.write(f, f"animes/{cache_id}/characters/{d.name}/{f.name}")
                        resumo.fotos += 1
    parcial.replace(destino)
    return resumo


def importar(db: Database, cache_path: Path, pacote: Path, modelo: dict) -> Resumo:
    """Lê o pacote e ensina os personagens a este AnCut.

    Só SOMA: personagem que já existe aqui fica com o centroide dele (o
    conhecimento local ganha), foto com o mesmo nome não é sobrescrita, e o
    cache do catálogo só entra se aqui não houver um.
    """
    resumo = Resumo()
    try:
        z = zipfile.ZipFile(pacote)
    except (OSError, zipfile.BadZipFile) as e:
        resumo.erro = f"não é um pacote de personagens ({e})"
        return resumo
    with z:
        try:
            manifesto = json.loads(z.read("elenco.json").decode("utf-8"))
        except (KeyError, ValueError):
            resumo.erro = "o arquivo não tem o elenco.json — não é um pacote do AnCut"
            return resumo
        if manifesto.get("formato") != FORMATO:
            resumo.erro = "o arquivo não é um pacote de personagens do AnCut"
            return resumo
        if int(manifesto.get("versao") or 0) > VERSAO:
            resumo.erro = "pacote feito por um AnCut mais novo — atualize o app"
            return resumo
        mesmo_modelo = manifesto.get("modelo") == modelo

        # 1) Fotos e cache do catálogo, pasta por pasta de franquia.
        nomes = z.namelist()
        for a in manifesto.get("animes") or []:
            cache_id = a.get("cache_id")
            if not cache_id:
                continue
            pasta = resolve_anime_dir(cache_path, cache_id)
            if not pasta.is_dir():
                # Pasta nova com o nome no formato da casa, "<título> [al…]".
                seguro = "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in a.get("titulo") or cache_id)
                pasta = (cache_path / "anime_db" / f"{seguro} [{cache_id}]"
                         if not cache_id.startswith("local-")
                         else cache_path / "anime_db" / cache_id)
            prefixo = f"animes/{cache_id}/"
            for nome in nomes:
                if not nome.startswith(prefixo) or nome.endswith("/"):
                    continue
                rel = nome[len(prefixo):]
                partes = Path(rel).parts
                # Só o que o formato prevê; nada de "..", nada fora da pasta.
                if ".." in partes:
                    continue
                if rel == "metadata.json":
                    alvo = pasta / "metadata.json"
                    if alvo.exists():
                        continue
                elif len(partes) == 3 and partes[0] == "characters" and Path(rel).suffix.lower() in _IMAGENS:
                    alvo = pasta / rel
                    if alvo.exists():
                        continue
                    resumo.fotos += 1
                else:
                    continue
                alvo.parent.mkdir(parents=True, exist_ok=True)
                alvo.write_bytes(z.read(nome))

        # 2) Banco: o anime e os personagens, com centroide se o modelo bate.
        for a in manifesto.get("animes") or []:
            anime_id = db.upsert_anime(
                anilist_id=a.get("anilist_id"),
                title=a.get("titulo") or "Anime",
                mal_id=a.get("mal_id"),
                title_english=a.get("titulo_ingles"),
            )
            resumo.animes += 1
            resumo.titulos.append(a.get("titulo") or "")
            existentes = {
                (r["name"] or "").lower(): r for r in db.get_characters_for_anime(anime_id)
            }
            for p in a.get("personagens") or []:
                nome = (p.get("nome") or "").strip()
                if not nome:
                    continue
                cid = db.upsert_character(
                    anime_id=anime_id, name=nome,
                    anilist_id=p.get("anilist_id"), mal_id=p.get("mal_id"),
                    role=p.get("papel") or "Main",
                )
                resumo.personagens += 1
                ja_tem = existentes.get(nome.lower())
                if (mesmo_modelo and p.get("centroide")
                        and not (ja_tem and ja_tem.get("embedding"))):
                    db.set_character_embedding(
                        cid, base64.b64decode(p["centroide"]), reference_count=0
                    )
                    resumo.centroides += 1
    return resumo
