from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# Combined season+episode patterns — strongest signal when both are present.
_SE_PATTERNS = [
    re.compile(r"\b[Ss](\d{1,2})[Ee](\d{1,3})\b"),
    re.compile(r"\bSeason\s*(\d{1,2}).*?Episode\s*(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2})x(\d{1,3})\b"),
]

# Season-only signals (when episode is specified separately).
# Order matters — "4th Season" must win over "Season 01" (where 01 is episode).
_SEASON_PATTERNS = [
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\s*Season\b", re.IGNORECASE), # 4th Season
    re.compile(r"\b[Ss](\d{1,2})\b"),                                   # S4 / s04
    re.compile(r"\bSeason\s*(\d{1,2})\b", re.IGNORECASE),                # Season 4
    re.compile(r"\bTemporada\s*(\d{1,2})\b", re.IGNORECASE),             # Portuguese
]

# Episode-only signals.
_EPISODE_PATTERNS = [
    re.compile(r"\s-\s*(\d{1,3})(?:\s*(?:v\d+)?(?:\s|$))"),               # " - 01"
    re.compile(r"\bEpis[oó]dio\s*(\d{1,3})\b", re.IGNORECASE),             # PT
    re.compile(r"\bEpisode\s*(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\bEp\.?\s*(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\s#(\d{1,3})\b"),
]

# Fansub / release tags and quality markers that should be stripped from names.
_BRACKETED = re.compile(r"\[[^\]]*\]|\{[^}]*\}")
_QUALITY_PAREN = re.compile(
    r"\((?:\d{3,4}p|\d{3,4}x\d{3,4}|BD|BDRip|DVD|DVDRip|WEB|WEB-?DL|HDTV|"
    r"Hi10p?|Ma10p?|x264|x265|HEVC|AAC|FLAC|Dual.?Audio|Multi.?Audio)\)",
    re.IGNORECASE,
)
_TRAILING_JUNK = re.compile(
    r"(?:\b(?:1080p|720p|480p|2160p|4k|x264|x265|HEVC|AAC|FLAC|BluRay|BDRip|WEB-?DL|HDTV)\b.*)$",
    re.IGNORECASE,
)


def clean_title(raw: str) -> str:
    s = _BRACKETED.sub(" ", raw)
    s = _QUALITY_PAREN.sub(" ", s)
    s = _TRAILING_JUNK.sub(" ", s)
    # Filenames often use dots/underscores as word separators. Normalize so
    # the regexes for "S4", "Season 4", etc. match.
    s = re.sub(r"[._]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -_.")
    return s


@dataclass
class EpisodeInfo:
    anime: str
    season: int
    episode: int
    source: Path
    skip_head_seconds: float = 0.0
    skip_tail_seconds: float = 0.0

    @property
    def slug(self) -> str:
        return f"S{self.season:02d}E{self.episode:02d}"


def parse_mmss(text: str) -> float:
    """Parse 'MM:SS' or plain seconds. Returns 0 on empty/invalid."""
    text = (text or "").strip()
    if not text:
        return 0.0
    try:
        if ":" in text:
            parts = text.split(":")
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return float(text)
    except Exception:
        return 0.0


def format_mmss(seconds: float) -> str:
    if seconds <= 0:
        return ""
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


_STRIP_ANIME = [
    re.compile(r"\s*-?\s*\b[Ss]\d{1,2}[Ee]\d{1,3}\b.*$"),
    re.compile(r"\s*-?\s*\b\d+(?:st|nd|rd|th)\s*Season\b.*$", re.IGNORECASE),  # "4th Season ..."
    re.compile(r"\s*-?\s*\bSeason\s*\d+\b.*$", re.IGNORECASE),
    re.compile(r"\s*-?\s*\bTemporada\s*\d+\b.*$", re.IGNORECASE),
    re.compile(r"\s*-?\s*\bEpisode\s*\d+\b.*$", re.IGNORECASE),
    re.compile(r"\s*-?\s*\bEpis[oó]dio\s*\d+\b.*$", re.IGNORECASE),
    re.compile(r"\s*-?\s*\bEp\.?\s*\d+\b.*$", re.IGNORECASE),
    re.compile(r"\s+\b[Ss]\d{1,2}\b.*$"),             # " S4 ..."
    re.compile(r"\s+-\s*\d{1,3}(?:\s|$).*$"),          # " - 01"
    re.compile(r"\s+\d{1,2}x\d{1,3}\b.*$"),            # " 4x01"
]

# Pastas que indicam ORGANIZAÇÃO por temporada — o nome do anime está um
# nível acima ("Mushoku Tensei/Season 1/S01E01-....mkv").
_SEASON_DIR = re.compile(
    r"^(?:season\s*\d*|s\d{1,2}|temporada\s*\d*|specials?|extras?|ovas?|"
    r"filmes?|movies?)$",
    re.IGNORECASE,
)
# Pastas de armazenamento genérico — subir além delas só acha lixo
# (nome de usuário, raiz do disco).
_STORAGE_DIR = re.compile(
    r"^(?:downloads?|desktop|documents?|documentos|videos?|vídeos|animes?|"
    r"series|séries|torrents?|users|home|midia|mídia|media)$",
    re.IGNORECASE,
)

# "S01E01-Título do Episódio" → tag no COMEÇO = arquivo sem nome de anime.
_LEADING_SE_TAG = re.compile(r"^\s*[Ss]\d{1,2}[Ee]\d{1,3}\s*[-–—_.]*\s*")
# "... V2" no fim = versão de fansub, não faz parte de nome nenhum.
_VERSION_SUFFIX = re.compile(r"\s+[Vv]\d+\s*$")


def _name_from_parents(path: Path, max_up: int = 3) -> str:
    """Nome do anime a partir das PASTAS ("Mushoku Tensei/Season 1/ep.mkv").
    Pula pastas de temporada; para de subir ao encontrar pasta de
    armazenamento genérico (Downloads etc. — dali pra cima é nome de usuário)."""
    for parent in list(path.parents)[:max_up]:
        raw = parent.name
        if not raw:
            break
        cand = clean_title(raw)
        if not cand:
            continue
        if _SEASON_DIR.match(cand):
            continue          # "Season 1" → o nome está um nível acima
        if _STORAGE_DIR.match(cand):
            break             # "Downloads" → desiste, acima é lixo
        for pat in _STRIP_ANIME:
            cand = pat.sub("", cand)   # "Mushoku Tensei S1" → "Mushoku Tensei"
        cand = _VERSION_SUFFIX.sub("", cand).strip(" -_.")
        if len(cand) >= 3 and not cand.isdigit():
            return cand
    return ""


def parse_filename(video_path: str | Path) -> EpisodeInfo:
    path = Path(video_path)
    stem = path.stem
    cleaned = clean_title(stem)

    season = 1
    episode = 1

    # Try combined Season+Episode patterns first (strongest signal)
    combined_hit = False
    for pat in _SE_PATTERNS:
        m = pat.search(cleaned)
        if m:
            season = int(m.group(1))
            episode = int(m.group(2))
            combined_hit = True
            break

    # Fall back to independent season + episode detection
    if not combined_hit:
        for pat in _SEASON_PATTERNS:
            m = pat.search(cleaned)
            if m:
                season = int(m.group(1))
                break
        for pat in _EPISODE_PATTERNS:
            m = pat.search(cleaned)
            if m:
                episode = int(m.group(1))
                break

    # Strip anything after the first season/episode marker to isolate anime name.
    name = cleaned
    for pat in _STRIP_ANIME:
        name = pat.sub("", name)
    name = _VERSION_SUFFIX.sub("", name).strip(" -_.")

    # Arquivo SEM nome de anime ("S01E01-Jobless Reincarnation V2.mkv" —
    # caso real): a tag vem primeiro e o strip acima zera o nome. Na ordem:
    # 1) nome da pasta ("Mushoku Tensei/Season 1/arquivo.mkv");
    # 2) o que vem DEPOIS da tag (título do episódio — a busca fuzzy da
    #    AniList costuma resolver, e é infinitamente melhor que mandar
    #    "S01E01-..." pra busca).
    if not name:
        name = _name_from_parents(path)
    if not name:
        name = _VERSION_SUFFIX.sub(
            "", _LEADING_SE_TAG.sub("", cleaned)
        ).strip(" -_.")
    if not name:
        name = cleaned.split(" - ")[0].strip() or stem

    return EpisodeInfo(anime=name, season=season, episode=episode, source=path)
