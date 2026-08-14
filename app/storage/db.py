from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS anime (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anilist_id INTEGER UNIQUE,
    mal_id INTEGER,
    title TEXT NOT NULL,
    title_english TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS episode (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anime_id INTEGER NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
    season INTEGER NOT NULL,
    episode INTEGER NOT NULL,
    source_file TEXT,
    processed_at TIMESTAMP,
    -- Pasta de saída REAL deste episódio. Guardada porque reconstruí-la
    -- (saída / sanitize(anime) / slug) depende do nome que o usuário digitou,
    -- que não fica em lugar nenhum — era o que impedia reabrir um resultado
    -- antigo sem reanalisar.
    output_root TEXT,
    -- '' = episódio, 'OP' = abertura, 'ED' = encerramento. Entra na chave
    -- porque a abertura da 2ª temporada NÃO é o episódio 1 dela: sem esta
    -- coluna as duas dividiam a mesma linha e uma apagava os shots da outra.
    kind TEXT NOT NULL DEFAULT '',
    UNIQUE(anime_id, season, episode, kind)
);

CREATE TABLE IF NOT EXISTS character (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anilist_id INTEGER UNIQUE,
    mal_id INTEGER,
    anime_id INTEGER NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    role TEXT,
    threshold REAL DEFAULT 0.74,
    reference_count INTEGER DEFAULT 0,
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS shot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL REFERENCES episode(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    file TEXT NOT NULL,
    keyframe TEXT,
    start REAL NOT NULL,
    end REAL NOT NULL,
    duration REAL NOT NULL,
    UNIQUE(episode_id, idx)
);

CREATE TABLE IF NOT EXISTS shot_character (
    shot_id INTEGER NOT NULL REFERENCES shot(id) ON DELETE CASCADE,
    character_id INTEGER NOT NULL REFERENCES character(id) ON DELETE CASCADE,
    confidence REAL NOT NULL,
    reviewed INTEGER DEFAULT 0,
    approved INTEGER,
    PRIMARY KEY (shot_id, character_id)
);

CREATE INDEX IF NOT EXISTS idx_shot_char_shot ON shot_character(shot_id);
CREATE INDEX IF NOT EXISTS idx_shot_char_char ON shot_character(character_id);
CREATE INDEX IF NOT EXISTS idx_character_anime ON character(anime_id);

-- Curadoria manual do usuário (remover/mover/aprovar na aba Resultados).
-- Presa ao NÚMERO da cena (shot_idx), não ao id da linha: a reanálise apaga
-- e recria os shots, mas os números são estáveis (cache de detecção), então
-- as decisões sobrevivem e são reaplicadas no fim de toda análise.
CREATE TABLE IF NOT EXISTS manual_override (
    episode_id INTEGER NOT NULL REFERENCES episode(id) ON DELETE CASCADE,
    shot_idx INTEGER NOT NULL,
    character_id INTEGER NOT NULL REFERENCES character(id) ON DELETE CASCADE,
    action TEXT NOT NULL CHECK(action IN ('add','block')),
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (episode_id, shot_idx, character_id)
);
"""


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.executescript(SCHEMA)
            self._migrate(c)

    @staticmethod
    def _migrate(c: sqlite3.Connection) -> None:
        """Migrações aditivas pra bancos que já existem.

        O SCHEMA usa CREATE TABLE IF NOT EXISTS, então coluna nova não chega
        em banco antigo — e o usuário tem análises acumuladas que não podem
        ser perdidas. Só adicionamos colunas anuláveis: bancos antigos seguem
        legíveis por versões antigas do app.
        """
        cols = {r["name"] for r in c.execute("PRAGMA table_info(episode)")}
        if "output_root" not in cols:
            c.execute("ALTER TABLE episode ADD COLUMN output_root TEXT")
        if "kind" not in cols:
            Database._backup_before_rebuild(c)
            Database._migrate_episode_kind(c)

    @staticmethod
    def _backup_before_rebuild(c: sqlite3.Connection) -> None:
        """Cópia do banco antes de reconstruir uma tabela.

        Migração aditiva (ALTER TABLE ADD COLUMN) não precisa disso; derrubar
        e recriar tabela, sim. O acervo do usuário são horas de análise, e um
        arquivo de 2 MB é barato demais pra não fazer.
        """
        import shutil

        origem = Path(
            c.execute("PRAGMA database_list").fetchone()[2] or ""
        )
        if not origem.is_file():
            return
        destino = origem.with_suffix(origem.suffix + ".antes-de-kind.bak")
        if destino.exists():
            return  # já existe de uma tentativa anterior: não sobrescrever
        try:
            shutil.copy2(origem, destino)
        except OSError:
            pass  # sem espaço/permissão: a migração é transacional de todo jeito

    @staticmethod
    def _migrate_episode_kind(c: sqlite3.Connection) -> None:
        """Acrescenta `kind` e troca a chave única de (anime, temporada,
        episódio) para (…, kind).

        Aqui não dá pra usar ALTER TABLE: o SQLite adiciona coluna, mas não
        remove um UNIQUE de tabela. Com o antigo no lugar, gravar a abertura
        da temporada 2 esbarraria no episódio 1 dela. Então é a reconstrução
        documentada pelo SQLite — tabela nova, cópia, troca de nome.

        Os ids são PRESERVADOS na cópia, então `shot` e `manual_override`
        continuam apontando pras mesmas linhas. É por isso que as chaves
        estrangeiras ficam desligadas durante a troca: por um instante a
        tabela `episode` não existe, e com elas ligadas o SQLite apagaria os
        shots em cascata.
        """
        c.execute("PRAGMA foreign_keys = OFF")
        try:
            c.execute("BEGIN IMMEDIATE")
            c.execute(
                """
                CREATE TABLE episode_novo (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    anime_id INTEGER NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
                    season INTEGER NOT NULL,
                    episode INTEGER NOT NULL,
                    source_file TEXT,
                    processed_at TIMESTAMP,
                    output_root TEXT,
                    kind TEXT NOT NULL DEFAULT '',
                    UNIQUE(anime_id, season, episode, kind)
                )
                """
            )
            c.execute(
                "INSERT INTO episode_novo "
                "(id, anime_id, season, episode, source_file, processed_at, "
                " output_root, kind) "
                "SELECT id, anime_id, season, episode, source_file, "
                "       processed_at, output_root, '' FROM episode"
            )
            c.execute("DROP TABLE episode")
            c.execute("ALTER TABLE episode_novo RENAME TO episode")
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise
        finally:
            c.execute("PRAGMA foreign_keys = ON")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- anime / episode ---

    def upsert_anime(
        self,
        anilist_id: int | None,
        title: str,
        mal_id: int | None = None,
        title_english: str | None = None,
    ) -> int:
        with self.connect() as c:
            if anilist_id is not None:
                row = c.execute("SELECT id FROM anime WHERE anilist_id = ?", (anilist_id,)).fetchone()
                if row:
                    c.execute(
                        "UPDATE anime SET title=?, mal_id=?, title_english=? WHERE id=?",
                        (title, mal_id, title_english, row["id"]),
                    )
                    return row["id"]
            row = c.execute("SELECT id FROM anime WHERE title = ? COLLATE NOCASE", (title,)).fetchone()
            if row:
                return row["id"]
            cur = c.execute(
                "INSERT INTO anime(anilist_id, mal_id, title, title_english) VALUES(?,?,?,?)",
                (anilist_id, mal_id, title, title_english),
            )
            return cur.lastrowid

    def upsert_episode(
        self,
        anime_id: int,
        season: int,
        episode: int,
        source: str,
        kind: str = "",
    ) -> int:
        with self.connect() as c:
            row = c.execute(
                "SELECT id FROM episode "
                "WHERE anime_id=? AND season=? AND episode=? AND kind=?",
                (anime_id, season, episode, kind),
            ).fetchone()
            if row:
                c.execute(
                    "UPDATE episode SET source_file=?, processed_at=CURRENT_TIMESTAMP WHERE id=?",
                    (source, row["id"]),
                )
                return row["id"]
            cur = c.execute(
                "INSERT INTO episode(anime_id, season, episode, source_file, "
                "processed_at, kind) VALUES(?,?,?,?,CURRENT_TIMESTAMP,?)",
                (anime_id, season, episode, source, kind),
            )
            return cur.lastrowid

    def set_episode_root(self, episode_id: int, root: str) -> None:
        """Grava a pasta de saída do episódio (ver comentário no SCHEMA)."""
        with self.connect() as c:
            c.execute(
                "UPDATE episode SET output_root=? WHERE id=?", (str(root), episode_id)
            )

    def all_episode_roots(self) -> list[dict]:
        """Todo episódio que tem pasta gravada — `id` e `output_root`.

        Sem `LIMIT` e sem exigir cenas, ao contrário de `recent_episodes`:
        quem junta pastas precisa reapontar TUDO que morava na origem,
        inclusive episódio antigo que não cabe no histórico da tela e análise
        que ficou sem cena. Deixar um pra trás cria caminho quebrado.
        """
        with self.connect() as c:
            rows = c.execute(
                "SELECT id, output_root FROM episode WHERE output_root IS NOT NULL"
            ).fetchall()
            return [dict(r) for r in rows]

    def recent_episodes(self, limit: int = 20) -> list[dict]:
        """Episódios analisados, do mais recente pro mais antigo.

        Só os que têm `output_root` gravado e ao menos um shot — sem isso a
        lista ofereceria entradas que não abrem (análises anteriores a esta
        coluna, ou runs que morreram no meio).
        """
        with self.connect() as c:
            rows = c.execute(
                "SELECT e.id, e.season, e.episode, e.kind, e.output_root, e.processed_at, "
                "       a.title AS anime_title, "
                "       (SELECT COUNT(*) FROM shot s WHERE s.episode_id = e.id) AS shot_count "
                "FROM episode e JOIN anime a ON a.id = e.anime_id "
                "WHERE e.output_root IS NOT NULL "
                "  AND EXISTS (SELECT 1 FROM shot s WHERE s.episode_id = e.id) "
                "ORDER BY e.processed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def clear_episode_shots(self, episode_id: int) -> None:
        with self.connect() as c:
            c.execute("DELETE FROM shot WHERE episode_id=?", (episode_id,))

    # --- characters ---

    def upsert_character(
        self,
        anime_id: int,
        name: str,
        anilist_id: int | None,
        mal_id: int | None = None,
        role: str | None = None,
    ) -> int:
        with self.connect() as c:
            if anilist_id is not None:
                row = c.execute(
                    "SELECT id FROM character WHERE anilist_id = ?", (anilist_id,)
                ).fetchone()
                if row:
                    c.execute(
                        "UPDATE character SET name=?, role=?, mal_id=? WHERE id=?",
                        (name, role, mal_id, row["id"]),
                    )
                    return row["id"]
            row = c.execute(
                "SELECT id FROM character WHERE anime_id=? AND name=? COLLATE NOCASE",
                (anime_id, name),
            ).fetchone()
            if row:
                return row["id"]
            # Mesmo personagem escrito de outro jeito ("Tempest, Rimuru" ≡
            # "Rimuru Tempest"; "Rimuru" do batismo quando inambíguo) reusa
            # a linha existente — senão cada formato de fonte criava um
            # personagem próprio no banco e nos Resultados.
            from ..naming import find_token_match
            rows = c.execute(
                "SELECT id, name FROM character WHERE anime_id=?", (anime_id,)
            ).fetchall()
            match = find_token_match(name, [r["name"] for r in rows])
            if match is not None:
                for r in rows:
                    if r["name"] == match:
                        return r["id"]
            cur = c.execute(
                "INSERT INTO character(anilist_id, mal_id, anime_id, name, role) VALUES(?,?,?,?,?)",
                (anilist_id, mal_id, anime_id, name, role),
            )
            return cur.lastrowid

    def set_character_embedding(
        self, character_id: int, embedding_bytes: bytes, reference_count: int
    ) -> None:
        with self.connect() as c:
            c.execute(
                "UPDATE character SET embedding=?, reference_count=? WHERE id=?",
                (embedding_bytes, reference_count, character_id),
            )

    def get_characters_for_anime(self, anime_id: int) -> list[dict]:
        with self.connect() as c:
            rows = c.execute(
                "SELECT id, anilist_id, mal_id, name, role, threshold, reference_count, embedding "
                "FROM character WHERE anime_id=? ORDER BY role, name",
                (anime_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def set_character_threshold(self, character_id: int, threshold: float) -> None:
        with self.connect() as c:
            c.execute("UPDATE character SET threshold=? WHERE id=?", (threshold, character_id))

    # --- shots ---

    def insert_shot(
        self,
        episode_id: int,
        idx: int,
        file: str,
        keyframe: str | None,
        start: float,
        end: float,
    ) -> int:
        with self.connect() as c:
            cur = c.execute(
                "INSERT INTO shot(episode_id, idx, file, keyframe, start, end, duration) "
                "VALUES(?,?,?,?,?,?,?)",
                (episode_id, idx, file, keyframe, start, end, end - start),
            )
            return cur.lastrowid

    def delete_shots(self, shot_ids: list[int]) -> None:
        """Remove shots e, por ON DELETE CASCADE, as atribuições deles."""
        if not shot_ids:
            return
        marks = ",".join("?" for _ in shot_ids)
        with self.connect() as c:
            c.execute(f"DELETE FROM shot WHERE id IN ({marks})", tuple(shot_ids))

    def assign_character(self, shot_id: int, character_id: int, confidence: float) -> None:
        with self.connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO shot_character(shot_id, character_id, confidence) "
                "VALUES(?,?,?)",
                (shot_id, character_id, confidence),
            )

    def shots_for_character(
        self, character_id: int, episode_id: int | None = None
    ) -> list[dict]:
        """Return shots tagged with this character.
        If `episode_id` is set, restricts to that episode (prevents shots
        from other runs from leaking into the current view)."""
        query = (
            "SELECT s.id, s.idx, s.file, s.keyframe, s.start, s.end, s.duration, "
            "sc.confidence, sc.approved "
            "FROM shot s JOIN shot_character sc ON sc.shot_id = s.id "
            "WHERE sc.character_id = ?"
        )
        args: list = [character_id]
        if episode_id is not None:
            query += " AND s.episode_id = ?"
            args.append(episode_id)
        query += " ORDER BY sc.confidence DESC"
        with self.connect() as c:
            rows = c.execute(query, args).fetchall()
            return [dict(r) for r in rows]

    def shots_for_episode(self, episode_id: int) -> list[dict]:
        with self.connect() as c:
            rows = c.execute(
                "SELECT id, idx, file, keyframe, start, end, duration FROM shot "
                "WHERE episode_id=? ORDER BY idx",
                (episode_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def characters_in_shot(self, shot_id: int) -> list[dict]:
        with self.connect() as c:
            rows = c.execute(
                """SELECT c.id, c.name, sc.confidence, sc.approved
                   FROM shot_character sc
                   JOIN character c ON c.id = sc.character_id
                   WHERE sc.shot_id = ?
                   ORDER BY sc.confidence DESC""",
                (shot_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def assignments_for_episode(self, episode_id: int) -> dict[int, list[dict]]:
        """Todas as atribuições do episódio de uma vez: {shot_id: [{id, name,
        confidence, approved}, ...]} ordenadas por confiança. Evita 1 query
        por shot na hora de montar as pastas."""
        with self.connect() as c:
            rows = c.execute(
                """SELECT sc.shot_id, c.id, c.name, sc.confidence, sc.approved
                   FROM shot_character sc
                   JOIN character c ON c.id = sc.character_id
                   JOIN shot s ON s.id = sc.shot_id
                   WHERE s.episode_id = ?
                   ORDER BY sc.confidence DESC""",
                (episode_id,),
            ).fetchall()
        out: dict[int, list[dict]] = {}
        for r in rows:
            out.setdefault(r["shot_id"], []).append(
                {
                    "id": r["id"],
                    "name": r["name"],
                    "confidence": r["confidence"],
                    "approved": r["approved"],
                }
            )
        return out

    # --- reanálise: substituir vs adicionar ---

    def has_analysis(
        self, source: str, anime_title: str, season: int, episode: int,
        kind: str = "",
    ) -> bool:
        """True se este episódio já tem atribuições salvas (pra UI perguntar
        'substituir ou adicionar?'). Casa por arquivo fonte OU por
        título+temporada+episódio.

        `kind` entra na comparação junto: a abertura da 2ª temporada não é o
        episódio 1 dela, e sem isto analisar a abertura perguntava
        'substituir ou somar?' apontando pro episódio errado."""
        with self.connect() as c:
            row = c.execute(
                """SELECT 1 FROM shot_character sc
                   JOIN shot s ON s.id = sc.shot_id
                   JOIN episode e ON e.id = s.episode_id
                   LEFT JOIN anime a ON a.id = e.anime_id
                   WHERE (e.source_file = ? AND e.kind = ?)
                      OR (a.title = ? COLLATE NOCASE
                          AND e.season = ? AND e.episode = ? AND e.kind = ?)
                   LIMIT 1""",
                (source, kind, anime_title, season, episode, kind),
            ).fetchone()
            return row is not None

    def assignments_snapshot(self, episode_id: int) -> list[dict]:
        """Foto das atribuições atuais POR NÚMERO de cena (sobrevive ao
        clear_episode_shots) — insumo do modo 'adicionar' da reanálise."""
        with self.connect() as c:
            rows = c.execute(
                """SELECT s.idx AS shot_idx, sc.character_id, sc.confidence,
                          sc.reviewed, sc.approved
                   FROM shot_character sc
                   JOIN shot s ON s.id = sc.shot_id
                   WHERE s.episode_id = ?""",
                (episode_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def merge_assignment(
        self,
        shot_id: int,
        character_id: int,
        confidence: float,
        reviewed: int = 0,
        approved: int | None = None,
    ) -> None:
        """Devolve uma atribuição antiga SEM sobrescrever a nova (a análise
        recente ganha quando o par já existe)."""
        with self.connect() as c:
            c.execute(
                "INSERT OR IGNORE INTO shot_character"
                "(shot_id, character_id, confidence, reviewed, approved) "
                "VALUES(?,?,?,?,?)",
                (shot_id, character_id, confidence, reviewed, approved),
            )

    # --- curadoria manual persistente ---

    def record_manual(
        self,
        episode_id: int,
        shot_idx: int,
        character_id: int,
        action: str,
        confidence: float | None = None,
    ) -> None:
        """Grava uma decisão manual ('add' ou 'block') pra cena shot_idx.
        REPLACE: a decisão mais recente pro mesmo par vence (ex.: removeu,
        depois moveu de volta → o 'add' substitui o 'block')."""
        with self.connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO manual_override"
                "(episode_id, shot_idx, character_id, action, confidence) "
                "VALUES(?,?,?,?,?)",
                (episode_id, shot_idx, character_id, action, confidence),
            )

    def manual_overrides(self, episode_id: int) -> list[dict]:
        with self.connect() as c:
            rows = c.execute(
                "SELECT shot_idx, character_id, action, confidence "
                "FROM manual_override WHERE episode_id = ?",
                (episode_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def assign_character_manual(
        self, shot_id: int, character_id: int, confidence: float | None
    ) -> None:
        """Atribuição vinda da curadoria manual: entra revisada e aprovada,
        o que a protege do drop por poucos-shots."""
        with self.connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO shot_character"
                "(shot_id, character_id, confidence, reviewed, approved) "
                "VALUES(?,?,?,1,1)",
                (shot_id, character_id, float(confidence or 1.0)),
            )

    def drop_low_count_character(self, episode_id: int, character_id: int) -> None:
        """Remove as atribuições AUTOMÁTICAS deste personagem no episódio
        (poucos shots = provável ruído). Escopado ao episódio — não mexe em
        outros eps — e poupa linhas aprovadas/manuais."""
        with self.connect() as c:
            c.execute(
                "DELETE FROM shot_character WHERE character_id = ? "
                "AND (approved IS NULL OR approved = 0) "
                "AND shot_id IN (SELECT id FROM shot WHERE episode_id = ?)",
                (character_id, episode_id),
            )

    def set_assignment_review(self, shot_id: int, character_id: int, approved: bool) -> None:
        with self.connect() as c:
            c.execute(
                "UPDATE shot_character SET reviewed=1, approved=? WHERE shot_id=? AND character_id=?",
                (1 if approved else 0, shot_id, character_id),
            )

    def remove_shot_character(self, shot_id: int, character_id: int) -> None:
        with self.connect() as c:
            c.execute(
                "DELETE FROM shot_character WHERE shot_id = ? AND character_id = ?",
                (shot_id, character_id),
            )

    def move_shot_to_character(
        self, shot_id: int, old_character_id: int, new_character_id: int, confidence: float | None = None
    ) -> None:
        """Reassign a shot from one character to another (manual correction).
        Preserves confidence if the new pair already exists."""
        with self.connect() as c:
            c.execute(
                "DELETE FROM shot_character WHERE shot_id = ? AND character_id = ?",
                (shot_id, old_character_id),
            )
            existing = c.execute(
                "SELECT 1 FROM shot_character WHERE shot_id = ? AND character_id = ?",
                (shot_id, new_character_id),
            ).fetchone()
            if existing is None:
                c.execute(
                    "INSERT INTO shot_character(shot_id, character_id, confidence, reviewed, approved) "
                    "VALUES(?,?,?,1,1)",
                    (shot_id, new_character_id, float(confidence or 1.0)),
                )
            else:
                c.execute(
                    "UPDATE shot_character SET reviewed=1, approved=1 "
                    "WHERE shot_id=? AND character_id=?",
                    (shot_id, new_character_id),
                )
