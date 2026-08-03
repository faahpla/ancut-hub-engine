from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .ai_review import (
    NavyAIClient,
    QuotaExhaustedError,
    classify_face_crops,
    classify_frame,
    classify_group,
)
from .config import Config
from .keyframe_extractor import cut_all_shots
from .matching.character_matcher import (
    CharacterEntry,
    CharacterMatcher,
    build_centroid,
    build_prototypes,
)
from .matching.cooccurrence import count_pairs
from .matching.credit_detector import is_credits_frame
from .matching.embedding_engine import EmbeddingEngine, from_bytes, to_bytes
from .matching.face_detector import (
    MODEL_SIGNATURE,
    AnimeFaceDetector,
    crops_from_boxes,
    ensure_cascade,
    smart_portrait_crop,
)
from .matching.feature_cache import FeatureCache
# Lightweight types re-exported here so callers can keep `from .pipeline
# import AIMode, PipelineResult, STAGES` without dragging in torch just to
# read a type name. UI modules should prefer `from .pipeline_types import ...`.
from .matching.face_clustering import FaceObservation, cluster_faces, pick_representatives
from .matching.group_rescue import decide, diverse_representatives, rank_characters
from .matching.second_pass import ShotFaces, build_episode_banks, rescue_unassigned
from .pipeline_types import (
    AIMode,
    DiscoveredGroup,
    DiscoveryResult,
    DiscoveryShot,
    InsufficientRefsError,
    PipelineResult,
    ProgressCb,
    STAGES,
    StageTimer,
)
from .providers.anime_provider import (
    AnimeBundle,
    AnimeProvider,
    CharacterRef,
    local_cache_id,
)
from .references.reference_store import ReferenceStore
from .shot_detection import ShotBounds, detect_shots
from .storage.db import Database
from .storage.metadata_writer import build_shot_payload, write_characters_json, write_shots_json
from .storage.anime_folders import AnimeFolderStore
from .storage.organizer import clear_grouping, organize_by_character, organize_by_pair, sanitize
from .video_ingest import EpisodeInfo


def _clip_needs_download(model_name: str, pretrained: str) -> bool:
    """Return True if the open_clip weights aren't in the local cache yet.
    Shows a helpful 'first-run download' message when we know we'd hit the
    network. False on lookup failure — better to say nothing than to warn
    incorrectly. open_clip today caches through huggingface_hub for the
    OpenAI-hosted checkpoints (repo `timm/vit_*_clip_224.openai`), so we
    ask huggingface_hub whether the file is already there.
    """
    try:
        import open_clip
        from huggingface_hub import try_to_load_from_cache

        cfg = open_clip.get_pretrained_cfg(model_name, pretrained) or {}
        hf_hub = cfg.get("hf_hub") if isinstance(cfg, dict) else None
        if not hf_hub:
            return False
        repo_id = hf_hub.rstrip("/")
        for filename in ("open_clip_model.safetensors", "open_clip_pytorch_model.bin"):
            hit = try_to_load_from_cache(repo_id=repo_id, filename=filename)
            if hit and hit != "_CACHED_NO_EXIST":
                return False
        return True
    except Exception:
        return False


def _cache_id_for(bundle) -> str:
    """Chave de cache do anime — a da FRANQUIA quando ela existe.

    Todas as temporadas dividem o mesmo banco de referências, e é esta mesma
    chave que amarra as temporadas numa pasta só de saída.
    """
    if bundle.cache_id_override:
        return bundle.cache_id_override   # banco local (Modo Descoberta)
    if bundle.franchise_root_id:
        return f"al{bundle.franchise_root_id}"
    if bundle.anilist_id:
        return f"al{bundle.anilist_id}"
    return f"mal{bundle.mal_id}"


def _noop(stage: str, frac: float, msg: str) -> None:
    pass


class Pipeline:
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.cfg.ensure_dirs()
        self.db = Database(self.cfg.cache_path / "index.db")

    def run(
        self,
        info: EpisodeInfo,
        on_progress: ProgressCb | None = None,
        use_ai_recognition: bool = False,
        ai_mode: AIMode | str = AIMode.FULL,
        ai_review_ambiguous: bool = False,
        merge_previous: bool = False,
        benchmark: bool = False,
    ) -> PipelineResult:
        """`benchmark=True`: rodada de MEDIÇÃO, não de produção.

        Duas diferenças, e as duas existem pelo mesmo motivo. A curadoria
        manual do usuário é o GABARITO do benchmark — se a rodada aplicasse
        os bloqueios e as adições dele, ela estaria copiando a resposta da
        prova e tirando 100 sempre. Então aqui a curadoria não entra nem na
        classificação nem no fim.

        E nada é organizado em pasta: medir não pode mexer no acervo. O
        banco é responsabilidade de quem chama (o runner usa uma cópia).
        """
        timer = StageTimer()
        cb = timer.wrap(on_progress or _noop)
        cfg = self.cfg

        cb("parse", 1.0, f"{info.anime} {info.slug}")

        # 3) Anime / characters — ANTES de preparar as pastas, de propósito.
        #
        # Quem decide em que pasta o episódio cai é a identidade do anime, e
        # a identidade só existe depois desta consulta. Enquanto o corte vinha
        # primeiro, a pasta era o nome digitado e nada mais — foi assim que
        # "Tensura", "Tensei Shitara Slime Datta Ken" e "That Time I Got
        # Reincarnated as a Slime" viraram três pastas do mesmo anime.
        #
        # Custa nada trocar a ordem: com o anime já conhecido esta etapa sai
        # do cache em ~0s (ver o disjuntor e o banco parcial na v1.8.1).
        cb("fetch_characters", -1.0, "Consultando AniList + Jikan...")
        provider = AnimeProvider(cfg.cache_path)
        try:
            bundle = provider.resolve(
                info.anime,
                max_characters=cfg.max_characters_per_anime,
                images_per_character=cfg.references_per_character,
                on_status=lambda m: cb("fetch_characters", -1.0, m),
                use_danbooru=cfg.use_danbooru,
                gallery_for_top=cfg.gallery_for_top_characters,
                season=info.season,
            )
        finally:
            provider.close()
        # Fonte fora do ar? Guarda pra avisar — no progresso agora e, se a
        # análise morrer por falta de refs, na mensagem de erro ("tenta de
        # novo mais tarde" só é bom conselho quando é confirmado).
        source_warnings = list(provider.source_warnings)
        for w in source_warnings:
            print(f"[CorteCenas] AVISO de fonte: {w}", flush=True)
        cb("fetch_characters", 1.0, f"{len(bundle.characters)} personagens")

        # A chave da FRANQUIA, não da temporada: no AniList cada temporada é
        # um anime diferente, e o usuário quer uma pasta pro anime inteiro.
        cache_id = _cache_id_for(bundle)
        pastas = AnimeFolderStore(cfg.cache_path)
        pastas.seed_from_history(cfg.cache_path, cfg.output_path)
        pasta_anime = (
            # `sanitize("")` devolve "unknown", não "" — por isso o teste vem
            # antes da chamada, e não depois dela.
            (sanitize(info.output_folder) if info.output_folder else "")
            or pastas.folder_for_name(info.anime)
            or pastas.folder_for_franchise(cache_id)
            or sanitize(info.anime)
        )
        if pasta_anime != sanitize(info.anime):
            print(
                f"[CorteCenas] Pasta do anime: '{pasta_anime}' "
                f"(digitado: '{info.anime}')",
                flush=True,
            )

        episode_root, metadata_dir, cut_results = self._prepare_shots(
            info, cb, pasta_anime
        )
        # Só depois do corte dar certo: uma análise que morreu no meio não
        # deve deixar combinação de pasta pra trás.
        pastas.remember(info.anime, pasta_anime, cache_id)

        anime_id = self.db.upsert_anime(
            anilist_id=bundle.anilist_id,
            mal_id=bundle.mal_id,
            title=bundle.title,
            title_english=bundle.title_english,
        )
        episode_id = self.db.upsert_episode(
            anime_id, info.season, info.episode, str(info.source), info.kind
        )
        # Guarda a pasta de saída real: é o que permite reabrir este resultado
        # depois sem reanalisar (reconstruir o caminho dependia do nome que o
        # usuário digitou, que não fica salvo em lugar nenhum).
        self.db.set_episode_root(episode_id, str(episode_root))

        # Modo "adicionar" da reanálise: fotografa as atribuições atuais
        # ANTES de limpar — elas voltam por cima do resultado novo no final
        # (a análise nova ganha nos empates; bloqueios manuais ganham de tudo).
        merge_snapshot: list[dict] = []
        if merge_previous and not benchmark:
            merge_snapshot = self.db.assignments_snapshot(episode_id)
            if merge_snapshot:
                print(
                    f"[CorteCenas] Modo adicionar: {len(merge_snapshot)} "
                    "atribuições da análise anterior serão preservadas.",
                    flush=True,
                )

        self.db.clear_episode_shots(episode_id)

        # Bloqueios da curadoria manual entram JÁ na classificação (não só na
        # reaplicação final): cena que o usuário removeu não pode ser
        # re-atribuída no meio da análise — senão ela vira fonte da segunda
        # passada e ESPALHA o erro que o usuário corrigiu.
        blocked_pairs: dict[int, set[int]] = {}
        for ov in [] if benchmark else self.db.manual_overrides(episode_id):
            if ov["action"] == "block":
                blocked_pairs.setdefault(int(ov["shot_idx"]), set()).add(
                    int(ov["character_id"])
                )
        if blocked_pairs:
            n_blocked = sum(len(v) for v in blocked_pairs.values())
            print(
                f"[CorteCenas] {n_blocked} bloqueio(s) manual(is) valendo "
                "desde a classificação.",
                flush=True,
            )

        for ch in bundle.characters:
            self.db.upsert_character(
                anime_id=anime_id,
                name=ch.name,
                anilist_id=ch.anilist_id,
                mal_id=ch.mal_id,
                role=ch.role,
            )

        # 4) Download references
        cb("download_refs", -1.0, "Baixando imagens de personagens...")
        ref_store = ReferenceStore(cfg.cache_path)
        # `cache_id` já foi calculado lá em cima: é ele que escolhe a pasta
        # do anime, e a pasta precisa ser decidida antes do corte.

        # Pastas locais de personagem que NÃO batem com o elenco online
        # também são personagens (Modo Descoberta com nome digitado, pastas
        # criadas à mão). Sem isto, as fotos delas ficavam invisíveis — bug
        # real: usuário batizou "Mitsuhime", o MAL chama de "Yukishiro,
        # Mitsuhime", e a análise dizia "sem referências suficientes".
        extra_locals = self._local_only_characters(ref_store, cache_id, bundle)
        for name in extra_locals:
            bundle.characters.append(
                CharacterRef(mal_id=None, anilist_id=None, name=name,
                             role="Main", image_urls=[])
            )
            self.db.upsert_character(
                anime_id=anime_id, name=name,
                anilist_id=None, mal_id=None, role="Main",
            )
        if extra_locals:
            cb("download_refs", -1.0,
               f"+{len(extra_locals)} personagens de pastas locais")

        refs_per_char = ref_store.ensure_references(
            cache_id, bundle, on_status=lambda m: cb("download_refs", -1.0, m)
        )
        cb("download_refs", 1.0, "Refs prontas")

        # === AI-only recognition path (alternative to CLIP) ===
        if use_ai_recognition:
            return self._run_ai_recognition(
                info=info,
                cb=cb,
                anime_id=anime_id,
                episode_id=episode_id,
                episode_root=episode_root,
                metadata_dir=metadata_dir,
                bundle=bundle,
                refs_per_char=refs_per_char,
                cut_results=cut_results,
                ai_mode=AIMode(ai_mode) if not isinstance(ai_mode, AIMode) else ai_mode,
            )

        # 5) Embeddings — modelos LAZY: com os caches de features cheios
        # (reanálise típica), nem o CLIP nem o YOLO chegam a ser carregados
        # e a análise vira só a matemática do matcher.
        get_engine, get_face_det = self._lazy_models(cb)
        feat_meta = self._feature_meta()
        ref_cache = FeatureCache(
            ref_store.anime_dir(cache_id) / "ref_features.npz", feat_meta
        )
        kf_cache = FeatureCache(metadata_dir / "face_cache.npz", feat_meta)

        db_chars = {c["name"]: c for c in self.db.get_characters_for_anime(anime_id)}
        entries: list[CharacterEntry] = []
        total_chars = len(bundle.characters)
        ref_stats: list[tuple[str, int, int]] = []  # (name, ref_count, face_count)
        skipped_few_refs: list[str] = []
        for i, ch in enumerate(bundle.characters, 1):
            cb("embed_refs", i / total_chars, f"Embeddando: {ch.name}")
            paths = refs_per_char.get(ch.name, [])
            if len(paths) < cfg.min_references_per_character:
                ref_stats.append((ch.name, len(paths), 0))
                skipped_few_refs.append(f"{ch.name}({len(paths)})")
                continue

            # Detect faces in each ref (cache primeiro; YOLO em lote só nos
            # arquivos novos). If a face is found, the face crops are used;
            # if not, a smart center-crop of the portrait (removes white
            # margins). Os embeddings viram os protótipos por modo visual
            # (multi_prototype) + um centroide global que vai pro DB pras
            # sugestões da Descoberta.
            emb_parts, faces_found = self._ref_features(
                paths, ref_cache, get_engine, get_face_det
            )
            ref_stats.append((ch.name, len(paths), faces_found))
            if not emb_parts:
                continue
            embs = np.concatenate(emb_parts, axis=0)
            centroid = build_centroid(embs)
            if centroid is None:
                continue
            db_row = db_chars.get(ch.name)
            if not db_row:
                continue
            protos = None
            if cfg.multi_prototype:
                protos = build_prototypes(
                    embs,
                    merge_threshold=cfg.prototype_merge_threshold,
                    max_prototypes=cfg.max_prototypes_per_character,
                )
            self.db.set_character_embedding(
                db_row["id"], to_bytes(centroid), reference_count=len(paths)
            )
            entries.append(
                CharacterEntry(
                    id=db_row["id"],
                    name=ch.name,
                    centroid=centroid,
                    threshold=cfg.default_threshold,
                    prototypes=protos,
                )
            )
        ref_cache.save()
        print(
            "[CorteCenas] Refs por personagem (refs/rostos):",
            ", ".join(f"{n}={rc}/{fc}" for n, rc, fc in ref_stats),
        )
        if skipped_few_refs:
            print(f"[CorteCenas] Ignorados (poucas refs): {', '.join(skipped_few_refs)}")
        if cfg.multi_prototype and entries:
            print(
                "[CorteCenas] Protótipos por personagem:",
                ", ".join(
                    f"{e.name}={1 if e.prototypes is None else len(e.prototypes)}"
                    for e in entries
                ),
            )
        cb("embed_refs", 1.0, f"{len(entries)} personagens com embedding")

        # Zero characters with embeddings = nothing can ever match. Seen in
        # the wild when Jikan's pictures endpoint 504s for every character:
        # each one falls back to a single portrait and the min_references
        # filter drops them all. Analyzing 300+ shots against an empty bank
        # would burn minutes of GPU to deliver a guaranteed-empty result —
        # fail now, with the actual reason and a way out.
        if not entries:
            if source_warnings:
                cause = (
                    "⚠️ CONFIRMADO: o MyAnimeList estava fora do ar durante esta "
                    "análise —\n" + "\n".join(f"• {w}" for w in source_warnings) +
                    "\n\nMuito provavelmente é SÓ isso. O que fazer:\n"
                    "• Tente de novo daqui a alguns minutos — os shots cortados "
                    "ficam em cache, a reanálise vai direto pro banco de "
                    "personagens;\n"
                    "• Ou use o Modo Descoberta agora: ele identifica os rostos "
                    "pelo próprio episódio, sem depender dessas fontes;\n"
                )
            else:
                cause = (
                    "Causa mais comum: as fontes de imagens (Jikan/MyAnimeList) "
                    "estão instáveis ou fora do ar agora. O que fazer:\n"
                    "• Tente de novo mais tarde (os shots cortados ficam em cache, "
                    "a próxima rodada pula direto pro banco de personagens);\n"
                )
            raise InsufficientRefsError(
                "Nenhum personagem ficou com fotos de referência suficientes "
                f"(mínimo {cfg.min_references_per_character} por personagem) — "
                "a análise não teria como identificar ninguém.\n\n"
                + cause +
                "• Ou adicione fotos manualmente: 'Abrir pasta de refs' neste "
                "aviso — cada personagem tem uma subpasta; prints do próprio "
                "episódio funcionam.\n\n"
                "Detalhes por personagem no app.log (Configurações → Abrir "
                "pasta de logs).",
                refs_dir=str(ref_store.anime_dir(cache_id) / "characters"),
            )

        # 1-2 usable characters while others got dropped: the run still
        # works, but the user should know most of the cast is invisible.
        refs_dir_str = str(ref_store.anime_dir(cache_id) / "characters")
        low_refs_warning = None
        if skipped_few_refs and len(entries) <= 2:
            low_refs_warning = (
                f"Só {len(entries)} personagem(ns) tinham fotos de referência "
                f"suficientes — {len(skipped_few_refs)} ficaram de fora e não "
                "podem ser identificados neste episódio."
            )
            if source_warnings:
                low_refs_warning += (
                    "\n\n⚠️ O MyAnimeList estava instável durante esta análise — "
                    "reanalisar mais tarde deve recuperar o elenco completo."
                )

        matcher = CharacterMatcher(entries)

        # 6) Analyze shots — YOLO+CLIP via cache, modelos lazy do passo 5

        per_shot_names: list[list[str]] = []
        shot_db_ids: list[int] = []
        total = len(cut_results)
        shots_with_faces = 0
        credit_shots = 0
        # Modo híbrido: shots que o CLIP deixou SEM personagem mas cuja melhor
        # similaridade chegou perto do threshold — a zona cinzenta que vale
        # uma segunda opinião da IA. Guardamos os crops já extraídos.
        ambiguous: list[dict] = []
        # Segunda passada: embeddings de rosto por shot (posição casa com
        # per_shot_names). Só rostos — o fallback de keyframe inteiro fica de
        # fora do banco (distribuição diferente contaminaria as comparações).
        shot_faces: list[ShotFaces] = []
        kf_hits = kf_total = 0
        for i, (shot, shot_file, kfs) in enumerate(cut_results, 1):
            cb("analyze_shots", i / max(total, 1), f"Shot {i}/{total}")
            main_kf = kfs[len(kfs) // 2] if kfs else None
            shot_id = self.db.insert_shot(
                episode_id=episode_id,
                idx=shot.idx,
                file=str(shot_file.relative_to(episode_root)),
                keyframe=str(main_kf.relative_to(episode_root)) if main_kf else None,
                start=shot.start,
                end=shot.end,
            )
            shot_db_ids.append(shot_id)

            # Leitura preguiçosa: cache cheio = nenhuma imagem aberta.
            imgs_loaded: dict[Path, np.ndarray | None] = {}

            def _img(p: Path) -> np.ndarray | None:
                if p not in imgs_loaded:
                    imgs_loaded[p] = cv2.imread(str(p))
                return imgs_loaded[p]

            # Credit / OP / ED detection — skip shots dominated by text overlay.
            if cfg.skip_credit_shots and kfs:
                credit_count = 0
                for kf_path in kfs:
                    flag = kf_cache.get(kf_path, "credit")
                    if flag is None:
                        img = _img(kf_path)
                        if img is None:
                            continue
                        flag = np.array(
                            [1 if is_credits_frame(img, cfg.credit_edge_threshold) else 0],
                            dtype=np.uint8,
                        )
                        kf_cache.put(kf_path, "credit", flag)
                    if int(flag[0]):
                        credit_count += 1
                if credit_count >= cfg.credit_min_keyframes:
                    credit_shots += 1
                    per_shot_names.append([])
                    shot_faces.append(ShotFaces(len(per_shot_names) - 1, None, []))
                    continue

            # --- Features por keyframe: cache → YOLO+CLIP em lote só nos
            # que faltam. boxes (brutos) e embeddings (crops com padding)
            # ficam pareados 1:1 — é o que permite rematerializar os crops
            # dos duvidosos depois sem guardar JPEG nenhum.
            feats: dict[Path, tuple[np.ndarray, np.ndarray | None]] = {}
            missing: list[Path] = []
            for kf_path in kfs:
                kf_total += 1
                boxes = kf_cache.get(kf_path, "boxes")
                embs_c = kf_cache.get(kf_path, "embs")
                if boxes is not None and (len(boxes) == 0 or embs_c is not None):
                    kf_hits += 1
                    feats[kf_path] = (boxes, embs_c)
                else:
                    missing.append(kf_path)
            if missing:
                good = [(p, _img(p)) for p in missing]
                good = [(p, im) for p, im in good if im is not None]
                if good:
                    batch = get_face_det().crop_faces_batch(
                        [im for _, im in good], pad=cfg.face_crop_padding
                    )
                    flat: list[np.ndarray] = []
                    spans: list[tuple[Path, list, int]] = []
                    for (p, _im), (crops, kept) in zip(good, batch):
                        spans.append((p, kept, len(crops)))
                        flat.extend(crops)
                    embs_all = (
                        get_engine().embed_images(flat)
                        if flat
                        else np.zeros((0, 1), dtype=np.float32)
                    )
                    if flat and len(embs_all) != len(flat):
                        # Pareamento quebrou — refaz keyframe a keyframe
                        # (caro, mas raro) pra não gravar cache torto.
                        for (p, _im), (crops, kept) in zip(good, batch):
                            e = (
                                get_engine().embed_images(crops)
                                if crops
                                else np.zeros((0, 1), dtype=np.float32)
                            )
                            if len(e) != len(kept):
                                continue
                            self._store_kf(kf_cache, feats, p, kept, e)
                    else:
                        off = 0
                        for p, kept, n_crops in spans:
                            e = embs_all[off:off + n_crops]
                            off += n_crops
                            self._store_kf(kf_cache, feats, p, kept, e)

            # Per-keyframe face-based assignments. Tracking votes across
            # keyframes lets us filter out single-frame cameos (a character
            # that shows up in only 1 of 3 keyframes is almost always noise
            # or a flash-through, not a real presence).
            per_kf_assigns: list[dict[int, float]] = []
            had_any_faces = False
            shot_best_sim = 0.0
            shot_boxes_ai: list[tuple[Path, np.ndarray]] = []
            face_embs_kf: list[np.ndarray] = []
            face_refs_kf: list[tuple] = []   # (kf_path, box_idx) por linha de embs
            collect_faces = cfg.second_pass or cfg.cluster_rescue
            for kf_path in kfs:
                boxes, embs = feats.get(kf_path, (None, None))
                if boxes is None or len(boxes) == 0 or embs is None or embs.size == 0:
                    continue
                had_any_faces = True
                if ai_review_ambiguous:
                    shot_boxes_ai.append((kf_path, boxes))
                if collect_faces:
                    face_embs_kf.append(embs)
                    face_refs_kf.extend((kf_path, bi) for bi in range(len(embs)))
                d = dict(matcher.assign_best_per_query(embs, margin=cfg.argmax_margin))
                per_kf_assigns.append(d)
                if ai_review_ambiguous:
                    best = matcher.best_overall(embs)
                    if best is not None:
                        shot_best_sim = max(shot_best_sim, best[1])

            if had_any_faces:
                shots_with_faces += 1

            assigns: list[tuple[int, float]] = []
            useful_kf = len(per_kf_assigns)
            if useful_kf > 0:
                votes: dict[int, int] = {}
                max_conf: dict[int, float] = {}
                for d in per_kf_assigns:
                    for cid, conf in d.items():
                        votes[cid] = votes.get(cid, 0) + 1
                        max_conf[cid] = max(max_conf.get(cid, 0.0), conf)
                required = min(cfg.min_keyframe_votes, useful_kf)
                assigns = sorted(
                    [(cid, max_conf[cid]) for cid, v in votes.items() if v >= required],
                    key=lambda x: x[1],
                    reverse=True,
                )
            elif kfs and not cfg.face_exclusive_when_detected:
                # No faces detected in any keyframe → whole-keyframe fallback.
                # Use a bumped margin/threshold internally to reduce background
                # noise contamination.
                q_parts: list[np.ndarray] = []
                for kf_path in kfs:
                    q = kf_cache.get(kf_path, "kfemb")
                    if q is None:
                        q = get_engine().embed_images([kf_path])
                        if q.size:
                            kf_cache.put(kf_path, "kfemb", q.astype(np.float32))
                    if q.size:
                        q_parts.append(q)
                if q_parts:
                    assigns = matcher.assign_best_per_query(
                        np.concatenate(q_parts, axis=0),
                        margin=max(cfg.argmax_margin, 0.05),
                    )

            # Par (cena, personagem) bloqueado pelo usuário não entra — nem
            # no banco, nem como fonte da segunda passada.
            if assigns and shot.idx in blocked_pairs:
                assigns = [
                    (cid, c) for cid, c in assigns
                    if cid not in blocked_pairs[shot.idx]
                ]

            if assigns:
                names: list[str] = []
                for char_id, conf in assigns:
                    self.db.assign_character(shot_id, char_id, conf)
                    for e in entries:
                        if e.id == char_id:
                            names.append(e.name)
                            break
                per_shot_names.append(names)
            else:
                per_shot_names.append([])
                # Ficou sem dono mas chegou perto? Candidato à revisão da IA.
                # Os crops são rematerializados AQUI, a partir dos boxes —
                # só os duvidosos pagam a releitura da imagem.
                if (
                    ai_review_ambiguous
                    and shot_boxes_ai
                    and cfg.ai_review_low <= shot_best_sim < cfg.default_threshold
                ):
                    crops_jpg: list[bytes] = []
                    for kf_path, boxes in shot_boxes_ai:
                        if len(crops_jpg) >= 5:
                            break
                        img = _img(kf_path)
                        if img is None:
                            continue
                        crops, _kept = crops_from_boxes(
                            img, boxes, cfg.face_crop_padding
                        )
                        for c in crops:
                            if len(crops_jpg) >= 5:
                                break
                            ch_, cw_ = c.shape[:2]
                            scale = 256 / max(ch_, cw_)
                            if scale < 1.0:
                                c = cv2.resize(c, (int(cw_ * scale), int(ch_ * scale)))
                            ok_, enc_ = cv2.imencode(
                                ".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 85]
                            )
                            if ok_:
                                crops_jpg.append(enc_.tobytes())
                    if crops_jpg:
                        ambiguous.append({
                            "pos": len(per_shot_names) - 1,  # índice em per_shot_names
                            "shot": shot,
                            "shot_id": shot_id,
                            "crops": crops_jpg,
                            "best_sim": shot_best_sim,
                        })
            if collect_faces:
                shot_faces.append(
                    ShotFaces(
                        pos=len(per_shot_names) - 1,
                        embs=np.vstack(face_embs_kf) if face_embs_kf else None,
                        assigned=list(assigns),
                        face_refs=face_refs_kf,
                    )
                )
        kf_cache.save()
        print(f"[CorteCenas] Rostos detectados em {shots_with_faces}/{total} shots.")
        if kf_total:
            print(
                f"[CorteCenas] Features de keyframe: {kf_hits}/{kf_total} "
                "vindas do cache (YOLO+CLIP pulados)."
            )
        if cfg.skip_credit_shots:
            print(f"[CorteCenas] Shots ignorados por créditos/texto: {credit_shots}/{total}")

        # === Segunda passada: resgate por semelhança no próprio episódio ===
        # As cenas identificadas viram refs temporárias (mesmo traço/ângulo);
        # as sem dono são recomparadas contra elas. Pega o clássico "mesma
        # cena, mesmo ângulo, uma identificada e a vizinha pulada". Roda ANTES
        # da revisão IA: cada resgate aqui é um duvidoso a menos gastando API.
        rescued_pos: set[int] = set()
        if cfg.second_pass and entries:
            cb("second_pass", -1.0, "Comparando cenas sem dono com as já identificadas...")
            banks = build_episode_banks(
                entries,
                shot_faces,
                min_sources=cfg.second_pass_min_sources,
                max_bank=cfg.second_pass_max_bank,
            )
            rescues = rescue_unassigned(
                banks, shot_faces, threshold=cfg.second_pass_threshold
            )
            id_to_name = {e.id: e.name for e in entries}
            for pos, hits in sorted(rescues.items()):
                idx_b = cut_results[pos][0].idx
                if idx_b in blocked_pairs:
                    hits = [
                        (cid, s) for cid, s in hits
                        if cid not in blocked_pairs[idx_b]
                    ]
                    if not hits:
                        continue
                names_r: list[str] = []
                for cid, sim in hits:
                    self.db.assign_character(shot_db_ids[pos], cid, sim)
                    names_r.append(id_to_name[cid])
                per_shot_names[pos] = names_r
                rescued_pos.add(pos)
                print(
                    f"[2a passada] Shot #{cut_results[pos][0].idx:04d} -> "
                    f"{'+'.join(names_r)} (sim {hits[0][1]:.2f})"
                )
            print(
                f"[2a passada] Banco de {len(banks)} personagens; "
                f"{len(rescues)} cenas resgatadas."
            )
            cb(
                "second_pass",
                1.0,
                f"{len(rescues)} cenas resgatadas por semelhança"
                if rescues
                else "Nenhuma cena extra recuperada",
            )
            if rescued_pos and ambiguous:
                before_n = len(ambiguous)
                ambiguous = [a for a in ambiguous if a["pos"] not in rescued_pos]
                if before_n != len(ambiguous):
                    print(
                        f"[2a passada] {before_n - len(ambiguous)} duvidosos "
                        "resolvidos sem gastar IA."
                    )
        else:
            cb("second_pass", 1.0, "—")

        # === Resgate por GRUPO: a Descoberta embutida na análise verde ===
        # Os rostos que AINDA estão sem dono são agrupados entre si (mesma
        # pessoa dentro do episódio) e cada grupo é comparado inteiro contra
        # os protótipos — mediana de representantes diversos, com margem e
        # concordância. Pega o personagem que a primeira passada nunca
        # enxerga: aquele cujos crops pontuam todos "quase" contra as refs
        # externas e por isso nunca ganha semente de banco.
        leftover_result: DiscoveryResult | None = None
        group_review: list[dict] = []
        leftover_clusters: list[dict] = []
        pool_embs: list[np.ndarray] = []
        pool_meta: list[tuple[int, tuple | None]] = []  # (pos, (kf, box_idx))
        # Rostos DETECTADOS nas refs de cada personagem — a régua de quão
        # confiáveis são os protótipos dele pra decisões em bloco.
        faces_by_name = {n: fc for n, _rc, fc in ref_stats}
        if cfg.cluster_rescue and entries:
            for sf in shot_faces:
                if sf.embs is None or sf.embs.size == 0 or per_shot_names[sf.pos]:
                    continue
                refs_sf = sf.face_refs or [None] * len(sf.embs)
                for row, fr in zip(sf.embs, refs_sf):
                    pool_embs.append(np.asarray(row, dtype=np.float32))
                    pool_meta.append((sf.pos, fr))
            clusters = []
            if len(pool_embs) >= cfg.cluster_min_faces:
                observations = [
                    FaceObservation(
                        shot_pos=pool_meta[i][0],
                        shot_id=shot_db_ids[pool_meta[i][0]],
                        shot_idx=cut_results[pool_meta[i][0]][0].idx,
                        embedding=pool_embs[i],
                        crop_jpg=b"",
                    )
                    for i in range(len(pool_embs))
                ]
                cb("second_pass", -1.0,
                   f"Agrupando {len(observations)} rostos que sobraram sem dono...")
                clusters = cluster_faces(observations)
            n_group_shots = 0
            n_groups_named = 0
            for cl in clusters:
                embs_cl = np.stack([pool_embs[i] for i in cl.members])
                positions = sorted({pool_meta[i][0] for i in cl.members})
                if (
                    len(cl.members) < cfg.cluster_min_faces
                    or len(positions) < cfg.cluster_min_shots
                ):
                    continue  # pequeno demais pra decisão em bloco ou batismo
                rep_local = diverse_representatives(embs_cl, cfg.cluster_max_reps)
                ranking = rank_characters(embs_cl[rep_local], entries)
                ginfo = {
                    "cluster": cl,
                    "positions": positions,
                    "rep_refs": [pool_meta[cl.members[j]][1] for j in rep_local],
                    "ranking": ranking,
                }
                winner = decide(
                    ranking,
                    min_sim=cfg.cluster_min_sim,
                    margin=cfg.cluster_margin,
                    min_agreement=cfg.cluster_agreement,
                )
                if (
                    winner is not None
                    and faces_by_name.get(winner.name, 0) < cfg.cluster_min_ref_faces
                ):
                    # Refs fracas (retratos sem rosto detectável) não
                    # sustentam nomear dezenas de cenas sozinho — caso real:
                    # grupo de 17 rostos batizado em cima de 2 retratinhos.
                    # Vira sugestão no batismo; o usuário confirma.
                    print(
                        f"[grupo] {len(cl.members)} rostos parecem "
                        f"{winner.name} (mediana {ranking[0].median:.2f}), mas "
                        f"ele só tem {faces_by_name.get(winner.name, 0)} "
                        "rosto(s) de referência — vai pro batismo como "
                        "sugestão, não como decisão."
                    )
                    winner = None
                if winner is not None:
                    n = self._assign_cluster(
                        winner, ranking[0].median, positions, blocked_pairs,
                        cut_results, shot_db_ids, per_shot_names,
                    )
                    n_group_shots += n
                    n_groups_named += 1
                    print(
                        f"[grupo] {len(cl.members)} rostos/{len(positions)} cenas "
                        f"-> {winner.name} (mediana {ranking[0].median:.2f}, "
                        f"{int(ranking[0].agreement * 100)}% dos reps)"
                    )
                elif (
                    ai_review_ambiguous
                    and ranking
                    and ranking[0].median >= cfg.cluster_review_low
                ):
                    group_review.append(ginfo)
                else:
                    leftover_clusters.append(ginfo)
            if clusters:
                print(
                    f"[grupo] {n_groups_named} grupos nomeados em bloco "
                    f"({n_group_shots} cenas); {len(group_review)} pra IA; "
                    f"{len(leftover_clusters)} pro batismo."
                )
                cb(
                    "second_pass", 1.0,
                    f"Grupos: {n_group_shots} cenas nomeadas em bloco"
                    if n_group_shots else "Grupos sem match direto",
                )

        # === Revisão IA por GRUPO (contact sheet) ===
        # Uma pergunta por grupo — o modelo vê várias expressões do mesmo
        # rosto e a resposta vale por todas as cenas do grupo. Muito mais
        # barato e estável que perguntar cena a cena.
        if ai_review_ambiguous and group_review:
            cb("ai_review", -1.0, f"{len(group_review)} grupos duvidosos → IA")
            try:
                gclient = self._build_ai_client()
            except RuntimeError as e:
                print(f"[AI grupo] Pulado: {e}", flush=True)
                gclient = None
                leftover_clusters.extend(group_review)
            if gclient is not None:
                name_to_entry = {e.name: e for e in entries}
                resolved_g = 0
                g_pt = g_ct = 0
                try:
                    for gi, ginfo in enumerate(group_review, 1):
                        cb("ai_review", -1.0, f"IA no grupo {gi}/{len(group_review)}")
                        crops = self._materialize_crops(
                            kf_cache, ginfo["rep_refs"], cfg.face_crop_padding_ai
                        )[:6]
                        cands = []
                        for gs in ginfo["ranking"][:3]:
                            # Candidato com refs fracas não entra no páreo da
                            # IA — ela só escolheria entre retratos ruins.
                            if (
                                faces_by_name.get(gs.entry.name, 0)
                                < cfg.cluster_min_ref_faces
                            ):
                                continue
                            jpgs = self._candidate_ref_jpgs(
                                refs_per_char.get(gs.entry.name, [])
                            )
                            if jpgs:
                                cands.append((gs.entry.name, jpgs))
                        if not crops or not cands:
                            leftover_clusters.append(ginfo)
                            continue
                        try:
                            vname, vconf, usage = classify_group(
                                gclient, crops, cands, bundle.title
                            )
                        except QuotaExhaustedError as e:
                            print(f"[AI grupo] Quota esgotou: {e}", flush=True)
                            leftover_clusters.append(ginfo)
                            leftover_clusters.extend(group_review[gi:])
                            break
                        except Exception as e:
                            print(f"[AI grupo] ERRO no grupo {gi}: {e}", flush=True)
                            leftover_clusters.append(ginfo)
                            continue
                        g_pt += int(usage.get("prompt_tokens") or 0)
                        g_ct += int(usage.get("completion_tokens") or 0)
                        if vname in name_to_entry and vconf >= cfg.default_threshold:
                            entry = name_to_entry[vname]
                            n = self._assign_cluster(
                                entry, vconf, ginfo["positions"], blocked_pairs,
                                cut_results, shot_db_ids, per_shot_names,
                            )
                            resolved_g += 1
                            print(
                                f"[AI grupo] {len(ginfo['positions'])} cenas -> "
                                f"{vname} (conf {vconf:.2f}, {n} atribuídas)"
                            )
                        else:
                            leftover_clusters.append(ginfo)
                finally:
                    gclient.close()
                print(
                    f"[AI grupo] {resolved_g}/{len(group_review)} grupos resolvidos "
                    f"| tokens: {g_pt:,}+{g_ct:,}",
                    flush=True,
                )
        elif group_review:
            leftover_clusters.extend(group_review)

        # Cena que ganhou dono via grupo sai da fila de revisão por cena.
        if ambiguous:
            before_g = len(ambiguous)
            ambiguous = [a for a in ambiguous if not per_shot_names[a["pos"]]]
            if before_g != len(ambiguous):
                print(
                    f"[grupo] {before_g - len(ambiguous)} duvidosos resolvidos "
                    "em bloco sem gastar IA por cena."
                )

        # Grupos que sobraram → ponte pro batismo no fim da análise.
        if leftover_clusters:
            groups_out: list[DiscoveredGroup] = []
            for gk, ginfo in enumerate(leftover_clusters):
                thumbs = self._materialize_crops(
                    kf_cache, ginfo["rep_refs"], cfg.face_crop_padding
                )
                if not thumbs:
                    continue
                cl = ginfo["cluster"]
                positions = ginfo["positions"]
                ranking = ginfo["ranking"]
                sug, sug_sim = "", 0.0
                if ranking and ranking[0].median >= 0.60:
                    sug, sug_sim = ranking[0].entry.name, ranking[0].median
                conf_map: dict[int, float] = {}
                for i in cl.members:
                    pos_i = pool_meta[i][0]
                    s = float(pool_embs[i] @ cl.centroid)
                    conf_map[pos_i] = max(conf_map.get(pos_i, 0.0), s)
                groups_out.append(DiscoveredGroup(
                    key=gk,
                    n_faces=len(cl.members),
                    n_shots=len(positions),
                    thumbs_jpg=thumbs[:6],
                    ref_crops_jpg=thumbs[:8],
                    shot_ids=[shot_db_ids[p] for p in positions],
                    shot_positions=positions,
                    shot_conf=conf_map,
                    centroid_bytes=to_bytes(cl.centroid.astype(np.float32)),
                    suggested_name=sug,
                    suggested_sim=sug_sim,
                ))
            if groups_out:
                leftover_result = DiscoveryResult(
                    anime_title=bundle.title,
                    season=info.season,
                    episode=info.episode,
                    anime_id=anime_id,
                    episode_id=episode_id,
                    episode_root=episode_root,
                    cache_id=cache_id,
                    shots=[
                        DiscoveryShot(
                            pos=p, shot_id=sid, idx=shot_b.idx,
                            file=str(sf_file), keyframes=[str(k) for k in kfs_b],
                            start=shot_b.start, end=shot_b.end,
                        )
                        for p, ((shot_b, sf_file, kfs_b), sid)
                        in enumerate(zip(cut_results, shot_db_ids))
                    ],
                    groups=groups_out,
                    total_faces=len(pool_embs),
                    online=True,
                    roster=[c.name for c in bundle.characters],
                )
                print(
                    f"[grupo] {len(groups_out)} grupos sem nome — batismo "
                    "será oferecido no fim.",
                    flush=True,
                )

        # === Revisão IA dos duvidosos (modo híbrido) ===
        # O CLIP resolveu o grosso de graça; só a zona cinzenta gasta API.
        if ai_review_ambiguous and ambiguous:
            name_to_id_review = {e.name: e.id for e in entries}
            roster = [e.name for e in entries]
            # Mais promissores primeiro; teto de custo por episódio.
            ambiguous.sort(key=lambda a: -a["best_sim"])
            dropped = len(ambiguous) - cfg.ai_review_max_shots
            if dropped > 0:
                print(f"[AI review] {dropped} duvidosos além do teto de "
                      f"{cfg.ai_review_max_shots} ficaram de fora.")
                ambiguous = ambiguous[: cfg.ai_review_max_shots]
            cb("ai_review", -1.0, f"{len(ambiguous)} shots duvidosos → IA")
            try:
                client = self._build_ai_client()
            except RuntimeError as e:
                print(f"[AI review] Pulado: {e}", flush=True)
                client = None
            if client is not None:
                from .ai_review import QuotaExhaustedError as _Quota
                top_refs = self._build_top_refs(bundle, refs_per_char)
                confirmed = 0
                errors = 0
                total_pt = total_ct = 0
                try:
                    for j, item in enumerate(ambiguous, 1):
                        cb("ai_review", j / len(ambiguous),
                           f"IA revisando {j}/{len(ambiguous)}")
                        try:
                            verdicts, usage = classify_face_crops(
                                client, item["crops"], roster,
                                bundle.title, top_refs=top_refs,
                            )
                        except _Quota as e:
                            print(f"[AI review] Quota esgotou no {j}º duvidoso — "
                                  f"mantendo o resultado do CLIP. {e}", flush=True)
                            break
                        except Exception as e:
                            errors += 1
                            print(f"[AI review] Shot #{item['shot'].idx:04d} ERRO: {e}",
                                  flush=True)
                            if errors >= 5:
                                print("[AI review] Muitos erros seguidos — parando a "
                                      "revisão; resultado do CLIP mantido.", flush=True)
                                break
                            continue
                        errors = 0
                        total_pt += int(usage.get("prompt_tokens") or 0)
                        total_ct += int(usage.get("completion_tokens") or 0)
                        names_in_shot: list[str] = []
                        for (vname, vconf) in verdicts:
                            if vname == "none" or vconf < cfg.default_threshold:
                                continue
                            if vname not in name_to_id_review or vname in names_in_shot:
                                continue
                            vcid = name_to_id_review[vname]
                            if vcid in blocked_pairs.get(item["shot"].idx, set()):
                                continue  # usuário já disse que não é
                            self.db.assign_character(item["shot_id"], vcid, vconf)
                            names_in_shot.append(vname)
                        if names_in_shot:
                            confirmed += 1
                            per_shot_names[item["pos"]] = names_in_shot
                            print(f"[AI review] Shot #{item['shot'].idx:04d} "
                                  f"(sim {item['best_sim']:.2f}) -> "
                                  f"{'+'.join(names_in_shot)}", flush=True)
                finally:
                    client.close()
                print(f"[AI review] === FIM === {confirmed}/{len(ambiguous)} duvidosos "
                      f"confirmados | tokens: {total_pt:,}+{total_ct:,}", flush=True)
                cb("ai_review", 1.0,
                   f"IA confirmou {confirmed} de {len(ambiguous)} duvidosos")
        elif ai_review_ambiguous:
            cb("ai_review", 1.0, "Nenhum shot duvidoso — CLIP resolveu tudo")
        else:
            cb("ai_review", 1.0, "—")

        result = self._finalize_episode(
            cb=cb,
            info=info,
            episode_id=episode_id,
            episode_root=episode_root,
            metadata_dir=metadata_dir,
            bundle=bundle,
            refs_per_char=refs_per_char,
            cut_results=cut_results,
            shot_db_ids=shot_db_ids,
            per_shot_names=per_shot_names,
            name_to_id={e.name: e.id for e in entries},
            characters_json=[
                {
                    "name": e.name,
                    "character_id": e.id,
                    "threshold": e.threshold,
                    "reference_count": len(refs_per_char.get(e.name, [])),
                }
                for e in entries
            ],
            refs_dir=refs_dir_str,
            low_refs_warning=low_refs_warning,
            merge_snapshot=merge_snapshot,
            benchmark=benchmark,
        )
        result.leftover_groups = leftover_result
        # Sem timings no modo medição: a rodada do benchmark pula o corte e
        # sai de cache: gravar esses tempos por cima estragaria o histórico
        # que responde "onde o tempo mora" (ver metadata/timings.json).
        if not benchmark:
            self._report_timings(timer, metadata_dir)
        return result

    def _finalize_episode(
        self,
        *,
        cb: ProgressCb,
        info: EpisodeInfo,
        episode_id: int,
        episode_root: Path,
        metadata_dir: Path,
        bundle,
        refs_per_char: dict,
        cut_results: list,
        shot_db_ids: list[int],
        per_shot_names: list[list[str]],
        name_to_id: dict[str, int],
        characters_json: list[dict],
        refs_dir: str | None = None,
        low_refs_warning: str | None = None,
        merge_snapshot: list[dict] | None = None,
        benchmark: bool = False,
    ) -> PipelineResult:
        """Shared end-of-pipeline stage for both CLIP and AI paths:
          - drop characters below min_shots_per_character (cleans DB too)
          - reapply the user's remembered manual curation (add/block)
          - write shots.json + characters.json
          - create by_character / by_pair hardlinks (from DB truth)
          - return PipelineResult
        """
        cfg = self.cfg

        from collections import Counter as _Counter
        char_counts = _Counter(n for names in per_shot_names for n in names)
        dropped: list[str] = []
        for name, count in char_counts.items():
            if count >= cfg.min_shots_per_character:
                continue
            cid = name_to_id.get(name)
            if cid is None:
                continue
            self.db.drop_low_count_character(episode_id, cid)
            dropped.append(f"{name}({count})")
        if dropped:
            print(f"[CorteCenas] Removidos (poucos shots): {', '.join(dropped)}")

        idx_to_dbid = {
            shot.idx: sid for (shot, _f, _k), sid in zip(cut_results, shot_db_ids)
        }

        # Modo "adicionar" da reanálise: devolve as atribuições da análise
        # anterior por cima do resultado novo (INSERT OR IGNORE — a análise
        # nova ganha quando o par já existe). Vem antes da curadoria manual,
        # que ainda ganha de tudo (bloqueio remove inclusive o que voltou).
        if merge_snapshot and not benchmark:
            merged_back = 0
            for snap in merge_snapshot:
                sid = idx_to_dbid.get(snap["shot_idx"])
                if sid is None:
                    continue
                self.db.merge_assignment(
                    sid, snap["character_id"], snap["confidence"],
                    reviewed=snap.get("reviewed") or 0,
                    approved=snap.get("approved"),
                )
                merged_back += 1
            print(
                f"[CorteCenas] Modo adicionar: {merged_back} atribuições "
                "antigas reaplicadas por cima do resultado novo."
            )
            cb("organize", -1.0, f"Somando análise anterior ({merged_back} atribuições)")

        # Curadoria manual lembrada de análises anteriores: bloqueios tiram o
        # que a IA re-adicionou, adições devolvem o que o usuário confirmou.
        # Vem DEPOIS do drop de poucos-shots pra decisão do usuário ganhar.
        overrides = [] if benchmark else self.db.manual_overrides(episode_id)
        if overrides:
            n_block = n_add = 0
            for ov in overrides:
                sid = idx_to_dbid.get(ov["shot_idx"])
                if sid is None:
                    continue  # cena não existe mais (bounds mudaram)
                if ov["action"] == "block":
                    self.db.remove_shot_character(sid, ov["character_id"])
                    n_block += 1
                else:
                    self.db.assign_character_manual(
                        sid, ov["character_id"], ov["confidence"]
                    )
                    n_add += 1
            print(
                f"[CorteCenas] Curadoria manual reaplicada: "
                f"{n_block} remoções, {n_add} adições/movidas."
            )
            cb("organize", -1.0, f"Curadoria manual: {n_block + n_add} decisões reaplicadas")

        cb("organize", -1.0,
           "Lendo o resultado (sem mexer nas pastas)" if benchmark
           else "Gerando pastas e metadados...")
        if not benchmark:
            clear_grouping(episode_root)

        # Pastas e contagens saem do BANCO (não das listas em memória): é o
        # banco que carrega o resultado final — automático + IA + curadoria.
        by_shot = self.db.assignments_for_episode(episode_id)
        final_names: list[list[str]] = []
        shots_payload = []
        for (shot, shot_file, kfs), shot_id in zip(cut_results, shot_db_ids):
            assigns = by_shot.get(shot_id, [])
            names = [a["name"] for a in assigns]
            final_names.append(names)
            shot_row = {
                "idx": shot.idx,
                "file": str(shot_file.relative_to(episode_root)).replace("\\", "/"),
                "keyframe": str((kfs[len(kfs) // 2]).relative_to(episode_root)).replace("\\", "/") if kfs else None,
                "start": shot.start,
                "end": shot.end,
            }
            shots_payload.append(
                build_shot_payload(shot_row, bundle.title, info.season, info.episode, assigns)
            )
            if names and not benchmark:
                organize_by_character(shot_file, episode_root, names)
                organize_by_pair(shot_file, episode_root, names)

        if not benchmark:
            write_shots_json(metadata_dir / "shots.json", shots_payload)
            write_characters_json(metadata_dir / "characters.json", characters_json)

        pair_counts = dict(count_pairs(final_names))
        identified = sorted({n for names in final_names for n in names})
        cb("organize", 1.0, "Concluído")
        return PipelineResult(
            episode_root=episode_root,
            total_shots=len(cut_results),
            total_characters=len(identified),
            identified_characters=identified,
            pair_counts=pair_counts,
            anime_title=bundle.title,
            season=info.season,
            episode=info.episode,
            episode_id=episode_id,
            kind=info.kind,
            low_refs_warning=low_refs_warning,
            refs_dir=refs_dir,
        )

    # ================= Modo Descoberta =================

    def run_discovery(
        self, info: EpisodeInfo, on_progress: ProgressCb | None = None
    ) -> DiscoveryResult:
        """Análise SEM banco online: corta o episódio, detecta e embedda os
        rostos, e agrupa por identidade. Retorna os grupos anônimos pra tela
        de batismo — commit_discovery() fecha o trabalho com os nomes."""
        timer = StageTimer()
        cb = timer.wrap(on_progress or _noop)
        cfg = self.cfg
        cb("parse", 1.0, f"{info.anime} {info.slug}")
        # Na Descoberta a identidade online é opcional, então aqui só vale a
        # memória que NÃO depende de rede: a pasta já combinada pra este nome.
        pastas = AnimeFolderStore(cfg.cache_path)
        pasta_anime = (
            (sanitize(info.output_folder) if info.output_folder else "")
            or pastas.folder_for_name(info.anime)
            or sanitize(info.anime)
        )
        episode_root, metadata_dir, cut_results = self._prepare_shots(
            info, cb, pasta_anime
        )
        pastas.remember(info.anime, pasta_anime)

        # Identidade online é opcional na descoberta: se o anime resolver,
        # os grupos reforçam o banco REAL (refs em al<root>) e os nomes são
        # pré-sugeridos pelos centroides já conhecidos; se não resolver,
        # segue 100% offline com banco local.
        cb("fetch_characters", -1.0, "Buscando anime (descoberta)...")
        online_bundle = None
        provider = AnimeProvider(cfg.cache_path)
        try:
            online_bundle = provider.resolve(
                info.anime,
                max_characters=cfg.max_characters_per_anime,
                images_per_character=cfg.references_per_character,
                on_status=lambda m: cb("fetch_characters", -1.0, m),
                use_danbooru=False,
                gallery_for_top=cfg.gallery_for_top_characters,
                season=info.season,
            )
        except Exception as e:
            print(f"[Descoberta] Sem identidade online ({type(e).__name__}) — "
                  "seguindo com banco local.", flush=True)
        finally:
            provider.close()

        known_centroids: list[tuple[str, np.ndarray]] = []
        if online_bundle is not None:
            anime_title = online_bundle.title
            anime_id = self.db.upsert_anime(
                anilist_id=online_bundle.anilist_id,
                mal_id=online_bundle.mal_id,
                title=online_bundle.title,
                title_english=online_bundle.title_english,
            )
            if online_bundle.cache_id_override:
                disc_cache_id = online_bundle.cache_id_override
            elif online_bundle.franchise_root_id:
                disc_cache_id = f"al{online_bundle.franchise_root_id}"
            elif online_bundle.anilist_id:
                disc_cache_id = f"al{online_bundle.anilist_id}"
            else:
                disc_cache_id = f"mal{online_bundle.mal_id}"
            for row in self.db.get_characters_for_anime(anime_id):
                if row.get("embedding"):
                    known_centroids.append((row["name"], from_bytes(row["embedding"])))
            cb("fetch_characters", 1.0,
               f"{anime_title} — descoberta reforçando o banco existente")
        else:
            anime_title = info.anime
            anime_id = self.db.upsert_anime(
                anilist_id=None, mal_id=None, title=info.anime, title_english=None
            )
            disc_cache_id = local_cache_id(info.anime)
            cb("fetch_characters", 1.0, "Modo Descoberta — sem banco online")

        # Refs pra SUGESTÃO: quando o anime é conhecido mas nunca foi
        # analisado (sem centroides no DB), até 1 foto por personagem —
        # pouco demais pra análise — já serve pra pré-nomear grupos.
        weak_refs: dict[str, list[Path]] = {}
        if online_bundle is not None and not known_centroids:
            cb("download_refs", -1.0, "Baixando fotos pra sugerir nomes...")
            try:
                weak_refs = ReferenceStore(cfg.cache_path).ensure_references(
                    disc_cache_id, online_bundle,
                    on_status=lambda m: cb("download_refs", -1.0, m),
                )
            except Exception as e:
                print(f"[Descoberta] Refs pra sugestão falharam: {e}", flush=True)
        cb("download_refs", 1.0, "—")

        get_engine, get_face_det = self._lazy_models(cb)
        kf_cache = FeatureCache(metadata_dir / "face_cache.npz", self._feature_meta())

        # Centroides provisórios das refs escassas (mesmo recorte de rosto da
        # análise normal). Só entram nomes que ainda não têm centroide do DB.
        if weak_refs:
            have = {n for n, _ in known_centroids}
            for name, paths in weak_refs.items():
                if name in have or not paths:
                    continue
                imgs = []
                for p in paths[:4]:
                    img = cv2.imread(str(p))
                    if img is None:
                        continue
                    crops = get_face_det().crop_faces(img, pad=cfg.face_crop_padding)
                    imgs.extend(crops if crops else [smart_portrait_crop(img)])
                if not imgs:
                    continue
                centroid = build_centroid(get_engine().embed_images(imgs))
                if centroid is not None:
                    known_centroids.append((name, centroid))
            print(f"[Descoberta] {len(known_centroids)} centroides provisórios "
                  "pra sugestão de nomes.", flush=True)
        cb("embed_refs", 1.0, "Modelos prontos")

        episode_id = self.db.upsert_episode(
            anime_id, info.season, info.episode, str(info.source), info.kind
        )
        # Gravar a pasta AQUI, e não só no commit: sem isto uma descoberta
        # abandonada (ou que ainda não foi batizada) deixava output_root NULL,
        # e episódio sem pasta some do histórico e não abre — a aba de
        # resultados caía na análise anterior como se a nova nunca tivesse
        # existido.
        self.db.set_episode_root(episode_id, str(episode_root))
        self.db.clear_episode_shots(episode_id)

        observations: list[FaceObservation] = []
        shots_out: list[DiscoveryShot] = []
        total = len(cut_results)
        kf_hits = kf_total = 0
        for i, (shot, shot_file, kfs) in enumerate(cut_results, 1):
            cb("analyze_shots", i / max(total, 1), f"Shot {i}/{total}")
            main_kf = kfs[len(kfs) // 2] if kfs else None
            shot_id = self.db.insert_shot(
                episode_id=episode_id,
                idx=shot.idx,
                file=str(shot_file.relative_to(episode_root)),
                keyframe=str(main_kf.relative_to(episode_root)) if main_kf else None,
                start=shot.start,
                end=shot.end,
            )
            shots_out.append(DiscoveryShot(
                pos=i - 1, shot_id=shot_id, idx=shot.idx,
                file=str(shot_file), keyframes=[str(k) for k in kfs],
                start=shot.start, end=shot.end,
            ))

            imgs_loaded: dict[Path, np.ndarray | None] = {}

            def _img(p: Path) -> np.ndarray | None:
                if p not in imgs_loaded:
                    imgs_loaded[p] = cv2.imread(str(p))
                return imgs_loaded[p]

            # Mesmo esquema da análise normal: cache de boxes+embeddings por
            # keyframe, YOLO+CLIP em lote só nos que faltam. A descoberta
            # ainda relê a imagem quando há rosto (precisa do JPEG do crop
            # pra tela de batismo), mas a parte cara de GPU é pulada.
            feats: dict[Path, tuple[np.ndarray, np.ndarray | None]] = {}
            missing: list[Path] = []
            for kf_path in kfs:
                kf_total += 1
                boxes = kf_cache.get(kf_path, "boxes")
                embs_c = kf_cache.get(kf_path, "embs")
                if boxes is not None and (len(boxes) == 0 or embs_c is not None):
                    kf_hits += 1
                    feats[kf_path] = (boxes, embs_c)
                else:
                    missing.append(kf_path)
            if missing:
                good = [(p, _img(p)) for p in missing]
                good = [(p, im) for p, im in good if im is not None]
                if good:
                    batch = get_face_det().crop_faces_batch(
                        [im for _, im in good], pad=cfg.face_crop_padding
                    )
                    flat: list[np.ndarray] = []
                    spans: list[tuple[Path, list, int]] = []
                    for (p, _im), (crops, kept) in zip(good, batch):
                        spans.append((p, kept, len(crops)))
                        flat.extend(crops)
                    embs_all = (
                        get_engine().embed_images(flat)
                        if flat
                        else np.zeros((0, 1), dtype=np.float32)
                    )
                    if flat and len(embs_all) != len(flat):
                        for (p, _im), (crops, kept) in zip(good, batch):
                            e = (
                                get_engine().embed_images(crops)
                                if crops
                                else np.zeros((0, 1), dtype=np.float32)
                            )
                            if len(e) != len(kept):
                                continue
                            self._store_kf(kf_cache, feats, p, kept, e)
                    else:
                        off = 0
                        for p, kept, n_crops in spans:
                            e = embs_all[off:off + n_crops]
                            off += n_crops
                            self._store_kf(kf_cache, feats, p, kept, e)

            for kf_path in kfs:
                boxes, embs = feats.get(kf_path, (None, None))
                if boxes is None or len(boxes) == 0 or embs is None or embs.size == 0:
                    continue
                img = _img(kf_path)
                if img is None:
                    continue
                crops, _kept = crops_from_boxes(img, boxes, cfg.face_crop_padding)
                if len(crops) != len(embs):
                    continue  # keyframe mudou entre o cache e a releitura
                for crop, emb in zip(crops, embs):
                    ch_, cw_ = crop.shape[:2]
                    scale = 256 / max(ch_, cw_)
                    if scale < 1.0:
                        crop = cv2.resize(crop, (int(cw_ * scale), int(ch_ * scale)))
                    ok_, enc_ = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if ok_:
                        observations.append(FaceObservation(
                            shot_pos=i - 1, shot_id=shot_id, shot_idx=shot.idx,
                            embedding=np.asarray(emb, dtype=np.float32),
                            crop_jpg=enc_.tobytes(),
                        ))
        kf_cache.save()
        if kf_total:
            print(
                f"[Descoberta] Features de keyframe: {kf_hits}/{kf_total} do cache.",
                flush=True,
            )

        cb("second_pass", 1.0, "—")
        cb("ai_review", 1.0, "—")
        cb("organize", -1.0, f"Agrupando {len(observations)} rostos por personagem...")
        clusters = cluster_faces(observations)
        # Sugestão de nome: centroide do grupo vs personagens já conhecidos
        # (existem quando o anime já foi analisado antes). 0.75 é conservador
        # — melhor campo vazio que sugestão errada pré-preenchida.
        _SUGGEST_MIN = 0.75
        groups: list[DiscoveredGroup] = []
        for key, cl in enumerate(clusters):
            reps = pick_representatives(cl, observations, k=10)
            positions = sorted({observations[i].shot_pos for i in cl.members})
            conf: dict[int, float] = {}
            for i in cl.members:
                p = observations[i].shot_pos
                s = float(observations[i].embedding @ cl.centroid)
                conf[p] = max(conf.get(p, 0.0), s)
            suggested, s_sim = "", 0.0
            for kname, kcent in known_centroids:
                s = float(cl.centroid @ kcent)
                if s > s_sim:
                    s_sim, suggested = s, kname
            if s_sim < _SUGGEST_MIN:
                suggested, s_sim = "", 0.0
            groups.append(DiscoveredGroup(
                key=key,
                n_faces=len(cl.members),
                n_shots=len(positions),
                thumbs_jpg=[observations[i].crop_jpg for i in reps[:6]],
                ref_crops_jpg=[observations[i].crop_jpg for i in reps[:8]],
                shot_ids=sorted({observations[i].shot_id for i in cl.members}),
                shot_positions=positions,
                shot_conf=conf,
                centroid_bytes=to_bytes(cl.centroid),
                suggested_name=suggested,
                suggested_sim=s_sim,
            ))
        cb("organize", 1.0, f"{len(groups)} personagens descobertos")
        print(f"[Descoberta] {len(observations)} rostos → {len(groups)} grupos "
              f"({', '.join(str(g.n_faces) for g in groups[:10])}...)", flush=True)

        self._report_timings(timer, metadata_dir)
        return DiscoveryResult(
            anime_title=anime_title,
            season=info.season,
            episode=info.episode,
            anime_id=anime_id,
            episode_id=episode_id,
            episode_root=episode_root,
            cache_id=disc_cache_id,
            shots=shots_out,
            groups=groups,
            total_faces=len(observations),
            online=online_bundle is not None,
            roster=(
                [ch.name for ch in online_bundle.characters]
                if online_bundle is not None else []
            ),
        )

    def commit_discovery(
        self,
        result: DiscoveryResult,
        names: dict[int, str],
        on_progress: ProgressCb | None = None,
        removed: dict[int, list[int]] | None = None,
    ) -> PipelineResult:
        """Fecha o Modo Descoberta com os nomes dados pelo usuário: cria os
        personagens, atribui os shots, salva os crops como referências (os
        próximos episódios rodam no modo normal) e organiza as pastas.
        Grupos sem nome são ignorados; dois grupos com o MESMO nome fundem.
        `removed`: índices de ref_crops_jpg que o usuário clicou pra tirar
        (rosto alheio infiltrado no grupo) — não viram referência."""
        cb = on_progress or _noop
        cfg = self.cfg
        removed = removed or {}

        by_name: dict[str, list[DiscoveredGroup]] = {}
        for g in result.groups:
            name = (names.get(g.key) or "").strip()
            if name:
                by_name.setdefault(name, []).append(g)
        if not by_name:
            raise RuntimeError(
                "Nenhum grupo recebeu nome — não há o que salvar. "
                "Dê nome a pelo menos um personagem e confirme de novo."
            )

        cb("organize", -1.0, "Salvando personagens descobertos...")
        store = ReferenceStore(cfg.cache_path)
        name_to_id: dict[str, int] = {}
        refs_per_char: dict[str, list[Path]] = {}

        for name, gs in by_name.items():
            cid = self.db.upsert_character(
                anime_id=result.anime_id, name=name,
                anilist_id=None, mal_id=None, role="Main",
            )
            name_to_id[name] = cid

            d = store.character_dir(result.cache_id, name)
            d.mkdir(parents=True, exist_ok=True)
            paths: list[Path] = []
            for g in gs:
                skip = set(removed.get(g.key) or [])
                for gi, jpg in enumerate(g.ref_crops_jpg):
                    if gi in skip:
                        continue
                    if len(paths) >= 10:
                        break
                    p = d / f"auto_disc_{len(paths):02d}.jpg"
                    p.write_bytes(jpg)
                    paths.append(p)
            refs_per_char[name] = paths

            cents = np.stack([from_bytes(g.centroid_bytes) for g in gs])
            weights = np.array([g.n_faces for g in gs], dtype=np.float32)
            centroid = (cents * weights[:, None]).sum(axis=0)
            centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
            self.db.set_character_embedding(
                cid, to_bytes(centroid.astype(np.float32)), reference_count=len(paths)
            )

            for g in gs:
                for pos in g.shot_positions:
                    self.db.assign_character(
                        result.shots[pos].shot_id, cid,
                        float(g.shot_conf.get(pos, 0.9)),
                    )

        per_shot_names: list[list[str]] = [[] for _ in result.shots]
        for name, gs in by_name.items():
            for g in gs:
                for pos in g.shot_positions:
                    if name not in per_shot_names[pos]:
                        per_shot_names[pos].append(name)

        bundle = AnimeBundle(
            anilist_id=None, mal_id=None,
            title=result.anime_title, title_english=None,
            characters=[
                CharacterRef(mal_id=None, anilist_id=None, name=n,
                             role="Main", image_urls=[])
                for n in by_name
            ],
            cache_id_override=result.cache_id,
        )
        if not result.online:
            # Banco local persistido: a próxima análise deste anime resolve
            # offline (anime_provider cai nele quando as buscas falham).
            # Quando o anime é conhecido, os refs já entraram na pasta real
            # (al<root>) e o metadata online se resolve sozinho.
            provider = AnimeProvider(cfg.cache_path)
            try:
                provider.save_cache(result.cache_id, bundle)
            finally:
                provider.close()

        cut_results = [
            (ShotBounds(idx=s.idx, start=s.start, end=s.end),
             Path(s.file), [Path(k) for k in s.keyframes])
            for s in result.shots
        ]
        info = EpisodeInfo(
            anime=result.anime_title, season=result.season,
            episode=result.episode,
            source=Path(result.shots[0].file) if result.shots else Path("."),
        )
        # Ponte verde→batismo: o commit roda DEPOIS de uma análise completa,
        # e o characters.json existente carrega o elenco inteiro — os nomes
        # batizados entram por cima sem apagar os demais.
        chars_json = [
            {
                "name": n,
                "character_id": name_to_id[n],
                "threshold": cfg.default_threshold,
                "reference_count": len(refs_per_char.get(n, [])),
            }
            for n in by_name
        ]
        meta_file = result.episode_root / "metadata" / "characters.json"
        try:
            if meta_file.exists():
                old = json.loads(meta_file.read_text(encoding="utf-8"))
                have = {c.get("name") for c in chars_json}
                chars_json.extend(
                    c for c in old
                    if isinstance(c, dict) and c.get("name") not in have
                )
        except Exception:
            pass

        return self._finalize_episode(
            cb=cb,
            info=info,
            episode_id=result.episode_id,
            episode_root=result.episode_root,
            metadata_dir=result.episode_root / "metadata",
            bundle=bundle,
            refs_per_char=refs_per_char,
            cut_results=cut_results,
            shot_db_ids=[s.shot_id for s in result.shots],
            per_shot_names=per_shot_names,
            name_to_id=name_to_id,
            characters_json=chars_json,
            refs_dir=str(store.anime_dir(result.cache_id) / "characters"),
        )

    def _prepare_shots(
        self, info: EpisodeInfo, cb: ProgressCb, pasta_anime: str = ""
    ):
        """Estágios comuns a TODAS as análises (normal e descoberta):
        layout de pastas, detecção de shots (com cache) e corte+keyframes.
        Retorna (episode_root, metadata_dir, cut_results).

        `pasta_anime` vazio = usa o nome digitado, como sempre foi. Quem
        chama com identidade em mãos (a análise normal) passa a pasta
        combinada pro anime — ver `storage/anime_folders.py`.
        """
        cfg = self.cfg

        episode_root = (
            cfg.output_path / (pasta_anime or sanitize(info.anime)) / info.slug
        )
        shots_dir = episode_root / "shots"
        keyframes_dir = episode_root / "keyframes"
        metadata_dir = episode_root / "metadata"
        for d in (shots_dir, keyframes_dir, metadata_dir):
            d.mkdir(parents=True, exist_ok=True)

        bounds_cache = metadata_dir / "shot_bounds.json"
        shots: list[ShotBounds] | None = None
        if bounds_cache.exists():
            try:
                data = json.loads(bounds_cache.read_text(encoding="utf-8"))
                if (
                    isinstance(data, dict)
                    and data.get("source") == str(info.source)
                    and abs(float(data.get("threshold", -1)) - cfg.scene_threshold) < 1e-6
                ):
                    shots = [
                        ShotBounds(idx=int(s["idx"]), start=float(s["start"]), end=float(s["end"]))
                        for s in data.get("shots", [])
                    ]
            except (json.JSONDecodeError, ValueError, KeyError, OSError) as e:
                print(f"[CorteCenas] Shot bounds cache inválido, recomputando: {e}")
                shots = None

        if shots:
            cb("detect_shots", 1.0, f"{len(shots)} shots (cache)")
        else:
            cb("detect_shots", -1.0, "Analisando mudanças de cena...")
            shots = detect_shots(
                info.source,
                threshold=cfg.scene_threshold,
                min_seconds=cfg.min_shot_seconds,
                on_progress=lambda f: cb(
                    "detect_shots", f, f"Analisando mudanças de cena... {int(f * 100)}%"
                ),
            )
            bounds_cache.write_text(
                json.dumps(
                    {
                        "source": str(info.source),
                        "threshold": cfg.scene_threshold,
                        "min_seconds": cfg.min_shot_seconds,
                        "shots": [
                            {"idx": s.idx, "start": s.start, "end": s.end} for s in shots
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            cb("detect_shots", 1.0, f"{len(shots)} shots detectados")

        if shots and (info.skip_head_seconds > 0 or info.skip_tail_seconds > 0):
            total_duration = max(s.end for s in shots)
            tail_cut = total_duration - info.skip_tail_seconds if info.skip_tail_seconds > 0 else total_duration + 1.0
            before = len(shots)
            shots = [
                s for s in shots
                if s.end > info.skip_head_seconds and s.start < tail_cut
            ]
            print(
                f"[CorteCenas] Skip manual: início {info.skip_head_seconds:.0f}s, "
                f"fim {info.skip_tail_seconds:.0f}s → {before - len(shots)} shots ignorados"
            )

        def cut_cb(done: int, total: int, skipped: int) -> None:
            suffix = f" ({skipped} já em cache)" if skipped else ""
            cb("cut_shots", done / max(total, 1), f"{done}/{total} shots{suffix}")

        cut_results = cut_all_shots(
            info.source,
            shots,
            shots_dir,
            keyframes_dir,
            keyframes_per_shot=cfg.keyframes_per_shot,
            reencode=cfg.reencode_shots,
            on_progress=cut_cb,
            render_mode=cfg.render_export_mode,
        )
        return episode_root, metadata_dir, cut_results

    def _lazy_models(self, cb: ProgressCb):
        """(get_engine, get_face_det) com carga adiada: os modelos só sobem
        quando alguma feature NÃO está no cache. Reanálise com cache cheio
        não paga o load do CLIP (~5s + VRAM) nem o do YOLO."""
        cfg = self.cfg
        holder: dict[str, object] = {"engine": None, "face_det": None}

        def get_engine() -> EmbeddingEngine:
            if holder["engine"] is None:
                clip_msg = "Carregando modelo CLIP..."
                if _clip_needs_download(cfg.clip_model, cfg.clip_pretrained):
                    clip_msg = (
                        "Baixando modelo CLIP (~890 MB) — só na primeira "
                        "execução, depois fica cacheado. Pode demorar 1-3 min."
                    )
                cb("embed_refs", -1.0, clip_msg)
                engine = EmbeddingEngine(
                    model_name=cfg.clip_model,
                    pretrained=cfg.clip_pretrained,
                    use_cuda=cfg.use_cuda,
                )
                if engine.on_device_fallback:
                    cb("embed_refs", -1.0, engine.on_device_fallback)
                    print(f"[CorteCenas] {engine.on_device_fallback}")
                holder["engine"] = engine
            return holder["engine"]

        def get_face_det() -> AnimeFaceDetector:
            if holder["face_det"] is None:
                # Face detector is also used for the references so the rep
                # space matches: face crop (ref) vs face crop (query), not
                # whole ref vs face crop. This removes background
                # contamination from the prototypes.
                holder["face_det"] = AnimeFaceDetector(ensure_cascade(cfg.models_path))
            return holder["face_det"]

        return get_engine, get_face_det

    def _feature_meta(self) -> dict:
        """Tudo que, mudando, invalida boxes/embeddings cacheados."""
        cfg = self.cfg
        return {
            "clip": f"{cfg.clip_model}/{cfg.clip_pretrained}",
            "detector": MODEL_SIGNATURE,
            "pad": cfg.face_crop_padding,
            "credit_thr": cfg.credit_edge_threshold,
        }

    def _ref_features(
        self, paths: list[Path], ref_cache: FeatureCache, get_engine, get_face_det
    ) -> tuple[list[np.ndarray], int]:
        """Embeddings das imagens de referência de UM personagem, com cache
        por arquivo e detecção/embedding em lote só nos que faltam.
        Retorna (blocos de embeddings, nº de rostos detectados)."""
        cfg = self.cfg
        emb_parts: list[np.ndarray] = []
        faces_found = 0
        misses: list[Path] = []
        for p in paths:
            boxes = ref_cache.get(p, "boxes")
            embs_c = ref_cache.get(p, "embs")
            if boxes is not None and embs_c is not None:
                faces_found += int(len(boxes))
                if embs_c.size:
                    emb_parts.append(embs_c)
            else:
                misses.append(p)
        if not misses:
            return emb_parts, faces_found

        imgs = [cv2.imread(str(p)) for p in misses]
        good = [(p, im) for p, im in zip(misses, imgs) if im is not None]
        if not good:
            return emb_parts, faces_found
        batch = get_face_det().crop_faces_batch(
            [im for _, im in good], pad=cfg.face_crop_padding
        )
        flat_crops: list[np.ndarray] = []
        spans: list[tuple[Path, list, int]] = []  # (arquivo, boxes mantidos, nº crops)
        for (p, im), (crops, kept) in zip(good, batch):
            if not crops:
                # Sem rosto na ref → retrato central (boxes ficam vazios; o
                # embedding do retrato ainda vale — silhueta/cabelo contam).
                crops, kept = [smart_portrait_crop(im)], []
            spans.append((p, kept, len(crops)))
            flat_crops.extend(crops)
        embs_all = get_engine().embed_images(flat_crops)
        if len(embs_all) != len(flat_crops):
            # Pareamento crop↔embedding quebrou (imagem indecodificável no
            # meio do lote) — usa o que veio, mas sem gravar cache torto.
            if embs_all.size:
                emb_parts.append(embs_all.astype(np.float32))
            return emb_parts, faces_found
        off = 0
        for p, kept, n_crops in spans:
            e = embs_all[off:off + n_crops].astype(np.float32)
            off += n_crops
            ref_cache.put(p, "boxes", np.array(kept, dtype=np.int32).reshape(-1, 4))
            ref_cache.put(p, "embs", e)
            faces_found += len(kept)
            if e.size:
                emb_parts.append(e)
        return emb_parts, faces_found

    def _assign_cluster(
        self,
        entry: CharacterEntry,
        conf: float,
        positions: list[int],
        blocked_pairs: dict[int, set[int]],
        cut_results: list,
        shot_db_ids: list[int],
        per_shot_names: list[list[str]],
    ) -> int:
        """Atribui um personagem a TODAS as cenas de um grupo resolvido,
        pulando os pares (cena, personagem) bloqueados pela curadoria."""
        n = 0
        for pos in positions:
            idx = cut_results[pos][0].idx
            if entry.id in blocked_pairs.get(idx, set()):
                continue
            self.db.assign_character(shot_db_ids[pos], entry.id, float(conf))
            if entry.name not in per_shot_names[pos]:
                per_shot_names[pos].append(entry.name)
            n += 1
        return n

    @staticmethod
    def _materialize_crops(
        kf_cache: FeatureCache,
        face_refs: list,
        pad: float,
        max_px: int = 256,
        quality: int = 86,
    ) -> list[bytes]:
        """Reconstrói os JPEGs dos crops a partir da proveniência
        (keyframe, índice do box) + boxes cacheados — só quem precisa de
        imagem paga a releitura do disco. Uma imagem por keyframe, memoizada."""
        out: list[bytes] = []
        crops_memo: dict[str, list] = {}
        for fr in face_refs:
            if not fr:
                continue
            kf_path, bi = fr
            key = str(kf_path)
            if key not in crops_memo:
                boxes = kf_cache.get(Path(kf_path), "boxes")
                crops: list = []
                if boxes is not None and len(boxes):
                    img = cv2.imread(key)
                    if img is not None:
                        crops = crops_from_boxes(img, boxes, pad)[0]
                crops_memo[key] = crops
            crops = crops_memo[key]
            if bi >= len(crops):
                continue
            c = crops[bi]
            h, w = c.shape[:2]
            scale = max_px / max(h, w)
            if scale < 1.0:
                c = cv2.resize(c, (int(w * scale), int(h * scale)))
            ok, enc = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                out.append(enc.tobytes())
        return out

    @staticmethod
    def _candidate_ref_jpgs(paths: list[Path], max_imgs: int = 2) -> list[bytes]:
        """Até `max_imgs` refs de um candidato, redimensionadas pro prompt."""
        out: list[bytes] = []
        for p in paths[:6]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = 256 / max(h, w)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                out.append(enc.tobytes())
            if len(out) >= max_imgs:
                break
        return out

    @staticmethod
    def _store_kf(
        cache: FeatureCache,
        feats: dict,
        p: Path,
        kept: list,
        embs: np.ndarray,
    ) -> None:
        """Grava boxes+embeddings de um keyframe no cache e no dict do shot.
        Invariante: len(boxes) == len(embs) — crop e embedding pareados."""
        boxes_arr = np.array(kept, dtype=np.int32).reshape(-1, 4)
        cache.put(p, "boxes", boxes_arr)
        e = embs.astype(np.float32) if embs.size else None
        if e is not None:
            cache.put(p, "embs", e)
        feats[p] = (boxes_arr, e)

    @staticmethod
    def _report_timings(timer: StageTimer, metadata_dir: Path) -> None:
        """Relatório final por etapa: app.log (print) + timings.json do
        episódio — a régua que diz onde a próxima otimização deve morar."""
        print(f"[Tempos] {timer.report()}", flush=True)
        try:
            (metadata_dir / "timings.json").write_text(
                json.dumps(timer.to_json(), indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    @staticmethod
    def _local_only_characters(ref_store: ReferenceStore, cache_id: str, bundle) -> list[str]:
        """Nomes de pastas em <anime>/characters/ com imagens dentro que não
        correspondem a nenhum personagem do bundle online — nem por slug,
        nem por tokens ("Rimuru" batizado casa com "Tempest, Rimuru" do
        elenco e NÃO vira personagem duplicado)."""
        from .naming import find_token_match
        from .references.reference_store import slug_for
        chars_dir = ref_store.anime_dir(cache_id) / "characters"
        if not chars_dir.exists():
            return []
        known = {slug_for(ch.name) for ch in bundle.characters}
        bundle_names = [ch.name for ch in bundle.characters]
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        out: list[str] = []
        for d in sorted(chars_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            if d.name in known:
                continue
            if find_token_match(d.name, bundle_names) is not None:
                continue  # mesmo personagem com nome noutro formato
            if any(f.suffix.lower() in exts for f in d.iterdir() if f.is_file()):
                out.append(d.name)
        return out

    def _build_ai_client(self) -> NavyAIClient:
        """NavyAI primário + Gemini nativo como fallback, conforme as keys
        configuradas. Levanta RuntimeError se nenhuma key existir."""
        cfg = self.cfg
        primary_key = cfg.navyai_api_key.strip()
        gemini_key = cfg.gemini_api_key.strip()
        if not primary_key and not gemini_key:
            raise RuntimeError(
                "Modo IA requer uma API key (NavyAI ou Gemini) em Configurações."
            )
        from .ai_review import GEMINI_OPENAI_BASE
        fallback = None
        if gemini_key:
            fallback = NavyAIClient(
                api_key=gemini_key,
                base_url=GEMINI_OPENAI_BASE,
                model=cfg.gemini_model or "gemini-2.5-flash",
            )
        if primary_key:
            return NavyAIClient(
                api_key=primary_key,
                base_url=cfg.navyai_base_url,
                model=cfg.navyai_model,
                fallback=fallback,
            )
        return fallback  # Gemini-only path

    @staticmethod
    def _build_top_refs(bundle, refs_per_char: dict) -> dict[str, list[bytes]]:
        """One reference image per character for the top 15 by popularity
        (role weight, then ref count) — the visual roster sent to the AI."""
        _ROLE_RANK = {"Main": 0, "MAIN": 0, "Supporting": 1, "SUPPORTING": 1}
        scored = []
        for ch in bundle.characters:
            paths = refs_per_char.get(ch.name)
            if not paths:
                continue
            role_w = _ROLE_RANK.get(ch.role, 2)
            scored.append((role_w, -len(paths), ch.name))  # negative for desc
        scored.sort()
        top_refs: dict[str, list[bytes]] = {}
        for _, _, name in scored[:15]:
            paths = refs_per_char.get(name, [])
            if not paths:
                continue
            try:
                img = cv2.imread(str(paths[0]))
                if img is None:
                    continue
                h, w = img.shape[:2]
                scale = 256 / max(h, w)
                if scale < 1.0:
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    top_refs[name] = [enc.tobytes()]
            except Exception:
                continue
        return top_refs

    def _run_ai_recognition(
        self,
        *,
        info: EpisodeInfo,
        cb: ProgressCb,
        anime_id: int,
        episode_id: int,
        episode_root: Path,
        metadata_dir: Path,
        bundle,
        refs_per_char: dict,
        cut_results: list,
        ai_mode: AIMode = AIMode.FULL,
    ) -> PipelineResult:
        """AI-only recognition: each shot is classified by sending its
        middle keyframe to the LLM (Gemini via NavyAI). No CLIP, no face
        detection. Slower and costs tokens, but leverages the model's
        prior anime knowledge — useful for well-known series.
        """
        cfg = self.cfg
        cb("embed_refs", -1.0, "Carregando conexão com IA...")
        client = self._build_ai_client()

        # Face detector for hybrid mode (YOLO -> face crops -> Gemini).
        face_det = None
        if ai_mode == AIMode.HYBRID:
            try:
                face_det = AnimeFaceDetector(ensure_cascade(cfg.models_path))
            except Exception as e:
                print(f"[AI analyze] Não consegui carregar o face detector: {e}")
                face_det = None
                ai_mode = AIMode.FULL

        db_chars = {c["name"]: c for c in self.db.get_characters_for_anime(anime_id)}
        character_names = [ch.name for ch in bundle.characters if refs_per_char.get(ch.name)]
        if bundle.cache_id_override:
            _ai_cache_id = bundle.cache_id_override
        elif bundle.franchise_root_id:
            _ai_cache_id = f"al{bundle.franchise_root_id}"
        elif bundle.anilist_id:
            _ai_cache_id = f"al{bundle.anilist_id}"
        else:
            _ai_cache_id = f"mal{bundle.mal_id}"
        refs_dir_str = str(
            ReferenceStore(cfg.cache_path).anime_dir(_ai_cache_id) / "characters"
        )
        if not character_names:
            raise InsufficientRefsError(
                "Nenhum personagem tem foto de referência — a IA não teria "
                "nomes nem rostos pra comparar.\n\n"
                "Causa mais comum: as fontes de imagens (Jikan/MyAnimeList) "
                "estão instáveis ou fora do ar agora. Tente de novo mais "
                "tarde (os shots cortados ficam em cache) ou adicione fotos "
                "manualmente pela pasta de refs.\n\n"
                "Detalhes no app.log (Configurações → Abrir pasta de logs).",
                refs_dir=refs_dir_str,
            )
        low_refs_warning = None
        n_missing = len(bundle.characters) - len(character_names)
        if len(character_names) <= 2 and n_missing > 0:
            low_refs_warning = (
                f"Só {len(character_names)} personagem(ns) tinham fotos de "
                f"referência — {n_missing} ficaram de fora e não podem ser "
                "identificados neste episódio."
            )
        top_refs = self._build_top_refs(bundle, refs_per_char)
        cb("embed_refs", 1.0, f"IA pronta com {len(character_names)} personagens")

        # 6) Analyze shots with Gemini
        per_shot_names: list[list[str]] = []
        shot_db_ids: list[int] = []
        total = len(cut_results)
        total_pt = 0
        total_ct = 0
        name_to_id = {n: db_chars[n]["id"] for n in character_names if n in db_chars}

        skipped_no_face = 0
        # A dead model / bad key fails EVERY request the same way. Without a
        # circuit breaker the run grinds through all N shots (minutes of
        # retries + wasted quota) to deliver an empty result.
        consecutive_ai_errors = 0
        max_consecutive_ai_errors = 8

        def _register_ai_error(shot_idx: int, err: Exception) -> None:
            nonlocal consecutive_ai_errors
            if isinstance(err, QuotaExhaustedError):
                # Every configured provider is out of quota for the day —
                # each further shot would fail identically.
                client.close()
                raise RuntimeError(
                    "Quota diária de IA esgotada em todos os provedores "
                    "configurados — análise abortada.\n\n"
                    f"Detalhe: {err}\n\n"
                    "Opções: esperar o reset da quota (NavyAI e Gemini resetam "
                    "1x por dia), configurar outra API key em Configurações, ou "
                    "usar o botão 'Analisar episódio' (CLIP local, sem IA e sem "
                    "limite de uso)."
                )
            consecutive_ai_errors += 1
            print(f"[AI analyze] Shot #{shot_idx:04d} ERRO: {err}", flush=True)
            if consecutive_ai_errors >= max_consecutive_ai_errors:
                client.close()
                raise RuntimeError(
                    f"A IA falhou em {consecutive_ai_errors} shots seguidos — análise "
                    f"abortada pra não desperdiçar tempo e quota.\n\n"
                    f"Último erro: {err}\n\n"
                    "Confira o modelo e as API keys em Configurações. O detalhe "
                    "completo de cada tentativa está no app.log (Configurações → "
                    "Abrir pasta de logs)."
                )

        print(f"[AI analyze] Iniciando classificação de {total} shots (modo={ai_mode})...", flush=True)
        for i, (shot, shot_file, kfs) in enumerate(cut_results, 1):
            cb("analyze_shots", i / max(total, 1), f"AI Shot {i}/{total}")
            main_kf = kfs[len(kfs) // 2] if kfs else None
            shot_id = self.db.insert_shot(
                episode_id=episode_id,
                idx=shot.idx,
                file=str(shot_file.relative_to(episode_root)),
                keyframe=str(main_kf.relative_to(episode_root)) if main_kf else None,
                start=shot.start,
                end=shot.end,
            )
            shot_db_ids.append(shot_id)

            if not main_kf or not main_kf.exists():
                per_shot_names.append([])
                continue

            try:
                img = cv2.imread(str(main_kf))
                if img is None:
                    per_shot_names.append([])
                    continue
            except Exception:
                per_shot_names.append([])
                continue

            # --- Hybrid mode: YOLO face crops sent to Gemini ---
            if ai_mode == AIMode.HYBRID and face_det is not None:
                # Use a wider pad than CLIP so hair/headband is visible —
                # critical for similar-faced mains (Chrome vs Senku, etc.).
                faces = face_det.crop_faces(img, pad=cfg.face_crop_padding_ai)
                if not faces:
                    skipped_no_face += 1
                    per_shot_names.append([])
                    continue

                face_bytes_list: list[bytes] = []
                for face in faces[:5]:  # cap 5 faces per shot
                    fh, fw = face.shape[:2]
                    scale = 256 / max(fh, fw)
                    if scale < 1.0:
                        face = cv2.resize(face, (int(fw * scale), int(fh * scale)))
                    ok, enc = cv2.imencode(".jpg", face, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        face_bytes_list.append(enc.tobytes())
                if not face_bytes_list:
                    per_shot_names.append([])
                    continue

                try:
                    verdicts, usage = classify_face_crops(
                        client, face_bytes_list, character_names, bundle.title, top_refs=top_refs
                    )
                except Exception as e:
                    _register_ai_error(shot.idx, e)
                    per_shot_names.append([])
                    continue

                pt = int(usage.get("prompt_tokens") or 0)
                ct = int(usage.get("completion_tokens") or 0)
                total_pt += pt
                total_ct += ct

                # Faces were sent but nothing usable came back (empty content /
                # unparseable JSON — a real "none" verdict still yields one
                # entry per face). That's a malfunction, not a miss: count it
                # toward the circuit breaker instead of silently burning the
                # whole episode's quota on 200-but-empty responses.
                if not verdicts:
                    _register_ai_error(
                        shot.idx,
                        RuntimeError(
                            f"resposta da IA sem conteúdo utilizável "
                            f"(tokens={pt}+{ct}) — veja 'resposta 200 mas "
                            f"VAZIA' no app.log"
                        ),
                    )
                    per_shot_names.append([])
                    continue
                consecutive_ai_errors = 0

                names_in_shot: list[str] = []
                seen: set[str] = set()
                for (fname, fconf) in verdicts:
                    if fname == "none" or fconf < cfg.default_threshold:
                        continue
                    if fname not in name_to_id:
                        continue
                    if fname in seen:
                        continue
                    seen.add(fname)
                    self.db.assign_character(shot_id, name_to_id[fname], fconf)
                    names_in_shot.append(fname)

                verdict_str = "+".join(names_in_shot) if names_in_shot else "NONE"
                print(
                    f"[AI analyze] Shot #{shot.idx:04d} ({len(face_bytes_list)} faces) -> {verdict_str} "
                    f"| tokens={pt}+{ct}",
                    flush=True,
                )
                per_shot_names.append(names_in_shot)
                continue

            # --- Full-frame mode ---
            try:
                h, w = img.shape[:2]
                scale = 512 / max(h, w)
                if scale < 1.0:
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not ok:
                    per_shot_names.append([])
                    continue
                frame_bytes = enc.tobytes()
            except Exception:
                per_shot_names.append([])
                continue

            try:
                name, conf, reason, usage = classify_frame(
                    client, frame_bytes, character_names, bundle.title, top_refs=top_refs
                )
            except Exception as e:
                _register_ai_error(shot.idx, e)
                per_shot_names.append([])
                continue
            consecutive_ai_errors = 0

            pt = int(usage.get("prompt_tokens") or 0)
            ct = int(usage.get("completion_tokens") or 0)
            total_pt += pt
            total_ct += ct

            verdict = name or "NONE"
            print(
                f"[AI analyze] Shot #{shot.idx:04d} -> {verdict} (conf={conf:.2f}) "
                f"| tokens={pt}+{ct} | {(reason or '')[:70]}",
                flush=True,
            )

            if name and conf >= cfg.default_threshold and name in name_to_id:
                self.db.assign_character(shot_id, name_to_id[name], conf)
                per_shot_names.append([name])
            else:
                per_shot_names.append([])

        client.close()

        cost_est = total_pt / 1_000_000 * 0.075 + total_ct / 1_000_000 * 0.30
        extra = f" | shots sem rosto (pulados): {skipped_no_face}" if ai_mode == AIMode.HYBRID else ""
        print(
            f"[AI analyze] === FIM === tokens: {total_pt:,}+{total_ct:,} "
            f"(total {total_pt + total_ct:,}) | custo estimado: ${cost_est:.4f}{extra}",
            flush=True,
        )
        cb("second_pass", 1.0, "—")  # estágios que não se aplicam no modo IA puro
        cb("ai_review", 1.0, "—")

        return self._finalize_episode(
            cb=cb,
            info=info,
            episode_id=episode_id,
            episode_root=episode_root,
            metadata_dir=metadata_dir,
            bundle=bundle,
            refs_per_char=refs_per_char,
            cut_results=cut_results,
            shot_db_ids=shot_db_ids,
            per_shot_names=per_shot_names,
            name_to_id=name_to_id,
            characters_json=[
                {
                    "name": n,
                    "character_id": name_to_id.get(n),
                    "threshold": cfg.default_threshold,
                    "reference_count": len(refs_per_char.get(n, [])),
                }
                for n in character_names
            ],
            refs_dir=refs_dir_str,
            low_refs_warning=low_refs_warning,
        )
