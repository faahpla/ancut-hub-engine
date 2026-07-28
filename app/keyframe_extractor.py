from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import cv2
import ffmpeg

from .ffmpeg_locate import nvenc_available, run_ffmpeg_hidden
from .shot_detection import ShotBounds

# Consumer GeForce cards cap concurrent NVENC sessions (3-8 depending on the
# driver generation). 3 parallel encodes is safe everywhere and already keeps
# the encode chip saturated for 1-4s clips.
_NVENC_WORKERS = 3
# libx264 path: each ffmpeg spawns its own encoder threads, so a modest pool
# is enough to keep every core busy without thrashing.
_CPU_WORKERS = 4


def cut_shot(
    video_path: str | Path,
    shot: ShotBounds,
    out_file: Path,
    reencode: bool = True,
    use_nvenc: bool = False,
    fps: float = 24.0,
    render_mode: str = "off",
) -> None:
    """Extract a shot to an mp4 file. Re-encode for frame accuracy, or stream-copy for speed.

    `render_mode` controla o formato de saída (ver config.render_export_mode):
    "off" mantém o comportamento antigo, "compat" força 8 bits e 23,976 CFR,
    "intra" acrescenta all-intra pra seek quadro a quadro sair barato.

    Overwrites any existing file at `out_file`.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        try:
            out_file.unlink()
        except OSError:
            pass

    # O "fim" do shot é o timestamp do PRIMEIRO frame do shot seguinte (fim
    # exclusivo, convenção do PySceneDetect). Cortar com "-to fim" na ENTRADA
    # deixava exatamente esse frame invadir o clipe quando o arredondamento de
    # timestamp caía pro lado errado (MKV marca em milissegundos) — o clássico
    # "frame de outra cena no final". A correção: duração na SAÍDA (-t, exata
    # pós-decodificação) com margem de MEIO frame — o intruso nunca entra e o
    # último frame legítimo nunca sai (ele termina 1 frame inteiro antes).
    duration = max(0.05, shot.duration - 0.5 / max(fps, 1.0))

    render = render_mode in ("compat", "intra")

    # Opções comuns aos dois encoders nos modos de render. O filtro fps com
    # fps_mode=cfr entrega cadência constante de 23,976: medido em 2,0s de
    # corte → 48 frames (esperado 47,95), nenhum duplicado, mesmo com o -ss
    # vindo ANTES do -i.
    common: dict[str, object] = {}
    if render:
        common = {"vf": "fps=24000/1001", "fps_mode": "cfr", "profile:v": "high"}

    if reencode:
        if use_nvenc:
            # GPU encode chip: ~5-10x faster than libx264 on CPU and leaves
            # the CPU free for parallel decode/keyframes.
            if render:
                # constqp é o "CRF" do NVENC: qualidade fixa, sem alvo de
                # bitrate — o certo pra material que ainda vai ser reprocessado.
                rate: dict[str, object] = {"rc": "constqp", "qp": 20}
            else:
                # rc=vbr + cq + b:v 0 é o modo de qualidade constante antigo.
                rate = {"rc": "vbr", "cq": 23, "b:v": "0"}
            if render_mode == "intra":
                # ATENÇÃO: aqui é -g 0 e no libx264 abaixo é -g 1. NÃO é erro
                # de digitação, não "conserte" pra ficarem iguais. O NVENC
                # rejeita -g 1 com "Gop Length should be greater than number
                # of B frames + 1" — mesmo com -bf 0 ele exige GOP >= 2.
                # Medido nesta máquina (RTX 3060), 72 frames:
                #   -g 1 -bf 0 → falha, não gera arquivo
                #   -g 2 -bf 0 → 36/72 keyframes
                #   -g 0 -bf 0 → 72/72 keyframes  ✓
                rate.update({"g": 0, "bf": 0})

            stream = ffmpeg.input(str(video_path), ss=shot.start).output(
                str(out_file),
                t=duration,
                vcodec="h264_nvenc",
                preset="p4",
                # OBRIGATÓRIO: o NVENC de H.264 não codifica 10 bits e falha
                # com "10 bit encode not supported / No capable devices".
                # Boa parte dos fansubs de anime é 10-bit (x265 Main 10), então
                # SEM isto o NVENC nunca entra: toda análise caía no fallback
                # de CPU sem avisar, e o corte virava 32% do tempo total.
                pix_fmt="yuv420p",
                acodec="aac",
                format="mp4",
                movflags="+faststart",
                loglevel="error",
                **rate,
                **common,
            )
        else:
            x264: dict[str, object] = {}
            if render:
                # ultrafast NÃO entrega High: ele desliga CABAC e 8x8dct, e o
                # x264 rebaixa a saída pra Constrained Baseline mesmo com
                # -profile:v high. Medido: superfast custa 8% mais tempo e
                # gera arquivo 33% MENOR (4,46 MB contra 6,67 MB).
                x264["preset"] = "superfast"
            else:
                x264["preset"] = "ultrafast"
            if render_mode == "intra":
                # No x264 é -g 1 mesmo (todo frame IDR). Ver o comentário do
                # NVENC acima pra entender por que os dois números diferem.
                x264.update({"g": 1, "bf": 0})

            stream = ffmpeg.input(str(video_path), ss=shot.start).output(
                str(out_file),
                t=duration,
                vcodec="libx264",
                crf=20,
                # Sem isto o libx264 HERDA o formato da fonte: episódio 10-bit
                # gerava clipe High 10 (verificado). H.264 High 10 não é
                # decodificado por WebCodecs nem pelo NVDEC da maioria das
                # placas, então esses clipes caem em decode por software no
                # pipeline de render. Vale SEMPRE, não só nos modos de render —
                # senão o fallback de CPU produz saída incompatível no meio de
                # um episódio que começou na GPU.
                pix_fmt="yuv420p",
                acodec="aac",
                format="mp4",
                movflags="+faststart",
                loglevel="error",
                **x264,
                **common,
            )
    else:
        stream = ffmpeg.input(str(video_path), ss=shot.start, to=shot.end).output(
            str(out_file),
            c="copy",
            format="mp4",
            avoid_negative_ts="make_zero",
            loglevel="error",
        )
    run_ffmpeg_hidden(stream)


def extract_keyframes(
    video_path: str | Path,
    shot: ShotBounds,
    out_dir: Path,
    n_frames: int = 3,
) -> list[Path]:
    """Sample N frames uniformly across the shot and save as JPGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    paths: list[Path] = []

    if n_frames <= 1:
        offsets = [0.5]
    else:
        offsets = [(i + 1) / (n_frames + 1) for i in range(n_frames)]

    for k, off in enumerate(offsets):
        t = shot.start + shot.duration * off
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out = out_dir / f"{shot.idx:04d}_{k}.jpg"
        cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(out)

    cap.release()
    return paths


def cut_all_shots(
    video_path: str | Path,
    shots: list[ShotBounds],
    shots_dir: Path,
    keyframes_dir: Path,
    keyframes_per_shot: int,
    reencode: bool,
    on_progress: Callable[[int, int, int], None] | None = None,
    skip_existing: bool = True,
    render_mode: str = "off",
) -> list[tuple[ShotBounds, Path, list[Path]]]:
    """Cut shots and extract keyframes, several shots at a time.

    Each shot is an independent (ffmpeg cut + cv2 keyframes) work unit, so
    they run in a thread pool: NVENC when the GPU has it (3 workers, safe for
    every session-limited GeForce), libx264 otherwise (4 workers). One shot
    at a time on CPU was ~86% of the whole pipeline's wall clock.

    If `skip_existing` is True, shots whose .mp4 is already on disk (non-empty)
    are not re-encoded, and keyframes already present are not re-extracted.

    `on_progress` is called from THIS thread as results complete (completion
    order, monotonic count) — raising from it (PipelineCancelled) cancels all
    queued shots; in-flight ffmpeg calls finish into the cache.
    """
    shots_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    # Clipe cortado noutro modo tem OUTRO formato, então reaproveitá-lo do
    # cache faria ligar a opção não mudar nada — o pior tipo de falha, a
    # silenciosa. O modo usado fica gravado ao lado dos clipes; se mudou,
    # todos são recortados.
    stamp = shots_dir / ".export_mode"
    try:
        previous = stamp.read_text(encoding="utf-8").strip()
    except OSError:
        # Sem marcador: ou é a primeira análise, ou os clipes vêm de uma versão
        # anterior a esta opção. Nos dois casos "off" é a suposição correta.
        previous = "off"
    if previous != render_mode:
        skip_existing = False
        if any(shots_dir.glob("*.mp4")):
            print(
                # Sem "→" nem travessão: o console do Windows é cp1252 e
                # estoura com UnicodeEncodeError em caractere fora dela,
                # derrubando a análise inteira por causa de um log.
                f"[CorteCenas] Formato de export mudou ({previous} -> {render_mode})"
                " - recortando os clipes",
                flush=True,
            )
    try:
        stamp.write_text(render_mode, encoding="utf-8")
    except OSError:
        pass  # marcador é otimização, não pode derrubar a análise

    # fps do vídeo, sondado uma vez: o corte usa meia duração de frame como
    # margem pra não deixar o primeiro frame do shot seguinte vazar pro clipe.
    probe = cv2.VideoCapture(str(video_path))
    video_fps = probe.get(cv2.CAP_PROP_FPS) if probe.isOpened() else 0.0
    probe.release()
    if not video_fps or video_fps <= 0 or video_fps != video_fps:
        video_fps = 24.0

    total = len(shots)
    # Shared, mutated on NVENC runtime failure (driver/session hiccup): the
    # remaining shots silently switch to libx264. Benign race — worst case a
    # couple extra NVENC attempts before every worker sees the flag.
    enc_state = {"nvenc": reencode and nvenc_available()}
    workers = _NVENC_WORKERS if enc_state["nvenc"] else _CPU_WORKERS
    workers = max(1, min(workers, os.cpu_count() or 4, total or 1))

    def process(shot: ShotBounds) -> tuple[ShotBounds, Path, list[Path], bool] | None:
        out_file = shots_dir / f"{shot.idx:04d}.mp4"
        expected_kfs = [keyframes_dir / f"{shot.idx:04d}_{k}.jpg" for k in range(keyframes_per_shot)]

        have_cut = out_file.exists() and out_file.stat().st_size > 0
        have_kfs = all(p.exists() and p.stat().st_size > 0 for p in expected_kfs)

        if not (skip_existing and have_cut):
            try:
                cut_shot(video_path, shot, out_file, reencode=reencode,
                         use_nvenc=enc_state["nvenc"], fps=video_fps,
                         render_mode=render_mode)
            except ffmpeg.Error:
                if enc_state["nvenc"]:
                    enc_state["nvenc"] = False
                    print(
                        f"[CorteCenas] NVENC falhou no shot {shot.idx} — "
                        "continuando na CPU (libx264)",
                        flush=True,
                    )
                    try:
                        cut_shot(video_path, shot, out_file, reencode=reencode,
                                 use_nvenc=False, fps=video_fps,
                                 render_mode=render_mode)
                    except ffmpeg.Error:
                        return None
                else:
                    return None

        if skip_existing and have_kfs:
            kfs = expected_kfs
        else:
            kfs = extract_keyframes(video_path, shot, keyframes_dir, n_frames=keyframes_per_shot)

        return shot, out_file, kfs, (skip_existing and have_cut and have_kfs)

    indexed: list[tuple[ShotBounds, Path, list[Path]] | None] = [None] * total
    done = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, shot): i for i, shot in enumerate(shots)}
        try:
            for fut in as_completed(futures):
                res = fut.result()
                done += 1
                if res is not None:
                    shot, out_file, kfs, was_skipped = res
                    indexed[futures[fut]] = (shot, out_file, kfs)
                    if was_skipped:
                        skipped += 1
                if on_progress:
                    on_progress(done, total, skipped)
        except BaseException:
            # PipelineCancelled (or anything else) — drop everything queued;
            # shots already encoding finish into the cache and get reused on
            # the next run.
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    # Original shot order, minus the ones ffmpeg couldn't cut.
    return [r for r in indexed if r is not None]
