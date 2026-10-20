from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from scenedetect import AdaptiveDetector, SceneManager, open_video

from .corte_v2 import quadros, refinar_cortes
from .corte_v3 import refinar_v3, sinais_v3

#: Identidades do corte dentro do cache (`shot_bounds.json`).
#:
#: As três primeiras são as MESMAS do Dangai (o "Gerar recap" do Levi), que
#: usa o mesmo motor: os dois apps aceitam o cache um do outro e dão os
#: mesmos cortes. Ver `corte_v2.py` e `corte_v3.py`, copiados de lá sem
#: mudança.
#:
#: v1 — o AdaptiveDetector sozinho (razão 2,0, piso 15, curta-cola).
DETECTOR_ID_V1 = "adaptive/ratio=2.0/floor=15.0/curta-cola"
#: v2 — mais flash, luz, fade na saída do escuro e 2ª chance.
DETECTOR_ID_V2 = "adaptive/ratio=2.0/floor=15.0/curta-cola/v2-flash-luz-fade"
#: v3 — mais cor, saída do branco e a nota aprendida no estudo do Levi.
DETECTOR_ID_V3 = "adaptive/ratio=2.0/floor=15.0/curta-cola/v3-cor-branco-nota"
#: O da 1.32.x do AnCut: o documento do aniFlow portado ao pé da letra. Saiu
#: porque o estudo do Levi achou dois defeitos nele — a regra de flash
#: confundia "a luz ainda não mudou" com "voltou" (12 de 23 cortes tirados
#: eram troca de câmera de verdade) e o dissolve inventava corte em pan e
#: efeito (34 de 50). Continua ACEITO no cache pelo motivo abaixo.
DETECTOR_ID_ANIFLOW = "adaptive/ratio=2.0/floor=15.0/curta-cola/refino-aniflow-1"

#: A versão do corte que episódio NOVO ganha.
VERSAO_DO_CORTE = 3
_IDENTIDADES = {1: DETECTOR_ID_V1, 2: DETECTOR_ID_V2, 3: DETECTOR_ID_V3}
DETECTOR_ID = _IDENTIDADES[min(max(VERSAO_DO_CORTE, 1), 3)]

#: Episódio com QUALQUER uma destas no cache nunca é recortado, nem na
#: reanálise.
#:
#: Antes, trocar o corte recortava o acervo inteiro na reanálise seguinte —
#: e os números das cenas mudavam. A curadoria manual (cena tirada ou movida
#: de personagem) é guardada pelo NÚMERO da cena, então passava a valer pra
#: outra cena. O Dangai já tinha essa regra; agora os dois concordam. Cache de
#: um corte anterior a estes (razão 3,0, ou sem identidade) continua sendo
#: refeito, como sempre foi.
DETECTORES_ACEITOS = frozenset(
    {DETECTOR_ID_V1, DETECTOR_ID_V2, DETECTOR_ID_V3, DETECTOR_ID_ANIFLOW}
)


class Cortes(list):
    """A lista de ShotBounds, com a identidade do corte que saiu de verdade
    (o v3 que falha volta ao v2, o v2 ao v1, e o cache grava o que saiu)."""

    def __init__(self, shots=(), detector: str = DETECTOR_ID_V1) -> None:
        super().__init__(shots)
        self.detector = detector


@dataclass
class ShotBounds:
    idx: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_shots(
    video_path: str | Path,
    min_content_val: float = 15.0,
    adaptive_ratio: float = 2.0,
    min_seconds: float = 0.6,
    on_progress: Callable[[float], None] | None = None,
    versao: int | None = None,
) -> Cortes:
    """Onde cada cena começa e termina: o AdaptiveDetector e, por cima, o
    corte v3 (ou o v2, se o v3 falhar), com os sinais tirados do mesmo quadro
    que o detector já usa, na mesma leitura do vídeo.

    **Por que adaptativo.** O `ContentDetector` comparava cada quadro com o
    anterior e cortava quando a diferença passava de um número fixo (27). Isso
    falha exatamente onde o anime mais precisa: cena escura, chuva, diálogo em
    close — dois planos de rostos escuros não chegam aos 27 e viram UM clipe.
    Medido no acervo: 334 clipes de 8s ou mais fora dos créditos em 11
    episódios; o pior, no Re:ZERO S04E11, com 40,6s e Subaru e Emilia se
    revezando quatro vezes. O `AdaptiveDetector` compara a diferença com a
    VIZINHANÇA dela, e um corte dentro de cena escura se destaca do escuro.

    **Razão 2,0, não 3,0.** Comparar com a vizinhança falha quando ela inteira
    já está agitada: numa batalha de magia nenhum corte chega a 3x a média.
    Num clipe do Slime S04E23 com ~9 cenas em 10,3s, a 3,0 achou zero cortes;
    a 2,0 acha 6. A 1,5 achava 12, mas com o dobro de fragmentos curtos.

    **O v3 por cima** (estudo do Levi, quadro a quadro, 6 episódios, 3 deles
    nunca vistos pelo ajuste): nos três novos, "dois planos viram um" caiu de
    21 pra 4 e os cortes inventados de 122 pra 78. Ver `corte_v3.py`.
    """
    if versao is None:
        versao = VERSAO_DO_CORTE
    v2 = versao >= 2
    nivel = 3 if versao >= 3 else (2 if v2 else 0)
    video = open_video(str(video_path))
    fps = float(video.frame_rate)
    minimo = quadros(15, fps) if v2 else 15
    sm = SceneManager()
    if v2:
        detector = _AdaptiveComSinais(
            adaptive_threshold=adaptive_ratio, min_content_val=min_content_val,
            min_scene_len=minimo, nivel=nivel,
        )
    else:
        detector = AdaptiveDetector(
            adaptive_threshold=adaptive_ratio, min_content_val=min_content_val,
            min_scene_len=minimo,
        )
    sm.add_detector(detector)
    # detect_scenes blocks for the whole episode (minutes). The per-cut
    # callback (fires every few seconds of video) feeds real progress to the
    # UI — and gives the cancel button a place to land mid-detection.
    callback = None
    if on_progress is not None:
        total_frames = max(int(video.duration.get_frames() or 0), 1)

        def callback(_image, frame_num) -> None:
            # scenedetect 0.7.x passa um FrameTimecode aqui (antes era int).
            fn = frame_num.get_frames() if hasattr(frame_num, "get_frames") else int(frame_num)
            on_progress(min(fn / total_frames, 1.0))

    sm.detect_scenes(video, show_progress=False, callback=callback)
    scenes = sm.get_scene_list()

    identidade = DETECTOR_ID_V1
    feito = False
    if nivel >= 3:
        try:
            scenes = _cenas_v2(scenes, detector.sinais, fps, adaptive_ratio, min_content_val, v3=True)
            identidade = DETECTOR_ID_V3
            feito = True
        except Exception as e:  # noqa: BLE001 -- o corte v2 continua valendo
            print(f"[corte] corte v3 nao serviu ({type(e).__name__}: {e}); tentando o v2.", flush=True)
    if v2 and not feito:
        try:
            scenes = _cenas_v2(scenes, detector.sinais, fps, adaptive_ratio, min_content_val)
            identidade = DETECTOR_ID_V2
        except Exception as e:  # noqa: BLE001 -- o corte v1 continua valendo
            print(f"[corte] corte v2 nao serviu ({type(e).__name__}: {e}); fica o corte v1.", flush=True)

    # Cena curta demais COLA na anterior, não some.
    #
    # Antes era `continue`: o trecho saía da lista e nenhum clipe cobria
    # aqueles segundos — episódio perdido em silêncio. Colando, ficar mais
    # sensível não custa material nenhum.
    shots: list[ShotBounds] = []
    for s, e in scenes:
        start = s.get_seconds()
        end = e.get_seconds()
        if end - start < min_seconds and shots:
            shots[-1].end = end
            continue
        shots.append(ShotBounds(idx=len(shots), start=start, end=end))

    if not shots:
        # fallback: whole video as one shot
        dur = video.duration.get_seconds()
        shots = [ShotBounds(idx=0, start=0.0, end=dur)]

    if on_progress is not None:
        on_progress(1.0)
    return Cortes(shots, identidade)


def _luz_e_hist(quadro):
    """A luz e o histograma do corte v2. ``quadro`` é o quadro que o
    SceneManager já reduziu (256x144 num 1080p), levado a 128x72 com
    INTER_AREA. luz = luma média / 255, com luma = (77R + 150G + 29B) >> 8;
    hist = a luma em 16 faixas (luma >> 4), somando 1."""
    import cv2
    import numpy as np

    p = cv2.resize(quadro, (128, 72), interpolation=cv2.INTER_AREA)
    b = p[..., 0].astype(np.uint32)
    g = p[..., 1].astype(np.uint32)
    r = p[..., 2].astype(np.uint32)
    y = (77 * r + 150 * g + 29 * b) >> 8
    return float(y.mean()) / 255, (np.bincount((y >> 4).ravel(), minlength=16) / y.size).astype(np.float32)


def _sinais_do_quadro(quadro, nivel: int) -> tuple:
    """(luz, hist) no v2; (luz, hist, hs, mini) no v3."""
    return sinais_v3(quadro) if nivel >= 3 else _luz_e_hist(quadro)


class _AdaptiveComSinais(AdaptiveDetector):
    """O AdaptiveDetector guardando, quadro a quadro, a posição, a nota, a luz
    e o histograma (e, no v3, o hs e a mini)."""

    def __init__(self, nivel: int = 2, **kw) -> None:
        super().__init__(**kw)
        self.nivel = nivel
        self.sinais: list = []

    def _calculate_frame_score(self, timecode, frame_img) -> float:
        nota = super()._calculate_frame_score(timecode, frame_img)
        self.sinais.append((timecode, nota, *_sinais_do_quadro(frame_img, self.nivel)))
        return nota


def _indices_dos_cortes(pos, inicios) -> list[int]:
    """O índice, na lista de quadros lidos, de cada início de cena.

    Casa pela ORDEM dos quadros: cada corte é procurado a partir do quadro
    seguinte ao do corte anterior. Vale o próprio objeto da posição (o
    detector devolve no corte a posição que recebeu do quadro); sem ele, o 1º
    quadro com a mesma posição exata (o PTS). NÃO use um dicionário
    frame_num → índice: vídeo com dois quadros do mesmo frame_num (o frame_num
    é o tempo arredondado; o Cyberpunk Edgerunners S02E01 tem dois 8098)
    derrubava o episódio inteiro pro v1.
    """
    cortes: list[int] = []
    j = 1
    for s in inicios:
        achou = igual = None
        i = j
        while i < len(pos):
            if pos[i] is s:
                achou = i
                break
            if pos[i] == s:
                if igual is None:
                    igual = i
            elif igual is not None:
                break  # passou da fileira de quadros com a mesma posição
            i += 1
        if achou is None:
            achou = igual
        if achou is None:
            raise ValueError(f"corte {s.frame_num} fora dos quadros lidos")
        cortes.append(achou)
        j = achou + 1
    return cortes


def _cenas_v2(scenes, sinais, fps: float, ratio: float, piso: float, v3: bool = False):
    """As cenas do AdaptiveDetector passadas pelo corte v2
    (corte_v2.refinar_cortes) ou, com ``v3``, pelo corte v3
    (corte_v3.refinar_v3)."""
    import numpy as np

    if not scenes:
        return scenes
    if not sinais:
        raise ValueError("sem os sinais dos quadros")
    pos = [x[0] for x in sinais]
    if pos[0] != scenes[0][0]:
        raise ValueError("a 1a cena nao comeca no 1o quadro lido")
    cortes = _indices_dos_cortes(pos, [s for s, _e in scenes[1:]])
    nota = np.array([x[1] for x in sinais], dtype=np.float64)
    luz = np.array([x[2] for x in sinais], dtype=np.float64)
    hist = np.stack([x[3] for x in sinais])
    if v3:
        if any(len(x) < 6 for x in sinais):
            raise ValueError("sem o hs e a mini dos quadros")
        final = refinar_v3(nota, luz, hist, np.stack([x[4] for x in sinais]),
                           np.stack([x[5] for x in sinais]), cortes, fps,
                           razao=ratio, piso=piso)
    else:
        final = refinar_cortes(nota, luz, hist, cortes, fps, razao=ratio, piso=piso)
    inicios = [scenes[0][0]] + [pos[c] for c in final]
    return list(zip(inicios, inicios[1:] + [scenes[-1][1]]))
