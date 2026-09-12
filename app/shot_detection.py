from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from scenedetect import AdaptiveDetector, SceneManager, open_video

#: Identidade do detector dentro do cache de cortes (`shot_bounds.json`).
#:
#: MUDE isto junto com qualquer coisa que altere ONDE os cortes caem. O cache
#: é por episódio e sobrevive à reanálise: sem trocar a identidade, episódio
#: já analisado reusaria os cortes velhos, e a melhora não apareceria
#: justamente pra quem já tem acervo.
DETECTOR_ID = "adaptive/ratio=3.0/floor=15.0"


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
    adaptive_ratio: float = 3.0,
    min_seconds: float = 0.6,
    on_progress: Callable[[float], None] | None = None,
) -> list[ShotBounds]:
    """Onde cada cena começa e termina, por detecção ADAPTATIVA.

    O `ContentDetector` comparava cada quadro com o anterior e cortava quando
    a diferença passava de um número fixo (27). Isso falha exatamente onde o
    anime mais precisa: cena escura, chuva, diálogo em close. Dois planos
    diferentes de rostos escuros têm diferença numérica pequena, não chegam
    aos 27, e viram UM clipe só.

    Medido no acervo: 334 clipes de 8s ou mais FORA dos créditos, em 11
    episódios — uns 30 por episódio. O pior, no Re:ZERO S04E11, tinha 40,6s
    com Subaru e Emilia se revezando quatro vezes. Um clipe assim não serve
    pra nenhum dos dois: ele não é de ninguém.

    O `AdaptiveDetector` compara a diferença de cada quadro com a VIZINHANÇA
    dela (`adaptive_ratio` vezes a média móvel) em vez de com uma constante,
    então um corte dentro de uma cena escura se destaca do próprio escuro ao
    redor. `min_content_val` é só um piso contra ruído de cena parada — não é
    mais o gatilho.

    Medido naquele clipe de 40,6s: acha os 4 cortes, nos lugares certos. No
    Bleach S01E06 inteiro, 419 -> 466 cenas (+11%) e ~14% mais rápido.
    """
    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(
        AdaptiveDetector(adaptive_threshold=adaptive_ratio, min_content_val=min_content_val)
    )
    # detect_scenes blocks for the whole episode (minutes). The per-cut
    # callback (fires every few seconds of video) feeds real progress to the
    # UI — and gives the cancel button a place to land mid-detection.
    callback = None
    if on_progress is not None:
        total_frames = max(int(video.duration.get_frames() or 0), 1)

        def callback(_image, frame_num) -> None:
            # scenedetect 0.7.x passa um FrameTimecode aqui (antes era int).
            # get_frames() dá o número inteiro do frame nas duas versões.
            fn = frame_num.get_frames() if hasattr(frame_num, "get_frames") else int(frame_num)
            on_progress(min(fn / total_frames, 1.0))

    sm.detect_scenes(video, show_progress=False, callback=callback)
    scenes = sm.get_scene_list()

    shots: list[ShotBounds] = []
    idx = 0
    for s, e in scenes:
        start = s.get_seconds()
        end = e.get_seconds()
        if end - start < min_seconds:
            continue
        shots.append(ShotBounds(idx=idx, start=start, end=end))
        idx += 1

    if not shots:
        # fallback: whole video as one shot
        dur = video.duration.get_seconds()
        shots = [ShotBounds(idx=0, start=0.0, end=dur)]

    if on_progress is not None:
        on_progress(1.0)
    return shots
