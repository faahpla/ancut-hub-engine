"""Refino dos cortes do detector adaptativo — portado do aniFlow (do Ajk).

O AdaptiveDetector acha os cortes; três passos depois limpam a lista:

1. **Flash não é corte.** Num golpe a tela estoura de branco por 2 a 6
   quadros e volta pro MESMO plano. O detector via dois saltos e cortava
   duas vezes. Se a luz se afasta forte e volta pro nível de antes, com o
   mesmo histograma, os cortes de dentro saem.
2. **Corte de verdade muda a luz.** Movimento dentro do plano (o personagem
   vira o rosto, passam linhas de velocidade) dá salto de conteúdo mas quase
   não mexe na distribuição de luz. O corte só fica com salto enorme (≥ 38)
   ou se o histograma médio dos 3 quadros antes e dos 3 depois diferir
   ≥ 0,06.
3. **Dissolve.** A cena muda devagar e nenhum quadro passa do limiar, mas o
   quadro 10 antes e o 10 depois não têm nada a ver: o corte vai no ponto de
   maior diferença.

Medido no AnCut (por cima do nosso detector, não no lugar dele), conferindo
cada corte mudado com os olhos:

- Slime S04E23 (luta, magia piscando): o passo 2 tirou 57 cortes, todos
  falsos; pedaços de menos de 1s caíram de 106 pra 70; 14 dissolves.
- Apothecary S02E05 (diálogo, dança): tirou 28, todos falsos; 70 → 53.
- Os cortes de verdade não mudaram: o clipe de 10s com 6 trocas de câmera
  do S04E23 sai cortado nos mesmos 6 lugares.

O medo era o passo 2 juntar plano e contraplano de diálogo (luz parecida);
no Apothecary não aconteceu em nenhum dos 28.

Os números de quadro assumem ~24 fps e são escalados por `fps / 24`.
"""

from __future__ import annotations

import cv2
import numpy as np
from scenedetect import SceneDetector


class Sinais(SceneDetector):
    """Detector que não detecta: anota luz e conteúdo de cada quadro.

    Entra no MESMO SceneManager do AdaptiveDetector, então o vídeo é lido uma
    vez só. Tudo no quadro reduzido a 128x72, como no aniFlow — os limiares
    abaixo foram medidos nessa escala.
    """

    def __init__(self) -> None:
        super().__init__()
        self.quadros: list[int] = []
        self.conteudo: list[float] = []
        self.luz: list[float] = []
        self.hist: list[np.ndarray] = []
        self._hsv_ant: np.ndarray | None = None

    def process_frame(self, timecode, frame_img):  # noqa: D401 — API do PySceneDetect
        p = cv2.resize(frame_img, (128, 72), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV).astype(np.int16)
        b, g, r = (p[..., k].astype(np.uint32) for k in range(3))
        y = (77 * r + 150 * g + 29 * b) >> 8
        self.conteudo.append(
            0.0 if self._hsv_ant is None
            else float(np.abs(hsv - self._hsv_ant).reshape(-1, 3).mean(axis=0).mean())
        )
        self.luz.append(float(y.mean()) / 255)
        self.hist.append((np.bincount((y >> 4).ravel(), minlength=16) / y.size).astype(np.float32))
        self.quadros.append(
            timecode.get_frames() if hasattr(timecode, "get_frames") else int(timecode)
        )
        self._hsv_ant = hsv
        return []


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum()) / 2


def suprimir_flashes(cortes, luz, hist, max_q=7):
    fora = set()
    for c in cortes:
        if c < 2 or c + max_q + 1 >= len(luz):
            continue
        antes = luz[c - 1]
        if max(abs(luz[k] - antes) for k in range(c, c + max_q)) < 0.2:
            continue
        for fim in range(c + 1, c + max_q + 1):
            if abs(luz[fim] - antes) < 0.07 and _dist(hist[c - 1], hist[fim]) < 0.18:
                fora.update(d for d in cortes if c <= d <= fim)
                break
    return [c for c in cortes if c not in fora]


def confirmar_cortes(cortes, conteudo, hist, raio=3):
    def media(de, ate):
        return np.mean(hist[max(0, de):min(len(hist), ate)], axis=0)

    return [
        c for c in cortes
        if conteudo[c] >= 38 or _dist(media(c - raio, c), media(c, c + raio)) >= 0.06
    ]


def achar_dissolves(conteudo, hist, cortes, piso=15, raio=10):
    novos: list[int] = []
    melhor: tuple[int, float] | None = None

    def fechar() -> None:
        nonlocal melhor
        if melhor and all(abs(c - melhor[0]) > raio + 5 for c in list(cortes) + novos):
            novos.append(melhor[0])
        melhor = None

    for i in range(raio, len(hist) - raio):
        d = _dist(hist[i - raio], hist[i + raio])
        janela = conteudo[i - raio + 1:i + raio + 1]
        if d > 0.4 and max(janela) < piso and sum(janela) / (2 * raio) > 2.5:
            if not melhor or d > melhor[1]:
                melhor = (i, d)
        elif melhor and i - melhor[0] > raio:
            fechar()
    fechar()
    return novos


def refinar(cortes_quadro: list[int], sinais: Sinais, fps: float) -> list[int]:
    """Cortes do detector (números de quadro) → cortes refinados.

    Os sinais são indexados pela posição na lista, e o vídeo pode não começar
    no quadro 0 do contador; por isso o mapa quadro → posição.
    """
    if len(sinais.quadros) < 30:
        return cortes_quadro
    pos = {q: i for i, q in enumerate(sinais.quadros)}
    conteudo = np.asarray(sinais.conteudo)
    luz = np.asarray(sinais.luz)
    hist = np.stack(sinais.hist)
    escala = max(fps, 1.0) / 24.0
    q = lambda n: max(1, round(n * escala))  # noqa: E731

    idx = sorted({pos[c] for c in cortes_quadro if c in pos})
    sem_flash = suprimir_flashes(idx, luz, hist, max_q=q(7))
    confirmados = confirmar_cortes(sem_flash, conteudo, hist, raio=q(3))
    dissolves = achar_dissolves(conteudo, hist, confirmados, raio=q(10))
    final = sorted(set(confirmados) | set(dissolves))
    return [sinais.quadros[i] for i in final]
