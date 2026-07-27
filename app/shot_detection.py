from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from scenedetect import ContentDetector, SceneManager, open_video


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
    threshold: float = 27.0,
    min_seconds: float = 0.6,
    on_progress: Callable[[float], None] | None = None,
) -> list[ShotBounds]:
    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
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
