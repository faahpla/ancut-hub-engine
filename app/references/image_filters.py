from __future__ import annotations

import re

import cv2
import numpy as np


# Files saved by image_downloader are named <sha1[:16]>.<ext>. Anything else
# in a character folder is user-added and should be preserved as-is.
HASH_FILENAME = re.compile(r"^[0-9a-f]{16}\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)

#: A imagem "NO IMAGE" da AniList (character/large|medium/default.jpg), pelo
#: nome que o downloader dá a ela. Baixada antes do filtro no provedor, ela
#: ficou nas pastas de refs de quem não tem foto e não serve de nada.
PLACEHOLDERS = {"e51c025d5f10f88c.jpg", "1106aa5a68b78735.jpg"}


def is_monochrome(img_bgr: np.ndarray, saturation_threshold: float = 18.0) -> bool:
    """Heuristic: HSV saturation mean below threshold → monochrome/manga panel.

    Color anime screenshots typically have mean saturation 60-150. Manga
    scans sit near 0-10 even when not perfectly grayscale (paper tone).
    """
    if img_bgr is None or img_bgr.size == 0:
        return False
    if img_bgr.ndim == 2:
        return True
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) < saturation_threshold
