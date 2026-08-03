# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AnCut HUB.

Build with:
    pyinstaller build.spec --noconfirm --clean

Output: dist/CorteCenas/ (onedir mode).
Zip this folder to share with users who have an NVIDIA GPU + CUDA 12.8 driver.
FFmpeg binary must be available on the target machine's PATH.
"""
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)


# --- Hidden imports ---------------------------------------------------------
# Heavy ML libs rely on runtime-discovered modules that PyInstaller can't
# find statically. We collect them explicitly.

hidden = []
hidden += collect_submodules("open_clip")
hidden += collect_submodules("ultralytics")
hidden += collect_submodules("torch")
hidden += collect_submodules("torchvision")
hidden += collect_submodules("cv2")
hidden += collect_submodules("PIL")

# Extra imports for model registries / tokenizer files these libs pull lazily.
hidden += [
    "ftfy",
    "regex",
    "huggingface_hub",
    "hf_xet",           # fast-path de download do HF Hub (senão cai em HTTP + warning)
    "safetensors",
    "timm",
    "yaml",
]

# --- QtMultimedia (preview de vídeo embutido + áudio no hover) --------------
# O import é feito sob try/except em preview_panel.py/character_grid.py e o
# PyInstaller não tem hook próprio pro QtMultimedia — então coletamos à mão o
# que ele deixaria de fora: os plugins de mídia e o backend FFmpeg do Qt.
# hiddenimports garante que os módulos entrem; os binaries garantem os plugins.
# Sem isto o app roda, mas cai no fallback (player externo, sem preview/hover).
import glob as _glob
import os as _os

import PySide6 as _ps6

_ps6_base = _os.path.dirname(_ps6.__file__)
binaries = []
# ffmpegmediaplugin.dll / windowsmediaplugin.dll -> PySide6/plugins/multimedia
for _dll in _glob.glob(_os.path.join(_ps6_base, "plugins", "multimedia", "*.dll")):
    binaries.append((_dll, "PySide6/plugins/multimedia"))
# Backend FFmpeg do Qt (avcodec/avformat/avutil/swresample/swscale) + DLLs core
# do multimedia -> raiz PySide6/
for _pat in ("av*.dll", "sw*.dll", "Qt6Multimedia*.dll"):
    for _dll in _glob.glob(_os.path.join(_ps6_base, _pat)):
        binaries.append((_dll, "PySide6"))

hidden += [
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtNetwork",   # dependência do QtMultimedia (streaming/rede)
    # CCIP: importado tarde (só quando há decisão apertada pra conferir), e
    # import tardio é justamente o que o PyInstaller não enxerga sozinho.
    "onnxruntime",
    "onnxruntime.capi._pybind_state",
]


# --- Data files -------------------------------------------------------------
datas = []
# Bundled FFmpeg (~200 MB). Put next to the app so users don't have to
# install ffmpeg separately or add anything to PATH. Populated by
# fetch_ffmpeg.py before this spec runs (see _build_all.bat).
datas += [("bin/ffmpeg.exe", "bin")]
datas += [("bin/ffprobe.exe", "bin")]
# Elevated helper for delta updates. Shipped alongside the exe so the
# updater can hand it a source dir + target dir and let it copy files.
datas += [("apply_update.ps1", ".")]
# Deps fingerprint (sha256 do requirements.txt, gerado por
# gen_deps_fingerprint.py). O updater compara o do delta zip com o local:
# deps diferentes = delta entregaria app quebrado -> forca setup completo.
datas += [("app/deps_fingerprint.txt", "app")]
# App icon (all sizes) — needed at runtime for QApplication.setWindowIcon().
datas += [("app/assets/icon.ico", "app/assets")]
datas += [("app/assets/icon_256.png", "app/assets")]
datas += [("app/assets/icon_128.png", "app/assets")]
datas += [("app/assets/icon_64.png", "app/assets")]
datas += [("app/assets/icon_48.png", "app/assets")]
datas += [("app/assets/icon_32.png", "app/assets")]
datas += [("app/assets/icon_16.png", "app/assets")]
# Marca do cabeçalho (logo recortada, sem a margem do ícone quadrado).
datas += [("app/assets/logo_header.png", "app/assets")]
datas += collect_data_files("open_clip")
datas += collect_data_files("ultralytics")
datas += collect_data_files("torchvision")
# Package metadata some libs inspect at runtime:
datas += copy_metadata("torch")
datas += copy_metadata("open_clip_torch")
datas += copy_metadata("ultralytics")
datas += copy_metadata("huggingface_hub")


a = Analysis(
    ["run.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib.tests",
        "notebook",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CorteCenas",
    icon="app/assets/icon.ico",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                # UPX often breaks torch/cuda libs; disable
    console=False,            # no CMD popup; crash traceback is captured
                              # by run.py's crash handler into a log + dialog
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="CorteCenas",
)
