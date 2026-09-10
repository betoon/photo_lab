"""Optional direct Nikon RAW development in an isolated native process."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import threading
import numpy as np

_lock = threading.Lock()
_RUNTIME_FILES = ('NkImgSDK.dll', 'Elm.dll', 'Elm.nlf', 'tbb.dll', 'tbbmalloc.dll', 'RCSigProc.dll', 'prm.bin')

def sdk_candidates():
    root = Path(__file__).resolve().parent
    yield root.parent / 'nikon_sdk' / 'Image SDK'
    yield Path.home() / 'Documents' / 'GitHub' / 'nikon_sdk' / 'Image SDK'

def find_sdk():
    from config import get_config
    configured = os.environ.get('PHOTOLAB_NIKON_SDK') or get_config().get('paths', 'nikon_sdk', '')
    candidates = [Path(configured)] if configured else list(sdk_candidates())
    for root in candidates:
        binary = root / 'Library' / 'win' / 'Bin' / 'x64' / 'Release'
        profiles = root / 'Library' / 'win' / 'Profiles'
        if all((binary / n).is_file() for n in _RUNTIME_FILES) and (profiles / 'NKsRGB.icm').is_file():
            return binary, profiles
    raise RuntimeError('Nikon Image SDK was not found. Set its Image SDK folder in Tools > Configuration.')

def _read_output(path):
    with open(path, 'rb') as f:
        header = f.read(16)
        if len(header) != 16:
            raise RuntimeError('Nikon decoder returned an incomplete header')
        magic, width, height, depth = struct.unpack('<4I', header)
        if magic != 0x4e4b5247 or depth not in (1, 2) or not width or not height:
            raise RuntimeError('Nikon decoder returned invalid image metadata')
        length = width * height * 3 * depth
        if length > 1024**3 or os.fstat(f.fileno()).st_size != 16 + length:
            raise RuntimeError('Nikon decoder returned an incomplete image')
        rgb = np.fromfile(f, dtype=np.uint8 if depth == 1 else '<u2').reshape(height, width, 3)
    return np.ascontiguousarray(rgb[:, :, ::-1])

def decode_nef(path, output_bps=8):
    if os.name != 'nt':
        raise RuntimeError('The Nikon SDK integration requires Windows')
    binary, profiles = find_sdk()
    helper = Path(__file__).resolve().parent / 'tools' / 'nikon_decoder.exe'
    if not helper.is_file():
        raise RuntimeError('Nikon decoder helper is missing. Run tools/build_nikon_decoder.ps1 before starting PhotoLab.')
    # Keep Nikon DLLs beside a dedicated EXE, as required by its RAW engine.
    # Never copy them into Python, PhotoLab, or the original SDK installation.
    with _lock:
        sources = [helper] + [binary / n for n in _RUNTIME_FILES]
        identity = '|'.join(f'{p}:{p.stat().st_size}:{p.stat().st_mtime_ns}' for p in sources)
        key = hashlib.sha256(identity.encode()).hexdigest()[:20]
        cache = Path(os.environ.get('LOCALAPPDATA', tempfile.gettempdir())) / 'PhotoLab' / 'nikon_runtime' / key
        cache.mkdir(parents=True, exist_ok=True)
        for source in sources:
            target = cache / source.name
            if not target.is_file() or target.stat().st_size != source.stat().st_size:
                temporary = target.with_suffix(target.suffix + '.tmp')
                shutil.copyfile(source, temporary)
                os.replace(temporary, target)
        with tempfile.TemporaryDirectory(prefix='photolab_nikon_') as temporary:
            output = Path(temporary) / 'image.rgb'
            result = subprocess.run([str(cache / helper.name), str(Path(path).resolve()), str(output),
                                     '16' if output_bps == 16 else '8', str(profiles)],
                                    cwd=cache, capture_output=True, timeout=120,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode:
                detail = result.stderr.decode('utf-8', errors='replace').strip()
                raise RuntimeError(detail or f'Nikon decoder exited with {result.returncode}')
            return _read_output(output)
