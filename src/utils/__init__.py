from .ema import ModelEMA

# Optional imports (only if files exist)
try:
    from .transcribe import transcribe_audio_files
    _has_transcribe = True
except ImportError:
    _has_transcribe = False

try:
    from .data_manifest import create_csv
    _has_manifest = True
except ImportError:
    _has_manifest = False

__all__ = ["ModelEMA"]

if _has_transcribe:
    __all__.append("transcribe_audio_files")
if _has_manifest:
    __all__.append("create_csv")
