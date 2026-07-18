import os
import re
import sys
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else default


def available_torch_devices() -> list[str]:
    """Compute devices this machine can actually use, best-first. CPU is always
    present; cuda/mps depend on the hardware + installed torch build. The
    Settings UI uses this to disable options that aren't available/detected so
    a user can't pick an impossible device."""
    devices: list[str] = []
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps")
    except ImportError:
        pass
    devices.append("cpu")
    return devices


def detect_torch_device() -> str:
    """Best available Torch device for Demucs by hardware probe: cuda > mps >
    cpu. Apple Silicon needs the explicit MPS check -- demucs's CLI default is
    "cuda if available else cpu" and macOS has no CUDA, leaving the integrated
    GPU idle and processing 3-5x slower than necessary.

    User-facing device selection lives in app.core.settings (demucs_device,
    default "auto" -> this probe); the STEMDECK_DEMUCS_DEVICE env var seeds
    that setting's default so env-based deployments keep working."""
    return available_torch_devices()[0]


ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = ROOT / "static"
STEM_NAMES: tuple[str, ...] = ("vocals", "drums", "bass", "guitar", "piano", "other")
JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# Runtime knobs -- env-backed so Docker / desktop packaging / local dev can
# tune without a code edit. STEMDECK_DATA_DIR is the portable app root for
# mutable runtime data; when unset, dev behavior remains the repo-local jobs/
# folder.
PORTABLE_DATA_DIR_ENABLED = bool(os.environ.get("STEMDECK_DATA_DIR", "").strip())
DATA_DIR = _env_path("STEMDECK_DATA_DIR", ROOT)
JOBS_DIR = _env_path(
    "STEMDECK_JOBS_DIR",
    (DATA_DIR / "jobs") if PORTABLE_DATA_DIR_ENABLED else (ROOT / "jobs"),
)
CACHE_DIR = _env_path("STEMDECK_CACHE_DIR", DATA_DIR / "cache")
DOWNLOADS_DIR = _env_path("STEMDECK_DOWNLOADS_DIR", DATA_DIR / "downloads")
MODELS_DIR = _env_path("STEMDECK_MODELS_DIR", DATA_DIR / "models")
LOGS_DIR = _env_path("STEMDECK_LOGS_DIR", DATA_DIR / "logs")
FFMPEG_DIR = _env_path("STEMDECK_FFMPEG_DIR", DATA_DIR / "ffmpeg")
FFMPEG_BIN = _env_path(
    "STEMDECK_FFMPEG",
    FFMPEG_DIR / ("ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"),
)
FFPROBE_BIN = _env_path(
    "STEMDECK_FFPROBE",
    FFMPEG_DIR / ("ffprobe.exe" if sys.platform.startswith("win") else "ffprobe"),
)
DEMUCS_MODEL = os.environ.get("STEMDECK_DEMUCS_MODEL", "htdemucs_6s").strip() or "htdemucs_6s"

# ── separation model registry ──
# Every selectable separator, keyed by its setting id (see settings.py's
# separation_model). Each entry declares:
#   backend    -- "demucs" (app/pipeline/demucs_worker) or "roformer"
#                 (app/pipeline/roformer_worker, needs the [roformer] extra)
#   checkpoint -- model name/file the backend loads (demucs bag name, or the
#                 audio-separator catalog filename for roformer)
#   stems      -- the stem names this model produces, in canonical order. NOT
#                 every model makes all 6: vocal models make just vocals+other.
#                 collect()/the mixer/the API derive the per-job set from this.
#   subdir     -- per-job stems directory name (job_dir/<subdir>/<source stem>)
#   label      -- human label for the Settings dropdown
# Stem names must stay within STEM_NAMES (the canonical superset) so the API
# validation and colour/label maps keep working -- a model emitting a genuinely
# new stem name would also need those widened.
SEPARATION_MODELS: dict[str, dict] = {
    "htdemucs_6s": {
        "backend": "demucs",
        "checkpoint": DEMUCS_MODEL,
        "stems": STEM_NAMES,
        "subdir": "htdemucs_6s",
        "label": "Demucs (6 stems)",
        "description": (
            "Meta's Demucs htdemucs_6s. Splits into 6 stems (vocals, drums, "
            "bass, guitar, piano, other). The default -- fast and reliable, "
            "no extra download. Piano/guitar are usable but the weakest stems."
        ),
    },
    "bs_roformer_sw": {
        "backend": "roformer",
        "checkpoint": "BS-Roformer-SW.ckpt",
        "stems": STEM_NAMES,
        "subdir": "bs_roformer_sw",
        "label": "BS-Roformer (6 stems, higher quality)",
        "description": (
            "6-stem BS-Roformer. Same stems as Demucs but noticeably cleaner "
            "separation, especially vocals and a more usable piano. Slower, and "
            "downloads a ~700 MB model on first use. Best all-round choice."
        ),
    },
    "kim_ft_vocal": {
        "backend": "roformer",
        "checkpoint": "mel_band_roformer_kim_ft_unwa.ckpt",
        "stems": ("vocals", "other"),  # "other" == the full instrumental
        "subdir": "kim_ft_vocal",
        "label": "Vocal Roformer (vocals only, highest quality)",
        "description": (
            "Mel-Band Roformer (Kim FT). Produces just 2 stems -- vocals and a "
            "full instrumental. The cleanest vocal isolation available; ideal "
            "when you only need the vocal or a karaoke/instrumental track. "
            "Downloads a ~900 MB model on first use."
        ),
    },
}

DEFAULT_SEPARATION_MODEL = "htdemucs_6s"


def _model_entry(separation_model: str) -> dict:
    """Registry entry for a model id, falling back to the default so callers
    never KeyError on a stale/unknown persisted value."""
    return SEPARATION_MODELS.get(separation_model, SEPARATION_MODELS[DEFAULT_SEPARATION_MODEL])


def model_backend(separation_model: str) -> str:
    return _model_entry(separation_model)["backend"]


def model_checkpoint(separation_model: str) -> str:
    return _model_entry(separation_model)["checkpoint"]


def model_stems(separation_model: str) -> tuple[str, ...]:
    """The stem names a model produces (e.g. all 6, or just vocals+other)."""
    return tuple(_model_entry(separation_model)["stems"])


def model_subdir(separation_model: str) -> str:
    """The job-dir subfolder a backend writes its <stem>.wav files into:
    `job_dir / <this> / <source stem>`. Keeps separate()/collect() from
    hardcoding a single model's dir so multiple backends can coexist."""
    return _model_entry(separation_model)["subdir"]


MAX_DURATION_SEC = max(60, _env_int("STEMDECK_MAX_DURATION_SEC", 1200))  # 20 min default
JOB_TTL_SECONDS = max(300, _env_int("STEMDECK_JOB_TTL_SECONDS", 24 * 3600))  # 24 h default
# TTL for quarantined failed-job dirs (jobs/failed/<id>, kept for diagnostics).
# Swept unconditionally -- even deployments with a persistent library must not
# accumulate failure evidence forever.
FAILED_TTL_SECONDS = max(3600, _env_int("STEMDECK_FAILED_TTL_SECONDS", 7 * 24 * 3600))  # 7 d
MAX_PENDING_JOBS = max(1, min(50, _env_int("STEMDECK_MAX_PENDING_JOBS", 3)))
TIMEOUT_FFMPEG = _env_int("STEMDECK_TIMEOUT_FFMPEG", 300)
TIMEOUT_ANALYZE = _env_int("STEMDECK_TIMEOUT_ANALYZE", 120)
TIMEOUT_DEMUCS_STALL = _env_int("STEMDECK_TIMEOUT_DEMUCS_STALL", 1800)
# Max height for the MP4 video stream pulled from YouTube (issue #219).
# Capped to keep downloads reasonable; 1080p of a full song is large.
VIDEO_MAX_HEIGHT = max(144, _env_int("STEMDECK_VIDEO_MAX_HEIGHT", 720))


def ffmpeg_executable() -> str:
    """Return the preferred FFmpeg executable.

    In portable mode, setup places FFmpeg under DATA_DIR/ffmpeg. Prefer that
    binary when present; otherwise fall back to PATH so local dev and Docker
    keep working exactly as before.
    """
    return str(FFMPEG_BIN) if FFMPEG_BIN.is_file() else "ffmpeg"


def ffprobe_executable() -> str:
    """Return the preferred ffprobe executable (same bundled dir as ffmpeg)."""
    return str(FFPROBE_BIN) if FFPROBE_BIN.is_file() else "ffprobe"


def configure_portable_environment() -> None:
    """Keep generated caches inside the portable data folder when requested.

    This is intentionally best-effort. It only sets variables that are still
    unset, so explicit caller/env choices win.
    """
    if FFMPEG_DIR.is_dir():
        path = os.environ.get("PATH", "")
        ffmpeg_path = str(FFMPEG_DIR)
        if ffmpeg_path not in path.split(os.pathsep):
            os.environ["PATH"] = ffmpeg_path + (os.pathsep + path if path else "")

    if PORTABLE_DATA_DIR_ENABLED:
        os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
        os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))


def ensure_runtime_dirs() -> None:
    paths = (
        (JOBS_DIR, CACHE_DIR, DOWNLOADS_DIR, MODELS_DIR, LOGS_DIR)
        if PORTABLE_DATA_DIR_ENABLED
        else (JOBS_DIR,)
    )
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
