"""Persistent Roformer separation worker (the audio-separator backend).

Run as its own process: `python -m app.pipeline.roformer_worker <device> <model_id>`.

Mirrors demucs_worker.py's contract exactly -- one JSON request per line on
stdin, tqdm `NN%` progress and a terminating `@@DONE@@` / `@@ERROR@@` line on
stderr -- so app/pipeline/separate.py drives both backends through a single
code path and neither needs to know which one it is talking to.

Persistent, unlike vocal_split_worker.py, for the reason given in #309: this is
the hot path every job takes, so the multi-second checkpoint load is worth
amortizing across consecutive jobs on the same (model, device). The vocal split
is an occasional user-triggered action, where it is not.

Which stems appear depends on the model: the 6-stem BS-Roformer produces all of
STEM_NAMES, a vocal model produces vocals + other (the whole instrumental). The
worker writes exactly the stems the registry declares for its model id, and
collect() picks up whatever landed.

The Roformer checkpoints are torch models: audio-separator runs them on the
same torch device the Demucs worker uses (CUDA, MPS, or CPU), so no extra
package is needed for GPU acceleration. onnxruntime is only involved for the
ONNX vocal-split model, never here.

The `audio_separator` import is deliberately inside the function rather than at
module scope. The package is platform-gated in pyproject (absent on Intel
macOS), and an install without it -- which can never select this backend, but
does import the package tree -- must not fail on it. A missing package becomes
a clean @@ERROR@@ line, not a traceback.

Protocol:
  - Parent writes one JSON line per job: {"source": "<path>", "job_dir": "<path>"}
    (no "shifts": that is a demucs flag, see separate.py)
  - Progress streams to stderr as tqdm `NN%` lines, which separate.py's _PCT_RE
    already parses -- no special handling needed here
  - Then exactly one of:
      "@@DONE@@"                        -- job ok, the worker keeps serving
      "@@ERROR@@<json-encoded message>" -- job failed, the worker exits(1)
  - EOF on stdin (the parent closed the pipe) ends the loop cleanly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from app.core.config import MODELS_DIR, model_checkpoint, model_stems, model_subdir
from app.core.process import arm_parent_watchdog


def _run_one_job(separator, subdir: str, output_names: dict[str, str], req: dict) -> None:
    source = Path(req["source"])
    job_dir = Path(req["job_dir"])

    # Match the demucs contract: stems land in job_dir/<subdir>/<source stem>/,
    # which is what separate.py returns and collect() then drains.
    out_dir = job_dir / subdir / source.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # The output directory that actually governs where files are written is the
    # one captured by the loaded model instance at load_model() time. Setting it
    # only on the reused top-level Separator leaves stems in the process CWD, so
    # both are updated for every job.
    separator.output_dir = str(out_dir)
    if getattr(separator, "model_instance", None) is not None:
        separator.model_instance.output_dir = str(out_dir)

    separator.separate(str(source), output_names)


def main() -> None:
    # Without this a Force-Quit of the app orphans this process holding the GPU:
    # nothing is read from stdin mid-inference, so EOF never arrives (#519).
    arm_parent_watchdog()

    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    model_id = sys.argv[2] if len(sys.argv) > 2 else "bs_roformer_sw"

    checkpoint = model_checkpoint(model_id)
    subdir = model_subdir(model_id)
    # audio-separator writes each stem to "<mapped name>.wav", so mapping every
    # stem to its own name yields the canonical filenames collect() looks for
    # and avoids depending on the library's default "<base>_(<stem>)_<model>"
    # format. Its keys are the model's own capitalised stem labels.
    output_names = {name.capitalize(): name for name in model_stems(model_id)}

    if device == "cpu":
        # Same policy as vocal_split_worker: hiding the CUDA devices is the
        # reliable way to force the CPU provider, and it must happen before
        # torch/onnxruntime are imported below. This is what makes separate.py's
        # GPU-failure CPU retry work for this backend too.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    try:
        from audio_separator.separator import Separator
    except Exception as e:  # ImportError, or a broken onnxruntime/torch link
        sys.stderr.write(
            "@@ERROR@@"
            + json.dumps(
                "The Roformer models need the audio-separator package, which is "
                f"not available on this platform. The import failed with: {e}"
            )
            + "\n"
        )
        sys.stderr.flush()
        sys.exit(1)

    # Cached under the app's models dir rather than a temp dir, so a ~700 MB
    # checkpoint survives a reboot and is covered by the app's data lifecycle.
    separator = Separator(
        log_level=40,  # logging.ERROR -- only the lines we emit ourselves matter
        model_file_dir=str(MODELS_DIR / "audio-separator"),
        output_format="WAV",
    )
    separator.load_model(model_filename=checkpoint)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            _run_one_job(separator, subdir, output_names, json.loads(line))
        except Exception as e:
            sys.stderr.write(f"@@ERROR@@{json.dumps(str(e))}\n")
            sys.stderr.flush()
            sys.exit(1)
        sys.stderr.write("@@DONE@@\n")
        sys.stderr.flush()


if __name__ == "__main__":
    main()
