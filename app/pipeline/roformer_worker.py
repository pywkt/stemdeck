"""Persistent Roformer worker (optional [roformer] backend).

Run as its own process: `python -m app.pipeline.roformer_worker <device> <model_id>`.
Mirrors demucs_worker.py exactly -- same stdin-JSON request / stderr
`@@DONE@@`/`@@ERROR@@`/tqdm-`NN%` protocol -- so separate.py drives both
backends through one code path. Loads the model named by `<model_id>` (looked
up in the SEPARATION_MODELS registry) once via the `audio-separator` package,
then serves jobs one at a time, keeping the model resident across consecutive
successful jobs on the same (model, device).

Different Roformer models produce different stem sets: the 6-stem BS-Roformer-SW
makes all of STEM_NAMES, while vocal models make just vocals+other. The worker
writes exactly the model's declared stems (registry `stems`) -- collect() then
picks up whatever landed.

The `audio-separator` import is deliberately kept at function scope, never at
module top level: it lives behind the optional [roformer] dependency extra, so
a base install (which never selects this backend) must be able to import the
app without it present. If the extra is missing, the import raises and we emit
`@@ERROR@@` so the parent surfaces a clean failure instead of crashing.

audio-separator emits standard tqdm `NN%` progress lines on stderr -- the same
format separate.py's _PCT_RE already parses -- so progress reporting needs no
special handling here.

Protocol (identical to demucs_worker):
  - Parent writes one JSON line to stdin per job:
      {"source": "<path>", "job_dir": "<path>"}
    (No "shifts": the demucs shift-averaging knob does not apply to Roformer;
    separation_quality only affects the demucs path. See settings.py.)
  - Progress streams to stderr as tqdm `NN%` lines during inference.
  - On completion the worker writes one more stderr line:
      "@@DONE@@"                          -- job ok, worker keeps serving
      "@@ERROR@@<json-encoded message>"   -- job failed, worker exits(1)
  - EOF on stdin (parent closed the pipe) ends the worker's loop cleanly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from app.core.config import MODELS_DIR, model_checkpoint, model_stems, model_subdir


def _run_one_job(sep, subdir: str, custom_output_names: dict, req: dict) -> None:
    source = Path(req["source"])
    job_dir = Path(req["job_dir"])

    # Match the demucs contract: stems land in job_dir/<subdir>/<source stem>/.
    out_dir = job_dir / subdir / source.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # The output_dir that actually governs where stems are written lives on the
    # loaded model separator (sep.model_instance), captured from config at
    # load_model() time -- setting it only on the top-level Separator (which we
    # reuse across jobs) leaves stems in the process CWD. Update both per job.
    sep.output_dir = str(out_dir)
    if getattr(sep, "model_instance", None) is not None:
        sep.model_instance.output_dir = str(out_dir)
    # custom_output_names maps each of the model's stems to "<name>.wav" so the
    # files land with canonical names (vocals.wav, other.wav, ...) that
    # collect.py picks up directly -- no post-separation renaming of
    # audio-separator's default "<base>_(<stem>)_<model>.wav" names.
    sep.separate(str(source), custom_output_names=custom_output_names)


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    model_id = sys.argv[2] if len(sys.argv) > 2 else "bs_roformer_sw"

    # Resolve the model's checkpoint, stems, and output subdir from the registry.
    checkpoint = model_checkpoint(model_id)
    subdir = model_subdir(model_id)
    # audio-separator's custom_output_names writes each stem to "<value>.wav";
    # an identity map over the model's declared stems yields canonical filenames.
    custom_output_names = {name: name for name in model_stems(model_id)}

    # audio-separator auto-selects the Torch device; the only case we must force
    # is the CPU fallback (separate.py retries a failed GPU attempt on "cpu").
    # Hiding CUDA from this process before torch is imported guarantees CPU,
    # which keeps the demucs GPU->CPU fallback contract working for Roformer too.
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        from audio_separator.separator import Separator
    except Exception as e:  # ImportError, or a broken onnxruntime/torch link
        sys.stderr.write(
            "@@ERROR@@"
            + json.dumps(
                "The Roformer backend requires the optional 'roformer' dependency "
                f"extra (uv sync --extra roformer). Import failed: {e}"
            )
            + "\n"
        )
        sys.stderr.flush()
        sys.exit(1)

    # Cache the checkpoint under the app's models dir (not /tmp), so it survives
    # reboots and is managed by the app's data lifecycle. audio-separator honors
    # the Torch device automatically; the CPU env override above keeps the
    # CPU-fallback path (separate.py) working the same as demucs.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sep = Separator(model_file_dir=str(MODELS_DIR), output_format="WAV")
    sep.load_model(model_filename=checkpoint)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            _run_one_job(sep, subdir, custom_output_names, req)
        except Exception as e:
            sys.stderr.write(f"@@ERROR@@{json.dumps(str(e))}\n")
            sys.stderr.flush()
            sys.exit(1)
        sys.stderr.write("@@DONE@@\n")
        sys.stderr.flush()


if __name__ == "__main__":
    main()
