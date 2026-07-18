"""Tests for the persistent demucs worker (#309) and the GPU->CPU separation
fallback (#276).

The worker invocation is swapped for stub Python scripts via the
_spawn_worker_cmd seam, so the real process machinery (Popen, stdin
dispatch, stderr streaming, watchdog, cancel translation, worker reuse) runs
end-to-end without demucs or a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.core.models import Job, JobCancelled
from app.pipeline import separate as sep_mod
from app.pipeline.errors import SeparationError

# A persistent worker stub: reads one JSON job request per line for as long
# as stdin stays open, always succeeding (writes a stem WAV where the real
# worker would, then "100%" + "@@DONE@@" to stderr) and keeps serving.
_SUCCESS_WORKER = """
import sys, json, os
for line in sys.stdin:
    req = json.loads(line)
    d = os.path.join(req["job_dir"], "htdemucs_6s", "source")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "vocals.wav"), "wb").write(b"RIFF")
    sys.stderr.write("100%\\n@@DONE@@\\n")
    sys.stderr.flush()
"""

# A worker stub that fails its first (only) dispatched job with a CUDA-OOM
# -shaped message, then exits -- matching demucs_worker.py's real behavior of
# never trying to keep serving after a failure.
_FAILING_WORKER = """
import sys, json
sys.stdin.readline()
sys.stderr.write("@@ERROR@@" + json.dumps("CUDA out of memory. Tried 2 GiB") + "\\n")
sys.stderr.flush()
sys.exit(1)
"""


def _stub_spawns(fail_devices: set[str], calls: list[str]):
    """A _spawn_worker_cmd replacement. `calls` records one entry per SPAWNED
    worker process (not per dispatched job) -- reuse across jobs on the same
    device means fewer calls than jobs, which the reuse tests assert on."""

    def fake_spawn(device: str, model: str) -> list[str]:
        calls.append(device)
        code = _FAILING_WORKER if device in fail_devices else _SUCCESS_WORKER
        return [sys.executable, "-c", code]

    return fake_spawn


@pytest.fixture()
def job(tmp_path: Path):
    j = Job(id="abcdefabc276")
    (tmp_path / "source.wav").write_bytes(b"RIFF")
    return j


@pytest.fixture(autouse=True)
def _reset_worker():
    """Each test starts and ends with no lingering worker reference -- a
    prior test's stub process must never leak into the next test."""
    sep_mod._worker.clear()
    yield
    sep_mod._kill_worker()


def test_gpu_failure_falls_back_to_cpu(job, tmp_path, monkeypatch, caplog):
    import logging

    calls: list[str] = []
    monkeypatch.setattr(sep_mod, "get_demucs_device", lambda: "cuda")
    monkeypatch.setattr(sep_mod, "_spawn_worker_cmd", _stub_spawns({"cuda"}, calls))

    with caplog.at_level(logging.WARNING, logger="stemdeck.pipeline"):
        stems_root = sep_mod.separate(job, tmp_path / "source.wav", tmp_path)

    assert calls == ["cuda", "cpu"]
    assert (stems_root / "vocals.wav").is_file()
    assert job.gpu_fallback is True
    assert job.compute_device == "cpu (fallback from cuda)"
    # Loud, never silent: the warning names device, cause, and stderr.
    warning = next(r.message for r in caplog.records if "retrying on CPU" in r.message)
    assert "cause=out-of-memory" in warning
    assert "CUDA out of memory" in warning


def test_dispatch_omits_extra_shifts_at_standard_quality(monkeypatch, tmp_path):
    monkeypatch.setattr(sep_mod, "get_separation_quality", lambda: "standard")
    # Drive a real job and have the stub echo the dispatched request's
    # "shifts" value back via a marker file -- simplest way to inspect what
    # separate() actually sent without patching json.dumps at the call site.
    echo_worker = """
import sys, json, os
for line in sys.stdin:
    req = json.loads(line)
    d = os.path.join(req["job_dir"], "htdemucs_6s", "source")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "vocals.wav"), "wb").write(b"RIFF")
    open(os.path.join(req["job_dir"], "shifts.txt"), "w").write(str(req["shifts"]))
    sys.stderr.write("100%\\n@@DONE@@\\n")
    sys.stderr.flush()
"""
    monkeypatch.setattr(sep_mod, "get_demucs_device", lambda: "cpu")
    monkeypatch.setattr(
        sep_mod, "_spawn_worker_cmd", lambda device, model: [sys.executable, "-c", echo_worker]
    )

    sep_mod.separate(Job(id="abcdefabc277"), tmp_path / "source.wav", tmp_path)

    assert (tmp_path / "shifts.txt").read_text() == "1"


def test_dispatch_includes_shifts_2_at_best_quality(monkeypatch, tmp_path):
    echo_worker = """
import sys, json, os
for line in sys.stdin:
    req = json.loads(line)
    d = os.path.join(req["job_dir"], "htdemucs_6s", "source")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "vocals.wav"), "wb").write(b"RIFF")
    open(os.path.join(req["job_dir"], "shifts.txt"), "w").write(str(req["shifts"]))
    sys.stderr.write("100%\\n@@DONE@@\\n")
    sys.stderr.flush()
"""
    monkeypatch.setattr(sep_mod, "get_separation_quality", lambda: "best")
    monkeypatch.setattr(sep_mod, "get_demucs_device", lambda: "cpu")
    monkeypatch.setattr(
        sep_mod, "_spawn_worker_cmd", lambda device, model: [sys.executable, "-c", echo_worker]
    )

    sep_mod.separate(Job(id="abcdefabc278"), tmp_path / "source.wav", tmp_path)

    assert (tmp_path / "shifts.txt").read_text() == "2"


def test_records_startup_timing_on_first_progress_line(job, tmp_path, monkeypatch):
    """#288/#309: time from dispatch to the first progress line demucs
    emits -- the full spawn/model-load cost for a fresh worker, near-zero
    for a reused warm one."""
    calls: list[str] = []
    monkeypatch.setattr(sep_mod, "get_demucs_device", lambda: "cpu")
    monkeypatch.setattr(sep_mod, "_spawn_worker_cmd", _stub_spawns(set(), calls))

    sep_mod.separate(job, tmp_path / "source.wav", tmp_path)

    assert job.stage_timings is not None
    assert "separate_startup" in job.stage_timings
    assert job.stage_timings["separate_startup"] >= 0.0


def test_gpu_success_needs_no_fallback(job, tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(sep_mod, "get_demucs_device", lambda: "cuda")
    monkeypatch.setattr(sep_mod, "_spawn_worker_cmd", _stub_spawns(set(), calls))

    stems_root = sep_mod.separate(job, tmp_path / "source.wav", tmp_path)

    assert calls == ["cuda"]
    assert stems_root.is_dir()
    assert job.gpu_fallback is False
    assert job.compute_device == "cuda"


def test_cpu_failure_does_not_retry(job, tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(sep_mod, "get_demucs_device", lambda: "cpu")
    monkeypatch.setattr(sep_mod, "_spawn_worker_cmd", _stub_spawns({"cpu"}, calls))

    with pytest.raises(SeparationError) as exc_info:
        sep_mod.separate(job, tmp_path / "source.wav", tmp_path)

    assert calls == ["cpu"]  # exactly one attempt
    assert job.gpu_fallback is False
    assert exc_info.value.device == "cpu"


def test_both_attempts_failing_raises_with_both_tails(job, tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(sep_mod, "get_demucs_device", lambda: "mps")
    monkeypatch.setattr(sep_mod, "_spawn_worker_cmd", _stub_spawns({"mps", "cpu"}, calls))

    with pytest.raises(SeparationError) as exc_info:
        sep_mod.separate(job, tmp_path / "source.wav", tmp_path)

    assert calls == ["mps", "cpu"]
    err = exc_info.value
    assert err.device == "mps, then cpu"
    # The quarantine's error.txt gets both attempts' evidence.
    joined = "\n".join(err.tail)
    assert "--- attempt on mps ---" in joined
    assert "--- cpu fallback attempt ---" in joined


def test_cancel_during_gpu_attempt_skips_fallback(job, tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(sep_mod, "get_demucs_device", lambda: "cuda")
    monkeypatch.setattr(sep_mod, "_spawn_worker_cmd", _stub_spawns({"cuda"}, calls))
    job.cancel_requested = True  # POST /cancel arrived before/mid attempt

    with pytest.raises(JobCancelled):
        sep_mod.separate(job, tmp_path / "source.wav", tmp_path)

    assert calls == ["cuda"]  # no CPU retry after a cancel


def test_partial_gpu_output_cleared_before_retry(job, tmp_path, monkeypatch):
    """A failed GPU attempt's partial stems must not leak into the CPU run."""
    calls: list[str] = []
    marker = tmp_path / "htdemucs_6s" / "partial-garbage.wav"
    marker_repr = str(marker).replace("\\", "\\\\")

    def fake_spawn(device: str, model: str) -> list[str]:
        calls.append(device)
        if device == "cuda":
            # Simulate the worker dying after writing partial output.
            code = (
                "import os, sys, json\n"
                "sys.stdin.readline()\n"
                f"os.makedirs(os.path.dirname('{marker_repr}'), exist_ok=True)\n"
                f"open('{marker_repr}', 'wb').write(b'junk')\n"
                "sys.stderr.write('@@ERROR@@' + json.dumps('CUDA error') + chr(10))\n"
                "sys.stderr.flush()\n"
                "sys.exit(1)\n"
            )
        else:
            code = _SUCCESS_WORKER
        return [sys.executable, "-c", code]

    monkeypatch.setattr(sep_mod, "get_demucs_device", lambda: "cuda")
    monkeypatch.setattr(sep_mod, "_spawn_worker_cmd", fake_spawn)

    sep_mod.separate(job, tmp_path / "source.wav", tmp_path)

    assert calls == ["cuda", "cpu"]
    assert not marker.exists(), "partial GPU output must be cleared before the CPU retry"


# ─── #309: worker reuse ────────────────────────────────────────────────────


def test_worker_reused_across_consecutive_jobs_on_same_device(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(sep_mod, "get_demucs_device", lambda: "cpu")
    monkeypatch.setattr(sep_mod, "_spawn_worker_cmd", _stub_spawns(set(), calls))

    for i in range(3):
        job = Job(id=f"abcdefabc30{i}")
        (tmp_path / "source.wav").write_bytes(b"RIFF")
        sep_mod.separate(job, tmp_path / "source.wav", tmp_path)

    assert calls == ["cpu"]  # one spawn serving all three jobs


def test_worker_respawned_on_device_change(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(sep_mod, "_spawn_worker_cmd", _stub_spawns(set(), calls))

    devices = iter(["cpu", "cuda"])
    monkeypatch.setattr(sep_mod, "get_demucs_device", lambda: next(devices))

    sep_mod.separate(Job(id="abcdefabc310"), tmp_path / "source.wav", tmp_path)
    sep_mod.separate(Job(id="abcdefabc311"), tmp_path / "source.wav", tmp_path)

    assert calls == ["cpu", "cuda"]


def test_worker_not_reused_after_failure(tmp_path, monkeypatch):
    """A failed job's worker is never handed the next job -- GPU/CUDA state
    afterward isn't something we can vouch for (see demucs_worker.py)."""
    calls: list[str] = []
    monkeypatch.setattr(sep_mod, "get_demucs_device", lambda: "cpu")
    monkeypatch.setattr(sep_mod, "_spawn_worker_cmd", _stub_spawns({"cpu"}, calls))

    with pytest.raises(SeparationError):
        sep_mod.separate(Job(id="abcdefabc320"), tmp_path / "source.wav", tmp_path)
    # The next job on the same device spawns a fresh worker rather than
    # reusing the one that just failed.
    monkeypatch.setattr(sep_mod, "_spawn_worker_cmd", _stub_spawns(set(), calls))
    sep_mod.separate(Job(id="abcdefabc321"), tmp_path / "source.wav", tmp_path)

    assert calls == ["cpu", "cpu"]  # two spawns: the failure, then the retry


def test_cancel_kills_worker_next_job_spawns_fresh(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(sep_mod, "get_demucs_device", lambda: "cpu")
    monkeypatch.setattr(sep_mod, "_spawn_worker_cmd", _stub_spawns({"cpu"}, calls))
    cancelled_job = Job(id="abcdefabc330")
    cancelled_job.cancel_requested = True

    with pytest.raises(JobCancelled):
        sep_mod.separate(cancelled_job, tmp_path / "source.wav", tmp_path)

    monkeypatch.setattr(sep_mod, "_spawn_worker_cmd", _stub_spawns(set(), calls))
    sep_mod.separate(Job(id="abcdefabc331"), tmp_path / "source.wav", tmp_path)

    assert calls == ["cpu", "cpu"]  # two spawns: cancelled, then fresh


# ─── separation model selection ────────────────────────────────────────────


def test_worker_respawned_on_model_change(tmp_path, monkeypatch):
    """Switching the separation model tears down the current worker and spawns
    the other backend, exactly like a device change -- the reuse cache is keyed
    on (model, device), not device alone."""
    spawned: list[tuple[str, str]] = []
    monkeypatch.setattr(sep_mod, "get_demucs_device", lambda: "cpu")

    def fake_spawn(device: str, model: str) -> list[str]:
        spawned.append((device, model))
        # Write stems into the subdir the router expects for this model, so
        # collect()'s lookup succeeds for both backends.
        subdir = "bs_roformer_sw" if model == "bs_roformer_sw" else "htdemucs_6s"
        worker = f"""
import sys, json, os
for line in sys.stdin:
    req = json.loads(line)
    d = os.path.join(req["job_dir"], "{subdir}", "source")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "vocals.wav"), "wb").write(b"RIFF")
    sys.stderr.write("100%\\n@@DONE@@\\n")
    sys.stderr.flush()
"""
        return [sys.executable, "-c", worker]

    monkeypatch.setattr(sep_mod, "_spawn_worker_cmd", fake_spawn)

    models = iter(["htdemucs_6s", "bs_roformer_sw", "bs_roformer_sw"])
    monkeypatch.setattr(sep_mod, "get_separation_model", lambda: next(models))

    for i in range(3):
        (tmp_path / "source.wav").write_bytes(b"RIFF")
        sep_mod.separate(Job(id=f"abcdefabc34{i}"), tmp_path / "source.wav", tmp_path)

    # demucs then roformer (respawn on change), then roformer reused (no respawn).
    assert spawned == [("cpu", "htdemucs_6s"), ("cpu", "bs_roformer_sw")]


def test_roformer_dispatch_selects_worker_and_omits_shifts(tmp_path, monkeypatch):
    """The bs_roformer_sw model routes to the roformer worker module and sends
    no `shifts` key (a demucs-only knob). The stub echoes what it received."""
    spawned_argv: list[list[str]] = []
    echo_worker = """
import sys, json, os
for line in sys.stdin:
    req = json.loads(line)
    d = os.path.join(req["job_dir"], "bs_roformer_sw", "source")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "vocals.wav"), "wb").write(b"RIFF")
    open(os.path.join(req["job_dir"], "req.json"), "w").write(json.dumps(req))
    sys.stderr.write("100%\\n@@DONE@@\\n")
    sys.stderr.flush()
"""

    # Capture the router's real module choice before swapping in the echo stub.
    real_spawn = sep_mod._spawn_worker_cmd

    def fake_spawn(device: str, model: str) -> list[str]:
        spawned_argv.append(real_spawn(device, model))
        return [sys.executable, "-c", echo_worker]

    monkeypatch.setattr(sep_mod, "get_demucs_device", lambda: "cpu")
    monkeypatch.setattr(sep_mod, "get_separation_model", lambda: "bs_roformer_sw")
    monkeypatch.setattr(sep_mod, "_spawn_worker_cmd", fake_spawn)

    stems_root = sep_mod.separate(Job(id="abcdefabc350"), tmp_path / "source.wav", tmp_path)

    # Routed to the roformer worker module, not demucs.
    assert any("roformer_worker" in " ".join(argv) for argv in spawned_argv)
    # Stems landed under the roformer subdir.
    assert stems_root == tmp_path / "bs_roformer_sw" / "source"
    assert (stems_root / "vocals.wav").is_file()
    # No demucs-only shifts key was dispatched to the roformer worker.
    import json as _json

    req = _json.loads((tmp_path / "req.json").read_text())
    assert "shifts" not in req


def test_2stem_vocal_model_routes_and_collects_two_stems(tmp_path, monkeypatch):
    """The kim_ft_vocal model spawns the roformer worker with its model id,
    writes into its own subdir, and collect() returns exactly vocals+other."""
    from app.pipeline.collect import collect

    # Stub worker: writes ONLY vocals.wav + other.wav into the kim_ft_vocal
    # subdir (what the real 2-stem model produces), then succeeds.
    worker = """
import sys, json, os
for line in sys.stdin:
    req = json.loads(line)
    d = os.path.join(req["job_dir"], "kim_ft_vocal", "source")
    os.makedirs(d, exist_ok=True)
    for name in ("vocals", "other"):
        open(os.path.join(d, name + ".wav"), "wb").write(b"RIFF")
    sys.stderr.write("100%\\n@@DONE@@\\n")
    sys.stderr.flush()
"""
    spawned_argv = []
    real_spawn = sep_mod._spawn_worker_cmd

    def fake_spawn(device, model):
        spawned_argv.append(real_spawn(device, model))
        return [sys.executable, "-c", worker]

    monkeypatch.setattr(sep_mod, "get_demucs_device", lambda: "cpu")
    monkeypatch.setattr(sep_mod, "get_separation_model", lambda: "kim_ft_vocal")
    monkeypatch.setattr(sep_mod, "_spawn_worker_cmd", fake_spawn)

    (tmp_path / "source.wav").write_bytes(b"RIFF")
    job = Job(id="abcdefkimft01")
    stems_root = sep_mod.separate(job, tmp_path / "source.wav", tmp_path)

    # Routed to the roformer worker with the model id in argv.
    assert any("roformer_worker" in " ".join(a) and "kim_ft_vocal" in a for a in spawned_argv)
    assert job.separation_model == "kim_ft_vocal"
    assert stems_root == tmp_path / "kim_ft_vocal" / "source"

    found = collect(job, stems_root, tmp_path)
    assert set(found) == {"vocals", "other"}  # exactly 2 stems, not 6
    assert (tmp_path / "stems" / "vocals.wav").is_file()
    assert (tmp_path / "stems" / "other.wav").is_file()
