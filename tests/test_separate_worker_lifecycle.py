"""The persistent demucs worker must not survive a failure, and cancel must
land promptly (#514).

ml-pipeline.md: the worker "is reused across consecutive **successful** jobs on
the same device, but torn down after **any** non-success (cancellation or
failure) -- post-exception CUDA state can't be trusted. Both halves of this
rule matter; don't relax either side."
"""

from __future__ import annotations

import pytest

from app.core.models import Job, JobCancelled
from app.pipeline import separate as _separate


@pytest.fixture(autouse=True)
def _no_worker():
    _separate._worker.clear()
    yield
    _separate._worker.clear()


def _job(**kw):
    return Job(id="a1b2c3d4e5f6", **kw)


def test_an_exception_in_the_stream_loop_still_tears_the_worker_down(tmp_path, monkeypatch):
    # The teardown used to sit after the finally, so anything raising out of
    # the read loop left a worker whose CUDA state followed an exception warm
    # for the next job.
    killed = []
    monkeypatch.setattr(_separate, "_kill_worker", lambda: killed.append(True))

    class _Boom:
        stdin = None
        stderr = None

    monkeypatch.setattr(_separate, "_get_worker", lambda device, model="htdemucs_6s": _Boom())

    job = _job(status="separating")
    with pytest.raises(RuntimeError):
        _separate._run_demucs(job, tmp_path / "s.wav", tmp_path, "cpu", "htdemucs_6s")

    assert killed, "a worker was left warm after an exception"


class _FakePipe:
    """stdin that accepts the request, stderr that dies mid-stream."""

    def __init__(self, on_read):
        self._on_read = on_read

    def write(self, _data):
        return None

    def flush(self):
        return None

    def read(self, _n):
        return self._on_read()


class _FakeProc:
    def __init__(self, on_read):
        self.stdin = _FakePipe(on_read)
        self.stderr = _FakePipe(on_read)

    def poll(self):
        return None

    def terminate(self):
        return None


def test_an_exception_inside_the_read_loop_still_tears_the_worker_down(tmp_path, monkeypatch):
    """The case #514 is actually about: teardown sat *after* the finally, so an
    OSError from proc.stderr.read(1) -- which happens when the API thread's
    terminate() races the read -- propagated with the worker left warm."""
    killed = []
    monkeypatch.setattr(_separate, "_kill_worker", lambda: killed.append(True))

    def _boom():
        raise OSError("broken pipe")

    monkeypatch.setattr(
        _separate, "_get_worker", lambda device, model="htdemucs_6s": _FakeProc(_boom)
    )
    monkeypatch.setattr(_separate, "set_proc", lambda *a, **kw: None)

    job = _job(status="separating")
    with pytest.raises(OSError):
        _separate._run_demucs(job, tmp_path / "s.wav", tmp_path, "cpu", "htdemucs_6s")

    assert killed, "a worker whose CUDA state followed an exception stayed warm"


def test_cancel_between_the_gpu_attempt_and_the_cpu_fallback_is_honoured(tmp_path, monkeypatch):
    # The expensive case: without this the entire CPU separation ran to
    # completion -- 10+ minutes -- while the UI showed "Cancelling".
    attempts = []

    def _fake_run(job, source, job_dir, device, model="htdemucs_6s"):
        attempts.append(device)
        if device != "cpu":
            job.cancel_requested = True  # the user cancels during the failure
            return 1, ["boom"]
        raise AssertionError("the CPU fallback must not run for a cancelled job")

    monkeypatch.setattr(_separate, "_run_demucs", _fake_run)
    monkeypatch.setattr(_separate, "get_demucs_device", lambda: "cuda")

    job = _job(status="separating")
    with pytest.raises(JobCancelled):
        _separate.separate(job, tmp_path / "s.wav", tmp_path)

    assert attempts == ["cuda"]
