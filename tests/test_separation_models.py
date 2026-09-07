"""The separation-model registry and the plumbing that reads it.

Covers the parts where a wrong answer is silent rather than loud: a model id
that no longer exists must not raise on a job that was queued before it went
away, a two-stem model must not have four empty lanes invented for it, and the
persistent worker must not be reused across two different checkpoints.
"""

from __future__ import annotations

import pytest

from app.core.config import (
    DEFAULT_SEPARATION_MODEL,
    SEPARATION_MODELS,
    STEM_NAMES,
    model_backend,
    model_checkpoint,
    model_stems,
    model_subdir,
)
from app.pipeline import separate as sep_mod


def test_every_registry_entry_declares_the_full_shape():
    for model_id, entry in SEPARATION_MODELS.items():
        for field in ("backend", "checkpoint", "stems", "subdir", "label"):
            assert field in entry, f"{model_id} is missing {field!r}"
        assert entry["backend"] in ("demucs", "roformer")
        assert entry["stems"], f"{model_id} declares no stems"


def test_model_stems_stay_within_the_canonical_superset():
    # Colours, labels and lane ordering are all keyed by STEM_NAMES; a model
    # emitting a name outside it would render as an unstyled, unlabelled lane.
    for model_id in SEPARATION_MODELS:
        for name in model_stems(model_id):
            assert name in STEM_NAMES, f"{model_id} emits unknown stem {name!r}"


def test_subdirs_are_unique_so_two_backends_cannot_read_each_others_output():
    subdirs = [entry["subdir"] for entry in SEPARATION_MODELS.values()]
    assert len(subdirs) == len(set(subdirs))


@pytest.mark.parametrize("unknown", ["", "removed_in_a_later_build", None])
def test_an_unknown_model_id_falls_back_instead_of_raising(unknown):
    # A persisted id can outlive the entry that named it -- a settings file from
    # a newer build, or a model dropped from the registry. The job still has to
    # run, on the default, rather than dying with a KeyError.
    default = SEPARATION_MODELS[DEFAULT_SEPARATION_MODEL]
    assert model_backend(unknown) == default["backend"]
    assert model_checkpoint(unknown) == default["checkpoint"]
    assert model_stems(unknown) == tuple(default["stems"])
    assert model_subdir(unknown) == default["subdir"]


def test_the_vocal_model_produces_two_stems():
    assert model_stems("kim_ft_vocal") == ("vocals", "other")


def test_the_six_stem_models_produce_all_of_them():
    for model_id in ("htdemucs_6s", "bs_roformer_sw"):
        assert model_stems(model_id) == tuple(STEM_NAMES)


def test_the_spawn_command_selects_the_backends_worker():
    demucs = sep_mod._spawn_worker_cmd("cpu", "htdemucs_6s")
    assert demucs[-2:] == ["app.pipeline.demucs_worker", "cpu"]

    # The roformer worker serves several checkpoints, so it is told which.
    roformer = sep_mod._spawn_worker_cmd("cuda", "bs_roformer_sw")
    assert "app.pipeline.roformer_worker" in roformer
    assert roformer[-2:] == ["cuda", "bs_roformer_sw"]


def test_a_worker_is_not_reused_across_models(monkeypatch):
    # The reuse key used to be the device alone. A worker holds ONE loaded
    # checkpoint, so reusing a demucs worker for a Roformer job would quietly
    # separate with the wrong model rather than fail.
    spawned: list[tuple[str, str]] = []

    class _Proc:
        def poll(self):
            return None

    monkeypatch.setattr(sep_mod, "_kill_worker", lambda: None)
    monkeypatch.setattr(
        sep_mod.subprocess,
        "Popen",
        lambda cmd, **kw: (spawned.append((cmd[-2], cmd[-1])), _Proc())[1],
    )
    sep_mod._worker.clear()
    try:
        sep_mod._get_worker("cpu", "htdemucs_6s")
        sep_mod._get_worker("cpu", "htdemucs_6s")  # same pair -> reused
        assert len(spawned) == 1
        sep_mod._get_worker("cpu", "bs_roformer_sw")  # different model -> fresh
        assert len(spawned) == 2
    finally:
        sep_mod._worker.clear()


def test_a_two_stem_job_gets_no_complement_backing_track(tmp_path):
    # "Original" is the sum of the stems the user did NOT pick, so the studio
    # can rebuild the song without doubling the picked ones. For a vocal model
    # that complement is just "other" -- which already IS the whole
    # instrumental, so building it would write a second copy of a file the job
    # has, under a name that claims to be something else.
    from app.core.models import Job
    from app.pipeline.collect import make_original_track

    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    for name in ("vocals", "other"):
        (stems_dir / f"{name}.wav").write_bytes(b"RIFF")

    job = Job(id="abcdef123456", separation_model="kim_ft_vocal", selected_stems=["vocals"])
    assert make_original_track(job, tmp_path, stems_dir) is None


def test_recovery_restores_the_stem_set_the_job_actually_produced(tmp_path):
    # An orphaned job dir used to be rescanned against all six stem names. A
    # two-stem job recovered that way keeps only the files that exist, but the
    # persisted list is what says the other four were never meant to be there.
    import json

    from app.core.registry import _recover_done_job

    job_dir = tmp_path / "abcdef123456"
    stems_dir = job_dir / "stems"
    stems_dir.mkdir(parents=True)
    for name in ("vocals", "other"):
        (stems_dir / f"{name}.wav").write_bytes(b"RIFF")
    (job_dir / "metadata.json").write_text(
        json.dumps(
            {
                "title": "Two stem track",
                "separation_model": "kim_ft_vocal",
                "stems": ["vocals", "other"],
            }
        ),
        encoding="utf-8",
    )

    job = _recover_done_job(job_dir)
    assert job is not None
    assert job.separation_model == "kim_ft_vocal"
    assert sorted(s["name"] for s in job.stems) == ["other", "vocals"]
    assert sorted(job.selected_stems) == ["other", "vocals"]


def test_recovery_of_a_job_with_no_persisted_stem_list_still_works(tmp_path):
    # Jobs written before the stem list was persisted have no "stems" key. They
    # are all six-stem Demucs jobs, so scanning the superset is right for them.
    import json

    from app.core.registry import _recover_done_job

    job_dir = tmp_path / "abcdef654321"
    stems_dir = job_dir / "stems"
    stems_dir.mkdir(parents=True)
    for name in ("vocals", "drums", "bass"):
        (stems_dir / f"{name}.wav").write_bytes(b"RIFF")
    (job_dir / "metadata.json").write_text(json.dumps({"title": "Old track"}), encoding="utf-8")

    job = _recover_done_job(job_dir)
    assert job is not None
    assert sorted(s["name"] for s in job.stems) == ["bass", "drums", "vocals"]
