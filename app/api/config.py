from __future__ import annotations

from fastapi import APIRouter

from app.core.config import EXTRA_STEM_NAMES, SEPARATION_MODELS, STEM_NAMES, model_stems
from app.core.settings import get_separation_model

router = APIRouter()


@router.get("/config")
def get_config() -> dict:
    # extra_stem_names (#275) are produced only when a job's on-demand
    # lead/backing vocal split has run -- kept separate from stem_names so
    # existing clients that assume "every job produces exactly these stems"
    # are unaffected; new clients merge it into their lane vocab (see
    # syncStemNamesFromAPI in static/js/constants.js).
    #
    # stem_names stays the canonical superset that colours, labels and lane
    # ordering are keyed by. active_stems is what the CURRENTLY selected model
    # produces, so the import UI can offer only the stems that model can make;
    # model_stems maps every model id to its set so the picker can update the
    # chips without waiting for a round trip.
    return {
        "stem_names": list(STEM_NAMES),
        "extra_stem_names": list(EXTRA_STEM_NAMES),
        "active_stems": list(model_stems(get_separation_model())),
        "model_stems": {mid: list(model_stems(mid)) for mid in SEPARATION_MODELS},
        "models": [
            {"id": mid, "label": m["label"], "description": m.get("description", "")}
            for mid, m in SEPARATION_MODELS.items()
        ],
    }
