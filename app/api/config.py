from __future__ import annotations

from fastapi import APIRouter

from app.core.config import SEPARATION_MODELS, STEM_NAMES, model_stems
from app.core.settings import get_separation_model

router = APIRouter()


@router.get("/config")
def get_config() -> dict:
    # stem_names is the canonical superset (all colours/labels/ordering derive
    # from it). active_stems is what the CURRENTLY selected model produces, so
    # the import UI can offer only the stems that model can extract (a vocal
    # model makes just vocals+other). model_stems maps every model id -> its set
    # so the frontend can update instantly when the model dropdown changes.
    return {
        "stem_names": list(STEM_NAMES),
        "active_stems": list(model_stems(get_separation_model())),
        "model_stems": {mid: list(model_stems(mid)) for mid in SEPARATION_MODELS},
    }
