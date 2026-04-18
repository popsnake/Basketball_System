from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from apps.api.deps import get_scoring_engine
from models_dl.inference import ScoringEngine
from models_dl.schemas import ScoreSequenceRequest, ScoringResult

router = APIRouter(prefix="/v1", tags=["scoring"])


@router.post("/score/sequence", response_model=ScoringResult)
def score_sequence(
    body: ScoreSequenceRequest,
    engine: ScoringEngine = Depends(get_scoring_engine),
) -> ScoringResult:
    if not body.sequence:
        raise HTTPException(status_code=400, detail="sequence must be non-empty")
    try:
        arr = np.asarray(body.sequence, dtype=np.float64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid sequence: {e}") from e
    if arr.ndim != 2 or arr.shape[1] != 75:
        raise HTTPException(
            status_code=400,
            detail=f"sequence must have shape (T,75), got {arr.shape}",
        )
    try:
        return engine.predict_from_flat(
            arr,
            return_angles=body.options.return_angles,
            return_keyframe=body.options.return_keyframe_index,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
