from __future__ import annotations

import secrets
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from crop_copilot.api.dependencies import (
    get_event_logger,
    get_orchestrator,
    get_review_store,
    get_stt_service,
)
from crop_copilot.config import get_settings
from crop_copilot.schemas import AdviceDraft, AdviceRequest, AdviceResponse, Location, ReviewRecord
from crop_copilot.services.speech import GTTSService

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/webm",
    "audio/ogg",
}

app = FastAPI(
    title="Multilingual Crop Health & Advisory Copilot",
    version="0.1.0",
    description="Decision support with calibrated abstention and agronomist review.",
)


class ReviewDecision(BaseModel):
    action: Literal["approve", "reject"]
    reviewer: str
    notes: str | None = None
    approved_text: AdviceDraft | None = None


class FeedbackRequest(BaseModel):
    request_id: str
    helpful: bool
    reason: Literal["incorrect", "unclear", "unsafe", "other"] | None = None


async def save_limited_upload(upload: UploadFile, allowed_types: set[str]) -> Path:
    if upload.content_type not in allowed_types:
        raise HTTPException(
            status_code=415, detail=f"Unsupported media type: {upload.content_type}"
        )
    suffix = Path(upload.filename or "upload.bin").suffix[:10]
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            path = Path(handle.name)
            total = 0
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Upload exceeds 10 MB")
                handle.write(chunk)
            if total == 0:
                raise HTTPException(status_code=422, detail="Upload is empty")
        return path
    except BaseException:
        if path is not None:
            path.unlink(missing_ok=True)
        raise


def verify_reviewer_key(x_reviewer_key: Annotated[str | None, Header()] = None) -> None:
    settings = get_settings()
    key = settings.reviewer_api_key
    if not key or (settings.environment.casefold() == "production" and key == "dev-review-key"):
        raise HTTPException(status_code=503, detail="Reviewer access is not configured")
    if not x_reviewer_key or not secrets.compare_digest(x_reviewer_key.encode(), key.encode()):
        raise HTTPException(status_code=401, detail="Invalid reviewer key")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/advice", response_model=AdviceResponse)
async def create_advice(
    image: Annotated[UploadFile, File()],
    crop: Annotated[str, Form(min_length=2, max_length=80)],
    question: Annotated[str, Form(min_length=2, max_length=2000)],
    language: Annotated[str, Form()] = "en",
    latitude: Annotated[float, Form(ge=-90, le=90)] = 14.4644,
    longitude: Annotated[float, Form(ge=-180, le=180)] = 75.9218,
    region: Annotated[str | None, Form(max_length=120)] = None,
    market: Annotated[str | None, Form(max_length=120)] = None,
) -> AdviceResponse:
    if language not in {"en", "kn", "hi"}:
        raise HTTPException(status_code=422, detail="Supported languages: en, kn, hi")
    image_path = await save_limited_upload(image, ALLOWED_IMAGE_TYPES)
    try:
        request = AdviceRequest(
            crop=crop,
            question=question,
            language=language,
            location=Location(
                latitude=latitude,
                longitude=longitude,
                region=region,
                market=market,
            ),
            image_path=str(image_path),
        )
        try:
            return await run_in_threadpool(lambda: get_orchestrator().advise(request))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        image_path.unlink(missing_ok=True)


@app.get("/v1/advice/reviews/{review_id}", response_model=AdviceResponse)
def review_status(review_id: str) -> AdviceResponse:
    try:
        return get_orchestrator().public_review_status(review_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Review not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get(
    "/v1/reviewer/queue",
    response_model=list[ReviewRecord],
    dependencies=[Depends(verify_reviewer_key)],
)
def review_queue() -> list[ReviewRecord]:
    return get_review_store().list_pending()


@app.post(
    "/v1/reviewer/{review_id}/decision",
    response_model=ReviewRecord,
    dependencies=[Depends(verify_reviewer_key)],
)
def review_decision(review_id: str, decision: ReviewDecision) -> ReviewRecord:
    store = get_review_store()
    try:
        if decision.action == "approve":
            if decision.approved_text is None:
                raise HTTPException(status_code=422, detail="Approved text is required")
            record = store.approve(
                review_id, decision.reviewer, decision.approved_text, decision.notes
            )
        else:
            record = store.reject(review_id, decision.reviewer, decision.notes or "")
        get_event_logger().log(
            "review_decided",
            record.request_id,
            review_id=record.id,
            decision=record.status.value,
        )
        return record
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/feedback", status_code=202)
def submit_feedback(feedback: FeedbackRequest) -> dict[str, str]:
    get_event_logger().log(
        "farmer_feedback",
        feedback.request_id,
        helpful=feedback.helpful,
        reason=feedback.reason,
    )
    return {"status": "accepted"}


@app.post("/v1/speech/transcribe")
async def transcribe_audio(
    audio: Annotated[UploadFile, File()],
    language: Annotated[str, Form()] = "kn",
) -> dict[str, str]:
    audio_path = await save_limited_upload(audio, ALLOWED_AUDIO_TYPES)
    try:
        try:
            text = await run_in_threadpool(
                lambda: get_stt_service().transcribe(audio_path, language)
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"text": text, "language": language}
    finally:
        audio_path.unlink(missing_ok=True)


@app.post("/v1/speech/synthesize")
def synthesize_speech(text: str, language: str = "kn") -> Response:
    if len(text) > 4000:
        raise HTTPException(status_code=413, detail="Text exceeds 4000 characters")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as handle:
        output_path = Path(handle.name)
    try:
        try:
            GTTSService().synthesize(text, language, output_path)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(content=output_path.read_bytes(), media_type="audio/mpeg")
    finally:
        output_path.unlink(missing_ok=True)
