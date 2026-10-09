
import logging

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import ProcessingStatus, Upload, transition
from app.schemas import StatusResponse, UploadResponse
from app.services.upload_service import save_upload
from app.workers.queue import enqueue_processing


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


def find_upload(processing_id: str, db: Session) -> Upload:
    record = db.get(Upload, processing_id)
    if not record:
        raise HTTPException(404, detail={"code": "PROCESSING_NOT_FOUND", "message": "No processing record exists for the provided ID."})
    return record


@router.post("/uploads", response_model=UploadResponse, status_code=202, summary="Upload and queue an image")
async def upload_image(file: UploadFile = File(..., description="JPEG, PNG, or WEBP vehicle image"), db: Session = Depends(get_db)):
    record = await save_upload(file, db)
    try:
        enqueue_processing(record.id)
    except Exception:
        transition(record, ProcessingStatus.failed)
        record.failure_reason = "Unable to queue processing. Please retry the upload."
        db.commit()
        raise HTTPException(503, detail={"code": "QUEUE_UNAVAILABLE", "message": record.failure_reason})
    logger.info("upload_queued id=%s", record.id)
    return UploadResponse(processing_id=record.id, status=record.status.value, message="Image uploaded successfully and queued for processing.")
@router.get("/processing/history", summary="Get image analysis history")
def processing_history(
    db: Session = Depends(get_db),
    search: str | None = Query(default=None, max_length=255),
    limit: int = Query(default=50, ge=1, le=100),
):
    query = db.query(Upload)

    if search and search.strip():
        term = f"%{search.strip()}%"
        query = query.filter(
            (Upload.original_filename.ilike(term))
            | (Upload.id.ilike(term))
        )

    records = (
        query.order_by(desc(Upload.created_at))
        .limit(limit)
        .all()
    )

    history = []

    for record in records:
        result_json = (
            record.result.result_json
            if record.result is not None
            else {}
        )

        # Look for common vehicle-number keys without assuming
        # a particular analysis-result structure.
        vehicle_number = None
        if isinstance(result_json, dict):
            possible_keys = {
                "vehicle_number",
                "detected_vehicle_number",
                "registration_number",
                "plate_number",
                "number_plate",
                "license_plate",
            }

            def find_vehicle_number(data):
                if isinstance(data, dict):
                    for key, value in data.items():
                        if (
                            key.lower() in possible_keys
                            and isinstance(value, str)
                            and value.strip()
                        ):
                            return value.strip()

                    for value in data.values():
                        found = find_vehicle_number(value)
                        if found:
                            return found

                elif isinstance(data, list):
                    for value in data:
                        found = find_vehicle_number(value)
                        if found:
                            return found

                return None

            vehicle_number = find_vehicle_number(result_json)

        history.append({
            "processing_id": record.id,
            "filename": record.original_filename,
            "status": record.status.value,
            "created_at": (
                record.created_at.isoformat()
                if record.created_at else None
            ),
            "processed_at": (
                record.processed_at.isoformat()
                if record.processed_at else None
            ),
            "overall_status": (
                record.result.overall_status
                if record.result is not None else None
            ),
            "vehicle_number": vehicle_number,
        })

    return {
        "count": len(history),
        "items": history,
    }

@router.get("/processing/{processing_id}", response_model=StatusResponse, summary="Get processing status")
def processing_status(processing_id: str, db: Session = Depends(get_db)):
    record = find_upload(processing_id, db)
    return StatusResponse(processing_id=record.id, status=record.status.value, failure_reason=record.failure_reason, created_at=record.created_at, processed_at=record.processed_at)


@router.get("/processing/{processing_id}/results", summary="Get completed analysis result")
def processing_results(processing_id: str, db: Session = Depends(get_db)):
    record = find_upload(processing_id, db)
    if record.status == ProcessingStatus.failed:
        raise HTTPException(409, detail={"code": "PROCESSING_FAILED", "message": record.failure_reason or "Processing failed."})
    if record.status != ProcessingStatus.completed or not record.result:
        raise HTTPException(409, detail={"code": "RESULT_NOT_READY", "message": "Analysis is not complete yet."})
    return record.result.result_json
