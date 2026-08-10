import logging
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
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
