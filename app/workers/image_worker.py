import logging
from datetime import datetime, timezone
from app.db.database import SessionLocal
from app.db.models import AnalysisResult, ProcessingStatus, Upload, transition
from app.services.analysis_service import analyze_upload

logger = logging.getLogger(__name__)


def process_image(processing_id: str) -> None:
    db = SessionLocal()
    try:
        upload = db.get(Upload, processing_id)
        # RQ delivery is at-least-once: a duplicate job must not regress an active/terminal record.
        if not upload or upload.status != ProcessingStatus.pending:
            return
        transition(upload, ProcessingStatus.processing)
        upload.failure_reason = None
        db.commit()
        payload = analyze_upload(upload, db)
        existing = db.query(AnalysisResult).filter_by(processing_id=processing_id).one_or_none()
        if existing:
            existing.overall_status, existing.result_json = payload["summary"]["overall_status"], payload
        else:
            db.add(AnalysisResult(processing_id=processing_id, overall_status=payload["summary"]["overall_status"], result_json=payload))
        transition(upload, ProcessingStatus.completed)
        upload.processed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("processing_completed id=%s", processing_id)
    except Exception as exc:
        db.rollback()
        upload = db.get(Upload, processing_id)
        if upload:
            if upload.status in (ProcessingStatus.pending, ProcessingStatus.processing):
                transition(upload, ProcessingStatus.failed)
            upload.failure_reason = "Image processing failed. Please contact support if the issue persists."
            db.commit()
        logger.exception("processing_failed id=%s error=%s", processing_id, exc)
    finally:
        db.close()
