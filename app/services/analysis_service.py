from datetime import datetime, timezone
import cv2
from sqlalchemy.orm import Session
from app.analyzers import image_checks
from app.analyzers.ocr import detect_vehicle_number, extract_text
from app.core.config import settings
from app.db.models import Upload


def aggregate(checks: dict[str, dict]) -> dict:
    """Convert non-authoritative check outcomes into a conservative review signal."""
    warnings = sum(check.get("status") == "warning" for check in checks.values())
    overall = "rejected" if warnings >= settings.rejection_warning_count else ("review_recommended" if warnings else "accepted")
    return {"overall_status": overall, "issue_count": warnings, "disclaimer": "Heuristic screening only; it is not a final authenticity determination."}


def analyze_upload(upload: Upload, db: Session) -> dict:
    image = cv2.imread(upload.file_path)
    if image is None:
        raise ValueError("Unable to decode uploaded image.")
    checks = {}
    checks["blur"] = image_checks.blur(image)
    checks["brightness"] = image_checks.brightness(image)
    checks["dimensions"] = image_checks.dimensions(image)
    candidates = db.query(Upload.sha256, Upload.perceptual_hash).filter(Upload.id != upload.id).all()
    checks["duplicate"] = image_checks.duplicate(upload.sha256, upload.perceptual_hash, candidates)
    checks["ocr"] = extract_text(image)
    checks["vehicle_number"] = detect_vehicle_number(image, checks["ocr"])
    checks["screenshot"] = image_checks.screenshot(image)
    checks["photo_of_photo"] = image_checks.photo_of_photo(image)
    checks["metadata"] = image_checks.metadata(upload.file_path)
    checks["tampering"] = image_checks.tampering(image, checks["metadata"])
    return {"processing_id": upload.id, "status": "completed", "summary": aggregate(checks), "checks": checks, "processed_at": datetime.now(timezone.utc).isoformat()}
