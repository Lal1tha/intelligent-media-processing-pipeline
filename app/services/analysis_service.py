from datetime import datetime, timezone

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.analyzers import image_checks
from app.analyzers.ocr import detect_vehicle_number, extract_text
from app.core.config import settings
from app.db.models import Upload
#from app.analyzers.vehicle_detection import detect_vehicles
from app.analyzers.registration_state import (
    identify_registration_location,
)


def aggregate(checks: dict[str, dict]) -> dict:
    """Convert non-authoritative check outcomes into a conservative review signal."""

    warnings = sum(
        check.get("status") == "warning"
        for check in checks.values()
    )

    overall = (
        "rejected"
        if warnings >= settings.rejection_warning_count
        else (
            "review_recommended"
            if warnings
            else "accepted"
        )
    )

    return {
        "overall_status": overall,
        "issue_count": warnings,
        "disclaimer": (
            "Heuristic screening only; it is not a final authenticity determination."
        ),
    }


def analyze_upload(upload: Upload, db: Session) -> dict:
    """
    Analyze an uploaded image.

    Production-safe approach:
    1. Prefer image_data stored in the database.
    2. Fall back to file_path for local development / older records.
    """

    image = None

    # ---------------------------------------------------------
    # 1. DATABASE IMAGE BYTES
    # ---------------------------------------------------------

    if upload.image_data:
        try:
            image_array = np.frombuffer(
                upload.image_data,
                dtype=np.uint8,
            )

            image = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR,
            )
        except Exception:
            image = None

    # ---------------------------------------------------------
    # 2. LOCAL FILESYSTEM FALLBACK
    # ---------------------------------------------------------

    if image is None and upload.file_path:
        image = cv2.imread(upload.file_path)

    # ---------------------------------------------------------
    # 3. FAIL CLEARLY
    # ---------------------------------------------------------

    if image is None:
        raise ValueError(
            "Unable to decode uploaded image. "
            "The image bytes are unavailable in the database "
            "and the local upload file is not accessible."
        )

    checks = {}

    # ---------------------------------------------------------
    # IMAGE QUALITY
    # ---------------------------------------------------------

    checks["blur"] = image_checks.blur(image)

    checks["brightness"] = image_checks.brightness(image)

    checks["dimensions"] = image_checks.dimensions(image)

    # ---------------------------------------------------------
    # DUPLICATE DETECTION
    # ---------------------------------------------------------

    candidates = (
        db.query(
            Upload.sha256,
            Upload.perceptual_hash,
        )
        .filter(
            Upload.id != upload.id
        )
        .all()
    )

    checks["duplicate"] = image_checks.duplicate(
        upload.sha256,
        upload.perceptual_hash,
        candidates,
    )

    # ---------------------------------------------------------
    # OCR
    # ---------------------------------------------------------

    checks["ocr"] = extract_text(image)


    # ---------------------------------------------------------
    # VEHICLE REGISTRATION NUMBER
    # ---------------------------------------------------------

    checks["vehicle_number"] = detect_vehicle_number(
        image,
        checks["ocr"],
    )

    # ---------------------------------------------------------
    # REGISTRATION STATE AND RTO REGION
    # ---------------------------------------------------------

    vehicle_number_result = checks["vehicle_number"]

    # Support common OCR result field names.
    # Adjust this list if your actual OCR response uses another key.
    registration_number = None

    if isinstance(vehicle_number_result, dict):
        for key in (
            "normalized_candidate",
            "normalized_number",
            "registration_number",
            "candidate",
        ):
            value = vehicle_number_result.get(key)

            if isinstance(value, str) and value.strip():
                registration_number = value
                break

    checks["registration_location"] = (
        identify_registration_location(registration_number)
    )


    # ---------------------------------------------------------
    # SCREENSHOT / PHOTO OF PHOTO
    # ---------------------------------------------------------

    checks["screenshot"] = image_checks.screenshot(
        image
    )

    checks["photo_of_photo"] = image_checks.photo_of_photo(
        image
    )

    # ---------------------------------------------------------
    # METADATA
    # ---------------------------------------------------------

    # Metadata requires an accessible file.
    #
    # If the API and worker are separate containers,
    # upload.file_path may not exist in the worker.
    #
    # Therefore, don't allow metadata inspection to crash
    # the entire analysis.

    try:
        checks["metadata"] = image_checks.metadata(
            upload.file_path
        )
    except Exception:
        checks["metadata"] = {
            "status": "warning",
            "message": (
                "Metadata could not be inspected because "
                "the original upload file is unavailable "
                "to the worker."
            ),
            "software": None,
            "camera_make": None,
            "camera_model": None,
            "gps_present": False,
            "orientation": None,
            "exif_present": False,
            "capture_timestamp": None,
        }

    # ---------------------------------------------------------
    # TAMPERING
    # ---------------------------------------------------------

    checks["tampering"] = image_checks.tampering(
        image,
        checks["metadata"],
    )

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    return {
        "processing_id": upload.id,
        "status": "completed",
        "summary": aggregate(checks),
        "checks": checks,
        "processed_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }
