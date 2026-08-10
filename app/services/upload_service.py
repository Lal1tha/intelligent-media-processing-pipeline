from __future__ import annotations
import hashlib
import io
import mimetypes
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session
from app.analyzers.image_checks import perceptual_hash
from app.core.config import settings
from app.db.models import ProcessingStatus, Upload

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}
FORMAT_TO_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
FORMAT_TO_EXTENSIONS = {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"}, "WEBP": {".webp"}}


async def save_upload(file: UploadFile, db: Session) -> Upload:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, detail={"code": "UNSUPPORTED_MEDIA_TYPE", "message": "Only JPEG, PNG, and WEBP images are supported."})
    if file.content_type and file.content_type not in ALLOWED_MIMES:
        raise HTTPException(415, detail={"code": "UNSUPPORTED_MEDIA_TYPE", "message": "The declared MIME type is not supported."})
    data = await file.read()
    if not data:
        raise HTTPException(400, detail={"code": "EMPTY_FILE", "message": "The upload is empty."})
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, detail={"code": "FILE_TOO_LARGE", "message": "The image exceeds the configured size limit."})
    try:
        with Image.open(io.BytesIO(data)) as im:
            image_format = im.format
            im.verify()
        actual_mime = FORMAT_TO_MIME.get(image_format or "")
        if not actual_mime or suffix not in FORMAT_TO_EXTENSIONS[image_format]:
            raise ValueError("image content does not match extension")
        if file.content_type and file.content_type != actual_mime:
            raise ValueError("image content does not match declared MIME type")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("decoder returned none")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(422, detail={"code": "INVALID_IMAGE", "message": "The file cannot be decoded as a valid image."})
    upload_id = str(uuid4())
    now = datetime.utcnow()
    directory = settings.upload_dir / str(now.year) / f"{now.month:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{upload_id}{suffix}"
    path = directory / stored_filename
    path.write_bytes(data)
    try:
        safe_original_name = Path((file.filename or "upload").replace("\\", "/")).name
        record = Upload(id=upload_id, original_filename=safe_original_name, stored_filename=stored_filename, file_path=str(path),image_data=data, mime_type=actual_mime or mimetypes.guess_type(stored_filename)[0] or "application/octet-stream", file_size=len(data), sha256=hashlib.sha256(data).hexdigest(), perceptual_hash=perceptual_hash(image), status=ProcessingStatus.pending)
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception:
        db.rollback()
        path.unlink(missing_ok=True)
        raise
    return record
