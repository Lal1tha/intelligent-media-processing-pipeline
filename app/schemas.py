from datetime import datetime
from pydantic import BaseModel


class UploadResponse(BaseModel):
    processing_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    processing_id: str
    status: str
    failure_reason: str | None = None
    created_at: datetime
    processed_at: datetime | None = None
