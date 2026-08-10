import enum
import uuid
from datetime import datetime
from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, String, Text, func, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.database import Base
from sqlalchemy import LargeBinary


class ProcessingStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


ALLOWED_STATUS_TRANSITIONS: dict[ProcessingStatus, set[ProcessingStatus]] = {
    ProcessingStatus.pending: {ProcessingStatus.processing, ProcessingStatus.failed},
    ProcessingStatus.processing: {ProcessingStatus.completed, ProcessingStatus.failed},
    ProcessingStatus.completed: set(),
    ProcessingStatus.failed: set(),
}


def transition(upload: "Upload", target: ProcessingStatus) -> None:
    """Apply only lifecycle transitions owned by the upload/worker workflow."""
    if target not in ALLOWED_STATUS_TRANSITIONS[upload.status]:
        raise ValueError(f"Invalid processing status transition: {upload.status.value} -> {target.value}")
    upload.status = target


class Upload(Base):
    __tablename__ = "uploads"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_filename: Mapped[str] = mapped_column(String(255), unique=True)
    file_path: Mapped[str] = mapped_column(String(1024))
    image_data: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    mime_type: Mapped[str] = mapped_column(String(100))
    file_size: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    perceptual_hash: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[ProcessingStatus] = mapped_column(Enum(ProcessingStatus), default=ProcessingStatus.pending, index=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    result: Mapped["AnalysisResult | None"] = relationship(back_populates="upload", uselist=False, cascade="all, delete-orphan")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    processing_id: Mapped[str] = mapped_column(ForeignKey("uploads.id"), unique=True, index=True)
    overall_status: Mapped[str] = mapped_column(String(40))
    result_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    upload: Mapped[Upload] = relationship(back_populates="result")


Index("ix_uploads_status_created", Upload.status, Upload.created_at)
