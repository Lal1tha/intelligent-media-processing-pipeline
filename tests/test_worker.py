from app.db.models import AnalysisResult, ProcessingStatus, Upload
from app.workers import image_worker


def make_upload(processing_id: str) -> Upload:
    return Upload(
        id=processing_id,
        original_filename="car.jpg",
        stored_filename=f"{processing_id}.jpg",
        file_path="unused-by-mocked-analysis.jpg",
        mime_type="image/jpeg",
        file_size=1,
        sha256="a" * 64,
        perceptual_hash="0" * 16,
        status=ProcessingStatus.pending,
    )


def test_worker_persists_result_and_completes(db, session_factory, monkeypatch):
    upload = make_upload("worker-success")
    db.add(upload)
    db.commit()
    processing_id = upload.id
    monkeypatch.setattr(image_worker, "SessionLocal", session_factory)
    monkeypatch.setattr(image_worker, "analyze_upload", lambda record, _: {"processing_id": record.id, "status": "completed", "summary": {"overall_status": "accepted"}, "checks": {}})

    image_worker.process_image(processing_id)

    with session_factory() as verification_db:
        refreshed = verification_db.get(Upload, processing_id)
        assert refreshed.status == ProcessingStatus.completed
        assert verification_db.query(AnalysisResult).filter_by(processing_id=processing_id).one().overall_status == "accepted"


def test_worker_records_failure(db, session_factory, monkeypatch):
    upload = make_upload("worker-failure")
    db.add(upload)
    db.commit()
    processing_id = upload.id
    monkeypatch.setattr(image_worker, "SessionLocal", session_factory)
    def broken_analysis(*_):
        raise ValueError("corrupt image")
    monkeypatch.setattr(image_worker, "analyze_upload", broken_analysis)

    image_worker.process_image(processing_id)

    with session_factory() as verification_db:
        refreshed = verification_db.get(Upload, processing_id)
        assert refreshed.status == ProcessingStatus.failed
        assert refreshed.failure_reason
