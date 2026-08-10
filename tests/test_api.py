from tests.conftest import image_bytes, png_bytes
from app.db.models import AnalysisResult, ProcessingStatus, Upload


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_and_status(client):
    response = client.post("/api/v1/uploads", files={"file": ("car.jpg", image_bytes(), "image/jpeg")})
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "pending"
    status = client.get(f"/api/v1/processing/{payload['processing_id']}")
    assert status.status_code == 200
    assert status.json()["status"] == "pending"


def test_invalid_and_unsupported_uploads(client):
    invalid = client.post("/api/v1/uploads", files={"file": ("bad.jpg", b"not-image", "image/jpeg")})
    unsupported = client.post("/api/v1/uploads", files={"file": ("bad.gif", image_bytes(), "image/gif")})
    assert invalid.status_code == 422
    assert unsupported.status_code == 415


def test_upload_rejects_content_mismatch_and_missing_file(client):
    mismatch = client.post("/api/v1/uploads", files={"file": ("car.jpg", png_bytes(), "image/jpeg")})
    missing = client.post("/api/v1/uploads")
    assert mismatch.status_code == 422
    assert missing.status_code == 422


def test_missing_pending_and_completed_result(client, db):
    assert client.get("/api/v1/processing/missing").status_code == 404
    response = client.post("/api/v1/uploads", files={"file": ("car.jpg", image_bytes(), "image/jpeg")})
    processing_id = response.json()["processing_id"]
    assert client.get(f"/api/v1/processing/{processing_id}/results").status_code == 409
    record = db.get(Upload, processing_id)
    record.status = ProcessingStatus.completed
    db.add(AnalysisResult(processing_id=processing_id, overall_status="accepted", result_json={"processing_id": processing_id, "status": "completed", "summary": {"overall_status": "accepted"}, "checks": {}}))
    db.commit()
    completed = client.get(f"/api/v1/processing/{processing_id}/results")
    assert completed.status_code == 200
    assert completed.json()["summary"]["overall_status"] == "accepted"
