# Intelligent Media Processing Pipeline

An asynchronous FastAPI service that accepts vehicle images, stores them locally, queues CPU-heavy screening with Redis/RQ, and persists explainable results in MySQL. It is a heuristic screening system for manual review—not forensic proof or an automated final decision.

## Architecture

```text
Client -> FastAPI upload API -> MySQL (metadata) + local upload volume
                              -> Redis/RQ -> Worker -> OpenCV/OCR/heuristics
                                                   -> MySQL (JSON result)
```

The upload request validates and stores the file, creates a `pending` record, and queues a job before returning `202`. The worker owns transitions from `pending` to `processing`, then `completed` or `failed`; it records a safe client-facing failure reason and logs the detailed exception.

## Features

- Secure JPEG/PNG/WEBP upload validation: extension, declared MIME type, maximum byte size, Pillow verification, and OpenCV decode.
- SHA-256 exact duplicate detection and 8x8 average-hash near-duplicate comparison.
- Laplacian-variance blur, lighting/contrast, and dimension/pixel-count checks.
- Full-image Tesseract OCR fallback plus lightweight OpenCV plate-region proposals, plate-specific preprocessing variants, PSM 7/8/13, character whitelist, and confidence-based OCR consensus.
- Broad Indian vehicle-registration extraction/validation, preserving raw OCR and exposing position-aware OCR ambiguity separately.
- Screenshot, photo-of-photo, EXIF, and editing/tampering signals. GPS presence is recorded but coordinates are never returned.
- Structured result JSON and configurable review aggregation.

## Project structure

`app/analyzers` contains focused, real image heuristics; `app/services` composes upload and analysis work; `app/workers` contains the RQ boundary; `app/db` contains SQLAlchemy persistence; `migrations` contains Alembic schema migration; and `tests` has deterministic unit/API coverage.

## Local setup

Prerequisites: Python 3.11+, MySQL, Redis, and Tesseract OCR.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # Windows (or cp .env.example .env)
mysql -u root -p -e "CREATE DATABASE media_pipeline;"
alembic upgrade head
uvicorn app.main:app --reload
```

In another terminal, run Redis (for example `redis-server`) and then the worker:

```bash
rq worker media-processing --url redis://localhost:6379/0
```

Open Swagger at `http://localhost:8000/docs`. `/health` is a cheap liveness probe; `/ready` checks database connectivity.

### Tesseract

Install with `winget install UB-Mannheim.TesseractOCR` on Windows, `brew install tesseract` on macOS, or `sudo apt install tesseract-ocr` on Debian/Ubuntu. If it is not on `PATH`, set `TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe` in `.env`. OCR returns `unavailable` rather than inventing text when the executable is missing.

## Configuration

`.env.example` documents `DATABASE_URL`, `REDIS_URL`, `UPLOAD_DIR`, `MAX_UPLOAD_SIZE_MB`, `BLUR_THRESHOLD`, `MIN_IMAGE_WIDTH`, `MIN_IMAGE_HEIGHT`, `MIN_IMAGE_PIXELS`, `REJECTION_WARNING_COUNT`, and `TESSERACT_CMD`. Thresholds are deliberately configuration, not hidden magic: tune them with representative data. The blur default is a common Laplacian-variance baseline, while lighting considers mean and the percentage of extreme pixels. The aggregate result becomes `rejected` only when warning count meets `REJECTION_WARNING_COUNT`; otherwise warnings result in `review_recommended`.

## Docker

```bash
docker compose up --build
```

Compose starts MySQL, Redis, a one-shot migration service, API, and worker. It installs Tesseract in the shared application image, sets `PYTHONPATH=/app` so Alembic, Uvicorn, and RQ can reliably import the application package, runs Alembic before API/worker startup without concurrent migration races, and persists both MySQL and uploads in named volumes.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/api/v1/uploads` | Upload and queue an image |
| GET | `/api/v1/processing/{id}` | Status/failure information |
| GET | `/api/v1/processing/{id}/results` | Completed analysis result |
| GET | `/health` | Liveness check |

```bash
curl -F "file=@vehicle.jpg" http://localhost:8000/api/v1/uploads
curl http://localhost:8000/api/v1/processing/<processing_id>
curl http://localhost:8000/api/v1/processing/<processing_id>/results
```

An upload returns `{"processing_id":"...","status":"pending","message":"Image uploaded successfully and queued for processing."}`. Completed results include each check's measurements, thresholds/signals where relevant, and summary `accepted`, `review_recommended`, or `rejected` status.

## Methodology and limits

Every check runs against image data or metadata. Blur uses edge variance; brightness measures grayscale distribution; dimensions assess minimum usable resolution; duplicates use hashes. OCR retains a full-image fallback, then proposes a small number of plate-like contours using contrast, rectangular geometry, edge density, and mild location scoring. Each crop is enlarged and tried with CLAHE, Otsu, and adaptive threshold variants plus plate-focused Tesseract modes. Matching registration candidates are grouped so repeated readings raise confidence. Number validation uses a deliberately broad state-prefix pattern and applies OCR substitutions only in expected numeric/letter positions. Screenshot/photo-of-photo/tampering checks combine simple signals such as screen dimensions, edge geometry, uniform blocks, and editing software metadata. Those signals are probabilistic, can have false positives/negatives, and never prove authenticity or tampering.

The design begins with local storage for simple development; replace `file_path` storage with an S3/object-store adapter for scale. RQ/Redis workers can be horizontally scaled; production follow-ups include retries/dead-letter queues, trained plate detection/OCR, authentication, rate limits, metrics, and monitoring.

## Testing

```bash
pytest
```

Tests use SQLite and mocked queue submission for isolated API checks, plus deterministic generated images for blur, lighting, dimensions, hashing, and plate validation. MySQL, Redis, an actual RQ worker, and Tesseract must be available to perform a full end-to-end local run.
