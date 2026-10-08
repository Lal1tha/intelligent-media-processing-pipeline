from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text
from app.api import router
from app.core.logging import configure_logging
from app.db.database import SessionLocal
from fastapi.middleware.cors import CORSMiddleware

configure_logging()
app = FastAPI(title="Intelligent Media Processing Pipeline", version="1.0.0", description="Asynchronous heuristic screening for uploaded vehicle images.")
app.include_router(router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "HTTP_ERROR", "message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content={"error": detail})


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, __: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": {"code": "VALIDATION_ERROR", "message": "The request is invalid or missing required data."}})


@app.exception_handler(Exception)
async def unhandled_error(_, __):
    return JSONResponse(status_code=500, content={"error": {"code": "INTERNAL_ERROR", "message": "An unexpected server error occurred."}})


@app.get("/health", summary="Liveness check")
def health():
    return {"status": "ok"}


@app.get("/ready", summary="Database readiness check")
def ready():
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
