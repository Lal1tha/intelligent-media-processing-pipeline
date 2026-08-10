from redis import Redis
from rq import Queue
from app.core.config import settings


def get_queue() -> Queue:
    return Queue("media-processing", connection=Redis.from_url(settings.redis_url))


def enqueue_processing(processing_id: str) -> str:
    job = get_queue().enqueue("app.workers.image_worker.process_image", processing_id, job_timeout="10m", result_ttl=86400)
    return job.id
