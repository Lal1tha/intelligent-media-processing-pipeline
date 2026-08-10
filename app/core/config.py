from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    database_url: str = "mysql+pymysql://root:password@localhost:3306/media_pipeline"
    redis_url: str = "redis://localhost:6379/0"
    upload_dir: Path = Path("uploads")
    max_upload_size_mb: int = 10
    blur_threshold: float = 100.0
    min_image_width: int = 640
    min_image_height: int = 480
    min_image_pixels: int = 307200
    rejection_warning_count: int = 3
    tesseract_cmd: str = ""

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


settings = Settings()
