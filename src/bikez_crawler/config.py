from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mongodb_url: str

    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    s3_raw_bucket: str
    s3_production_bucket: str
    s3_region: str = "us-east-1"

    bikez_http_concurrency: int = 2
    bikez_http_interval_ms: int = 1000
    bikez_http_interval_cap: int = 3

    pending_links_concurrency: int = 3
    image_upload_concurrency: int = 3
    queue_max_attempts: int = 5
    queue_lease_seconds: int = 300
    max_bikes_to_crawl: int | None = None
    reseed_interval_days: int = 7

    pipeline_concurrency: int = 2
    pipeline_bike_tag: str | None = None
    pipeline_intermediate_prefix: str = "pipeline/intermediate"


def load_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
