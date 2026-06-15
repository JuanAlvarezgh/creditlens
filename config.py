from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "creditlens"
    postgres_user: str = "creditlens"
    postgres_password: str = "creditlens_pass"
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "credit_applications"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_model_name: str = "credit_risk_model"
    log_level: str = "INFO"

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = {"env_file": ".env"}


settings = Settings()
