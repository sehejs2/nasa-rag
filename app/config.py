from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    OPENAI_API_KEY: str = ""
    NASA_API_KEY: str = ""
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/nasa_rag"

    # Comma-separated list of allowed CORS origins for the FastAPI app, e.g.
    # "https://my-app.vercel.app,https://my-app-git-preview.vercel.app". Defaults
    # to the local frontend dev origin.
    CORS_ORIGINS: str = "http://localhost:3000"

    RETRIEVAL_CANDIDATE_POOL_SIZE: int = 20
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_RERANK_ENABLED: bool = True

    # Model used to judge faithfulness/key-facts in the Phase 7 eval harness.
    # Defaults to gpt-4o-mini for cost; see app/evals/faithfulness.py for the
    # same-model-family leniency caveat this knob exists to let you work around.
    JUDGE_MODEL: str = "gpt-4o-mini"

    @field_validator("DATABASE_URL")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        # Some platforms (Railway, Heroku-style) hand out postgres:// URLs;
        # psycopg only recognizes the postgresql:// scheme.
        if value.startswith("postgres://"):
            return "postgresql://" + value[len("postgres://") :]
        return value

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()
