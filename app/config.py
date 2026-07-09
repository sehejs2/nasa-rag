from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    OPENAI_API_KEY: str = ""
    NASA_API_KEY: str = ""
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/nasa_rag"

    RETRIEVAL_CANDIDATE_POOL_SIZE: int = 20
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_RERANK_ENABLED: bool = True

    # Model used to judge faithfulness/key-facts in the Phase 7 eval harness.
    # Defaults to gpt-4o-mini for cost; see app/evals/faithfulness.py for the
    # same-model-family leniency caveat this knob exists to let you work around.
    JUDGE_MODEL: str = "gpt-4o-mini"


settings = Settings()
