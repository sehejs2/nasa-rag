from app.config import Settings


def test_cors_origins_list_splits_and_strips_comma_separated_origins() -> None:
    settings = Settings(CORS_ORIGINS="https://a.example.com, https://b.example.com ,,")

    assert settings.cors_origins_list == ["https://a.example.com", "https://b.example.com"]


def test_cors_origins_list_defaults_to_local_frontend_origin() -> None:
    settings = Settings()

    assert settings.cors_origins_list == ["http://localhost:3000"]


def test_database_url_normalizes_postgres_scheme_to_postgresql() -> None:
    settings = Settings(DATABASE_URL="postgres://user:pass@host:5432/db")

    assert settings.DATABASE_URL == "postgresql://user:pass@host:5432/db"


def test_database_url_leaves_postgresql_scheme_unchanged() -> None:
    settings = Settings(DATABASE_URL="postgresql://user:pass@host:5432/db")

    assert settings.DATABASE_URL == "postgresql://user:pass@host:5432/db"
