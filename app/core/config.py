import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "local")
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
    frontend_origins: str = os.getenv(
        "FRONTEND_ORIGINS",
        os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"),
    )
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_anon_key: str = os.getenv("SUPABASE_ANON_KEY", "")
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    supabase_jwt_secret: str = os.getenv("SUPABASE_JWT_SECRET", "")
    developer_user_ids: str = os.getenv("DEVELOPER_USER_IDS", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # Microsoft SQL Server alternative to Supabase (app.services.database_service).
    # Unused unless that module is wired in instead of app.services.supabase_service.
    mssql_server: str = os.getenv("MSSQL_SERVER", "")
    mssql_database: str = os.getenv("MSSQL_DATABASE", "")
    mssql_user: str = os.getenv("MSSQL_USER", "")
    mssql_password: str = os.getenv("MSSQL_PASSWORD", "")
    mssql_driver: str = os.getenv("MSSQL_DRIVER", "ODBC Driver 17 for SQL Server")
    mssql_trusted_connection: bool = (
        os.getenv("MSSQL_TRUSTED_CONNECTION", "false").lower() == "true"
    )
    jwt_secret: str = os.getenv("JWT_SECRET", "")
    jwt_access_token_expires_minutes: int = int(
        os.getenv("JWT_ACCESS_TOKEN_EXPIRES_MINUTES", "60")
    )
    jwt_refresh_token_expires_days: int = int(
        os.getenv("JWT_REFRESH_TOKEN_EXPIRES_DAYS", "30")
    )


settings = Settings()
