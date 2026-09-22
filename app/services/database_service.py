"""Microsoft SQL Server backend for the app -- replaces app.services.supabase_service.

app/core/auth.py, app/api/auth.py, app/api/currency.py, and app/api/scrape.py
import from this module. It stores scrape jobs, API cost logs, currency
rates, and user accounts in your own SQL Server instance instead of Supabase.
To switch back to Supabase, point those four files' imports back at
app.services.supabase_service (every public function here has the same
name, arguments, and return shape as its supabase_service.py counterpart).

Because there is no Supabase Auth here, user accounts (email + password hash)
are stored in this database too (see database/mssql_schema.sql, table
dbo.users) and sessions are plain JWTs signed with JWT_SECRET, not Supabase
tokens. register_user() is provided for sign-up; wire it behind a new
POST /signup route if you want self-service account creation. The frontend's
login/signup pages currently call the Supabase JS client directly, so they
would also need to be pointed at your own backend routes instead -- that is
a separate change from this file.

Required environment variables (see backend/.env.example for the Supabase
equivalents this replaces):

    MSSQL_SERVER                e.g. localhost or myserver.database.windows.net
    MSSQL_DATABASE              e.g. scrapper
    MSSQL_USER                  SQL login (omit if using MSSQL_TRUSTED_CONNECTION)
    MSSQL_PASSWORD              SQL login password (omit if using MSSQL_TRUSTED_CONNECTION)
    MSSQL_TRUSTED_CONNECTION    "true" to use Windows auth instead of MSSQL_USER/MSSQL_PASSWORD
    MSSQL_DRIVER                default "ODBC Driver 17 for SQL Server"
    JWT_SECRET                  random long string used to sign access/refresh tokens
    JWT_ACCESS_TOKEN_EXPIRES_MINUTES   default 60
    JWT_REFRESH_TOKEN_EXPIRES_DAYS     default 30

Requires the "pyodbc" package and a SQL Server ODBC driver installed on the
host (see database/mssql_schema.sql for the matching table definitions).
"""

import hashlib
import hmac
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
import pyodbc
from fastapi import HTTPException, status

from app.core.config import settings

PBKDF2_ITERATIONS = 390_000


@dataclass(frozen=True)
class AuthSessionTokens:
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int | None = None


# --------------------------------------------------------------------------
# Connection handling
# --------------------------------------------------------------------------


def _require_mssql_config() -> None:
    if not settings.mssql_server or not settings.mssql_database:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SQL Server host or database is not configured.",
        )
    if not settings.mssql_trusted_connection and not (
        settings.mssql_user and settings.mssql_password
    ):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SQL Server credentials are not configured.",
        )
    if not settings.jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT_SECRET is not configured.",
        )


def get_db_connection() -> "pyodbc.Connection":
    _require_mssql_config()
    if settings.mssql_trusted_connection:
        conn_str = (
            f"DRIVER={{{settings.mssql_driver}}};"
            f"SERVER={settings.mssql_server};"
            f"DATABASE={settings.mssql_database};"
            "Trusted_Connection=yes;"
            "Encrypt=yes;TrustServerCertificate=yes;"
        )
    else:
        conn_str = (
            f"DRIVER={{{settings.mssql_driver}}};"
            f"SERVER={settings.mssql_server};"
            f"DATABASE={settings.mssql_database};"
            f"UID={settings.mssql_user};PWD={settings.mssql_password};"
            "Encrypt=yes;TrustServerCertificate=yes;"
        )
    try:
        return pyodbc.connect(conn_str)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not connect to SQL Server: {exc}",
        ) from exc


def _row_to_dict(
    columns: list[str],
    row: Any,
    json_columns: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for col, value in zip(columns, row):
        if isinstance(value, uuid.UUID):
            value = str(value)
        elif isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            value = value.isoformat()
        elif col in json_columns and value is not None:
            value = json.loads(value)
        record[col] = value
    return record


def _rows_to_dicts(
    columns: list[str],
    rows: list[Any],
    json_columns: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    return [_row_to_dict(columns, row, json_columns) for row in rows]


# --------------------------------------------------------------------------
# Password hashing (stdlib PBKDF2-SHA256, no extra dependency)
# --------------------------------------------------------------------------


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${derived.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations, salt_hex, hash_hex = password_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        return False
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, int(iterations)
    )
    return hmac.compare_digest(derived, expected)


# --------------------------------------------------------------------------
# JWT session tokens
# --------------------------------------------------------------------------


def _invalid_login_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _invalid_refresh_token_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired refresh token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _create_token(user_id: str, token_type: str, expires_delta: timedelta) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def _issue_session_tokens(user_id: str) -> AuthSessionTokens:
    access_minutes = settings.jwt_access_token_expires_minutes
    refresh_days = settings.jwt_refresh_token_expires_days
    access_token = _create_token(
        user_id, "access", timedelta(minutes=access_minutes)
    )
    refresh_token = _create_token(user_id, "refresh", timedelta(days=refresh_days))
    return AuthSessionTokens(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=access_minutes * 60,
    )


def _user_exists(user_id: str) -> bool:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM dbo.users WHERE id = ?", user_id)
        return cursor.fetchone() is not None
    except Exception:
        return False
    finally:
        conn.close()


def authenticate_user_with_password(email: str, password: str) -> AuthSessionTokens:
    """Verify email/password against dbo.users and issue a new session."""
    if not email or not password:
        raise _invalid_login_exception()

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, password_hash FROM dbo.users WHERE email = ?",
            email.strip().lower(),
        )
        row = cursor.fetchone()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authentication service is unavailable.",
        ) from exc
    finally:
        conn.close()

    if not row or not verify_password(password, row.password_hash):
        raise _invalid_login_exception()

    return _issue_session_tokens(str(row.id))


def refresh_access_token(refresh_token: str) -> AuthSessionTokens:
    """Exchange a refresh token for a newly rotated session."""
    if not refresh_token:
        raise _invalid_refresh_token_exception()

    try:
        payload = jwt.decode(refresh_token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise _invalid_refresh_token_exception() from exc

    if payload.get("type") != "refresh":
        raise _invalid_refresh_token_exception()

    user_id = payload.get("sub")
    if not user_id or not _user_exists(user_id):
        raise _invalid_refresh_token_exception()

    return _issue_session_tokens(user_id)


def get_user_id_from_access_token(access_token: str) -> str:
    try:
        payload = jwt.decode(access_token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
        ) from exc

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token is missing a user id.",
        )
    return user_id


def register_user(email: str, password: str) -> AuthSessionTokens:
    """Create a new dbo.users row and return a session for it.

    Not wired to a route by default -- add a POST /signup endpoint that
    calls this if you want self-service sign-up without Supabase Auth.
    """
    email = (email or "").strip().lower()
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")
    if len(password) < 6:
        raise HTTPException(
            status_code=400, detail="Password must be at least 6 characters."
        )

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM dbo.users WHERE email = ?", email)
        if cursor.fetchone():
            raise HTTPException(
                status_code=409, detail="An account with this email already exists."
            )
        user_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO dbo.users (id, email, password_hash) VALUES (?, ?, ?)",
            user_id,
            email,
            hash_password(password),
        )
        conn.commit()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to create the account: {exc}",
        ) from exc
    finally:
        conn.close()

    return _issue_session_tokens(user_id)


# --------------------------------------------------------------------------
# Scrape jobs
# --------------------------------------------------------------------------


def insert_scrape_job(
    *,
    url: str,
    scraping_method: str,
    selected_page_count: int,
    use_ai: bool,
) -> dict[str, Any]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO dbo.scrape_jobs
                (url, scraping_method, selected_page_count, use_ai, status)
            OUTPUT inserted.id, inserted.url, inserted.scraping_method,
                   inserted.status, inserted.use_ai, inserted.selected_page_count,
                   inserted.failure_reason, inserted.scraped_text, inserted.ai_output,
                   inserted.created_at, inserted.completed_at
            VALUES (?, ?, ?, ?, 'running')
            """,
            url,
            scraping_method,
            selected_page_count,
            use_ai,
        )
        row = cursor.fetchone()
        columns = [col[0] for col in cursor.description]
        conn.commit()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to save scrape job to SQL Server: {exc}",
        ) from exc
    finally:
        conn.close()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="SQL Server did not return the saved scrape job.",
        )
    return _row_to_dict(columns, row, json_columns=frozenset({"ai_output"}))


def update_scrape_job_completed(
    *,
    scrape_job_id: str,
    scraped_text: str,
    ai_output: dict[str, Any] | None,
) -> None:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE dbo.scrape_jobs
            SET status = 'completed',
                scraped_text = ?,
                ai_output = ?,
                failure_reason = NULL,
                completed_at = SYSUTCDATETIME()
            WHERE id = ?
            """,
            scraped_text,
            json.dumps(ai_output) if ai_output is not None else None,
            scrape_job_id,
        )
        conn.commit()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to mark scrape job as completed in SQL Server.",
        ) from exc
    finally:
        conn.close()


def update_scrape_job_failed(*, scrape_job_id: str, failure_reason: str) -> None:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE dbo.scrape_jobs
            SET status = 'failed',
                failure_reason = ?,
                completed_at = SYSUTCDATETIME()
            WHERE id = ?
            """,
            failure_reason[:4000],
            scrape_job_id,
        )
        conn.commit()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to mark scrape job as failed in SQL Server.",
        ) from exc
    finally:
        conn.close()


def list_scrape_jobs_for_user(*, limit: int = 50) -> list[dict[str, Any]]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT TOP (?) id, url, scraping_method, status, use_ai,
                   selected_page_count, failure_reason, scraped_text, ai_output,
                   created_at, completed_at
            FROM dbo.scrape_jobs
            ORDER BY created_at DESC
            """,
            limit,
        )
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to load scrape history from SQL Server.",
        ) from exc
    finally:
        conn.close()

    return _rows_to_dicts(columns, rows, json_columns=frozenset({"ai_output"}))


# --------------------------------------------------------------------------
# API cost logs
# --------------------------------------------------------------------------


def insert_api_cost_log(
    *,
    scrape_job_id: str,
    provider: str,
    model: str,
    usage: dict[str, Any],
) -> dict[str, Any] | None:
    input_cost = (usage.get("input_cost_usd", 0) or 0) + (
        usage.get("cached_input_cost_usd", 0) or 0
    )
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO dbo.api_cost_logs
                (scrape_job_id, provider, model, input_tokens, output_tokens,
                 total_tokens, input_cost_usd, output_cost_usd, total_cost_usd)
            OUTPUT inserted.id, inserted.scrape_job_id, inserted.provider,
                   inserted.model, inserted.input_tokens, inserted.output_tokens,
                   inserted.total_tokens, inserted.input_cost_usd, inserted.output_cost_usd,
                   inserted.total_cost_usd, inserted.created_at
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            scrape_job_id,
            provider,
            model,
            usage.get("input_tokens", 0) or 0,
            usage.get("output_tokens", 0) or 0,
            usage.get("total_tokens", 0) or 0,
            round(input_cost, 8),
            usage.get("output_cost_usd", 0) or 0,
            usage.get("estimated_total_cost_usd", 0) or 0,
        )
        row = cursor.fetchone()
        columns = [col[0] for col in cursor.description]
        conn.commit()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to save API cost log to SQL Server.",
        ) from exc
    finally:
        conn.close()

    return _row_to_dict(columns, row) if row else None


def list_api_cost_logs_for_user(*, limit: int = 50) -> list[dict[str, Any]]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT TOP (?) id, scrape_job_id, provider, model, input_tokens,
                   output_tokens, total_tokens, input_cost_usd, output_cost_usd,
                   total_cost_usd, created_at
            FROM dbo.api_cost_logs
            ORDER BY created_at DESC
            """,
            limit,
        )
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to load API cost logs from SQL Server.",
        ) from exc
    finally:
        conn.close()

    return _rows_to_dicts(columns, rows)


# --------------------------------------------------------------------------
# Currency rates
# --------------------------------------------------------------------------


def get_currency_rate(*, base_currency: str, quote_currency: str) -> dict[str, Any]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT base_currency, quote_currency, rate, updated_at
            FROM dbo.currency_rates
            WHERE base_currency = ? AND quote_currency = ?
            """,
            base_currency,
            quote_currency,
        )
        row = cursor.fetchone()
        columns = [col[0] for col in cursor.description]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to load the currency rate from SQL Server.",
        ) from exc
    finally:
        conn.close()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"The {base_currency}/{quote_currency} currency rate is not configured.",
        )
    return _row_to_dict(columns, row)


def upsert_currency_rate(
    *,
    base_currency: str,
    quote_currency: str,
    rate: float,
    updated_by: str,
) -> dict[str, Any]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            MERGE dbo.currency_rates AS target
            USING (SELECT ? AS base_currency, ? AS quote_currency) AS src
                ON target.base_currency = src.base_currency
               AND target.quote_currency = src.quote_currency
            WHEN MATCHED THEN
                UPDATE SET rate = ?, updated_at = SYSUTCDATETIME(), updated_by = ?
            WHEN NOT MATCHED THEN
                INSERT (base_currency, quote_currency, rate, updated_at, updated_by)
                VALUES (?, ?, ?, SYSUTCDATETIME(), ?);
            """,
            base_currency,
            quote_currency,
            rate,
            updated_by,
            base_currency,
            quote_currency,
            rate,
            updated_by,
        )
        conn.commit()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to save the currency rate to SQL Server.",
        ) from exc
    finally:
        conn.close()

    return get_currency_rate(base_currency=base_currency, quote_currency=quote_currency)
