import asyncio
import logging
import os
import sys
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Playwright launches the browser via an asyncio subprocess. On Windows, the
# default event loop uvicorn ends up using (especially under --reload)
# doesn't support subprocess creation and raises NotImplementedError -- the
# Proactor loop is required for that. Has no effect on Linux/macOS.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from app.core.logging_config import API_LOGGER_NAME, setup_logging

setup_logging()
api_logger = logging.getLogger(API_LOGGER_NAME)

# /token and /token/refresh are unused -- there is no user model left
# (authentication is disabled in app/core/auth.py, and nothing tracks a
# user_id anymore). Uncomment to bring login back.
# from app.api.auth import router as auth_router
# Currency-rate conversion is unused -- scrape costs are already returned in
# USD by app.services.cost_calculator. Uncomment to bring the feature back.
# from app.api.currency import router as currency_router
from app.api.scrape import router as scrape_router
from app.core.config import settings

web_scraper = FastAPI(
    title="AI Website Scraper API",
    swagger_ui_parameters={"persistAuthorization": True},
)

web_scraper.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.frontend_origins.split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_scraper.middleware("http")
async def log_requests(request: Request, call_next):
    pid = os.getpid()
    full_url = str(request.url)
    api_logger.info("[%s] START %s %s", pid, request.method, full_url)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration = time.perf_counter() - started
        api_logger.exception(
            "[%s] END %s status=500 duration=%.2fs (unhandled exception)",
            pid,
            full_url,
            duration,
        )
        raise
    duration = time.perf_counter() - started
    api_logger.info(
        "[%s] END %s status=%s duration=%.2fs",
        pid,
        full_url,
        response.status_code,
        duration,
    )
    return response


# app.include_router(auth_router)
# app.include_router(currency_router)
web_scraper.include_router(scrape_router)


@web_scraper.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
