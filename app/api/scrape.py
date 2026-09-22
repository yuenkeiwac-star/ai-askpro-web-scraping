import json
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import settings
from app.core.auth import AuthenticatedUser, require_authenticated_user
from app.schemas.scrape import (
    DiscoverLinksRequest,
    DiscoverLinksResponse,
    ScrapeRequest,
    ScrapeResponse,
)
# /scrape-history and /api-cost-logs are disabled below -- uncomment with them.
# from app.schemas.scrape import ApiCostLogRecord, ScrapeJobRecord
from app.services.ai import (
    ai_classify_page_access,
    ai_extract_from_scraped_pages,
    ai_extract_full_page_json,
)
from app.services.cost_calculator import combine_usage
from app.services.full_page_json import join_page_text
from app.services.scraper import (
    SCRAPING_METHOD,
    is_scrapable_page_url,
    same_domain,
    scrape_page,
)
# Scrape job / API cost persistence is disabled -- /scrape already returns
# token usage and cost directly in ScrapeResponse, and /scrape-history and
# /api-cost-logs are disabled below since nothing writes to those tables
# anymore. Uncomment these and the matching code below to bring it all back.
# from app.services.database_service import (
#     insert_api_cost_log,
#     insert_scrape_job,
#     list_api_cost_logs_for_user,
#     list_scrape_jobs_for_user,
#     update_scrape_job_completed,
#     update_scrape_job_failed,
# )

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/discover-links", response_model=DiscoverLinksResponse)
async def discover_links(
    request: DiscoverLinksRequest,
    _user: AuthenticatedUser = Depends(require_authenticated_user),
) -> DiscoverLinksResponse:
    start_url = str(request.url)
    logger.info("discover_links start url=%s", start_url)
    page = await scrape_page(start_url)
    if page.get("error"):
        logger.error("discover_links scrape failed url=%s error=%s", start_url, page["error"])
        raise HTTPException(status_code=502, detail=page["error"])
    if page.get("possible_login_page"):
        logger.warning("discover_links blocked possible login page url=%s", start_url)
        raise HTTPException(status_code=400, detail="The URL appears to be a login or restricted page.")

    links = [
        {
            "text": page.get("title") or "Starting page",
            "url": page.get("final_url") or start_url,
        }
    ]
    links.extend(page.get("internal_links", []))
    unique_links = []
    seen = set()
    for link in links:
        link_url = link.get("url", "")
        if link_url and link_url not in seen and is_scrapable_page_url(link_url):
            seen.add(link_url)
            unique_links.append(
                {"text": link.get("text") or "Untitled page", "url": link_url}
            )

    logger.info("discover_links done url=%s links_found=%d", start_url, len(unique_links))
    return DiscoverLinksResponse(
        start_url=page.get("final_url") or start_url,
        links=unique_links,
        warning="" if unique_links else "No public same-domain page links were found.",
    )


@router.post("/scrape", response_model=ScrapeResponse)
async def scrape(
    request: ScrapeRequest,
    _user: AuthenticatedUser = Depends(require_authenticated_user),
) -> ScrapeResponse:
    provider = request.ai_provider.lower()
    url = str(request.url)
    selected_urls = [str(selected_url) for selected_url in request.selected_urls]
    logger.info(
        "scrape start url=%s pages=%d provider=%s model=%s extraction_mode=%s use_ai=%s",
        url,
        len(selected_urls),
        provider,
        request.ai_model,
        request.extraction_mode,
        request.use_ai,
    )
    # scrape_job = insert_scrape_job(
    #     url=url,
    #     scraping_method=SCRAPING_METHOD,
    #     selected_page_count=len(selected_urls),
    #     use_ai=request.use_ai,
    # )

    try:
        api_key = settings.gemini_api_key if provider == "gemini" else settings.openai_api_key
        if request.use_ai and not api_key:
            key_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
            raise HTTPException(status_code=500, detail=f"{key_name} is not configured.")

        if any(not same_domain(url, selected_url) for selected_url in selected_urls):
            raise HTTPException(
                status_code=400,
                detail="Every selected URL must use the same domain as the starting URL.",
            )
        results = [await scrape_page(selected_url) for selected_url in selected_urls]
        scrape_errors = [result.get("error") for result in results if result.get("error")]
        if scrape_errors:
            logger.error("scrape page fetch failed url=%s error=%s", url, scrape_errors[0])
            raise HTTPException(status_code=502, detail=str(scrape_errors[0]))
        logger.info("scrape pages fetched url=%s count=%d", url, len(results))

        warnings = []
        classification_usage = None
        if request.use_ai:
            logger.info("scrape ai_classify_page_access start url=%s", url)
            classification = ai_classify_page_access(
                results,
                api_key,
                request.ai_model,
                provider,
            )
            if classification.get("error"):
                logger.warning(
                    "scrape ai_classify_page_access failed url=%s error=%s",
                    url,
                    classification["error"],
                )
                warnings.append(
                    "The AI page-access check failed, so conservative local login-page "
                    "detection was used."
                )
            else:
                candidate_indices = {
                    index
                    for index, result in enumerate(results)
                    if result.get("login_page_candidate") or result.get("possible_login_page")
                }
                for decision in classification.get("data", {}).get("decisions", []):
                    page_index = decision.get("page_index")
                    is_restricted = decision.get("is_restricted")
                    if page_index in candidate_indices and isinstance(is_restricted, bool):
                        results[page_index]["possible_login_page"] = is_restricted
                        results[page_index]["access_classification_reason"] = decision.get(
                            "reason", ""
                        )
                classification_usage = classification.get("usage")

        public_results = [result for result in results if not result.get("possible_login_page")]
        skipped_login_pages = len(results) - len(public_results)
        if skipped_login_pages:
            warnings.append(
                f"Skipped {skipped_login_pages} possible login or restricted page(s). "
                "No credentials or private account data were collected."
            )

        scraped_text = join_page_text(public_results)
        usage = {}
        ai_output = None
        if request.use_ai:
            logger.info(
                "scrape ai_extract start url=%s extraction_mode=%s model=%s",
                url,
                request.extraction_mode,
                request.ai_model,
            )
            if request.extraction_mode == "full_page":
                extraction = ai_extract_full_page_json(
                    public_results,
                    api_key,
                    request.ai_model,
                    provider,
                )
            else:
                extraction = ai_extract_from_scraped_pages(
                    public_results,
                    request.task,
                    api_key,
                    request.ai_model,
                    provider,
                )
            if extraction.get("error"):
                logger.error("scrape ai_extract failed url=%s error=%s", url, extraction["error"])
                raise HTTPException(status_code=502, detail=extraction["error"])
            usage = combine_usage([
                classification_usage,
                extraction.get("usage"),
            ]) or {}
            ai_output = extraction.get("data")
            logger.info(
                "scrape ai_extract done url=%s total_tokens=%s estimated_cost_usd=%s",
                url,
                usage.get("total_tokens", 0),
                usage.get("estimated_total_cost_usd", 0),
            )

            # if usage.get("total_tokens", 0) > 0:
            #     insert_api_cost_log(
            #         scrape_job_id=scrape_job["id"],
            #         provider=usage.get("provider", provider.title()) or provider.title(),
            #         model=usage.get("model_used", request.ai_model) or request.ai_model,
            #         usage=usage,
            #     )

        # update_scrape_job_completed(
        #     scrape_job_id=scrape_job["id"],
        #     scraped_text=scraped_text,
        #     ai_output=ai_output,
        # )
    except Exception as exc:
        reason = str(exc.detail) if isinstance(exc, HTTPException) else f"{type(exc).__name__}: {exc}"
        # try:
        #     update_scrape_job_failed(scrape_job_id=scrape_job["id"], failure_reason=reason)
        # except HTTPException:
        #     pass
        if isinstance(exc, HTTPException):
            logger.error("scrape failed url=%s status=%s reason=%s", url, exc.status_code, reason)
            raise
        logger.exception("scrape failed unexpectedly url=%s reason=%s", url, reason)
        raise HTTPException(status_code=500, detail=reason) from exc

    logger.info(
        "scrape done url=%s pages_scraped=%d total_tokens=%s estimated_cost_usd=%s warnings=%d",
        url,
        len(public_results),
        usage.get("total_tokens", 0),
        usage.get("estimated_total_cost_usd", 0),
        len(warnings),
    )
    if ai_output is not None:
        logger.info(
            "scrape final ai_json_output url=%s output=%s",
            url,
            json.dumps(ai_output, ensure_ascii=False),
        )
    return ScrapeResponse(
        scraping_method=SCRAPING_METHOD,
        extraction_mode=(request.extraction_mode if request.use_ai else "text"),
        scraped_text=scraped_text,
        ai_json_output=ai_output,
        ai_provider=(
            usage.get("provider", provider.title()) or provider.title()
            if request.use_ai
            else "Disabled"
        ),
        ai_model=(
            usage.get("model_used", request.ai_model)
            if request.use_ai
            else "Text only"
        ),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        estimated_total_cost=usage.get("estimated_total_cost_usd", 0),
        pages_scraped=len(public_results),
        warnings=warnings,
    )


# Disabled -- nothing writes to scrape_jobs/api_cost_logs anymore (see the
# commented-out persistence calls in scrape() above), so these would always
# return an empty list. Uncomment along with the imports above to bring back.
# @router.get("/scrape-history", response_model=list[ScrapeJobRecord])
# async def scrape_history(
#     _user: AuthenticatedUser = Depends(require_authenticated_user),
#     limit: int = Query(default=50, ge=1, le=100),
# ) -> list[ScrapeJobRecord]:
#     return list_scrape_jobs_for_user(limit=limit)
#
#
# @router.get("/api-cost-logs", response_model=list[ApiCostLogRecord])
# async def api_cost_logs(
#     _user: AuthenticatedUser = Depends(require_authenticated_user),
#     limit: int = Query(default=50, ge=1, le=100),
# ) -> list[ApiCostLogRecord]:
#     return list_api_cost_logs_for_user(limit=limit)
