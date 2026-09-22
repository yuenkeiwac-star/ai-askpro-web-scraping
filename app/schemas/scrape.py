from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class ScrapeRequest(BaseModel):
    url: HttpUrl
    ai_provider: Literal["openai", "gemini"] = "openai"
    ai_model: str = Field(default="gpt-4o-mini", min_length=1)
    extraction_mode: Literal["records", "full_page"] = "records"
    selected_urls: list[HttpUrl] = Field(min_length=1, max_length=50)
    use_ai: bool = True
    task: str = Field(
        default=(
            "Goal: Extract the website's public offerings and contact resources.\n"
            "One record per: service or product.\n"
            "Fields for each record: name, category, description, price (when shown), availability, "
            "key features, public contact details, source URL, and supporting quote.\n"
            "Include: only facts shown on the selected public pages. Also include relevant downloads, "
            "useful public links, and public forms.\n"
            "Exclude: navigation, cookie notices, duplicate content, and unrelated information."
        ),
        min_length=0,
        max_length=4000,
    )

    @model_validator(mode="after")
    def require_task_for_record_extraction(self):
        if self.extraction_mode == "full_page" and not self.use_ai:
            raise ValueError("Full page JSON requires AI extraction")
        if self.use_ai and self.extraction_mode == "records" and not self.task.strip():
            raise ValueError("task is required for AI record extraction")
        return self


class DiscoverLinksRequest(BaseModel):
    url: HttpUrl


class DiscoveredLink(BaseModel):
    text: str
    url: str


class DiscoverLinksResponse(BaseModel):
    start_url: str
    links: list[DiscoveredLink]
    warning: str = ""


class DynamicExtractionPage(BaseModel):
    """The only fixed fields for one AI-detected page."""

    model_config = ConfigDict(extra="forbid")

    title: str
    content: dict[str, Any]


class DynamicExtractionOutput(BaseModel):
    """Minimal envelope around content whose keys are selected by the model."""

    model_config = ConfigDict(extra="forbid")

    pages: list[DynamicExtractionPage]
    unclassified_content: list[Any]
    metadata: dict[str, Any]


class ScrapeResponse(BaseModel):
    scraping_method: str
    extraction_mode: Literal["text", "records", "full_page"]
    scraped_text: str
    ai_json_output: DynamicExtractionOutput | dict[str, Any] | None
    ai_provider: str
    ai_model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_total_cost: float
    pages_scraped: int
    warnings: list[str] = Field(default_factory=list)


class ScrapeJobRecord(BaseModel):
    id: str
    url: str
    scraping_method: str
    status: str
    use_ai: bool
    selected_page_count: int
    failure_reason: str | None = None
    scraped_text: str | None = None
    ai_output: dict[str, Any] | None = None
    created_at: str
    completed_at: str | None = None


class ApiCostLogRecord(BaseModel):
    id: str
    scrape_job_id: str | None = None
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    created_at: str
