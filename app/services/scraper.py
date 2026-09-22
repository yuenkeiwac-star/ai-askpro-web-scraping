import logging
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

SCRAPING_METHOD = "Playwright (Chromium)"

SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".css", ".js", ".zip", ".rar", ".7z", ".mp4", ".mp3", ".avi",
    ".mov", ".wmv", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
)
DOWNLOAD_EXTENSIONS = (".pdf", ".csv", ".txt")
SENSITIVE_FIELD_TYPES = {"password", "hidden"}
SENSITIVE_PATTERNS = [
    re.compile(r"(?i)\b(password|passcode|api[_ -]?key|secret|token)\b\s*[:=]\s*\S+"),
    re.compile(r"(?i)\b(session|auth)[_ -]?(cookie|token)\b\s*[:=]\s*\S+"),
]
LOGIN_KEYWORDS = (
    "login", "log in", "sign in", "signin", "member login", "customer login",
    "portal", "username", "password", "forgot password",
)
AUTH_PATH_SEGMENTS = {
    "auth", "authenticate", "login", "log-in", "signin", "sign-in", "sso",
}
RESTRICTED_PAGE_PHRASES = (
    "access denied",
    "authentication required",
    "please log in to continue",
    "please login to continue",
    "please sign in to continue",
    "you must be logged in",
    "you must sign in",
)
MOJIBAKE_REPLACEMENTS = {
    "â€™": "’",
    "â€˜": "‘",
    "â€œ": "“",
    "â€": "”",
    "â€“": "–",
    "â€”": "—",
    "â€¦": "…",
    "â€‹": "",
    "Â©": "©",
    "Â®": "®",
    "Â ": " ",
    "ï»¿": "",
}


def repair_mojibake(text: str) -> str:
    repaired = text
    for corrupted, replacement in MOJIBAKE_REPLACEMENTS.items():
        repaired = repaired.replace(corrupted, replacement)
    return repaired


def normalize_url(base_url: str, href: str | None) -> str | None:
    if not href:
        return None
    href = href.strip()
    if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
        return None
    normalized = urljoin(base_url, href)
    parsed = urlparse(normalized)
    if parsed.scheme not in ("http", "https"):
        return None
    return parsed._replace(fragment="").geturl()


def same_domain(base_url: str, target_url: str) -> bool:
    return urlparse(base_url).netloc.lower() == urlparse(target_url).netloc.lower()


def is_download_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(DOWNLOAD_EXTENSIONS)


def is_scrapable_page_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return not path.endswith(SKIP_EXTENSIONS) and not is_download_url(url)


def dedupe_links(links: list[dict]) -> list[dict]:
    seen = set()
    output = []
    for link in links:
        url = link.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        output.append(link)
    return output


def clean_lines(text: str, keep_duplicates: bool = False) -> str:
    lines = []
    seen = set()
    for line in text.splitlines():
        line = repair_mojibake(line)
        line = " ".join(line.strip().split())
        if not line:
            continue
        if keep_duplicates:
            lines.append(line)
            continue
        key = line.lower()
        if key not in seen:
            seen.add(key)
            lines.append(line)
    return "\n".join(lines)


def redact_sensitive_text(text: str) -> str:
    redacted = text
    for pattern in SENSITIVE_PATTERNS:
        redacted = pattern.sub("[REDACTED_SENSITIVE_VALUE]", redacted)
    return redacted


def html_to_text(html: str, include_header_footer: bool = True, keep_duplicates: bool = True) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "canvas"]):
        tag.decompose()
    if not include_header_footer:
        for selector in ["header", "nav", "footer"]:
            for tag in soup.select(selector):
                tag.decompose()
    return redact_sensitive_text(clean_lines(soup.get_text("\n", strip=True), keep_duplicates=keep_duplicates))


def extract_links_from_html(base_url: str, html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for anchor in soup.find_all("a"):
        href = normalize_url(base_url, anchor.get("href"))
        if not href:
            continue
        links.append({
            "text": repair_mojibake(
                anchor.get_text(" ", strip=True)
            ) or "Untitled link",
            "url": href,
            "same_domain": same_domain(base_url, href),
            "is_download": is_download_url(href),
        })
    return dedupe_links(links)


def extract_sections_from_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "canvas"]):
        tag.decompose()
    sections = []
    current = None
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = repair_mojibake(tag.get_text(" ", strip=True))
        if not text:
            continue
        if tag.name in ["h1", "h2", "h3", "h4"]:
            current = {"heading": redact_sensitive_text(text), "content": []}
            sections.append(current)
        elif current:
            current["content"].append(redact_sensitive_text(text))
    return sections


def has_password_field(forms: list[dict]) -> bool:
    return any(
        any((field.get("type") or "").lower() == "password" for field in form.get("inputs", []))
        for form in forms
    )


def has_login_page_signals(url: str, title: str, text: str, forms: list[dict]) -> bool:
    """Return broad signals used to decide whether an AI review is worthwhile."""
    combined = f"{url} {title} {text}".lower()
    return any(keyword in combined for keyword in LOGIN_KEYWORDS) or has_password_field(forms)


def is_possible_login_page(url: str, title: str, text: str, forms: list[dict]) -> bool:
    """Conservative non-AI fallback for pages that are clearly access gates.

    A navigation link such as "Sign in" is deliberately not enough. That text is
    common on public product pages and caused the previous false positives.
    """
    path_segments = {
        segment.lower()
        for segment in urlparse(url).path.split("/")
        if segment
    }
    normalized_title = " ".join(title.lower().split())
    normalized_text = " ".join(text[:8_000].lower().split())
    auth_focused_url = bool(path_segments & AUTH_PATH_SEGMENTS)
    auth_focused_title = any(
        phrase in normalized_title
        for phrase in ("login", "log in", "sign in", "signin", "authentication required")
    )
    explicitly_restricted = any(
        phrase in normalized_text for phrase in RESTRICTED_PAGE_PHRASES
    )

    return explicitly_restricted or (
        has_password_field(forms) and (auth_focused_url or auth_focused_title)
    )


def sanitize_forms(forms: list[dict]) -> list[dict]:
    safe_forms = []
    for form in forms:
        safe_inputs = []
        for field in form.get("inputs", []):
            field_type = (field.get("type") or "").lower()
            if field_type in SENSITIVE_FIELD_TYPES:
                continue
            safe_inputs.append({
                "tag": field.get("tag", ""),
                "type": field_type,
                "label": repair_mojibake(field.get("label", "")),
                "placeholder": repair_mojibake(field.get("placeholder", "")),
                "required": field.get("required", False),
            })
        safe_forms.append({
            "form_index": form.get("form_index"),
            "method": form.get("method", "GET"),
            "inputs": safe_inputs,
            "buttons": [
                {
                    "text": repair_mojibake(
                        button.get("text", "") if isinstance(button, dict) else button
                    ),
                    "type": button.get("type", "") if isinstance(button, dict) else "",
                }
                for button in form.get("buttons", [])
            ],
        })
    return safe_forms


async def extract_form_details(page) -> list[dict]:
    return await page.evaluate("""
    () => {
        const forms = Array.from(document.querySelectorAll("form"));
        return forms.map((form, formIndex) => {
            const inputs = Array.from(form.querySelectorAll("input, textarea, select"))
                .map(input => {
                    let labelText = "";
                    if (input.id) {
                        const safeId = CSS.escape(input.id);
                        const label = document.querySelector(`label[for="${safeId}"]`);
                        if (label) labelText = label.innerText.trim();
                    }
                    const parentLabel = input.closest("label");
                    if (!labelText && parentLabel) {
                        labelText = parentLabel.innerText.trim();
                    }
                    return {
                        tag: input.tagName.toLowerCase(),
                        type: input.getAttribute("type") || "",
                        name: input.getAttribute("name") || "",
                        id: input.getAttribute("id") || "",
                        label: labelText,
                        placeholder: input.getAttribute("placeholder") || "",
                        autocomplete: input.getAttribute("autocomplete") || "",
                        required: input.required || false
                    };
                });
            const buttons = Array.from(form.querySelectorAll("button, input[type='submit'], input[type='button']"))
                .map(btn => ({
                    text: btn.innerText || btn.value || "",
                    type: btn.getAttribute("type") || ""
                }));
            return {
                form_index: formIndex + 1,
                action: form.getAttribute("action") || "",
                method: form.getAttribute("method") || "GET",
                inputs: inputs,
                buttons: buttons
            };
        });
    }
    """)


logger = logging.getLogger(__name__)


async def scrape_page(page_url: str, include_header_footer: bool = True, keep_duplicates: bool = True) -> dict:
    from playwright.async_api import async_playwright

    logger.info("scrape_page start url=%s", page_url)
    started = time.perf_counter()
    title = ""
    final_url = page_url
    html = ""
    visible_text = ""
    forms = []
    status_code = None
    error = ""
    browser = None

    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": 1366, "height": 1400})
            try:
                try:
                    response = await page.goto(page_url, wait_until="networkidle", timeout=60000)
                except Exception:
                    response = await page.goto(page_url, wait_until="domcontentloaded", timeout=60000)

                for text in [
                    "OUR EXPERTISE", "Our Expertise", "SERVICES", "Services", "SOLUTIONS", "Solutions",
                    "PRODUCTS", "Products", "ABOUT", "About", "CONTACT", "Contact", "BLOG", "Blog",
                    "CAREERS", "Careers",
                ]:
                    try:
                        await page.get_by_text(text, exact=False).first.hover(timeout=1000)
                        await page.wait_for_timeout(250)
                    except Exception:
                        pass

                for _ in range(12):
                    await page.mouse.wheel(0, 1000)
                    await page.wait_for_timeout(250)

                title = repair_mojibake(await page.title())
                final_url = page.url
                html = await page.content()
                try:
                    visible_text = await page.locator("body").inner_text(timeout=10000)
                except Exception:
                    visible_text = ""
                forms = await extract_form_details(page)
                status_code = response.status if response else None
            finally:
                await browser.close()
    except Exception as exc:
        error = (
            "Playwright could not start or scrape the page: "
            f"{type(exc).__name__}: {exc!r}"
        )
        logger.exception("scrape_page failed url=%s", page_url)

    login_page_candidate = has_login_page_signals(final_url, title, visible_text, forms)
    possible_login_page = (
        status_code in {401, 403}
        or is_possible_login_page(final_url, title, visible_text, forms)
    )
    all_links = extract_links_from_html(final_url, html) if html else []
    internal_links = [
        link for link in all_links
        if link.get("same_domain") and is_scrapable_page_url(link.get("url", ""))
    ]
    download_links = [link for link in all_links if link.get("is_download")]
    cleaned_text = html_to_text(html, include_header_footer, keep_duplicates) if html else ""
    visible_text = redact_sensitive_text(clean_lines(visible_text, keep_duplicates=keep_duplicates))

    duration = round(time.perf_counter() - started, 3)
    if error:
        logger.warning("scrape_page done url=%s status_code=%s duration=%.3fs error=%s", page_url, status_code, duration, error)
    else:
        logger.info("scrape_page done url=%s status_code=%s duration=%.3fs", page_url, status_code, duration)

    return {
        "input_url": page_url,
        "final_url": final_url,
        "title": title,
        "status_code": status_code,
        "error": error,
        "text": cleaned_text,
        "visible_text": visible_text,
        "sections": extract_sections_from_html(html) if html else [],
        "links": all_links,
        "internal_links": internal_links,
        "download_links": download_links,
        "forms": sanitize_forms(forms),
        "login_page_candidate": login_page_candidate,
        "has_password_field": has_password_field(forms),
        "possible_login_page": possible_login_page,
        "scrape_seconds": duration,
    }


async def whole_site_crawl(
    start_url: str,
    max_pages: int,
    max_depth: int = 1,
    include_header_footer: bool = True,
    keep_duplicates: bool = True,
) -> list[dict]:
    results = []
    visited = set()
    queue = deque([(start_url, 0)])

    while queue and len(results) < max_pages:
        page_url, depth = queue.popleft()
        if page_url in visited or not same_domain(start_url, page_url) or not is_scrapable_page_url(page_url):
            continue
        visited.add(page_url)

        page_data = await scrape_page(page_url, include_header_footer, keep_duplicates)
        final_url = page_data.get("final_url") or page_url
        visited.add(final_url)
        page_data["crawl_depth"] = depth
        results.append(page_data)

        if page_data.get("possible_login_page"):
            continue

        if depth < max_depth:
            for link in page_data.get("internal_links", []):
                next_url = link.get("url", "")
                if next_url and next_url not in visited and same_domain(start_url, next_url):
                    queue.append((next_url, depth + 1))

    return results
