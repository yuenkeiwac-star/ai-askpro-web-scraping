from typing import Any


PAGE_SEPARATOR = "\n\n"


def _collect_page_texts(
    results: list[dict[str, Any]],
) -> list[tuple[int, str]]:
    return [
        (page_index, text)
        for page_index, result in enumerate(results)
        if isinstance((text := result.get("text")), str) and text
    ]


def join_page_text(results: list[dict[str, Any]]) -> str:
    return PAGE_SEPARATOR.join(
        text for _, text in _collect_page_texts(results)
    )
