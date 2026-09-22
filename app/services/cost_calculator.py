OPENAI_PRICING = {
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
}

GEMINI_PRICING = {
    "gemini-3.1-flash-lite": {"input": 0.25, "cached_input": 0.025, "output": 1.50},
    "gemini-3.1-pro-preview": {"input": 2.00, "cached_input": 0.20, "output": 12.00},
    "gemini-3.5-flash": {"input": 1.50, "cached_input": 0.15, "output": 9.00},
}

DEFAULT_PRICING_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_PRICING_MODEL = "gemini-2.5-flash"


def estimate_openai_cost(model: str, usage) -> dict:
    pricing = OPENAI_PRICING.get(model)
    pricing_model = model
    if pricing is None:
        pricing = OPENAI_PRICING[DEFAULT_PRICING_MODEL]
        pricing_model = f"{DEFAULT_PRICING_MODEL} (fallback)"

    input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached_input = getattr(details, "cached_tokens", 0) or 0
    normal_input = max(input_tokens - cached_input, 0)

    input_cost = (normal_input / 1_000_000) * pricing["input"]
    cached_cost = (cached_input / 1_000_000) * pricing["cached_input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + cached_cost + output_cost

    return {
        "provider": "OpenAI",
        "model_used": model,
        "pricing_model": pricing_model,
        "input_tokens": input_tokens,
        "normal_input_tokens": normal_input,
        "cached_input_tokens": cached_input,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost_usd": round(input_cost, 8),
        "cached_input_cost_usd": round(cached_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "estimated_total_cost_usd": round(total_cost, 8),
        "pricing_usd_per_1m_tokens": pricing,
    }


def estimate_gemini_cost(model: str, usage_metadata) -> dict:
    pricing = GEMINI_PRICING.get(model)
    pricing_model = model
    if pricing is None:
        pricing = GEMINI_PRICING[DEFAULT_GEMINI_PRICING_MODEL]
        pricing_model = f"{DEFAULT_GEMINI_PRICING_MODEL} (fallback)"

    input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
    total_tokens = getattr(usage_metadata, "total_token_count", None) or input_tokens + output_tokens
    cached_input = getattr(usage_metadata, "cached_content_token_count", 0) or 0
    normal_input = max(input_tokens - cached_input, 0)

    input_cost = (normal_input / 1_000_000) * pricing["input"]
    cached_cost = (cached_input / 1_000_000) * pricing["cached_input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + cached_cost + output_cost

    return {
        "provider": "Gemini",
        "model_used": model,
        "pricing_model": pricing_model,
        "input_tokens": input_tokens,
        "normal_input_tokens": normal_input,
        "cached_input_tokens": cached_input,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": round(input_cost, 8),
        "cached_input_cost_usd": round(cached_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "estimated_total_cost_usd": round(total_cost, 8),
        "pricing_usd_per_1m_tokens": pricing,
    }


def combine_usage(usages: list[dict | None]) -> dict | None:
    valid = [usage for usage in usages if isinstance(usage, dict)]
    if not valid:
        return None

    return {
        "provider": valid[-1].get("provider", ""),
        "model_used": valid[-1].get("model_used", ""),
        "pricing_model": valid[-1].get("pricing_model", ""),
        "input_tokens": sum(usage.get("input_tokens", 0) or 0 for usage in valid),
        "normal_input_tokens": sum(usage.get("normal_input_tokens", 0) or 0 for usage in valid),
        "cached_input_tokens": sum(usage.get("cached_input_tokens", 0) or 0 for usage in valid),
        "output_tokens": sum(usage.get("output_tokens", 0) or 0 for usage in valid),
        "total_tokens": sum(usage.get("total_tokens", 0) or 0 for usage in valid),
        "input_cost_usd": round(sum(usage.get("input_cost_usd", 0) or 0 for usage in valid), 8),
        "cached_input_cost_usd": round(sum(usage.get("cached_input_cost_usd", 0) or 0 for usage in valid), 8),
        "output_cost_usd": round(sum(usage.get("output_cost_usd", 0) or 0 for usage in valid), 8),
        "estimated_total_cost_usd": round(
            sum(usage.get("estimated_total_cost_usd", 0) or 0 for usage in valid),
            8,
        ),
        "pricing_usd_per_1m_tokens": valid[-1].get("pricing_usd_per_1m_tokens", {}),
    }
