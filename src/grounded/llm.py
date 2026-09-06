"""Thin LLM client. Defaults to a local Ollama; optionally proxies to a paid
provider when ``LLM_PROVIDER`` is set (the only place a key is ever used)."""

from __future__ import annotations

import json
import time
from typing import Any, get_args

from pydantic import BaseModel

from grounded.config import settings

_Message = dict[str, Any]


def chat(
    messages: list[_Message],
    *,
    model: str | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
) -> str:
    """Return the assistant's text for a chat completion."""
    model = _resolve_model(model or settings.llm_model)
    temperature = settings.llm_temperature if temperature is None else temperature

    if settings.llm_provider == "ollama":
        import ollama

        client = ollama.Client(host=settings.ollama_host)
        t0 = time.perf_counter()
        resp = client.chat(
            model=model,
            messages=messages,
            format="json" if json_mode else None,
            options={
                "temperature": temperature,
                "num_ctx": settings.llm_num_ctx,
            },
        )
        _track_ollama_usage(model, resp, time.perf_counter() - t0)
        return resp["message"]["content"]

    if settings.llm_provider == "openai":  # pragma: no cover - optional path
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"} if json_mode else None,
        )
        return resp.choices[0].message.content or ""

    if settings.llm_provider == "anthropic":  # pragma: no cover - optional path
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
        sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
        if json_mode:
            sys += "\n\nReply with a single valid JSON object and nothing else."
        turns = [
            {"role": m["role"], "content": m["content"]}
            for m in messages if m["role"] != "system"
        ]
        # The Claude 5 family rejects `temperature`; its default sampling is fine
        # for grounded QA, so it's simply not sent on this path.
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=model, system=sys, messages=turns, max_tokens=4096,
        )
        _track_anthropic_usage(
            model, resp.usage.input_tokens, resp.usage.output_tokens,
            time.perf_counter() - t0,
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


def vision_available() -> bool:
    """Whether ``chat_with_image`` has an actual provider to call. Anthropic
    always qualifies (every Claude model in ``_ANTHROPIC_PRICE_PER_MTOK``
    reads images); the local Ollama path only if a vision model is explicitly
    configured (``VLM_MODEL`` in .env) — most local text models can't see
    images at all. Callers check this before reading an image off disk, so a
    document ingested without vision access still degrades to its OCR/text
    fallback instead of erroring."""
    return settings.llm_provider == "anthropic" or bool(settings.vlm_model)


def chat_with_image(
    messages: list[_Message],
    image_bytes: bytes,
    *,
    media_type: str = "image/png",
    model: str | None = None,
    max_tokens: int = 2048,
) -> str:
    """Like :func:`chat`, but attaches ``image_bytes`` to the final user turn
    — lets the model read a chart/diagram/figure directly instead of trusting
    only a static text description of it (an OCR pass, or an earlier vision
    call baked in at ingest time). Call :func:`vision_available` first; this
    raises on a provider with no vision path rather than silently degrading,
    since a caller reaching here has already decided it wants a real image
    read, not a text-only guess."""
    if not vision_available():
        raise ValueError(
            "No vision-capable provider configured — check vision_available() first."
        )
    model = _resolve_model(model or settings.llm_model)
    sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [dict(m) for m in messages if m["role"] != "system"]
    if not turns:
        turns = [{"role": "user", "content": ""}]

    if settings.llm_provider == "anthropic":
        import base64

        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
        b64 = base64.standard_b64encode(image_bytes).decode()
        last = turns[-1]
        turns[-1] = {
            "role": last["role"],
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                },
                {"type": "text", "text": last["content"]},
            ],
        }
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=model, system=sys, messages=turns, max_tokens=max_tokens,
        )
        _track_anthropic_usage(
            model, resp.usage.input_tokens, resp.usage.output_tokens,
            time.perf_counter() - t0,
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    # Ollama, with a VLM configured (the only other branch vision_available()
    # allows through).
    import ollama

    client = ollama.Client(host=settings.ollama_host)
    last = turns[-1]
    chat_messages = (
        ([{"role": "system", "content": sys}] if sys else [])
        + turns[:-1]
        + [{"role": last["role"], "content": last["content"], "images": [image_bytes]}]
    )
    t0 = time.perf_counter()
    resp = client.chat(
        model=settings.vlm_model, messages=chat_messages,
        options={"temperature": settings.llm_temperature},
    )
    _track_ollama_usage(settings.vlm_model, resp, time.perf_counter() - t0)
    return resp["message"]["content"].strip()


# $ per million tokens (input, output). Update if pricing changes.
_ANTHROPIC_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
}
_usage_totals = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0}


def _track_anthropic_usage(
    model: str, input_tokens: int, output_tokens: int, seconds: float,
) -> None:
    """Print + accumulate real, measured cost — replaces estimates with facts
    once a key is live. See ``usage_summary()`` for the running total."""
    in_price, out_price = next(
        (p for name, p in _ANTHROPIC_PRICE_PER_MTOK.items() if name in model),
        (0.0, 0.0),
    )
    cost = input_tokens / 1e6 * in_price + output_tokens / 1e6 * out_price
    _usage_totals["input_tokens"] += input_tokens
    _usage_totals["output_tokens"] += output_tokens
    _usage_totals["cost_usd"] += cost
    _usage_totals["calls"] += 1
    print(
        f"[llm] {model} (anthropic, {seconds:.1f}s): {input_tokens} in / "
        f"{output_tokens} out = ${cost:.5f}  (running total: "
        f"${_usage_totals['cost_usd']:.4f} over {_usage_totals['calls']} calls)",
        flush=True,
    )


def _track_ollama_usage(model: str, resp: dict, seconds: float) -> None:
    """Same live line as the anthropic tracker, for the free local path — no
    cost, but the same at-a-glance visibility into what's running and how long
    it's taking."""
    in_tok = resp.get("prompt_eval_count", 0)
    out_tok = resp.get("eval_count", 0)
    _usage_totals["input_tokens"] += in_tok
    _usage_totals["output_tokens"] += out_tok
    _usage_totals["calls"] += 1
    print(
        f"[llm] {model} (ollama, local, {seconds:.1f}s): {in_tok} in / "
        f"{out_tok} out  (free — {_usage_totals['calls']} calls this run)",
        flush=True,
    )


def usage_summary() -> dict[str, float]:
    """Running token/cost total for this process (cost is $0 for ollama calls)."""
    return dict(_usage_totals)


_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"


def _resolve_model(model: str) -> str:
    """Guard against a stale Ollama tag (``qwen2.5:7b``) leaking through when
    ``LLM_PROVIDER=anthropic`` — set LLM_MODEL / ROUTER_MODEL to a ``claude-*``
    id in .env, but fall back sanely if they weren't."""
    if settings.llm_provider == "anthropic" and "claude" not in model:
        print(
            f"[llm] LLM_PROVIDER=anthropic but model {model!r} is not a Claude "
            f"model; using {_DEFAULT_ANTHROPIC_MODEL}. Set LLM_MODEL / ROUTER_MODEL "
            f"in .env to silence this."
        )
        return _DEFAULT_ANTHROPIC_MODEL
    return model


def chat_structured[T: BaseModel](
    messages: list[_Message],
    schema: type[T],
    *,
    model: str | None = None,
    fallback: T | None = None,
) -> T:
    """Chat and coerce the JSON reply into a Pydantic model.

    Small local models are inconsistent JSON emitters, so this tries hard: it
    strips fences, unwraps the ``{"properties": {...}}`` schema-echo that Qwen
    sometimes produces, and does one repair round-trip. If everything fails and a
    ``fallback`` was given, it returns that instead of raising.
    """
    fields = ", ".join(schema.model_fields)
    hint = {
        "role": "system",
        "content": (
            f"Reply with ONE JSON object with exactly these keys: {fields}. "
            "No prose, no markdown, no schema — just the object with real values.\n"
            "Shape:\n" + json.dumps(_example(schema))
        ),
    }
    raw = chat([hint, *messages], model=model, json_mode=True)
    parsed = _try_parse(schema, raw)
    if parsed is None:
        repair = {
            "role": "user",
            "content": f"Return ONLY the JSON object with keys {fields}. "
            f"Your reply was:\n{raw}",
        }
        raw = chat([hint, *messages, repair], model=model, json_mode=True)
        parsed = _try_parse(schema, raw)
    if parsed is not None:
        return parsed
    if fallback is not None:
        return fallback
    raise ValueError(f"Could not parse {schema.__name__} from model output: {raw[:400]}")


def _try_parse[T: BaseModel](schema: type[T], text: str) -> T | None:
    candidates: list[object] = []
    blob = _extract_json(text)
    try:
        obj = json.loads(blob)
    except Exception:
        return None
    candidates.append(obj)
    if isinstance(obj, dict):
        for key in ("properties", "parameters", "value", "result", "output"):
            if key in obj and isinstance(obj[key], dict):
                candidates.append(obj[key])
    for cand in candidates:
        try:
            return schema.model_validate(cand)
        except Exception:
            continue
    return None


def _example(schema: type[BaseModel]) -> dict[str, object]:
    out: dict[str, object] = {}
    for name, field in schema.model_fields.items():
        ann = field.annotation
        if type(None) in get_args(ann):  # Optional[X] / X | None
            out[name] = None
        elif getattr(ann, "__origin__", None) is list:
            out[name] = []
        else:
            out[name] = {bool: False, int: 0, float: 0.0, str: "", list: []}.get(ann, "")
    return out


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end != -1 else text
