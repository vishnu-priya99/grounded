"""Gold-set evaluation harness.

    python eval/run_eval.py            # run everything, write eval/results.md
    python eval/run_eval.py -n 5       # first 5 items only (quick check)

Metrics
-------
answerable items:
  - correct   : an expected answer string appears in the response
  - cited     : the response carries at least one citation
  - not-refused
unanswerable items:
  - refused   : the system correctly declined

The bundled corpus must be ingested first (`python -m grounded.cli ingest "data/samples/*"`).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from grounded.agents.graph import answer  # noqa: E402

GOLD = Path(__file__).parent / "gold.jsonl"
OUT = Path(__file__).parent / "results.md"


def _load() -> list[dict]:
    lines = GOLD.read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=0, help="limit number of items")
    ap.add_argument("-k", type=str, default="", help="only items whose question contains this")
    ap.add_argument("--kind", type=str, default="", help="only this kind, e.g. doc/table/code")
    args = ap.parse_args()

    items = _load()
    full_run = not (args.k or args.kind or args.n)
    if args.k:
        items = [it for it in items if args.k.lower() in it["q"].lower()]
    if args.kind:
        items = [it for it in items if it["kind"] == args.kind]
    if args.n:
        items = items[: args.n]

    rows = []
    for i, item in enumerate(items, start=1):
        t0 = time.perf_counter()
        res = answer(item["q"])
        dt = time.perf_counter() - t0
        ans_lower = res.answer.lower()

        if item.get("expect_refuse"):
            ok = res.refused
            detail = "refused" if res.refused else f"answered: {res.answer[:60]}"
        else:
            hit = any(e.lower() in ans_lower for e in item.get("expect_any", []))
            ok = hit and not res.refused
            detail = (
                ("correct" if hit else "MISS")
                + ("" if res.citations else ", no-cite")
                + ("" if not res.refused else ", REFUSED")
            )
        rows.append(
            {
                "n": i, "kind": item["kind"], "q": item["q"], "ok": ok,
                "confidence": res.confidence, "seconds": dt, "detail": detail,
            }
        )
        print(f"[{i:2d}/{len(items)}] {'PASS' if ok else 'FAIL'}  {item['kind']:12s} "
              f"conf={res.confidence:.2f} {dt:5.1f}s  {item['q'][:60]}")
        if full_run:
            _report(rows, partial=i < len(items))  # checkpoint after every item

    if full_run:
        _report(rows)
    passed = sum(r["ok"] for r in rows)
    dest = f"  (details in {OUT})" if full_run else ""
    print(f"\n{passed}/{len(rows)} passed{dest}")
    return 0 if passed == len(rows) else 1


def _report(rows: list[dict], partial: bool = False) -> None:
    ans = [r for r in rows if r["kind"] != "unanswerable"]
    una = [r for r in rows if r["kind"] == "unanswerable"]
    status = " (in progress)" if partial else ""
    lines = [
        "# Evaluation results", "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M')} · {len(rows)} items{status}_", "",
        "| metric | score |", "| --- | --- |",
        f"| Answerable correct | {sum(r['ok'] for r in ans)}/{len(ans)} |",
        f"| Refusal accuracy (unanswerable) | {sum(r['ok'] for r in una)}/{len(una)} |",
        f"| Mean confidence (answerable) | {statistics.mean([r['confidence'] for r in ans]):.2f} |",
        f"| Median latency | {statistics.median([r['seconds'] for r in rows]):.1f}s |",
        "", "## Per-item", "",
        "| # | kind | question | result | conf | s |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        mark = "✅" if r["ok"] else "❌"
        lines.append(
            f"| {r['n']} | {r['kind']} | {r['q']} | {mark} {r['detail']} | "
            f"{r['confidence']:.2f} | {r['seconds']:.1f} |"
        )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
