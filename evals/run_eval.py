"""Run the eval: drive the real agent through each case against the mock home,
then score the final home state with the case's checks.

    python3 -m evals.run_eval            # run every case
    python3 -m evals.run_eval movie_night wind_down   # run named cases only

Each case runs the real model, so this costs a little. The mock home makes the
*scoring* free and deterministic: we read the final device states and check them.

Scoring is programmatic (no LLM judge): one point for "the agent finished" plus
one point per check that holds. A per-case and overall pass count is printed.
"""
from __future__ import annotations

import sys

from agent.loop import ExperienceAgent
from evals.cases import CASES, Case
from smarthome.mock import MockHome


class _Counter:
    """Wraps a real client to count how many model calls a case takes."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        return self._inner.messages.create(**kwargs)


def run_case(case: Case, client) -> dict:
    home = MockHome(fail_device_ids=case.fail_devices)
    counter = _Counter(client)
    agent = ExperienceAgent(home, client=counter, verbose=False)

    rows: list[tuple[str, bool]] = []
    try:
        result = agent.run(case.request)
        finished = not result.startswith("Stopped:")
        rows.append(("agent finished", finished))
        by_id = home.get_home_graph().by_id()
        for chk in case.checks:
            try:
                rows.append((chk.desc, bool(chk.ok(by_id))))
            except Exception as exc:  # a check that errors is a fail, not a crash
                rows.append((f"{chk.desc} [check error: {exc}]", False))
    except Exception as exc:
        result = f"ERROR: {exc}"
        rows.append((f"agent ran without crashing [{exc}]", False))

    passed = sum(1 for _, ok in rows if ok)
    return {"name": case.name, "rows": rows, "passed": passed,
            "total": len(rows), "calls": counter.calls, "result": result}


def main() -> int:
    wanted = set(sys.argv[1:])
    cases = [c for c in CASES if not wanted or c.name in wanted]
    if not cases:
        print(f"no cases match {sorted(wanted)}; known: {[c.name for c in CASES]}")
        return 2

    try:
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    except Exception as exc:
        print(f"Can't create the Anthropic client ({exc}).")
        print("Install deps (pip install -r requirements.txt) and set ANTHROPIC_API_KEY.")
        return 2

    total_pass = total = 0
    summary = []
    for case in cases:
        r = run_case(case, client)
        print(f"\n=== {r['name']}  ({r['passed']}/{r['total']} checks, {r['calls']} model calls) ===")
        for desc, ok in r["rows"]:
            print(f"  [{'x' if ok else ' '}] {desc}")
        total_pass += r["passed"]
        total += r["total"]
        summary.append((r["name"], r["passed"], r["total"]))

    print("\n--- summary ---")
    for name, p, t in summary:
        print(f"  {name:22} {p}/{t}")
    print(f"  {'TOTAL':22} {total_pass}/{total}")
    return 0 if total_pass == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
