"""Zeilen der README-Tabellen, gerechnet über dieselben Funktionen wie in der App (zkt_evaluation) und in der Rundung der App.

`tests/test_claims.py` prüft jede dieser Zeilen gegen die README; `python tools/readme_rows.py` druckt sie zum Einfügen.
Alle Zahlen sind deterministisch (feste Seeds SWEEP_SEEDS, je SWEEP_RUNS Läufe je Seed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import zkt_algorithm as A  # noqa: E402
import zkt_constants as C  # noqa: E402
import zkt_evaluation as ev  # noqa: E402
from zkt_scenario import make_network  # noqa: E402

N, B = 40, 0.1
S_TABLE = (0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0)
S_RING = (0.05, 0.15, 0.25, 0.55, 0.95, 1.05, 1.15, 3.0)
THRESHOLD_CASES = ((0.05, 2), (0.05, 4), (0.1, 4), (0.1, 8), (0.2, 8), (0.1, 3), (0.2, 4))
S_COST = (0.2, 0.5, 1.0, 2.0)


def c(v: float, d: int = 2) -> str:
    return f"{round(v, d) + 0.0:.{d}f}".replace(".", ",")


def p(v: float, d: int = 1) -> str:
    return c(v, d) + " %"


def forms_rows() -> list[str]:
    rows = []
    for k in (1, 4):
        forms = ev.sweep_forms(N, B, k)
        for kind in C.KINDS:
            by_s = {r["s"]: r for r in forms[kind]}
            rows.append(f"| {k} | {C.KIND_LABEL[kind]} | " + " | ".join(p(by_s[s]["mean"]) for s in S_TABLE) + " |")
    return rows


def ring_rows() -> list[str]:
    w = make_network("ring", N, 0).w
    rows = []
    for s in S_RING:
        e = A.apply_shock(A.external_assets(w, B), [7], s)
        pv, L, _ = A.clear(w, e)
        rows.append(f"| {c(s)} | {A.ring_defaults(s, B, N)} | {int(A.failed(pv, L).sum())} |")
    return rows


def threshold_rows() -> list[str]:
    rows = []
    for b, k in THRESHOLD_CASES:
        star, below, above = ev.complete_below_above(N, b, k)
        if star is None:
            rows.append(f"| {c(b)} | {k} | {c(b * (N - 1))} | – | keine Kaskade | – |")
        else:
            rows.append(f"| {c(b)} | {k} | {c(b * (N - 1))} | {c(star, 3)} | {p(below)} | {p(above)} |")
    return rows


def cost_rows() -> list[str]:
    free = ev.sweep_forms(N, B, 4)
    cost = ev.sweep_forms(N, B, 4, C.COST_ALPHA)
    rows = []
    for kind in C.KINDS:
        cells = []
        for s in S_COST:
            f = [r for r in free[kind] if r["s"] == s][0]["mean"]
            k = [r for r in cost[kind] if r["s"] == s][0]["mean"]
            cells.append(f"{p(f)} / {p(k)}")
        rows.append(f"| {C.KIND_LABEL[kind]} | " + " | ".join(cells) + f" | {ev.count_nonunique(kind)} von {C.NONUNIQUE_NETS} |")
    return rows


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for title, rows in (("FORMS", forms_rows()), ("RING", ring_rows()), ("THRESHOLD", threshold_rows()), ("COST", cost_rows())):
        print(f"--- {title}")
        print("\n".join(rows))
