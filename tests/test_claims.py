"""Jede Zahl der README steht hier und wird über dieselben Funktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4, §11a).

Die Tabellen der README werden Zeile für Zeile gegen `tools/readme_rows.py` geprüft (dieselbe Rundung wie die App); die einzelnen Sätze der README
haben je einen Test unter der Bedingung, die in der README daneben steht."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import zkt_algorithm as A  # noqa: E402
import zkt_constants as C  # noqa: E402
import zkt_evaluation as ev  # noqa: E402
import zkt_scenario as S  # noqa: E402
import readme_rows as R  # noqa: E402
from test_oracle_zahlungsketten import _clear_by_state_enumeration  # noqa: E402

README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_tables_in_the_readme_are_the_computed_rows():
    for name, rows in (("Tabelle 1", R.forms_rows()), ("Tabelle 2", R.ring_rows()), ("Tabelle 3", R.threshold_rows()), ("Tabelle 4", R.cost_rows())):
        for row in rows:
            assert row in README, (name, row)


def test_standard_setting_numbers():
    a = ev.analyse(ev.Settings())
    assert (a.net.n, a.n_failed, a.steps) == (40, 4, 13) and a.residual < 1e-12 and a.gap < 1e-9
    assert ev.analyse(ev.Settings(s=1.2)).n_failed == 40
    for part in ("nach 13 Schritten", "4 von 40 Firmen fallen aus", "Bei s = 1,2 fallen alle 40 aus"):
        assert part in README, part


def test_clearing_vector_claims_of_the_readme():
    rng = np.random.default_rng(2)
    found_ok = 0
    for _ in range(60):
        n = int(rng.integers(3, 6))
        w = rng.random((n, n)) * (rng.random((n, n)) < 0.6)
        np.fill_diagonal(w, 0.0)
        e = (rng.random(n) - 0.3) * 1.5
        p, _, _ = A.clear(w, e)
        found_ok += any(np.allclose(p, f, atol=1e-8) for f in _clear_by_state_enumeration(w, e))
    assert found_ok == 60
    rng = np.random.default_rng(4)
    unique = 0
    for _ in range(40):
        w = rng.random((4, 4)) + 0.1
        np.fill_diagonal(w, 0.0)
        unique += len(_clear_by_state_enumeration(w, (rng.random(4) - 0.4) * 2.0)) == 1
    assert unique == 40
    sp = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(7)
    lp_ok = 0
    for kind in C.KINDS:
        for _ in range(15):
            w = S.graph_from_rng(kind, 12, rng)
            e = rng.random(12) * 0.8
            L, pi = A.relative_liabilities(w)
            res = sp.linprog(-np.ones(12), A_ub=np.eye(12) - pi.T, b_ub=e, bounds=[(0, L[i]) for i in range(12)])
            lp_ok += bool(np.allclose(A.clear(w, e)[0], res.x, atol=1e-8))
    assert lp_ok == 60
    rng = np.random.default_rng(9)
    same = 0
    for kind in C.KINDS:
        for _ in range(20):
            w = S.graph_from_rng(kind, 15, rng)
            e = rng.random(15) * 0.8 - rng.random(15) * 1.5
            same += bool(np.allclose(A.clear(w, e, "gross")[0], A.clear(w, e, "klein")[0], atol=1e-9))
    assert same == 80
    for part in ("In 60 von 60 kleinen Zufallsnetzen", "bei 40 von 40 Netzen", "in 60 von 60 Netzen auf 10⁻⁸", "in 80 von 80 Netzen gleich"):
        assert part in README, part


def test_ring_grid_claim_of_the_readme():
    count = 0
    for b in (0.05, 0.1, 0.2):
        w = S.ring(40)
        for s in np.arange(b + 0.011, 3.0, 0.005):
            if abs((s - b) / b - round((s - b) / b)) < 1e-6:
                continue
            e = A.apply_shock(A.external_assets(w, b), [7], s)
            p, L, _ = A.clear(w, e)
            assert int(A.failed(p, L).sum()) == A.ring_defaults(float(s), b, 40)
            count += 1
    assert count == 1724 and "1 724 Schockstärken" in README


def test_threshold_claims_of_the_readme():
    cascade = none = 0
    for n in (10, 20, 30, 40):
        w = S.complete(n)
        for b in (0.05, 0.1, 0.2, 0.3):
            for k in range(1, n // 2 + 1):
                e0 = A.external_assets(w, b)
                if A.cascade_possible(b, n, k):
                    star = A.complete_threshold(b, n, k)
                    for s, expected in ((star - 0.002, k / n), (star + 0.002, 1.0)):
                        p, L, _ = A.clear(w, A.apply_shock(e0, range(k), s))
                        assert float(A.failed(p, L).mean()) == pytest.approx(expected)
                    cascade += 1
                else:
                    p, L, _ = A.clear(w, A.apply_shock(e0, range(k), 50.0))
                    assert float(A.failed(p, L).mean()) == pytest.approx(k / n)
                    none += 1
    assert (cascade, none) == (150, 50) and "Geprüft in 200 von 200 Fällen" in README and "in 150 gibt es eine Kaskade" in README and "in 50 fallen" in README


def test_deficit_formula_claim_of_the_readme():
    n, b, k = 40, 0.1, 4
    w = S.complete(n)
    for s in (0.3, 0.6, 0.9):
        p, L, _ = A.clear(w, A.apply_shock(A.external_assets(w, b), range(k), s))
        assert (L - p)[0] == pytest.approx((s - b) * (n - 1) / (n - k), abs=1e-9)
    assert "(s - b)(n-1)/(n-k)" in README


def test_structure_claims_of_the_readme():
    forms4 = ev.sweep_forms(40, 0.1, 4)
    forms1 = ev.sweep_forms(40, 0.1, 1)
    assert max(r["mean"] for r in forms1["vollstaendig"]) == 2.5
    assert max(r["mean"] for r in forms4["ring"]) == pytest.approx(71.7, abs=0.05) and max(r["mean"] for r in forms4["ring"]) < 100.0
    comp = {r["s"]: r["mean"] for r in forms4["vollstaendig"]}
    assert comp[1.0] == 10.0 and comp[1.1] == 100.0
    kern = {r["s"]: r["mean"] for r in forms4["kernrand"]}
    assert kern[0.5] == pytest.approx(8.7, abs=0.05) and kern[0.5] < comp[0.5]
    for part in ("im Mittel höchstens etwa 70 % (größter Mittelwert der Tabelle: 71,7 %)", "bis s = 1,0 nur diese vier (10,0 %)", "ab s = 1,1 alle", "8,7 % und damit weniger als das vollständige Netz (10,0 %)"):
        assert part in README, part


def test_cost_claims_of_the_readme():
    free = ev.sweep_forms(40, 0.1, 4)
    cost = ev.sweep_forms(40, 0.1, 4, C.COST_ALPHA)
    assert all(r["mean"] == 100.0 for r in cost["ring"] if r["s"] >= 0.2)
    assert [r["mean"] for r in free["vollstaendig"] if r["s"] == 1.0] == [10.0] and [r["mean"] for r in cost["vollstaendig"] if r["s"] == 1.0] == [100.0]
    assert all(c["mean"] >= f["mean"] - 1e-9 for kind in C.KINDS for f, c in zip(free[kind], cost[kind]))
    assert {kind: ev.count_nonunique(kind) for kind in C.KINDS} == {"ring": 0, "zufall": 23, "kernrand": 0, "vollstaendig": 0}
    assert "schon ein Schock von 0,2 (das Doppelte des Puffers) mit Insolvenzkosten alle Firmen zahlungsunfähig" in README


def test_hand_values_named_in_the_readme(ring3):
    e = A.apply_shock(A.external_assets(ring3, 0.1), [0], 0.5)
    assert A.clear(ring3, e)[0] == pytest.approx([0.0, 0.1, 0.2]) and "p = (0; 0,1; 0,2)" in README
