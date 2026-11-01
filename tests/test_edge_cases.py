"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import zkt_algorithm as A
import zkt_constants as C
import zkt_evaluation as ev
import zkt_scenario as S


def test_exactly_two_firms_and_no_shock():
    w = np.array([[0.0, 1.0], [1.0, 0.0]])
    p, L, _ = A.clear(w, A.external_assets(w, 0.1))
    assert p == pytest.approx([1.0, 1.0]) and not A.failed(p, L).any()


def test_a_single_firm_without_creditors_owes_nothing():
    w = np.zeros((1, 1))
    p, L, _ = A.clear(w, np.array([-1.0]))
    assert p.tolist() == [0.0] and L.tolist() == [0.0] and not A.failed(p, L).any()


def test_all_firms_shocked_and_shock_zero():
    """k = n: jede Firma verliert s; s = 0: nichts passiert (Grenze inklusive)."""
    for kind in C.KINDS:
        a = ev.analyse(ev.Settings(kind, C.N_MIN, 0.1, C.N_MIN, 0.0, 1.0, 3))
        assert a.n_failed == 0, kind
        a = ev.analyse(ev.Settings(kind, C.N_MIN, 0.1, C.N_MIN, 3.0, 1.0, 3))
        assert a.n_failed == C.N_MIN and len(a.hit) == C.N_MIN, kind


def test_ties_in_the_shock_threshold_follow_the_documented_rule():
    """Genau auf dem Puffer (s = b) bricht keine Firma: ein Schock höchstens so groß wie der Puffer lässt niemanden ausfallen, im Ring wie im vollständigen Netz."""
    for kind in ("ring", "vollstaendig"):
        a = ev.analyse(ev.Settings(kind, 20, 0.1, 1, 0.1, 1.0, 3))
        assert a.n_failed == 0, kind
    assert A.ring_defaults(0.1, 0.1, 20) == 0


def test_bounds_are_inclusive():
    """Alle Regler an Minimum und Maximum, jede Netzform, mit und ohne Insolvenzkosten."""
    for kind in C.KINDS:
        for n, b, k, s, alpha in ((C.N_MIN, C.B_MIN, C.K_MIN, C.S_MIN, C.ALPHA_MIN), (C.N_MAX, C.B_MAX, C.K_MAX, C.S_MAX, C.ALPHA_MAX)):
            a = ev.analyse(ev.Settings(kind, n, b, k, s, alpha, 3))
            assert a.residual < 1e-9 and 0 <= a.n_failed <= n and np.isfinite(a.p).all()


def test_generators_reject_unknown_forms_and_small_cores():
    with pytest.raises(ValueError):
        S.graph_from_rng("unbekannt", 10, np.random.default_rng(0))
    w = S.core_periphery(C.N_MIN, np.random.default_rng(0))
    assert w.sum(axis=1) == pytest.approx(np.ones(C.N_MIN))


def test_complete_below_above_without_a_cascade_returns_none():
    assert ev.complete_below_above(40, 0.1, 3) == (None, None, None)
    star, below, above = ev.complete_below_above(40, 0.1, 4)
    assert star == pytest.approx(1.0) and below == pytest.approx(10.0) and above == pytest.approx(100.0)


def test_threshold_table_marks_cells_without_a_cascade():
    """Zeilen k = 1, 2, 4, Spalten b = 0,05 und 0,1 bei n = 40: b (n - 1) = 1,95 bzw. 3,9, eine Kaskade gibt es nur für k > b (n - 1)."""
    table = ev.threshold_table(40, (0.05, 0.1), (1, 2, 4))
    assert table[0] == [None, None]
    assert table[1][0] == pytest.approx(1.0) and table[1][1] is None
    assert table[2][0] == pytest.approx(0.5) and table[2][1] == pytest.approx(1.0)


def test_nearest_row_prefers_the_smaller_s_on_an_exact_tie():
    rows = [{"s": 0.5, "mean": 1.0}, {"s": 1.0, "mean": 2.0}]
    assert ev.nearest_row(rows, 0.75)["s"] == 0.5                                      # genau in der Mitte (in Binärzahlen exakt): das kleinere
    assert ev.nearest_row(rows, 0.76)["s"] == 1.0 and ev.nearest_row(rows, 0.1)["s"] == 0.5


def test_first_above_returns_none_when_the_level_is_never_exceeded():
    assert ev.first_above([{"s": 0.1, "mean": 10.0}, {"s": 0.2, "mean": 50.0}]) is None
    assert ev.first_above([{"s": 0.1, "mean": 10.0}, {"s": 0.2, "mean": 50.1}]) == 0.2


def test_worst_case_at_the_slider_maximum_finishes():
    """n = 40, b = 0,05, Ring, s = 3: die längste Kette, und ein Sweep über alle Formen bei k = 10 laufen durch (kein Zeit-Assert)."""
    a = ev.analyse(ev.Settings("ring", C.N_MAX, C.B_MIN, 1, C.S_MAX, 0.5, 3))
    assert a.residual < 1e-9
    forms = ev.sweep_forms(C.N_MAX, C.B_MIN, C.K_MAX, 0.6, grid=(0.0, 1.0, 3.0))
    assert set(forms) == set(C.KINDS) and all(len(rows) == 3 for rows in forms.values())
