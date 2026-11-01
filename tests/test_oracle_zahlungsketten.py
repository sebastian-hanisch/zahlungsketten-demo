"""Orakel-Tests des Kerns gegen unabhängige Rechenwege (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2).

Orakel: (1) Aufzählung aller Zustandsmuster (zahlt nichts / zahlt Teil / zahlt voll) mit linearem Gleichungssystem je Muster (kein Iterieren), (2) das LP von Eisenberg und Noe
(scipy, e >= 0), (3) die Handformeln für Ring und vollständiges Netz gegen die Verrechnung auf einem Raster, (4) Iteration von beiden Seiten. Alle < 10 s."""

import itertools
import math

import numpy as np
import pytest

import zkt_algorithm as A
import zkt_constants as C
import zkt_scenario as S


def _clear_by_state_enumeration(w, e):
    """Eindeutiger Verrechnungsvektor durch Aufzählung der Zustände (0 = zahlt nichts, 1 = zahlt einen Teil, 2 = zahlt voll): je Muster löst ein lineares System die
    Teil-Zahler, danach wird die Verträglichkeit geprüft (kein Iterationsverfahren, kein LP)."""
    n = len(e)
    L, pi = A.relative_liabilities(w)
    found = []
    for state in itertools.product((0, 1, 2), repeat=n):
        part = [i for i in range(n) if state[i] == 1]
        p = np.array([0.0 if s == 0 else L[i] if s == 2 else 0.0 for i, s in enumerate(state)])
        if part:
            m = np.eye(len(part)) - pi.T[np.ix_(part, part)]
            rhs = e[part] + pi.T[np.ix_(part, [i for i in range(n) if i not in part])] @ p[[i for i in range(n) if i not in part]]
            if abs(np.linalg.det(m)) < 1e-12:
                continue
            p[part] = np.linalg.solve(m, rhs)
        assets = e + pi.T @ p
        ok = True
        for i, s in enumerate(state):
            if s == 0 and not (assets[i] <= 1e-12 and L[i] > -1e-12):
                ok = False
            elif s == 1 and not (-1e-12 < p[i] < L[i] - 1e-12 and abs(p[i] - assets[i]) < 1e-9):
                ok = False
            elif s == 2 and not (assets[i] >= L[i] - 1e-12):
                ok = False
        if ok:
            found.append(p)
    return found


def test_the_clearing_vector_equals_the_state_enumeration_on_small_random_nets():
    rng = np.random.default_rng(2)
    for trial in range(60):
        n = int(rng.integers(3, 6))
        w = rng.random((n, n)) * (rng.random((n, n)) < 0.6)
        np.fill_diagonal(w, 0.0)
        e = (rng.random(n) - 0.3) * 1.5
        found = _clear_by_state_enumeration(w, e)
        p, L, _ = A.clear(w, e)
        assert any(np.allclose(p, f, atol=1e-8) for f in found), (trial, n)
        assert len(found) >= 1


def test_clearing_vector_is_unique_by_enumeration_when_every_firm_owes():
    rng = np.random.default_rng(4)
    for trial in range(40):
        n = 4
        w = rng.random((n, n)) + 0.1
        np.fill_diagonal(w, 0.0)
        e = (rng.random(n) - 0.4) * 2.0
        found = _clear_by_state_enumeration(w, e)
        assert len(found) == 1, (trial, len(found))


def test_the_clearing_vector_solves_the_lp_of_eisenberg_and_noe():
    sp = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(7)
    for kind in C.KINDS:
        for _ in range(15):
            w = S.graph_from_rng(kind, 12, rng)
            e = rng.random(12) * 0.8
            L, pi = A.relative_liabilities(w)
            res = sp.linprog(-np.ones(12), A_ub=np.eye(12) - pi.T, b_ub=e, bounds=[(0, L[i]) for i in range(12)])
            assert A.clear(w, e)[0] == pytest.approx(res.x, abs=1e-8), kind


def test_greatest_and_least_fixed_point_coincide_also_with_negative_external_assets():
    rng = np.random.default_rng(9)
    for kind in C.KINDS:
        for _ in range(20):
            w = S.graph_from_rng(kind, 15, rng)
            e = rng.random(15) * 0.8 - rng.random(15) * 1.5
            assert A.clear(w, e, "gross")[0] == pytest.approx(A.clear(w, e, "klein")[0], abs=1e-9), kind


def test_ring_formula_against_the_clearing_on_a_fine_grid():
    for b in (0.05, 0.1, 0.2):
        w = S.ring(40)
        for s in np.arange(b + 0.011, 3.0, 0.005):
            if abs((s - b) / b - round((s - b) / b)) < 1e-6:
                continue                                   # genau auf der Stufe: Gleichheit, Rundungsfrage
            e = A.apply_shock(A.external_assets(w, b), [7], s)
            p, L, _ = A.clear(w, e)
            assert int(A.failed(p, L).sum()) == A.ring_defaults(float(s), b, 40), (b, round(float(s), 3))


def test_ring_formula_independent_chain_argument():
    """Zweiter Weg zur Formel: der Fehlbetrag sinkt je Firma um b, solange er positiv ist (Kette per Schleife statt Iteration über das ganze Netz)."""
    for b in (0.05, 0.1, 0.2):
        for s in (b + 0.013, 0.5, 0.77, 1.3, 2.0):
            deficit = min(s - b, 1.0)
            count = 0
            while deficit > 1e-12 and count < 40:
                count += 1
                deficit -= b
            assert count == A.ring_defaults(s, b, 40), (b, s)


@pytest.mark.parametrize("n", [10, 20, 30, 40])
def test_complete_threshold_formula_against_the_clearing(n):
    w = S.complete(n)
    for b in (0.05, 0.1, 0.2, 0.3):
        for k in range(1, n // 2 + 1):
            e0 = A.external_assets(w, b)
            if A.cascade_possible(b, n, k):
                star = A.complete_threshold(b, n, k)
                for s, expected in ((star - 0.002, k / n), (star + 0.002, 1.0)):
                    p, L, _ = A.clear(w, A.apply_shock(e0, range(k), s))
                    assert float(A.failed(p, L).mean()) == pytest.approx(expected), (n, b, k, round(s, 4))
            else:
                p, L, _ = A.clear(w, A.apply_shock(e0, range(k), 50.0))
                assert float(A.failed(p, L).mean()) == pytest.approx(k / n), (n, b, k)


def test_complete_deficit_formula_below_the_cap():
    """Fehlbetrag einer geschockten Firma im vollständigen Netz: (s - b)(n - 1)/(n - k) unter dem Deckel 1 (zweiter Rechenweg zur Herleitung der Schwelle)."""
    n, b, k = 40, 0.1, 4
    w = S.complete(n)
    for s in (0.3, 0.6, 0.9):
        p, L, _ = A.clear(w, A.apply_shock(A.external_assets(w, b), range(k), s))
        assert (L - p)[0] == pytest.approx((s - b) * (n - 1) / (n - k), abs=1e-9)


def test_insolvency_costs_never_raise_the_payments_and_make_the_ring_collapse():
    rng = np.random.default_rng(5)
    for kind in C.KINDS:
        for _ in range(10):
            w = S.graph_from_rng(kind, 20, rng)
            e = A.apply_shock(A.external_assets(w, 0.1), A.pick_hit(20, 3, rng), 0.9)
            assert (A.clear(w, e, "gross", 0.8)[0] <= A.clear(w, e)[0] + 1e-9).all(), kind
    w = S.ring(40)
    p, L, _ = A.clear(w, A.apply_shock(A.external_assets(w, 0.1), [0, 10, 20, 30], 0.2), "gross", 0.8)
    assert A.failed(p, L).all()


def test_the_generators_give_rows_that_sum_to_one_and_the_documented_shapes():
    rng = np.random.default_rng(1)
    for kind in C.KINDS:
        w = S.graph_from_rng(kind, 14, rng)
        assert w.sum(axis=1) == pytest.approx(np.ones(14)) and (np.diag(w) == 0).all() and (w >= 0).all()
    assert (S.complete(6) > 0).sum() == 30 and (S.ring(6) > 0).sum() == 6
    assert S.core_periphery(12, np.random.default_rng(0))[:5, :5].sum() == pytest.approx(5.0)
    assert math.isclose(S.make_network("zufall", 14, 3).w.sum(), 14.0)
