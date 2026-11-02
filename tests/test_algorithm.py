"""Kern gegen Handrechnung und Randfälle (DEMO-PLAYBOOK §1b und §12)."""

import numpy as np
import pytest

import zkt_algorithm as A
import zkt_constants as C


def test_two_firms_by_hand(two_firms):
    p, L, _ = A.clear(two_firms, np.array([0.2, 0.2]))
    assert p == pytest.approx([1.0, 1.0]) and L.tolist() == [1.0, 1.0] and not A.failed(p, L).any()
    p, L, _ = A.clear(two_firms, np.array([-0.5, 0.2]))
    assert p == pytest.approx([0.0, 0.2]) and A.failed(p, L).tolist() == [True, True]


def test_ring3_by_hand(ring3):
    e = A.apply_shock(A.external_assets(ring3, 0.1), [0], 0.5)
    assert e == pytest.approx([-0.4, 0.1, 0.1], abs=1e-12)
    assert A.external_assets(ring3, 0.1) == pytest.approx([0.1, 0.1, 0.1])               # Forderungen 1 decken die Schuld, nur der Puffer bleibt
    p, L, _ = A.clear(ring3, e)
    assert p == pytest.approx([0.0, 0.1, 0.2]) and A.failed(p, L).all()
    assert A.ring_defaults(0.5, 0.1, 3) == 3                                               # ceil(0,4 / 0,1) = 4, begrenzt auf n = 3


def test_firm_without_debt(chain3):
    p, L, _ = A.clear(chain3, np.array([0.0, 0.5, 0.5]))
    assert p == pytest.approx([0.0, 0.5, 0.0]) and L.tolist() == [1.0, 1.0, 0.0]
    assert A.failed(p, L).tolist() == [True, True, False]                                  # Firma 0 hat nichts, Firma 1 zahlt die Hälfte, Firma 2 schuldet nichts


def test_relative_liabilities_rows_sum_to_one_or_zero(chain3):
    L, pi = A.relative_liabilities(chain3)
    assert L.tolist() == [1.0, 1.0, 0.0] and pi.sum(axis=1).tolist() == [1.0, 1.0, 0.0]


def test_greatest_and_least_fixed_point_agree_without_costs(ring3):
    e = A.apply_shock(A.external_assets(ring3, 0.1), [0], 0.5)
    hi, _, _ = A.clear(ring3, e, "gross")
    lo, _, _ = A.clear(ring3, e, "klein")
    assert hi == pytest.approx(lo, abs=1e-12)


def test_residual_detects_a_wrong_vector(two_firms):
    e = np.array([0.2, 0.2])
    p, _, _ = A.clear(two_firms, e)
    assert A.residual(two_firms, e, p) < 1e-12
    assert A.residual(two_firms, e, p - np.array([0.3, 0.0])) > 0.2


def test_insolvency_costs_by_hand():
    """Zwei Firmen: Firma 0 hat nach dem Schock Mittel 0,5 und schuldet 1 an Firma 1, die ihr nichts schuldet. Mit alpha = 0,8 zahlt Firma 0 nur 0,4."""
    w = np.array([[0.0, 1.0], [0.0, 0.0]])
    p, L, _ = A.clear(w, np.array([0.5, 0.0]), "gross", alpha=0.8)
    assert p[0] == pytest.approx(0.4) and A.failed(p, L).tolist() == [True, False]
    p_free, _, _ = A.clear(w, np.array([0.5, 0.0]))
    assert p_free[0] == pytest.approx(0.5)


def test_alpha_one_equals_no_costs(ring3):
    e = A.apply_shock(A.external_assets(ring3, 0.1), [0], 0.5)
    assert A.clear(ring3, e, alpha=1.0)[0] == pytest.approx(A.clear(ring3, e)[0], abs=1e-12)


def test_ring_defaults_formula_values():
    assert A.ring_defaults(0.05, 0.1, 40) == 0 and A.ring_defaults(0.1, 0.1, 40) == 0         # Schock höchstens der Puffer
    assert A.ring_defaults(0.15, 0.1, 40) == 1 and A.ring_defaults(0.25, 0.1, 40) == 2
    assert A.ring_defaults(1.1, 0.1, 40) == 10 and A.ring_defaults(5.0, 0.1, 40) == 10          # Fehlbetrag auf 1 begrenzt
    assert A.ring_defaults(5.0, 0.1, 6) == 6                                                    # höchstens n


def test_threshold_formulas_by_hand():
    assert A.cascade_possible(0.1, 40, 4) and not A.cascade_possible(0.1, 40, 3)               # b (n - 1) = 3,9
    assert not A.cascade_possible(0.1, 11, 1)                                                   # b (n - 1) = 1,0: k = 1 reicht nicht (strikt größer)
    assert A.complete_threshold(0.1, 40, 4) == pytest.approx(1.0)
    assert A.complete_threshold(0.1, 40, 3) is None
    assert A.complete_threshold(0.5, 4, 2) == pytest.approx(1.0)


def test_pick_hit_is_sorted_distinct_and_reproducible():
    h1 = A.pick_hit(20, 5, np.random.default_rng(3))
    h2 = A.pick_hit(20, 5, np.random.default_rng(3))
    assert h1.tolist() == h2.tolist() == sorted(set(h1.tolist())) and len(h1) == 5 and 0 <= h1.min() and h1.max() < 20


def test_apply_shock_does_not_modify_the_input():
    e = np.array([0.1, 0.1, 0.1])
    out = A.apply_shock(e, [1], 0.5)
    assert e.tolist() == [0.1, 0.1, 0.1] and out.tolist() == pytest.approx([0.1, -0.4, 0.1])


def test_max_iterations_bound_is_respected():
    w = np.array([[0.0, 1.0], [1.0, 0.0]])
    _, _, steps = A.clear(w, np.array([0.2, 0.2]), "klein", max_iter=3)
    assert steps == 3                                                                          # vom kleinsten Punkt aus nicht konvergiert, aber begrenzt
    assert C.MAX_ITER >= 1000


def test_iteration_cap_is_visible_in_the_step_count_so_the_app_can_warn():
    """Erreicht die Iteration den Höchstwert, steht das in der Schrittzahl (die App zeigt dann einen Hinweis); sonst bleibt sie darunter."""
    w = np.array([[0.0, 1.0], [1.0, 0.0]])
    _, _, steps = A.clear(w, np.array([0.2, 0.2]), "klein", max_iter=3)
    assert steps == 3
    _, _, steps_full = A.clear(w, np.array([0.2, 0.2]), "gross")
    assert steps_full < C.MAX_ITER
