"""Eigenschaftstests (ORACLE-PLAYBOOK §5): Aussagen, die für jedes Netz gelten müssen, unabhängig von einem Vergleichswert."""

import numpy as np
import pytest

import zkt_algorithm as A
import zkt_constants as C
import zkt_evaluation as ev
import zkt_scenario as S


def _nets(n=16, count=4):
    rng = np.random.default_rng(1)
    for kind in C.KINDS:
        for _ in range(count):
            yield kind, S.graph_from_rng(kind, n, rng), rng


def test_payments_lie_between_zero_and_the_debt():
    for kind, w, rng in _nets():
        e = A.apply_shock(A.external_assets(w, 0.1), A.pick_hit(16, 3, rng), 1.5)
        p, L, _ = A.clear(w, e)
        assert (p >= -1e-12).all() and (p <= L + 1e-12).all(), kind


def test_without_a_shock_nobody_fails():
    for kind, w, _ in _nets():
        for b in (0.05, 0.1, 0.3):
            p, L, _ = A.clear(w, A.external_assets(w, b))
            assert not A.failed(p, L).any(), (kind, b)


def test_payments_rise_with_the_external_assets():
    for kind, w, rng in _nets(count=3):
        e = A.apply_shock(A.external_assets(w, 0.1), A.pick_hit(16, 3, rng), 1.0)
        more = e + rng.random(16) * 0.3
        assert (A.clear(w, more)[0] >= A.clear(w, e)[0] - 1e-9).all(), kind


def test_failures_rise_with_the_shock_and_with_more_shocked_firms():
    for kind, w, rng in _nets(count=3):
        hit = A.pick_hit(16, 6, rng)
        counts = []
        for s in (0.2, 0.5, 1.0, 2.0):
            p, L, _ = A.clear(w, A.apply_shock(A.external_assets(w, 0.1), hit[:3], s))
            counts.append(int(A.failed(p, L).sum()))
        assert counts == sorted(counts), (kind, counts)
        few = A.clear(w, A.apply_shock(A.external_assets(w, 0.1), hit[:3], 1.0))
        many = A.clear(w, A.apply_shock(A.external_assets(w, 0.1), hit, 1.0))
        assert (A.failed(many[0], many[1]) | ~A.failed(few[0], few[1])).all(), kind          # wer bei 3 Schocks ausfällt, fällt bei 6 auch aus


def test_more_buffer_means_fewer_failures():
    for kind, w, rng in _nets(count=3):
        hit = A.pick_hit(16, 4, rng)
        counts = []
        for b in (0.05, 0.1, 0.2, 0.3):
            p, L, _ = A.clear(w, A.apply_shock(A.external_assets(w, b), hit, 1.2))
            counts.append(int(A.failed(p, L).sum()))
        assert counts == sorted(counts, reverse=True), (kind, counts)


def test_relabelling_the_firms_permutes_the_result():
    rng = np.random.default_rng(3)
    for kind, w, _ in _nets(count=3):
        perm = rng.permutation(16)
        e = A.apply_shock(A.external_assets(w, 0.1), [2, 5, 9], 1.1)
        p, _, _ = A.clear(w, e)
        p2, _, _ = A.clear(w[np.ix_(perm, perm)], e[perm])
        assert p2 == pytest.approx(p[perm], abs=1e-9), kind


def test_scaling_all_debts_and_assets_scales_the_payments():
    for kind, w, rng in _nets(count=3):
        e = A.apply_shock(A.external_assets(w, 0.1), A.pick_hit(16, 4, rng), 1.0)
        p, L, _ = A.clear(w, e)
        p3, L3, _ = A.clear(3.0 * w, 3.0 * e)
        assert p3 == pytest.approx(3.0 * p, abs=1e-8) and (A.failed(p3, L3) == A.failed(p, L)).all(), kind


def test_costs_only_ever_reduce_payments_and_alpha_one_is_free():
    for kind, w, rng in _nets(count=3):
        e = A.apply_shock(A.external_assets(w, 0.1), A.pick_hit(16, 3, rng), 0.8)
        free = A.clear(w, e)[0]
        assert (A.clear(w, e, "gross", 0.9)[0] <= free + 1e-9).all() and (A.clear(w, e, "gross", 0.6)[0] <= A.clear(w, e, "gross", 0.9)[0] + 1e-9).all(), kind
        assert A.clear(w, e, "gross", 1.0)[0] == pytest.approx(free, abs=1e-12)


def test_without_costs_the_vector_is_unique_for_every_net_form():
    for kind, w, rng in _nets(count=5):
        e = A.apply_shock(A.external_assets(w, 0.1), A.pick_hit(16, 4, rng), 2.0)
        assert ev_gap(w, e) < 1e-9, kind


def ev_gap(w, e):
    return float(np.abs(A.clear(w, e, "gross")[0] - A.clear(w, e, "klein")[0]).max())


def test_the_ring_never_loses_everybody_when_few_firms_are_shocked_but_the_complete_net_does():
    n, b, k = 40, 0.1, 4
    ring_share = []
    for seed in range(5):
        a = ev.analyse(ev.Settings("ring", n, b, k, 3.0, 1.0, seed))
        ring_share.append(a.share)
    assert max(ring_share) < 1.0
    assert ev.analyse(ev.Settings("vollstaendig", n, b, k, 3.0, 1.0, 1)).share == 1.0


def test_failure_share_is_independent_of_the_seed_in_the_complete_net():
    """Vollständiges Netz, n = 30, b = 0,1, k = 4: s* = 0,75; bei s = 0,6 fallen immer genau die 4 geschockten Firmen aus, welche es auch sind."""
    shares = {ev.analyse(ev.Settings("vollstaendig", 30, 0.1, 4, 0.6, 1.0, seed)).share for seed in (1, 2, 3, 4)}
    assert shares == {4 / 30}
