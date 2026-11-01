"""Kennzahlen, Verdikt-Logik und Sweeps (Komposition der Regeln aus `zkt_algorithm.py`; Verdikte sind reine Funktionen)."""

from dataclasses import dataclass

import numpy as np

import zkt_algorithm as A
import zkt_constants as C
from zkt_scenario import Network, graph_from_rng, make_network


@dataclass(frozen=True)
class Settings:
    kind: str = C.DEFAULT_KIND
    n: int = C.DEFAULT_N
    b: float = C.DEFAULT_B
    k: int = C.DEFAULT_K
    s: float = C.DEFAULT_S
    alpha: float = C.DEFAULT_ALPHA        # Anteil, den eine ausfallende Firma auszahlt (1 = keine Insolvenzkosten)
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True, eq=False)
class Analysis:
    net: Network
    e: np.ndarray            # externe Mittel nach dem Schock
    hit: np.ndarray          # geschockte Firmen
    p: np.ndarray            # Verrechnungsvektor
    L: np.ndarray
    failed: np.ndarray       # Ausfallmaske
    steps: int               # Schritte der Fixpunktiteration
    residual: float          # Abweichung von der Fixpunktgleichung
    gap: float               # größter minus kleinster Fixpunkt (0 = eindeutig)
    alpha: float | None

    @property
    def n_failed(self) -> int:
        return int(self.failed.sum())

    @property
    def share(self) -> float:
        return float(self.failed.mean()) if len(self.failed) else 0.0


def _alpha(settings_alpha: float) -> float | None:
    return None if settings_alpha >= 1.0 else float(settings_alpha)


def analyse(settings: Settings) -> Analysis:
    net = make_network(settings.kind, settings.n, settings.seed)
    rng = np.random.default_rng(settings.seed + 7)
    hit = A.pick_hit(net.n, min(settings.k, net.n), rng)
    e = A.apply_shock(A.external_assets(net.w, settings.b), hit, settings.s)
    alpha = _alpha(settings.alpha)
    p, L, steps = A.clear(net.w, e, "gross", alpha)
    low, _, _ = A.clear(net.w, e, "klein", alpha)
    return Analysis(net, e, hit, p, L, A.failed(p, L), steps, A.residual(net.w, e, p, alpha), float(np.abs(p - low).max()), alpha)


# --- Verdikte als reine Funktionen -------------------------------------------------------------------------------------------------------

def _num(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def _pct(v: float) -> str:
    return _num(v, 1)


def verdict_shock(n_failed: int, n_hit: int, n: int, alpha: float | None = None) -> tuple[str, str]:
    """Kernfrage: Bleibt der Ausfall bei den geschockten Firmen, oder läuft er durch das Netz?"""
    tail = "" if alpha is None else f" (Insolvenzkosten: eine ausfallende Firma zahlt {_pct(100 * alpha)} % ihrer Mittel)"
    if n_failed == 0:
        return "success", f"Kein Ausfall: Die {n_hit} geschockten Firmen fangen den Schock mit ihren Mitteln ab{tail}."
    if n_failed <= n_hit:
        return "info", f"Es fallen {n_failed} von {n} Firmen aus, höchstens die geschockten: Das Netz gibt den Ausfall nicht weiter{tail}."
    ratio = n_failed / max(n_hit, 1)
    kind = "warning" if n_failed == n else "info"
    return kind, (f"Es fallen **{n_failed}** von {n} Firmen aus ({_pct(100 * n_failed / n)} %), bei {n_hit} geschockten: Der Ausfall läuft durch das Netz "
                  f"(das {_num(ratio, 1)}-Fache der geschockten Firmen){tail}.")


def verdict_threshold(b: float, n: int, k: int, s_star: float | None, below: float | None, above: float | None) -> tuple[str, str]:
    """Schwelle des vollständigen Netzes s* = b n / k und ihre Gegenprobe: Ausfallanteil knapp darunter und knapp darüber (in %)."""
    if s_star is None:
        return "success", (f"Bei k = {k} ≤ b (n − 1) = {_num(b * (n - 1))} gibt es im vollständigen Netz **keine Kaskade**: Selbst bei beliebig großem Schock fallen nur die {k} geschockten Firmen aus.")
    return "warning", (f"Schwelle des vollständigen Netzes: s* = b·n/k = {_num(s_star, 3)}. Knapp darunter fallen {_pct(below)} % der Firmen aus (nur die {k} geschockten), "
                       f"knapp darüber {_pct(above)} %.")


def verdict_cost(alpha: float, with_costs: float, without_costs: float, nonunique: int, nets: int) -> tuple[str, str]:
    """Insolvenzkosten: Ausfallanteil im Mittel mit und ohne Kosten (eine Schockstärke) und die Zahl der Netze mit mehreren Verrechnungsvektoren."""
    if with_costs <= without_costs + 1e-12:
        return "info", (f"Bei α = {_num(alpha)} ändern die Insolvenzkosten hier nichts: {_pct(100 * with_costs)} % Ausfälle mit und ohne Kosten. "
                        f"{nonunique} von {nets} Zufallsnetzen haben mehrere Verrechnungsvektoren.")
    return "warning", (f"Bei α = {_num(alpha)} steigt der Ausfallanteil von {_pct(100 * without_costs)} % auf **{_pct(100 * with_costs)} %**. "
                       f"{nonunique} von {nets} Zufallsnetzen haben mit Insolvenzkosten mehrere Verrechnungsvektoren.")


# --- Sweeps über feste Netze -------------------------------------------------------------------------------------------------------------

def _prepared(kind: str, n: int, k: int, seed: int, runs: int):
    """Je Seed feste Netze und feste geschockte Firmen: Der Sweep über s ist damit ein gepaarter Vergleich."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(runs):
        w = graph_from_rng(kind, n, rng)
        out.append((w, A.pick_hit(n, min(k, n), rng)))
    return out


def sweep_shock(kind: str, n: int, b: float, k: int, alpha: float | None = None, grid=C.S_GRID,
                seeds=C.SWEEP_SEEDS, runs: int = C.SWEEP_RUNS) -> list[dict]:
    """Ausfallanteil (in %) über die Schockstärke s: Mittel über alle Läufe, Band = kleinster und größter Seed-Mittelwert."""
    prepared = [_prepared(kind, n, k, seed, runs) for seed in seeds]
    rows = []
    for s in grid:
        per_seed = []
        for runs_of_seed in prepared:
            shares = []
            for w, hit in runs_of_seed:
                e = A.apply_shock(A.external_assets(w, b), hit, s)
                p, L, _ = A.clear(w, e, "gross", alpha)
                shares.append(float(A.failed(p, L).mean()))
            per_seed.append(100.0 * float(np.mean(shares)))
        rows.append({"s": s, "mean": float(np.mean(per_seed)), "low": float(min(per_seed)), "high": float(max(per_seed))})
    return rows


def sweep_forms(n: int, b: float, k: int, alpha: float | None = None, grid=C.S_GRID) -> dict:
    """Alle vier Netzformen mit denselben Einstellungen: {Form: Zeilen von sweep_shock}."""
    return {kind: sweep_shock(kind, n, b, k, alpha, grid) for kind in C.KINDS}


def first_above(rows: list[dict], level: float = 50.0) -> float | None:
    """Kleinstes s im Raster, bei dem der mittlere Ausfallanteil über `level` Prozent liegt (None, wenn nie)."""
    for r in rows:
        if r["mean"] > level:
            return r["s"]
    return None


def complete_below_above(n: int, b: float, k: int, eps: float = C.FORMULA_EPS):
    """Gegenprobe der Schwelle im vollständigen Netz: (s*, Ausfallanteil in % bei s* - eps, bei s* + eps); (None, None, None), wenn es keine Kaskade gibt."""
    star = A.complete_threshold(b, n, k)
    if star is None:
        return None, None, None
    w = make_network("vollstaendig", n, 0).w
    e0 = A.external_assets(w, b)
    shares = []
    for s in (star - eps, star + eps):
        e = A.apply_shock(e0, range(k), s)
        p, L, _ = A.clear(w, e)
        shares.append(100.0 * float(A.failed(p, L).mean()))
    return star, shares[0], shares[1]


def threshold_table(n: int, bs, ks) -> list[list[float | None]]:
    """s* = b n / k für jedes (b, k); None, wo es keine Kaskade gibt (k <= b (n - 1))."""
    return [[A.complete_threshold(b, n, k) for b in bs] for k in ks]


def count_nonunique(kind: str, alpha: float = C.COST_ALPHA, n: int = C.NONUNIQUE_N, nets: int = C.NONUNIQUE_NETS, b: float = 0.1, k: int = 3, s: float = 0.9) -> int:
    """Zahl der Netze (von `nets`) mit mehreren Verrechnungsvektoren (größter und kleinster Fixpunkt unterscheiden sich) bei Insolvenzkosten alpha."""
    rng = np.random.default_rng(21)
    count = 0
    for _ in range(nets):
        w = graph_from_rng(kind, n, rng)
        e = A.apply_shock(A.external_assets(w, b), A.pick_hit(n, k, rng), s)
        hi, _, _ = A.clear(w, e, "gross", alpha)
        lo, _, _ = A.clear(w, e, "klein", alpha)
        count += bool(np.abs(hi - lo).max() > 1e-9)
    return count


def nearest_row(rows: list[dict], s: float) -> dict:
    """Zeile des Sweeps, deren s dem gewünschten s am nächsten liegt (bei Gleichstand das kleinere)."""
    return min(rows, key=lambda r: (abs(r["s"] - s), r["s"]))


def verdict_sweep_forms(forms: dict, n: int, k: int) -> tuple[str, str]:
    """Netzform gegen Schockstärke: Sprung des vollständigen Netzes, Höchstwerte der anderen Formen."""
    comp, ring, rand, core = forms["vollstaendig"], forms["ring"], forms["zufall"], forms["kernrand"]
    top = lambda rows: max(r["mean"] for r in rows)
    jump = first_above(comp)
    if jump is None:
        head = f"Im vollständigen Netz fallen bei jeder Schockstärke höchstens {_pct(top(comp))} % der Firmen aus (die {k} geschockten): Es gibt keine Kaskade."
    else:
        before = [r for r in comp if r["s"] < jump][-1]
        head = (f"Im vollständigen Netz bleibt der Ausfall bis s = {_num(before['s'], 1)} bei {_pct(before['mean'])} % (die {k} geschockten Firmen), ab s = {_num(jump, 1)} fallen "
                f"**{_pct(comp[-1]['mean'])} %** aus.")
    return "info", head + f" Der Ring verliert höchstens {_pct(top(ring))} %, das Zufallsnetz {_pct(top(rand))} %, das Kern–Rand-Netz {_pct(top(core))} %."
