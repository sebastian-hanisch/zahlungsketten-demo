"""Kern der Demo (nur numpy): Verrechnung von Zahlungsketten nach Eisenberg und Noe, Handformeln der Zerbrechlichkeit, Insolvenzkosten.

Firma i schuldet Firma j den Betrag W[i, j]; L_i = Summe der Schulden von i; Pi = W / L (relative Schuld). Externe Mittel e_i. Verrechnungsvektor:
p_i = min(L_i, max(e_i + sum_j Pi_ji p_j, 0)), der größte Fixpunkt (Iteration von p = L aus). Mit Insolvenzkosten zahlt eine ausfallende Firma nur den Anteil alpha
ihrer Mittel (vereinfachte Fassung nach Rogers und Veraart).

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b); die Komposition steht in `zkt_evaluation.py`."""

import math

import numpy as np

import zkt_constants as C


def relative_liabilities(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gesamtschuld L und relative Schuld Pi = W / L (Firmen ohne Schulden haben eine Nullzeile)."""
    L = w.sum(axis=1)
    pi = np.divide(w, L[:, None], out=np.zeros_like(w), where=L[:, None] > 0)
    return L, pi


def external_assets(w: np.ndarray, b: float) -> np.ndarray:
    """Externe Mittel so, dass ohne Schock niemand ausfällt: max(1 - Forderungen, 0) + Puffer b (jede Firma schuldet insgesamt 1)."""
    return np.maximum(1.0 - w.sum(axis=0), 0.0) + b


def apply_shock(e: np.ndarray, hit, s: float) -> np.ndarray:
    """Die Firmen `hit` verlieren je s von ihren externen Mitteln (e darf negativ werden: externe Verluste)."""
    out = np.array(e, dtype=float)
    out[list(hit)] -= s
    return out


def pick_hit(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """k zufällig gewählte, verschiedene geschockte Firmen (aufsteigend sortiert)."""
    return np.sort(rng.choice(n, size=k, replace=False))


def clear(w: np.ndarray, e: np.ndarray, start: str = "gross", alpha: float | None = None,
          tol: float = C.FIX_TOL, max_iter: int = C.MAX_ITER) -> tuple[np.ndarray, np.ndarray, int]:
    """Verrechnungsvektor als Fixpunkt von p -> min(L, max(e + Pi^T p, 0)) (mit Insolvenzkosten: zahlt eine Firma weniger als L, nur alpha * Mittel).

    start = "gross": Iteration von p = L aus (größter Fixpunkt), "klein": von p = 0 aus. Gibt (p, L, Zahl der Schritte) zurück."""
    L, pi = relative_liabilities(w)
    p = L.copy() if start == "gross" else np.zeros_like(L)
    for step in range(1, max_iter + 1):
        assets = np.maximum(e + pi.T @ p, 0.0)
        nxt = np.minimum(L, assets) if alpha is None else np.where(assets >= L - 1e-15, L, alpha * assets)
        if np.abs(nxt - p).max() < tol:
            return nxt, L, step
        p = nxt
    return p, L, max_iter


def residual(w: np.ndarray, e: np.ndarray, p: np.ndarray, alpha: float | None = None) -> float:
    """Unabhängiger Prüfer: größte Abweichung von p_i = min(L_i, max(e_i + sum_j Pi_ji p_j, 0)) (mit Kosten: der Verzweigung)."""
    L, pi = relative_liabilities(w)
    assets = np.maximum(e + pi.T @ p, 0.0)
    expect = np.minimum(L, assets) if alpha is None else np.where(assets >= L - 1e-15, L, alpha * assets)
    return float(np.abs(p - expect).max()) if len(p) else 0.0


def failed(p: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Welche Firmen zahlen weniger als ihre Schuld?"""
    return p < L - C.FAIL_TOL


# --- Handformeln --------------------------------------------------------------------------------------------------------------------------

def ring_defaults(s: float, b: float, n: int) -> int:
    """Zahl der ausfallenden Firmen im Ring bei EINEM Schock s: ceil(min(s - b, 1) / b), höchstens n (0, wenn der Schock den Puffer nicht übersteigt).
    Jede Firma fängt b des Fehlbetrags auf und reicht den Rest weiter."""
    deficit = min(s - b, 1.0)
    if deficit <= 0:
        return 0
    return min(n, math.ceil(deficit / b - 1e-12))


def cascade_possible(b: float, n: int, k: int) -> bool:
    """Im vollständigen Netz gibt es eine Kaskade über die k geschockten Firmen hinaus genau dann, wenn k > b (n - 1)."""
    return k > b * (n - 1)


def complete_threshold(b: float, n: int, k: int) -> float | None:
    """Schwelle s* = b n / k des vollständigen Netzes: darunter fallen nur die k geschockten Firmen aus, darüber alle. None, wenn es keine Kaskade gibt."""
    return b * n / k if cascade_possible(b, n, k) else None
