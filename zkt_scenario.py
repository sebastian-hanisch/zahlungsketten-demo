"""Szenario: synthetische Netze von Zahlungszielen (nur numpy). Zufall nur über den übergebenen Generator, nie über `hash()`.

W[i, j] = Betrag, den Firma i an Firma j schuldet; jede Firma schuldet insgesamt 1. Keine Personen-, Firmen- oder Bankdaten."""

from dataclasses import dataclass

import numpy as np

import zkt_constants as C


@dataclass(frozen=True, eq=False)
class Network:
    w: np.ndarray       # n x n Schuldenmatrix, Diagonale 0, Zeilensummen 1
    kind: str

    @property
    def n(self) -> int:
        return len(self.w)


def ring(n: int) -> np.ndarray:
    """Jede Firma schuldet der nächsten den ganzen Betrag 1 (Kette im Kreis)."""
    w = np.zeros((n, n))
    for i in range(n):
        w[i, (i + 1) % n] = 1.0
    return w


def complete(n: int) -> np.ndarray:
    """Jede Firma schuldet jeder anderen gleich viel: 1/(n - 1)."""
    w = np.full((n, n), 1.0 / (n - 1))
    np.fill_diagonal(w, 0.0)
    return w


def random_creditors(n: int, rng: np.random.Generator, k: int = C.ZUFALL_CREDITORS) -> np.ndarray:
    """Jede Firma schuldet k zufällig gewählten anderen Firmen je 1/k."""
    w = np.zeros((n, n))
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for j in rng.choice(others, size=k, replace=False):
            w[i, j] = 1.0 / k
    return w


def core_periphery(n: int, rng: np.random.Generator, core: int = C.CORE_SIZE) -> np.ndarray:
    """Kern–Rand: die ersten `core` Firmen schulden einander gleich viel; jede Randfirma schuldet zwei zufällig gewählten Kernfirmen je die Hälfte."""
    w = np.zeros((n, n))
    for i in range(n):
        if i < core:
            for j in range(core):
                if j != i:
                    w[i, j] = 1.0 / (core - 1)
        else:
            for j in rng.choice(core, size=2, replace=False):
                w[i, j] = 0.5
    return w


def graph_from_rng(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "ring":
        return ring(n)
    if kind == "vollstaendig":
        return complete(n)
    if kind == "zufall":
        return random_creditors(n, rng)
    if kind == "kernrand":
        return core_periphery(n, rng)
    raise ValueError(f"unbekannte Netzform {kind!r}")


def make_network(kind: str, n: int, seed: int) -> Network:
    """Deterministisch aus (kind, n, seed): gleiche Eingabe, gleiches Netz."""
    return Network(graph_from_rng(kind, n, np.random.default_rng(seed)), kind)


def circle_layout(n: int) -> np.ndarray:
    """Firmen gleichmäßig auf dem Kreis, Firma 0 oben, im Uhrzeigersinn; für die Darstellung, ohne Zufall."""
    angle = np.pi / 2 - 2 * np.pi * np.arange(n) / max(n, 1)
    return np.column_stack([np.cos(angle), np.sin(angle)])
