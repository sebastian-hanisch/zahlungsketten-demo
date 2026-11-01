"""Gemeinsame Fixtures: die Mini-Netze, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def two_firms():
    """Zwei Firmen, die einander 1 schulden. Mit e = (0,2; 0,2) zahlen beide voll (p = (1, 1)); mit e = (-0,5; 0,2) ist p = (0; 0,2)."""
    return np.array([[0.0, 1.0], [1.0, 0.0]])


@pytest.fixture
def ring3():
    """Ring mit 3 Firmen (0 -> 1 -> 2 -> 0, Schuld je 1). Mit b = 0,1 und Schock 0,5 auf Firma 0: p = (0; 0,1; 0,2), alle drei fallen aus."""
    w = np.zeros((3, 3))
    for i in range(3):
        w[i, (i + 1) % 3] = 1.0
    return w


@pytest.fixture
def chain3():
    """Kette 0 -> 1 -> 2: Firma 2 schuldet nichts (L = 0). Mit e = (0; 0,5; 0,5) ist p = (0; 0,5; 0)."""
    w = np.zeros((3, 3))
    w[0, 1] = 1.0
    w[1, 2] = 1.0
    return w
