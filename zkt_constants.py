"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

KINDS = ("ring", "zufall", "kernrand", "vollstaendig")
KIND_LABEL = {"ring": "Ring", "zufall": "Zufall (3 Gläubiger)", "kernrand": "Kern–Rand", "vollstaendig": "vollständig"}

# Regler. Komplexität am Maximum: Eine Verrechnung ist eine Fixpunktiteration mit einer n x n Matrix-Vektor-Multiplikation je Schritt (n = 40: Mikrosekunden je Schritt,
# höchstens MAX_ITER Schritte); ein Sweep rechnet SWEEP_SEEDS * SWEEP_RUNS Läufe je Zelle (Worst-Case-Test in tests/test_edge_cases.py).
N_MIN, N_MAX, DEFAULT_N = 10, 40, 40
B_MIN, B_MAX, DEFAULT_B, B_STEP = 0.05, 0.3, 0.1, 0.05
K_MIN, K_MAX, DEFAULT_K = 1, 10, 4
S_MIN, S_MAX, DEFAULT_S, S_STEP = 0.0, 3.0, 0.9, 0.05
ALPHA_MIN, ALPHA_MAX, DEFAULT_ALPHA, ALPHA_STEP = 0.5, 1.0, 1.0, 0.05
SEED_MAX, DEFAULT_SEED = 999999, 35
DEFAULT_KIND = "vollstaendig"

CORE_SIZE = 5            # Kern–Rand: so viele Kernfirmen
ZUFALL_CREDITORS = 3     # Zufall: so viele Gläubiger je Firma
MAX_ITER, FIX_TOL = 200000, 1e-13
FAIL_TOL = 1e-9          # eine Firma gilt als ausgefallen, wenn sie weniger als L - FAIL_TOL zahlt

# Sweeps: feste Instanzen, nie ein einzelner Seed (DEMO-PLAYBOOK §4)
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_RUNS = 10                                                  # Läufe je Seed und Zelle: 50 je Zelle
S_GRID = tuple(round(0.1 * i, 1) for i in range(0, 26))          # 0,0 bis 2,5 in Schritten von 0,1
COST_ALPHA = 0.8                                                 # Insolvenzkosten im Sweep des vierten Experiments
NONUNIQUE_N, NONUNIQUE_NETS = 20, 100                            # Netze zur Zahl der Netze mit mehreren Verrechnungsvektoren
FORMULA_EPS = 0.002                                              # Abstand zu s*, bei dem die Schwellenformel geprüft wird

COLOR_FAIL = "#c0392b"
COLOR_OK = "#2F6B65"
COLOR_SHOCK = "#e08a2c"
COLOR_EDGE = "rgba(150,150,150,0.45)"
COLORS = {"ring": "#7a6aa8", "zufall": "#4c78a8", "kernrand": "#2F6B65", "vollstaendig": "#c0392b"}

PRESETS = {
    "Standardfall (Voreinstellung)": {"kind": "vollstaendig", "n": DEFAULT_N, "b": DEFAULT_B, "k": DEFAULT_K, "s": DEFAULT_S, "alpha": DEFAULT_ALPHA, "seed": DEFAULT_SEED},
    "Über der Schwelle": {"kind": "vollstaendig", "n": DEFAULT_N, "b": DEFAULT_B, "k": DEFAULT_K, "s": 1.2, "alpha": DEFAULT_ALPHA, "seed": DEFAULT_SEED},
    "Ring: die Kette läuft": {"kind": "ring", "n": DEFAULT_N, "b": DEFAULT_B, "k": 1, "s": 1.1, "alpha": DEFAULT_ALPHA, "seed": DEFAULT_SEED},
    "Kern–Rand: wenige Ausfälle": {"kind": "kernrand", "n": DEFAULT_N, "b": DEFAULT_B, "k": DEFAULT_K, "s": 1.5, "alpha": DEFAULT_ALPHA, "seed": DEFAULT_SEED},
    "Insolvenzkosten im Ring": {"kind": "ring", "n": DEFAULT_N, "b": DEFAULT_B, "k": DEFAULT_K, "s": 0.2, "alpha": 0.8, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Vollständiges Netz mit 40 Firmen, vier Firmen verlieren 0,9: knapp unter der Schwelle, nur die vier geschockten Firmen fallen aus.",
    "Über der Schwelle": "Dasselbe Netz, Schock 1,2: über der Schwelle s* = b·n/k = 1,0 fallen alle Firmen aus.",
    "Ring: die Kette läuft": "Eine Firma im Ring verliert 1,1: Jede Firma fängt einen Puffer von 0,1 auf, die Kette ist zehn Firmen lang.",
    "Kern–Rand: wenige Ausfälle": "Wenige Kernfirmen halten viele Forderungen: Auch ein großer Schock lässt nur einen kleinen Teil ausfallen.",
    "Insolvenzkosten im Ring": "Eine ausfallende Firma zahlt nur 80 % ihrer Mittel: Schon ein kleiner Schock lässt im Ring alle Firmen ausfallen.",
}
