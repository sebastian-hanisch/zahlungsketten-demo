"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import zkt_evaluation as ev


def test_shock_verdict_branches():
    kind, text = ev.verdict_shock(0, 4, 40)
    assert kind == "success" and "Kein Ausfall" in text and "4 geschockten" in text
    kind, text = ev.verdict_shock(4, 4, 40)                                           # genau die geschockten Firmen
    assert kind == "info" and "4 von 40" in text and "nicht weiter" in text
    kind, text = ev.verdict_shock(10, 1, 40)
    assert kind == "info" and "**10** von 40" in text and "25,0 %" in text and "10,0-Fache" in text
    kind, text = ev.verdict_shock(40, 4, 40)
    assert kind == "warning" and "**40** von 40" in text and "100,0 %" in text and "10,0-Fache" in text


def test_shock_verdict_names_the_insolvency_costs():
    assert "Insolvenzkosten: eine ausfallende Firma zahlt 80,0 % ihrer Mittel" in ev.verdict_shock(5, 2, 20, 0.8)[1]
    assert "Insolvenzkosten" not in ev.verdict_shock(5, 2, 20)[1]


def test_threshold_verdict_both_branches():
    kind, text = ev.verdict_threshold(0.1, 40, 3, None, None, None)
    assert kind == "success" and "k = 3 ≤ b (n − 1) = 3,90" in text and "keine Kaskade" in text
    kind, text = ev.verdict_threshold(0.1, 40, 4, 1.0, 10.0, 100.0)
    assert kind == "warning" and "s* = b·n/k = 1,000" in text and "10,0 %" in text and "100,0 %" in text and "die 4 geschockten" in text


def test_cost_verdict_both_branches():
    kind, text = ev.verdict_cost(0.8, 1.0, 0.107, 25, 100)
    assert kind == "warning" and "α = 0,80" in text and "10,7 %" in text and "**100,0 %**" in text and "25 von 100" in text
    kind, text = ev.verdict_cost(0.8, 0.1, 0.1, 0, 100)
    assert kind == "info" and "ändern die Insolvenzkosten hier nichts" in text and "10,0 %" in text and "0 von 100" in text
    assert ev.verdict_cost(0.8, 0.1, 0.1 + 1e-13, 0, 100)[0] == "info"                # Rundungsrauschen zählt nicht als Anstieg


def _rows(values):
    return [{"s": round(0.1 * i, 1), "mean": v} for i, v in enumerate(values)]


def test_forms_verdict_with_a_jump():
    forms = {"vollstaendig": _rows([0.0, 10.0, 10.0, 100.0, 100.0]), "ring": _rows([0.0, 20.0, 60.0, 70.0, 71.7]),
             "zufall": _rows([0.0, 5.0, 20.0, 26.5, 27.0]), "kernrand": _rows([0.0, 8.7, 8.7, 9.1, 8.7])}
    kind, text = ev.verdict_sweep_forms(forms, 40, 4)
    assert kind == "info"
    for part in ("bis s = 0,2 bei 10,0 %", "die 4 geschockten Firmen", "ab s = 0,3 fallen **100,0 %**", "Der Ring verliert höchstens 71,7 %", "das Zufallsnetz 27,0 %", "Kern–Rand-Netz 9,1 %"):
        assert part in text, part


def test_forms_verdict_without_a_jump():
    forms = {"vollstaendig": _rows([0.0, 2.5, 2.5]), "ring": _rows([0.0, 25.0, 25.0]), "zufall": _rows([0.0, 5.0, 6.4]), "kernrand": _rows([0.0, 2.2, 2.2])}
    kind, text = ev.verdict_sweep_forms(forms, 40, 1)
    assert kind == "info" and "höchstens 2,5 %" in text and "keine Kaskade" in text and "Ring verliert höchstens 25,0 %" in text
