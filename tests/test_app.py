"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def _app_with(**state) -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    for key, value in state.items():
        at.session_state[key] = value
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_closed_group_without_means_shows_the_multiple_fixed_points_hint():
    """Ring, alle Firmen geschockt mit s = b: geschlossene Gruppe ohne externe Mittel, größter und kleinster Fixpunkt unterscheiden sich (Lücke 1,0). Die App sagt es, statt Eindeutigkeit zu zeigen."""
    at = _app_with(kind_radio="ring", n_slider=10, b_slider=0.1, k_slider=10, s_slider=0.1)
    assert any("Mehrere Verrechnungsvektoren" in i.value for i in at.info)
    assert at.metric[3].value == "mehrere"


def test_default_net_shows_no_multiple_fixed_points_hint_and_no_iteration_warning():
    at = _fresh_app()
    assert not any("Mehrere Verrechnungsvektoren" in i.value for i in at.info)
    assert not any("Höchstwert" in w.value for w in at.warning)
