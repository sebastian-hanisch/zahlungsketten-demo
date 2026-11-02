"""Zahlungsketten und Ausfallkaskaden: robust und zerbrechlich - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 6 der Reihe „Spiele auf Netzen“: Firmen schulden einander Beträge (Zahlungsziele), einige verlieren externe Mittel. Die Verrechnung nach Eisenberg und Noe
bestimmt, wer wie viel zahlt; die Demo zeigt, wie sich ein Ausfall je nach Netzform ausbreitet und wo die Schwelle liegt, an der das dichteste Netz kippt.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import zkt_algorithm as A
import zkt_constants as C
from zkt_evaluation import (
    Settings,
    analyse,
    complete_below_above,
    count_nonunique,
    first_above,
    nearest_row,
    sweep_forms,
    threshold_table,
    verdict_cost,
    verdict_shock,
    verdict_sweep_forms,
    verdict_threshold,
)
from zkt_layout import dunkles_design, kopf_css
from zkt_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from zkt_scenario import make_network
from zkt_visualization import (
    build_cost_sweep,
    build_forms_sweep,
    build_network,
    build_payments,
    build_threshold_heatmap,
)

st.set_page_config(page_title="Zahlungsketten – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Rechne die vier Netzformen über viele Schockstärken …")
def _forms(n, b, k, alpha):
    return sweep_forms(n, b, k, alpha)


@st.cache_data(show_spinner=False)
def _nonunique(kind, alpha):
    return count_nonunique(kind, alpha)


def _de(v, d=2):
    return f"{v:.{d}f}".replace(".", ",")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in zkt_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("💸 Zahlungsketten und Ausfallkaskaden: robust und zerbrechlich")
st.markdown(
    """
Firmen schulden einander **Zahlungsziele**. Verliert eine Firma externe Mittel, kann sie ihre Schulden nicht voll zahlen, ihre Gläubiger bekommen weniger und fallen vielleicht selbst aus. Die **Verrechnung** (Eisenberg und Noe)
legt fest, wer wie viel zahlt: Jede Firma zahlt, was sie hat, höchstens ihre Schuld. Die Demo zeigt, **wie weit ein Ausfall läuft**, je nach Form des Netzes, und warum das **dichteste Netz** nur bis zu einer **Schwelle**
sicherer als Ring und Zufallsnetz ist: Darunter fallen nur die geschockten Firmen aus, darüber alle. Mit Insolvenzkosten kippt sogar der Ring. Stück 6 der Reihe **Spiele auf Netzen**; die Netze sind synthetisch, es gibt keine Daten echter Firmen.

Weiter unten: vier Abschnitte **🔬 Experiment**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Verrechnung", expanded=True):
    st.markdown(
        """
1. **Die Schulden.** Firma $i$ schuldet Firma $j$ den Betrag $W_{ij}$; jede Firma schuldet insgesamt 1. Jede Firma hat externe Mittel $e_i$, mit einem **Puffer** $b$ über dem, was sie braucht, um ihre Schuld zu decken.
2. **Der Schock.** $k$ Firmen verlieren je $s$ ihrer externen Mittel; sie können dann nicht mehr voll zahlen.
3. **Die Verrechnung.** Jede Firma zahlt $p_i = \\min(L_i, \\max(e_i + \\sum_j \\Pi_{ji} p_j, 0))$: ihre Mittel plus das, was sie von ihren Schuldnern bekommt, höchstens ihre Schuld $L_i$. Der **größte Fixpunkt** (Iteration von $p = L$ aus) ist der Verrechnungsvektor; ohne Insolvenzkosten ist er in den meisten Konstellationen eindeutig.
4. **Die Ausbreitung.** Eine Firma fällt aus, wenn $p_i < L_i$. Eine Firma fängt höchstens ihren Puffer plus ihre überschüssigen Forderungen auf und reicht den Rest weiter.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz**")
    kind = st.radio("Netzform", C.KINDS, key="kind_radio", format_func=lambda k: C.KIND_LABEL[k], help="Wer wem schuldet.")
    n = st.slider("Firmen n", *bounds("n_slider"), key="n_slider", help="Wie viele Firmen im Netz sind.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Bestimmt das Zufalls- und das Kern–Rand-Netz und die geschockten Firmen.")
    st.markdown("**Schock**")
    b = st.slider("Puffer b", *bounds("b_slider"), key="b_slider", step=C.B_STEP, help="Wie viel externe Mittel jede Firma über ihre Schuld hinaus hat.")
    k = st.slider("Geschockte Firmen k", *bounds("k_slider"), key="k_slider", help="Wie viele Firmen externe Mittel verlieren.")
    s = st.slider("Schock s je Firma", *bounds("s_slider"), key="s_slider", step=C.S_STEP, help="Wie viel jede geschockte Firma verliert (ihre Schuld ist 1).")
    st.markdown("**Insolvenzkosten**")
    alpha = st.slider("Auszahlung α einer ausfallenden Firma", *bounds("alpha_slider"), key="alpha_slider", step=C.ALPHA_STEP,
                      help="1 = keine Kosten. Darunter zahlt eine ausfallende Firma nur diesen Anteil ihrer Mittel (vereinfachte Fassung nach Rogers und Veraart).")
    st.button("🎲 Neues Netz und neue Firmen", width="stretch", on_click=randomize_seed)

sync_query_params({"kind_radio": kind, "n_slider": int(n), "b_slider": round(float(b), 2), "k_slider": int(k), "s_slider": round(float(s), 2),
                   "alpha_slider": round(float(alpha), 2), "seed_input": int(seed)})

settings = Settings(kind, int(n), round(float(b), 2), int(k), round(float(s), 2), round(float(alpha), 2), int(seed))
a = _analysis(settings)
cost_alpha = a.alpha

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Ausgefallene Firmen", f"{a.n_failed} von {settings.n}", delta=f"{_de(100 * a.share, 1)} %", delta_color="off")
m2.metric("Geschockte Firmen", f"{len(a.hit)}", delta=f"je {_de(settings.s)} Schock", delta_color="off")
m3.metric("Schritte der Verrechnung", f"{a.steps}", delta=f"Abweichung {a.residual:.1e}", delta_color="off")
m4.metric("Eindeutigkeit", "eindeutig" if a.gap < 1e-9 else "mehrere", delta=f"größter − kleinster Fixpunkt {a.gap:.1e}", delta_color="off")
st.plotly_chart(build_network(a.net.w, a.failed, a.hit, a.p, a.L), width="stretch", key="main_chart")
st.caption("Rot: fällt aus (zahlt weniger als die Schuld), grün: zahlt voll, oranger Rand: geschockt. Die Anordnung auf dem Kreis folgt nur der Nummerierung.")
if a.residual > 1e-9:
    st.error("Die Verrechnung erfüllt die Fixpunktgleichung nicht.")
with st.expander("Zahlung je Firma"):
    st.plotly_chart(build_payments(a.p, a.L, a.hit), width="stretch", key="payments_chart")
    st.caption("Zahlung im Verhältnis zur Schuld; geschockte Firmen orange, ausgefallene rot.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Bleibt der Ausfall bei den geschockten Firmen, oder läuft er durch das Netz?")
kind_v, text_v = verdict_shock(a.n_failed, len(a.hit), settings.n, a.alpha)
getattr(st, kind_v)(text_v)

st.markdown("---")

# --- Experiment 1: Netzform gegen Schockstärke ---------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Netzform gegen Schockstärke – robust und zerbrechlich")
st.caption(f"Ausfallanteil über die Schockstärke für alle vier Netzformen (n = {settings.n}, b = {_de(settings.b)}, k = {settings.k}, ohne Insolvenzkosten), "
           f"über {len(C.SWEEP_SEEDS)} feste Seeds mit je {C.SWEEP_RUNS} Läufen; Band = kleinster und größter Seed-Mittelwert.")
forms = _forms(settings.n, settings.b, settings.k, None)
star = A.complete_threshold(settings.b, settings.n, settings.k)
st.plotly_chart(build_forms_sweep(forms, star), width="stretch", key="forms_chart")
kind_f, text_f = verdict_sweep_forms(forms, settings.n, settings.k)
getattr(st, kind_f)(text_f)

st.markdown("---")

# --- Experiment 2: Puffer und Zahl der Schocks ------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Puffer und Zahl der Schocks – die Schwelle in geschlossener Form")
st.markdown("Im **vollständigen Netz** fallen darunter nur die $k$ geschockten Firmen aus, darüber alle. Die Schwelle ist $s^* = b\\,n/k$, und sie gibt es nur, wenn $k > b\\,(n-1)$. Im **Ring** fängt jede Firma den Puffer $b$ des Fehlbetrags auf: "
            "Ein Schock $s$ lässt $\\lceil \\min(s-b, 1)/b \\rceil$ Firmen ausfallen.")
bs = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
ks = tuple(range(1, 11))
st.plotly_chart(build_threshold_heatmap(bs, ks, threshold_table(settings.n, bs, ks), settings.n), width="stretch", key="threshold_chart")
star_now, below, above = complete_below_above(settings.n, settings.b, settings.k)
kind_t, text_t = verdict_threshold(settings.b, settings.n, settings.k, star_now, below, above)
getattr(st, kind_t)(text_t)
ring_w = make_network("ring", settings.n, 0).w
e_ring = A.apply_shock(A.external_assets(ring_w, settings.b), [0], settings.s)
p_ring, L_ring, _ = A.clear(ring_w, e_ring)
measured = int(A.failed(p_ring, L_ring).sum())
st.markdown(f"**Ring, ein Schock s = {_de(settings.s)}:** Formel ⌈min(s − b, 1)/b⌉ = **{A.ring_defaults(settings.s, settings.b, settings.n)}** ausfallende Firmen, gemessen **{measured}**.")

st.markdown("---")

# --- Experiment 3: Insolvenzkosten ------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Insolvenzkosten – wenn ein Ausfall Werte vernichtet")
alpha_used = cost_alpha if cost_alpha is not None else C.COST_ALPHA
st.caption(f"Eine ausfallende Firma zahlt nur {_de(100 * alpha_used, 0)} % ihrer Mittel (α = {_de(alpha_used)}; Regler links, 0,80 wenn dort 1 steht). Gestrichelt: ohne Kosten, durchgezogen: mit Kosten (n = {settings.n}, b = {_de(settings.b)}, k = {settings.k}).")
forms_cost = _forms(settings.n, settings.b, settings.k, alpha_used)
st.plotly_chart(build_cost_sweep(forms, forms_cost), width="stretch", key="cost_chart")
row_free, row_cost = nearest_row(forms[settings.kind], settings.s), nearest_row(forms_cost[settings.kind], settings.s)
nonunique = _nonunique(settings.kind, alpha_used)
kind_c, text_c = verdict_cost(alpha_used, row_cost["mean"] / 100, row_free["mean"] / 100, nonunique, C.NONUNIQUE_NETS)
st.markdown(f"**{C.KIND_LABEL[settings.kind]}, s = {_de(row_free['s'], 1)}** (nächster Rasterpunkt):")
getattr(st, kind_c)(text_c)
st.caption(f"Netze mit mehreren Verrechnungsvektoren: je {C.NONUNIQUE_NETS} Zufallsnetze mit n = {C.NONUNIQUE_N}, drei geschockten Firmen, s = 0,9 und b = 0,1 (die größte Lösung wird gezeigt).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine Verrechnungsrunde, keine Zeit** | Zahlungsziele laufen zu verschiedenen Terminen; die Demo rechnet den Endzustand, nicht den Ablauf. | – |
| **Alle Schulden gleichrangig, alle Firmen gleich groß** (Schuld 1) | Bei verschieden großen Firmen verschiebt sich die Schwelle; die Formeln gelten nur für diese Netze. | – |
| **Keine Notverkäufe, keine Marktpreise** | Wer Vermögen verkaufen muss, drückt den Preis und schadet anderen: ein weiterer Kanal, der hier fehlt. | – |
| **Insolvenzkosten vereinfacht** (α · Mittel) | Das ist keine Nachbildung des Modells von Rogers und Veraart; Zahl und Lage der Verrechnungsvektoren hängen vom Modell ab. | – |
| **Synthetische Netze, feste Parameter** | Echte Zahlungsnetze sind weder Ring noch vollständig; die Aussage „dicht ist bis zur Schwelle am sichersten“ gilt hier gegenüber Ring und Zufallsnetz, nicht gegenüber Kern–Rand. | – |
| **Festes Netz** | Wer sich neu verbindet, ändert das Spiel: Das Netz entsteht hier nicht. | Stück 5 (Netzbildung unter Ausfallrisiko) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Schulden und Verrechnung.** $W_{ij} \ge 0$ ist die Schuld von $i$ an $j$, $L_i = \sum_j W_{ij}$, $\Pi_{ij} = W_{ij}/L_i$. Externe Mittel $e_i$. Ein **Verrechnungsvektor** $p \in [0, L]$ erfüllt
$p_i = \min\bigl(L_i,\ \max(e_i + \sum_j \Pi_{ji} p_j,\ 0)\bigr)$. Die Abbildung ist monoton; der größte Fixpunkt ergibt sich durch Iteration von $p = L$ aus. Für $e \ge 0$ löst der größte Fixpunkt das LP
$\max \sum_i p_i$ unter $p \le L$, $p \le e + \Pi^\top p$ (Eisenberg und Noe 2001); in den geprüften Netzen ist er auch bei negativen $e$ (große Schocks) eindeutig (Messbefund an Stichproben, durch kleinsten Fixpunkt geprüft); mit externen Mitteln 0 für eine geschlossene Gruppe (z. B. alle Firmen eines Rings geschockt mit $s = b$) gibt es mehrere Fixpunkte.

**Aufbau der Demo.** Jede Firma schuldet insgesamt 1; $e_i = \max(1 - \text{Forderungen}_i,\ 0) + b$ (ohne Schock fällt niemand aus). Geschockte Firmen: $e_i \leftarrow e_i - s$.

**Ring, ein Schock.** Der Fehlbetrag der geschockten Firma ist $d_0 = \min(s - b, 1)$, jede Firma auf der Kette fängt $b$ auf: Die Ausfälle sind $\lceil d_0 / b \rceil$ Firmen (höchstens $n$).

**Vollständiges Netz, $k$ Schocks.** Der Fehlbetrag einer geschockten Firma ist $(s-b)(n-1)/(n-k)$ (solange er unter 1 liegt). Die übrigen Firmen verlieren daraus je $k \cdot \text{Fehlbetrag}/(n-1)$; übersteigt das ihren Puffer $b$, fallen alle aus.
Das geschieht für $s > s^* = b\,n/k$, und nur, wenn der größtmögliche Fehlbetrag 1 dafür reicht: $k > b\,(n-1)$. Darunter fallen nur die $k$ geschockten Firmen aus. (Eigene Herleitung, in der Demo gegen die Verrechnung geprüft.)

**Insolvenzkosten (vereinfacht).** Eine Firma, die nicht voll zahlen kann, zahlt nur $\alpha \cdot \max(e_i + \sum_j \Pi_{ji} p_j, 0)$. Die Abbildung bleibt monoton; der größte Fixpunkt ist der Verrechnungsvektor, kleinere Fixpunkte können existieren.

**Literatur** (nur Titel, Abstract und Sekundärquellen geprüft, keine Volltexte). Eisenberg, Noe: *Systemic Risk in Financial Systems.* Management Science 47(2), 2001, S. 236–249.
Acemoglu, Ozdaglar, Tahbaz-Salehi: *Systemic Risk and Stability in Financial Networks.* American Economic Review 105(2), 2015, S. 564–608.
Elliott, Golub, Jackson: *Financial Networks and Contagion.* American Economic Review 104(10), 2014, S. 3115–3153.
Rogers, Veraart: *Failure and Rescue in an Interbank Network.* Management Science 59(4), 2013, S. 882–898.
Glasserman, Young: *How likely is contagion in financial networks?* Journal of Banking & Finance 50, 2014, S. 383–399.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
