"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import zkt_constants as C
from zkt_scenario import circle_layout


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.28), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_network(w, failed, hit, p, L):
    """Das Netz der Schulden auf einem Kreis: ausgefallene Firmen rot, geschockte mit orangem Rand, übrige grün. Kanten grau (Dicke wie die Schuld)."""
    n = len(w)
    pos = circle_layout(n)
    fig = go.Figure()
    thin = w.max() if w.size else 1.0
    ex, ey = [], []
    for i in range(n):
        for j in range(n):
            if w[i, j] > 0:
                ex += [pos[i, 0], pos[j, 0], None]
                ey += [pos[i, 1], pos[j, 1], None]
    dense = int((w > 0).sum()) > 300      # vollständiges Netz: sehr dünne, blasse Kanten, sonst verdeckt das Gewirr die Firmen
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(150,150,150,0.14)" if dense else C.COLOR_EDGE, width=0.5 if dense else (0.6 if n > 20 else 1.0)), hoverinfo="skip", showlegend=False))
    hitset = set(int(i) for i in hit)
    fail = np.asarray(failed, dtype=bool)
    groups = ((~fail, "zahlt voll", C.COLOR_OK), (fail, "fällt aus", C.COLOR_FAIL))
    size = 22 if n <= 20 else 15
    for mask, name, color in groups:
        idx = np.flatnonzero(mask)
        lines = [dict(color=C.COLOR_SHOCK if int(i) in hitset else "rgba(0,0,0,0.3)", width=3 if int(i) in hitset else 1) for i in idx]
        fig.add_trace(go.Scatter(
            x=pos[idx, 0], y=pos[idx, 1], mode="markers+text", text=[str(int(i) + 1) for i in idx], textposition="middle center", name=name, textfont=dict(size=9),
            marker=dict(size=size, color=color, line=dict(color=[l["color"] for l in lines], width=[l["width"] for l in lines])),
            customdata=np.column_stack([p[idx], L[idx]]) if len(idx) else None,
            hovertemplate="Firma %{text}<br>zahlt %{customdata[0]:.3f} von %{customdata[1]:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="orange Rand: geschockt", marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(color=C.COLOR_SHOCK, width=3))))
    # Gleicher Maßstab nur mit autorange (DEMO-PLAYBOOK §3): Der Rand kommt von zwei unsichtbaren Markern an den Ecken, nicht von einem festen Bereich.
    fig.add_trace(go.Scatter(x=[-1.2, 1.2], y=[-1.15, 1.15], mode="markers", marker=dict(size=1, opacity=0), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=540, margin=dict(l=0, r=0, t=0, b=40), legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center"), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_payments(p, L, hit):
    """Zahlung je Firma im Verhältnis zur Schuld (1 = zahlt voll); geschockte Firmen orange."""
    labels = [str(i + 1) for i in range(len(p))]
    ratio = [float(a / b) if b > 0 else 1.0 for a, b in zip(p, L)]
    hitset = set(int(i) for i in hit)
    colors = [C.COLOR_SHOCK if i in hitset else (C.COLOR_FAIL if r < 1 - C.FAIL_TOL else C.COLOR_OK) for i, r in enumerate(ratio)]
    fig = go.Figure(go.Bar(x=labels, y=ratio, marker_color=colors, hovertemplate="Firma %{x}: zahlt %{y:.1%} der Schuld<extra></extra>"))
    fig.update_xaxes(title_text="Firma", type="category")
    fig.update_yaxes(title_text="Zahlung / Schuld", range=[0, 1.05])
    return _base(fig, 280)


def build_forms_sweep(forms, markers=None):
    """Ausfallanteil über die Schockstärke s für jede Netzform (Mittel, Band über die Seeds); `markers`: senkrechte Linie bei s* (vollständiges Netz)."""
    fig = go.Figure()
    for kind, rows in forms.items():
        xs = [r["s"] for r in rows]
        color = C.COLORS[kind]
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[r["high"] for r in rows] + [r["low"] for r in rows][::-1], mode="lines", fill="toself", fillcolor=color + "22",
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=[r["mean"] for r in rows], mode="lines", name=C.KIND_LABEL[kind], line=dict(color=color, width=3), hovertemplate="s = %{x:.1f}: %{y:.1f} %<extra></extra>"))
    if markers is not None:
        fig.add_vline(x=markers, line=dict(color="rgba(120,120,120,0.8)", dash="dash"), annotation_text="s* vollständig", annotation_position="top left")
    fig.update_xaxes(title_text="Schock s je geschockter Firma")
    fig.update_yaxes(title_text="ausgefallene Firmen (%)", range=[0, 102])
    return _base(fig, 380)


def build_threshold_heatmap(bs, ks, table, n):
    """s* = b n / k über (Puffer b, Zahl der geschockten Firmen k); Felder ohne Kaskade (k <= b (n - 1)) sind grau und als „keine Kaskade“ beschriftet.
    Statt fehlender Werte (Plotly zeichnet dafür Text bei „-Infinity“) steht dort 0 mit eigener Farbe am unteren Ende der Farbskala."""
    top = max([v for row in table for v in row if v is not None] or [1.0])
    z = [[0.0 if v is None else v for v in row] for row in table]
    text = [["keine Kaskade" if v is None else f"{v:.2f}" for v in row] for row in table]
    scale = [[0.0, "rgba(128,128,128,0.35)"], [1e-6, "#d7eeea"], [1.0, "#2F6B65"]]
    fig = go.Figure(go.Heatmap(x=[f"{b:.2f}" for b in bs], y=[str(k) for k in ks], z=z, zmin=0.0, zmax=top, text=text, texttemplate="%{text}", colorscale=scale,
                               colorbar=dict(title=dict(text="s*", side="top"), thickness=10, nticks=4), hovertemplate="b = %{x}, k = %{y}: %{text}<extra></extra>"))
    fig.update_xaxes(title_text="Puffer b", type="category")
    fig.update_yaxes(title_text="geschockte Firmen k", type="category", autorange="reversed")
    return _base(fig, 420)


def build_cost_sweep(forms_free, forms_cost):
    """Ausfallanteil ohne (gestrichelt) und mit Insolvenzkosten (durchgezogen) für jede Netzform."""
    fig = go.Figure()
    for kind in forms_free:
        color = C.COLORS[kind]
        fig.add_trace(go.Scatter(x=[r["s"] for r in forms_free[kind]], y=[r["mean"] for r in forms_free[kind]], mode="lines", name=f"{C.KIND_LABEL[kind]}, ohne Kosten", line=dict(color=color, dash="dash")))
        fig.add_trace(go.Scatter(x=[r["s"] for r in forms_cost[kind]], y=[r["mean"] for r in forms_cost[kind]], mode="lines", name=f"{C.KIND_LABEL[kind]}, mit Kosten", line=dict(color=color, width=3)))
    fig.update_xaxes(title_text="Schock s je geschockter Firma")
    fig.update_yaxes(title_text="ausgefallene Firmen (%)", range=[0, 102])
    return _base(fig, 400)
