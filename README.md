# Zahlungsketten und Ausfallkaskaden: robust und zerbrechlich – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-zahlungsketten-demo.streamlit.app/)** -->

Drittes Stück der Reihe **Spiele auf Netzen** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (Vorgänger: [netzwerkspiel-demo](https://github.com/sebastian-hanisch/netzwerkspiel-demo) und [oeffentliche-gueter-netz-demo](https://github.com/sebastian-hanisch/oeffentliche-gueter-netz-demo); geplante Folge: Netzbildung unter Ausfallrisiko). Anders als die Fall-Demos zeigt sie **ein** Verfahren an einem wachsenden Beispiel: die **Verrechnung von Zahlungsketten** nach Eisenberg und Noe. Vehikel: ein synthetisches Netz von Firmen (Zulieferer, Spediteure, Frachtbörse), die einander Zahlungsziele schulden; einige verlieren externe Mittel. Keine Banken, keine Namen, keine Daten echter Firmen.

## Warum dieses Problem

Wer seine Rechnung nicht zahlt, schwächt seine Gläubiger. Ob daraus eine Kette wird, hängt von der Form des Netzes und vom Puffer ab, und die Antwort ist nicht „dichter ist sicherer“: Das **dichteste Netz** (jeder schuldet jedem) fängt kleine Schocks vollständig ab und kippt an einer **Schwelle**, an der **alle** Firmen ausfallen. Die Demo zeigt diese Schwelle in geschlossener Form, vergleicht vier Netzformen und fragt, was Insolvenzkosten ändern.

## Befunde und Korrekturen gegenüber dem Plan

- **Die erste Formel für die Schwelle des vollständigen Netzes war falsch.** Der Plan nannte $b + b(n-1)/k$ und übersah, dass auch die anderen geschockten Firmen weniger zahlen. Richtig ist **$s^* = b\,n/k$**, und es gibt eine Kaskade nur, wenn $k > b\,(n-1)$ (Tabelle 3, geprüft in 200 von 200 Fällen).
- **Die Messgröße „Fehlbetrag / Schock“ ist verworfen.** Sie zählt dieselbe Schuld auf jeder Stufe der Kette erneut und ist kein Verlustmaß. Gemessen wird der **Anteil ausgefallener Firmen**.
- **„Dicht ist unterhalb der Schwelle am sichersten“ gilt nur gegenüber dem Ring.** Das Kern–Rand-Netz verliert bei vier Schocks 8,7 % und damit weniger als das vollständige Netz (10,0 %), selbst oberhalb von dessen Schwelle (Tabelle 1): Einige Firmen haben dort mehr externe Mittel (bei wenigen Forderungen ist $e = 1 - 	ext{Forderungen} + b$ größer), und die Ausfälle bleiben bei den geschockten Firmen.

## Modell

Firma $i$ schuldet Firma $j$ den Betrag $W_{ij}$, $L_i = \sum_j W_{ij}$ (jede Firma schuldet insgesamt 1), $\Pi_{ij} = W_{ij}/L_i$. Externe Mittel $e_i = \max(1 - \text{Forderungen}_i, 0) + b$ mit Puffer $b$: Ohne Schock fällt niemand aus. Schock: $k$ zufällige Firmen verlieren je $s$. **Verrechnungsvektor** $p_i = \min(L_i, \max(e_i + \sum_j \Pi_{ji} p_j, 0))$, der größte Fixpunkt (Iteration von $p = L$ aus); eine Firma fällt aus, wenn $p_i < L_i$. **Insolvenzkosten (vereinfacht):** eine ausfallende Firma zahlt nur $\alpha \cdot$ ihre Mittel. Netzformen: Ring, Zufall (3 Gläubiger), Kern–Rand (5 Kernfirmen, jede Randfirma schuldet zwei Kernfirmen je die Hälfte), vollständig. Formeln im 📐-Expander der App.

## Methodik

Die Ergebnisansicht zeigt eine Verrechnung (Firmen, Zahlungen, Fixpunktprüfer, Eindeutigkeit); drei Experimente rechnen über feste Seeds (`SWEEP_SEEDS`, 5 Seeds × 10 Läufe je Zelle, je Seed feste Netze und feste geschockte Firmen: der Vergleich über $s$ ist gepaart):

1. **Netzform gegen Schockstärke:** Ausfallanteil über $s$ für alle vier Formen.
2. **Puffer und Zahl der Schocks:** die Schwelle $s^*$ als Fläche über $(b, k)$, Gegenprobe gegen die Verrechnung; Ring: Kettenlänge $\lceil \min(s-b, 1)/b \rceil$.
3. **Insolvenzkosten:** Ausfallanteil ohne und mit Kosten; Zahl der Netze mit mehreren Verrechnungsvektoren.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen sind in `tests/test_claims.py` nachgerechnet; die Tabellen werden dort Zeile für Zeile gegen diese Datei geprüft (`tools/readme_rows.py` erzeugt sie). Tabellen 1 und 4: n = 40, b = 0,1, 50 Läufe je Zelle.

**Standardeinstellung** (vollständiges Netz, n = 40, b = 0,1, k = 4, s = 0,9, Seed 35): Die Verrechnung konvergiert nach 13 Schritten, 4 von 40 Firmen fallen aus, nur die geschockten. Bei s = 1,2 fallen alle 40 aus.

**Verrechnungsvektor.** In 60 von 60 kleinen Zufallsnetzen (3 bis 5 Firmen, auch mit negativen externen Mitteln) steht der Iterationswert in der Liste, die die **Aufzählung aller Zustandsmuster** liefert (zahlt nichts / einen Teil / voll, je Muster ein lineares System); bei 40 von 40 Netzen, in denen jede Firma etwas schuldet, ist dieser Vektor **der einzige**. Gegen das LP von Eisenberg und Noe (e ≥ 0) stimmt er in 60 von 60 Netzen auf 10⁻⁸ überein; größter und kleinster Fixpunkt sind in 80 von 80 Netzen gleich, auch mit negativen e.

**Tabelle 1: Netzform gegen Schockstärke.** Anteil ausgefallener Firmen in %, Mittel über 50 Läufe.

| k | Netzform | s = 0,1 | 0,5 | 0,9 | 1,0 | 1,1 | 1,5 | 2,0 |
|---|---|---|---|---|---|---|---|---|
| 1 | Ring | 0,0 % | 10,0 % | 20,0 % | 22,5 % | 25,0 % | 25,0 % | 25,0 % |
| 1 | Zufall (3 Gläubiger) | 0,0 % | 4,0 % | 4,8 % | 4,8 % | 5,4 % | 5,9 % | 6,0 % |
| 1 | Kern–Rand | 0,0 % | 2,2 % | 2,2 % | 2,2 % | 2,2 % | 2,2 % | 2,2 % |
| 1 | vollständig | 0,0 % | 2,5 % | 2,5 % | 2,5 % | 2,5 % | 2,5 % | 2,5 % |
| 4 | Ring | 0,0 % | 42,9 % | 68,2 % | 70,1 % | 71,7 % | 71,7 % | 71,7 % |
| 4 | Zufall (3 Gläubiger) | 0,0 % | 15,0 % | 20,6 % | 21,3 % | 23,2 % | 25,4 % | 26,5 % |
| 4 | Kern–Rand | 0,0 % | 8,7 % | 8,7 % | 8,7 % | 8,7 % | 8,7 % | 8,7 % |
| 4 | vollständig | 0,0 % | 10,0 % | 10,0 % | 10,0 % | 100,0 % | 100,0 % | 100,0 % |

Das ist der Befund „robust und zerbrechlich“ (Acemoglu, Ozdaglar, Tahbaz-Salehi 2015): Im vollständigen Netz fallen bei einem Schock nie mehr als die geschockten Firmen aus (2,5 % = 1 von 40), bei vier Schocks bis s = 1,0 nur diese vier (10,0 %), ab s = 1,1 alle. Der Ring verliert nie alles (höchstens 71,7 %), dafür schon bei kleinen Schocks viel.

**Tabelle 2: Ring, ein Schock.** Ausfälle nach der Formel ⌈min(s − b, 1)/b⌉ und nach der Verrechnung (n = 40, b = 0,1); jede Firma fängt den Puffer $b$ des Fehlbetrags auf und reicht den Rest weiter.

| s | Formel | Verrechnung |
|---|---|---|
| 0,05 | 0 | 0 |
| 0,15 | 1 | 1 |
| 0,25 | 2 | 2 |
| 0,55 | 5 | 5 |
| 0,95 | 9 | 9 |
| 1,05 | 10 | 10 |
| 1,15 | 10 | 10 |
| 3,00 | 10 | 10 |

Auf einem Raster von 1 724 Schockstärken (Schritt 0,005, b = 0,05; 0,1; 0,2) stimmen Formel und Verrechnung in jedem Punkt überein.

**Tabelle 3: Schwelle des vollständigen Netzes** (n = 40). Ausfallanteil knapp unter und knapp über $s^* = b\,n/k$ (Abstand 0,002); eine Kaskade gibt es nur für $k > b\,(n-1)$.

| b | k | b (n − 1) | s* | knapp darunter | knapp darüber |
|---|---|---|---|---|---|
| 0,05 | 2 | 1,95 | 1,000 | 5,0 % | 100,0 % |
| 0,05 | 4 | 1,95 | 0,500 | 10,0 % | 100,0 % |
| 0,10 | 4 | 3,90 | 1,000 | 10,0 % | 100,0 % |
| 0,10 | 8 | 3,90 | 0,500 | 20,0 % | 100,0 % |
| 0,20 | 8 | 7,80 | 1,000 | 20,0 % | 100,0 % |
| 0,10 | 3 | 3,90 | – | keine Kaskade | – |
| 0,20 | 4 | 7,80 | – | keine Kaskade | – |

Geprüft in 200 von 200 Fällen (n = 10, 20, 30, 40; b = 0,05 bis 0,3; k = 1 bis n/2): in 150 gibt es eine Kaskade (knapp unter $s^*$ nur die $k$ geschockten Firmen, knapp darüber alle), in 50 fallen auch bei $s = 50$ nur die geschockten aus. Der Fehlbetrag einer geschockten Firma ist unterhalb des Deckels 1 genau $(s - b)(n-1)/(n-k)$.

**Tabelle 4: Insolvenzkosten** (α = 0,8; k = 4). Ausfallanteil ohne / mit Kosten in %; letzte Spalte: Zahl der Netze mit mehreren Verrechnungsvektoren unter 100 Zufallsnetzen mit n = 20, drei Schocks, s = 0,9, b = 0,1.

| Netzform | s = 0,2 | 0,5 | 1,0 | 2,0 | mehrere Vektoren |
|---|---|---|---|---|---|
| Ring | 10,9 % / 100,0 % | 42,9 % / 100,0 % | 70,1 % / 100,0 % | 71,7 % / 100,0 % | 0 von 100 |
| Zufall (3 Gläubiger) | 6,0 % / 7,7 % | 15,0 % / 19,7 % | 21,3 % / 34,2 % | 26,5 % / 39,9 % | 23 von 100 |
| Kern–Rand | 8,7 % / 8,7 % | 8,7 % / 8,7 % | 8,7 % / 8,7 % | 8,7 % / 8,7 % | 0 von 100 |
| vollständig | 10,0 % / 10,0 % | 10,0 % / 10,0 % | 10,0 % / 100,0 % | 100,0 % / 100,0 % | 0 von 100 |

Im Ring macht schon ein Schock von 0,2 (das Doppelte des Puffers) mit Insolvenzkosten alle Firmen zahlungsunfähig; im vollständigen Netz sinkt die Schwelle unter 1,0. Mit Kosten ist der Verrechnungsvektor nicht mehr eindeutig, aber nicht in jeder Netzform.

## Ehrliche Grenzen

- **Eine Verrechnungsrunde ohne Zeit:** Die Demo rechnet den Endzustand, nicht den Ablauf; gleich große Firmen (Schuld 1), alle Schulden gleichrangig, keine Notverkäufe und keine Marktpreise.
- **Insolvenzkosten vereinfacht:** Das Modell „zahlt $\alpha \cdot$ Mittel“ ist keine Nachbildung von Rogers und Veraart; Zahl und Lage der Verrechnungsvektoren hängen vom Modell ab.
- **Eindeutigkeit bei negativen externen Mitteln** ist ein Messbefund (Aufzählung, größter gegen kleinster Fixpunkt) und kein bewiesener Satz; die Sekundärquelle (Stachurski) nennt Eindeutigkeit bei positiven Schulden immer.
- **Nur n = 40 für die Sweeps** (Tabellen 1 und 4) und die genannten Netzformen; die Formeln der Tabellen 2 und 3 gelten für beliebiges n (geprüft für n = 10 bis 40).
- **Synthetische Netze:** Die Quoten gelten für diese Netzformen und Parameter, nicht für echte Zahlungsnetze; „dicht ist bis zur Schwelle sicher“ gilt gegenüber dem Ring (siehe oben).
- **Literatur:** Alle Angaben beruhen auf Titel, Abstract und Sekundärquellen, nicht auf Volltexten.

## Verifikation

- **Kern:** Verrechnung gegen Handrechnung (zwei Firmen; Ring mit 3 Firmen: p = (0; 0,1; 0,2); Kette mit einer Firma ohne Schulden), Prüfer, Insolvenzkosten (n = 2 von Hand).
- **Orakel (anderer Rechenweg):** Aufzählung aller Zustandsmuster mit linearem System je Muster (kein Iterieren), das LP von Eisenberg und Noe (`scipy`), Iteration von beiden Seiten, Ring-Formel gegen die Verrechnung auf einem Raster und gegen ein zweites, schleifenbasiertes Argument, Schwellenformel des vollständigen Netzes gegen 200 Fälle, Fehlbetragsformel $(s-b)(n-1)/(n-k)$.
- **Eigenschaftstests:** Zahlung zwischen 0 und Schuld, ohne Schock kein Ausfall, Zahlung steigt mit den externen Mitteln, Ausfälle steigen mit Schock und Zahl der Schocks und fallen mit dem Puffer, Umbenennen der Firmen, Skalierung aller Beträge, Insolvenzkosten senken nie die Zahlungen, ohne Kosten eindeutig, Ring verliert nie alle, das vollständige Netz schon.
- **Randfälle:** n = 2, eine Firma ohne Gläubiger, k = n, s = 0, genau auf dem Puffer, Reglergrenzen inklusive, Worst-Case bei n = 40 und b = 0,05.
- **Verdikte** als reine Funktionen an handgesetzten Werten; **Presets, Layout, AppTest-Rauchtest** (jeder Button, jeder Regler an Minimum und Maximum), **Footer**, **Laufzeit-Importe**.
- **Behauptungen:** `tests/test_claims.py` rechnet jede Zahl dieser Datei nach.
- **Umfang:** 93 Tests, rund drei Minuten (AppTest, Orakel mit scipy); `tools/readme_rows.py` ist kein Test, sondern erzeugt die Tabellen, die `test_claims.py` prüft.
- **Nicht gelaufen:** Mutationstests (seit 2026-10-28 nicht mehr Teil des Ablaufs); die Aussagenprüfung durch unabhängige Prüfer steht noch aus.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Ergebnis, Kernfrage, drei Experimente, Grenzen, Formeln |
| `zkt_algorithm.py` | Kern: Verrechnung, Prüfer, Handformeln (Ring, vollständiges Netz), Insolvenzkosten |
| `zkt_constants.py` | Regler-Grenzen, Sweep-Seeds, Presets |
| `zkt_evaluation.py` | Analyse, Sweeps über feste Netze, Schwellentabelle, Verdikte (reine Funktionen) |
| `zkt_layout.py` | Fester Kopf (Stylesheet) |
| `zkt_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button, Auswahlregler |
| `zkt_scenario.py` | Netzformen (Ring, vollständig, Zufall, Kern–Rand), Kreis-Layout |
| `zkt_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/readme_rows.py` | erzeugt die README-Tabellen über dieselben Funktionen wie die App |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur für die LP-Gegenprobe) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Mini-Netze, Orakel, Eigenschaften, Randfälle, Claims, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Notverkäufe und Marktpreise** (Fire Sales), **zeitliche Ansteckung** und Zahlungstermine; **verschieden große Firmen** und Rangfolgen der Gläubiger.
- **Nachbau des Modells von Rogers und Veraart** mit mehrstufigen Insolvenzkosten und Rettungskonsortien.
- **Netzbildung** (Stück 5), in dem die Firmen die Netzform selbst wählen; **PDF-Export** der Ergebnisse.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die CI läuft mit Python 3.12; für die Streamlit Community Cloud ist die Python-Version im Dashboard einzustellen (3.12 empfohlen).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
