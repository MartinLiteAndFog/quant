# Cloudgarden / Quant — unabhängiger Tradingade-Exit-Test

**Stand: 2. November 2026.** Forschungsmodus, keine Orders, keine Liveänderungen, keine Buchung in Cloudgarden-Kapital/Vitalität. Hier wird **nur der Ausstieg unabhängig** berechnet. Als Startbedingung jedes Tests dient weiterhin ein aus dem Strategy-Builder-Excel stammender TradingView-Einstieg (Zeitpunkt, Richtung, Renko-Einstiegskurs). Insbesondere sind die Ein- und Richtungswechsel-Einstiegssignale **nicht** unabhängig produziert.

## Testbasis

- Vertraulicher Quelle: Binance SOLUSDT 1-Minuten-Kerzen, `SOLUSDT (3).csv(1).gz`; vollständige Daten bis 17.10.2026 **10:30 UTC** auf 5-Minuten-Quellkerzenebene.
- Vertraulicher Vergleich: die sechs Binance-SOL-Strategy-Builder-Exporte, deren geschlossene Referenztrades im normalisierten privaten JSON-Manifest `quant_tv_first_plants_corrected_utc_v2.json` stehen.
- Aufbauregel: historische TradingView-H1-Schlusskerzen bis 07.09.2026 00:00 UTC, danach Binance-5-Minuten-Schlusskerzen, Renko-Box 0,1 USDT, 2-Box-Umkehr, separate Millisekundenstempel und Dochte aus Zwischen-Schlusskurs-Extremen. Aus den **Binance-Rohdaten** rekonstruiert; die TradingView-OHLC-Kerzen werden nicht für die Stop-Berechnung verwendet.
- Für jeden referenzierten Einstieg wird ab dem **nächsten** gebildeten Renko-Brick berechnet: fester SL von 0,9 %, anfänglich **nicht aktiviertes** Trailing-TP mit 0,8 % Rücklauf. Nach einem erstmals beobachteten *günstigen* Hoch/Tief wird der Trailing-Stop aktiv, ohne aktuelle Bar-Extrema rückwirkend auf einen bereits erreichten Stop anzuwenden. Das ist ein durch Referenzabgleich gestütztes, **nicht als Original-Quellcode bestätigtes** Modell.
- Ein erzielter Treffer liefert ein **synthetisches Stopniveau auf Renko**, keinen echten Binance-Fillpreis. Exakter Millisekundenwert bezeichnet die historische Chart-Zeitposition, **nicht** die reale Kenntnis-/Orderzeit. Ausdrücklich keine Strategieperformance.

## Gemessene Exit-Parität

| Prüfung | Datensätze | Bis auf Millisekunde gleicher Exit-Zeitpunkt | Preis innerhalb 0,02 USDT **und** exakt gleicher Zeitpunkt |
|---|---:|---:|---:|
| Alle vergleichbaren SL-/TP-Referenzen | 534 | 486 | **486** (91,0 %) |
| Davon Trailing-TP (`TP`) | 262 | **260** (99,2 %) | **260** |
| Davon Stop-Loss (`SL`) | 272 | **226** (83,1 %) | **226** |
| Nach Deduplikation überlappender Exporte | 278 | **237** (85,3 %) | **237** |

Bei **497/534** ist die korrekte 5-Minuten-Quellkerze getroffen (einschließlich 1–4 ms Verschiebung innerhalb eines Mehrfach-Brick-Bündels). Das erklärt den Unterschied zum strikteren 486-Zähler. Die Rohdatenrekonstruktion umfasst 11.639 Bricks einschließlich Warm-up; der Vergleich betrachtet nur vollständig von Binance abgedeckte Referenzdaten.

Von 763 geschlossenen Binance-Trades sind 534 für diesen SL/TP-Test vergleichbar, 218 haben die Exit-Labels `long`/`short` (Richtungswechsel, hier noch kein unabhängig produzierter Flip) und 11 SL/TP-Trades liegen außerhalb der vollständigen Marktdatenabdeckung.

### Welche Variante weicht ab?

| Seed-SHA-Präfix | Kompatible SL/TP | Exakte Zeit-/Preis-Matches | Einordnung |
|---|---:|---:|---|
| `e667c46efc4b` | 78 | **78** | 17.10. Binance-SOL-Baseline |
| `7d1a0113a983` | 115 | **108** | Entry-Operator-Variante |
| `21ab4b55912e` | 30 | **0** | **RSI deaktiviert, nur Short/SL**: eigene noch nicht erklärte Mechanik |
| `1f159166edd1` | 109 | **101** | Entry-Operator-Variante |
| `acc3c1e067bc` | 124 | **121** | Gemischte Operatoren |
| `cccf89c57ded` | 78 | **78** | zweite Baseline, teilweise dieselben Trades |

Der isolierte Chart-Renko-Test mit direkt aus TradingView gelieferten OHLC-Bricks erreichte zuvor 489/536 exakte Millisekunden-/Preis-Treffer. Unsere unabhängige Binance-Rekonstruktion ergibt **486/534**. Die Differenz hängt unter anderem an der begrenzten Rohdatenabdeckung und an wenigen bekannten Renko-Zeitverschiebungen; sie ist **kein Beweis**, dass ein Live-Bot 91 % seiner realen Exits treffen wird.

**Neue Regelhypothese:** Bei sofort aktivem Trail an der Einstiegsschwelle würde ein Short unter Umständen zu früh geschlossen. Die Referenzzeiten passen deutlich besser, wenn der Trail **erst nach mindestens einer günstigen Bewegung** scharf ist; das Modell kombiniert ihn dann mit dem festen SL. Auch die günstigen Docht-Extrema statt nur der Schlusskurse verbessern den Trefferanteil. Das sind datengetestete Hypothesen, nicht die verifizierte Original-Tradingade-Implementierung.

## Flip-Kontinuität

In den neun ursprünglichen XLSX-Referenzbändern gibt es **383 `long`/`short`-Ausstiege**. **383/383** lassen sich mit einer anschließenden Gegenposition **zum selben Zeitstempel und selben Renko-Preis** verbinden, einschließlich zweier offener Endpositionen. Das ist ein struktureller Konsistenztest des exportierten Journals. Weder Flip-Auslöser noch Ordergrößen, Gebühren, Margin oder Kapitalbuchung sind damit unabhängig validiert. In den sechs Binance-Exporte sind davon 218 Flip-Ausstiege.

## Zwei überprüfbare Beispiele

1. **Passender Short-TP (7. September)**: Referenz-Short bei 105,90 um 01:05 UTC, unabhängig rekonstruierter Trailing-Stop bei **105,84 um 02:35 UTC**, exakt wie TV-TP. Datei `tradingade_exit_match_beispiel.png`.
2. **Nicht passender Short-SL (RSI aus)**: Einstieg bei 106,10 um 02:35 UTC. Unser Modell schließt bei **105,0336 um 09:25 UTC**; TradingView-Referenz sagt **105,75 um 13:15 UTC**. Datei `tradingade_exit_mismatch_rsi_aus.png`. Diese Variante ist bis zur Klärung der Exit-Mechanik für Cloudgarden-Trades **gesperrt**.

## Grenzen und nächster Schritt

- **Entry-Parität** bleibt offen: Die bisherige Elliott-Wave-Parität 147/147 verwendete noch den exportierten TradingView-MTF-Kontext; ein unabhängiger 1h/4h-Pine-`request.security`-Nachbau fehlt. Der RSI ist numerisch gut rekonstruiert; der zuvor festgestellte Unterschied zwischen rückblickender Chartdarstellung und strikt bestätigtem RSI muss für echte Einträge geklärt werden.
- **Exit-Label- und Flip-Parität** sind nicht bewiesen: `SL` im Excel ist nicht zwangsläufig nur ein initial fester Stop; auch nach einem Trailing-Stop kann der ursprüngliche Builder `SL` schreiben. Unsere Maschine berechnet nur den Stopkurs und Zeitpunkt, **nicht** seine proprietäre Labelentscheidung.
- Die **synthetische Renko-Stoppreis-Parität** ist kein Nachweis tatsächlicher Börsen-Fills. Für wirtschaftliche Beurteilungen sind bei Kenntnis des Renko-Signals erst **danach** verfügbare echte Binance-1-Minuten-Marktkurse, tatsächliche Gebühren, Slippage, Funding, Positionsgrößen und konservative Intrabar-Reihenfolge zu verwenden. Die vorherige Fixzeitpunkt-Realpreisprüfung ergab im Schnitt 13,58 bp Ein- und 18,15 bp Ausstiegsdifferenz.
- Keine nachträgliche Kalibrierung an der RSI-aus-Anomalie nur zur Maximierung des Match-Scores. Unabhängige Hypothese und Reproduzierbarkeit zuerst.
- Nicht den aus TV bekannten Referenz-Exit zurück in einen neuen Testlauf einspeisen. **Die Exit-Funktion liest nie den Referenz-Exit.** Referenz-Exit taucht ausschließlich im Vergleicher auf.
- KuCoin/BTC und Kraken erfordern eigene Venue-Quellen; die Binance-Rechnung ist nicht auf ihre Preishistorie übertragbar.

Nächster Bauabschnitt: **vollständige Zustandsmaschine für Entry, Position, Flip, Stop/Trail** aus *unabhängig* berechneten Pine-Signalen und streng bestätigter MTF-RSI-Bedingung. Anschließend echte 1m-Preis-Fills, eine einzige Kapitalbuchung pro Trade, und erst dann Aktivierung der passenden Samen als Cloudgarden-Testpflanzen.

### Reproduktion ohne Live-System

```sh
python3 -m unittest discover -s apps/cloudgarden/scripts -p 'test_tradingade_exit_parity.py'
python3 apps/cloudgarden/scripts/tradingade_exit_parity.py \
  --binance-1m-gz '/private/Quant/SOLUSDT (3).csv(1).gz' \
  --manifest-v2 '/private/Quant/quant_tv_first_plants_corrected_utc_v2.json' \
  --out-dir '/private/Quant/reports/ExitParity_2026-11-02'
```

Ausgaben: `tradingade_exit_reference_anchored_summary.json` (inkl. per Seed und dedupliziert), `tradingade_exit_reference_anchored_details.csv` (jeder Trade und jede Abweichung). Keine privaten TradingView-Tapes, Binance-Rohdaten, Chart-Preise oder Tokens gehören ins GitHub-Repo.