# ECO-07: getrennte Fitness, aktuelle Vitalität und Evidenz

**Stand:** Forschungsmodell v0.1, **kein** Kapital- oder Fraßentscheidungs-Automatismus. Das Modul `quant.ecosystem.metrics.vitality` berechnet drei voneinander unabhängige Skalen. Es wertet *ausschließlich* bereits kausal bekannte, cashflow-bereinigte tägliche **Netto**-TWR-Werte aus; es erfindet keine Trades, Fees, Funding-Informationen oder Marktgelegenheiten.

## Messgrößen

- **Lifetime Fitness (0–100)** als experimenteller skalierter Kombinationswert aus profitbezogenem annualisiertem Nettoertrag (55%), Sharpe-Komponente (25%), maximalem Drawdown (20%) und Tail-Risiko. Alle Gewichte sind `MetricConfig`-Parameter; **kein** Komplexitätsmalus. Dies ist ein Vergleichsindikator, kein beweisbarer Alpha-Nachweis.
- **Current Vitality (0–1)**: jüngere *Gelegenheits-/Mark-to-Market-aktive* Ergebnisse werden exponentiell nach der Anzahl unabhängig beobachteter Möglichkeiten gewichtet, **nicht blind nach Kalendertagen**. Zusätzlich berücksichtigt: aktueller Drawdown und Leistungsabfall gegenüber eigenem langjährigen Verhalten. Frühere zweijährige Gewinner verlieren Vitalität, wenn sie sechs Monate unter tatsächlich vorhandenem Edge-/Marktregime leiden.
- **Credibility (0–1)**: effektive unabhängige Stichprobe wird konservativ auf die Anzahl Handelstage/Trades beschränkt und bei positiver serieller Autokorrelation gesenkt. OOS-Status und Datenqualität müssen von externen, prüfbaren Daten stammen; fehlende OOS-Validierung senkt Glaubwürdigkeit. Kleines N bedeutet breite Konfidenz und keine voreilige Jagd-/Kapitalentscheidung.
- **Bark Evidence (0–1)** ist die Langfrist-Robustheits-*Evidenz* (positiver Nettoertrag, geringe Drawdowns, glaubwürdige Stichprobe). Sie ist **nicht** das genetische Bark-Gene selbst, schützt niemals vor harten Stopps und ersetzt nicht das Fressermodell ECO-09.
- **Risiko/ Aktivität:** annualisierte Sharpe (365 Sessions als SOL-Default, für FX/Öl andere Session-Konvention nötig), MDD, aktueller Drawdown, Time Under Water, Ulcer Index, Expected Shortfall unterer 5% der Tage, möglicher Paar-Korrelationswert, ausgeführte Trades, verfügbare `eligible_opportunities`, `expected_signals`, ggf. nicht bediente Signale, Anzahl fehlender Kalendertage.
- **Stille ≠ Versagen:** Ein seltener „Kaktus“, der 10 Monate keine *ex ante* identifizierten Eintrittsgelegenheiten, keine Position/Mark-to-Market-Änderung und keine Signale hatte, wird nicht künstlich monatlich herunterbewertet. Wenn er während dieser Zeit aufgrund offener Positionen Verluste markiert, zählen diese trotzdem. Ex-ante-Gelegenheiten müssen unabhängig/kausal nachvollziehbar sein, sonst kann eine untätige Strategie ihre Vitalität durch „keine Gelegenheit“ vortäuschen.

## Methodische Grenzen

1. **Eingangsdatenvertrag:** `NetDay(day, net_return, trades, eligible_opportunities, expected_signals, known_at)`. `net_return` ist eine tägliche, **cashflow-adjusted** Netto-TWR inkl. Gebühren und Funding aus einer echten oder synthetischen Paper-Execution-Engine. Nicht kumuliertes PnL oder Renko-Signalpreise. Ein realer Datenanbieter muss pro Tag auch gültige Null-Tage liefern; fehlende Börsen-Sessions werden getrennt als Beobachtungslücke ausgewiesen.
2. **Keine Lookahead-Daten:** `known_at` muss spätestens UTC Tagesende sein. OOS-Flag darf nur mit dokumentierten, bislang nicht für die Optimierung genutzten Daten gesetzt werden. Nach einer Optimierung millionenfach getesteter Genome sind multiple Tests und nachträgliche Auswahl für klassische Konfidenzschranken entscheidend; der angenäherte Lower Bound ist ausdrücklich **kein multipeltestbereinigter Signifikanznachweis**.
3. **Bewusste konservative Effektivstichprobe:** aus **Tages**renditen kann man nicht die unabhängige Tradezahl einer intraday gehandelten Strategie genau ermitteln. Wir reduzieren einen naiven Tages-n nach positivem Lag-Autocorrelation-Test und begrenzen auf max. `min(active days, executed trades)`. Mit echten Trade-/Block-Bootstrap-Daten sollte ECO-12 später die Unsicherheit genauer schätzen.
4. **Anfälligkeit für unsaubere Skalierung:** Annualized Return aus 1m-Daten ist kein Beleg, dass Markt-/Asset-Kapital unbegrenzt hebelbar ist. Sharpe ist bei null empirischer Standardabweichung `None`, kein „unendlicher Sharpe“. `MetricResult` trägt getrennte Komponenten für spätere empirisch kalibrierte Gewichtung.
5. **Offene Positionen während Dormanz:** An Tagen ohne neue Einstiege aber mit mark-to-market-geändertem NAV wird Vitalität aktualisiert. Nur echte Ruhe bei fehlender Gelegenheit und fehlendem PnL ist neutral.
6. **Korrelation:** optional und nur bei mindestens 20 aligned Tagen; fehlende Werte sind nicht Nullkorrelation. Finanz-/Sicherheits-Gene sind nie Teil dieser Scorefunktion.
7. **Sicherheit und Kapital:** Dieses Modul vergibt keine Entitlements, trifft keine Predator-Löschung und umgeht keine Loss-/Margin-/Leverage-Caps. ECO-05/06 und ECO-09 konsumieren später nur geprüfte Metriken. Keine Änderung am Legacy Blaupunkt Live-Executor.

## Kommandos / Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests/ecosystem -p test_vitality.py -v
```

Testfälle: ehemaliger Champion 2 Jahre gut/6 Monate schlechter; profitabler zyklischer Busch; zehn Monate ruhender Kaktus; mark-to-market Verlust während Dormanz; acht vs. achthundert Trades; positive serielle Abhängigkeit; echte OOS-Evidenz; unvollständige Tage/Portfoliokorrelation; Future-Data- und Negative-NAV-Return-Rejection.
