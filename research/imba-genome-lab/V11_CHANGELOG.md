# IMBA Genome Parameter Lab v11 – Ranking-Zeiträume

Ergänzt die Auswahlfenster **7 Tage (1 Woche)** und **10 Tage**.
Bestehende Optionen: 14 Tage (Standard), 30 Tage, 90 Tage.

## Semantik
- Zum Beginn jedes neuen 14-Tage-Zyklus werden die sechs besten, **positiven**
  Strategien gemäß der unmittelbar vorherigen 7/10/14/30/90 Kalendertage
  (tatsächliche Bot-Tagesrenditen, 1x) bestimmt.
- Falls nur drei Kandidaten positiv sind, handeln drei; falls keiner positiv ist,
  wird kein neuer Bot eröffnet.
- **Die Neu-Auswahl selbst bleibt alle 14 Tage**, unabhängig vom Ranking-Fenster.
- Sonstige Einstellungen unverändert: 10x in Tagesmodell, individuell einstellbarer
  Drawdown-Stopp, 0,06 % pro Orderseite in den zugrunde liegenden Kurven,
  interne 20-%-Kapitalallokation und Gewinntransfers.
- Kein Lookahead: Auswahl-Rendite endet am Tag vor dem Wechsel.
- Auswahlfenster ist im Portfolio-Cacheschlüssel enthalten: ein Wechsel löst
  eine neue Simulation und eine Aktualisierung des HUD aus.
- Alle historischen Jahre 2022–2026 bleiben verfügbar.

## Qualitätssicherung
Playwright mit lokalem Chromium: Optionen 7/10/14/30 geladen, Top-100-Pool
umschaltbar, Portfolio-Kennzahlen unterscheiden sich, kein JS-Pageerror.
Das bestätigt UI und Berechnungsauslösung, aber nicht TradingView-Live-Parität.

## Dateien
- `imba_genome_parameter_lab_v11_ranking_7_10_tage.html`: autarke Offline-Website
- `imba_genome_app_v11.js`: Stand-alone-App-Code (für Builder)
- `imba_genome_build_v11.py`: Patch für v10-Eingaben