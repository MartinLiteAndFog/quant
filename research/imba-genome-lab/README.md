# IMBA Genome Parameter Lab — Forschungsstand 05.11.2026

**Status:** Forschungssoftware, **keine Live-Änderungen**. Die hier dokumentierte Browser-App ist v10 (UI-Ausbau auf v9). Historische Renditen und Portfolios sind keine Bestätigung von TradingView-Signalparität, Liquidationssicherheit oder einer live handelbaren Strategie.

## 1. Ziel und Begriffe

Die App soll **keinen statischen Bot-Katalog** darstellen, sondern einen **visuellen Strategie-Geneditor**. Ein Punkt in der Parameterlandschaft bezeichnet eine vollständige, ausführbare Strategie-Konfiguration. Diese besteht aus modularen Genen. Die Benutzerin/der Benutzer soll einen Punkt an beliebigen Parameterkoordinaten setzen, ein Einstiegssignal und optionale Filter/Exits festlegen, eine echte neue Backtest-Reihe berechnen lassen und die Strategie danach in einen Pool und einen Top-6-Portfolio-Test aufnehmen können.

- **Gen:** Baustein einer Strategie (z. B. IMBA-Signal, Renko oder Candle-Zeitbasis, TP, TTP, später RSI, Entry-/Exit-Filter).
- **Parameterfeld:** Visualisierung zweier frei gewählter Gen-Parameter als X/Y; Z und Farbe zeigen tatsächlich berechnete Performance. Achsen z. B. Sensitivity×Renko-Box, TP-Ziel×Sensitivity, TTP-Aktivierung×Sensitivity, TTP-Abstand×Sensitivity.
- **Bot/Strategie:** eindeutig definierte Kombination aller aktiven Gene und ihrer Parameter; nicht bloß ein Punkt mit erfundener Rendite.
- **Genpool:** Kandidatenuniversum tatsächlich berechneter Strategien. Raster ist *eine Auswahl-/Erzeugungsmethode* für Kandidaten, nicht gleichbedeutend mit Bot-Pool oder aktuell handelnden Top 6.
- **Büschel:** laufend aus dem Kandidatenuniversum ausgewählte maximal sechs Handelsstrategien. Gewählte, tatsächlich aktive und individuell gestoppte Bots sind grafisch getrennt.

## 2. Daten und Backtests

1. Ursprüngliche Renko-/IMBA-Familie: 462 diskrete Kombinationen (Sensitivity 5–120 in Fünferschritten, 21 feste Renko-Boxwerte 0,10–0,50 USD). Bei zusätzlichen Modi umfassen die v8/v9-Forschungsdaten 7.326 Varianten: fester USD-Renko, prozentualer Renko und normale 5-Minuten-Candles; mit 9 Exit-Programmen (ohne TP/TTP, TP, TTP und beides).
2. Betrachtungszeitraum: separate historische 2022-, 2023-, 2024-, 2025- und 2026-Tagesdaten. 2026 reicht in diesem Snapshot bis 04.11.2026. Für die 3D-Animation 14/30/90 Tage oder seit Jahresbeginn.
3. **Gebührenentscheidung: 0,06 % pro Orderseite** (0,12 % für vollständigen Entry+Exit); 0,08 % ist überholt.
4. Forschungssimulation: 5-Minuten-Kurse/IMBA-Renko und synthetische Tagesrendite-Auswertung für das 10×-Portfolio. Diese Engine reproduziert **nicht automatisch** die historische 462er-Referenz; Variationen in Signalberechnung, Gebührenbasis, Renko-Anker und TP/TTP-Ausführung sind separat zu validieren.
5. Alle sichtbaren und wählbaren Strategiepunkte müssen eigene echte Backtest-Reihen haben. Bei identischen X/Y-Koordinaten mit verschiedenen anderen Genen: aggregierte Heatmap-Zelle zeigt Max oder Mittel der real vorhandenen Strategien. Fehlende Gene-/Parameterkombinationen **niemals** mit nächstgelegenem Bot gleichsetzen oder durch bloßes Snapping als neu berechnet ausgeben.
6. Neu erzeugte freie Parameterpunkte brauchen zukünftig einen On-demand-Backtest (noch **nicht** fertig implementiert); bis dahin die Raster-/Punkt-Auswahl nur für tatsächlich gerechnete Strategien freigeben und klar anzeigen.

## 3. Pool und Top-6-Regeln

- Standard: alle 14 Tage nach **rein vergangener** 14-Tage-Rendite rangieren. 30 und 90 Tage sind als Forschungsalternative verfügbar. Nur positive Vorperiodenrenditen auswählen; handeln dürfen **null bis sechs** Bots, nicht zwangsläufig sechs.
- Einstiegskapital je separat getestetes Jahr: 1.000 Euro, keine externen Einzahlungen. Kapitaleinsatz aus Reserve monatlich 20 %, mindestens 100 Euro soweit verfügbar. Zehnfacher Hebel **einmal** auf Renditen. Nach +100-%-Gewinnmeilensteinen 50 % des Überschusses in Reserve überführen. Kein Wetterfilter.
- Individuelle Handelspause je Bot: Verlustschwelle einstellbar, Stopp betrifft nur diese Strategie und wird bei der nächsten 14-Tage-Neuwahl zurückgesetzt; Kapital bleibt bis Rotation Cash. **Nicht** das gesamte Portfolio sperren, wenn ein alter Bot gestoppt wurde.
- Im 3D-HUD: Depotwert, Tradingkapital, Reserve, Gewinnrückführungen, gestoppte Bots, aktuelle aktive Anzahl n/6 und Vermögenskurve. Große/warme Marker zeigen tatsächlich handelnde Bots; gewählte, aber pausierte Bots anders; übrige Kandidaten auf Wunsch ausgeblendet.
- Separate Forschungs-Presets: 462 Originale, 2022 Top 100 Raw, 2022 Top 100 Balanced, gemischte Universen, manuelle Punkte, 20×20 und 30×30 Gitter **nur mit tatsächlich getesteten Koordinaten**.

## 4. Frühere Erkenntnisse und Fallstricke

- Archiviertes ursprüngliches 462er-Modell mit **0,08-%-Gebührenbasis**: unabhängig gestartete Jahresdepots 2023 ca. 6,12 EUR, 2024 ca. 498,53 EUR, 2025 ca. 8.305,64 EUR, 2026 bis 16.10. ca. 14.514,31 EUR. Das ursprüngliche 2023er Portfolio wurde anhand der Originaldaten exakt reproduziert. Diese Summen sind **nicht** die heutigen 0,06-%-Ergebnisse.
- Mit 0,06-%-Modell ergeben sich andere Kontrollwerte; frühere Korrektur z. B. Original 462 bis 16.10.2026 ca. 15.588 EUR. Verschiedene Tage/Stichtage und Simulationsmodelle dürfen nicht vermischt werden.
- Ein erweiterter Pool aus historisch starken Strategien verbessert Verlustrisiken **nicht automatisch**. Top-100-2022 mit Korrelation-Cutoff 0,85 war in allen vier *separat* gerechneten Jahren positiv; Cutoff 0,90 zeigte stark andere Werte. **Nachträglich gewählter Cutoff ist overfitting-verdächtig**; Ergebnis nicht als Live-Beweis werten.
- Hohe 3D-Spikes können wenig Trades aufweisen; Plateau-/Diversifikationsauswahl als *Hypothese*, nicht validierte Zukunftsperformance.
- 10× auf Tagesrenditen ignoriert mögliche Intraday-Liquidation; 5-Minuten-Backtests sind ebenfalls keine endgültige TradingView-/Live-Parität. Funding, Slippage und tatsächliche Zeitpunkte fehlen bzw. sind vereinfacht.

## 5. UI-Stand v9 und v10

- **v9**: vier **separate** 3D-Flächen für `none`, `tp`, `ttp`, `both`, individuell ein-/ausblendbar und halbtransparent; Mehrjahresdaten 2022–2026 mit Tagesslider; Preisbasis USD-Renko/%-Renko/5m-Candles; Top-6-HUD; gene badges und Bot-Inspector; 2D-Fallback.
- **v10** ergänzt rein optische Glättung der 3D-Mesh-Werte mit wählbarer Stärke 0/1/2. Originalwerte in Hover, Punktmarkern, Renditekurven und Portfolio-Engine bleiben unverändert. Räumliche Glättung darf niemals Handelsentscheidungen beeinflussen.
- **v10**: `requestAnimationFrame` statt Interval-Queue für zeitliches Abspielen und Zurückspulen, Rendering verarbeitet nur die aktuelle angeforderte Lage (keine Render-Warteschlange). WebGL bleibt je nach Grafikleistung ein Flaschenhals; das ist keine Zusage von 60 fps.
- **v10**: rechte Menüs einzeln über Titel auf-/zuklappbar; ganzer Bereich ausblendbar und Plot vergrößerbar. Dunkle, abgerundete Oberfläche. Browser-Tests für Jahreswechsel, Glättung, Rückwärtsplay, Panels und Console-Errors sind bestanden.
- Später: Punkt durch Klick auf freie Flächenkoordinate erzeugen + **echten neuen Backtest anstoßen**, Genliste bearbeiten, Set/Remove/Delete, konfigurierbare Rasterdichte bis z. B. 30×30 oder mehr, speicherbare eigene Strategiekonfigurationen, Walk-forward-/Train-Test-Auswertung ohne Leakage.

## 6. Quellcode, Daten und Reproduktion

**Quellcode in GitHub:** `research/imba-genome-lab/` auf dem Forschungsbranch `research/imba-genome-lab-2026-11-05`, Dateien `app_v8.js`, `app_v10.js`, `build_v9.py`, `build_v10.py`, `encode_history_v9.py`, `backtest_prior_years_v9.py`, `test_v10.py`, `README.md`.

Die vollständige offline HTML-App ist ein **großes generiertes Artefakt (ca. 24 MB)**. Aus Performance-/Repository-Gründen sind HTML-Bundle und umfangreiche Backtest-Kurven **nicht automatisch Teil des Git-Commmits**. Die Originalinputdaten, insbesondere `history_payload.json`, `historical_7326_equity_2021Dec_2022_2023_2024.npz`, v7/v8-Kurven und die v8-HTML-Basis müssen als Daten-/Build-Artefakte bereitgestellt bzw. separat versioniert werden. Der Build-Snapshot arbeitet noch mit `/mnt/data/`-Pfaden und wird erst nach Parametrisierung als vollständig portable Pipeline gelten.

**Tests:** Browserkontrolle mit Chromium/Playwright, v10 initial 2026, sidebar ein/aus, einzelnes Panel auf/zu, Glättung 2, Slider Tag 15, Jahreswechsel zu 2023, Rückwärtswiedergabe. Keine Page-Errors beobachtet. Dies beweist keine Performance-Parität oder stabile Darstellung auf beliebiger GPU.

**Sicherheit:** Keine Änderungen an Live-Executor, Redis, Railway, Kraken, KuCoin, TradingView-Alarme oder produktiven Branches. Nur eine isolierte Forschungsverzweigung; kein automatischer Merge/Deploy.