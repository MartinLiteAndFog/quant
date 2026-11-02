# Quant / Cloudgarden — Elliott-Wave-Pine, RSI und TradingView-Trades

**Ergebnis: sehr hohe Parität der Elliott-Wave-Signale im exportierten Binance-Renko-Chart. Keine vollständige unabhängige Strategie-/Execution-Parität.**

Stand: 2. November 2026. Research only; kein Live-Trading, keine automatischen Orders, kein Gewinn-/Vitalitäts-Booking.

## Eingaben

- Benutzer-Pine: `EW_v3_2_MTF_AUDIT.pine`, vorliegender Upload `Eingefügter Text(4).txt`, 1.405 Zeilen, SHA-256 `5017ca99c4d3ee378b09381c41f2a946e86d772ac21b4eda24a338659a1e5875`.
- TradingView: `BINANCE_SOLUSDT, 5 (5)(1).csv`, 10.264 0,10-USDT-Traditional-Renko-Bricks, enthält `Long Entry`, `Short Entry`, Primary-Pivot-, Pattern- und MTF-Diagnostik. SHA-256 `795f77224dddc873d6b445440b3f03512e2b7274b029ba4d23a1f1c098dd3771`.
- KuCoin: `KUCOIN_SOLUSDT, 5 (2)(1).csv`, 2.744 Renko-Bricks, 2. bis 17. Oktober, SHA-256 `4d9a73ab007344fc97abe2dc16733a9e7ef2672d86865e042815bc4205c40373`.
- Neun Tradingade Strategy Builder XLSX, insgesamt **1.193 geschlossene Trades und vier offene Positionen**. Die originalen Exporte verbleiben in der privaten Quant-Ablage.
- Binance-SOL 1m Rohdaten `.gz` (nicht mit ins Repository übernehmen).

## 1. Elliott-Wave-Logik: überprüfte Ergebnisse

Der unabhängige Python-Port verwendet für die erste Prüfung **die originalen TradingView-Renko-OHLC-Preise**, um Unterschiede in der Renko-Rekonstruktion von der EW-Signallogik zu trennen. Implementiert wurden Pine-Pivot-Tie-Break, Wilder-ATR, Primary-/Secondary-Zigzag, Retracements, Impulse, Zigzag, Flat, Triangle, Diagonal, `findBestPattern`, Wave-3- und Wave-C-Signale. Die Long/Short-Markierungen wurden anhand echter Chart-Spalten verglichen, ohne diese Markierungen für die Signalbildung einzulesen.

| Prüfpunkt | Referenz | Unser Port | Exakt | Interpretation |
|---|---:|---:|---:|---|
| `primaryPushed`, ab Brick 100 | 422 | 422 | **422/422** | Richtige Swing-Einstellungen **Primary 13 / Secondary 5 / ATR 14**, Primary-Multiplikator 3×, Secondary 1,5× |
| Long-/Short-Entry, ab Brick 100 | 149 | 148 | **148/149** | Ein frühes Long-Signal bei Brick 194 fehlt infolge abgeschnittener früher Chart-Historie / Kontext |
| Long-/Short-Entry, ab Brick **206** | **147** | **147** | **147/147**, keine Zusätzlichen | EW/W3/WC mit **importierter** TV-MTF-Konfidenz und TV-HTF1-Trendkontext |
| Pattern bei Primary-Änderungen, ab Brick 100 | 434 | 434 | **432/434** | Zwei frühe Pattern-Differenzen (Bricks 194, 205); nach Brick 206 vollständig passend |

**Wichtige Einschränkung:** Der Port rekonstruiert die gesamte hier getestete CTF-Elliott-Wave-/Zigzag-/Pattern-Logik selbst, aber nutzt `Audit CTF MTF confidence`, `Audit alignment label` und `Audit htf1 trend bias` als TradingView-Referenzfelder. **Die HTF-`request.security`-Berechnung (auf Renko 1h/4h) ist noch kein unabhängiger Nachbau.** Ein vollständiger Paritätsnachweis muss diese drei Spalten selbst erzeugen. Deshalb ist 147/147 nicht als end-to-end unabhängige Feed-/Signal-/Order-Parität zu verstehen.

Originale Pine-Einstiegssignale werden bei bestätigtem neuen Primary-Pivot ausgelöst (`barstate.isconfirmed`, `primaryPushed`). Der separat gezeichnete `generateForecast()` läuft nur am letzten Chart-Bar und ist **kein historischer Strategy-Builder-Einstiegsauftrag**.

## 2. Neuer Fehler behoben: TradingView-XLSX-Zeitzone

Die alte Referenz-Konvertierung interpretierte Excel-Wandzeiten fälschlich als UTC. Die Quellen sind tatsächlich **Europe/Berlin (CEST = UTC+2 in den geprüften September-/Oktoberperioden)**. Beispiel:

- Strategie-Builder-Excel: **07.09.2026 03:05 Uhr** (Berlin)
- Korrekt: **07.09.2026 01:05 UTC**, nicht 03:05 UTC.

Validierung: **41/41** im KuCoin-Chart-Zeitausschnitt liegende Strategy-Builder-Eintritte stimmen nach Korrektur **in Richtung, Sekunde, beobachteter Renko-Kerze und Entry-Preis** mit TradingView-Marker und -Kerzenkörper überein. **Alle 41** erfüllen auch den im Chart angezeigten 30m-RSI-Filter (**10 Long >55 / 31 Short <75**). Die 68 weiteren KuCoin-Trades liegen außerhalb des vorhandenen KuCoin-CSV-Zeitfensters und sind auf diese Weise nicht gegengeprüft.

Die ursprüngliche Manifest-Version 1 enthält die falschen `*Utc`-Labels. Ein neuer, separat gespeicherter, unveränderter PnL-**v2**-Export enthält echte UTC-Zeitstempel, `sourceTimeZone: Europe/Berlin`, verifizierbare Quell-Hashes und 1.193+4 Referenztrades. Die v1-Version muss für zeitkritische Signalaudits zurückgewiesen werden.

## 3. RSI: Formel vorhanden, aber Chartdaten noch nicht paritätisch

Im Repo ist `src/quant/backtest/tradingade_signal_exits.py` vorhanden: Wilder-RSI(14), bestätigte 30m-Buckets, Long >55 und Short <75. Die Formel und der Filter sind unproblematisch, **die Eingabekerzen sind es nicht**.

Ein bewusst unabhängiger Gegenversuch mit normalen Binance-1m-Daten → 30m-Close → Wilder-RSI(14) verglichen mit der Chartspalte `30m` ergibt über **10.226 Vergleichswerte** eine mittlere absolute Differenz von **14,61 RSI-Punkten**, Korrelation **0,6565**. Der Vergleich beweist nicht einen Rechenfehler; er beweist, dass gewöhnliche Binance-30m-Kerzen **nicht** ohne Weiteres die passende Quelle für den RSI auf einem Renko-Chart sind. Auch eine naive Resampling-Variante der exportierten Renko-Kerzen löst die Parität noch nicht. Wir müssen das synthetische `request.security`/MTF-Quellverhalten der konkreten TradingView-Renko-Konfiguration nachbauen und validieren.

## 4. Strategy-Builder-Trades ≠ reine Elliott-Wave-Markierungen

Der Originalcode ist ein Pine-`indicator()`, **keine `strategy()`**. Die Tradingade-Software trifft eigene Ausführungsentscheidungen aus **Operatoren** (`PRICE =`, `Price >`, `Source <`), RSI-Schaltern, Positionsstatus und SL/Trailing-TP. Deshalb dürfen wir nicht jede Excel-Entry direkt als ursprünglichen EW-`Long Entry`/`Short Entry`-Impuls interpretieren. Bei den sechs Binance-Varianten liegen (bei zutreffender Zeitzone) je nach Builder-Variante **0 bis 25** Entry-Zeitpunkte auf den Plotmarkierungen **dieses einen** Binance-CSV-Snapshots. Für den Rest muss die konkrete Operator-/Signalquellen-Konfiguration unabhängig nachgebaut werden; fehlende Markierungen beweisen nicht, dass der Trade ungültig ist.

Beim KuCoin SOL sind Entry-Zeit und -Preis zu 41/41 konsistent, **Exits aber noch nicht vollständig reproduziert**: 41 geschlossene Exits liegen im gleichen Chart-Zeitfenster, davon nur 11/41 auf exakt passenden ausgegebenen `TP S`/`SL S`-Markern, 26/41 auf irgendeinem passenden exportierten Exit-/Trailing-Level. Diese Diagnose ist **keine** unabhängige Fill-Parität, da Tradingade Stop-/Trailing-Orders auf Renko-Daten synthetisch verbuchen kann. Die Excel-Netto-PnL-Abrechnung lässt sich dagegen für alle 1.193 geschlossenen Trades auf Rundungsniveau nachvollziehen (maximale Rundungsabweichung 0,0319 USDT); das beweist Import/Buchung, nicht Kausalität.

## 5. Nächste Gates für echte erste Pflanzen

1. Die noch offenen, chartbedingten Renko-MTF-HTF-Werte (1h/4h), insbesondere `request.security` mit bestätigten Renko-Kerzen, und RSI-30m-Renko-Quelle **kausal und unabhängig** rekonstruieren.
2. Tradingade Strategy Builder Operatoren sowie die verifizierten SL-/Trailing-TP-/Flip-Ausführungen über die Quelle konfigurationsgenau reproduzieren. Eingang/Exit unbedingt zwischen Original-Renko-Preis und realen 1m-Marktpreis-Fills trennen.
3. Abgleich pro Seed: EW-Event, RSI-Wert, Filterentscheidung, Trigger, Entry, Exitgrund, Exitpreis, Gebühren, PnL. Jede Abweichung einzeln erfassen.
4. Erst nach erfolgreichen Test-Gates Seed-Strategien aktiv in Cloudgarden simulierbar machen. Bis dahin `reference_only_no_live_orders`; niemals Original-Trades oder Excel-PnL als bekannte Zukunft für Fitness, Vitalität oder Kapitalkonto verwenden.

## Reproduktionsartefakte

- `ew_wave3_pine_probe.py` / `ew_pine_entry_parity.py`: Pine-Port und Signalvergleich auf exportiertem Renko-OHLC.
- `tradingview_trade_entry_audit.py`: sichere Zeitzonen-/Entry-/RSI-Referenzprüfung für neun Tradesnapshots.
- `quant_tv_seed_import.py`: v2-Importer mit Berlin-Wandzeiten → echter UTC.
- `tradingViewSeedCatalog.ts` + `.test.ts`: reine Referenzpflanzen, sichere v2-Prüfung und Verweigerung alter v1-Manifeste.
- `quant_tv_first_plants_corrected_utc_v2.json`: korrigiertes Referenz-Manifest (privat, nicht nach GitHub pushen).
- `ew_pine_signal_match.csv`, `tv_ew_trade_audit_summary.csv`, `tv_ew_trade_audit_entries.csv`, `rsi30_tv_vs_binance_prices.csv`: diagnostische Vergleichsdaten.

**Weder TradingView-Kontotokens noch private Marktdateien werden in GitHub abgelegt.**