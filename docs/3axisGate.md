# _ 3axis Gate Backtest

## Ziel

Dieses Dokument beschreibt den vollständigen Ablauf, wie wir im aktuellen `quant-main`-Projekt ein **3-axis PC Gate** für ein **kurzes 7-Tage-Renko/IMBA-Backtestfenster** erzeugt, angepasst und getestet haben.

Ziel war **nicht** ein allgemeines Langfrist-Gate, sondern ein **kurzfristig empfindlicheres Regime-Gate** für ein ultrakurzes IMBA-Setup auf:

- **Symbol:** SOL-USDT
- **Quelle:** Binance
- **Fenster:** letzte 7 Tage des gezogenen Datensatzes
- **Renko:** `box = 0.03`
- **IMBA:** verschiedene Lookbacks, später Fokus auf `lb150`
- **Fee-Stress:** v. a. `15 bps roundtrip`

---

## Wichtige Grundbeobachtungen vor dem Gate

### 1. Basisproblem im Backtest
Am Anfang liefen Signale und Runner nicht auf derselben Basis:

- IMBA-Signale wurden korrekt aus **echten Renko-Bricks** erzeugt
- der Backtest-Runner lief teilweise auf **1m-Bars**, nicht auf echter Renko-Basis

Das führte zu falschen Flip-Folgen und unplausiblen Trades.

### 2. Fix der Basis
Wir haben dann die Renko-Basis normalisiert:

- echtes Renko-Parquet gebaut
- Duplicate-Timestamps per **Nanosekunden-Offsets** eindeutig gemacht
- IMBA-Signale auf genau dieser NS-normalisierten Renko-Datei neu erzeugt
- Runner auf genau diese Datei gesetzt

**Wichtige Dateien:**

- Renko input:
  - `/Users/martinpeter/Desktop/quant/data/renko/SOL-USDT_renko_box0.03_last7d_binance_ns.parquet`
- 1m fills/raw:
  - `/Users/martinpeter/Desktop/quant/data/raw/exchange=binance/symbol=SOL-USDT/timeframe=1m/SOL-USDT_1m_last7d_from_binance_trades.parquet`
- IMBA signals:
  - `/Users/martinpeter/Desktop/quant/data/signals/SOL-USDT/imba_renko_box0.03_lb150_last7d_binance_ns.jsonl`

---

## Relevante Projektdateien

### Backtest / Strategie
- `src/quant/backtest/renko_runner.py`
- `src/quant/backtest/renko_runner_tp2.py`
- `src/quant/strategies/flip_engine.py`
- `src/quant/strategies/imba.py`

### Predictive Coding / Gate
- `scripts/run_pc_trade_renko.py`
- `scripts/make_pc_3axis_gate_v2.py`
- temporär angepasst:
  - `scripts/_tmp_make_pc_3axis_gate_v2_last7d.py`

### State space / Sensoren
- `src/quant/state_space/pipeline.py`
- `src/quant/state_space/sensors_x.py`
- `src/quant/state_space/sensors_y.py`
- `src/quant/state_space/sensors_z.py`

### Regime-Dateien
- Haupt-Gate:
  - `data/regimes/pc_3axis_gate_FULLRANGE_nolookahead_v2.csv`
- invertierte OFF-Gates:
  - `data/regimes/pc_3axis_gate_last7d_OFF_base2of3.csv`
  - `data/regimes/pc_3axis_gate_last7d_OFF_base3of3.csv`

---

## Teil A — PC Predictions für das 7-Tage-Renko-Fenster erzeugen

### Verwendetes Skript
`run_pc_trade_renko.py`

### Aufruf
```bash
cd /Users/martinpeter/Desktop/quant-main && \
PYTHONPATH=/Users/martinpeter/Desktop/quant-main/src \
python3 scripts/run_pc_trade_renko.py \
  --input /Users/martinpeter/Desktop/quant/data/renko/SOL-USDT_renko_box0.03_last7d_binance_ns.parquet \
  --run-id PC_RENKO_last7d_box003
```

### Ergebnis
Output geschrieben nach:

- `data/runs/PC_RENKO_last7d_box003/pc_v02_renko`

Dort liegt insbesondere:

- `predictions.parquet`

Die PC predictions enthalten u. a.:

- `ts`
- `close`
- `v_temporal`
- optional `v_obs_mean`

Diese Felder sind die Grundlage für das 3-axis Gate.

---

## Teil B — Warum das Original-Gate-Skript nicht direkt lief

### Problem 1: hart verdrahteter Input-Pfad
`scripts/make_pc_3axis_gate_v2.py` war nicht CLI-gesteuert, sondern nutzte direkt:

```python
PRED_PATH = Path("data/runs/PC_GATE_FULLRANGE/pc_v02/predictions.parquet")
```

Das passte nicht zu unserem neuen 7-Tage-Run.

### Workaround
Wir haben eine temporäre Variante erzeugt:

- `scripts/_tmp_make_pc_3axis_gate_v2_last7d.py`

und dort den `PRED_PATH` auf den aktuellen 7-Tage-Run umgestellt:

- `data/runs/PC_RENKO_last7d_box003/pc_v02_renko/predictions.parquet`

---

## Teil C — Ursprüngliche Gate-Logik und warum sie zu träge war

### Originale Schwellen
Im alten Gate-Skript wurden diese Schwellensätze benutzt:

#### Base
- `instab q40`
- `elas q30`
- `|drift| q60`

#### Loose
- `instab q50`
- `elas q20`
- `|drift| q70`

### Problem
Diese Schwellen waren für längere Horizonte gedacht und im kurzen 7-Tage-Setup zu träge / zu wenig selektiv.

Das sah man daran, dass z. B. `gate_base_2of3` viel zu hohe ON-Rates erzeugte und Choppiness nicht genug filterte.

---

## Teil D — Neue, kurzfristigere Gate-Schwellen

Wir haben die temporäre Datei `scripts/_tmp_make_pc_3axis_gate_v2_last7d.py` vollständig neu geschrieben und die Quantile für den kurzen Horizont angepasst.

### Neue Base-Schwellen
- `instab q35`
- `elas q25`
- `|drift| q55`

### Neue Loose-Schwellen
- `instab q30`
- `elas q20`
- `|drift| q50`

### Neuer Aufruf
```bash
cd /Users/martinpeter/Desktop/quant-main && \
PYTHONPATH=/Users/martinpeter/Desktop/quant-main/src \
python3 scripts/_tmp_make_pc_3axis_gate_v2_last7d.py
```

### Ergebnis
Die Datei wurde geschrieben nach:

- `data/regimes/pc_3axis_gate_FULLRANGE_nolookahead_v2.csv`

### Neue ON-Rates
Die neu erzeugten ON-Rates waren:

- `base_2of3`: **64.18%**
- `base_3of3`: **8.08%**
- `loose_2of3`: **61.40%**
- `loose_3of3`: **4.03%**

Das war deutlich sensibler für den kurzen Horizont.

---

## Teil E — Timestamp-/Merge-Probleme beim Regime-Mapping

### Problem
Beim Laden des Gates in `renko_runner.py` gab es mehrfach:

- `MergeError: incompatible merge keys datetime64[ns, UTC] and datetime64[us, UTC]`

### Ursache
Die Gate-CSV und die Renko-Bars hatten unterschiedliche Timestamp-Präzision.

### Fix 1 — Runner-seitig
In `src/quant/backtest/renko_runner.py` wurde `_load_external_regime_to_bricks(...)` so angepasst, dass beide Seiten auf **`datetime64[ns, UTC]`** normalisiert werden.

### Fix 2 — CSV-seitig
Zusätzlich wurde die Gate-CSV so geschrieben, dass `ts` explizit mit **9 Stellen Nanosekunden-Präzision** vorliegt, z. B.:

- `2026-03-16T19:58:00.000000000Z`

Das half, dass `pandas` die CSV nicht wieder als `us` einliest.

---

## Teil F — ON-Zweig: Countertrend / Flip / TTP

### Setup
Runner:
- `src/quant/backtest/renko_runner.py`

Strategie:
- IMBA + Flip/TTP
- Lookback am Ende: `lb150`
- TTP: `0.25%`
- Fee: `15 bps`

### Bestes ON-Regime im kurzen Fenster
Das beste ON-Gate wurde am Ende:

- **`gate_base_3of3`** aus der neu erzeugten Gate-Datei

### Aufruf
```bash
cd /Users/martinpeter/Desktop/quant-main/src && python3 -m quant.backtest.renko_runner \
  --parquet /Users/martinpeter/Desktop/quant/data/renko/SOL-USDT_renko_box0.03_last7d_binance_ns.parquet \
  --box 0.03 \
  --signals-jsonl /Users/martinpeter/Desktop/quant/data/signals/SOL-USDT/imba_renko_box0.03_lb150_last7d_binance_ns.jsonl \
  --ttp-trail-pct 0.0025 \
  --fee-bps 15 \
  --min-sl-pct 0.010 \
  --max-sl-pct 0.080 \
  --swing-lookback 180 \
  --regime-csv /Users/martinpeter/Desktop/quant-main/data/regimes/pc_3axis_gate_FULLRANGE_nolookahead_v2.csv \
  --regime-col gate_base_3of3 \
  --fills-parquet /Users/martinpeter/Desktop/quant/data/raw/exchange=binance/symbol=SOL-USDT/timeframe=1m/SOL-USDT_1m_last7d_from_binance_trades.parquet \
  --fill-col close \
  --run-id SOLUSDT_box003_lb150_ttp025_fee15_PCgateNEW_base3of3
```

### Ergebnis
- ON-rate: **40.15%**
- `trades_real`: **54**
- `total_return_pct_real`: **-3.25%**
- `max_drawdown_pct_real`: **-4.40%**

### Interpretation
Das war der bisher beste **Countertrend-ON**-Kandidat im kurzen Testfenster, aber immer noch leicht negativ.

---

## Teil G — OFF-Zweig: Trendfollower / TP-only / no-flip

### Warum ein eigener OFF-Zweig
Für trendigere Phasen war die Countertrend-Flip-Logik ungeeignet.

Deshalb haben wir den OFF-Zweig separat mit `renko_runner_tp2.py` getestet:

- Trendfolger
- kleine TP-Ziele
- **kein Flip auf Opposite-Signal**
- Fokus auf schnelles Mitnehmen kurzer Trends

### Wichtiger Punkt
Zunächst wurde vergessen:

- `--regime external`

Dadurch ignorierte `renko_runner_tp2.py` das Gate trotz `--regime-csv`.

Das erkennt man daran, dass in den Stats `regime: 'none'` stand.

### Korrektes OFF-Setup
Wir haben die OFF-Gate-Datei invertiert aus dem neuen `base_3of3` erzeugt:

- `data/regimes/pc_3axis_gate_last7d_OFF_base3of3.csv`

Mit:
- `gate_off_base3of3 = 1 - gate_base_3of3`

### OFF-Gate-Rate
- `OFF rate`: **91.92%**

### Bester OFF-Runner-Test
Runner:
- `renko_runner_tp2.py`

Einstellungen:
- `tp1 = tp2 = 0.005`
- `tp1_frac = 1.0`
- `--no-flip-on-opposite`
- `fee = 15 bps`
- `lb150`
- `OFF base3of3`
- `regime external`

### Aufruf
```bash
cd /Users/martinpeter/Desktop/quant-main/src && python3 -m quant.backtest.renko_runner_tp2 \
  --parquet /Users/martinpeter/Desktop/quant/data/renko/SOL-USDT_renko_box0.03_last7d_binance_ns.parquet \
  --box 0.03 \
  --signals-jsonl /Users/martinpeter/Desktop/quant/data/signals/SOL-USDT/imba_renko_box0.03_lb150_last7d_binance_ns.jsonl \
  --fee-bps 15 \
  --tp1-pct 0.005 \
  --tp2-pct 0.005 \
  --tp1-frac 1.0 \
  --min-sl-pct 0.03 \
  --max-sl-pct 0.08 \
  --swing-lookback 180 \
  --no-flip-on-opposite \
  --regime external \
  --regime-csv /Users/martinpeter/Desktop/quant-main/data/regimes/pc_3axis_gate_last7d_OFF_base3of3.csv \
  --regime-col gate_off_base3of3 \
  --regime-default-off \
  --fills-parquet /Users/martinpeter/Desktop/quant/data/raw/exchange=binance/symbol=SOL-USDT/timeframe=1m/SOL-USDT_1m_last7d_from_binance_trades.parquet \
  --fill-col close \
  --run-id SOLUSDT_box003_lb150_OFFbase3of3_fee15_tp005_noflip_REGIME
```

### Ergebnis
- `trades`: **45**
- `total_return_pct_real`: **-1.51%**
- `total_return_pct_real_net`: **-7.94%**
- `max_drawdown_pct_real`: **-3.70%**
- `max_drawdown_pct_real_net`: **-9.00%**

### Interpretation
Das war der bisher beste **OFF-/Trendfolger-Zweig**.

---

## Teil H — Was sich als nicht hilfreich herausgestellt hat

### 1. Zu kurzes IMBA-Lookback
`lb50` war zu nervös und erzeugte zu viele kleine Fibostrukturen und Gegensignale.

### 2. Mindestabstand für Opposite-Flips
Wir haben `min_opposite_bricks` eingebaut, aber die Logik war wirkungslos, weil die Referenz aktuell an `entry_px` hing und durch Re-Anchorings nicht den gewünschten Effekt brachte.

Dieser Strang wurde vorerst zurückgestellt.

### 3. Base 2of3 im kurzen Horizont
Selbst mit angepassten Schwellen blieb `base_2of3` zu loose und war auf dem Countertrend-Zweig deutlich schwächer als `base_3of3`.

### 4. OFF-Runner mit Flip-on-opposite
Für den Trendfolger war `flip_on_opposite=True` schlecht, weil es die Swing-Idee zerstörte.  
Erst `--no-flip-on-opposite` machte den OFF-Runner sinnvoller.

---

## Teil I — Bisher bester Zwischenstand

### ON / Countertrend
- Runner: `renko_runner.py`
- IMBA `lb150`
- TTP `0.25%`
- Gate: **`gate_base_3of3`**
- Fee: `15 bps`

**Ergebnis:**
- `total_return_pct_real = -3.25%`
- `max_drawdown_pct_real = -4.40%`

### OFF / Trendfollower
- Runner: `renko_runner_tp2.py`
- IMBA `lb150`
- `tp1 = tp2 = 0.5%`
- `tp1_frac = 1.0`
- `--no-flip-on-opposite`
- Gate: **`gate_off_base3of3`**
- Fee: `15 bps`

**Ergebnis:**
- `total_return_pct_real = -1.51%`
- `total_return_pct_real_net = -7.94%`
- `max_drawdown_pct_real = -3.70%`

---

## Teil J — Was als Nächstes sinnvoll ist

### 1. Visuelle Diagnose-Tools bauen
Ziel:
- ON/OFF-Phasen sichtbar machen
- Trade-Exits/Entries mit Gate-Status überlagern
- verstehen, welche Trades vom Gate richtig/falsch geroutet werden

### 2. ON + OFF zusammenführen
Nächster großer Schritt:
- ON-Zeiten → Countertrend-Runner
- OFF-Zeiten → Trendfolger-Runner
- danach kombinierte Equity / Trade-Liste

### 3. Gate nicht nur über starre Quantile, sondern ggf. mit echter Kurzfrist-Kalibrierung
Mögliche weitere Justierung:
- andere Quantile
- andere Drift-/Elasticity-Horizonte
- ggf. Tradezahl vs. Qualität explizit optimieren

---

## Referenz-Dateien für den aktuellen Stand

### Daten
- Renko:
  - `/Users/martinpeter/Desktop/quant/data/renko/SOL-USDT_renko_box0.03_last7d_binance_ns.parquet`
- Fills/1m:
  - `/Users/martinpeter/Desktop/quant/data/raw/exchange=binance/symbol=SOL-USDT/timeframe=1m/SOL-USDT_1m_last7d_from_binance_trades.parquet`

### Signale
- `lb150`:
  - `/Users/martinpeter/Desktop/quant/data/signals/SOL-USDT/imba_renko_box0.03_lb150_last7d_binance_ns.jsonl`

### PC
- Predictions:
  - `/Users/martinpeter/Desktop/quant-main/data/runs/PC_RENKO_last7d_box003/pc_v02_renko/predictions.parquet`

### Gates
- Haupt-Gate:
  - `/Users/martinpeter/Desktop/quant-main/data/regimes/pc_3axis_gate_FULLRANGE_nolookahead_v2.csv`
- OFF invertiert:
  - `/Users/martinpeter/Desktop/quant-main/data/regimes/pc_3axis_gate_last7d_OFF_base3of3.csv`

### Runner
- ON/Countertrend:
  - `/Users/martinpeter/Desktop/quant-main/src/quant/backtest/renko_runner.py`
- OFF/Trendfolger:
  - `/Users/martinpeter/Desktop/quant-main/src/quant/backtest/renko_runner_tp2.py`

---

## Kurzfazit

Für dieses kurze 7-Tage-Fenster war der entscheidende Lernpunkt:

- Das alte 3-axis Gate war zu sehr auf längere Horizonte ausgerichtet.
- Mit kürzerfristigeren Quantilschwellen wurde das Gate brauchbarer.
- Der **Countertrend-ON-Zweig** funktioniert am besten mit **`base_3of3`**.
- Der **Trendfollower-OFF-Zweig** funktioniert nur sinnvoll mit **`--no-flip-on-opposite`** und sehr kleinem TP.
- Beide Zweige sind einzeln noch leicht negativ, aber deutlich näher an brauchbaren Kurzfrist-Setups als der ungefilterte No-Gate-Ansatz.
