# Live-Betrieb & Cloud-Deployment

## Woher kommt der SOL-Kurs (Ticker)?

Der **Kurs kommt von KuCoin**: Sobald der Live-Service läuft, holt er Bid/Ask über die **KuCoin Futures API** (Ticker-Endpoint). Es wird also dieselbe Quelle genutzt wie für die Orders – keine separate Datenquelle. Der `KucoinFuturesBroker` ruft dafür `get_best_bid_ask("SOL-USDT")` auf und erhält die aktuellen Preise von KuCoin.

## API-Key eintragen

**Nicht im Code.** Nur über Umgebungsvariablen:

| Variable | Beschreibung |
|----------|--------------|
| `KUCOIN_FUTURES_API_KEY` | API Key aus KuCoin (API Management) |
| `KUCOIN_FUTURES_API_SECRET` | API Secret |
| `KUCOIN_FUTURES_PASSPHRASE` | Passphrase, die du beim Anlegen des Keys gesetzt hast |

**Lokal:**

1. Im Projektroot `.env` anlegen (wird von git ignoriert, wenn in `.gitignore`).
2. Inhalt an `.env.example` anlehnen und die Platzhalter durch echte Werte ersetzen.
3. Beim Start der App werden die Werte geladen (z.B. mit `python-dotenv` in deinem Startscript).

**Cloud (24/7):**

- In der jeweiligen Plattform die **Environment Variables** setzen (keine `.env`-Datei committen).
- Beispiele:
  - **Railway:** Project → Variables → Add Variable
  - **Fly.io:** `fly secrets set KUCOIN_FUTURES_API_KEY=...` (und Secret/Passphrase)
  - **Render / Heroku / AWS:** jeweilige „Environment“- oder „Config Vars“-Sektion

KuCoin: **API Management** → Create API → **Futures**-Berechtigung aktivieren, IP-Whitelist optional.

## 24/7 in der Cloud laufen lassen

Ein möglicher Aufbau:

1. **Ein Service** führt aus:
   - Webhook-Server (empfängt TradingView-Signale),
   - optional ein Worker/Loop, der bei neuem Signal den OMS aufruft und Orders an KuCoin sendet,
   - **Dashboard** (gleiche App, andere Pfade) für Status und Oberfläche.

2. **Deployment-Optionen:**
   - **Railway / Render / Fly.io:** Repo verbinden, Build-Befehl (z.B. `pip install -e .`), Start: `uvicorn quant.execution.webhook_server:app --host 0.0.0.0 --port $PORT`. Env-Variablen wie oben setzen.
   - **VPS (Ubuntu):** Systemd-Service oder Docker, gleicher uvicorn-Befehl, `0.0.0.0` damit der Server von außen erreichbar ist.

3. **Webhook von TradingView:** URL auf deine Cloud-URL setzen (z.B. `https://deine-app.railway.app/webhook/tradingview`). Wenn `WEBHOOK_TOKEN` gesetzt ist, im TradingView-Webhook den Header `x-webhook-token` mitschicken.

4. **Persistenz:** `data/signals` und `data/live` sollten auf einem Volume/verzeichnis liegen, das bei Restarts erhalten bleibt (z.B. Railway Volume, oder gebundenes Verzeichnis auf dem VPS).

## Desktop-Oberfläche

Es gibt eine **Web-Dashboard-Grundlage** (siehe unten). Du öffnest sie im Browser (lokal oder unter der Cloud-URL). Später kann daraus eine „richtige“ Desktop-App werden (z.B. Electron/Tauri), die dieselben API-Routen nutzt.

- **Lokal:** Nach Start des Servers: `http://127.0.0.1:8000/dashboard`
- **Cloud:** `https://deine-app.railway.app/dashboard`

Die gleichen API-Routen (`/api/status`, `/api/position`, …) können dann von einer eigenen Desktop-App angefragt werden.

## Regime Store und Dashboard-Overlays

Neue Standardpfade (optional per Env konfigurierbar):

- `REGIME_DB_PATH` (Default: `data/live/regime.db`)
- `DASHBOARD_RENKO_PARQUET` (Default: `data/live/renko_latest.parquet`)
- `DASHBOARD_TRADES_PARQUET` (Default: `data/live/trades.parquet`)
- `DASHBOARD_LEVELS_JSON` (Default: `data/live/execution_state.json`)

Neue Dashboard-APIs:

- `/api/regime/latest?symbol=SOL-USDT`
- `/api/regime/transitions?symbol=SOL-USDT&limit=50`
- `/api/dashboard/chart?symbol=SOL-USDT&hours=336&max_points=4000`

Die Dashboard-Chart zeigt:

- Renko-Chart (scroll/zoom),
- Gate-Shading (ON=grün, OFF=blau) mit transparenzbasierter Confidence,
- Trades (Marker),
- aktive Level (`SL`, `TTP`, `TP1`, `TP2`) aus `DASHBOARD_LEVELS_JSON`.

### Optional: Renko-Cache automatisch aktualisieren

Wenn du willst, kann der Webservice selbst im Hintergrund den Dashboard-Renko-Cache aktualisieren:

- `ENABLE_DASHBOARD_RENKO_UPDATER=1`
- `DASHBOARD_RENKO_BOX=0.1`
- `DASHBOARD_RENKO_DAYS_BACK=14`
- `DASHBOARD_RENKO_STEP_HOURS=6`
- `DASHBOARD_RENKO_POLL_SEC=300`

Hinweis: Das aktualisiert nur die Dashboard-Renko-Datei (`DASHBOARD_RENKO_PARQUET`), nicht die Trading-Logik an sich.

## Live-Ausführung (Signal + Gate-Routing + Trailing + Executor)

Der Live-Flow besteht aus zwei Worker-Prozessen:

1. `live_signal_worker`
   - erzeugt IMBA-Signale mit Lookback-Historie,
   - baut parallel einen inversen Trendfolger-Stream,
   - wählt je nach Gate den aktiven Stream:
     - `gate_on=1` -> `countertrend` (IMBA)
     - `gate_on=0` -> `trendfollower`
   - schreibt aktive Signale nach `SIGNALS_DIR/<SYMBOL>/<day>.jsonl`
   - persistiert Strategy-Streams zusätzlich unter:
     - `.../countertrend/<day>.jsonl`
     - `.../trendfollower/<day>.jsonl`

2. `live_executor`
   - liest den aktiven Stream,
   - führt OMS-Entry/Flip aus,
   - berechnet und schreibt laufende `SL/TTP/TP1/TP2` Trailing-Level in `DASHBOARD_LEVELS_JSON`.

Empfohlene Safety-Defaults:

- `LIVE_TRADING_ENABLED=0` (hart aus)
- `LIVE_EXECUTOR_DRY_RUN=1` (nur simulieren)
- `LIVE_EXECUTOR_MAX_EUR=20`
- `LIVE_EXECUTOR_LEVERAGE=1`
- `LIVE_EXECUTOR_SYMBOL_ALLOWLIST=SOLUSDT,SOL-USDT`
- `LIVE_TTP_TRAIL_PCT=0.012`
- `LIVE_WAIT_SL_PCT=0.02`
- `LIVE_DEFAULT_GATE_ON=1`

Worker-Start (z. B. in zweitem Railway-Service):

```bash
sh -lc "python -u -m quant.execution.live_signal_worker --symbol SOLUSDT --signals-dir /data/live/signals & python -u -m quant.execution.live_executor --symbol SOLUSDT --signals-dir /data/live/signals; wait"
```

Go-live Schalter:

1. Dry-Run beobachten (Logs + expected_trades)
2. Dann `LIVE_TRADING_ENABLED=1`
3. Dann `LIVE_EXECUTOR_DRY_RUN=0`

Ohne diese beiden letzten Schritte werden keine echten Orders gesendet.

## Smoke-Check lokal (Regime + Dashboard)

1. Regime-Daten in SQLite schreiben:

```bash
python scripts/update_regime_store.py --input data/regimes/your_gate.csv --symbol SOL-USDT --gate-col gate_on --ts-col ts
```

2. Webserver starten:

```bash
uvicorn quant.execution.webhook_server:app --host 127.0.0.1 --port 8000
```

3. API prüfen:

```bash
curl "http://127.0.0.1:8000/api/regime/latest?symbol=SOL-USDT"
curl "http://127.0.0.1:8000/api/dashboard/chart?symbol=SOL-USDT&hours=168"
```

4. Dashboard prüfen:

- `http://127.0.0.1:8000/dashboard`

## Migration zu Postgres (später)

- Die Regime-Logik ist über ein Store-Interface gekapselt (`RegimeStore`).
- Für Postgres bleibt der API-Vertrag gleich; nur die Store-Implementierung wird getauscht.
- Empfohlener Weg: parallele Writes (SQLite + Postgres) im Übergang, dann Dashboard/Worker auf Postgres umstellen.
