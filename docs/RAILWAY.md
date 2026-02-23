# Deployment auf Railway

Nachdem du Railway mit GitHub verbunden hast:

## 1. Projekt auswählen

- Im Railway-Dashboard: **New Project** → **Deploy from GitHub repo**
- Repo `quant` (oder dein Repo-Name) auswählen
- Branch: `main` (oder dein Standard-Branch)

## 2. Build & Start (per Dockerfile)

Im Repo liegt ein **Dockerfile**. Wenn Railway es erkennt, werden Build und Start daraus ausgeführt – **dasselbe Python** und dieselben Pakete (uvicorn, fastapi, quant) in Build und Laufzeit. Das behebt „No module named uvicorn“.

- **Build Command / Start Command** in Railway kannst du leer lassen, sobald das Dockerfile genutzt wird.
- Falls Railway nicht automatisch das Dockerfile nutzt: unter **Settings** prüfen, ob „Dockerfile“ als Build-Option ausgewählt ist.

**Root Directory:** leer lassen (Projekt-Root).

## 3. Umgebungsvariablen

Im Service → **Variables** (oder **Settings** → **Variables**) hinzufügen.  
**Nur Futures** – es werden keine Spot-Keys verwendet:

| Name | Wert |
|------|------|
| `KUCOIN_FUTURES_API_KEY` | dein KuCoin **Futures** API Key |
| `KUCOIN_FUTURES_API_SECRET` | dein KuCoin **Futures** Secret |
| `KUCOIN_FUTURES_PASSPHRASE` | deine Passphrase (beim Anlegen des Keys gesetzt) |

In KuCoin: API mit Berechtigung **Futures** (nicht Spot) erstellen.

Optional (für Webhook-Sicherheit):

| Name | Wert |
|------|------|
| `WEBHOOK_TOKEN` | ein geheimer String; diesen dann in TradingView im Webhook als Header `x-webhook-token` eintragen |

Optional (Dashboard-Symbol):

| Name | Wert |
|------|------|
| `DASHBOARD_SYMBOL` | z.B. `SOL-USDT` (Standard) |

Optional (Regime + Dashboard-Quellen):

| Name | Wert |
|------|------|
| `REGIME_DB_PATH` | z.B. `data/live/regime.db` |
| `DASHBOARD_RENKO_PARQUET` | z.B. `data/live/renko_latest.parquet` |
| `DASHBOARD_TRADES_PARQUET` | z.B. `data/live/trades.parquet` |
| `DASHBOARD_LEVELS_JSON` | z.B. `data/live/execution_state.json` |

Nach dem Speichern baut/startet Railway neu.

## 4. Öffentliche URL

- **Settings** → **Networking** → **Generate Domain** (oder **Public Networking**)
- Du bekommst eine URL wie `https://quant-production-xxxx.up.railway.app`

Dann:

- **Dashboard:** `https://deine-url.up.railway.app/dashboard`
- **Health:** `https://deine-url.up.railway.app/health`
- **Regime latest:** `https://deine-url.up.railway.app/api/regime/latest?symbol=SOL-USDT`
- **Chart payload:** `https://deine-url.up.railway.app/api/dashboard/chart?symbol=SOL-USDT&hours=336`
- **Webhook für TradingView:** `https://deine-url.up.railway.app/webhook/tradingview`  
  (POST, Body JSON; wenn `WEBHOOK_TOKEN` gesetzt ist: Header `x-webhook-token: dein_token`)

## 5. Persistenz (Signale/Files)

Railway-Container sind ephemer: Bei jedem Redeploy ist das Dateisystem wieder leer. Wenn du eingehende Signale (`data/signals/...`) dauerhaft brauchst:

- **Option A:** Einen **Railway Volume** an den Service hängen und im Code (oder über `SIGNALS_DIR`) auf einen Pfad in diesem Volume schreiben.
- **Option B:** Signale in einer externen Datenquelle schreiben (z.B. S3, DB), statt nur lokal in Dateien.

Für den Einstieg reicht es, erst ohne Volume zu testen; die Webhook-Antwort bestätigt, dass der Request ankam. Dauerhafte Speicherung kannst du danach ergänzen.

## Kurz-Checkliste

- [ ] Repo verbunden, Branch stimmt
- [ ] Build läuft per **Dockerfile** (kein getrenntes Nixpacks-Python mehr)
- [ ] `KUCOIN_FUTURES_API_KEY`, `_SECRET`, `_PASSPHRASE` gesetzt
- [ ] Domain generiert
- [ ] `/dashboard` und `/health` im Browser getestet
- [ ] `/api/regime/latest` und `/api/dashboard/chart` getestet
