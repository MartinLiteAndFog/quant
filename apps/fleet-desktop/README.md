# Fleet Desktop Cockpit

Native multi-bot % performance cockpit (Tauri 2 + React). Hero chart is compounded live trade PnL %; account equity %, activity, trades, stats, and capitalization live in drawers.

## Prerequisites

- Node 20+
- Rust / Cargo (`curl https://sh.rustup.rs -sSf | sh`)
- Fleet API host: the existing quant webhook/dashboard process with `/api/fleet/*`

## Dev (browser via Vite)

```bash
cd apps/fleet-desktop
npm install
# Terminal A — dashboard/webhook with Postgres
# Terminal B
npm run dev
```

## Desktop connection

Default API base: `https://quant-production-5533.up.railway.app`.

1. Open **Settings → Test connection**
2. Modes:
   - `fleet_api` — `/api/fleet/*` reachable (full curves + drawers)
   - `direct_health` — Railway bot `/health` only (status rail works; curves need fleet API deploy or local webhook)
   - `offline` — nothing reachable
3. Presets: Railway quant / Local webhook (`http://127.0.0.1:8000`)

Tauri uses native HTTP (`@tauri-apps/plugin-http` + `probe_url` command) so cross-origin calls do not depend on browser CORS. The webhook server also enables CORS for `/api/fleet` GET via `FLEET_CORS_ORIGINS` (default `*`).

Until production includes the fleet routes, point API base at a local webhook with `POSTGRES_URL` for curves, or keep Railway for health-only mode.


## Dev (Tauri window)

```bash
source "$HOME/.cargo/env"
npm run tauri:dev
```

## Production build

```bash
npm run tauri:build
```

## Bot registry

Defaults match live Railway pilots + Kraken:

| Display | `strategy_instance` | Health |
|---|---|---|
| Imba Runner | `sol-pilot-canonical` | sol-pilot-canonical-production |
| Pure ImbaTP | `sol-pilot-pc3axis` | sol-pilot-pc3axis-production |
| Countervariante | `sol-pilot-countertrend` | … |
| Counter SL Reverse | `sol-pilot-countertrend-sl-reverse` | … |
| Kraken Legacy | `kraken_bot` | kraken-production-cb57 |

Override server-side with `FLEET_BOTS_JSON`, or edit Settings locally (stored in `localStorage`).

Kraken history is read server-side; exchange credentials are never stored in
the desktop app. The Kraken service exposes the authenticated, read-only
`/api/fleet/kraken-position-events` route. Activity and Price Move use this same
ledger; split closing fills are aggregated before BPS are calculated. Configure
`FLEET_KRAKEN_DIRECT_EVENTS_URL` and the dedicated
`FLEET_KRAKEN_READ_TOKEN`. The older position-trades route remains a
compatibility fallback only.

## API

Mounted on the webhook server:

- `GET /api/fleet/bots`
- `GET /api/fleet/performance?hours=&instances=`
- `GET /api/fleet/activity?hours=`
- `GET /api/fleet/trades?instance=&limit=`
- `GET /api/fleet/kraken-position-trades?since_ms=&limit=` (Kraken service; read-only)
- `GET /api/fleet/kraken-position-events?since_ms=&limit=&include_funding=` (Kraken service; authenticated read-only)
- `GET /api/fleet/capitalization`

Auth: `WEBHOOK_TOKEN` via `Authorization: Bearer …`, `X-Webhook-Token`, or `?token=`.
