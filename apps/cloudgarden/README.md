# Cloudgarden

Cloudgarden is a standalone synthetic evolution game. It has its own React/Vite app under `apps/cloudgarden`; it does not start Fleet Cockpit, connect to an account or venue, fetch market data, or place orders.

## Play the garden

```sh
npm install
npm run dev
```

The app starts at `http://127.0.0.1:5184`. Step one day, a week, or a 140-day season, or let the world advance automatically. The same world seed and actions produce the same climate, births, predation, and archive events.

Cloudgarden automatically saves its versioned world snapshot, selected view, research inputs, and latest closed replay in this browser. Reloading resumes the garden and restores the replay with the plant and signal genes used for that run. Existing first-version saves migrate to the current format without losing the garden or replay. Save data stays in local browser storage; there is no account sync. Use **Reset world** to start over on this device.

The initial ledger has 100 synthetic USDT of NAV, with a hard 90 USDT allocation ceiling and a protected 10 USDT reserve. Three parent plants begin with 20 USDT each. The remaining 30 USDT is held against 20 single-use seedling entitlements of 1.50 USDT each. Conception transfers that entitlement into deployed capital; it does not create equity. Predation returns an active plant's allocation to free capital only after it is archived. It never restores a spent seed entitlement.

Plant forms (Grass, Bush, and Cactus) are derived from root and leaf genes. Those genes share a bounded daily growth budget. Daily habitat fit measures only the plant's match to the current synthetic climate; vigor is its accumulated state, while synthetic evidence reflects simulated observation and is not market evidence. Dormancy is a reversible climate response: dormant plants remain visible and capitalized, recover when conditions improve, and are protected from predators. The predator has a minimum observation gate, 70/80 vulnerability thresholds, and a stronger bark-like age protection after 120 observation days. Archived plants leave the visible meadow and remain only as research records. The recent event window is capped at 2,000 records during long autoplay.

The Understory view aggregates archive and seed-bank activity and can run matched Random, Local, Hive, and Hybrid cohorts over the same world seeds and day horizon. Reports include synthetic daily habitat fit and vigor alongside population outcomes. The experiment compares only Cloudgarden's synthetic rules. Its result is marked `not_comparable` to market performance and must not be treated as evidence for a trading strategy.

Each plant carries heritable trading traits for lookback, entry, and exit thresholds alongside its habitat genes. Local bees blend and mutate parental traits; Hive and Hybrid bees bring in deterministic hive traits; Random bees sample independently. The selected plant's traits drive its 7-, 30-, or 90-day signal-to-paper fixture over generated UTC minute bars. Each range is deterministic for its garden seed and starting week, and later garden cycles receive a distinct synthetic fixture. The versioned synthetic Momentum v1 recipe reads completed closes, reports executable opportunity observations before the reserved forced-flat close, and emits targets for the causal replay. Strictly-next-open fills, bounded delay, gap/zero-volume rejection, spread/slippage embedded in price, one-time fees, and signed funding remain enforced. A versioned evidence report reconciles PnL, costs, cash, daily drawdown, generated targets, fills, and rejected signals. All synthetic horizons are marked `insufficient_sample`; independent opportunity count is unavailable. A closed result books into the same garden NAV and advances the daily clock by the replay's exact horizon. A no-opportunity run is labeled `no_trade` without inventing a trading loss. Executable opportunity observations use the selected plant's synthetic entry threshold and are not independent market opportunities. The signal fixture is not alpha or market evidence and does not change habitat fit, vigor, or evidence.

## Scope and evidence

All garden state and the built-in market week are generated fixtures. No historical SOL market source, venue/contract provenance, license, or quality-qualified dataset is connected. Costs and funding are configurable assumptions, not venue-calibrated values. The fixture cannot qualify or promote a strategy, and no account, exchange, order, Fleet API, or real capital is available to any plant.

## Verify

```sh
npm test
npm run build
```
