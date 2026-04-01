<script>
  import { chartStore, statusStore } from '../lib/stores.js';

  const EM = '\u2014';

  /** @param {unknown} n */
  function numOk(n) {
    const x = Number(n);
    return Number.isFinite(x);
  }

  /** @param {unknown} n @param {number} places */
  function fmtNum(n, places = 4) {
    if (!numOk(n)) return EM;
    return Number(n).toFixed(places);
  }

  /** @param {unknown} n */
  function fmtPrice(n) {
    if (!numOk(n)) return EM;
    const x = Math.abs(Number(n));
    const d = x >= 100 ? 2 : x >= 1 ? 3 : 4;
    return Number(n).toFixed(d);
  }

  /** @param {unknown} n */
  function fmtMoney(n) {
    if (!numOk(n)) return EM;
    return `$${Number(n).toFixed(2)}`;
  }

  /** @param {unknown} n */
  function fmtSignedPct(n) {
    if (!numOk(n)) return EM;
    const v = Number(n);
    const sign = v > 0 ? '+' : '';
    return `${sign}${v.toFixed(2)}%`;
  }

  /** @param {unknown} n */
  function pctClass(n) {
    if (!numOk(n)) return '';
    const v = Number(n);
    if (v > 0) return 'pos';
    if (v < 0) return 'neg';
    return '';
  }

  /** @param {unknown} n */
  function fmtWinrate(n) {
    if (!numOk(n)) return EM;
    return `${Number(n).toFixed(2)}%`;
  }

  /** @param {unknown} side @param {unknown} qty */
  function kucoinPositionLabel(side, qty) {
    const q = Number(qty);
    const flatQty = !Number.isFinite(q) || Math.abs(q) < 1e-12;
    if (side == null || side === '') {
      return flatQty ? 'Flat' : EM;
    }
    const s = String(side).toLowerCase();
    if (s === 'none' || s === 'flat') return 'Flat';
    if (flatQty) return 'Flat';
    const label =
      s === 'long' || s === 'buy' || s === 'l' ? 'Long' : s === 'short' || s === 'sell' || s === 's' ? 'Short' : null;
    if (!label) return 'Flat';
    const abs = Math.abs(q);
    const qStr = abs >= 100 ? abs.toFixed(0) : abs >= 10 ? abs.toFixed(1) : abs.toFixed(2);
    return `${label} ${qStr}`;
  }

  /** @param {Record<string, unknown> | null | undefined} km */
  function krakenPositionLabel(km) {
    if (!km || typeof km !== 'object') return EM;
    const pos = km.position;
    if (typeof pos === 'string' && pos.trim()) return pos.trim();
    const size = Number(km.venue_pos_size ?? km.size_rem);
    if (!Number.isFinite(size) || size === 0) return 'Flat';
    const sideN = Number(km.venue_pos_side ?? km.pos_side);
    let label = 'Flat';
    if (sideN >= 1) label = 'Long';
    else if (sideN <= -1) label = 'Short';
    else label = size > 0 ? 'Long' : 'Short';
    const abs = Math.abs(size);
    const qStr = abs >= 100 ? abs.toFixed(0) : abs >= 10 ? abs.toFixed(1) : abs.toFixed(2);
    return `${label} ${qStr}`;
  }

  /** @param {Record<string, unknown> | null | undefined} km */
  function krakenPrice(km) {
    if (!km || typeof km !== 'object') return EM;
    const p = km.price ?? km.mark_price;
    return fmtPrice(p);
  }

  /** @param {Record<string, unknown> | null | undefined} km */
  function krakenBalance(km) {
    if (!km || typeof km !== 'object') return EM;
    const b = km.balance ?? km.equity_usd ?? km.wallet_usd;
    return fmtMoney(b);
  }

  /** @param {unknown} side */
  function capitalizeSide(side) {
    if (side == null || side === '') return EM;
    if (typeof side === 'number' && Number.isFinite(side)) {
      if (side > 0) return 'Long';
      if (side < 0) return 'Short';
      return 'Flat';
    }
    const s = String(side).toLowerCase();
    if (s === 'long' || s === '1') return 'Long';
    if (s === 'short' || s === '-1') return 'Short';
    if (s === 'flat' || s === '0') return 'Flat';
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  /** @param {unknown} n */
  function fmtInt(n) {
    if (n == null || n === '') return EM;
    const x = Number(n);
    if (!Number.isFinite(x)) return EM;
    return String(Math.round(x));
  }
</script>

<div class="panel">
  <section class="block">
    <h3 class="section-head">Status</h3>
    <div class="two-col">
      <div>
        <div class="venue">KuCoin</div>
        <div class="mono val">{fmtPrice($chartStore?.bars?.at(-1)?.close)}</div>
      </div>
      <div>
        <div class="venue">Kraken</div>
        <div class="mono val">{krakenPrice($chartStore?.kraken_metrics)}</div>
      </div>
    </div>
    <div class="kv mono val top-gap">
      <span class="k">Regime</span>
      <span>{$chartStore?.day_regime_state ?? $chartStore?.regime_state ?? EM}</span>
    </div>
  </section>

  <section class="block">
    <h3 class="section-head">Position</h3>
    <div class="two-col">
      <div>
        <div class="venue">KuCoin</div>
        <div class="mono val">{kucoinPositionLabel($statusStore?.position?.side, $statusStore?.position?.position)}</div>
      </div>
      <div>
        <div class="venue">Kraken</div>
        <div class="mono val">{krakenPositionLabel($chartStore?.kraken_metrics)}</div>
      </div>
    </div>
  </section>

  <section class="block">
    <h3 class="section-head">Capital</h3>
    <div class="two-col">
      <div>
        <div class="venue">KuCoin</div>
        <div class="mono val">{fmtMoney($statusStore?.status?.balance?.equity)}</div>
      </div>
      <div>
        <div class="venue">Kraken</div>
        <div class="mono val">{krakenBalance($chartStore?.kraken_metrics)}</div>
      </div>
    </div>
  </section>

  <section class="block">
    <h3 class="section-head">Performance</h3>
    <div class="kv mono val">
      <span class="k">PnL %</span>
      <span class={pctClass($statusStore?.performance?.pnl_pct)}>{fmtSignedPct($statusStore?.performance?.pnl_pct)}</span>
    </div>
    <div class="kv mono val row-gap">
      <span class="k">Winrate</span>
      <span>{fmtWinrate($statusStore?.performance?.winrate)}</span>
    </div>
    <div class="kv mono val row-gap">
      <span class="k">Monthly growth</span>
      <span class={pctClass($statusStore?.performance?.monthly_growth)}
        >{fmtSignedPct($statusStore?.performance?.monthly_growth)}</span
      >
    </div>
    <div class="kv mono val row-gap">
      <span class="k">Average trade</span>
      <span class={pctClass($statusStore?.performance?.average_gain)}
        >{fmtSignedPct($statusStore?.performance?.average_gain)}</span
      >
    </div>
    <div class="kv mono val row-gap">
      <span class="k">Trades</span>
      <span>{fmtInt($statusStore?.performance?.trade_count)}</span>
    </div>
    <div class="kv mono val row-gap">
      <span class="k">Wins</span>
      <span class="pos">{fmtInt($statusStore?.performance?.winning_trade_count)}</span>
    </div>
    <div class="kv mono val row-gap">
      <span class="k">Losses</span>
      <span class="neg">{fmtInt($statusStore?.performance?.losing_trade_count)}</span>
    </div>
  </section>

  <section class="block">
    <h3 class="section-head">Levels</h3>
    <div class="kv mono val">
      <span class="k">Side</span>
      <span>{capitalizeSide($chartStore?.levels?.side)}</span>
    </div>
    <div class="kv mono val row-gap">
      <span class="k">Mode</span>
      <span>{$chartStore?.levels?.mode ?? EM}</span>
    </div>
    <div class="kv mono val row-gap">
      <span class="k">Entry</span>
      <span>{fmtNum($chartStore?.levels?.entry_px, 4)}</span>
    </div>
    <div class="kv mono val row-gap">
      <span class="k">SL</span>
      <span class="neg">{fmtNum($chartStore?.levels?.sl, 4)}</span>
    </div>
    <div class="kv mono val row-gap">
      <span class="k">TP1</span>
      <span>{fmtNum($chartStore?.levels?.tp1, 4)}</span>
    </div>
    <div class="kv mono val row-gap">
      <span class="k">TP2</span>
      <span>{fmtNum($chartStore?.levels?.tp2, 4)}</span>
    </div>
  </section>
</div>

<style>
  .panel {
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 16px;
    background: transparent;
  }

  .block {
    margin: 0;
  }

  .section-head {
    font-size: 0.7rem;
    letter-spacing: 0.05em;
    color: var(--muted);
    text-transform: uppercase;
    font-weight: 600;
    margin: 0 0 8px 0;
  }

  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px 16px;
  }

  .venue {
    font-size: 0.65rem;
    letter-spacing: 0.04em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 4px;
  }

  .mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }

  .val {
    color: var(--text);
  }

  .pos {
    color: var(--green);
  }

  .neg {
    color: var(--red);
  }

  .kv {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
  }

  .k {
    color: var(--muted);
    font-size: 0.75rem;
    flex-shrink: 0;
  }

  .row-gap {
    margin-top: 6px;
  }

  .top-gap {
    margin-top: 10px;
  }
</style>
