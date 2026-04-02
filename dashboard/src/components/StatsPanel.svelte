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
  <div class="row"><span class="label">Ticker (KuCoin)</span><span class="mono">{fmtPrice($chartStore?.bars?.at(-1)?.close)}</span></div>
  <div class="row"><span class="label">Ticker (Kraken)</span><span class="mono">{krakenPrice($chartStore?.kraken_metrics)}</span></div>
  <div class="row"><span class="label">Position</span><span class="mono">{kucoinPositionLabel($statusStore?.position?.side, $statusStore?.position?.position)}</span></div>
  <div class="row"><span class="label">Notional (est)</span><span class="mono">{fmtMoney($statusStore?.status?.balance?.equity)}</span></div>
  <hr />
  <div class="row"><span class="label">Capital</span><span class="mono ok">{fmtMoney($statusStore?.status?.balance?.equity)}</span></div>
  <div class="row"><span class="label">Regime</span><span>{$chartStore?.day_regime_state ?? $chartStore?.regime_state ?? EM}</span></div>
  <div class="row"><span class="label">Confidence</span><span class="mono confidence">{fmtSignedPct($statusStore?.performance?.pnl_pct)}</span></div>
  <div class="row"><span class="label">Bar time</span><span class="mono">{EM}</span></div>
  <div class="row"><span class="label">Exit mode</span><span>{$chartStore?.levels?.mode ?? EM}</span></div>
  <div class="row"><span class="label">SL</span><span class="mono">{fmtNum($chartStore?.levels?.sl, 4)}</span></div>
  <div class="row"><span class="label">TTP</span><span class="mono">{fmtNum($chartStore?.levels?.ttp, 4)}</span></div>
  <div class="row"><span class="label">TP1</span><span class="mono">{fmtNum($chartStore?.levels?.tp1, 4)}</span></div>
  <div class="row"><span class="label">TP2</span><span class="mono">{fmtNum($chartStore?.levels?.tp2, 4)}</span></div>
</div>

<style>
  .panel {
    padding: 0.75rem;
  }

  .row {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    margin: 0.4rem 0;
  }

  .label {
    color: var(--muted);
  }

  .mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }

  .ok {
    color: var(--green);
  }

  .confidence {
    font-weight: 700;
  }

  hr {
    border-color: #2a3044;
    border-style: solid;
    border-width: 1px 0 0 0;
    margin: 0.8rem 0;
  }
</style>
