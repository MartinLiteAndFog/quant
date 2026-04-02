<script>
  import { chartStore, statusStore } from '../lib/stores.js';

  function fmt(v, digits = 2) {
    if (v == null || !isFinite(Number(v))) return '—';
    return Number(v).toLocaleString(undefined, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function fmtPct(v, digits = 2) {
    if (v == null || !isFinite(Number(v))) return '—';
    return `${fmt(v, digits)}%`;
  }

  function fmtInt(v) {
    if (v == null || !isFinite(Number(v))) return '—';
    return Math.round(Number(v)).toLocaleString();
  }

  function sideColor(side) {
    if (!side) return 'flat';
    const s = String(side).toLowerCase();
    if (s === 'long' || s === 'buy' || s === '1') return 'long';
    if (s === 'short' || s === 'sell' || s === '-1') return 'short';
    return 'flat';
  }

  function sideLabel(side) {
    if (side == null) return 'Flat';
    if (typeof side === 'number') {
      if (side > 0) return 'Long';
      if (side < 0) return 'Short';
      return 'Flat';
    }
    const s = String(side).toLowerCase();
    if (s === 'long' || s === 'buy' || s === '1') return 'Long';
    if (s === 'short' || s === 'sell' || s === '-1') return 'Short';
    if (s === 'flat' || s === '0' || s === '') return 'Flat';
    return side;
  }

  $: st = $statusStore?.status ?? null;
  $: pos = $statusStore?.position ?? null;
  $: perf = $statusStore?.performance ?? null;
  $: chart = $chartStore;
  $: kr = chart?.kraken_metrics ?? null;
  $: levels = chart?.levels ?? null;
  $: terminal = levels?.terminal ?? null;

  $: kucoinPrice = st?.ticker?.last ?? st?.ticker?.mid ?? null;
  $: kucoinEquity = st?.balance?.equity ?? null;
  $: strategyLabel = $statusStore?.strategy?.strategy_label ?? '—';

  $: kucoinSide = levels?.side ?? pos?.side ?? null;
  $: kucoinSizeNum = Number(levels?.live_pos ?? pos?.position ?? NaN);
  $: kucoinHasPos = Number.isFinite(kucoinSizeNum) && kucoinSizeNum !== 0;
  $: kucoinDisplaySize = Number.isFinite(kucoinSizeNum) ? Math.abs(kucoinSizeNum) / 10 : 0;

  $: krakenEquity = kr?.equity_usd ?? null;
  $: krakenSide = kr?.venue_pos_side ?? kr?.pos_side ?? null;
  $: krakenSizeNum = Number(kr?.venue_pos_size ?? kr?.size_rem ?? NaN);
  $: krakenMode = kr?.mode ?? null;
  $: krakenMark = kr?.mark_price ?? null;
  $: krakenHasPos = Number.isFinite(krakenSizeNum) && krakenSizeNum !== 0;

  $: entryPx = terminal?.entry_px ?? levels?.entry_px ?? null;
  $: sl = terminal?.sl ?? levels?.sl ?? null;
  $: ttp = terminal?.ttp ?? levels?.ttp ?? null;
  $: tp1 = levels?.tp1 ?? null;
  $: tp2 = levels?.tp2 ?? null;
  $: mode = terminal?.mode ?? levels?.mode ?? null;
  $: lvlSide = terminal?.side ?? levels?.side ?? null;
  $: hasAnyLevels = entryPx != null || sl != null || ttp != null;
</script>

<aside class="sidebar">
  <!-- STATUS -->
  <div class="card">
    <h3 class="card-title">Status</h3>
    <div class="dual-col">
      <div>
        <div class="venue-header">
          <span class="dot blue"></span>
          <span class="venue-name">KuCoin</span>
        </div>
        <div class="venue-detail">
          <div>Price: <span class="dim">{fmt(kucoinPrice, 4)}</span></div>
          <div>Regime: <span class="dim">{strategyLabel}</span></div>
        </div>
      </div>
      <div>
        <div class="venue-header">
          <span class="dot amber"></span>
          <span class="venue-name">Kraken</span>
        </div>
        <div class="venue-detail">
          <div>Price: <span class="dim">{fmt(krakenMark, 4)}</span></div>
          {#if krakenMode}
            <div>Mode: <span class="dim">{krakenMode}</span></div>
          {/if}
        </div>
      </div>
    </div>
  </div>

  <!-- POSITION -->
  <div class="card">
    <h3 class="card-title">Position</h3>
    <div class="dual-col">
      <div>
        <div class="venue-header">
          <span class="dot blue"></span>
          <span class="venue-name">KuCoin</span>
        </div>
        <div class="pos-value">
          {#if kucoinHasPos}
            <span class="side-{sideColor(kucoinSide)}">{sideLabel(kucoinSide)} {kucoinDisplaySize.toFixed(1)}</span>
          {:else}
            <span class="flat">Flat</span>
          {/if}
        </div>
      </div>
      <div>
        <div class="venue-header">
          <span class="dot amber"></span>
          <span class="venue-name">Kraken</span>
        </div>
        <div class="pos-value">
          {#if krakenHasPos}
            <span class="side-{sideColor(String(krakenSide))}">{sideLabel(krakenSide)} {Math.abs(krakenSizeNum).toFixed(1)}</span>
          {:else}
            <span class="flat">Flat</span>
          {/if}
        </div>
      </div>
    </div>
  </div>

  <!-- CAPITAL -->
  <div class="card">
    <h3 class="card-title">Capital</h3>
    <div class="dual-col">
      <div>
        <div class="venue-header">
          <span class="dot blue"></span>
          <span class="venue-name">KuCoin</span>
        </div>
        <div class="pos-value">
          {#if kucoinEquity != null}
            <span class={kucoinEquity >= 0 ? 'green' : 'red'}>${fmt(kucoinEquity)}</span>
          {:else}
            <span class="flat">—</span>
          {/if}
        </div>
      </div>
      <div>
        <div class="venue-header">
          <span class="dot amber"></span>
          <span class="venue-name">Kraken</span>
        </div>
        <div class="pos-value">
          {#if krakenEquity != null}
            <span class={krakenEquity >= 0 ? 'green' : 'red'}>${fmt(krakenEquity)}</span>
          {:else}
            <span class="flat">—</span>
          {/if}
        </div>
      </div>
    </div>
  </div>

  <!-- PERFORMANCE -->
  <div class="card">
    <h3 class="card-title">Performance</h3>
    <div class="kv-list">
      <div class="kv"><span class="k">PnL %</span><span>{fmtPct(perf?.pnl_pct)}</span></div>
      <div class="kv"><span class="k">Winrate</span><span>{fmtPct(perf?.winrate)}</span></div>
      <div class="kv"><span class="k">Monthly growth</span><span>{fmtPct(perf?.monthly_growth)}</span></div>
      <div class="kv"><span class="k">Average trade</span><span>{fmtPct(perf?.average_gain)}</span></div>
      <div class="divider"></div>
      <div class="kv"><span class="k">Trades</span><span>{fmtInt(perf?.trade_count)}</span></div>
      <div class="kv"><span class="k">Wins</span><span class="green">{fmtInt(perf?.winning_trade_count)}</span></div>
      <div class="kv"><span class="k">Losses</span><span class="red">{fmtInt(perf?.losing_trade_count)}</span></div>
    </div>
  </div>

  <!-- LEVELS -->
  <div class="card">
    <h3 class="card-title">Levels</h3>
    {#if !hasAnyLevels}
      <div class="pos-value flat">—</div>
    {:else}
      <div class="kv-list">
        {#if lvlSide}
          <div class="side-{sideColor(lvlSide)}">Side: {sideLabel(lvlSide)}</div>
        {/if}
        {#if mode}
          <div class="kv"><span class="k">Mode</span><span>{mode}</span></div>
        {/if}
        {#if entryPx != null}
          <div class="kv"><span class="k">Entry</span><span>{fmt(entryPx, 4)}</span></div>
        {/if}
        {#if sl != null}
          <div class="kv"><span class="k">SL</span><span>{fmt(sl, 4)}</span></div>
        {/if}
        {#if ttp != null}
          <div class="kv"><span class="k">TTP</span><span>{fmt(ttp, 4)}</span></div>
        {/if}
        {#if tp1 != null}
          <div class="kv"><span class="k">TP1</span><span>{fmt(tp1, 4)}</span></div>
        {/if}
        {#if tp2 != null}
          <div class="kv"><span class="k">TP2</span><span>{fmt(tp2, 4)}</span></div>
        {/if}
      </div>
    {/if}
  </div>
</aside>

<style>
  .sidebar {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.75rem;
    overflow-y: auto;
    width: 100%;
  }

  .card {
    border-radius: 0.5rem;
    border: 1px solid #27272a;
    padding: 0.75rem;
  }

  .card-title {
    margin: 0 0 0.5rem 0;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #a1a1aa;
  }

  .dual-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
  }

  .venue-header {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    margin-bottom: 0.375rem;
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
  }
  .dot.blue { background: #3b82f6; }
  .dot.amber { background: #f59e0b; }

  .venue-name {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #71717a;
  }

  .venue-detail {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.75rem;
    color: #f4f4f5;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .venue-detail .dim { color: #d4d4d8; }

  .pos-value {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.875rem;
  }

  .flat { color: #71717a; }

  .side-long { color: #34d399; }
  .side-short { color: #f87171; }
  .side-flat { color: #71717a; }

  .green { color: #34d399; }
  .red { color: #f87171; }

  .kv-list {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.875rem;
    color: #f4f4f5;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .kv {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
  }

  .k {
    color: #a1a1aa;
  }

  .divider {
    border-top: 1px solid #27272a;
    margin: 0.375rem 0;
  }
</style>
