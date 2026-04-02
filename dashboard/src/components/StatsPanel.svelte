<script>
  import { chartStore, statusStore } from '../lib/stores.js';
  import { liveRegimeScore } from '../lib/chartHelpers.js';
  import { scoreToColor } from '../lib/colors.js';

  function fmtNum(v) {
    if (v == null || !Number.isFinite(Number(v))) return '-';
    return Number(v).toFixed(4);
  }

  function tickerText(st) {
    if (!st?.ticker) return st?.ticker_error || '-';
    const t = st.ticker;
    const b = typeof t.bid === 'number' ? t.bid.toFixed(4) : t.bid;
    const a = typeof t.ask === 'number' ? t.ask.toFixed(4) : t.ask;
    const m = t.mid != null ? (typeof t.mid === 'number' ? t.mid.toFixed(4) : t.mid) : null;
    return m != null ? `${m} (bid ${b} / ask ${a})` : `${b} / ${a}`;
  }

  function positionText(pos) {
    if (pos?.position == null) return pos?.error || '-';
    const lev = (pos.leverage != null && Number.isFinite(Number(pos.leverage)))
      ? ' x' + Number(pos.leverage).toFixed(1) : '';
    return String(pos.position) + lev;
  }

  function notionalText(pos, st) {
    if (pos?.position == null || !st?.ticker?.mid) return '-';
    const mult = Number(pos.contract_multiplier || 1);
    const notional = Math.abs(Number(pos.position)) * mult * Number(st.ticker.mid);
    return Number.isFinite(notional) ? notional.toFixed(2) + ' USDT' : '-';
  }

  function capitalText(st) {
    const bal = st?.balance;
    if (bal?.equity != null) return Number(bal.equity).toFixed(2) + ' USDT';
    return '-';
  }

  function apiStatusText(st) {
    return st?.api_configured ? 'configured' : 'missing';
  }

  function apiStatusClass(st) {
    return st?.api_configured ? 'ok' : 'err';
  }

  function exitModeText(payload) {
    if (!payload?.levels) return '-';
    const levels = payload.levels;
    const sl = Number(levels.sl);
    const ttp = Number(levels.ttp);
    const hasSl = Number.isFinite(sl);
    const hasTtp = Number.isFinite(ttp);
    if (hasTtp) return 'TTP (trailing)';
    if (hasSl) return 'SL (stop loss)';
    return '-';
  }

  function exitModeClass(payload) {
    if (!payload?.levels) return '';
    const ttp = Number(payload.levels.ttp);
    const sl = Number(payload.levels.sl);
    if (Number.isFinite(ttp)) return 'ok';
    if (Number.isFinite(sl)) return 'err';
    return '';
  }

  function barTimeText(payload) {
    let barTs = null;
    const levels = payload?.levels;
    if (levels?.entry_bar_ts != null && Number.isFinite(Number(levels.entry_bar_ts))) {
      barTs = Number(levels.entry_bar_ts);
    } else if (payload?.open_position?.entry_time != null && Number.isFinite(Number(payload.open_position.entry_time))) {
      barTs = Number(payload.open_position.entry_time);
    }
    if (barTs == null) return '-';
    const d = new Date(barTs * 1000);
    const yy = d.getUTCFullYear();
    const mo = String(d.getUTCMonth() + 1).padStart(2, '0');
    const dd = String(d.getUTCDate()).padStart(2, '0');
    const hh = String(d.getUTCHours()).padStart(2, '0');
    const mm = String(d.getUTCMinutes()).padStart(2, '0');
    return `${yy}-${mo}-${dd} ${hh}:${mm} UTC`;
  }

  function confidenceText(payload) {
    const conf = payload?.confidence == null ? null : Number(payload.confidence);
    if (conf == null) return '-';
    return conf.toFixed(3);
  }

  function confidenceColor(payload) {
    const conf = payload?.confidence == null ? null : Number(payload.confidence);
    if (conf == null) return 'inherit';
    const score = liveRegimeScore(payload);
    if (Number.isFinite(score)) {
      return score >= 0 ? '#9ece6a' : '#f7768e';
    }
    const rs = String(payload.regime_state || '').toLowerCase();
    if (rs === 'trend') {
      return conf >= 0.7 ? '#9ece6a' : (conf >= 0.5 ? '#e0af68' : '#f7768e');
    }
    return conf >= 0.7 ? '#f7768e' : (conf >= 0.5 ? '#e0af68' : '#9ece6a');
  }
</script>

<div class="panel">
  <div class="row"><span class="label">API (KuCoin)</span><span class={apiStatusClass($statusStore?.status)}>{apiStatusText($statusStore?.status)}</span></div>
  <div class="row"><span class="label">Ticker</span><span class="mono">{tickerText($statusStore?.status)}</span></div>
  <div class="row"><span class="label">Position</span><span class="mono">{positionText($statusStore?.position)}</span></div>
  <div class="row"><span class="label">Notional (est)</span><span class="mono">{notionalText($statusStore?.position, $statusStore?.status)}</span></div>
  <hr />
  <div class="row"><span class="label">Capital</span><span class="mono ok">{capitalText($statusStore?.status)}</span></div>
  <div class="row"><span class="label">Regime</span><span>{$chartStore?.day_regime_state ?? $chartStore?.regime_state ?? '-'}</span></div>
  <div class="row"><span class="label">Confidence</span><span class="confidence-pill" style:color={confidenceColor($chartStore)}>{confidenceText($chartStore)}</span></div>
  <div class="row"><span class="label">Bar time</span><span class="mono">{barTimeText($chartStore)}</span></div>
  <div class="row"><span class="label">Exit mode</span><span class={exitModeClass($chartStore)}>{exitModeText($chartStore)}</span></div>
  <div class="row"><span class="label">SL</span><span class="mono">{fmtNum($chartStore?.levels?.sl)}</span></div>
  <div class="row"><span class="label">TTP</span><span class="mono">{fmtNum($chartStore?.levels?.ttp)}</span></div>
  <div class="row"><span class="label">TP1</span><span class="mono">{fmtNum($chartStore?.levels?.tp1)}</span></div>
  <div class="row"><span class="label">TP2</span><span class="mono">{fmtNum($chartStore?.levels?.tp2)}</span></div>
  <p class="hint">{$statusStore?.status?.hint ?? ''}</p>
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

  .err {
    color: var(--red);
  }

  .confidence-pill {
    font-weight: 700;
  }

  .hint {
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 0.5rem;
  }

  hr {
    border-color: #2a3044;
    border-style: solid;
    border-width: 1px 0 0 0;
    margin: 0.8rem 0;
  }
</style>
