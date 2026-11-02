import type { MarketBarInput } from "./paperReplay.ts";
import type { KrakenProfile } from "./krakenProfiles.ts";
import { advanceKrakenEw, validateKrakenEw, type KrakenEwState } from "./krakenEw.ts";

type Candle = { timestampMs: number; open: number; high: number; low: number; close: number; minutes: number };
type Brick = { high: number; low: number; close: number };
export type KrakenPaperOrder = {
  availableAtMs: number; targetSide: -1 | 0 | 1; closeFraction: number; reason: string;
  nextMode: "TTP" | "WAIT" | "TF"; protectivePrice?: number;
};
/** JSON-serializable execution and indicator state; persist it with each plant, including pending orders. */
export type KrakenPaperState = {
  ew?: KrakenEwState;
  schemaVersion: 1; profileId: string; cashEquity: number; quantity: number; entryPrice: number;
  mode: "TTP" | "WAIT" | "TF"; bestFavorable: number; partialDone: boolean; tp1Done: boolean;
  breakEvenArmed: boolean; renkoClose: number | null; bricks: Brick[]; imbaTrend: -1 | 0 | 1;
  bucket: Candle | null; lastTimestampMs: number | null; lastMarkPrice: number;
  pending: KrakenPaperOrder[]; fillCount: number; feesPaid: number; realizedPnl: number;
};
export type KrakenPaperFill = {
  id: string; profileId: string; side: "buy" | "sell"; quantity: number; price: number;
  referencePrice: number; fee: number; executionCost: number; executedAt: string; reason: string;
  realizedPnl: number;
};
export type KrakenPaperResult = {
  state: KrakenPaperState; fills: KrakenPaperFill[]; netPnl: number; fees: number;
  markedEquity: number; realizedPnl: number;
};
export const KRAKEN_PAPER_POLICY = "Source-derived 5-minute-close Renko proxy, not TradingView Renko parity. "
  + "Signals and exits use completed five-minute observations; orders fill at the next minute open. "
  + "Mandatory 1x exposure cap and protective hard stop. Stop/TP ambiguity chooses stop first; favorable TP gaps fill at target, adverse gaps at worse open. "
  + "No synthetic Renko execution prices, funding, leverage, Kelly or recent-window lookahead.";

export function createKrakenPaperState(profileId: string, initialEquity: number): KrakenPaperState {
  if (!profileId || !Number.isFinite(initialEquity) || initialEquity <= 0) throw new RangeError("Positive initial paper allocation required.");
  return { schemaVersion: 1, profileId, cashEquity: initialEquity, quantity: 0, entryPrice: 0,
    mode: "TTP", bestFavorable: 0, partialDone: false, tp1Done: false, breakEvenArmed: false,
    renkoClose: null, bricks: [], imbaTrend: 0, bucket: null, lastTimestampMs: null,
    lastMarkPrice: 0, pending: [], fillCount: 0, feesPaid: 0, realizedPnl: 0 };
}

export function validateKrakenPaperState(state: KrakenPaperState, profile: KrakenProfile): void {
  if (profile.ew) validateKrakenEw(profile.ew, state.ew);
  else if (state.ew) throw new RangeError("EW state requires frozen EW profile genes.");
  if (!state || state.schemaVersion !== 1 || state.profileId !== profile.id || !["TTP", "WAIT", "TF"].includes(state.mode)
    || ![state.cashEquity, state.quantity, state.entryPrice, state.bestFavorable, state.lastMarkPrice, state.feesPaid, state.realizedPnl]
      .every(Number.isFinite) || state.entryPrice < 0 || state.lastMarkPrice < 0 || state.bestFavorable < 0
    || !Number.isInteger(state.fillCount) || state.fillCount < 0
    || ![-1, 0, 1].includes(state.imbaTrend) || !Array.isArray(state.bricks) || state.bricks.length > 500
    || !Array.isArray(state.pending) || state.pending.length > 3
    || (state.lastTimestampMs !== null && (!Number.isSafeInteger(state.lastTimestampMs) || state.lastTimestampMs % 60_000 !== 0))
    || (state.renkoClose !== null && (!Number.isFinite(state.renkoClose) || state.renkoClose <= 0))
    || [state.partialDone, state.tp1Done, state.breakEvenArmed].some((value) => typeof value !== "boolean")) {
    throw new RangeError("Invalid persistent Kraken paper state.");
  }
  for (const brick of state.bricks) if (![brick.high, brick.low, brick.close].every((v) => Number.isFinite(v) && v > 0)
    || brick.high < brick.low || brick.close > brick.high || brick.close < brick.low) throw new RangeError("Invalid Renko history.");
  for (const order of state.pending) if (!Number.isSafeInteger(order.availableAtMs) || ![-1, 0, 1].includes(order.targetSide)
    || !Number.isFinite(order.closeFraction) || order.closeFraction < 0 || order.closeFraction > 1
    || !["TTP", "WAIT", "TF"].includes(order.nextMode) || typeof order.reason !== "string"
    || (order.protectivePrice !== undefined && (!Number.isFinite(order.protectivePrice) || order.protectivePrice <= 0))) {
    throw new RangeError("Invalid persistent Kraken paper order.");
  }
  if (state.bucket && (![state.bucket.timestampMs, state.bucket.open, state.bucket.high, state.bucket.low,
    state.bucket.close, state.bucket.minutes].every(Number.isFinite) || !Number.isInteger(state.bucket.minutes)
    || state.bucket.minutes < 1 || state.bucket.minutes > 4 || state.bucket.timestampMs % 300_000 !== 0
    || state.bucket.low <= 0 || state.bucket.high < Math.max(state.bucket.open, state.bucket.close)
    || state.bucket.low > Math.min(state.bucket.open, state.bucket.close))) throw new RangeError("Invalid incomplete five-minute bucket.");
}

function marked(state: KrakenPaperState, price: number): number {
  return state.cashEquity + state.quantity * (price - state.entryPrice);
}

function executeKrakenOrder(state: KrakenPaperState, order: KrakenPaperOrder, bar: Candle,
  profile: KrakenProfile, allocationUsdt: number, fills: KrakenPaperFill[]): void {
  const currentSide = Math.sign(state.quantity);
  let desired = 0;
  const equity = Math.max(0, marked(state, bar.open));
  if (order.closeFraction > 0) desired = state.quantity * (1 - order.closeFraction);
  else if (order.targetSide !== 0) {
    // Fee and slippage reserve ensures gross opening notional + fee cannot exceed the allocation.
    const budget = Math.max(0, Math.min(allocationUsdt, equity));
    desired = order.targetSide * budget / (bar.open * (1 + (profile.commissionBps + profile.paperOverrides.slippageBps) / 10_000));
  }
  desired = Math.sign(desired) * Math.floor((Math.abs(desired) + 1e-12) * 1e8) / 1e8;
  // Reverse uses two observable fills so closing and new opening fees are accounted independently.
  const targets = state.quantity !== 0 && desired !== 0 && Math.sign(desired) !== currentSide ? [0, desired] : [desired];
  for (let target of targets) {
    if (Math.abs(target) < 1e-12) target = 0;
    const delta = Math.round((target - state.quantity) * 1e8) / 1e8;
    if (Math.abs(delta) < 1e-8) continue;
    const buying = delta > 0;
    // Pending source exits execute next open. Protective price only prevents optimistic favorable TP gaps.
    let reference = bar.open;
    if (order.protectivePrice !== undefined && order.reason.includes("tp") && !order.reason.includes("ttp")) {
      reference = currentSide > 0 ? Math.min(reference, order.protectivePrice) : Math.max(reference, order.protectivePrice);
    }
    const price = reference * (1 + (buying ? 1 : -1) * profile.paperOverrides.slippageBps / 10_000);
    const closed = state.quantity !== 0 && Math.sign(delta) !== Math.sign(state.quantity)
      ? Math.min(Math.abs(delta), Math.abs(state.quantity)) : 0;
    const realizedPnl = closed * Math.sign(state.quantity) * (price - state.entryPrice);
    const fee = Math.abs(delta) * price * profile.commissionBps / 10_000;
    state.cashEquity += realizedPnl - fee;
    state.realizedPnl += realizedPnl; state.feesPaid += fee;
    const oldQuantity = state.quantity;
    state.quantity = target;
    if (target === 0) state.entryPrice = 0;
    else if (oldQuantity === 0) state.entryPrice = price;
    else if (Math.abs(target) > Math.abs(oldQuantity)) {
      state.entryPrice = (Math.abs(oldQuantity) * state.entryPrice + Math.abs(delta) * price) / Math.abs(target);
    }
    state.fillCount += 1;
    fills.push({ id: `${profile.id}-${state.fillCount}`, profileId: profile.id, side: buying ? "buy" : "sell",
      quantity: Math.abs(delta), price, referencePrice: bar.open, fee,
      executionCost: Math.abs(delta) * Math.abs(price - bar.open), executedAt: new Date(bar.timestampMs).toISOString(),
      reason: order.reason, realizedPnl });
  }
  if (order.closeFraction === 0) {
    state.mode = order.nextMode; state.bestFavorable = state.entryPrice;
    state.partialDone = false; state.tp1Done = false; state.breakEvenArmed = false;
  }
}

function swingStop(state: KrakenPaperState, profile: KrakenProfile, side: number): number {
  const history = state.bricks.slice(-profile.swingLookback);
  const min = (profile.mode === "CT" ? profile.waitMinSlPct : profile.tfMinSlPct) / 100;
  const max = (profile.mode === "CT" ? profile.waitMaxSlPct : profile.tfMaxSlPct) / 100;
  const extreme = history.length ? (side > 0 ? Math.min(...history.map((b) => b.low)) : Math.max(...history.map((b) => b.high))) : state.entryPrice;
  const raw = side > 0 ? (state.entryPrice - extreme) / state.entryPrice : (extreme - state.entryPrice) / state.entryPrice;
  const distance = Math.min(max, Math.max(min, raw));
  return state.entryPrice * (1 - side * distance);
}

/** Consume only a completed five-minute source candle. Renko bricks are signals, never fill prices. */
function decideKrakenBar(state: KrakenPaperState, bar: Candle, profile: KrakenProfile, allowDecisions = true): void {
  const side = Math.sign(state.quantity);
  const stopBeforeCurrentBricks = side !== 0 ? swingStop(state, profile, side) : 0;
  let signal: -1 | 0 | 1 = 0;
  const box = profile.renkoBoxUsd;
  if (state.renkoClose === null) state.renkoClose = Math.floor(bar.close / box) * box;
  const count = Math.floor((Math.abs(bar.close - state.renkoClose) + box * 1e-9) / box);
  if (count > 100_000) throw new RangeError("Renko move exceeds bounded processing limit.");
  const direction = bar.close > state.renkoClose ? 1 : -1;
  for (let i = 0; i < count; i += 1) {
    const open: number = state.renkoClose;
    const close: number = Math.round((open + direction * box) * 1e8) / 1e8;
    state.renkoClose = close;
    state.bricks.push({ high: Math.max(open, close), low: Math.min(open, close), close });
    if (state.bricks.length > 500) state.bricks.shift();
    const channel = state.bricks.slice(-profile.channelLookback);
    const high = Math.max(...channel.map((b) => b.high)), low = Math.min(...channel.map((b) => b.low));
    const range = high - low;
    const next = close >= high - range * .236 ? 1 : close <= high - range * .786 ? -1 : state.imbaTrend;
    if (next !== 0 && next !== state.imbaTrend) {
      if (state.imbaTrend !== 0 || profile.bootstrap) signal = next;
      state.imbaTrend = next;
    }
  }
  let waveExit = 0;
  if (profile.ew) { const ew = advanceKrakenEw(state.ew, bar, profile.ew); state.ew = ew.state; waveExit = ew.waveExit; }
  if (!allowDecisions) return;
  if (profile.ew?.filter && signal !== 0 && signal !== state.ew?.bias) signal = 0;
  const enqueue = (targetSide: number, reason: string, nextMode: KrakenPaperOrder["nextMode"],
    closeFraction = 0, protectivePrice?: number) => {
    state.pending.push({ availableAtMs: bar.timestampMs + 300_000, targetSide: targetSide as -1 | 0 | 1,
      reason, nextMode, closeFraction, ...(protectivePrice === undefined ? {} : { protectivePrice }) });
  };
  if (!side) {
    if (signal !== 0) enqueue(signal, "imba_entry", profile.mode === "CT" ? "TTP" : "TF");
    return;
  }
  const hardDistance = Math.min(profile.hardSlPct > 0 ? profile.hardSlPct : Infinity,
    profile.paperOverrides.mandatoryHardSlPct) / 100;
  const hardStop = state.entryPrice * (1 - side * hardDistance);
  const hardHit = side > 0 ? bar.low <= hardStop : bar.high >= hardStop;
  // Protective stops take precedence when a completed OHLC candle also touches a TP.
  if (hardHit) {
    enqueue(profile.reverseHardSl ? -side : 0, "hard_sl", profile.armTtpAfterSlReverse ? "TTP" : "WAIT");
    return;
  }
  if (profile.ew?.exit && waveExit === side) { enqueue(0, "ew_confirmed_impulse_exit", state.mode); return; }
  if (profile.mode === "CT") {
    if (signal === -side) { enqueue(signal, "imba_flip", "TTP"); return; }
    if (signal === side && state.mode === "WAIT") { state.mode = "TTP"; state.bestFavorable = bar.close; }
    if (state.mode === "TTP") {
      // Trailing high water uses completed closes, per canonical Pine ctBestFav.
      state.bestFavorable = side > 0 ? Math.max(state.bestFavorable, bar.close) : Math.min(state.bestFavorable, bar.close);
      const ttp = state.bestFavorable * (1 - side * profile.ttpTrailPct / 100);
      if (side > 0 ? bar.close <= ttp : bar.close >= ttp) { enqueue(-side, "ttp_reverse", "WAIT"); return; }
    } else if (side > 0 ? bar.close <= stopBeforeCurrentBricks : bar.close >= stopBeforeCurrentBricks) {
      enqueue(profile.reverseWaitSl ? -side : 0, "wait_sl", profile.armTtpAfterSlReverse ? "TTP" : "WAIT"); return;
    }
    const tp1 = state.entryPrice * (1 + side * profile.ctTp1Pct / 100);
    if (!state.tp1Done && profile.ctTp1Pct > 0 && profile.ctTp1ClosePct > 0
      && (side > 0 ? bar.high >= tp1 : bar.low <= tp1)) {
      enqueue(side, "ct_tp1", state.mode, profile.ctTp1ClosePct / 100, tp1); state.tp1Done = true;
    }
    const partial = state.entryPrice * (1 + side * profile.partialTpPct / 100);
    if (!state.partialDone && profile.partialTpPct > 0 && profile.partialClosePct > 0
      && (side > 0 ? bar.high >= partial : bar.low <= partial)) {
      enqueue(side, "ct_partial_tp", state.mode, profile.partialClosePct / 100, partial); state.partialDone = true;
    }
  } else {
    const stop = state.breakEvenArmed ? state.entryPrice : stopBeforeCurrentBricks;
    if (side > 0 ? bar.low <= stop : bar.high >= stop) { enqueue(0, state.breakEvenArmed ? "tf_be" : "tf_sl", "TF"); return; }
    if (signal === -side) { enqueue(profile.flipTrendOnOpposite ? signal : 0, "tf_imba_flip", "TF"); return; }
    const tp2 = state.entryPrice * (1 + side * profile.tfTp2Pct / 100);
    if (side > 0 ? bar.high >= tp2 : bar.low <= tp2) { enqueue(0, "tf_tp2", "TF", 0, tp2); return; }
    const tp1 = state.entryPrice * (1 + side * profile.tfTp1Pct / 100);
    if (!state.tp1Done && (side > 0 ? bar.high >= tp1 : bar.low <= tp1)) {
      enqueue(side, "tf_tp1", "TF", profile.tfTp1ClosePct / 100, tp1);
      state.tp1Done = true; state.breakEvenArmed = profile.breakEvenAfterTp1;
    }
  }
}

/**
 * Advance continuous UTC one-minute OHLC. State may be JSON round-tripped between arbitrary chunks or days.
 * `allocationUsdt` caps the next position's notional, never adds cash or resets strategy history.
 * netPnl is the change in marked equity over this call, including fees and unrealized movement.
 */
export function advanceKrakenPaper(input: KrakenPaperState, bars: readonly MarketBarInput[],
  profile: KrakenProfile, allocationUsdt: number, options: { exitOnly?: boolean } = {}): KrakenPaperResult {
  validateKrakenPaperState(input, profile);
  if (!Number.isFinite(allocationUsdt) || allocationUsdt < 0) throw new RangeError("Nonnegative bounded allocation required.");
  const state: KrakenPaperState = JSON.parse(JSON.stringify(input));
  const fills: KrakenPaperFill[] = [];
  const before = marked(state, state.lastMarkPrice || state.entryPrice);
  const feesBefore = state.feesPaid, realizedBefore = state.realizedPnl;
  for (const inputBar of bars) {
    const timestampMs = Date.parse(inputBar.timestamp);
    const [open, high, low, close, volume] = [inputBar.open, inputBar.high, inputBar.low, inputBar.close, inputBar.volume].map(Number);
    if (!Number.isSafeInteger(timestampMs) || timestampMs % 60_000 !== 0 || ![open, high, low, close].every((v) => Number.isFinite(v) && v > 0)
      || !Number.isFinite(volume) || volume < 0 || high < Math.max(open, close) || low > Math.min(open, close) || high < low
      || (state.lastTimestampMs !== null && timestampMs !== state.lastTimestampMs + 60_000)) {
      throw new RangeError("Kraken paper requires positive continuous ordered UTC minute candles.");
    }
    const minute: Candle = { timestampMs, open, high, low, close, minutes: 1 };
    if (options.exitOnly) {
      // Caller declares the archive terminal horizon before replay. No orders reopen during liquidation retries.
      state.pending = state.quantity === 0 ? [] : [{ availableAtMs: timestampMs, targetSide: 0,
        closeFraction: 0, reason: "archive_exit", nextMode: state.mode }];
    }
    const hardPct = Math.min(profile.hardSlPct > 0 ? profile.hardSlPct : Infinity,
      profile.paperOverrides.mandatoryHardSlPct) / 100;
    if (state.quantity !== 0 && (open / state.entryPrice - 1) * Math.sign(state.quantity) <= -hardPct) {
      // The protective stop was already armed: a gap through it executes at this worse actual open.
      state.pending = [{ availableAtMs: timestampMs, targetSide: profile.reverseHardSl && !options.exitOnly
        ? -Math.sign(state.quantity) as -1 | 1 : 0, closeFraction: 0, reason: "hard_sl_gap",
        nextMode: profile.armTtpAfterSlReverse ? "TTP" : "WAIT" }];
    }
    const ready = state.pending.filter((order) => order.availableAtMs <= timestampMs);
    if (volume > 0) {
      state.pending = state.pending.filter((order) => order.availableAtMs > timestampMs);
      for (const order of ready) executeKrakenOrder(state, order, minute, profile, allocationUsdt, fills);
    }
    // Enforce a lowered allocation at a real open, including a zero allocation after ecology retires a plant.
    if (volume > 0 && Math.abs(state.quantity) * open > Math.min(allocationUsdt, Math.max(0, marked(state, open))) + 1e-9) {
      const cap = Math.min(allocationUsdt, Math.max(0, marked(state, open)));
      const fraction = Math.min(1, 1 - cap / (Math.abs(state.quantity) * open) + profile.commissionBps / 10_000);
      executeKrakenOrder(state, { availableAtMs: timestampMs, targetSide: Math.sign(state.quantity) as -1 | 1,
        closeFraction: fraction, reason: "allocation_cap", nextMode: state.mode }, minute, profile, allocationUsdt, fills);
    }
    const bucketMs = Math.floor(timestampMs / 300_000) * 300_000;
    if (!state.bucket && timestampMs === bucketMs) state.bucket = { ...minute, timestampMs: bucketMs };
    else if (state.bucket && state.bucket.timestampMs === bucketMs) {
      state.bucket.high = Math.max(state.bucket.high, high); state.bucket.low = Math.min(state.bucket.low, low);
      state.bucket.close = close; state.bucket.minutes += 1;
    }
    if (state.bucket?.minutes === 5) {
      // If execution is waiting for liquidity, defer new source orders rather than overwrite an unfilled exit.
      decideKrakenBar(state, state.bucket, profile, !state.pending.length && !options.exitOnly);
      state.bucket = null;
    }
    state.lastTimestampMs = timestampMs; state.lastMarkPrice = close;
  }
  validateKrakenPaperState(state, profile);
  const markedEquity = marked(state, state.lastMarkPrice || state.entryPrice);
  return { state, fills, netPnl: markedEquity - before, fees: state.feesPaid - feesBefore,
    markedEquity, realizedPnl: state.realizedPnl - realizedBefore };
}

/** Explicit data-gap discontinuity. Preserve account/position/exit state; rebuild indicators from observed data.
 * Existing queued risk reductions survive. Stale queued entries are cancelled; queued reversals become closes.
 * A pre-existing hard stop crossed by the first observed open executes at that actual open, including gap loss.
 */
export function resumeKrakenPaperAfterGap(input: KrakenPaperState): KrakenPaperState {
  const state: KrakenPaperState = JSON.parse(JSON.stringify(input));
  state.lastTimestampMs = null; state.bucket = null; state.renkoClose = null;
  state.bricks = []; state.imbaTrend = 0; delete state.ew;
  state.pending = state.pending.flatMap((order) => {
    if (state.quantity === 0) return [];
    return [{ ...order, availableAtMs: 0, targetSide: 0 as const,
      reason: order.closeFraction > 0 ? order.reason : "gap_stale_signal_close" }];
  });
  return state;
}
