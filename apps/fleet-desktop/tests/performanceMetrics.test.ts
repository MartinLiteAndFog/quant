import assert from "node:assert/strict";
import test from "node:test";

import {
  correctedCurveOrJumpTwr,
  rawCommonScopeReturnPct,
  returnPctForView,
} from "../src/lib/performanceMetrics.ts";
import type { PortfolioSeries } from "../src/types.ts";

const portfolio: PortfolioSeries = {
  id: "portfolio",
  display_name: "Portfolio",
  account_curve: [
    { t: 1, equity_pct: 0 },
    { t: 2, equity_pct: 4.43 },
  ],
  account_curve_abs: [
    { t: 1, equity: 100 },
    { t: 2, equity: 500 },
    { t: 3, equity: 520 },
  ],
  corrected_curve: [
    { t: 1, equity_pct: 0 },
    { t: 2, equity_pct: 3.75 },
  ],
  cashflow_return: {
    available: true,
    return_pct: 8.5,
    net_cashflow: 400,
    flow_count: 1,
    scope_label: "5 futures accounts",
    boundary_note: "Futures account boundaries",
    method: "ledger_segmented_equity",
    excluded_bot_ids: ["counter-sl-reverse"],
    unavailable_bot_ids: [],
    reason: null,
    as_of: "2026-07-27T00:00:00Z",
  },
};

test("Equity % mirrors the selected portfolio percent chart", () => {
  assert.equal(returnPctForView("account", portfolio, 88), 4.43);
});

test("Corrected Return mirrors the cashflow-adjusted portfolio curve", () => {
  assert.equal(returnPctForView("corrected", portfolio, 88), 3.75);
});

test("older API payloads get a local jump-TWR compatibility curve", () => {
  const curve = correctedCurveOrJumpTwr(undefined, [
    { t: 1, equity: 100 },
    { t: 2, equity: 102 },
    { t: 3, equity: 300 },
    { t: 4, equity: 306 },
  ]);
  assert.equal(curve.at(-1)?.equity_pct, 4.04);
});

test("an explicit empty corrected curve stays unavailable", () => {
  assert.deepEqual(
    correctedCurveOrJumpTwr([], [
      { t: 1, equity: 100 },
      { t: 2, equity: 110 },
    ]),
    [],
  );
});

test("the desktop can recover an empty server curve from absolute equity", () => {
  const curve = correctedCurveOrJumpTwr(
    [],
    [
      { t: 1, equity: 100 },
      { t: 2, equity: 102 },
      { t: 3, equity: 300 },
    ],
    10,
    true,
  );
  assert.equal(curve.at(-1)?.equity_pct, 2);
});

test("the desktop can recover from an older percent-only API payload", () => {
  const curve = correctedCurveOrJumpTwr(
    [],
    [],
    10,
    true,
    [
      { t: 1, equity_pct: 0 },
      { t: 2, equity_pct: 2 },
      { t: 3, equity_pct: 200 },
      { t: 4, equity_pct: 206 },
    ],
  );
  assert.equal(curve.at(-1)?.equity_pct, 4.04);
});

test("Equity $ uses the confirmed-ledger cashflow-corrected return", () => {
  assert.equal(returnPctForView("account_abs", portfolio, 88), 8.5);
});

test("Price Move preserves the BPS value supplied by the caller", () => {
  assert.equal(returnPctForView("trade", portfolio, 7.25), 7.25);
});

test("Strategy return is never inferred from portfolio equity", () => {
  assert.equal(returnPctForView("strategy", portfolio, null), null);
});

test("Equity $ falls back to cashflow-neutralized return while ledger sync is unavailable", () => {
  assert.equal(
    returnPctForView(
      "account_abs",
      { ...portfolio, cashflow_return: { ...portfolio.cashflow_return!, available: false, return_pct: null } },
      null, 6.25,
    ),
    3.75,
  );
});

test("Equity $ corrected fallback still requires two equity observations", () => {
  assert.equal(
    returnPctForView(
      "account_abs",
      {
        ...portfolio,
        corrected_curve: [],
        account_curve: [],
        account_curve_abs: [{ t: 1, equity: 100 }],
        cashflow_return: { ...portfolio.cashflow_return!, available: false, return_pct: null },
      },
      null, null,
    ),
    null,
  );
});

test("raw fallback uses value-weighted common account coverage", () => {
  const value = rawCommonScopeReturnPct([
    {
      id: "early",
      display_name: "Early",
      strategy_instance: "early",
      venue: "kucoin",
      symbol: "SOL-USDT",
      trade_curve: [],
      account_curve: [],
      account_curve_abs: [
        { t: 90, equity: 100 },
        { t: 190, equity: 110 },
      ],
      stats: { return_pct: 0, max_drawdown_pct: 0, trade_count: 0, win_rate: null, profit_factor: null },
    },
    {
      id: "late",
      display_name: "Late",
      strategy_instance: "late",
      venue: "kucoin",
      symbol: "SOL-USDT",
      trade_curve: [],
      account_curve: [],
      account_curve_abs: [
        { t: 100, equity: 200 },
        { t: 190, equity: 220 },
      ],
      stats: { return_pct: 0, max_drawdown_pct: 0, trade_count: 0, win_rate: null, profit_factor: null },
    },
  ]);
  assert.ok(value != null && Math.abs(value - 10) < 1e-9);
});
