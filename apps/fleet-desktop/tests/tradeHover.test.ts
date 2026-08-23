import assert from "node:assert/strict";
import test from "node:test";
import { latestTradeTime, tradesAtHover, visibleTradeDetails } from "../src/lib/tradeHover";
import type { BotSeries } from "../src/types";

function bot(id: string, entry: number, exit: number): BotSeries {
  return {
    id,
    display_name: id,
    strategy_instance: id,
    venue: "test",
    symbol: "SOL-USDT",
    trade_curve: [],
    account_curve: [],
    stats: { return_pct: 0, max_drawdown_pct: 0, trade_count: 1, win_rate: null, profit_factor: null },
    trade_details: [{
      trade_id: `${id}-trade`, bot_id: id, display_name: id,
      strategy_instance: id, entry_t: entry, exit_t: exit,
      side: "long", price_move_bps: 10,
    }],
  };
}

test("hover returns every simultaneously active trade across visible bots", () => {
  const trades = visibleTradeDetails(
    [bot("a", 100, 300), bot("b", 150, 250), bot("hidden", 100, 400)],
    new Set(["a", "b"]),
    null,
  );
  assert.deepEqual(tradesAtHover(trades, 200).map((trade) => trade.bot_id), ["b", "a"]);
});

test("hover includes all trades realized within the chart interval", () => {
  const trades = visibleTradeDetails([bot("a", 10, 100), bot("b", 20, 104)], new Set(), null);
  assert.equal(tradesAtHover(trades, 102, 10).length, 2);
  assert.equal(latestTradeTime(trades), 104);
});

test("isolation excludes every other bot", () => {
  const trades = visibleTradeDetails([bot("a", 10, 100), bot("b", 10, 100)], new Set(), "b");
  assert.deepEqual(trades.map((trade) => trade.bot_id), ["b"]);
});
