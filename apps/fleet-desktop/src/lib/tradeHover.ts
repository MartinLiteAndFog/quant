import type { BotSeries, TradeDetail } from "../types";

export interface HoverTrade extends TradeDetail {
  color: string;
}

export function visibleTradeDetails(
  series: BotSeries[],
  visibleIds: Set<string>,
  isolatedId: string | null,
): HoverTrade[] {
  return series
    .filter((bot) =>
      isolatedId
        ? bot.id === isolatedId
        : visibleIds.size === 0 || visibleIds.has(bot.id),
    )
    .flatMap((bot) =>
      (bot.trade_details || []).map((trade) => ({
        ...trade,
        color: bot.color || "#c9a65a",
      })),
    );
}

/** All economic trades active or realized at the selected chart instant. */
export function tradesAtHover(
  trades: HoverTrade[],
  hoverTime: number,
  intervalSec = 60,
): HoverTrade[] {
  if (!Number.isFinite(hoverTime)) return [];
  const tolerance = Math.max(1, intervalSec / 2);
  const relevant = trades.filter((trade) => {
    const entry = trade.entry_t;
    const active = entry != null && entry <= hoverTime && hoverTime <= trade.exit_t;
    const realizedNow = Math.abs(trade.exit_t - hoverTime) <= tolerance;
    return active || realizedNow;
  });
  return relevant.sort((a, b) => a.exit_t - b.exit_t || a.bot_id.localeCompare(b.bot_id));
}

export function latestTradeTime(trades: HoverTrade[]): number | null {
  if (!trades.length) return null;
  return Math.max(...trades.map((trade) => trade.exit_t));
}
