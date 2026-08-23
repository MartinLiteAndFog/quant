import type {
  AbsCurvePoint,
  BotSeries,
  ChartMode,
  CurvePoint,
  PortfolioSeries,
} from "../types";

/** Compatibility fallback for older Fleet APIs that do not emit corrected_curve. */
export function correctedCurveOrJumpTwr(
  corrected: CurvePoint[] | undefined,
  absolute: AbsCurvePoint[] | undefined,
  jumpThresholdPct = 10,
  fallbackWhenEmpty = false,
  rawPercent?: CurvePoint[],
): CurvePoint[] {
  if (corrected !== undefined && (corrected.length > 0 || !fallbackWhenEmpty)) {
    return corrected;
  }
  let clean = (absolute || [])
    .filter(
      (point) =>
        Number.isFinite(point.t) &&
        Number.isFinite(point.equity) &&
        point.equity > 0,
    )
    .slice()
    .sort((a, b) => a.t - b.t);
  if (!clean.length) {
    clean = (rawPercent || [])
      .filter(
        (point) =>
          Number.isFinite(point.t) &&
          Number.isFinite(point.equity_pct) &&
          point.equity_pct > -100,
      )
      .map((point) => ({
        t: point.t,
        equity: 1 + point.equity_pct / 100,
      }))
      .sort((a, b) => a.t - b.t);
  }
  if (!clean.length) return [];
  let growth = 1;
  const out: CurvePoint[] = [{ t: clean[0].t, equity_pct: 0 }];
  for (let index = 1; index < clean.length; index += 1) {
    const previous = clean[index - 1].equity;
    const current = clean[index].equity;
    let intervalReturn = current / previous - 1;
    if (Math.abs(intervalReturn) * 100 > jumpThresholdPct) intervalReturn = 0;
    growth *= 1 + intervalReturn;
    out.push({
      t: clean[index].t,
      equity_pct: Number(((growth - 1) * 100).toFixed(6)),
    });
  }
  return out;
}

function lastFinitePercent(points: CurvePoint[]): number | null {
  for (let index = points.length - 1; index >= 0; index -= 1) {
    const value = points[index]?.equity_pct;
    if (value != null && Number.isFinite(value)) return value;
  }
  return null;
}

function equityAt(
  points: NonNullable<BotSeries["account_curve_abs"]>,
  timestamp: number,
): number | null {
  let value: number | null = null;
  for (const point of points) {
    if (point.t > timestamp) break;
    if (Number.isFinite(point.equity) && point.equity > 0) value = point.equity;
  }
  return value;
}

/** Value-weighted raw return on the clock shared by every included account. */
export function rawCommonScopeReturnPct(series: BotSeries[]): number | null {
  const eligible = series.filter(
    (item) =>
      item.id !== "counter-sl-reverse" &&
      (item.account_curve_abs?.length || 0) >= 2,
  );
  if (!eligible.length) return null;
  const start = Math.max(
    ...eligible.map((item) => item.account_curve_abs![0].t),
  );
  const end = Math.min(
    ...eligible.map(
      (item) => item.account_curve_abs![item.account_curve_abs!.length - 1].t,
    ),
  );
  if (end <= start) return null;
  let startTotal = 0;
  let endTotal = 0;
  for (const item of eligible) {
    const startValue = equityAt(item.account_curve_abs!, start);
    const endValue = equityAt(item.account_curve_abs!, end);
    if (startValue == null || endValue == null) return null;
    startTotal += startValue;
    endTotal += endValue;
  }
  if (startTotal <= 0) return null;
  return 100 * (endTotal / startTotal - 1);
}

/**
 * Return KPI semantics follow the active chart:
 * - Equity % mirrors the raw selected portfolio percent series.
 * - Corrected Return mirrors the deposit/withdrawal-adjusted portfolio curve.
 * - Equity $ prefers the confirmed-ledger return and falls back to the
 *   cashflow-neutralized compatibility curve while ledger sync is pending.
 * - Price Move is supplied by the caller in its display unit (BPS in the UI).
 * - Strategy return stays unavailable here unless a complete per-bot curve is
 *   rendered directly; it is never inferred from account equity.
 */
export function returnPctForView(
  mode: ChartMode,
  portfolio: PortfolioSeries | null,
  tradeReturnPct: number | null,
  rawAccountReturnPct: number | null = null,
): number | null {
  if (mode === "trade") return tradeReturnPct;
  if (mode === "strategy") return null;
  if (!portfolio) return null;
  if (mode === "account") return lastFinitePercent(portfolio.account_curve || []);
  const correctedReturn = (): number | null => {
    const curve = correctedCurveOrJumpTwr(
      portfolio.corrected_curve,
      portfolio.account_curve_abs,
      10,
      true,
      portfolio.account_curve,
    );
    return curve.length >= 2 ? lastFinitePercent(curve) : null;
  };
  if (mode === "corrected") {
    return correctedReturn();
  }
  const metric = portfolio.cashflow_return;
  if (
    metric?.available &&
    (metric.return_pct == null || !Number.isFinite(metric.return_pct))
  ) return null;
  if (metric?.available) return metric.return_pct;
  // Kept in the signature for older callers, but raw equity must never become
  // the performance KPI because user deposits/withdrawals would distort it.
  void rawAccountReturnPct;
  return correctedReturn();
}
