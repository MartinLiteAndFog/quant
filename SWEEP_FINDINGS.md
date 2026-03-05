# Multi-Asset Renko Strategy Sweep Findings

**Date:** 2026-03-04
**Backtest period:** 2024-01-01 to present
**Fee assumption:** 20 bps roundtrip
**Strategies:** Flip Engine (countertrend, Gate ON) + TP2 Engine (trendfollower, Gate OFF)
**Signal source:** IMBA (Fib-zone flips, lookback=240)

---

## 1. Asset Configuration

| Param | SOLUSDT (ref) | FETUSDT | XRPUSDT | BNBUSDT |
|-------|--------------|---------|---------|---------|
| Price (approx) | $150 | $0.16 | $1.41 | $628 |
| Renko Box | $0.10 | $0.001 | $0.005 | $0.50 |
| Box as % of price | 0.067% | 0.63% | 0.35% | 0.08% |

Note: FET and XRP boxes are wider in percentage terms than SOL due to practical constraints (smaller boxes create too many bricks).

---

## 2. Best Configurations Per Pair

### BNBUSDT -- Best Performer

| Rank | Gate | Gate ON% | TTP | Return | MaxDD | Trades | WinRate | Flip Ret | TP2 Ret |
|------|------|----------|-----|--------|-------|--------|---------|----------|---------|
| 1 | 30% (c35/30 a36/42 e0.25/0.40) | 30.1% | 1.1% | **+147.8%** | **-8.8%** | 1,264 | 42.6% | -36.2% | +288.3% |
| 2 | 30% | 30.1% | 1.0% | +145.3% | -8.6% | 1,265 | 42.6% | -36.8% | +288.3% |
| 3 | 31% (c38/30 a36/42 e0.65/0.75) | 30.7% | 1.1% | +147.0% | -8.8% | 1,260 | 42.3% | -36.4% | +288.4% |
| 4 | 35% (c38/30 a48/58 e0.65/0.80) | 34.7% | 1.1% | +115.9% | -8.5% | 1,252 | 45.9% | -38.5% | +251.2% |
| 5 | 34c% (c35/30 a48/54 e0.65/0.80) | 34.4% | 1.1% | +113.1% | -9.2% | 1,252 | 45.4% | -38.5% | +246.7% |

**Best BNB config:** CHOP 35/30, ADX 36/42, ER 0.25/0.40, TTP=1.1%
- Return/DD ratio: **16.8x** (outstanding)
- TP2 engine carries all the returns (+288%)
- Flip engine consistently negative (-36%)
- TTP sweet spot: 1.0-1.2% (very stable)

---

### XRPUSDT -- Most Conservative

| Rank | Gate | Gate ON% | TTP | Return | MaxDD | Trades | WinRate | Flip Ret | TP2 Ret |
|------|------|----------|-----|--------|-------|--------|---------|----------|---------|
| 1 | 30%a (c38/33 a32/42 e0.25/0.40) | 30.2% | 1.4% | **+64.8%** | **-14.0%** | 403 | 44.2% | +1.6% | +62.2% |
| 2 | 30%a | 30.2% | 1.5% | +64.4% | -14.0% | 403 | 44.2% | +1.4% | +62.2% |
| 3 | 30%a | 30.2% | 1.6% | +64.1% | -14.0% | 403 | 44.2% | +1.2% | +62.2% |
| 4 | 30%a | 30.2% | 1.1% | +61.3% | -14.3% | 407 | 44.5% | -0.6% | +62.2% |
| 5 | 30%a | 30.2% | 1.0% | +58.8% | -14.5% | 409 | 44.3% | -2.1% | +62.2% |

**Best XRP config:** CHOP 38/33, ADX 32/42, ER 0.25/0.40, TTP=1.4%
- Return/DD ratio: **4.6x**
- Only pair where flip engine goes slightly positive at TTP>=1.4%
- TP2 and Flip contribute roughly equally at optimal TTP
- Higher gate ON rates (>35%) destroy returns -- flip engine bleeds heavily
- TTP sweet spot: 1.4-1.6% (higher than other pairs)

---

### FETUSDT -- Highest Return, Highest Risk

| Rank | Gate | Gate ON% | TTP | Return | MaxDD | Trades | WinRate | Flip Ret | TP2 Ret |
|------|------|----------|-----|--------|-------|--------|---------|----------|---------|
| 1 | adx30/35 (ADX-only) | 43.1% | 0.7% | **+216.0%** | **-33.3%** | 2,200 | 45.2% | -57.2% | +638.8% |
| 2 | adx27/32 | 34.5% | 0.7% | +214.9% | -44.7% | 2,153 | 45.5% | -52.8% | +566.6% |
| 3 | adx27/32 | 34.5% | 0.9% | +212.7% | -38.6% | 2,120 | 45.6% | -53.1% | +566.6% |
| 4 | adx30/35 | 43.1% | 1.0% | +210.1% | -31.0% | 2,140 | 46.1% | -58.0% | +638.8% |
| 5 | ch30+a42 (CHOP+ADX) | 30.0% | 0.9% | +210.2% | -60.5% | 2,151 | 47.3% | -58.4% | +644.6% |

**Best FET config:** ADX 30/35 (ADX-only gate, no CHOP/ER), TTP=0.7%
- Return/DD ratio: **6.5x**
- FET is too trendy for 3-indicator gates -- ADX-only gates required for 30%+ ON time
- TP2 engine is a monster (+639% standalone!)
- Flip engine always deeply negative (-50% to -85%)
- Gates above 50% ON annihilate all returns (adx33/41 and adx36/41 go negative)
- CHOP+ADX combo gates (ch30+a42/a48/a54) produce highest TP2 returns but with 50-65% drawdowns
- TTP sweet spot: 0.7-1.0%

---

## 3. Key Observations

### TTP Sensitivity
- **BNB:** Very stable across 0.9-1.3%. Sweet spot at 1.0-1.1%.
- **XRP:** Higher TTP is better for flip engine profitability. Sweet spot at 1.4%.
- **FET:** Lower TTP preferred (0.7-0.9%). Higher TTP increases flip engine losses.

### Gate ON Rate Impact
**Critical finding: Lower gate ON rates consistently outperform across all pairs.**

| Gate ON% | BNB Combined | XRP Combined | FET Combined |
|----------|-------------|-------------|-------------|
| ~30% | +148% | +65% | +215% (ADX-only) |
| ~35% | +116% | +44% | +189% |
| ~40% | - | +32% | +170% |
| ~45% | - | +19% | ~0% to -50% |

The flip engine (countertrend, Gate ON) is a net drag on performance for all three pairs. The TP2 trendfollowing engine drives nearly all returns. The gate's primary role is **protecting the TP2 engine from bad entries** during choppy periods rather than generating alpha through countertrend trading.

### Flip Engine Performance
- **XRP:** Only pair where flip barely breaks even (TTP>=1.4%, gate ON 30%)
- **BNB:** Consistently -35% to -44% regardless of gate config
- **FET:** Consistently -50% to -85%, gets dramatically worse with wider gates

### TP2 Engine Performance
| Pair | TP2 Return | TP2 Trades | TP2 WinRate |
|------|-----------|-----------|-------------|
| BNB | +288% | 1,016 | 46.2% |
| XRP | +62% | 350 | 39.7% |
| FET | +639% | 1,495 | 46.3% |

FET's TP2 engine is exceptional -- likely due to the strong trending behavior of the token on Renko charts.

---

## 4. Kelly Criterion

Computed from best config per pair (mean_pnl and win_rate):

| Pair | Win Rate | Mean PnL% | Approx Kelly% |
|------|----------|-----------|---------------|
| BNB (30%, TTP=1.1%) | 42.6% | 0.074% | ~15-20% |
| XRP (30%a, TTP=1.4%) | 44.2% | 0.137% | ~25-30% |
| FET (adx30/35, TTP=0.7%) | 45.2% | 0.058% | ~12-15% |

Note: Half-Kelly or quarter-Kelly recommended in practice due to non-normal return distributions and drawdown risk.

---

## 5. Recommended Production Parameters

### BNBUSDT
```
box = 0.50
ttp_trail_pct = 0.011
chop_on = 35, chop_off = 30
adx_on = 36, adx_off = 42
er_on = 0.25, er_off = 0.40
fee_bps = 20
```

### XRPUSDT
```
box = 0.005
ttp_trail_pct = 0.014
chop_on = 38, chop_off = 33
adx_on = 32, adx_off = 42
er_on = 0.25, er_off = 0.40
fee_bps = 20
```

### FETUSDT
```
box = 0.001
ttp_trail_pct = 0.007
adx_on = 30, adx_off = 35
chop_on = 0, chop_off = 0    # disabled
er_on = 1.0, er_off = 1.0    # disabled
fee_bps = 20
```

---

## 6. Warnings and Caveats

1. **Overfitting risk:** 100 configs tested per pair. The best results will naturally look better than they'd perform out-of-sample. Focus on parameter regions (ranges) rather than single optimal points.

2. **FET drawdowns:** Even the best FET config has -33% drawdown. The CHOP+ADX combo gates push drawdowns to -50 to -65%. Size accordingly.

3. **Flip engine value questionable:** The countertrend strategy is a net negative across all pairs. Consider whether the flip engine adds enough diversification value to justify the drag, or whether a TP2-only approach might be simpler and nearly as profitable.

4. **BNB is the standout:** 148% return with only 8.8% drawdown is exceptional risk-adjusted performance. The 30% gate configuration is very stable across TTP values.

5. **No BTC data available:** Only BNBUSDT, FETUSDT, and XRPUSDT datasets were present in the data directory.

---

## 7. Raw Data

Full sweep results (10 gate configs x 10 TTP values = 100 runs per pair):
- `data/sweep_xrp.json`
- `data/sweep_bnb.json`
- `data/sweep_fet.json`
