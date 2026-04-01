<script>
  import StatsPanel from './components/StatsPanel.svelte';
  import PriceChart from './components/PriceChart.svelte';
  import RegimeBand from './components/RegimeBand.svelte';
  import AxisBars from './components/AxisBars.svelte';
  import PolarChart from './components/PolarChart.svelte';
  import EquityCurve from './components/EquityCurve.svelte';
</script>

<div class="dashboard">
  <div class="stats-panel">
    <StatsPanel />
  </div>
  <div class="chart-area">
    <PriceChart />
    <RegimeBand />
  </div>
  <div class="right-panel">
    <AxisBars />
    <PolarChart />
  </div>
  <div class="equity-bar">
    <EquityCurve />
  </div>
</div>

<style>
  .dashboard {
    display: grid;
    grid-template-columns: 220px 1fr 240px;
    grid-template-rows: 1fr auto;
    height: 100vh;
    gap: 0;
  }
  .stats-panel { grid-column: 1; grid-row: 1; overflow-y: auto; }
  .chart-area { grid-column: 2; grid-row: 1; display: flex; flex-direction: column; min-height: 0; }
  .right-panel { grid-column: 3; grid-row: 1; display: flex; flex-direction: column; }
  .equity-bar { grid-column: 1 / -1; grid-row: 2; }

  @media (max-width: 1200px) {
    .dashboard {
      grid-template-columns: 220px 1fr;
      grid-template-rows: 1fr auto auto;
    }
    .right-panel { grid-column: 1 / -1; grid-row: 2; flex-direction: row; }
    .equity-bar { grid-row: 3; }
  }
  @media (max-width: 800px) {
    .dashboard {
      grid-template-columns: 1fr;
      grid-template-rows: auto 1fr auto auto;
      overflow-y: auto;
    }
    .stats-panel { grid-column: 1; grid-row: 1; }
    .chart-area { grid-column: 1; grid-row: 2; min-height: 400px; }
    .right-panel { grid-column: 1; grid-row: 3; flex-direction: row; }
    .equity-bar { grid-column: 1; grid-row: 4; }
  }
</style>
