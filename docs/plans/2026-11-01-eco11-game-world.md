# ECO-11 game-world cockpit

This package adds a selectable Fleet Cockpit research surface for the ECO-11 game-world issue. It is deliberately a frontend-only prototype with deterministic fixtures and no connection to execution, account, portfolio, webhook, or order APIs.

## Views

- **Surface meadow:** only the selected strategy family is plotted. The two axes name actual genome fields and versioned ranges from the selected parameter meadow (IMBA sensitivity × TTP, or RSI period × stop distance). Plants with missing axis genes are counted explicitly and omitted from neither the society roster nor a hidden bucket; a future catalog integration can place them in an unassigned category.
- **Phenotypes:** simple SVG grass, flower, shrub, cactus, and orchid marks; visible roots, family color, vitality, sun, water table, and a sparse bee animation. Plant height switches between simulated profit and logarithmic biomass.
- **Underground:** paper candidates are shown as aggregate research clusters only, never as one SVG node per candidate. The count and scores are labelled synthetic.
- **Clock:** play/pause advances the local demo day; the step control advances one day. These controls mutate only component state.

## Evidence boundary

All plants, genome values, groundwater, ecological fields, research clusters, and scores are reproducible illustrative fixtures. No historical market data or strategy results are read. The surface displays `SYNTHETISCHE FIXTURES`; underwater values explicitly identify their simulated status. This view must remain separate from empirical evidence and live trading surfaces.

## Rendering limits and later integration

The current fixture has 36 surface plants and the renderer caps its nodes at 1,200. Underworld candidate counts do not affect render count. A future backend may supply paginated/clustered snapshots through a typed view-model adapter; this package does not add arbitrary genome execution, capital allocation, or trading behavior. The game surface is selected through a dedicated button and leaves the existing meadow and primary hero chart available.
