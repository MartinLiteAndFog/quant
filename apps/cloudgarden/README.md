# Cloudgarden

Cloudgarden is a standalone, synthetic strategy-ecosystem game. It has its own React/Vite entry point under `apps/cloudgarden`; it does not boot Fleet Cockpit, poll Fleet APIs, read exchange accounts, or place orders.

The meadow shows five strategy-seeds as plant phenotypes. A deterministic local day-step changes biomass, vigor, fitness, light, and the shared water table. The Understory view displays synthetic research candidates and bounded pollination concepts. A stable world seed makes the same sequence of day steps reproducible.

All values are toy simulation inputs and outputs. “Fitness,” “vigor,” and observation days are not market evidence, performance claims, or permission to promote a candidate. Nothing in the app receives capital.

## Run

```sh
npm install
npm run dev
```

The app opens at `http://127.0.0.1:5184`. To verify the simulation model and production build:

```sh
npm test
npm run build
```

The game is intentionally self-contained. Persisted worlds, real strategy executors, market-data ingestion, and registry or venue adapters are outside this first standalone prototype.
