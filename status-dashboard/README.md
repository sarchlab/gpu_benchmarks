# GPU Bench Status Dashboard

Small React/Vite status page for a live `gpubench run`.

## Run

For live reload while editing the React app:

```bash
cd status-dashboard
npm install
npm run dev
```

Then open the Vite URL, usually <http://127.0.0.1:5173>.

For the production-style local server:

```bash
cd status-dashboard
npm install
npm run build
npm run preview
```

Then open <http://127.0.0.1:4174>.

By default the server reads `../results.db`. To point it at another database:

```bash
GPUBENCH_DB=/path/to/results.db npm run preview
```

The page polls `/api/status` every 5 seconds.

## Results Explorer

The dashboard includes an execution-time chart for collected benchmark results.
Each property has a multi-select filter and a channel assignment:

- `x`: x-axis for the line chart
- `y`: execution time
- `color`: line/color grouping; multiple color properties form combination labels
- `aggregate`: average over this property

By default the scaling parameter is `x`, execution time is `y`, non-scaling
parameters are `color`, and repetitions are aggregated. Repeated samples are
aggregated into a mean, with shaded bands and error bars showing the 95%
confidence interval.
