# bayesian-supplier-risk-engine-jit
Forecast which late supplier deliveries would cause parts shortages on just-in-time assembly lines within 72 hours, across 3 simulated plants and 366 suppliers.

## TL:DR
- Question: do supplier clusters improve delay forecasts beyond hierarchical pooling?
- Deliverable: 72-hour shortage-risk scores feeding a Power BI report

## Problem definition
JIT is a form of lean inventory management in manufacturing processes that describes parts are being ordered in quantities equal to those that were sold. 

This form of inventory management has many advantages, including storge cost savings, limiting obsolescence and high adaptability to fluctuating demand levels. However, it also faces unique challenges, such as high vulnerability to supply chain disruptions, high administrative and freight costs, and most importantly, a heavy reliance on on-time deliveries from suppliers. "Late deliveries by supplier" are often a key metric in JIT reporting, despite the difficulty of forcasting it.

This report aims to help senior managers of a manufacturing plant take preventative actions to mitigate production impact by showing the predicted delayed delivery rate for each supplier within the next 72 hours.

## Data


## Methodology


## Results


<!-- - One results table: models x metrics, per scenario, with intervals
- 1–2 key plots (e.g. calibration, cold-start vs. established suppliers)
- What it means in plain language -->

## From model to decision


<!-- How forecasts become 72h shortage risk, and what the dashboard shows. -->

## Limitations and next steps


<!-- Simulated data, assumptions that may not hold in real plants,
what you'd do with real data (e.g. a Bayesian mixture model). -->

## Reproduce it


<!-- Setup, then one command to rerun everything. Repo structure tree. -->

## Tech stack
Python, PyMC, scikit-learn, Power BI