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
- The dataset for this study will be simulated from a prior distribution of plausable values. 
- Each row represents 1 delivery from 1 supplier of 1 part to 1 plant at a certain time of day.
- Key assumptions:
    - The simulation only focuses on a fraction of parts from a real-life industrial manufacturing plant
    - Delays are zero-inflated with a log-normal tail
    - Usage-per-part (and therefore, product demand) are held constant
    - Suppliers are expected to ship 1 part per day for a month
- Using simulated data calls for seperate scenarios to be run to remove artefacts of generated data. 4 models will be fit within each scenario, and results will be derived only if they are observable across all scenarios.
    1. Scenario 1: True supplier clusters exist. The generated dataset will include clear indicators of supplier reliability that allow for clear clusters to be formed. 
    2. Scenario 2: No clusters exist. The generated dataset will make each supplier as unique as possible
    3. Scenario 3: Clusters exist but drift over time. The generated dataset will start each supplier off with a cluster, but then have their performance drift to another cluster near the end of the simulation period.
- Below is a brief description of the key columns in the dataset:

| Column | Description | Used As |
| :--- | :--- | :--- |
| delay_hours | Hours late vs. scheduled arrival (0 if on time) | Target |
| lead_time_days | Planned order-to-deliver time | Feature |
| num_overflow | Number of boxes stored in Overflow area | Feature |
| usage_per_hour | Part's usage per hour | Feature |
| num_pulled_in | Number of times a later invoice had to be pulled in due to empty lineside inventory | Feature |
| true_cluster | Supplier's latent cluster from the siumulator | Evaluation only |


## Methodology
- The 4 models are laid out in a 2x2 design, plus the naive baseline, as ilustrated below:
|  | No Clustering | Clustering |
| :--- | :--- | :--- |
| Non-Heirarchical Bayes | Model 1 | Model 2 |
| Heirarchical Bayes | Model 3 | Model 4 |
- The final model that will be selected to power the PowerBI dashboard is the model the performs the best against PR-AUC and ROC-AUC and stable clusters even through walk-forward validation

- Clustering will be done based on supplier features, such as mean delay, on-time rate, usage per hour, etc.
- Each cluster is only fit insidea its training window to prevent leakage

- Candidate models that will be tested are: 
    - K-means clustering as a baseline
    - Gaussian Mixed Models as cluster probabilities fit better in a Bayesian pipeline
    - Heirarchical clustering to check that the clustering structure holds up

- The number of clusters will be chosen based on peaks on a Silhouette score plus a bootstrap stability check


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