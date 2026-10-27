# Bayesian Supplier Risk Engine for JIT Manufacturing
Forecast which late supplier deliveries would cause parts shortages on just-in-time assembly lines within 72 hours, across 3 simulated plants and 366 suppliers.

## TL;DR
- Question: do supplier clusters improve delay forecasts beyond hierarchical pooling?
- Deliverable: 72-hour shortage-risk scores feeding a Power BI report

## Problem definition
JIT is a form of lean inventory management in manufacturing processes where parts arrive just as production needs them.

This form of inventory management has many advantages, including storage cost savings, limiting obsolescence and high adaptability to fluctuating demand levels. However, it also faces unique challenges, such as high vulnerability to supply chain disruptions, high administrative and freight costs, and most importantly, a heavy reliance on on-time deliveries from suppliers. "Late deliveries by supplier" are often a key metric in JIT reporting, despite the difficulty of forecasting it.

This project aims to help senior managers of a manufacturing plant take preventative actions to mitigate production impact by showing the predicted delayed delivery rate for each supplier within the next 72 hours.

## Data
- The dataset for this study will be simulated from a prior distribution of plausible values. 
- Each row represents 1 delivery of 1 part from 1 supplier to 1 plant.
- Key assumptions:
    - The parts base will consist of a simplified set of parts with parameters chosen to be plausible for an assembly plant.
    - Delays are zero-inflated with a log-normal tail
    - Usage-per-part (and therefore, product demand) is held constant
    - Suppliers are expected to ship each part at least once in the simulation period
    - The simulated plants only produce 1 model 
- Using simulated data calls for separate scenarios to be run to remove artefacts of generated data. 4 models will be fit within each scenario, and ranking discrepencies between scenarios will be analyzed.
    1. Scenario 1: True supplier clusters exist. The generated dataset will include clear indicators of supplier reliability that allow for clear clusters to be formed. 
    2. Scenario 2: No clusters exist. Supplier reliability is drawn from one continuous distribution, with no group structure
    3. Scenario 3: Clusters exist but drift over time. The generated dataset will start each supplier off with a cluster, but then have their performance drift to another cluster near the end of the simulation period. Specifically, 20% of suppliers will switch clusters at month 3, another 10% will switch at month 7
- Below is a brief description of the key columns in the dataset:

| Column | Description | Used As |
| :--- | :--- | :--- |
| planned_delivery_datetime | Planned date and time of delivery | Identifier |
| shipment_id | ID of shipment | Identifier |
| part_code | Internal part code | Identifier |
| supplier_code | Internal supplier code | Identifier |
| plant | Delivery destination | Identifier |
| lead_time_days | Planned order-to-deliver time | Feature |
| safety_inv | Amount of in-house safety stock | Feature |
| usage_per_hour | Part's usage per hour | Feature |
| prev_num_pulled_in | Number of times a later invoice had to be pulled in due to empty lineside inventory from past shortages | Feature |
| prev_shortages | Number of shortages within the previous 30 days | Feature |
| avail_in_other_plants | Whether the part is available in other plants | Feature |
| actual_delivery_datetime | Actual date and time of delivery | Target |
| delay_hours | Number of hours the shipment was delayed | Target |
| true_cluster | Supplier's latent cluster from the simulator | Evaluation only |


## Methodology
- The 4 models are laid out in a 2x2 design, as illustrated below:

|  | No Clustering | Clustering |
| :--- | :--- | :--- |
| Non-Hierarchical Bayes | Model 1 (naive) | Model 2 |
| Hierarchical Bayes | Model 3 | Model 4 |
- The final model that will be selected to power the PowerBI dashboard is the model that has the best CRPS score

- Clustering will be done based on supplier features, such as mean delay, on-time rate, usage per hour, etc.
- Each cluster is only fit inside its training window to prevent leakage, and metrics will be derived from walk-forward folds plus a final held-out testng period.

- Candidate clustering algorithms that will be tested are: 
    - K-means clustering as a baseline
    - Gaussian Mixed Models as cluster probabilities fit better in a Bayesian pipeline
    - Hierchical clustering to check that the clustering structure holds up

- The number of clusters will be chosen based on peaks on a Silhouette score , BIC plus a bootstrap stability check

## Results


<!-- - One results table: models x metrics, per scenario, with intervals
- 1–2 key plots (e.g. calibration, cold-start vs. established suppliers)
- What it means in plain language -->

## From model to decision
The output pipeline will be as follows:
- The model will output a predicted delayed delivery rate for each supplier
- Each supplier's delayed delivery rate will be split linearly between the parts supplied by that supplier, based on the number of deliveries made for each part
- All predicted delayed delivery rate will be converted into a probability of being a production constraint within the next 72 hours
    - Factors such as our in-house safety stock, usage per hour, and availability in other plants will determine how much a delayed delivery will increase the probability of that part becoming a production constraint
- The top 5 highest predicted delayed delivery rate and calculated probability of production constraint will be displayed on a PowerBI dashboard in a table format for investigation by analysts, directed by the senior manager


## Limitations and next steps


<!-- Simulated data, assumptions that may not hold in real plants,
what you'd do with real data (e.g. a Bayesian mixture model). -->

## Reproduce it


<!-- Setup, then one command to rerun everything. Repo structure tree. -->

## Tech stack
Python, PyMC, scikit-learn, Power BI