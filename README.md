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
    - The simulated dataset will consist of made-up orders for 1 year of production
    - The parts base will consist of a simplified set of parts with parameters chosen to be plausible for an assembly plant.
    - Delays are zero-inflated with a log-normal tail
    - Usage-per-part (and therefore, product demand) is held constant
    - Suppliers are expected to ship each part at least once in the simulation period
    - The simulated plants only produce 1 model 
- Using simulated data calls for separate scenarios to be run to remove artefacts of generated data. 4 models will be fit within each scenario, and ranking discrepancies between scenarios will be analyzed.
    1. Scenario 1: True supplier clusters exist. The generated dataset will include clear indicators of supplier reliability that allow for clear clusters to be formed. 
    2. Scenario 2: No clusters exist. Supplier reliability is drawn from one continuous distribution, with no group structure
    3. Scenario 3: Clusters exist but drift over time. The generated dataset will start each supplier off with a cluster, but then have their performance drift to another cluster throughout the simulation period. Specifically, 20% of suppliers will switch clusters at month 3, another 10% will switch at month 7
- Below is a brief description of the key columns in the dataset:

| Column | Description | Used As | Used In |
| :--- | :--- | :--- | :--- |
| planned_delivery_datetime | Planned date and time of delivery | Identifier | Delay Model |
| shipment_id | ID of shipment | Identifier | Delay Model |
| part_code | Internal part code | Identifier | Delay Model |
| supplier_code | Internal supplier code | Identifier | Delay Model |
| plant | Delivery destination | Identifier | Delay Model |
| lead_time_days | Planned order-to-deliver time | Feature | Delay Model |
| sup_late_rate_30d | 30-day trailing average delay | Feature | Delay Model |
| safety_inv | Amount of in-house safety stock | Feature | Prod. Constraint Calculation |
| usage_per_hour | Part's usage per hour | Feature | Prod. Constraint Calculation |
| prev_num_pulled_in | Number of times a later invoice had to be pulled in due to empty lineside inventory from past shortages | Feature | Prod. Constraint Calculation |
| part_shortages_30d | Number of shortages within the previous 30 days | Feature | Prod. Constraint Calculation |
| avail_in_other_plants | Whether the part is available in other plants | Feature | Prod. Constraint Calculation |
| delay_hours | Number of hours the shipment was delayed | Target | Delay Model |
| actual_delivery_datetime | Actual date and time of delivery | Target Source | Delay Model |
| true_cluster | Supplier's latent cluster from the simulator | Evaluation only | Delay Model |


## Methodology
- The 4 models are laid out in a 2x2 design, as illustrated below:

|  | No Clustering | Clustering |
| :--- | :--- | :--- |
| Non-Hierarchical Bayes | Model 1 | Model 2 |
| Hierarchical Bayes | Model 3 | Model 4 |

- Note: A naive model of each supplier's own past delays in the training window will act a baseline that all 4 models have to beat. Suppliers with fewer than 20 past deliveries use all suppliers’ delays pooled.

- The final model that will be selected to power the PowerBI dashboard is the model that has the best CRPS score. If performance gain between 2 models are close, the simplest model will be chosen

- Clustering will be done based on supplier features, such as late rate, mean late delay and delay spread.
- Each cluster is only fit inside its training window to prevent leakage, and metrics will be derived from walk-forward folds plus a final held-out testing period.

- Candidate clustering algorithms that will be tested are: 
    - K-means clustering as a baseline
    - Gaussian Mixture Models as cluster probabilities fit better in a Bayesian pipeline
    - Hierarchical clustering to check that the clustering structure holds up

- The number of clusters will be chosen based on peaks on a Silhouette score , BIC plus a bootstrap stability check

## Results

### Clustering Results
The results from the `build_folds.ipynb` notebook all support a 4-cluster structure within the data. 

While most algorithms across the 4 folds and the holdout set point to this conclusion, hierarchical and k-means clustering found a 3-cluster structure within the first 2 folds. However, folds 3 and 4 for both of the algorithms reverted back to the majority conclusion of a 4-cluster structure. The `true_ari` score for fold 2 further point to this result being unreliable, as a significantly lower score was observed for this fold between the two algorithms.

Below is a table summarising the key characteristics of each cluster, ordered by supplier membership:

| Cluster | # of Suppliers | Late Rate | Mean Delay Hours |
| :--- | :--- | :--- | :--- |
| C1 | 149 | 0.08 | 0.553 |
| C2 | 133 | 0.215 | 2.635 |
| C3 | 76 | 0.211 | 0.869 |
| NEW | 8 | 0.124 | 0.594 |

C1 having the largest population of suppliers and the lowest mean delay hours makes sense, as most suppliers should be able to make the majority of their shipments on time, with occasional minor delays. C2 presents an interesting case, as it shows that the second largest population of suppliers are ones with the highest late rates and longest mean delay hours. C3 seems to be middle-of-the-road suppliers who provide late delivers some of the time but not by much. Finally, the NEW cluster is assigned to suppliers who started supplying parts mid-year. This cluster's results should be taken carefully as they won't have as much data as others, and each individual delay pulls the mean delay hours metric a lot more due to the smaller sample size.

With these results, each cluster can be labeled as the following:

| Cluster | Label |
| :--- | :--- |
| C1 | Reliable Suppliers |
| C2 | Inconsistant Suppliers |
| C3 | Average Suppliers | 
| NEW | New Suppliers |

### Cross-Validation Results
After examining the results across each fold per scenario, it can be concluded that supplier clusters do not improve delay and shortage forecasts beyond hierarchical pooling. Adding clusters changed CRPS by about 0.1% when true clusters existed, made forecasts consistently worse when suppliers drifted between clusters (+0.3%, all 4 folds), and hurt new suppliers that joined mid-year. Partial pooling also added little performance (≤0.25%).

Another key finding is that the models only beat a naive baseline when supplier behaviour drifts. With stable suppliers and long histories, replaying each supplier’s past delays was as accurate as any model. Under drift, every model beat it in all 4 folds (about 2% CRPS, growing to 2.4% by the last fold), removed the baseline’s tendency to under-predict lateness, and improved shortage flagging (precision@10 from 0.375 to 0.41). The gain comes from recent-window features rather than model structure, and the simplest model (model 1) captured nearly all of it at a fraction of the training time of the hierarchical models. Therefore, this will be the model that advances to holdout testing.

An anomaly of 7 divergences was detected when training fold 2 of model 4 in the `no_clusters` scenario, while every other fit had no divergences detected. 

### Holdout Results
Since model 1 captures nearly all accuracy gains from recent-window feature calculation and being much faster to fit, I will be proceeding with holdout testing using this model versus the naive baseline to validate this conditional finding. The results are shown below:

- **<Holdout test results>**
<!-- - One results table: models x metrics, per scenario, with intervals
- 1–2 key plots (e.g. calibration, cold-start vs. established suppliers)
- What it means in plain language -->

## From model to decision
The output pipeline will be as follows:
1. The model will output a distribution of predicted delayed delivery rates for each upcoming delivery in the next 72 hours
2. All predicted delayed delivery rate will be converted into a probability of being a production constraint within the next 72 hours by the following:
    1. Coverage for the part will be calculated as Safety Stock / Usage Per Hour
    2. Predictive draws will be made from the posterior where delay exceeds coverage
3. The top 10 highest predicted delayed delivery rate and calculated probability of production constraint will be displayed on a PowerBI dashboard in a table format for investigation by analysts, directed by the senior manager


## Limitations and next steps
- The biggest caveat to this project is the use of simulated data in the place of real operational data. Therefore, ensuring fairness in the simulated dataset and trying to mimic, as much as possible, the volume and abnormalities experienced in a real manufacturing environment has been a top priority for this project. This was done by making broad assumptions about delivery metrics (such as on-time rate, delay amount, etc.) and exploring the possibility of supplier cluster structure existing, not existing, or shifting with time, and how this affects model performance. 
- Despite this, there is still room for improvement:
1. Assumptions such as there being steady demand throughout the year, the number of parts and deliveries made, and only 1 model being produced by the plant will probably not hold up in a real manufacturing plant. The impact of these assumptions being violated would introduce more variability to the model, which could make the hierarchical models harder to fit. However, with more delivery volume, the models will have more data to digest, which may outweigh the additional variability. 
2. Delays in a real manufacturing plant are likely to be messier than the zero-inflated log-normal distribution used in this simulation. Hence, the results generated are the "best case" for fitting a model to.
3. Only winter and short regional disruptions are simulated; drivers such as weather, border crossings, traffic and carrier performance are not
4. Real suppliers share carriers and routes, so delays typically involve more than one shipment per supplier

- Further, performing the experiment with real data would allow me to, as next steps:
1. Confirm whether suppliers form stable clusters at all
2. Incorporate short-time horizon signals to improve model accuracy, such as border wait times, weather, and carrier ETA
3. Confirm an appropriate delay distribution was chosen to fit a model to
4. Confirm the impact of this analysis pipeline through logging actions taken, since a prevented shortage can look like a wrong prediction on paper.


## Reproducibility

### Environment Setup
1. Install Anaconda and Git Bash, and navigate to the working directory in which this repo was cloned
2. To replicate the environment, simply run `conda env create -f environment.yml`
3. Activate the environment with `conda activate supplier-risk-engine`
- This creates a virtual environment with all required packages installed

### Running The Experiment
Follow this sequence to build the data, engineer the features and build each training fold to ensure no leakage:
1. Run `python ./src/generate_all.py`
    - This script creates the simulated data based on a random seed. Extra flags can be passed when running this command to generate data for a specific scenario. See the Python file for the specific syntax.
2. Run `python ./src/build_features.py`
    - This script uses the master panel to build modelling features from the master panel. A JSON file describing each column's role and a `features.parquet` file will be saved under `./data/processed/<scenario>/` for each scenario. A built-in leakage check is ran alongside feature calculation.
3. Run `python ./src/build_folds.py`
    - This script performs the training and testing data split and builds training folds for cross-validation. Imputation and clustering algorithms are also fit at this step. Outputs will be stored in `data/processed/<scenario>/folds/`
4. Run `python ./src/evaluate_models.py`
    - This script fits and scores the naive baseline and models 1-4 on every walk-forward fold. Each scenario will have 4 outputs: 
        1. `results/<scenario>/metrics.csv` - one row per fold x model (all metrics, fit time, diagnostics)
        2. `results/<scenario>/summary.csv` - mean across CV folds + paired CRPS difference vs naive
        3. `predictions/<fold>_<model>_shipments.parquet` - per-shipment forecast summary
        4. `predictions/<fold>_<model>_rows.parquet` - Calculated probability of being a shortage per delivery (row)
    - The script can be configured to run only a subset of models, a certain scenario or a quick smoke test based on the flag it is executed with. Specific flag usage can be found in the Python file itself.

Notes: 
- By default, the final holdout set is not ran, as it should only be run once at the end for report generation by running the command with a `--holdout` flag
- Step 5 takes a while to run (~1 hour on a 2-core machine)
- A (fold, model) pair already in metrics.csv is skipped unless --overwrite is passed, so a long run can be stopped and restarted.

The notebooks in `notebooks/` walk through each step in detail and run the same code as the scripts.


## Tech stack
Python, PyMC, scikit-learn, Power BI