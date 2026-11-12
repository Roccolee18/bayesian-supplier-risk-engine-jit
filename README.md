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
| trailing_mean_delay | 30-day trailing average delay | Feature | Delay Model |
| on_time_rate | Proportion of deliveries made on-time versus experiencing a delay | Feature | Delay Model |
| safety_inv | Amount of in-house safety stock | Feature | Prod. Constraint Calculation |
| usage_per_hour | Part's usage per hour | Feature | Prod. Constraint Calculation |
| prev_num_pulled_in | Number of times a later invoice had to be pulled in due to empty lineside inventory from past shortages | Feature | Prod. Constraint Calculation |
| prev_shortages | Number of shortages within the previous 30 days | Feature | Prod. Constraint Calculation |
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

- Note: A naive model of supplier's trailing average delays will act a baseline that all 4 models have to beat

- The final model that will be selected to power the PowerBI dashboard is the model that has the best CRPS score

- Clustering will be done based on supplier features, such as mean delay, on-time rate, usage per hour, etc.
- Each cluster is only fit inside its training window to prevent leakage, and metrics will be derived from walk-forward folds plus a final held-out testing period.

- Candidate clustering algorithms that will be tested are: 
    - K-means clustering as a baseline
    - Gaussian Mixture Models as cluster probabilities fit better in a Bayesian pipeline
    - Hierarchical clustering to check that the clustering structure holds up

- The number of clusters will be chosen based on peaks on a Silhouette score , BIC plus a bootstrap stability check

## Results
- WIP

<!-- - One results table: models x metrics, per scenario, with intervals
- 1–2 key plots (e.g. calibration, cold-start vs. established suppliers)
- What it means in plain language -->

## From model to decision
The output pipeline will be as follows:
1. The model will output a distribution of predicted delayed delivery rates for each upcoming delivery in the next 72 hours
2. All predicted delayed delivery rate will be converted into a probability of being a production constraint within the next 72 hours by the following:
    1. Coverage for the part will be calculated as (In-House Inventory + Safety Stock) / Usage Per Hour
    2. Predictive draws will be made from the posterior where delay exceeds coverage
3. The top 5 highest predicted delayed delivery rate and calculated probability of production constraint will be displayed on a PowerBI dashboard in a table format for investigation by analysts, directed by the senior manager


## Limitations and next steps
- The biggest caveat to this project is the use of simulated data in the place of real operational data. Therefore, ensuring fairness in the simulated dataset and trying to mimic, as much as possible, the volume and abnormalities experienced in a real manufacturing environment has been a top priority for this project. This was done by making broad, conservative assumptions about delivery metrics (such as on-time rate, delay amount, etc.) and exploring the possibility of supplier cluster structure existing, not existing, or shifting with time, and how that will impact model performance. 
- Despite this, there is still room for improvement:
1. Assumptions such as there being steady demand throughout the year, the number of parts and deliveries made, and only 1 model being produced by the plant will probably not hold up in a real manufacturing plant. The impact of these assumptions being violated would introduce more variability to the model, potentially causing further non-convergence in the Hierarchical models. However, with more delivery volume, the models will have more data to digest, which may outweigh the additional variability. 
2. Delays in a real manufacturing plant are likely to be messier than the zero-inflated log-normal distribution used in this simulation. Hence, the results generated are the "best case" for fitting a model to.
3. No real-world delay drivers (such as weather, boarder crossing, traffic, etc.) are included in the model 


- Further, performing the experiment with real data would allow me to, as next steps:
1. Confirm whether suppliers form stable clusters at all
2. Incorperate short-time horizon signals to improve model accuracy, such as border wait times, weatehr, and carrier ETA
3. Confirm an appropriate delay distrubution was chosen to fit a model to
4. Confirm the impact of this analysis pipeline through logging actions taken, since a prevented shortage can look like a wrong prediction on paper.


## Reproducibility

### Environment Setup
1. Install Anaconda and Git Bash, and navigate to the working directory in which this repo was cloned
2. To replicate the environment, simply run `conda env create -f environment.yml`
- This creates a virtual environment with all required packages installed

### Running The Experiment
Follow this sequence to build the data, engineer the features and build each training fold to ensure no leakage:
1. Run `python ./src/generate_all.py`
    - This script creates the simulated data based on a random seed. Extra flags can be passed when running this command to generate data for a specific scenario. See the Python file for the specific syntax
2. Run `python ./src/build_master_panel.py`
    - This script builds the master panel from which features and, eventually, models will be built from. Each row represents 1 delivery, and the output master_panel.parquet file is saved under `./data/raw/`
3. Run `python ./src/build_features.py`
    - This script uses the master panel to build leakage-safe modelling features from the master panel. A JSON file describing each column's role and a `features.parquet` file will be saved under `./data/processed/<scenario>/` for each scenario
4. Run `python ./src/build_folds.py`
    - This script performs the training and testing data split and builds training folds for cross-validation. Imputation and clustering algorithms are also fit at this step. Outputs will be stored in `data/processed/<scenario>/folds/`
5. Run `python ./src/evaluate_models.py`
    - This script fits and scores the naive baseline and models 1-4 on every walk-forward fold. Each scenario will have 4 outputs: 
        1. `metrics.csv` - one row per fold x model (all metrics, fit time, diagnostics)
        2. `summary.csv` - mean across CV folds + paired CRPS difference vs naive
        3. `predictions/<fold>_<model>_shipments.parquet` - per-shipment forecast summary
        4. `predictions/<fold>_<model>_rows.parquet` - Calculated probability of being a shortage per delivery (row)
    - The script can be configured to run only a subset of models, a certain scenario, a quick smoke test or the final holdout set based on the flag it is executed with. Specific flag usage can be found in the Python file itself.
    - Note: Step 5 takes a while to run
    - Note: a (fold, model) pair already in metrics.csv is skipped unless --overwrite is passed, so a long run can be stopped and restarted.

## Tech stack
Python, PyMC, scikit-learn, Power BI