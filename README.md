# Bayesian Supplier Risk Engine for JIT Manufacturing
Forecast which late supplier deliveries would cause parts shortages on just-in-time assembly lines within 72 hours, across 3 simulated plants and 366 suppliers.

## TL;DR
- Question: do supplier clusters improve delay forecasts beyond hierarchical pooling?
- Deliverable: 72-hour shortage-risk scores feeding a Power BI report

## Project Outputs

A detailed report outlining the dataset, methodology, results and next steps can be found under `reports/report.pdf`

## Repo Structure

bayesian-supplier-risk-engine-jit/
├── .gitignore                  # Specifies files and folders Git should ignore
├── environment.yml             # Dependencies for reproducing the environment
├── README.md                   # Top-level project overview and setup instructions
├── img/                        # Generated graphics and figures for reporting
├── notebooks/                  # Jupyter notebooks for validation and experimentation
|   └── build_features.ipynb    # Notebook to validate feature engineering script
|   └── build_folds.ipynb       # Notebook to validate train/test split script
|   └── evaluate_models.ipynb   # Notebook to validate model evaluation script
├── reports/                    # Generated analysis as PDF and the raw quarto file
|   └── report.qmd              # Raw report file to be rendered to PDF
├── results/                    # Model metrics per fold and a summary of training, by scenario
|   └── clusters/
|   └── drift/
|   └── no_clusters/
├── src/                        # Source code for use in this project
|   └── build_*.py              # Scripts to preprocess raw data into modeling features and train/test splits
|   └── config.py               # Shared settings and helpers for the supplier-risk data simulator
|   └── evaluate_models.py      # Script to fit and score the naive baseline and models 1-4 on every walk-forward fold
|   └── generate_*.py           # Scripts to create simulated dataset
|   └── models.py               # Script of functions that fit the naive baseline and the four Bayesian hurdle models.
|   └── scoring.py              # Script of functions to score models for probabilistic delay forecasts.


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