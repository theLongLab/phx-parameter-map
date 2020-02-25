# phx-parameter-map
==============================

Map PoolHapX parameter sets to MCC/JSD ratios. This repository serves as the code base for the
inner model of the Deep Learning for Haplotype Reconstruction workflow. The code here serves to:
* Set the hyperparameter search space of a ML model.
* Perform hyperparameter optimization through optuna.
* Train and score the ML model with optimal hyperparameters.
* Serialize the trained model.

## Getting Started
### Prerequisites
* Python 3.7.6
* pandas 0.24.2
* optuna 0.19.0
* scikit-learn 0.22
* xgboost 0.90
* pytorch 1.3.1
* cudatoolkit 10.1.243 (optional, if using GPU for xgboost)


## Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models as well as the hyperparameter search space
    │   │                     settings file.
    │   ├── hyperparameter_search.json
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to tune, train, and score models and then serialize trained models.
    │   │   ├── base_tuner.py
    │   │   ├── model_tuners.py
    │   │   └── score_models.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project partially based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
