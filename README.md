# phx-parameter-map
Map PoolHapX parameter sets to MCC/JSD ratios. This repository serves as the code base for the
inner model of the Deep Learning for Haplotype Reconstruction workflow. The code here serves to:
* Set the hyperparameter search space of a ML model.
* Perform hyperparameter optimization through optuna.
* Train and score the ML model with optimal hyperparameters.
* Serialize the trained model.

## Getting Started
### Prerequisites
* Python 3.7.6
* Pandas 0.24.2
* Optuna 0.19.0
* Scikit-Learn 0.22
* XGBoost 0.90
* PyTorch 1.3.1
* CudaToolKit 10.1.243 (optional, if using GPU for xgboost)


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

## Running the Scripts
#### Data Compilation
```bash
python src/data/make_dataset.py <phx_dir_path> <output_file_paths>
```
* ```phx_dir_path```: the directory path for the base PoolHapX folder.
* ```output_file_paths```: the file name for the compiled list of output directories placed under ```data/raw/```.

#### Data Pre-Processing
```base
python src/features/build_features.py <output_file_paths>
```
* ```output_file_paths```: the aformentioned file name under ```data/raw/```.

#### Hyperparameter Search, Model Scoring, and Model Serialization
Prior to running the following, ensure that the hyperparameters and their search spaces are properly
specified in ```models/hyperparameter_search.json```.
```base
python src/models/score_models.py <num_trials> <train_test_split_proportion> <num_folds> <model_num> <optional_seed>
```
* ```num_trials```: number of optuna trials.
* ```train_test_split_proportion```: proportion of data set aside for testing.
* ```num_folds```: number of folds for *k*-fold cross validation.
* ```model_num```: the model to tune and train (1-5).
    * 1: LASSO
    * 2: Ridge regression
    * 3: Elastic net
    * 4: Random forest
    * 5: XGBRegressor
* ```optional_seed```: optional seed value.

## Built With
* [XGBoost](https://xgboost.ai/) - gradient boosted decision tree.
* [Scikit-Learn](https://scikit-learn.org/stable/) - machine learning models.
* [Pandas](https://pandas.pydata.org/) - data manipulation.
* [Optuna](https://optuna.org/) - hyperparameter optimization.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
Project partially based on the
[cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/).
