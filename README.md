# TELCO CHURN ANALYSIS AND PREDICTION

A full-scale analytical and machine learning project focused on customer churn for a telecom company. The goal is to identify churn patterns, evaluate feature impact, and develop predictive models using structured modular code and reproducible pipelines.

## Project Structure

- `churn_project.ipynb` - main notebook with exploratory analysis, feature engineering, and modeling.
- `Churn Project Report.docx` - formal report with conclusions and business recommendations.
- `LICENSE` - project license (MIT).
- `README.md` - project description and documentation.
- `requirements.txt` - list of Python dependencies required for the project.

- `datasets/` - input data files in `.xlsx` format:
  - `CustomerChurn.xlsx`
  - `Telco_customer_churn.xlsx`
  - `Telco_customer_churn_demographics.xlsx`
  - `Telco_customer_churn_location.xlsx`
  - `Telco_customer_churn_population.xlsx`
  - `Telco_customer_churn_services.xlsx`
  - `Telco_customer_churn_status.xlsx`

- `logs/` - logging directory (currently minimal use; architecture prepared for future logs).

- `utility_functions/` - custom utility module with core project functions:
  - `__init__.py`
  - `data_check.py` - functions for data validation and compatibility checks.
  - `feature_selection.py` - feature selection utilities.
  - `logger.py` - logging configuration.
  - `modeling.py` - functions for building and evaluating ML models.
  - `parameters.py` - hyperparameter grids and aggregation function.
  - `preprocessing.py` - preprocessing logic and pipeline builder.
  - `stats_analysis.py` - distribution checks, statistical profiling, and outlier detection.
  - `visualization.py` - simplified plotting functions and visual helpers.

- `utility_transformers/` - custom transformers (built on `BaseEstimator` and `TransformerMixin`):
  - `__init__.py`
  - `transformers.py` - generic reusable transformers for pipeline integration.


## Key Features

- **Modular and reusable pipeline architecture** for outlier handling, preprocessing, and feature engineering.
- **Careful** dataset **merging** and validation, following **strict rules** to ensure correct **alignment and integrity** of the combined data.
- Extensive set of **utility functions** for automating repetitive tasks across the workflow.
- Strict adherence to **best practices** at each stage of the data science process.
- All transformations implemented as **pipeline steps** to ensure **modularity** and **reproducibility**.
- Scalable design aimed at **semi-automated processing** for future data science / data analysis projects.


## Data Source

Multiple `.xlsx` files sourced from the publicly available Telco Customer Churn datasets:

- Main churn dataset and extended metadata (demographics, services, etc.)
- Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)

## Technologies & Libraries

This project leverages a robust Python data science stack, including:

- **Core Data Processing:**  
  `pandas`, `numpy`, `openpyxl` (for Excel file handling)

- **Data Preparation and Feature Engineering:**  
  `scikit-learn` (pipelines, custom transformers, preprocessing, feature selection),  
  `copy` (deepcopy utility)

- **Machine Learning Models:**  
  `xgboost` (XGBClassifier, XGBRegressor),  
  `lightgbm` (LGBMClassifier, LGBMRegressor),  
  `catboost` (CatBoostClassifier),  
  `scikit-learn` (linear models, tree-based models, SVM, ensemble methods, naive bayes, dummy estimators)

- **Model Evaluation Metrics:**  
  Regression and classification metrics from `scikit-learn.metrics`

- **Visualization:**  
  `matplotlib`, `seaborn`

- **Statistical Analysis:**  
  `scipy.stats` (e.g., `pearsonr`, `f_oneway`),  
  `statsmodels` (variance inflation factor for multicollinearity analysis)

- **Logging and Utility:**  
  `logging`, `colorlog`, `datetime`, `os`

- **Custom Modules:**  
  Project-specific utilities and transformers developed with `scikit-learn` API (`BaseEstimator`, `TransformerMixin`)

---

This combination ensures a scalable, reproducible, and modular pipeline for robust churn analysis and prediction.
## Technologies & Libraries

This project leverages a robust Python data science stack, including:

- **Core Data Processing:**  
  `pandas`, `numpy`, `openpyxl` (for Excel file handling)

- **Data Preparation and Feature Engineering:**  
  `scikit-learn` (pipelines, custom transformers, preprocessing, feature selection),  
  `copy` (deepcopy utility)

- **Machine Learning Models:**  
  `xgboost` (XGBClassifier, XGBRegressor),  
  `lightgbm` (LGBMClassifier, LGBMRegressor),  
  `catboost` (CatBoostClassifier),  
  `scikit-learn` (linear models, tree-based models, SVM, ensemble methods, naive bayes, dummy estimators)

- **Model Evaluation Metrics:**  
  Regression and classification metrics from `scikit-learn.metrics`

- **Visualization:**  
  `matplotlib`, `seaborn`

- **Statistical Analysis:**  
  `scipy.stats` (e.g., `pearsonr`, `f_oneway`),  
  `statsmodels` (variance inflation factor for multicollinearity analysis)

- **Logging and Utility:**  
  `logging`, `colorlog`, `datetime`, `os`

- **Custom Modules:**  
  Project-specific utilities and transformers developed with `scikit-learn` API (`BaseEstimator`, `TransformerMixin`)

---

This combination ensures a scalable, reproducible, and modular pipeline for robust churn analysis and prediction.

## Installation

Clone the repo and install the required packages:

```bash
git clone https://github.com/alexyepur/telco-churn-analysis-and-prediction.git
cd telco-churn-analysis-and-prediction
pip install -r requirements.txt


## License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.
