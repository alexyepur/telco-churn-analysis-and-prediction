# Basic
import numpy as np
import pandas as pd
import openpyxl

# Data Preparation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Modeling
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    GridSearchCV, 
    RandomizedSearchCV, 
    ParameterGrid
)

from sklearn.svm import (
    SVR, 
    SVC,
    LinearSVC
)

from sklearn.tree import (
    DecisionTreeRegressor, 
    DecisionTreeClassifier
)

from xgboost import (
    XGBRegressor, 
    XGBClassifier
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import (
    LinearRegression, 
    Lasso, 
    Ridge, 
    ElasticNet, 
    LogisticRegression,
    SGDClassifier
)
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier, 
    RandomForestRegressor, 
    RandomForestClassifier,
    ExtraTreesClassifier
)

from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.dummy import DummyRegressor, DummyClassifier

# Evaluation Metrics for Regression
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    explained_variance_score, 
    median_absolute_error, 
    max_error
)

# Evaluation Metrics for Classification
from sklearn.metrics import (
    accuracy_score, 
    precision_score,
    recall_score, 
    f1_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    roc_auc_score, 
    log_loss
)

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns



#from utility_functions.logger import get_logger
#logger = get_logger(__name__)


def create_basic_models(
    task: str,
    random_state: int=56
) -> dict[str, object]:
    """
    Returns a dictionary of basic ML models depending on the task type.
    
    Supports predefined model sets for regression, binary classification, and multiclass classification tasks.
    
    Args:
        task (str): Type of ML task, one of {'regression', 'binary_class', 'multi_class'}
        random_state (type, optional): Seed for the random number generator to ensure reproducibility. Defaults to 56.
    
    Returns:
        dict[str, object]: Dictionary of scikit-learn-compatible model instances with basic hyperparameters
    """
    
    if task not in ['regression', 'binary_class', 'multi_class']:
        raise ValueError(f"Task type must be on of: 'regression', 'binary_class', 'multi_class'. Got '{task}' instead")
    if not isinstance(random_state, int):
        raise TypeError("random_state must be 'int' type")

    if task == 'regression':
        return {
            'LinearRegression': LinearRegression(),
            'Lasso': Lasso(),
            'Ridge': Ridge(),
            'ElasticNet': ElasticNet(),
            'SVR': SVR(kernel='rbf', C=1.0),
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=random_state),
            'RandomForestRegressor': RandomForestRegressor(n_estimators=200, random_state=random_state),
            'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state),
            'XGBRegressor': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state)
        }
        
    elif task == 'binary_class':
        return {
            'LogisticRegression': LogisticRegression(random_state=random_state, class_weight='balanced', max_iter=1000),
            'SGDLogisticClassifier': SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, max_iter=1000, random_state=random_state, class_weight='balanced'),
            'DecisionTreeClassifier': DecisionTreeClassifier(criterion='entropy', random_state=random_state, class_weight='balanced'),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced'),
            'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
            'LinearSVC': LinearSVC(random_state=random_state, max_iter=1000, class_weight='balanced'),
            'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=100, random_state=random_state, class_weight='balanced'),
            'XGBClassifier': XGBClassifier(verbosity=0, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state),
            'LGBMClassifier': LGBMClassifier(verbosity=-1, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state)
        }

    elif task == 'multi_class':
        return {
            'LogisticRegression': LogisticRegression(random_state=random_state, class_weight='balanced', max_iter=1000),
            'SGDLogisticClassifier': SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, max_iter=1000, random_state=random_state, class_weight='balanced'),
            'DecisionTreeClassifier': DecisionTreeClassifier(criterion='entropy', random_state=random_state, class_weight='balanced'),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced'),
            'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
            'LinearSVC': LinearSVC(random_state=random_state, max_iter=1000, class_weight='balanced'),
            'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=100, random_state=random_state, class_weight='balanced'),
            'XGBClassifier': XGBClassifier(verbosity=0, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state),
            'LGBMClassifier': LGBMClassifier(verbosity=-1, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state),
        }


def evaluate_regression_models(
    X: pd.DataFrame, 
    y: pd.Series, 
    prep_pipeline_steps: list, 
    models: dict[str, object], 
    test_size: float=0.25, 
    random_state: int=56
) -> pd.DataFrame:
    """
    Evaluates multiple regression models using a shared preprocessing pipeline.

    For each model, a complete sklearn Pipeline is created by combining the provided preprocessing steps
    with the model as the final step. Each pipeline is trained on a portion of the data and evaluated
    on a separate test set not seen during training.
    
    Args:
        X (pd.DataFrame): Features matrix
        y (pd.Series): Target vector
        prep_pipeline_steps (list): List of preprocessing steps in (str, transformer) format, compatible with sklearn Pipeline
        models (dict[str, object]): Dictionary of estimators with model names as keys
        test_size (float, optional): Proportion of the dataset to include in the test split. Must be between 0 and 1. Defaults to 0.25
        random_state (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to 56
    
    Returns:
        pd.DataFrame: DataFrame with evaluation metrics for each model.
    """
    
    # validation
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a pandas DataFrame')
    if not isinstance(y, pd.Series):
        raise TypeError('y must be a pandas Series')
    if not isinstance(prep_pipeline_steps, list) or not all(isinstance(step, tuple) and len(step) == 2 for step in prep_pipeline_steps):
        raise TypeError("prep_pipeline_steps must be a list of (str, transformer) tuples")

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state) 

    # calculations
    results = []
    for mod_name, model in models.items():
        steps = prep_pipeline_steps + [('estimator', model)]
        pipe = Pipeline(steps=steps)
        pipe.fit(X_train, y_train)
        
        yhat = pipe.predict(X_test)
        mse = np.round(mean_squared_error(y_test, yhat), 2)
        r2 = np.round(r2_score(y_test, yhat), 2)
        mae = np.round(mean_absolute_error(y_test, yhat), 2)
        rmse = np.round(np.sqrt(mse), 2)
        med_ae = np.round(median_absolute_error(y_test, yhat), 2)
        ex_var = np.round(explained_variance_score(y_test, yhat), 2)
        max_err = np.round(max_error(y_test, yhat), 2)
        results.append(
            {
                'Model':mod_name,
                'R2_Score':r2,
                'Explained_Variance':ex_var,
                'MSE':mse,
                'Median_AE':med_ae,
                'MAE':mae,
                'RMSE':rmse,
                'Max_Error':max_err,
            }
        )
    result_df = pd.DataFrame(results).sort_values('R2_Score', ascending=False).reset_index(drop=True)
    return result_df


def evaluate_final_regression_models(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series, 
    prep_pipeline_steps: list, 
    models: dict[str, object], 
) -> pd.DataFrame:
    """
    Evaluates multiple regression models using a shared preprocessing pipeline.

    For each model, a complete sklearn Pipeline is created by combining the provided preprocessing steps
    with the model as the final step. Each pipeline is trained on a portion of the data and evaluated
    on a separate test set not seen during training. 
    It is designed to test final models (e.g., after hyperparameter tuning) on a dedicated,
    previously unseen test set that was not used during training or validation.
    However, it is not limited to such usage and can be applied more broadly.
    
    Args:
        X_train (pd.DataFrame): Features matrix for the training set
        X_test (pd.DataFrame): Features matrix for the test set (unseen by the model on previous steps)
        y_train (pd.Series): Target vector for the training set
        y_test (pd.Series): Target vector for the test set (unseen by the model on previous steps)
        prep_pipeline_steps (list): List of preprocessing steps in (str, transformer) format, compatible with sklearn Pipeline
        models (dict[str, object]): Dictionary of chosen estimators with model names as keys.
    
    Returns:
        pd.DataFrame: DataFrame with evaluation metrics for each model.
    """
    
    # validation
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError('X_train must be a pandas DataFrame')
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError('X_test must be a pandas DataFrame')
    if not isinstance(y_train, pd.Series):
        raise TypeError('y_train must be a pandas Series')
    if not isinstance(y_test, pd.Series):
        raise TypeError('y_test must be a pandas Series')
    if not isinstance(prep_pipeline_steps, list) or not all(isinstance(step, tuple) and len(step) == 2 for step in prep_pipeline_steps):
        raise TypeError("prep_pipeline_steps must be a list of (str, transformer) tuples")

    # calculations
    results = []
    for mod_name, model in models.items():
        steps = prep_pipeline_steps + [('estimator', model)]
        pipe = Pipeline(steps=steps)
        pipe.fit(X_train, y_train)
        
        yhat = pipe.predict(X_test)
        mse = np.round(mean_squared_error(y_test, yhat), 4)
        r2 = np.round(r2_score(y_test, yhat), 4)
        mae = np.round(mean_absolute_error(y_test, yhat), 4)
        rmse = np.round(np.sqrt(mse), 4)
        med_ae = np.round(median_absolute_error(y_test, yhat), 4)
        ex_var = np.round(explained_variance_score(y_test, yhat), 4)
        max_err = np.round(max_error(y_test, yhat), 4)
        results.append(
            {
                'Model':mod_name,
                'R2_Score':r2,
                'Explained_Variance':ex_var,
                'MSE':mse,
                'Median_AE':med_ae,
                'MAE':mae,
                'RMSE':rmse,
                'Max_Error':max_err,
            }
        )
    result_df = pd.DataFrame(results).sort_values('R2_Score', ascending=False).reset_index(drop=True)
    return result_df


def evaluate_binary_class_models(
    X: pd.DataFrame, 
    y: pd.Series, 
    prep_pipeline_steps: list[tuple[str, object]], 
    models: dict[str, object], 
    test_size: float = 0.25, 
    random_state: int = 100, 
    stratify_y: bool = False, 
    target_mapping: dict[str, str] = None
)-> pd.DataFrame:
    """
    Evaluates multiple classification models (binary) using a shared preprocessing pipeline.

    For each model, a complete sklearn Pipeline is created by combining the provided preprocessing steps
    with the model as the final step. Each pipeline is trained on a portion of the data and evaluated
    on a separate test set not seen during training.
    
    Args:
        X (pd.DataFrame): Features matrix
        y (pd.Series): Target vector
        prep_pipeline_steps (list): List of preprocessing steps in (str, transformer) format, compatible with sklearn Pipeline.
        models (dict[str, object]): Dictionary of estimators with model names as keys
        test_size (float, optional): Proportion of the dataset to include in the test split. Must be between 0 and 1. Defaults to 0.25
        random_state (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to 56
        stratify_y (bool, optional): Whether to stratify split by target. Defaults to False 
        target_mapping (dict[str, str], optional): Mapping of target classes to numerical labels. If not provided, LabelEncoder is used. Defaults to None
    
    Returns:
        pd.DataFrame: DataFrame with evaluation metrics for each model.
    """
    
    # validation
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a pandas DataFrame')
    if not isinstance(y, pd.Series):
        raise TypeError('y must be a pandas Series')
    if not isinstance(prep_pipeline_steps, list) or not all(isinstance(step, tuple) and len(step) == 2 for step in prep_pipeline_steps):
        raise TypeError("prep_pipeline_steps must be a list of (str, transformer) tuples")
    if not isinstance(models, dict):
        raise TypeError('models must be a dict')
    if not isinstance(test_size, float):
        raise TypeError('test_size must be float')
    if not isinstance(random_state, int):
        raise TypeError('random_state must be int')
        

    if y.nunique() != 2:
        raise ValueError(f"Chosen target has {y.nunique()} unique values, when 2 expected.")
    
    if set(y.unique()) != set([1,0]):
        if target_mapping:
            y = y.map(target_mapping)
        else:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            #logger.info(f"Target was label-encoded: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # train test split
    if stratify_y:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y) 
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state) 

    # calculations
    results = []
    for mod_name, model in models.items():
        steps = prep_pipeline_steps + [('estimator', model)]
        pipe = Pipeline(steps=steps)
        pipe.fit(X_train, y_train)
        
        yhat = pipe.predict(X_test)
        
        yhat_prob = None
        if hasattr(pipe, 'predict_proba'):
             yhat_prob = pipe.predict_proba(X_test)[:,1]
        
        acc = accuracy_score(y_test, yhat)
        prec = precision_score(y_test, yhat, zero_division=0)
        rec = recall_score(y_test, yhat, zero_division=0)
        f1 = f1_score(y_test, yhat, zero_division=0)
        roc_auc = roc_auc_score(y_test, yhat_prob) if yhat_prob is not None else None
        ll_score = log_loss(y_test, yhat_prob) if yhat_prob is not None else None
        results.append(
            {
                'Model':mod_name,
                'F1 Score': round(f1,  3),
                'Accuracy': round(acc, 3),
                'Precision': round(prec, 3),
                'Recall':round(rec, 3),
                'ROC AUC': round(roc_auc, 3) if roc_auc is not None else None,
                'Log Loss' : round(ll_score, 3) if ll_score is not None else None
            }
        )
    result_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False).reset_index(drop=True)
    return result_df


def evaluate_final_binary_class_models(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series, 
    prep_pipeline_steps: list[tuple[str, object]], 
    models: dict[str, object], 
    target_mapping: dict[str, str] = None
)-> pd.DataFrame:
    """
    Evaluates multiple classification models (binary) using a shared preprocessing pipeline.

    For each model, a complete sklearn Pipeline is created by combining the provided preprocessing steps
    with the model as the final step. Each pipeline is trained on a portion of the data and evaluated
    on a separate test set not seen during training.
    It is designed to test final models (e.g., after hyperparameter tuning) on a dedicated,
    previously unseen test set that was not used during training or validation.
    However, it is not limited to such usage and can be applied more broadly.
    
    Args:
        X_train (pd.DataFrame): Features matrix for the training set
        X_test (pd.DataFrame): Features matrix for the test set (unseen by the model on previous steps)
        y_train (pd.Series): Target vector for the training set
        y_test (pd.Series): Target vector for the test set (unseen by the model on previous steps)
        prep_pipeline_steps (list): List of preprocessing steps in (str, transformer) format, compatible with sklearn Pipeline.
        models (dict[str, object]): Dictionary of chosen estimators with model names as keys.
        target_mapping (dict[str, str], optional): Mapping of target classes to numerical labels. If not provided, LabelEncoder is used. Defaults to None
    
    Returns:
        pd.DataFrame: DataFrame with evaluation metrics for each model.
    """
    
    # validation
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError('X_train must be a pandas DataFrame')
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError('X_test must be a pandas DataFrame')
    if not isinstance(y_train, pd.Series):
        raise TypeError('y_train must be a pandas Series')
    if not isinstance(y_test, pd.Series):
        raise TypeError('y_test must be a pandas Series')
    if not isinstance(prep_pipeline_steps, list) or not all(isinstance(step, tuple) and len(step) == 2 for step in prep_pipeline_steps):
        raise TypeError("prep_pipeline_steps must be a list of (str, transformer) tuples")
    if not isinstance(models, dict):
        raise TypeError('models must be a dict')

    if y_train.nunique() != 2:
        raise ValueError(f"Chosen target has {y_train.nunique()} unique values, when 2 expected.")
    if y_test.nunique() != 2:
        raise ValueError(f"Chosen target has {y_test.nunique()} unique values, when 2 expected.")

    if set(y_train.unique()) != set([1,0]):
        if target_mapping:
            y_train_encoded = y_train.map(target_mapping)
        else:
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
    else:
        y_train_encoded = y_train.copy()

    if set(y_test.unique()) != set([1,0]):
        if target_mapping:
            y_test_encoded = y_test.map(target_mapping)
        else:
            label_encoder = LabelEncoder()
            y_test_encoded = label_encoder.fit_transform(y_test)
    else:
        y_test_encoded = y_test.copy()

    # calculations
    results = []
    for mod_name, model in models.items():
        steps = prep_pipeline_steps + [('estimator', model)]
        pipe = Pipeline(steps=steps)
        pipe.fit(X_train, y_train_encoded)
        
        yhat = pipe.predict(X_test)
        
        yhat_prob = None
        if hasattr(pipe, 'predict_proba'):
             yhat_prob = pipe.predict_proba(X_test)[:,1]
        
        acc = accuracy_score(y_test_encoded, yhat)
        prec = precision_score(y_test_encoded, yhat, zero_division=0)
        rec = recall_score(y_test_encoded, yhat, zero_division=0)
        f1 = f1_score(y_test_encoded, yhat, zero_division=0)
        roc_auc = roc_auc_score(y_test_encoded, yhat_prob) if yhat_prob is not None else None
        ll_score = log_loss(y_test_encoded, yhat_prob) if yhat_prob is not None else None
        results.append(
            {
                'Model':mod_name,
                'F1 Score': round(f1,  3),
                'Accuracy': round(acc, 3),
                'Precision': round(prec, 3),
                'Recall':round(rec, 3),
                'ROC AUC': round(roc_auc, 3) if roc_auc is not None else None,
                'Log Loss' : round(ll_score, 3) if ll_score is not None else None
            }
        )
    result_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False).reset_index(drop=True)
    return result_df


def evaluate_multi_class_models(
    X: pd.DataFrame, 
    y: pd.Series, 
    prep_pipeline_steps: list[tuple[str, object]], 
    models: dict[str, object], 
    test_size: float = 0.25, 
    random_state: int = 100, 
    stratify_y: bool = False, 
    average_metrics: str='weighted',
    target_mapping: dict[str, str] = None
)-> pd.DataFrame:
    """
    Evaluates multiple classification models (non-binary) using a shared preprocessing pipeline.

    For each model, a complete sklearn Pipeline is created by combining the provided preprocessing steps
    with the model as the final step. Each pipeline is trained on a portion of the data and evaluated
    on a separate test set not seen during training.
    
    Args:
        X (pd.DataFrame): Features matrix
        y (pd.Series): Target vector
        prep_pipeline_steps (list): List of preprocessing steps in (str, transformer) format, compatible with sklearn Pipeline.
        models (dict[str, object]): Dictionary of estimators with model names as keys
        test_size (float, optional): Proportion of the dataset to include in the test split. Must be between 0 and 1. Defaults to 0.25
        random_state (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to 56
        stratify_y (bool, optional): Whether to stratify split by target. Defaults to False 
        average_metrics (str, optional): Averaging strategy for multiclass metrics. Defaults to 'weighted'
        target_mapping (dict[str, str], optional): Mapping of target classes to numerical labels. If not provided, LabelEncoder is used.Defaults to None
    
    Returns:
        pd.DataFrame: DataFrame with evaluation metrics for each model.
    """
    
    # validatioт
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a pandas DataFrame')
    if not isinstance(y, pd.Series):
        raise TypeError('y must be a pandas Series')
    if not isinstance(prep_pipeline_steps, list) or not all(isinstance(step, tuple) and len(step) == 2 for step in prep_pipeline_steps):
        raise TypeError("prep_pipeline_steps must be a list of (str, transformer) tuples")
    if not isinstance(models, dict):
        raise TypeError('models must be a dict')
    if not isinstance(test_size, float):
        raise TypeError('test_size must be float')
    if not isinstance(random_state, int):
        raise TypeError('random_state must be int')
        
    # target encoding
    if target_mapping:
        y_encoded = y.map(target_mapping)
    else:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

    # train test split
    if stratify_y:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded) 
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state) 

    # calculations
    results = []
    for mod_name, model in models.items():
        steps = prep_pipeline_steps + [('estimator', model)]
        pipe = Pipeline(steps=steps)
        pipe.fit(X_train, y_train)
        
        yhat = pipe.predict(X_test)
        
        yhat_prob = None
        if hasattr(pipe, 'predict_proba') and callable(pipe.predict_proba):
             yhat_prob = pipe.predict_proba(X_test)
        
        acc = accuracy_score(y_test, yhat)
        prec = precision_score(y_test, yhat, average=average_metrics, zero_division=0)
        rec = recall_score(y_test, yhat, average=average_metrics, zero_division=0)
        f1 = f1_score(y_test, yhat, average=average_metrics, zero_division=0)

        if yhat_prob is not None:
            roc_auc = roc_auc_score(y_test, yhat_prob, multi_class='ovr', average=average_metrics)
            ll_score = log_loss(y_test, yhat_prob)
        else:
            roc_auc = None
            ll_score = None
            
        results.append(
            {
                'Model': mod_name,
                'F1 Score': round(f1,  3),
                'Accuracy': round(acc, 3),
                'Precision': round(prec, 3),
                'Recall': round(rec, 3),
                'ROC AUC': round(roc_auc, 3) if roc_auc is not None else None,
                'Log Loss': round(ll_score, 3) if ll_score is not None else None
            }
        )

    
    result_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False).reset_index(drop=True)
    return result_df


def evaluate_final_multi_class_models(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series, 
    prep_pipeline_steps: list[tuple[str, object]], 
    models: dict[str, object], 
    average_metrics: str='weighted',
    target_mapping: dict[str, str] = None
)-> pd.DataFrame:
    """
    Evaluates multiple classification models (non-binary) using a shared preprocessing pipeline.

    For each model, a complete sklearn Pipeline is created by combining the provided preprocessing steps
    with the model as the final step. Each pipeline is trained on a portion of the data and evaluated
    on a separate test set not seen during training.
    It is designed to test final models (e.g., after hyperparameter tuning) on a dedicated,
    previously unseen test set that was not used during training or validation.
    However, it is not limited to such usage and can be applied more broadly.
    
    Args:
        X_train (pd.DataFrame): Features matrix for the training set
        X_test (pd.DataFrame): Features matrix for the test set (unseen by the model on previous steps)
        y_train (pd.Series): Target vector for the training set
        y_test (pd.Series): Target vector for the test set (unseen by the model on previous steps)
        prep_pipeline_steps (list): List of preprocessing steps in (str, transformer) format, compatible with sklearn Pipeline
        models (dict[str, object]): Dictionary of chosen estimators with model names as keys
        average_metrics (str, optional): Averaging strategy for multiclass metrics. Defaults to 'weighted'
        target_mapping (dict[str, str], optional): Mapping of target classes to numerical labels. If not provided, LabelEncoder is used. Defaults to None
    
    Returns:
        pd.DataFrame: DataFrame with evaluation metrics for each model.
    """
    
    # validatioт
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError('X_train must be a pandas DataFrame')
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError('X_test must be a pandas DataFrame')
    if not isinstance(y_train, pd.Series):
        raise TypeError('y_train must be a pandas Series')
    if not isinstance(y_test, pd.Series):
        raise TypeError('y_test must be a pandas Series')
    if not isinstance(prep_pipeline_steps, list) or not all(isinstance(step, tuple) and len(step) == 2 for step in prep_pipeline_steps):
        raise TypeError("prep_pipeline_steps must be a list of (str, transformer) tuples")
    if not isinstance(models, dict):
        raise TypeError('models must be a dict')

    # target encoding
    if target_mapping:
        y_train_encoded = y_train.map(target_mapping)
        y_test_encoded = y_test.map(target_mapping)
    else:
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.fit_transform(y_test)

    # calculations
    results = []
    for mod_name, model in models.items():
        steps = prep_pipeline_steps + [('estimator', model)]
        pipe = Pipeline(steps=steps)
        pipe.fit(X_train, y_train_encoded)
        
        yhat = pipe.predict(X_test)
        
        yhat_prob = None
        if hasattr(pipe, 'predict_proba') and callable(pipe.predict_proba):
             yhat_prob = pipe.predict_proba(X_test)
        
        acc = accuracy_score(y_test_encoded, yhat)
        prec = precision_score(y_test_encoded, yhat, average=average_metrics, zero_division=0)
        rec = recall_score(y_test_encoded, yhat, average=average_metrics, zero_division=0)
        f1 = f1_score(y_test_encoded, yhat, average=average_metrics, zero_division=0)

        if yhat_prob is not None:
            roc_auc = roc_auc_score(y_test_encoded, yhat_prob, multi_class='ovr', average=average_metrics)
            ll_score = log_loss(y_test_encoded, yhat_prob)
        else:
            roc_auc = None
            ll_score = None
            
        results.append(
            {
                'Model': mod_name,
                'F1 Score': round(f1,  3),
                'Accuracy': round(acc, 3),
                'Precision': round(prec, 3),
                'Recall': round(rec, 3),
                'ROC AUC': round(roc_auc, 3) if roc_auc is not None else None,
                'Log Loss': round(ll_score, 3) if ll_score is not None else None
            }
        )

    
    result_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False).reset_index(drop=True)
    return result_df


def get_baseline_scores(
    task: str,
    X: pd.DataFrame, 
    y: pd.Series,
    random_state: int=100,
    test_size: float=0.25,
    stratify_y: bool=True
):
    """
    Compute baseline scores using simple dummy models for a given ML task.
    
    Fits dummy models (e.g., mean, median, most frequent) to establish reference performance for regression 
    or classification tasks.
    
    Args:
        task (str): Type of ML task, one of {'regression', 'binary_class', 'multi_class'}
        X (pd.DataFrame): Features matrix
        y (pd.Series): Target vector
        test_size (float, optional): Proportion of the dataset to include in the test split. Must be between 0 and 1. Defaults to 0.25
        random_state (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to 56
        stratify_y (bool): Whether to stratify split by target. Defaults to True
    
    Returns:
        pd.DataFrame: Evaluation metrics per dummy model.
    """
    
    # validation
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a pandas DataFrame')
    if not isinstance(y, pd.Series):
        raise TypeError('y must be a pandas Series')
    if not isinstance(test_size, float):
        raise TypeError('test_size must be float')
    if not isinstance(random_state, int):
        raise TypeError('random_state must be int')
    if not isinstance(stratify_y, bool):
        raise TypeError('stratify_y must be bool')
    if task not in ['regression', 'binary_class', 'multi_class']:
        raise ValueError(f"Task type must be on of: 'regression', 'binary_class', 'multi_class'. Got '{task}' instead")

    # calculations
    prep_pipeline_steps = []
    
    if task == 'regression':
        dummy_models = {
            'dummy_mean': DummyRegressor(strategy='mean'),
            'dummy_median': DummyRegressor(strategy='median'),
            'dummy_quantile_0.25': DummyRegressor(strategy='quantile', quantile=0.25)
        }
        
        dummy_res_df = evaluate_regression_models(
            X=X, 
            y=y, 
            prep_pipeline_steps=prep_pipeline_steps, 
            models=dummy_models)

    elif task == 'binary_class' or task == 'multi_class':
        dummy_models = {
            'dummy_most_frequent':DummyClassifier(strategy='most_frequent', random_state=random_state), 
            'dummy_stratified':DummyClassifier(strategy='stratified', random_state=random_state), 
            'dummy_uniform':DummyClassifier(strategy='uniform', random_state=random_state)
        }

        if task == 'binary_class':
            dummy_res_df = evaluate_binary_class_models(
                X=X, 
                y=y, 
                prep_pipeline_steps=prep_pipeline_steps, 
                models=dummy_models,
                stratify_y=stratify_y)

        else:
            dummy_res_df = evaluate_multi_class_models(
                X=X, 
                y=y, 
                prep_pipeline_steps=prep_pipeline_steps, 
                models=dummy_models)
    else:
        raise RuntimeError('Something went wrong')
        
    return dummy_res_df