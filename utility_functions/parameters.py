import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV, 
    ParameterGrid
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDClassifier, LogisticRegression
from sklearn.svm import SVR, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier





# REGRESSION PARAMETERS
###########################################################################
# LinearRegression
###########################################################################
linear_reg_params = {
    'fit_intercept': True,
    'copy_X': True,
    'n_jobs': None,
    'positive': False,
}

linear_reg_params_grid = {}
###########################################################################
# Lasso
###########################################################################
lasso_reg_params = {
    'alpha': 1.0,
    'fit_intercept': True,
    'max_iter': 1000,
    'tol': 1e-4,
    'random_state': 100,
    'selection': 'cyclic',
}

lasso_reg_params_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'max_iter': [1000, 5000],
    'tol': [1e-4, 1e-3, 1e-2],
    'selection': ['cyclic', 'random'],
}

###########################################################################
# Ridge
###########################################################################
ridge_reg_params = {
    'alpha': 1.0,
    'fit_intercept': True,
    'max_iter': None,
    'tol': 1e-3,
    'solver': 'auto',
    'random_state': 100,
}

ridge_reg_params_grid = {
    'alpha': [0.1, 1, 10, 100, 200],
    'tol': [1e-4, 1e-3, 1e-2],
    'solver': ['auto', 'svd', 'cholesky', 'sparse_cg', 'sag', 'saga'],
}
###########################################################################
# ElasticNet 
###########################################################################
enet_reg_params = {
    'alpha': 1.0,
    'l1_ratio': 0.5,
    'fit_intercept': True,
    'max_iter': 1000,
    'tol': 1e-4,
    'selection': 'cyclic',
    'random_state': 100,
}

enet_reg_params_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
    'max_iter': [1000, 5000],
    'tol': [1e-4, 1e-3],
    'selection': ['cyclic', 'random'],
}
###########################################################################
# SVR 
###########################################################################
svr_reg_params = {
    'kernel': 'rbf',
    'degree': 3,
    'gamma': 'scale',
    'coef0': 0.0,
    'tol': 1e-3,
    'C': 1.0,
    'epsilon': 0.1,
    'shrinking': True,
    'cache_size': 200,
    'max_iter': -1,
}

svr_reg_params_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'epsilon': [0.01, 0.1, 0.2, 0.5, 1],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],
}
###########################################################################
# DecisionTreeRegressor
###########################################################################
dec_tree_reg_params = {
    'criterion': 'squared_error',
    'splitter': 'best',
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'min_weight_fraction_leaf': 0.0,
    'max_features': None,
    'random_state': 100,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
}

dec_tree_reg_params_grid = {
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': [None, 'auto', 'sqrt', 'log2'],
    'max_leaf_nodes': [None, 10, 20, 50, 100],
    'min_impurity_decrease': [0.0, 0.01, 0.1],
    'splitter': ['best', 'random'],
}
###########################################################################
# RandomForestRegressor
###########################################################################
rand_for_reg_params = {
    'n_estimators': 100,
    'criterion': 'squared_error',
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'min_weight_fraction_leaf': 0.0,
    'max_features': 'auto',
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'bootstrap': True,
    'oob_score': False,
    'n_jobs': -1,
    'random_state': 100,
    'verbose': 0,
    'warm_start': False,
}

rand_for_reg_params_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'bootstrap': [True, False],
    'max_leaf_nodes': [None, 20, 50, 100],
    'min_impurity_decrease': [0.0, 0.01, 0.1],
}
###########################################################################
# GradientBoostingRegression
###########################################################################
grad_boost_reg_params = {
    'loss': 'squared_error',
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 1.0,
    'criterion': 'friedman_mse',
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_depth': 3,
    'max_features': None,
    'random_state': 100,
    'validation_fraction': 0.1,
    'n_iter_no_change': None,
    'tol': 1e-4,
    'verbose': 0,
    'max_leaf_nodes': None,
    'warm_start': False,
}

grad_boost_reg_params_grid = {
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'subsample': [0.8, 1.0],
    'max_features': ['sqrt', None],
    'max_leaf_nodes': [None, 50],
}
###########################################################################
# XGBRegressor
###########################################################################
xg_boost_reg_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': 100,
    'verbosity': 1,
    'n_jobs': -1,
    'booster': 'gbtree',
    'tree_method': 'auto',
}
xg_boost_reg_params_grid = {
    'n_estimators': [100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.3],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1],
    'min_child_weight': [1, 3],
}


# BINARY CLASS PARAMETERS
###########################################################################
# LogisticRegression
###########################################################################
logreg_bin_class_params = {
    'random_state': 100,
    'solver': 'lbfgs',
    'max_iter': 1000,
    'n_jobs': -1,
    'fit_intercept': True,
    'class_weight': None,
}

logreg_bin_class_params_grid = {
    'solver': ['lbfgs', 'saga'],
    'penalty': ['l2'],
    'C': [0.01, 0.1, 1, 10],
    'max_iter': [5000], 
}
###########################################################################
# SGDLogisticClassifier
###########################################################################
sgd_log_bin_class_params = {
    'loss': 'log_loss',
    'penalty': 'l2',
    'alpha': 0.0001,
    'max_iter': 1000,
    'tol': 1e-3,
    'random_state': 100,
    'early_stopping': False,
    'n_jobs': -1,
    'learning_rate': 'optimal',
    'eta0': 0.0,
}

sgd_log_bin_class_params_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
    'max_iter': [1000, 2000],
    'tol': [1e-4, 1e-3],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'eta0': [0.001, 0.01, 0.1],
    'l1_ratio': [0, 0.5, 1],
    'early_stopping': [True, False],
    'class_weight': [None, 'balanced'],
}
###########################################################################
# DecisionTreeClassifier
###########################################################################
dec_tree_bin_class_params = {
    'criterion': 'gini',
    'splitter': 'best',
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'min_weight_fraction_leaf': 0.0,
    'max_features': None,
    'random_state': 100,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'class_weight': None,
}
dec_tree_bin_class_params_grid = {
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': [None, 'auto', 'sqrt', 'log2'],
    'max_leaf_nodes': [None, 10, 20, 50, 100],
    'min_impurity_decrease': [0.0, 0.01, 0.1],
    'splitter': ['best', 'random'],
    'criterion': ['gini', 'entropy'],
    'class_weight': [None, 'balanced'],
}
###########################################################################
# RandomForestClassifier
###########################################################################
rand_for_bin_class_params = {
    'random_state': 100,
    'n_jobs': -1,
}
rand_for_bin_class_params_grid = {
    'n_estimators': [100, 200, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced'],
}
###########################################################################
# KNeighborsClassifier
###########################################################################
kn_bin_class_params = {}
kn_bin_class_params_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
}
###########################################################################
# LinearSVC
###########################################################################
lin_svc_bin_class_params = {
    'random_state': 100,
    'dual': False,
}

lin_svc_bin_class_params_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'loss': ['squared_hinge'],
    'max_iter': [1000, 5000, 10000],
}
###########################################################################
# ExtraTreesClassifier
###########################################################################
ex_tree_bin_class_params = {
    'random_state': 100,
    'n_jobs': -1,
}
ex_tree_bin_class_params_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [False, True],
    'criterion': ['gini', 'entropy'],
}
###########################################################################
# XGBClassifier
###########################################################################
xgb_bin_class_params = {
    'objective': 'binary:logistic',
    'random_state': 100,
    'n_jobs': -1,
    'eval_metric': 'logloss',
}
xgb_bin_class_params_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 10],
}
###########################################################################
# LGBMClassifier
###########################################################################
lgbm_bin_class_params = {
    'objective': 'binary',
    'random_state': 42,
    'n_jobs': -1,
    'metric': 'binary_logloss',
}
lgbm_bin_class_params_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100],
    'max_depth': [5, 10, 20],
    'min_child_samples': [20, 50],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0], 
    'reg_alpha': [0, 0.1],
    'reg_lambda': [0, 0.1],
    'scale_pos_weight': [1]   
}


# MULTI CLASS PARAMETERS
###########################################################################
# LogisticRegression
###########################################################################
logreg_multi_class_params = {
    'random_state': 100,
    'solver': 'lbfgs',
    'multi_class': 'multinomial',
    'max_iter': 1000,
    'n_jobs': -1,
    'fit_intercept': True,
    'class_weight': None,
}

logreg_multi_class_params_grid = {
    'penalty': ['l2'],
    'C': [0.01, 0.1, 1, 10, 100],
    'class_weight': [None, 'balanced'],
    'fit_intercept': [True, False],
}
###########################################################################
# SGDLogisticClassifier
###########################################################################
sgd_log_multi_class_params = {
    'loss': 'log_loss',
    'penalty': 'l2',
    'max_iter': 1000,
    'tol': 1e-3,
    'random_state': 100,
    'early_stopping': False,
    'n_jobs': -1,
    'learning_rate': 'optimal',
    'eta0': 0.0,
    'class_weight': None,
    'multi_class': 'ovr',
}

sgd_log_multi_class_params_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
    'max_iter': [1000, 2000],
    'tol': [1e-4, 1e-3],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'eta0': [0.001, 0.01, 0.1],
    'l1_ratio': [0, 0.5, 1],
    'early_stopping': [True, False],
    'class_weight': [None, 'balanced'],
    'multi_class': ['ovr', 'multinomial'],
}
###########################################################################
# DecisionTreeClassifier
###########################################################################
dec_tree_multi_class_params = {
    'criterion': 'gini',
    'splitter': 'best',
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'min_weight_fraction_leaf': 0.0,
    'max_features': None,
    'random_state': 100,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'class_weight': None,
}

dec_tree_multi_class_params_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'max_leaf_nodes': [None, 10, 30, 50],
    'min_impurity_decrease': [0.0, 0.01, 0.1],
    'class_weight': [None, 'balanced'],
}
###########################################################################
# RandomForestClassifier 
###########################################################################
rand_for_multi_class_params = {
    'random_state': 100,
    'n_jobs': -1,
}
rand_for_multi_class_params_grid = {
    'n_estimators': [100, 200, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced'],
}
###########################################################################
# KNeighborsClassifier
###########################################################################
kn_multi_class_params = {}
kn_multi_class_params_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
}
###########################################################################
# LinearSVC
###########################################################################
lin_svc_multi_class_params = {
    'random_state': 100,
    'dual': False,
}
lin_svc_multi_class_params_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'loss': ['squared_hinge'],
    'max_iter': [1000, 5000, 10000],
    'multi_class': ['ovr', 'crammer_singer']
}
###########################################################################
# ExtraTreesClassifier
###########################################################################
ex_tree_multi_class_params = {
    'random_state': 100,
    'n_jobs': -1,
}
ex_tree_multi_class_params_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [False, True],
    'criterion': ['gini', 'entropy'],
}
###########################################################################
# XGBClassifier
###########################################################################
xgb_multi_class_params = {
    'objective': 'multi:softprob',
    'num_class': None,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
}
xgb_multi_class_params_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 10],
}
###########################################################################
# LGBMClassifier
###########################################################################
lgbm_multi_class_params = {
    'objective': 'multiclass',
    'random_state': 100,
    'n_jobs': -1,
    'metric': 'multi_logloss'
}
lgbm_multi_class_params_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100],
    'max_depth': [5, 10, 20],
    'min_child_samples': [20, 50],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0], 
    'reg_alpha': [0, 0.1],
    'reg_lambda': [0, 0.1],
    'scale_pos_weight': [1] 
}
###########################################################################
# Regression Params Dict
###########################################################################
regression_params_dict = {
    'LinearRegression': {
        'basic': linear_reg_params,
        'grid': linear_reg_params_grid
    },
    'Lasso': {
        'basic': lasso_reg_params,
        'grid': lasso_reg_params_grid
    },
    'Ridge': {
        'basic': ridge_reg_params,
        'grid': ridge_reg_params_grid
    },
    'ElasticNet': {
        'basic': enet_reg_params,
        'grid': enet_reg_params_grid
    },
    'SVR': {
        'basic': svr_reg_params,
        'grid': svr_reg_params_grid
    },
    'DecisionTreeRegressor': {
        'basic': dec_tree_reg_params,
        'grid': dec_tree_reg_params_grid
    },
    'RandomForestRegressor': {
        'basic': rand_for_reg_params,
        'grid': rand_for_reg_params_grid
    },
    'GradientBoostingRegressor': {
        'basic': grad_boost_reg_params,
        'grid': grad_boost_reg_params_grid
    },
    'XGBRegressor': {
        'basic': xg_boost_reg_params,
        'grid': xg_boost_reg_params_grid
    }
}
###########################################################################
# Binary Class Params Dict
###########################################################################
binary_class_params_dict = {
    'LogisticRegression': {
        'basic': logreg_bin_class_params,
        'grid': logreg_bin_class_params_grid
    },
    'SGDLogisticClassifier': {
        'basic': sgd_log_bin_class_params,
        'grid': sgd_log_bin_class_params_grid
    },
    'DecisionTreeClassifier': {
        'basic': dec_tree_bin_class_params,
        'grid': dec_tree_bin_class_params_grid
    },
    'RandomForestClassifier': {
        'basic': rand_for_bin_class_params,
        'grid': rand_for_bin_class_params_grid
    },
    'KNeighborsClassifier': {
        'basic': kn_bin_class_params,
        'grid': kn_bin_class_params_grid
    },
    'LinearSVC': {
        'basic': lin_svc_bin_class_params,
        'grid': lin_svc_bin_class_params_grid
    },
    'ExtraTreesClassifier': {
        'basic': ex_tree_bin_class_params,
        'grid': ex_tree_bin_class_params_grid
    },
    'XGBClassifier': {
        'basic': xgb_bin_class_params,
        'grid': xgb_bin_class_params_grid
    },
    'LGBMClassifier': {
        'basic': lgbm_bin_class_params,
        'grid': lgbm_bin_class_params_grid 
    }
}
###########################################################################
# Multi Class Params Dict
###########################################################################
multi_class_params_dict = {
    'LogisticRegression': {
        'basic': logreg_multi_class_params,
        'grid': logreg_multi_class_params_grid
    },
    'SGDLogisticClassifier': {
        'basic': sgd_log_multi_class_params,
        'grid': sgd_log_multi_class_params_grid
    },
    'DecisionTreeClassifier': {
        'basic': dec_tree_multi_class_params,
        'grid': dec_tree_multi_class_params_grid
    },
    'RandomForestClassifier': {
        'basic': rand_for_multi_class_params,
        'grid': rand_for_multi_class_params_grid
    },
    'KNeighborsClassifier': {
        'basic': kn_multi_class_params,
        'grid': kn_multi_class_params_grid
    },
    'LinearSVC': {
        'basic': lin_svc_multi_class_params,
        'grid': lin_svc_multi_class_params_grid
    },
    'ExtraTreesClassifier': {
        'basic': ex_tree_multi_class_params,
        'grid': ex_tree_multi_class_params_grid
    },
    'XGBClassifier': {
        'basic': xgb_multi_class_params,
        'grid': xgb_multi_class_params_grid
    },
    'LGBMClassifier': {
        'basic': lgbm_multi_class_params,
        'grid': lgbm_multi_class_params_grid
    }
}

###########################################################################
###########################################################################
###########################################################################
###########################################################################
# FINALIZATION
###########################################################################
###########################################################################
###########################################################################
###########################################################################
grid_search_parameters = {
    'regression':regression_params_dict,
    'binary_class': binary_class_params_dict,
    'multi_class': multi_class_params_dict
}

grid_search_models_dict = {
    'regression': {
        'LinearRegression': LinearRegression,
        'Lasso': Lasso,
        'Ridge': Ridge,
        'ElasticNet': ElasticNet,
        'SVR': SVR,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'RandomForestRegressor': RandomForestRegressor,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'XGBRegressor': XGBRegressor,
    },
    'binary_class': {
        'LogisticRegression': LogisticRegression,
        'SGDLogisticClassifier': SGDClassifier,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'KNeighborsClassifier': KNeighborsClassifier,
        'LinearSVC': LinearSVC,
        'ExtraTreesClassifier': ExtraTreesClassifier,
        'XGBClassifier': XGBClassifier,
        'LGBMClassifier': LGBMClassifier,
    },
    'multi_class': {
        'LogisticRegression': LogisticRegression,
        'SGDLogisticClassifier': SGDClassifier,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'KNeighborsClassifier': KNeighborsClassifier,
        'LinearSVC': LinearSVC,
        'ExtraTreesClassifier': ExtraTreesClassifier,
        'XGBClassifier': XGBClassifier,
        'LGBMClassifier': LGBMClassifier,
    },
}

###########################################################################
###########################################################################
###########################################################################
###########################################################################
def do_random_grid_search(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    prep_pipeline_steps: list[tuple[str, object]],
    models: list[str],
    random_state: int=100,
    n_iter_min: int=100,
    cv: int=5,
    test_size: float=0.25,
    stratify_y: bool=True
) -> pd.DataFrame:
    """
    Run randomized hyperparameter search for multiple models using sklearn pipeline.
    
    Fits each model with given preprocessing steps and evaluates with cross-validation.
    
    Args:
        X (pd.DataFrame): Features matrix
        y (pd.Series): Target vector
        task (str): Type of ML task, one of {'regression', 'binary_class', 'multi_class'}
        prep_pipeline_steps (list): List of preprocessing steps in (str, transformer) format, compatible with sklearn Pipeline.
        models (dict[str, object]): Dictionary of estimators with model names as keys
        random_state (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to 56
        n_iter_min (int, optional): Max allowed number of sampled parameter combinations in RandomizedSearchCV. Defaults to 100
        cv (int, optional): Number of cross-validation folds. Defaults to 5
        test_size (float, optional): Proportion of the dataset to include in the test split. Must be between 0 and 1. Defaults to 0.25
        stratify_y (bool, optional): Whether to stratify split by target. Defaults to False 
    Returns:
        pd.DataFrame: Sorted table with model name, best params, and CV score.
    """
    
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be pandas DataFrame')
    if not isinstance(y, pd.Series):
        raise TypeError('y must be pandas Series')
    if not isinstance(prep_pipeline_steps, list):
        raise TypeError('prep_pipeline_steps must be list')
    if not (all(isinstance(step, tuple)) and len(step) == 2 for step in prep_pipeline_steps):
        raise ValueError('prep_pipeline_steps must be list of (str, transformer) tuples')
    if not isinstance(models, list):
        raise TypeError('models must be a list')
    if not isinstance(test_size, float):
        raise TypeError('test_size must be float')
    if not isinstance(random_state, int):
        raise TypeError('random_state must be int')
    if not isinstance(cv, int):
        raise TypeError('cv must be int')
    if not isinstance(stratify_y, bool):
        raise TypeError('stratify_y must be bool')
    if task not in ['regression', 'binary_class', 'multi_class']:
        raise ValueError(f"Task type must be on of: 'regression', 'binary_class', 'multi_class'. Got '{task}' instead")

    # scoring_determination
    if task == 'regression':
        scoring = 'r2'
    elif task == 'binary_class':
        scoring = 'f1'
    elif task == 'multi_class':
        scoring = 'f1_weighted'
    else:
        raise ValueError(f"Unknown task: {task}")\

    # train test split
    if stratify_y:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    model_step_name = 'estimator'
    tested_models = []
    best_params = []
    best_score = []

    # main cycle
    for mod_name in models:
        model = grid_search_models_dict[task][mod_name]
        basic_params = grid_search_parameters[task][mod_name]['basic']
        param_grid = grid_search_parameters[task][mod_name]['grid']

        # setting of random seeds
        if 'random_state' in basic_params.keys():
            basic_params['random_state'] = random_state
        if 'random_seed' in basic_params.keys():
            basic_params['random_seed'] = random_state
    
        
        # adding model prefix for parameters
        updated_param_grid = {}
        for parameter in param_grid:
            updated_param_grid[f"{model_step_name}__{parameter}"] = param_grid[parameter]

        # steps assembly
        pipeline_steps = prep_pipeline_steps + [(model_step_name, model(**basic_params))]
        pipeline = Pipeline(steps=pipeline_steps)
        total_params_combinations = len(list(ParameterGrid(param_grid)))

        # grid
        grid = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=updated_param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=random_state,
            n_iter=min(n_iter_min, total_params_combinations)
        )
        
        grid.fit(X_train, y_train)

        # results
        tested_models.append(mod_name)
        best_params.append(grid.best_params_)
        best_score.append(grid.best_score_)
    
    res_df = pd.DataFrame({
        'model':tested_models,
        'best_params': best_params,
        scoring: best_score
    }).sort_values(scoring, ascending=False).reset_index(drop=True)
    
    return res_df

    
def assemble_models_after_grid_search(
    task: str,
    models_names_list: list[str],
    parametes_res_df: pd.DataFrame,
    random_state: int = 100
):
    """
    Assemble model instances with best parameters after grid search.

    Creates initialized models using their best hyperparameters found in grid search.
    
    Args:
        task (str): Type of ML task, one of {'regression', 'binary_class', 'multi_class'}
        models_names_list (list[str]): List of model names to assemble.
        parametes_res_df (pd.DataFrame): Output DataFrame from 'do_random_grid_search'
        random_state (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to 100
    
    Returns:
        dict: Dictionary where keys are model names and values are models initialized with best found parameters
    """
    
    if not isinstance(random_state, int):
        raise TypeError('random_state must be int')
    if not isinstance(models_names_list, list):
        raise TypeError(f"'models_names_list' must be a list")
    if not all(isinstance(mod_name, str) for mod_name in models_names_list):
        raise TypeError(f"All elements in 'models_names_list' must be str")
    if task not in ['regression', 'binary_class', 'multi_class']:
        raise ValueError(f"Task type must be on of: 'regression', 'binary_class', 'multi_class'. Got '{task}' instead")
    
    PREFIX = 'estimator__'
    
    models_dict = {}
    for mod_name in models_names_list:
    
        model = grid_search_models_dict[task][mod_name]
        params = grid_search_parameters[task][mod_name]['basic']
        best_params_raw = parametes_res_df[parametes_res_df['model'] == mod_name]['best_params'].values.tolist()[0]
        
        for param_name, param_value in best_params_raw.items():
            if param_name.startswith(PREFIX):
                param_name = param_name.removeprefix(PREFIX)
            params[param_name] = param_value
    
        if 'random_state' in params.keys():
            params['random_state'] = random_state
        if 'random_seed' in params.keys():
            params['random_seed'] = random_state
    
        models_dict[mod_name] = model(**params)
    
    return models_dict