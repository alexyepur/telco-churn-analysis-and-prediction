import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from copy import deepcopy
from sklearn.model_selection import KFold

from utility_transformers.transformers import FrequencyEncoder, ArrayToDFTransformer

#from utility_functions.logger import get_logger
#logger = get_logger(__name__)


def target_splitter(
    data: pd.DataFrame, 
    target_list: list[str]
) -> dict[str, dict[str, pd.Series]]:
    """
    Splits provided target columns into numeric and categorical based on data types.

    Validates input and returns a dictionary with separate groups for numeric and categorical targets.
    
    Args:
    data (pd.DataFrame): Full input dataset containing both features and target
    target_list (list[str]): List of target column names to be split.
    
    Returns:
        dict[str, dict[str, pd.Series]]: 
            Dictionary with two keys:
                - 'num_targets': mapping of numeric target names to Series.
                - 'cat_targets': mapping of categorical target names to Series.
    """
    
    if not isinstance(target_list, list):
        #logger.error("Invalid type of target_list. Split interrupted.")
        raise TypeError(f"'target_list' is expected to be list of strings. Got {type(target_list)} instead.")
    
    if not isinstance(data, pd.DataFrame):
        #logger.error("Invalid type of data. Split interrupted.")
        raise TypeError(f"'data' is expected to be pandas.DataFrame. Got {type(data)} instead.")

    if not all(isinstance(col, str) for col in target_list):
        #logger.error("All elements of target_list must be strings.")
        raise TypeError("All elements of target_list must be strings.")

    num_target_dict = {}
    cat_target_dict = {}
    
    for col in target_list:
        if col not in data.columns:
            #logger.error(f"Column '{col}' was not found in dataframe columns. Execution stopped.")
            raise ValueError(f"Column '{col}' was not found in dataframe columns. Execution stopped.")

        target = data[col]
        
        if pd.api.types.is_numeric_dtype(target.dtype):
            num_target_dict[f"{col}"] = target
        elif pd.api.types.is_object_dtype(target.dtype) or isinstance(target.dtype, pd.CategoricalDtype):
            cat_target_dict[f"{col}"] = target
        else:
            pass
            #logger.warning(f"Target '{col}' has unsupported dtype: {target.dtype}. Skipped.")

        #logger.info(f" Successfully separated target '{col}'. Shape: {target.shape}")

    #logger.info('=' * 40)

    final_dict = {'num_targets': num_target_dict, 'cat_targets': cat_target_dict}
    return final_dict


def classify_columns(
    data : pd.DataFrame, 
    nunique_threshold : int =10
) -> dict[str, list[str]]:
    """
    Classifies columns in a DataFrame into detailed data type and cardinality-based groups.

    Columns are categorized into:
    - Boolean
    - Binary numeric
    - Binary categorical
    - Discrete numeric (numeric with limited unique values)
    - Discrete categorical (categorical with limited unique values)
    - Continuous numeric
    - High-cardinality categorical

    The classification is based on both data type and number of unique values
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target 
        nunique_threshold (int, optional): Upper limit of unique values to classify a feature as discrete. Defaults to 10
    
    Returns:
        dict[str, list[str]]: Dictionary containing lists of column names by category:
            - 'bool_cols': Boolean columns
            - 'binary_num_cols': Numeric columns with 2 unique values
            - 'binary_cat_cols': Object/Categorical columns with 2 unique values
            - 'discrete_num_cols': Numeric columns with unique values in (2, threshold]
            - 'discrete_cat_cols': Categorical columns with unique values in (2, threshold]
            - 'num_cols': Numeric columns with unique values > threshold
            - 'cat_cols': Categorical columns with unique values > threshold
    """
    bool_cols = []
    
    binary_num_cols = []
    binary_cat_cols = []
    
    discrete_num_cols = []
    discrete_cat_cols = []
    
    num_cols = []
    cat_cols = []

    for col in data.columns:
        n_unique = data[col].nunique(dropna=False)
        d_type = data[col].dtype

        if pd.api.types.is_bool_dtype(d_type):
            bool_cols.append(col)
            
        elif pd.api.types.is_numeric_dtype(d_type):
            if n_unique == 2:
                binary_num_cols.append(col)
            elif (n_unique > 2) and (n_unique <= nunique_threshold):
                discrete_num_cols.append(col)
            elif n_unique > nunique_threshold:
                num_cols.append(col)

        elif pd.api.types.is_object_dtype(d_type) or pd.api.types.is_categorical_dtype(d_type):
            if n_unique == 2:
                binary_cat_cols.append(col)
            elif (n_unique > 2) and (n_unique <= nunique_threshold):
                discrete_cat_cols.append(col)
            elif n_unique > nunique_threshold:
                cat_cols.append(col)
    return {
        "bool_cols": bool_cols,
        "binary_num_cols": binary_num_cols,
        "binary_cat_cols": binary_cat_cols,
        "discrete_num_cols": discrete_num_cols,
        "discrete_cat_cols": discrete_cat_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


def set_encoding_col_transformer(
    data: pd.DataFrame,
    previous_pipeline_steps: list[tuple],
    ordinal_mapping: list[list]=None, 
    ordinal_columns: list[str]=None,
    nunique_threshold: int=15,
    verbose_feature_names_out: bool=True
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that encodes categorical features based on their type and cardinality.

    This function performs a temporary fit_transform using the provided preprocessing steps to analyze
    the transformed feature set. Based on the number of unique values per feature, it applies:
    - OrdinalEncoder to user-defined ordinal columns (with a provided ordering),
    - OneHotEncoder to nominal and binary categorical features,
    - FrequencyEncoder to high-cardinality categorical features.
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        previous_pipeline_steps (list[tuple]): List of preprocessing steps in (str, transformer) format, compatible with sklearn Pipeline
        ordinal_mapping (list[list], optional): List of category orders for each ordinal column. Defaults to None
        ordinal_columns (list[str], optional): List of column names corresponding to ordinal_mapping. Defaults to None
        nunique_threshold (int, optional): Upper limit of unique values to classify a feature as discrete. Defaults to 15
        verbose_feature_names_out (bool, optional): If True, transformed features will have prefixed names. Defaults to True
    
    Returns:
        ColumnTransformer: ColumnTransformer with appropriate encoders based on column types.
    """
    
    # validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')
    if not isinstance(previous_pipeline_steps, list) or not all(isinstance(step, tuple) for step in previous_pipeline_steps):
        raise TypeError('previous_pipeline_steps must be a list of tuples')
    if not isinstance(nunique_threshold, int):
        raise TypeError('nunique_threshold must be int')

    
    ordinal_columns = ordinal_columns if ordinal_columns else []
    ordinal_mapping = ordinal_mapping if ordinal_mapping else []

                
    if not isinstance(ordinal_columns, list) or not all(isinstance(col, str) for col in ordinal_columns):
        raise TypeError('ordinal_columns must be a list of strings')
            
    if not isinstance(ordinal_mapping, list):
        raise TypeError('ordinal_mapping must be a list of lists of strings')
    for single_map in ordinal_mapping:
        if not isinstance(single_map, list):
            raise TypeError('ordinal_mapping must be a list of lists of strings')
        for cat in single_map:
            if not isinstance(cat, str):
                raise TypeError('ordinal_mapping must be a list of lists of strings')
    
    if len(ordinal_columns) != len(ordinal_mapping):
        raise ValueError(f'Length mismatch: ordinal_columns has {len(ordinal_columns)} elements, '
                     f'but ordinal_mapping has {len(ordinal_mapping)} elements')


    # Create a Pipeline from previous steps and apply fit_transform
    # WARNING: This fit_transform is performed solely for temporary analysis of the data structure,
    # to obtain the current features after prior transformations.
    # The fitted state of this Pipeline is NOT preserved and NOT used in the final training.
    # The final Pipeline will be fitted separately on the training data.
    ct_set_pipeline = Pipeline(steps=previous_pipeline_steps)
    data_transformed = ct_set_pipeline.fit_transform(data)
    
    col_segm = classify_columns(data=data_transformed, nunique_threshold=nunique_threshold)
    columns_transformer_steps = []

    
    # ordinal categorical columns 
    if ordinal_columns:
        columns_transformer_steps.append(
            ('ordinal_encoder', OrdinalEncoder(categories=ordinal_mapping, handle_unknown='use_encoded_value', unknown_value=-1), ordinal_columns)
        )

    
    # nominal discrete categorical columns + binary categorical columns 
    nominal_cat_cols = [col for col in col_segm['discrete_cat_cols'] 
                        if col not in ordinal_columns] + col_segm['binary_cat_cols']
    if nominal_cat_cols:
        columns_transformer_steps.append(('one_hot_encoder', OneHotEncoder(drop='first'), nominal_cat_cols))

    
    # high cardinality categorical columns
    high_card_cat_cols = [col for col in col_segm['cat_cols']
                          if (col not in ordinal_columns) 
                          and (col not in nominal_cat_cols)]
    if high_card_cat_cols:
        columns_transformer_steps.append(('frequency_encoder', FrequencyEncoder(cat_cols=high_card_cat_cols), high_card_cat_cols))

    # final assembly
    ct = ColumnTransformer(
        transformers=columns_transformer_steps, 
        remainder='passthrough', 
        verbose_feature_names_out=verbose_feature_names_out, 
        force_int_remainder_cols=False)

    return ct


def set_scaling_col_transformer(
    data: pd.DataFrame,
    previous_pipeline_steps: list[tuple],
    nunique_threshold: int=15, 
    verbose_feature_names_out: bool=True
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that scales numeric features after previous transformations.
    
    Designed to be used **after categorical encoding**, this function identifies numeric columns
    (including newly created ones from encoding) and applies `StandardScaler` to them.
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        previous_pipeline_steps (list[tuple]): List of preprocessing steps in (str, transformer) format, compatible with sklearn Pipeline
        nunique_threshold (int, optional): Upper limit of unique values to classify a feature as discrete. Defaults to 15
        verbose_feature_names_out (bool, optional): If True, transformed features will have prefixed names. Defaults to True
    
    Returns:
        ColumnTransformer: A transformer that applies `StandardScaler` to numeric features while passing through all others.
    """
    
    # validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')
    if not isinstance(nunique_threshold, int):
        raise TypeError('nunique_threshold must be int')
    if not isinstance(previous_pipeline_steps, list) or not all(isinstance(step, tuple) for step in previous_pipeline_steps):
        raise TypeError('previous_pipeline_steps must be a list of tuples')

    
    # Create a Pipeline from previous steps and apply fit_transform
    # WARNING: This fit_transform is performed solely for temporary analysis of the data structure,
    # to obtain the current features after prior transformations.
    # The fitted state of this Pipeline is NOT preserved and NOT used in the final training.
    # The final Pipeline will be fitted separately on the training data.
    ct_set_pipeline = Pipeline(steps=previous_pipeline_steps)
    data_transformed = ct_set_pipeline.fit_transform(data)
        
    col_seg = classify_columns(data=data_transformed, nunique_threshold=nunique_threshold)
    cols_to_scale = col_seg['discrete_num_cols'] + col_seg['num_cols']

    if cols_to_scale:
        ct = ColumnTransformer(
            transformers=[('scaler', StandardScaler(), cols_to_scale)], 
            remainder='passthrough',
            verbose_feature_names_out = verbose_feature_names_out,
            force_int_remainder_cols=False
        )
        return ct
    else: 
        raise ValueError('No columns to scale found in the dataset')


def set_preprocessing_pipeline_steps(
    data: pd.DataFrame,
    previous_pipeline_steps: list[tuple],
    nunique_threshold: int=15,
    verbose_feature_names_out: bool=True,
    ordinal_columns: list[str]=None,
    ordinal_mapping: list[list[str]]=None,
) -> list[tuple[str, object]]:
    """
    Builds a complete preprocessing pipeline by adding encoding and scaling steps sequentially.
    
    This function takes previously defined pipeline steps and appends:
    1. A column transformer for encoding categorical features,
    2. A column transformer for scaling numeric features (including those generated during encoding).
    
    It ensures proper transformation ordering and handles intermediate state internally.
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        previous_pipeline_steps (list[tuple]): List of preprocessing steps in (str, transformer) format, compatible with sklearn Pipeline
        nunique_threshold (int, optional): Upper limit of unique values to classify a feature as discrete. Defaults to 15
        verbose_feature_names_out (bool, optional): If True, transformed features will have prefixed names. Defaults to True
        ordinal_columns (list[str], optional): List of column names corresponding to ordinal_mapping. Defaults to None
        ordinal_mapping (list[list], optional): List of category orders for each ordinal column. Defaults to None
    
    Returns:
        list[tuple[str, object]]: Updated list of preprocessing steps including encoding and scaling transformers.
    """
    
    pipeline_steps = deepcopy(previous_pipeline_steps)

    # add encoding steps
    in_func_data_copy = data.copy()
    ct_encoder = set_encoding_col_transformer(
        data=in_func_data_copy,
        previous_pipeline_steps=pipeline_steps,
        ordinal_columns=ordinal_columns, 
        ordinal_mapping=ordinal_mapping, 
        verbose_feature_names_out=verbose_feature_names_out)

    pipeline_steps.append(('ct_encoder', ArrayToDFTransformer(ct_encoder)))
    

    # add scaling steps
    in_func_data_copy = data.copy()
    ct_scaler = set_scaling_col_transformer(
        data=in_func_data_copy, 
        nunique_threshold=nunique_threshold, 
        verbose_feature_names_out=verbose_feature_names_out, 
        previous_pipeline_steps=pipeline_steps)

    pipeline_steps.append(('ct_scaler', ArrayToDFTransformer(ct_scaler)))


    return pipeline_steps


def generate_oof_feature(
    X: pd.DataFrame, 
    y: pd.Series, 
    model: object, 
    prep_pipeline_steps: list[tuple[str, object]],
    n_splits: int=5, 
    random_state: int=100
) -> pd.Series:
    """
    Generates an out-of-fold (OOF) feature using cross-validated model predictions.
    
    This function trains a model on multiple train-validation splits of the input data and 
    aggregates predictions for each sample when it is part of the validation fold. The purpose 
    is to generate meta-features for stacking or model ensembling, ensuring no data leakage.
    
    Args:
        X (pd.DataFrame): Feature matrix 
        y (pd.Series): Target vector 
        model (object): Estimator with `fit()` and either `predict()` or `predict_proba()` methods. 
        prep_pipeline_steps (list[tuple[str, object]]): List of preprocessing steps in (str, transformer) format, compatible with sklearn Pipeline
        n_splits (int, optional): Number of KFold splits. Defaults to 5 
        random_state (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to 100
    
    Returns:
        pd.Series: Out-of-fold predictions aligned with original input index. 
    """
    
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be pandas DataFrame')
    if not isinstance(y, pd.Series):
        raise TypeError('y must be pandas Series')
    if not isinstance(prep_pipeline_steps, list):
        raise TypeError('prep_pipeline_steps must be list')
    if not (all(isinstance(step, tuple)) and len(step) == 2 for step in prep_pipeline_steps):
        raise ValueError('prep_pipeline_steps must be list of (str, transformer) tuples')
    if not hasattr(model, 'fit') and (not hasattr(model, 'predict') or not hasattr(model, 'predict_proba')):
        raise AttributeError('model must have fit(), predict()/predict_proba() methods')
        

    kf_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_generated = np.zeros(len(X))

    for train_idx, gen_idx in kf_splitter.split(X):
        X_train, X_gen = X.iloc[train_idx], X.iloc[gen_idx]
        y_train = y.iloc[train_idx]

        oof_pipeline_steps = prep_pipeline_steps + [('estimator', model)]
        oof_gen_pipeline = Pipeline(steps=oof_pipeline_steps)
        oof_gen_pipeline.fit(X_train, y_train)
        
        if hasattr(oof_gen_pipeline, "predict_proba"):
            probs = oof_gen_pipeline.predict_proba(X_gen)
            preds = np.argmax(probs, axis=1)
        else:
            preds = oof_gen_pipeline.predict(X_gen)
        
        oof_generated[gen_idx] = preds

    return pd.Series(oof_generated, index=X.index, name="oof_feature")