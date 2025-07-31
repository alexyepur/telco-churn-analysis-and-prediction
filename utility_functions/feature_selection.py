import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, chi2
from scipy.stats import pearsonr, f_oneway

from utility_functions.preprocessing import classify_columns


#from utility_functions.logger import get_logger
#logger = get_logger(__name__)


def get_corr(
    data: pd.DataFrame, 
    num_features: list[str], 
    num_target: pd.Series, 
    corr_threshold: float=0.05, 
    p_threshold: float=0.05
):
    """
    Compute Pearson correlation and p-value for numerical features against a numerical target
    
    Filters out features with weak or statistically insignificant correlations based on given thresholds
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        num_features (list[str]): List of numeric feature names to evaluate
        num_target (pd.Series): Numeric target vector
        corr_threshold (float, optional): Minimum absolute correlation required to consider a feature as strong. Defaults to 0.05
        p_threshold (float, optional): Maximum p-value allowed for statistical significance. Defaults to 0.05

    Returns:
        tuple:
            pd.DataFrame: DataFrame with correlation and p-value for each feature
            list[str]: List of features considered weak (low correlation or high p-value)
    """
    
    results = []

    # validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')
    if not isinstance(num_target, pd.Series):
        raise TypeError('num_target must be pandas Series')    
    if not isinstance(corr_threshold, (float, int)):
        raise TypeError('corr_threshold must be float or int')
    if not isinstance(p_threshold, (float, int)):
        raise TypeError('p_threshold must be float or int')
    if not isinstance(num_features, list) or not all(isinstance(col, str) for col in num_features):
        raise TypeError('num_features must be a list of strings')
    
    missing_cols = [col for col in num_features if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in dataset: {missing_cols}")

    non_num_cols = [col for col in num_features if not pd.api.types.is_numeric_dtype(data[col])]
    if non_num_cols:
        raise TypeError(f"Columns are not numeric: {non_num_cols}")
    
    
    # calculation
    for col in num_features:
        corr, p_value = pearsonr(data[col], num_target)
        results.append({'feature': col, f'correlation(>{corr_threshold})': round(corr, 3), f'corr_p(<{p_threshold})':round(p_value, 3)})

    res_df = pd.DataFrame(results).sort_values(by=f'correlation(>{corr_threshold})', key=abs, ascending=False).reset_index(drop=True)
    
    weak_features_list = res_df[
        (res_df[f'correlation(>{corr_threshold})'].abs() < corr_threshold) | (res_df[f'corr_p(<{p_threshold})'] > p_threshold)
    ]['feature'].tolist()
    
    
    return res_df, weak_features_list


def get_mutual_info(
    data: pd.DataFrame, 
    target: pd.Series, 
    num_features: list[str], 
    mi_threshold: float=0.01,
    task_type: str='auto',  # 'auto', 'regression', 'classification'
    random_state: int=100,
    nunique_threshold: int=15
):
    """
    Compute mutual information scores between features and target
    
    Determines task type (classification or regression) and calculates mutual information 
    to evaluate feature relevance. Filters out features with low MI scores.
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        target (pd.Series): Target vector
        num_features (list[str]): List of numeric feature names to evaluate
        mi_threshold (float, optional): Minimum MI score required to keep feature. Defaults to 0.01
        task_type (str, optional): Task type, one of {'auto', 'regression', 'classification'}. Defaults to 'auto'
        random_state (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to 100
        nunique_threshold(int, optional): Threshold for unique values to distinguish classification vs regression in 'auto' mode. Defaults to 15
    Returns:
        tuple:
            pd.DataFrame: DataFrame with MI scores for each feature
            list[str]: List of features with MI score below threshold
    """
    
    # validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')
    if not isinstance(target, pd.Series):
        raise TypeError('target must be pandas Series')    
    if not isinstance(mi_threshold, (float, int)):
        raise TypeError('mi_threshold must be float or int')
    if not isinstance(random_state, int):
        raise TypeError('random_state must be int')
    if not isinstance(num_features, list) or not all(isinstance(col, str) for col in num_features):
        raise TypeError('num_features must be a list of strings')
    
    missing_cols = [col for col in num_features if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in dataset: {missing_cols}")

    non_num_cols = [col for col in num_features if not pd.api.types.is_numeric_dtype(data[col])]
    if non_num_cols:
        raise TypeError(f"Columns are not numeric: {non_num_cols}")
        
    # task determination
    n_unique = target.nunique()
    if task_type == 'regression':
        selected_task = 'regression'
        
    elif task_type == 'classification':
        if n_unique > nunique_threshold:
            raise ValueError(f"Target has {n_unique} unique values, which is more than nunique_threshold={nunique_threshold} for classification.")
        selected_task = 'classification'
    
    elif task_type == 'auto':
        if pd.api.types.is_numeric_dtype(target) and n_unique > nunique_threshold:
            selected_task = 'regression'
        else:
            selected_task = 'classification'
    else:
        raise ValueError("task_type must be 'auto', 'regression' or 'classification'")

    # calculation
    if selected_task == 'regression':
        mi_scores = np.round(mutual_info_regression(data[num_features], target, random_state=random_state), 3)
    elif selected_task == 'classification':
        mi_scores = np.round(mutual_info_classif(data[num_features], target, random_state=random_state), 3)

    res_df = pd.DataFrame({'feature':num_features,f'mi(>{mi_threshold})': mi_scores}
                         ).sort_values(by=f'mi(>{mi_threshold})', ascending=False).reset_index(drop=True)

    weak_features_list = res_df[res_df[f'mi(>{mi_threshold})'] < mi_threshold]['feature'].tolist()
    
    return res_df, weak_features_list


def get_anova_scores(
    data: pd.DataFrame,
    num_target: pd.Series,
    disc_features: list[str],
    p_threshold: float=0.05,
    nunique_threshold: int=15
):
    """
    Perform one-way ANOVA F-test between numerical target and discrete features.

    Evaluates how well each discrete feature separates the numerical target by calculating
    the ANOVA F-statistic and p-value for each feature. Also flags features with weak statistical
    association based on the provided p-value threshold.
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        num_target (pd.Series): Numeric target vector
        disc_features (list[str]): List of column names corresponding to discrete features
        p_threshold (float, optional): Maximum p-value allowed for statistical significance. Defaults to 0.05
        nunique_threshold (int, optional): Maximum number of unique values allowed in a feature. Defaults to 15
    
    Returns:
        tuple:
            pd.DataFrame: DataFrame with feature names, F-scores, and p-values sorted by F-score.
            list[str]: List of features that do not meet the p-value threshold (i.e., weak features).
    """
    
    # validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')
    if not isinstance(num_target, pd.Series):
        raise TypeError('num_target must be pandas Series')    
    if not isinstance(p_threshold, (float, int)):
        raise TypeError('p_threshold must be float or int')
    if not isinstance(disc_features, list) or not all(isinstance(col, str) for col in disc_features):
        raise TypeError('disc_features must be a list of strings')
    
    missing_cols = [col for col in disc_features if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in dataset: {missing_cols}")

    if not pd.api.types.is_numeric_dtype(num_target):
        raise TypeError('num_target must be numeric for ANOVA')

    high_card_cols = [col for col in disc_features if data[col].nunique() > nunique_threshold]
    if high_card_cols:
        raise ValueError(f"These features have too many unique values (> {nunique_threshold}): {high_card_cols}")

    # calculation
    results = []
    for col in disc_features:
        unique_categories = data[col].dropna().unique()
        groups = [num_target[data[col] == category] for category in unique_categories]

        # remove groups with <2 elements
        groups = [g for g in groups if len(g) > 1]

        # ANOVE expects at least 2 groups
        if len(groups) >= 2:
            try:
                f_score, p_value = f_oneway(*groups)
                results.append({
                    'feature': col,
                    'f_score': f_score,
                    f'p_anova(<{p_threshold})': round(p_value, 4)
                })
            except Exception as e:
                print(f"ANOVA failed for '{col}': {e}")
                results.append({
                    'feature': col,
                    'f_score': np.nan,
                    f'p_anova(<{p_threshold})': np.nan
                })
        else:
            # no enough groups
            results.append({
                'feature': col,
                'f_score': np.nan,
                f'p_anova(<{p_threshold})': np.nan
            })
        
    res_df = pd.DataFrame(results).sort_values(by='f_score', ascending=False).reset_index(drop=True)

    weak_features_list = res_df[res_df[f'p_anova(<{p_threshold})'] > p_threshold]['feature'].tolist()

    return res_df, weak_features_list


def get_chi2_scores(
    data: pd.DataFrame,
    cat_target: pd.Series,
    num_disc_features: list[str],
    chi2_threshold: float=1.0,
    p_threshold: float=0.05,
    nunique_threshold: int=15,
):
    """
    Computes chi2 scores and p-values between discrete numeric features and categorical target.

    Validates input, filters high-cardinality features, and returns weak features based on thresholds.
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        cat_target (pd.Series): Categorical target vector
        num_disc_features (list[str]):  List of columns names corresponding to discrete (numeric) features
        chi2_threshold (float, optional): Minimum chi2 score to keep feature. Defaults to 1.0
        p_threshold (float, optional): Maximum p-value allowed for statistical significance. Defaults to 0.05
        nunique_threshold (int, optional): Maximum number of unique values allowed in a feature. Defaults to 15
    
    Returns:
        tuple:
            pd.DataFrame: Table with chi² scores and p-values.
            list[str]: Features failing chi² or p-value thresholds.
    """
    
    # types validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')
    if not isinstance(cat_target, pd.Series):
        raise TypeError('cat_target must be pandas Series')    
    if not isinstance(chi2_threshold, (float, int)):
        raise TypeError('chi2_threshold must be float or int')
    if not isinstance(num_disc_features, list) or not all(isinstance(col, str) for col in num_disc_features):
        raise TypeError('num_disc_features must be a list of strings')
    if not pd.api.types.is_numeric_dtype(cat_target):
        raise TypeError('cat_target must be numerically encoded (e.g., 0/1/2)')
    
    # missing columns
    missing_cols = [col for col in num_disc_features if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in dataset: {missing_cols}")

    # numerical features check
    non_num_cols = [col for col in num_disc_features if not pd.api.types.is_numeric_dtype(data[col])]
    if non_num_cols:
        raise TypeError(f"Features must be numerically encoded for chi2: {non_num_cols}")


    # high cardinality check
    high_card_cols = [col for col in num_disc_features if data[col].nunique() > nunique_threshold]
    if high_card_cols:
        raise ValueError(f"These features have too many unique values (> {nunique_threshold}): {high_card_cols}")
    if cat_target.nunique() > nunique_threshold:
        raise ValueError(f"Target has too many unique values (> {nunique_threshold}), which may not be suitable for chi2.")

    # columns with negative values
    neg_cols = [col for col in num_disc_features if (data[col] < 0).any()]
    if neg_cols:
        raise ValueError(f"Columns contain negative values, which is invalid for chi² test: {neg_cols}")

    # calculation
    scores, p_values = chi2(data[num_disc_features], cat_target)
    
    res_df = pd.DataFrame({
        'feature': num_disc_features,
        f'chi2_score(>{chi2_threshold})':np.round(scores, 3),
        f'p_chi2(<{p_threshold})': np.round(p_values, 4)
    }).sort_values(by=f'chi2_score(>{chi2_threshold})', ascending=False).reset_index(drop=True)
    
    weak_chi2_list = res_df[res_df[f'chi2_score(>{chi2_threshold})'] < chi2_threshold]['feature'].tolist()
    weak_p_list = res_df[res_df[f'p_chi2(<{p_threshold})'] > p_threshold]['feature'].tolist()
    weak_features_list = list(set([*weak_chi2_list, *weak_p_list]))

    return res_df, weak_features_list


def feature_selection_orchestrator(
    data: pd.DataFrame,
    target: pd.Series,
    p_threshold: float=0.05,
    nunique_threshold: int=15,
    corr_threshold: float=0.05,
    mi_task_type: str='auto',
    mi_threshold: float=0.01,
    chi2_threshold: float=1.0,
    random_state: int=100,
):
    """
    Runs a set of statistical tests to evaluate feature relevance to the target.

    Applies mutual information, correlation, ANOVA, and chi2 tests depending on feature/target types.
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        target (pd.Series): Target vector
        p_threshold (float, optioanal): Maximum p-value allowed for statistical significance. Defaults to 0.05
        nunique_threshold (int, optional): Maximum number of unique values allowed in a feature. Defaults to 15
        corr_threshold (float, optional): Minimum absolute correlation required to consider a feature as strong. Defaults to 0.05
        mi_task_type (str, optional): Task type, one of {'auto', 'regression', 'classification'}. Defaults to 'auto'
        mi_threshold (float, optional): Minimum MI score required to keep feature. Defaults to 0.01
        chi2_threshold (float, optional): Minimum chi2 score to keep feature. Defaults to 1.0
        random_state (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to 100
    
    Returns:
        tuple:
            pd.DataFrame: Summary table with test results per feature.
            list[str]: Features that failed at least one test.
            list[str]: Features that failed all tests they participated in.
    """
    
    # types validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if not isinstance(target, pd.Series):
        raise TypeError("target must be a pandas Series")

    if not isinstance(p_threshold, (float, int)):
        raise TypeError("p_threshold must be float or int")
    if not 0 < p_threshold < 1:
        raise ValueError("p_threshold must be between 0 and 1")

    if not isinstance(nunique_threshold, int):
        raise TypeError("nunique_threshold must be int")
    if nunique_threshold <= 1:
        raise ValueError("nunique_threshold must be greater than 1")

    if not isinstance(corr_threshold, (float, int)):
        raise TypeError("corr_threshold must be float or int")
    if not 0 <= abs(corr_threshold) <= 1:
        raise ValueError("corr_threshold must be between -1 and 1")

    if not isinstance(mi_task_type, str):
        raise TypeError("mi_task_type must be a string")
    if mi_task_type not in {'auto', 'classification', 'regression'}:
        raise ValueError("mi_task_type must be one of {'auto', 'classification', 'regression'}")

    if not isinstance(mi_threshold, (float, int)):
        raise TypeError("mi_threshold must be float or int")
    if not 0 <= mi_threshold <= 1:
        raise ValueError("mi_threshold must be between 0 and 1")

    if not isinstance(chi2_threshold, (float, int)):
        raise TypeError("chi2_threshold must be float or int")
    if chi2_threshold < 0:
        raise ValueError("chi2_threshold must be non-negative")

    if not isinstance(random_state, int):
        raise TypeError("random_state must be int")
        
    # calculation
    classified_features = classify_columns(data=data, nunique_threshold=nunique_threshold)

    bin_num_feat = classified_features['binary_num_cols']
    disc_num_feat = classified_features['discrete_num_cols']
    rest_num_feat = classified_features['num_cols']
    
    bin_cat_feat = classified_features['binary_cat_cols']
    disc_cat_feat = classified_features['discrete_cat_cols']
    
    target_type = target.dtype
    target_nunique = target.nunique()

    base_df = pd.DataFrame({'feature': data.columns})
    mi_df = pd.DataFrame()
    corr_df = pd.DataFrame()
    anova_df = pd.DataFrame()
    chi2_df = pd.DataFrame()
    weak_features = []
    very_weak_features = []
    tested_features = []
    
    mi_features = [*bin_num_feat, *disc_num_feat]
    if mi_features:
        mi_df, weak_mi_features = get_mutual_info(
            data=data, 
            target=target, 
            num_features=mi_features, 
            mi_threshold=mi_threshold, 
            task_type=mi_task_type, 
            random_state=random_state, 
            nunique_threshold=nunique_threshold)
        
        base_df = pd.merge(base_df, mi_df, on='feature', how='outer')
        tested_features.extend(mi_features)
        weak_features.extend(weak_mi_features)

    if pd.api.types.is_numeric_dtype(target):
        corr_features = [*bin_num_feat, *disc_num_feat, *rest_num_feat]
        
        if corr_features:
            corr_df, weak_corr_features = get_corr(
                data=data, 
                num_features=corr_features, 
                num_target=target, 
                corr_threshold=corr_threshold, 
                p_threshold=p_threshold)
            
            base_df = pd.merge(base_df, corr_df, on='feature', how='outer')
            tested_features.extend(corr_features)
            weak_features.extend(weak_corr_features)

        anova_features = [*disc_num_feat, *disc_cat_feat, *bin_num_feat, *bin_cat_feat]
        if anova_features:
            anova_df, weak_anova_features = get_anova_scores(
                data=data, 
                num_target=target, 
                disc_features=anova_features, 
                p_threshold=p_threshold, 
                nunique_threshold=nunique_threshold)
            
            base_df = pd.merge(base_df, anova_df, on='feature', how='outer')
            tested_features.extend(anova_features)
            weak_features.extend(weak_anova_features)

    if target_nunique <= nunique_threshold:
        chi2_features = [*bin_num_feat, *disc_num_feat]
        if chi2_features:
            chi2_df, weak_chi2_features = get_chi2_scores(
                data=data, 
                cat_target=target, 
                num_disc_features=chi2_features, 
                chi2_threshold=chi2_threshold, 
                p_threshold=p_threshold, 
                nunique_threshold=nunique_threshold)
            
            base_df = pd.merge(base_df, chi2_df, on='feature', how='outer')
            tested_features.extend(chi2_features)
            weak_features.extend(weak_chi2_features)

    

    base_df['tests_failed'] = base_df['feature'].apply(lambda x: weak_features.count(x))
    base_df['tests_participated'] = base_df['feature'].apply(lambda x: tested_features.count(x))
    base_df['all_passed'] = base_df['feature'].apply(lambda x: x not in weak_features)
    base_df['no_success'] = base_df['tests_failed'] == base_df['tests_participated']
    
    base_df = base_df.sort_values(by=['all_passed','no_success'], ascending=[False, True]).reset_index(drop=True)
    no_success_features = base_df[base_df['no_success'] == True]['feature'].values.tolist()
    
    return base_df, weak_features, no_success_features