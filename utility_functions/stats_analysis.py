# Basic
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def identify_outliers(
    data: pd.DataFrame, 
    features: list[str]
) -> set:
    """
    Identifies outliers in specified numeric features using the IQR method.
    
    For each given feature, outliers are defined as values outside the range 
    [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. The function returns a set of all unique 
    row indices where outliers were detected
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        features (list[str]): List of numeric columns names to check for outliers
    
    Returns:
        For each feature with detected outliers, prints the outlier indices, their values, 
        and the feature's median
    """
    
    outliers_set = set()
    for col in features:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5*IQR
        lower_bound = Q1 - 1.5*IQR
        outliers = data[(data[col]>upper_bound)|(data[col]<lower_bound)]
        outliers_set.update(i for i in outliers.index)
        if not outliers.empty:
            print(f"[IQR] {col.upper()} \ncontains outlier(s) with index(es):\n{outliers.index.values} with values of \n{outliers[col].values} \nand median value of {data[col].median()}\n" )
    if not outliers_set:
        print("No outliers detected using IQR.")

    return outliers_set


def evaluate_kurtosis(
    data: pd.DataFrame, 
    features: list[str], 
    detailed: bool=False
) -> list[str]:
    """
    Evaluates kurtosis for specified numeric features and reports high kurtosis values
    
    For each feature, computes the Fisher kurtosis. Features with kurtosis > 3 
    (indicating heavy tails compared to a normal distribution) are flagged and returned
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        features (list[str]): List of numeric columns names to check for kurtosis
        detailed (bool, optional): If True, prints kurtosis for all features, not just those with high values. Defaults to False
    
    Returns:
        list[str]: List of column names with kurtosis > 3
    """
    
    high_kurtosis_cols = []
    message=True
    
    for col in features:
        col_kurt = data[col].kurt()
        if (col_kurt > 3):
            print(f"!HIGH! kurtosis found on column {col.capitalize()} with value of {col_kurt:.2f}\n")
            message=False
            high_kurtosis_cols.append(col)
        elif detailed:
            print(f"NORMAL Kurtosis on column {col.capitalize()} with value of {col_kurt:.2f}\n")
            
    if message:
        print("Kurtosis of dataset is ok!")
        
    return high_kurtosis_cols

    
def generate_skew_instructions(
    data: pd.DataFrame, 
    features: list[str], 
    detailed: bool=False, 
    high_skew_transformer: str='log', 
    extra_high_skew_transformer: str='yeo-johnson'
) -> dict[str, str]:    
    """
    Analyzes skewness of given features and generates transformation instructions
    
    Prints skewness level per feature and returns a mapping of features requiring transformation
    based on skewness thresholds.
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        features (list[str]): List of numeric columns names to check for skewness
        detailed (bool, optional): If True, prints skewness for all features, not just those with high values. Defaults to False
        high_skew_transformer (str, optional): Transformation method to apply when absolute skewness is between 0.5 and 1.0. Defaults to 'log'
        extra_high_skew_transformer (str, optional): Transformation method to apply when absolute skewness exceeds 1.0. Defaults to 'yeo-johnson'
    
    Returns:
        dict[str, str]: Mapping of feature names to recommended transformation methods
    """
    
    transform_instructions = {}

    for col in features:
        col_skew = data[col].skew()
        
        if abs(col_skew) > 1:
            print(f"!EXTRA HIGH! Skewness found on column \"{col}\" with value of {col_skew:.2f}\n")
            transform_instructions[col] = extra_high_skew_transformer
        elif abs(col_skew) > 0.5:
            print(f"!HIGH! Skewness found on column \"{col}\" with value of {col_skew:.2f}\n")
            transform_instructions[col] = high_skew_transformer
        elif detailed:
            print(f"NORMAL Skewness on column \"{col}\" with value of {col_skew:.2f}\n")

    if not transform_instructions:
        print("Skewness of dataset is ok!")
    
    return transform_instructions

    
def transform_skewed_features(
    data: pd.DataFrame, 
    transform_instructions: dict[str, str]
) -> None:
    """
    Applies specified transformations to skewed features in a DataFrame
    
    For each feature in the provided instruction dictionary, applies the designated transformation
    (method must be one of: 'log', 'reciprocal', 'sqrt', 'boxcox', 'yeojohnson') if data constraints allow.
    Creates new columns with transformed values using the pattern: original_name_method
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        transform_instructions (dict[str, str]): Mapping of feature names to recommended transformation methods
    
    Returns:
        None
    """
    
    affected_columns = []
    for col, method in transform_instructions.items():
        
        if col not in data.columns:
            print(f"Column \"{col}\" was not found in dataframe.")
            continue

        if data[col].isnull().any():
            print(f"Column \"{col}\" has NaN values. Consider handling them before transformation.")
            
        if method not in ['log', 'reciprocal', 'sqrt', 'boxcox', 'yeojohnson']:
            print(f"Chosen method \"{method}\" is not supported")
            continue
        
        if method == 'log':
            if (data[col] <= -1).any():
                print(f"Column \"{col}\" contains non-positive values. Method \"{method}\" is not available. Choose another method.")
                continue
            data[f"{col}_{method}"] = np.log(data[col]+1)
            affected_columns.append(col)
        
        elif method == 'reciprocal':
            if (data[col] <= -1).any():
                print(f"Column \"{col}\" contains non-positive values. Method \"{method}\" is not available. Choose another method.")
                continue
            data[f"{col}_{method}"] = 1 / (data[col]+1)
            affected_columns.append(col)
            
        elif method == 'sqrt':
            if (data[col] < 0).any():
                print(f"Column \"{col}\" contains negative values. Method \"{method}\" is not available. Choose another method.")
                continue
            data[f"{col}_{method}"] = np.sqrt(data[col])
            affected_columns.append(col)
        
        elif method == 'boxcox':
            if (data[col] <= 0).any():
                print(f"Column \"{col}\" contains non-positive values. Method \"{method}\" is not available. Choose another method.")
                continue
            data[f"{col}_{method}"] = boxcox(data[col])[0]
            affected_columns.append(col)
        
        elif method == 'yeojohnson':
            pt = PowerTransformer(method='yeo-johnson')
            data[f"{col}_{method}"] = pt.fit_transform(data[[col]])
            affected_columns.append(col)

    if affected_columns:
        return print(f"\n\nSuccesfully transformed columns:{', '.join(affected_columns)}")
    else:
        return print("\n\nNone columns were affected.")


def compare_columns(
    data: pd.DataFrame, 
    features: list[str], 
    normalize: bool = False, 
    sort: bool = True
) -> None:
    """
    Displays value counts for a list of categorical features
    
    Prints the distribution of values for each specified column. Optionally normalizes counts to proportions
    and sorts the output by count
        
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        features (list[str]): List of columns names to be compared
        normalize (bool, optional): If True, displays relative frequencies instead of absolute counts. Defaults to False
        sort (bool, optional): If True, sorts values by frequency. Defaults to True
    
    Returns:
        None
    """
    print("Comparison of columns: \n",features)
    print('='*40)
    for col in features:
        print(data[col].value_counts(normalize=normalize, sort=sort, dropna=False))
        print()


def describe_disc_features(
    data: pd.DataFrame, 
    features: list[str], 
    show_counts: bool = True
) -> None:
    """
    Displays basic descriptive statistics for discrete (categorical) features
    
    Prints either value counts or a list of unique values for each feature. 
    Useful for quick inspection of categorical variables
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        features (list[str]): List of discrete columns names to be described
        show_counts (bool, optional): If True, prints value counts. If False, prints unique values only. Defaults to True
    
    Returns:
        None
    """
    print("Describing discrete categorical features:")
    print('=' * 40)
    for col in features:
        print(f"Column: '{col}'")
        if show_counts:
            print(data[col].value_counts(dropna=False))
        else:
            print("Values:", data[col].unique())
        print(f"{data[col].nunique()} unique values.")
        print('-' * 40)


def calculate_vif(
    data: pd.DataFrame
)-> pd.DataFrame:
    """
    Calculates Variance Inflation Factor (VIF) for a set of numeric features
    
    Performs multicollinearity diagnostics by computing VIF scores for each feature.
    Also classifies features by VIF severity and returns both a detailed report and a dictionary of grouped features
    
    Args:
        data (pd.DataFrame): DataFrame containing only numeric, non-null, non-infinite columns
    
    Returns:
        - pd.DataFrame: Table with features, VIF scores, and severity grades ('low', 'moderate', 'high', 'infinite')
        - dict: Feature groups by VIF severity:
            - 'low_vif_cols': VIF ≤ 2
            - 'moderate_vif_cols': 2 < VIF ≤ 10
            - 'high_vif_cols': VIF > 10
            - 'inf_vif_cols': VIF == np.inf
    """
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')

    if not all(np.issubdtype(dtype, np.number) for dtype in data.dtypes):
        raise TypeError('All columns must be numeric for VIF calculation')
    
    if data.isnull().any().any():
        raise ValueError('Input data contains NaN values')
    if np.isinf(data.values).any():
        raise ValueError('Input data contains infinite values')
        
    vif_report = pd.DataFrame()
    vif_report['feature'] = data.columns

    vif_report['vif_score'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

    conditions = [
        (vif_report['vif_score'] <= 2),
        (vif_report['vif_score'] <= 10) & ((vif_report['vif_score'] > 2)),
        (vif_report['vif_score'] < np.inf) & ((vif_report['vif_score'] > 10)),
        (vif_report['vif_score'] == np.inf)
    ]
    choices = ['low', 'moderate', 'high', 'infinite']
    vif_report['vif_grade'] = np.select(condlist=conditions, choicelist=choices, default = 'None')

    low_vif_cols = vif_report[vif_report['vif_grade'] == 'low']['feature'].values.tolist()
    moderate_vif_cols = vif_report[vif_report['vif_grade'] == 'moderate']['feature'].values.tolist()
    high_vif_cols = vif_report[vif_report['vif_grade'] == 'high']['feature'].values.tolist()
    inf_vif_cols = vif_report[vif_report['vif_grade'] == 'infinite']['feature'].values.tolist()

    vif_report_dict = {
        'low_vif_cols': vif_report[vif_report['vif_grade'] == 'low']['feature'].values.tolist(),
        'moderate_vif_cols': vif_report[vif_report['vif_grade'] == 'moderate']['feature'].values.tolist(),
        'high_vif_cols': vif_report[vif_report['vif_grade'] == 'high']['feature'].values.tolist(),
        'inf_vif_cols': vif_report[vif_report['vif_grade'] == 'infinite']['feature'].values.tolist()
    }

    return vif_report.sort_values(by='vif_score').reset_index(drop=True), vif_report_dict