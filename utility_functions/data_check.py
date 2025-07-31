import pandas as pd


#from utility_functions.logger import get_logger
#logger = get_logger(__name__)


def detect_missing_values(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Detects and reports columns with missing values.

    Computes the number and percentage of missing values in each column of the dataset.
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
    
    Returns:
        pd.DataFrame: Table with columns 'missing_count' and 'missing_%' for features with at least one missing value.
    """
    
    miss_rep = pd.concat([data.isnull().sum(), data.isnull().mean()*100], axis=1)
    miss_rep.columns = ['missing_count', 'missing_%']
    miss_rep = miss_rep[miss_rep['missing_count'] > 0]
    if not miss_rep.empty:
        print('Some missing values were detected!')
        return miss_rep
    else: 
        print('No missing values detected!')

        
def check_datasets_before_merge(
    first_df: pd.DataFrame, 
    second_df: pd.DataFrame, 
    uniting_column: str
) -> None:
    """
    Checks two datasets for consistency before merging by a common key.

    Verifies existence and uniqueness of the key column, presence of nulls, column name overlaps (excluding the key), 
    and matching key values between the datasets. Reports all issues found.
    
    Args:
        first_df (pd.DataFrame): First feature matrix to be merged
        second_df (pd.DataFrame): Second feature matrix to be merged
        uniting_column (str) : Name of the column to merge on
    
    Returns:
        None
    """
    
    issues = []

    if uniting_column not in first_df.columns:
        raise KeyError(f"'{uniting_column}' is missing in first_df.")
    if uniting_column not in second_df.columns:
        raise KeyError(f"'{uniting_column}' is missing in second_df.")


    first_nulls = first_df.isnull().sum()[first_df.isnull().sum() > 0]
    second_nulls = second_df.isnull().sum()[second_df.isnull().sum() > 0]
    if not first_nulls.empty:
        issues.append(f"Missing values in first_df:\n{first_nulls}")
    if not second_nulls.empty:
        issues.append(f"Missing values in second_df:\n{second_nulls}")

    # Проверка уникальности ключа
    n_first_dup = first_df[uniting_column].duplicated().sum()
    n_second_dup = second_df[uniting_column].duplicated().sum()
    if n_first_dup > 0 or n_second_dup > 0:
        issues.append(f"Duplicates in '{uniting_column}': {n_first_dup} in first_df, {n_second_dup} in second_df")

    # Проверка совпадения значений ключа
    if set(first_df[uniting_column]) != set(second_df[uniting_column]):
        issues.append(f"Values in '{uniting_column}' do not match between datasets")

    # Проверка дублирования колонок кроме ключа
    first_cols = set(first_df.columns) - {uniting_column}
    second_cols = set(second_df.columns) - {uniting_column}
    duplicated_cols = first_cols.intersection(second_cols)
    if duplicated_cols:
        issues.append(f"Duplicated columns (except '{uniting_column}'): {list(duplicated_cols)}")

    # Итог
    print("\nFinal check for merging datasets on '{}':".format(uniting_column))
    print("------------------------------------------------------")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"- {issue}")
        print("\nDatasets are NOT ready to be merged.")
    else:
        print("No issues found. Datasets are ready to be merged.")
        print(f"first_df shape : {first_df.shape}")
        print(f"second_df shape: {second_df.shape}")
        print(f"Expected merged shape: ({first_df.shape[0]}, {first_df.shape[1] + second_df.shape[1] - 1})")
    print("------------------------------------------------------")

