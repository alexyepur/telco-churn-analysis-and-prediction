import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


#from utility_functions.logger import get_logger
#logger = get_logger(__name__)

# ================================================================
# Standardize String Values
# ================================================================
class StandardizeStringValues(BaseEstimator, TransformerMixin):
    """
    Standardizes string values in object-type columns

    Automatically detects columns of type `object` or `string` and applies a standardization process: 
    - conversion to lowercase and stripping 
    - leading/trailing whitespaces

    Parameters:
        some_param (int, optional): Dummy parameter to avoid fitting warnings (ignored). Defaults to 1

    Methods:
        fit(X, y=None): Identifies string columns in the input DataFrame
        transform(X): Applies string standardization to identified columns
    """
    
    def __init__(self, some_param=1):
        """
        Initializes the StandardizeStringValues transformer

        Parameters:
            some_param (int, optional): Dummy parameter to avoid fitting warnings (ignored). Defaults to 1
        """
        self.some_param = some_param

    def fit(self, X, y=None):
        """
        Identifies string columns to standardize.

        Parameters:
            X (pd.DataFrame): Features matrix
            y (pd.Series, optional): Target vector (ignored). Defaults to None

        Returns:
            StandardizeStringValues: Self, fitted with identified string columns
        """
        self.is_fitted = True
        self.str_cols_ = X.select_dtypes(include='object').columns.tolist()
        return self

    def transform(self, X):
        """
        Transforms string columns by lowercasing and trimming whitespace

        Parameters:
            X (pd.DataFrame): Features matrix

        Returns:
            pd.DataFrame: Transformed DataFrame with standardized string columns
        """
        X_ = X.copy()
        for col in self.str_cols_:
            X_[col] = X_[col].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
        return X_


# ================================================================
# Fixed Column Discretization
# ================================================================
class FixedColumnDiscretization(BaseEstimator, TransformerMixin):
    """
    Discretizes a specific numeric feature into categorical bins

    Applies fixed-bin discretization using `pd.cut()` to a single specified numeric column.
    The user defines explicit bin edges and corresponding labels. 
    Optionally, the original numeric column can be dropped after discretization

    Parameters:
        bins (list[float]): List of bin edges (must be monotonically increasing)
        labels (list[str]): List of category labels corresponding to bins. Length must be len(bins) - 1
        feature_name (str): Name of the column to be discretized
        drop_original (bool, optional): If True, drops the original column after transformation. Defaults to False

    Methods:
        fit(X, y=None): Validates input and stores column name
        transform(X): Applies discretization to the specified column and returns modified DataFrame
    """
    
    def __init__(self, bins: list[float], labels: list[str], feature_name: str, drop_original: bool = False):
        """
        Initializes the FixedColumnDiscretization transformer

        Parameters:
            bins (list[float]): List of bin edges (must be monotonically increasing)
            labels (list[str]): List of category labels corresponding to bins. Length must be len(bins) - 1
            feature_name (str): Name of the column to be discretized
            drop_original (bool, optional): If True, drops the original column after transformation. Defaults to False
        """
        
        self.bins = bins
        self.labels = labels
        self.feature_name = feature_name
        self.drop_original = drop_original

    def fit(self, X, y=None):
        """
        Validates input data and checks configuration.

        Parameters:
            X (pd.DataFrame): Feature matrix
            y (pd.Series, optional): Target vector (ignored). Defaults to None

        Returns:
            FixedColumnDiscretization: Fitted transformer
        """
        
        if self.feature_name not in X.columns:
            raise ValueError(f"Column '{self.feature_name}' not found in input data")

        if len(self.labels) != (len(self.bins)-1):
            raise ValueError("Length of labels must be equal to len(bins)-1")
            
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Applies binning to the specified feature

        Parameters:
            X (pd.DataFrame): Feature matrix

        Returns:
            pd.DataFrame: Transformed DataFrame with binned feature added or replacing original
        """
        X_ = X.copy()
        if not hasattr(self, 'is_fitted_'):
            raise RuntimeError("You must fit transformer before transform")
            
        X_[self.feature_name + '_binned'] = pd.cut(X_[self.feature_name], bins=self.bins, labels=self.labels, include_lowest=True)
        if self.drop_original:
            X_ = X_.drop(columns=self.feature_name)
        return X_


# ================================================================
# Quantile Column Discretization
# ================================================================
class QuantileColumnDiscretization(BaseEstimator, TransformerMixin):
    """
    Discretizing a numeric feature into quantile-based bins.

    Applies quantile-based discretization using `pd.cut()` to a specified numeric column. 
    Bin edges are determined by empirical quantiles, computed using `np.linspace(0, 1, n_bins + 1)`. 
    Optionally, the original numeric column can be dropped after transformation

    Parameters:
        n_bins (int): Number of quantile bins to compute
        labels (list[str]): Category labels for each bin. Must have length equal to `n_bins`
        feature_name (str): Name of the feature to be discretized
        drop_original (bool, optional): If True, drops the original column after transformation. Defaults to False

    Methods:
        fit(X, y=None): Validates configuration and computes quantile bin edges
        transform(X): Applies discretization to the specified column and returns the modified DataFrame
    """
    
    def __init__(self, n_bins: int, labels: list[str], feature_name: str, drop_original: bool = False):
        """
        Initializes the QuantileColumnDiscretization transformer.

        Parameters:
            n_bins (int): Number of quantile bins to compute
            labels (list[str]): Category labels for each bin. Must have length equal to `n_bins`
            feature_name (str): Name of the feature to be discretized
            drop_original (bool, optional): If True, drops the original column after transformation. Defaults to False
        """
        self.n_bins = n_bins
        self.labels = labels
        self.feature_name = feature_name
        self.drop_original = drop_original

    def fit(self, X, y=None):
        """
        Computes quantile-based bin edges and validates the column

        Parameters:
            X (pd.DataFrame): Feature matrix
            y (pd.Series, optional): Target vector (ignored). Defaults to None

        Returns:
            QuantileColumnDiscretization: Fitted transformer
        """
        
        if self.feature_name not in X.columns:
            raise ValueError(f"Column '{self.feature_name}' not found in input data")

        if len(self.labels) != self.n_bins:
            raise ValueError("Length of labels must be equal to n_bins value")

        quantiles = np.linspace(0, 1, self.n_bins+1)
        self.bins_ = X[self.feature_name].quantile(quantiles).unique()
        if len(self.bins_) <= 1:
            raise ValueError(f"Column '{self.feature_name}' has constant value or not enough unique values to bin.")
            
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Applies quantile binning to the specified feature

        Parameters:
            X (pd.DataFrame): Feature matrix

        Returns:
            pd.DataFrame: Transformed DataFrame with binned feature
        """
        
        if not hasattr(self, 'is_fitted_'):
            raise RuntimeError("You must fit transformer before transform")
        
        X_ = X.copy()
            
        X_[self.feature_name + '_binned'] = pd.cut(X_[self.feature_name], bins=self.bins_, labels=self.labels, include_lowest=True)
        if self.drop_original:
            X_ = X_.drop(columns=self.feature_name)
        return X_


# ================================================================
# Positive Value Flag Transformer
# ================================================================
class PositiveValueFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Creates binary flags indicating whether values in specified features are positive

    Optionally, the original column can be dropped after transformation
        
    Parameters:
        features (list[str]): List of column names to apply the transformation to
        prefix (str, optional): Prefix to use for new flag columns. Defaults to 'has_'
        drop_original (bool, optional): If True, removes the original columns. Defaults to False

    Methods:
        fit(X, y=None): Validates the presence of specified columns
        transform(X): Adds binary flag columns indicating positive values in the original features
    """
    
    def __init__(self, features: list[str], prefix: str = 'has_', drop_original: bool=False):
        """
        Initializes the PositiveValueFlagTransformer.

        Parameters:
            features (list[str]): List of column names to apply the transformation to
            prefix (str, optional): Prefix to use for new flag columns. Defaults to 'has_'
            drop_original (bool, optional): If True, removes the original columns. Defaults to False
        """
        self.features = features
        self.prefix = prefix
        self.drop_original = drop_original

    def fit(self, X, y=None):
        """
        Validates input columns.

        Parameters:
            X (pd.DataFrame): Feature matrix
            y (pd.Series, optional): Target vector (ignored). Defaults to None

        Returns:
            PositiveValueFlagTransformer: Fitted transformer instance
        """
        missing_cols = [col for col in self.features if col not in X.columns]
                
        if missing_cols:
            raise ValueError(f"Columns '{missing_cols}' not found in input data")
        
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Creates binary flag columns indicating positive values

        Parameters:
            X (pd.DataFrame): Feature matrix

        Returns:
            pd.DataFrame: Transformed DataFrame with binary indicator columns added
        """
        if not hasattr(self, 'is_fitted_'):
            raise RuntimeError("You must fit transformer before transform")
        
        X_ = X.copy()

        for col in self.features:
            X_[f"{self.prefix}{col}"] = (X_[col] > 0).astype(int)

        if self.drop_original:
            X_ = X_.drop(columns=self.features)
        
        return X_


# ================================================================
# Skewed Features Transformer
# ================================================================
class SkewedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Applies custom skewness correction to numeric features

    Applies various mathematical transformations to specified features to reduce skewness. 
    Supported transformations include:
        - 'log':      log(x + 1), for strictly non-negative values.
        - 'reciprocal': 1 / (x + 1), for strictly non-negative values.
        - 'sqrt':     sqrt(x), for non-negative values.
        - 'boxcox':   Box-Cox transform, requires strictly positive values.
        - 'yeo-johnson': Yeo-Johnson transform, supports zero and negative values.

    Each transformed feature is added as a new column with a suffix corresponding to the method used.
    Optionally, the original column can be dropped

    Parameters:
        transform_instructions (dict[str, str]): Mapping of column names to transformation methods
        drop_original (bool, optional): If True, drops original features after transformation. Defaults to False

    Methods:
        fit(X, y=None): Validates columns, checks for NaNs and stores fitted transformers for 'boxcox'/'yeo-johnson'
        transform(X): Applies selected transformations and returns modified DataFrame

    """
    
    def __init__(self, transform_instructions: dict[str,str], drop_original: bool=False):
        """
        Initializes the SkewedFeaturesTransformer.

        Parameters:
            transform_instructions (dict[str, str]): Mapping of column names to transformation methods
            drop_original (bool, optional): If True, drops original features after transformation. Defaults to False
        """
        self.transform_instructions = transform_instructions
        self.pt_dict = {}
        self.drop_original = drop_original
    
    def fit(self, X, y=None):
        """
        Validates input and prepares transformers

        Parameters:
            X (pd.DataFrame): Feature matrix
            y (pd.Series, optional): Target vector (ignored). Defaults to None

        Returns:
            SkewedFeaturesTransformer: Fitted transformer instance
        """
        
        for col, method in self.transform_instructions.items():
        
            if col not in X.columns:
                raise KeyError(f"Column \"{col}\" was not found in dataframe.")
            
            if X[col].isnull().any():
                raise ValueError(f"Column \"{col}\" has NaN values. Consider handling them before transformation.")
            
            if method not in ['log', 'reciprocal', 'sqrt', 'boxcox', 'yeo-johnson']:
                raise ValueError(f"Chosen method \"{method}\" is not supported")

            if method in ['boxcox', 'yeo-johnson']:
                pt = PowerTransformer(method='yeo-johnson' if method=='yeo-johnson' else 'box-cox')
                pt.fit(X[[col]])
                self.pt_dict[col] = pt
        
        self.is_fitted_ = True
        return self

        
    def transform(self, X):
        """
        Applies specified transformations to input DataFrame

        Parameters:
            X (pd.DataFrame): Feature matrix

        Returns:
            pd.DataFrame: Transformed DataFrame with new columns
        """
        if not hasattr(self, 'is_fitted_'):
            raise RuntimeError("You must fit transformer before transform")
            
        X_ = X.copy()
        affected_columns = []
        
        for col, method in self.transform_instructions.items():
            if method == 'log':
                if (X_[col] + 1 <= 0).any():
                    raise ValueError(f"Column \"{col}\" contains non-positive values. Method \"{method}\" is not available. Choose another method.")
                X_[f"{col}_{method}"] = np.log(X_[col]+1)
                affected_columns.append(f"{col}_{method}")
            
            elif method == 'reciprocal':
                if (X_[col] +1 <= 0).any():
                    raise ValueError(f"Column \"{col}\" contains non-positive values. Method \"{method}\" is not available. Choose another method.")
                X_[f"{col}_{method}"] = 1 / (X_[col]+1)
                affected_columns.append(f"{col}_{method}")
                
            elif method == 'sqrt':
                if (X_[col] < 0).any():
                    raise ValueError(f"Column \"{col}\" contains negative values. Method \"{method}\" is not available. Choose another method.")
                X_[f"{col}_{method}"] = np.sqrt(X_[col])
                affected_columns.append(f"{col}_{method}")
            
            elif method == 'boxcox':
                if (X_[col] <= 0).any():
                    raise ValueError(f"Column \"{col}\" contains non-positive values. Method \"{method}\" is not available. Choose another method.")
                
                if col not in self.pt_dict:
                    raise RuntimeError(f"Missing fitted transformer for column {col}")
                    
                X_[f"{col}_{method}"] = self.pt_dict[col].transform(X_[[col]])
                affected_columns.append(f"{col}_{method}")
            
            elif method == 'yeo-johnson':
                if col not in self.pt_dict:
                    raise RuntimeError(f"Missing fitted transformer for column {col}")
                    
                X_[f"{col}_{method}"] = self.pt_dict[col].transform(X_[[col]])
                affected_columns.append(f"{col}_{method}")
            
        #if affected_columns:
            #logger.info(f"Successfully transformed columns: {', '.join(affected_columns)}")

        for col, method in self.transform_instructions.items():
            if self.drop_original:
                X_ = X_.drop(columns=col)
                
        return X_


# ================================================================
# Mapping Transformer
# ================================================================
class MappingTransformer(BaseEstimator, TransformerMixin):
    """
    Applies predefined mapping dictionaries to specific columns

    Replaces values in selected columns according to user-provided mapping dictionaries.
    Intended for categorical feature encoding or standardizing non-numeric labels

    Args:
        mapping_dict (dict[str, dict[Any, Any]]): Dictionary where keys are column names and values are mapping dictionaries for each column

    Methods:
        fit(X, y=None): Validates that all values in specified columns are covered by provided mappings
        transform(X): Applies the mappings to each corresponding column
    """
    
    def __init__(self, mapping_instruction: dict[str, dict]):
        """
        Checks that all columns and their values in `mapping_dict` exist in the input DataFrame.

        Args:
            X (pd.DataFrame): Feature matrix
            y (ignored): Not used, present for compatibility.

        Returns:
            MappingTransformer: Fitted transformer instance
        """

        self.mapping_instruction = mapping_instruction

    def fit(self, X, y=None):
        """
        Fits the transformer on the data.
    
        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series, optional): Target vector (ignored). Defaults to None
    
        Returns:
            TransformerName: The fitted transformer
        """
        self.columns_to_map_ = list(self.mapping_instruction.keys())
        
        missing_columns = [col for col in self.columns_to_map_ if col not in X.columns]
        if missing_columns:
            raise KeyError(f"Column(s) '{missing_columns}' was/were not found in dataset.")

        for col in self.columns_to_map_:
            map_values = set(self.mapping_instruction[col].keys())
            dataset_values = set(X[col].unique())

            if not dataset_values.issubset(map_values):
                missing_values = dataset_values - map_values
                raise ValueError(f"Column '{col}' has values {missing_values} not covered by mapping.")
         
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Applies the mapping to each specified column.

        Args:
            X (pd.DataFrame): Feature matrix

        Returns:
            pd.DataFrame: DataFrame with mapped values.
        """
        
        if not hasattr(self, 'is_fitted_'):
            raise RuntimeError("You must fit transformer before transform")

        X_ = X.copy()
        for col in self.columns_to_map_:
            X_[col] = X_[col].map(self.mapping_instruction[col])
            
        return X_


# ================================================================
# Yes No To Binary Mapper
# ================================================================
class YesNoToBinaryMapper(BaseEstimator, TransformerMixin):
    """
    Converts 'yes'/'no' categorical columns to binary 1/0 encoding.

    It validates that only 'yes'/'no' (case-sensitive) values are present in
    the selected columns.

    Args:
        yes_no_columns (list[str]): List of column names to be binary-encoded

    Methods:
        fit(X, y=None): Validates column presence and value consistency
        transform(X): Applies binary mapping to the selected columns
    """
    
    def __init__(self, yes_no_columns):
        """
        Initializes the YesNoToBinaryMapper
    
        Args:
            yes_no_columns (list[str]): List of column names to be binary-encoded
        """
        
        self.yes_no_columns = yes_no_columns
    
    def fit(self, X, y=None):
        """
        Checks that the specified columns exist and contain only 'yes'/'no'.

        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series, optional):Target vector (ignored). Defaults to None

        Raises:
            KeyError: If any specified column is missing from the input.
            ValueError: If unexpected values are found in any target column.

        Returns:
            YesNoToBinaryMapper: The fitted transformer.
        """
        # missing columns check
        missing_columns = [col for col in self.yes_no_columns if col not in X.columns]
        if missing_columns:
            raise KeyError(f"Column(s) '{missing_columns}' was/were not found in dataset.")

        # non "yes/no" columns check
        invalid_columns = {}
        for col in self.yes_no_columns:
            extra_values = set(X[col].dropna().unique()) - {'yes', 'no'}
            if extra_values:
                invalid_columns[col] = extra_values
        if invalid_columns:
            raise ValueError(f"Columns with unexpected values: {invalid_columns}")        
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Applies binary mapping to 'yes'/'no' columns

        Args:
            X (pd.DataFrame): Feature matrix

        Returns:
            pd.DataFrame: Transformed feature matrix with binary values
        """
        if not hasattr(self, 'is_fitted_'):
            raise RuntimeError("You must fit transformer before transform")

        X_ = X.copy()

        for col in self.yes_no_columns:
            X_[col] = X_[col].map({'yes':1, 'no':0})

        return X_


# ================================================================
# Unused Features Dropper
# ================================================================
class UnusedFeaturesDropper(BaseEstimator, TransformerMixin):
    """
    Drops specified columns from the input DataFrame

    Removes chosen columns. Can be used as part of a preprocessing pipeline

    Args:
        cols_to_drop (list[str]): List of column names to drop

    Methods:
        fit(X, y=None): Marks the transformer as fitted
        transform(X): Drops the specified columns from the input DataFrame
    """
    
    def __init__(self, cols_to_drop: list[str]):
        """
        Initializes the UnusedFeaturesDropper.
    
        Args:
            cols_to_drop (list[str]): List of column names to drop
        """
        
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        """
        Marks the transformer as fitted. No statistics are learned

        Args:
            X (pd.DataFrame): Features matrix
            y (pd.Series, optional): Target vector (ignored). Defaults to None

        Returns:
            UnusedFeaturesDropper: Fitted transformer
        """
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Drops specified columns from the input DataFrame.

        Args:
            X (pd.DataFrame): Features matrix

        Returns:
            pd.DataFrame: Transformed DataFrame with specified columns removed
        """
        
        if not hasattr(self, 'is_fitted_'):
            raise RuntimeError('You must fit transformer before transform')

        missing_cols = [col for col in self.cols_to_drop if col not in X.columns]
        if missing_cols:
            print(f"Column(s) {missing_cols} was/were not found and skipped")
        
        X_ = X.copy()
        X_ = X_.drop(columns=self.cols_to_drop, errors='ignore')

        return X_

# ================================================================
# Array To DF Transformer
# ================================================================
class ArrayToDFTransformer(BaseEstimator, TransformerMixin):
    """
    Wraps a column transformer and returns its output as a pandas DataFrame

    Allows for integration with pandas-based workflows by preserving feature names and DataFrame structure.

    Args:
        col_transformer (TransformerMixin): Fitted transformer that must implement `fit`, `transform`, and `get_feature_names_out`

    Methods:
        fit(X, y=None): Fits the internal transformer
        transform(X): Transforms input and returns a DataFrame with named columns
    """
    
    def __init__(self, col_transformer):
        """
        Initializes the transformer.

        Args:
            col_transformer (TransformerMixin): Fitted transformer that must implement `fit`, `transform`, and `get_feature_names_out`
        """
        
        self.col_transformer = col_transformer
    
    def fit(self, X, y=None):
        """
        Fits the internal transformer on the provided data

        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series, optional): Target vector (ignored). Defaults to None

        Returns:
            ArrayToDFTransformer: The fitted transformer instance
        """
        
        if not hasattr(self.col_transformer, 'fit'):
            raise TypeError('col_transformer must be able to fit')
        
        self.col_transformer.fit(X,y)
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transforms the data using the internal transformer and returns a DataFrame

        Args:
            X (pd.DataFrame): Feature matrix

        Returns:
            pd.DataFrame: Transformed DataFrame with proper column names.
        """
        
        if not hasattr(self, 'is_fitted_'):
            raise RuntimeError('You must fit transformer before transform')

        X_transformed = self.col_transformer.transform(X)
        columns = self.col_transformer.get_feature_names_out()
        df_transformed = pd.DataFrame(X_transformed, columns=columns, index=X.index)
        
        return df_transformed



# ================================================================
# Frequency Encoder
# ================================================================
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features using frequency encoding.

    For each specified column, replaces category labels with their relative frequency in the training data. 
    Unknown values during transform are encoded as 0.

    Args:
        cat_cols (list[str]): List of categorical columns names to encode

    Methods:
        fit(X, y=None): Learns the frequency of each category in specified columns
        transform(X): Applies frequency encoding to the specified columns
    """
    
    def __init__(self, cat_cols):
        """
        Initializes the transformer.

        Args:
            cat_cols (list[str]): List of categorical columns names to encode
        """
        
        self.cat_cols = cat_cols
        self.freq_dict_ = {}
    
    def fit(self, X, y=None):
        """
        Learns frequency mappings for each specified categorical column.

        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series, optional): Target vector (ignored). Defaults to None

        Returns:
            FrequencyEncoder: Fitted transformer instance.
        """
        
        missing_cols = [col for col in self.cat_cols if col not in X.columns]
        if missing_cols:
            raise KeyError(f'Columns not found in dataset during fit: {missing_cols}')

        for col in self.cat_cols:
            freq = X[col].value_counts(normalize=True).to_dict()
            self.freq_dict_[col] = freq
            
        self.is_fitted_ = True
        
        return self

    def transform(self, X):
        """
        Applies frequency encoding to the specified categorical columns

        Args:
            X (pd.DataFrame): Feature matrix

        Returns:
            pd.DataFrame: Transformed DataFrame with encoded categorical features.
        """
        
        if not hasattr(self, 'is_fitted_'):
            raise RuntimeError('You must fit transformer before transform') 

        missing_cols = [col for col in self.cat_cols if col not in X.columns]
        if missing_cols:
            raise KeyError(f'Columns not found in dataset during transform: {missing_cols}')

        X_ = X.copy()
        for col, freq in self.freq_dict_.items():
            X_[col] = X_[col].map(freq).fillna(0)

        return X_.to_numpy()