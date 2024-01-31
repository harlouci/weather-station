"""
This script is to make transforms so that it's return dataframe so that we keep the columns names for easier interpretability and debuging
"""
from typing import List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin



class OneHotEncoderDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.encoder = OneHotEncoder(**kwargs)
        self.column_names = None

    def fit(self, X:pd.DataFrame, y=None)->None:
        self.encoder.fit(X)
        self.column_names = self.encoder.get_feature_names_out(X.columns)
        return self

    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        return pd.DataFrame(self.encoder.transform(X).toarray(), columns=self.column_names)

    def get_feature_names_out(self)->List[str]:
        return self.column_names


class TransformerToDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, base_transformer):
        self.base_transformer = base_transformer
        self.column_names = None  # Initialize column names as None

    def fit(self, X:pd.DataFrame, y=None)->None:
        self.base_transformer.fit(X, y)
        # Capture the column names during fitting
        self.column_names = X.columns.tolist()
        return self

    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        # Apply the transformation
        X_transformed = self.base_transformer.transform(X)

        # Convert the transformed array back to a DataFrame
        return pd.DataFrame(X_transformed, columns=self.column_names, index=X.index)

    def get_feature_names_out(self)->List[str]:
        return self.column_names


class DataFrameColumnTransformer(ColumnTransformer):
    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        # Transform the data using the original ColumnTransformer
        X_array = super().transform(X)

        # Get the output feature names
        feature_names = self.get_feature_names_out()

        # Convert the array to a DataFrame
        return pd.DataFrame(X_array, columns=feature_names, index=X.index)

    def fit_transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:
        # Fit and transform the data and convert to DataFrame in one step
        X_array = super().fit_transform(X, y)

        # Get the output feature names
        feature_names = self.get_feature_names_out()

        # Convert the array to a DataFrame
        return pd.DataFrame(X_array, columns=feature_names, index=X.index)



class RemoveLastNRowsTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that removes the last N rows from a DataFrame.

    Parameters:
    - n_rows: int. The number of rows to remove from the end of the DataFrame.
    """

    def __init__(self, n_rows=1):
        self.n_rows = n_rows

    def fit(self, X, y=None):
        # Nothing to fit, so we just return the instance
        return self

    def transform(self, X):
        # Check if X is a DataFrame
        if not hasattr(X, 'iloc'):
            raise ValueError("Input is not a pandas DataFrame")

        # Remove the last n_rows from X
        X_transformed = X.iloc[:-self.n_rows]
        return X_transformed


class SimpleCustomPipeline(Pipeline):
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        # Check if the pipeline has a final step that is a transformer
        if hasattr(self.steps[-1][1], 'get_feature_names_out'):
            # If the last step is a transformer with the method 'get_feature_names_out'
            return self.steps[-1][1].get_feature_names_out()
        else:
            raise AttributeError("The last step of the pipeline does not support 'get_feature_names_out'.")



