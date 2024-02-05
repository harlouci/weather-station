"""
This script is to make transforms so that it's return dataframe so that we keep the columns names
for easier interpretability and debuging.
"""

from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class TransformerToDataFrame(BaseEstimator, TransformerMixin):
    """As some transformers, e.g. StandardScaler(), return a np.array, wrapped
    with this transformer they return a dataframe, with the same column names."""

    def __init__(self, base_transformer):
        self.base_transformer = base_transformer
        self.column_names = None  # Initialize column names as None

    def fit(self, x: pd.DataFrame, y=None) -> None:
        self.base_transformer.fit(x, y)
        # Capture the column names during fitting
        self.column_names = x.columns.tolist()
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        # Apply the transformation
        x_transformed = self.base_transformer.transform(x)

        # Convert the transformed array back to a DataFrame
        return pd.DataFrame(x_transformed, columns=self.column_names, index=x.index)

    def get_feature_names_out(self) -> List[str]:
        return self.column_names


class DataFrameColumnTransformer(ColumnTransformer):
    """As the method `transform()` of ColumnTransformer() returns a np.array, wrapped
    with  this transformer it returns a dataframe, with the same column names.
    """

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        # Transform the data using the original ColumnTransformer
        x_array = super().transform(x)

        # Get the output feature names
        feature_names = self.get_feature_names_out()

        # Convert the array to a DataFrame
        return pd.DataFrame(x_array, columns=feature_names, index=x.index)

    def fit_transform(self, x: pd.DataFrame, y=None) -> pd.DataFrame:
        # Fit and transform the data and convert to DataFrame in one step
        x_array = super().fit_transform(x, y)

        # Get the output feature names
        feature_names = self.get_feature_names_out()

        # Convert the array to a DataFrame
        return pd.DataFrame(x_array, columns=feature_names, index=x.index)


class SimpleCustomPipeline(Pipeline):
    """Wraps the class Pipeline() to add it the method `get_feature_names_out`."""

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        # Check if the pipeline has a final step that is a transformer
        if hasattr(self.steps[-1][1], "get_feature_names_out"):
            # If the last step is a transformer with the method 'get_feature_names_out'
            return self.steps[-1][1].get_feature_names_out()
        else:
            msg = "The last step of the pipeline does not support 'get_feature_names_out'."
            raise AttributeError(msg)
