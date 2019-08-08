import pandas as pd
from featuretools.primitives.base.transform_primitive_base import (
    TransformPrimitive
)
from featuretools.variable_types import Categorical, Numeric


class OneHotEnc(TransformPrimitive):
    """Applies one hot encoding for the specific category value to the column

    Parameters:
        value: str or nan
            The category value.

    Examples:
        >>> enc = Encoder(method='one_hot')
        >>> enc.fit_transform(feature_matrix, features)
        >>> encoder = OneHotEnc(value='coke zero')
        >>> encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
        [0, 0, 1, 1]
    """
    name = "one_hot_enc"
    input_types = [Categorical]
    return_type = [Numeric]

    def __init__(self, value=None):
        self.value = value

    def get_function(self):
        def transform(X):
            if pd.isnull(self.value):
                return pd.isna(X).astype(int)
            return (pd.Series(X).astype(pd.Series([self.value]).dtype) == self.value).astype(int)
        return transform

    def generate_name(self, base_feature_names):
        return u"%s = %s" % (base_feature_names[0], str(self.value))
