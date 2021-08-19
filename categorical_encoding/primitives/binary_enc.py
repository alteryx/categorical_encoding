import numpy as np
from featuretools.primitives.base.transform_primitive_base import (
    TransformPrimitive
)
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, Integer


class BinaryEnc(TransformPrimitive):
    """Applies a fitted Binary Encoder to the values. Requires an already fitted encoder.

    Parameters:
        fitted_encoder: encoder
            Encoder that has already learned encoding mappings from fitting to a data table.
        category: str or int
            String or integer corresponding to the name of the particular category.
            If integer, is the nth category encoded in the data table.

    Examples:
        >>> enc = Encoder(method='binary')
        >>> enc.fit_transform(feature_matrix, features)
        >>> encoder = BinaryEnc(fitted_encoder=enc, category='product_id')
        >>> encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
        [[0, 0, 0, 0],
         [1, 1, 0, 0],
         [0, 1, 1, 1]]
    """
    name = "binary_enc"
    input_types = [ColumnSchema(logical_type=Categorical)]
    return_type = ColumnSchema(semantic_tags={'numeric'})

    def __init__(self, fitted_encoder, category):
        self.mapping, self.mapping_ord = fitted_encoder.get_mapping(category)
        self.n = self.mapping.shape[1]
        self.number_output_features = self.n

    def get_function(self):
        def transform(X):
            mapped = []
            if self.mapping_ord is not None:
                X = X.map(self.mapping_ord)
            if self.mapping is not None:
                for value in X:
                    mapped.append(self.mapping.loc[value].values)
            return np.swapaxes(mapped, 0, 1)
        return transform

    def generate_name(self, base_feature_names):
        return u"%s_%s" % (base_feature_names[0].upper(), 'binary')
