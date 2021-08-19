from featuretools.primitives.base.transform_primitive_base import (
    TransformPrimitive
)
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, Integer

class OrdinalEnc(TransformPrimitive):
    """Applies a fitted Ordinal Encoder to the values.
    Requires an already fitted encoder.

    Parameters:
        fitted_encoder: encoder
            Encoder that has already learned encoding mappings from fitting to a data table.
        category: str or int
            String or integer corresponding to the name of the particular category.
            If integer, is the nth category encoded in the data table.

    Examples:
        >>> enc = Encoder(method='ordinal')
        >>> enc.fit_transform(feature_matrix, features)
        >>> encoder = OrdinalEnc(fitted_encoder=enc, category='product_id')
        >>> encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
        [2, 3, 1, 1]
    """
    name = "ordinal_enc"
    input_types = [ColumnSchema(logical_type=Categorical)]
    return_type = [ColumnSchema(logical_type=Integer, semantic_tags={'numeric'})]

    def __init__(self, fitted_encoder, category):
        self.mapping = fitted_encoder.get_mapping(category)

    def get_function(self):
        def transform(X):
            if self.mapping is not None:
                X = X.map(self.mapping)
            return X
        return transform

    def generate_name(self, base_feature_names):
        return u"%s_%s" % (base_feature_names[0].upper(), 'ordinal')
