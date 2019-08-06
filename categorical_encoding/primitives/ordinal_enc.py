from featuretools.primitives.base.transform_primitive_base import (
    TransformPrimitive
)
from featuretools.variable_types import Categorical, Ordinal


class OrdinalEnc(TransformPrimitive):
    """Applies a fitted Ordinal Encoder to the values.
    Requires an already fitted encoder.

    Parameters:
        fitted_encoder: encoder
            encoder that has already learned encoding mappings from fitting to a data table.
        category: str or int
            string or integer corresponding to the name of the particular category.
            If integer, is the nth category encoded in the data table.

    Examples:
        >>> enc = Encoder(method='ordinal')
        >>> enc.fit_transform(feature_matrix, features)
        >>> encoder = OrdinalEnc(fitted_encoder=enc, category='product_id')
        >>> encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
        [2, 3, 1, 1]
    """
    name = "ordinal_enc"
    input_types = [Categorical]
    return_type = Ordinal

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
