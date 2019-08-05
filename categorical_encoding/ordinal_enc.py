from featuretools.primitives.base.transform_primitive_base import (
    TransformPrimitive
)
from featuretools.variable_types import Categorical, Ordinal


class OrdinalEnc(TransformPrimitive):
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
