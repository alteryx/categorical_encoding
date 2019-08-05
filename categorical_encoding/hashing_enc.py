from featuretools.primitives.base.transform_primitive_base import (
    TransformPrimitive
)
from featuretools.variable_types import Categorical, Numeric
from category_encoders import HashingEncoder
import numpy as np

class HashingEnc(TransformPrimitive):
    name = "hashing_enc"
    input_types = [Categorical]
    return_type = [Numeric]

    def __init__(self, fitted_encoder):
        self.hash_method = fitted_encoder.get_hash_method()
        self.n = fitted_encoder.get_n_components()
        self.number_output_features = self.n

    def get_function(self):
        def transform(X):
            new_encoder = HashingEncoder(hash_method=self.hash_method, n_components=self.n)
            return np.swapaxes(new_encoder.fit_transform(X).values, 0, 1)
        return transform

    def generate_name(self, base_feature_names):
        return u"%s_%s" % (base_feature_names[0].upper(), 'hashing')
