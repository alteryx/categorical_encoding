from featuretools.primitives.base.transform_primitive_base import (
    TransformPrimitive
)
from featuretools.variable_types import Categorical, Numeric
import pandas as pd
import numpy as np


class BinaryEnc(TransformPrimitive):
    name = "binary_enc"
    input_types = [Categorical]
    return_type = [Numeric]

    def __init__(self, mapping=None, mapping_ord=None):
        self.mapping = mapping
        self.mapping_ord = mapping_ord
        self.n = mapping.shape[1]
        self.number_output_features = self.n

    def get_function(self):
        def transform(X):
            mapped = []
            if self.mapping_ord is not None:
                X = X.map(self.mapping_ord)
            if self.mapping is not None:
                for col in range(self.n):
                    mapped.append([self.mapping.loc[value][col] for value in X])
            return mapped
        return transform

    def generate_name(self, base_feature_names):
        return u"%s_%s" % (base_feature_names[0].upper(), 'binary')
