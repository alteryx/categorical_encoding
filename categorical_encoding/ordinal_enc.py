import featuretools as ft
import numpy as np
import pandas as pd

from featuretools.primitives.base.transform_primitive_base import (
    TransformPrimitive
)
from featuretools.utils import convert_time_units
from featuretools.variable_types import (
    Boolean,
    Datetime,
    DatetimeTimeIndex,
    LatLong,
    Numeric,
    Ordinal,
    Text,
    Variable,
    Categorical
)
from category_encoders import OrdinalEncoder

class OrdinalEnc(TransformPrimitive):
    name = "ordinal_enc"
    input_types = [Categorical]
    return_type = [Numeric]

    def __init__(self, mapping=None):
        self.mapping = mapping

    def get_function(self):
        def transform(X):
            if self.mapping is not None:  
                X = X.map(self.mapping)
            return X
        return transform

    def generate_name(self, base_feature_names):
        return u"%s_%s" % (base_feature_names[0].upper(), 'ordinal')

