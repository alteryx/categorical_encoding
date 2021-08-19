from featuretools.primitives.base.transform_primitive_base import (
    TransformPrimitive
)
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, Integer

class LeaveOneOutEnc(TransformPrimitive):
    """Applies a fitted LeaveOneOut Encoder to the values.
    Requires an already fitted encoder.

    Parameters:
        fitted_encoder: encoder
            Encoder that has already learned encoding mappings from fitting to a data table.
        category: str or int
            String or integer corresponding to the name of the particular category.
            If integer, is the nth category encoded in the data table.

    Examples:
        >>> enc = Encoder(method='leave_one_out')
        >>> enc.fit_transform(feature_matrix, features)
        >>> encoder = OrdinalEnc(fitted_encoder=enc, category='product_id')
        >>> encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
        [2, 3, 1, 1]
    """
    name = "leave_one_out"
    input_types = [ColumnSchema(logical_type=Categorical)]
    return_type = [ColumnSchema(logical_type=Integer, semantic_tags={'numeric'})]

    def __init__(self, fitted_encoder, category):
        self.mapping = fitted_encoder.get_mapping(category)
        total = 0
        count = 0
        for index in self.mapping.index:
            total += self.mapping['sum'][index]
            count += self.mapping['count'][index]
        self.global_mean = total / count

    def get_function(self):
        def transform(X):
            mapped = []
            for value in X:
                if self.mapping['sum'][value] == 0:
                    mapped.append(self.global_mean)
                else:
                    mapped.append(self.mapping['sum'][value] / self.mapping['count'][value])
            return mapped
        return transform

    def generate_name(self, base_feature_names):
        return u"%s_%s" % (base_feature_names[0].upper(), 'leave_one_out')
