import featuretools as ft
from category_encoders import BinaryEncoder as Binary

from categorical_encoding.primitives import BinaryEnc


class BinaryEncoder():
    """
        Maps each categorical value to several columns using binary encoding.

        Parameters:
        cols: [str]
            list of column names to encode.

        Functions:
        fit:
            fits encoder to data table
            returns self
        transform:
            encodes matrix and updates features accordingly
            returns encoded matrix (dataframe)
        fit_transform:
            first fits, then transforms matrix
            returns encoded matrix (dataframe)
        get_mapping:
            gets the mapping for the binary encoder and underlying ordinal encoder
            returns tuple (binary_encoder_mapping, ordinal_encoder_mapping)
    """

    def __init__(self, cols):
        self.encoder = Binary(cols=cols)

    def fit(self, X, y=None):
        self.encoder.fit(X, y)
        return self

    def transform(self, X, features):
        X_new = self.encoder.transform(X)
        feature_names = []
        for feature in features:
            for fname in feature.get_feature_names():
                feature_names.append(fname)
        X_new.columns = feature_names

        return X_new

    def fit_transform(self, X, features, y=None):
        return self.fit(X, y).transform(X, features)

    def get_mapping(self, category):
        def mapping_helper(method, category):
            if isinstance(category, str):
                for map in method.mapping:
                    if map['col'] == category:
                        return map['mapping']
            return method.mapping[category]['mapping']

        return mapping_helper(self.encoder.base_n_encoder, category), \
            mapping_helper(self.encoder.base_n_encoder.ordinal_encoder, category)

    def encode_features_list(self, X, features):
        feature_list = []
        index = 0
        for f in features:
            if f.get_name() in self.encoder.base_n_encoder.cols:
                f = ft.Feature([f], primitive=BinaryEnc(self, index))
                index += 1
            feature_list.append(f)
        return feature_list
