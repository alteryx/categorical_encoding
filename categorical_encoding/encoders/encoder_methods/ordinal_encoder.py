import featuretools as ft
from category_encoders import OrdinalEncoder as Ordinal

from categorical_encoding.primitives import OrdinalEnc


class OrdinalEncoder():
    """
        Maps each categorical value to one column using ordinal encoding.

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
            gets the mapping the ordinal encoder
            returns mapping (dict)
    """

    def __init__(self, cols):
        self.encoder = Ordinal(cols=cols)

    def fit(self, X, y=None):
        self.encoder.fit(X, y=None)
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
        if isinstance(category, str):
            for map in self.encoder.mapping:
                if map['col'] == category:
                    return map['mapping']
        return self.encoder.mapping[category]['mapping']

    def encode_features_list(self, X, features):
        feature_list = []
        index = 0
        for f in features:
            if f.get_name() in self.encoder.cols:
                f = ft.Feature([f], primitive=OrdinalEnc(self, index))
                index += 1
            feature_list.append(f)
        return feature_list
