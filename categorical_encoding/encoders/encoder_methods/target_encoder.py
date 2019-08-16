import featuretools as ft
from category_encoders import TargetEncoder as Target

from categorical_encoding.primitives import TargetEnc


class TargetEncoder():
    """Maps each categorical value to one column using target encoding.

    Parameters:
        cols: [str]
            list of column names to encode.
    """
    name = 'target'

    def __init__(self, cols=None):
        self.encoder = Target(cols=cols)

    def fit(self, X, features, y):
        """Fits encoder to data table based on given target column.
        returns self
        """
        self.encoder.fit(X, y)
        self.features = self.encode_features_list(X, features)
        return self

    def transform(self, X):
        """Encodes matrix and updates features accordingly.
        returns encoded matrix (dataframe)
        """
        X_new = self.encoder.transform(X)
        feature_names = []
        for feature in self.features:
            for fname in feature.get_feature_names():
                feature_names.append(fname)
        X_new.columns = feature_names
        return X_new

    def fit_transform(self, X, features, y=None):
        """First fits, then transforms matrix.
        returns encoded matrix (dataframe)
        """
        return self.fit(X, features, y).transform(X)

    def get_mapping(self, category):
        """Gets the mapping for the target encoder. Only takes strings of the column name, not the index number.
        returns tuple of dict (mapping, mapping of corresponding ordinal encoder)
        """
        def mapping_helper(method, category):
            if isinstance(category, str):
                for map in method.mapping:
                    if map['col'] == category:
                        return map['mapping']
            return method.mapping[category]['mapping']

        return self.encoder.mapping[category], mapping_helper(self.encoder.ordinal_encoder, category)

    def encode_features_list(self, X, features):
        feature_list = []
        for f in features:
            if f.get_name() in self.encoder.cols:
                f = ft.Feature([f], primitive=TargetEnc(self, f.get_name()))
            feature_list.append(f)
        return feature_list

    def get_features(self):
        return self.features

    def get_name(self):
        return self.name
