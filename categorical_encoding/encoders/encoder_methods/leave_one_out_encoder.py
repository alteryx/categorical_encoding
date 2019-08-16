import featuretools as ft
from category_encoders import LeaveOneOutEncoder as LeaveOneOut

from categorical_encoding.primitives import LeaveOneOutEnc


class LeaveOneOutEncoder():
    """Maps each categorical value to one column using LeaveOneOut encoding.

    Parameters:
        cols: [str]
            list of column names to encode.
    """
    name = 'leave_one_out'

    def __init__(self, cols=None):
        self.encoder = LeaveOneOut(cols=cols)

    def fit(self, X, features, y):
        """Fits encoder to data table.
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
        X_new.columns = self._rename_columns(self.features)
        return X_new

    def fit_transform(self, X, features, y=None):
        """First fits, then transforms matrix.
        returns encoded matrix (dataframe)
        """
        self.encoder.fit(X, y)
        self.features = self.encode_features_list(X, features)
        X_new = self.encoder.fit_transform(X, y)
        X_new.columns = self._rename_columns(self.features)
        return X_new

    def get_mapping(self, category):
        """Gets the mapping for the LeaveOneOut encoder. Only takes strings of the column name, not the index number.
        returns mapping (dict)
        """
        return self.encoder.mapping[category]

    def encode_features_list(self, X, features):
        feature_list = []
        for f in features:
            if f.get_name() in self.encoder.cols:
                f = ft.Feature([f], primitive=LeaveOneOutEnc(self, f.get_name()))
            feature_list.append(f)
        return feature_list

    def _rename_columns(self, features):
        feature_names = []
        for feature in features:
            for fname in feature.get_feature_names():
                feature_names.append(fname)
        return feature_names

    def get_features(self):
        return self.features

    def get_name(self):
        return self.name
