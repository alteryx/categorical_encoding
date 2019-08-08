import featuretools as ft
import numpy as np
from category_encoders import OneHotEncoder as OneHot

from categorical_encoding.primitives import OneHotEnc


class OneHotEncoder():
    def __init__(self, cols):
        self.encoder = OneHot(cols=cols)

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
        for f in features:
            if f.get_name() in self.encoder.cols:
                val_counts = X[f.get_name()].value_counts().to_frame()
                val_counts.sort_values(f.get_name(), ascending=False)
                unique = val_counts.index.tolist()
                for label in unique:
                    add = ft.Feature([f], primitive=OneHotEnc(label))
                    feature_list.append(add)
                has_unknown = X[f.get_name()].isnull().values.any()
                if has_unknown:
                    unknown = ft.Feature([f], primitive=OneHotEnc(np.nan))
                    feature_list.append(unknown)
            else:
                feature_list.append(f)
        return feature_list
