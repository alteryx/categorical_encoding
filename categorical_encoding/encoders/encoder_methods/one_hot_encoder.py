import logging

import featuretools as ft
import numpy as np
from category_encoders import OneHotEncoder as OneHot

from categorical_encoding.primitives import OneHotEnc

logger = logging.getLogger('featuretools')


class OneHotEncoder():
    def __init__(self, cols, top_n=15):
        self.encoder = OneHot(cols=cols)
        self.matrix = None
        self.top_n = top_n

    def fit(self, X, y=None):
        self.encoder.fit(X, y=None)
        return self

    def transform(self, X, features=None):
        assert(self.matrix is not None), "Check that the encoder is fitted."
        return self.matrix

    def fit_transform(self, X, features=None, y=None):
        return self.fit(X, y).transform(X)

    def get_mapping(self, category):
        if isinstance(category, str):
            for map in self.encoder.mapping:
                if map['col'] == category:
                    return map['mapping']
        return self.encoder.mapping[category]['mapping']

    def encode_features_list(self, X, features):
        X_new = X.copy()
        feature_list = []
        for f in features:
            if f.number_output_features > 1:
                logger.warning("Feature %s has multiple columns. One-Hot Encoder may not properly encode."
                               "Consider using another encoding method or the `encoder` property value assigned "
                               "to this OneHotEncoder class instance." % (f))
            if f.get_name() in self.encoder.cols:
                val_counts = X[f.get_name()].value_counts().to_frame()
                val_counts.sort_values(f.get_name(), ascending=False)
                unique = val_counts.head(self.top_n).index.tolist()

                index = X_new.columns.get_loc(f.get_name())
                for label in unique:
                    add = ft.Feature([f], primitive=OneHotEnc(label))
                    feature_list.append(add)
                    X_new.insert(index, add.get_name(), (X_new[f.get_name()] == label).astype(int), allow_duplicates=True)
                    index += 1
                has_unknown = X[f.get_name()].isnull().values.any()
                if has_unknown:
                    unknown = ft.Feature([f], primitive=OneHotEnc(np.nan))
                    feature_list.append(unknown)
                    X_new.insert(index, unknown.get_name(), (~X_new[f.get_name()].isin(unique)).astype(int), allow_duplicates=True)
                X_new.drop([f.get_name()], axis=1, inplace=True)
            else:
                feature_list.append(f)
        self.matrix = X_new
        return feature_list
