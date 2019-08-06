import featuretools as ft
from category_encoders import HashingEncoder as Hashing

from categorical_encoding.primitives import HashingEnc


class HashingEncoder():
    def __init__(self, cols, hash_method='md5', n_components=8):
        self.encoder = {}
        # may add hashing method here
        self.cols = cols
        self.hash_method = 'md5'
        self.n_components = n_components

    def fit(self, X, features, y=None):
        self.cols = Hashing(cols=self.cols, hash_method=self.hash_method, n_components=self.n_components).fit(X, y).cols
        for col in self.cols:
            self.encoder[col] = Hashing(cols=[col], hash_method=self.hash_method, n_components=self.n_components)
            self.encoder[col].fit(X[col], y)
        return self

    def transform(self, X, features):
        index = 0
        for col in self.cols:
            new_columns = self.encoder[col].transform(X[col])
            index = X.columns.get_loc(col)
            X.drop([col], axis=1, inplace=True)
            for col_enc in new_columns.iloc[:, ::-1]:
                X.insert(index, col_enc, new_columns[col_enc], allow_duplicates=True)

        feature_names = []
        for feature in features:
            for fname in feature.get_feature_names():
                feature_names.append(fname)
        X.columns = feature_names

        return X

    def fit_transform(self, X, features, y=None):
        return self.fit(X, features, y).transform(X, features)

    def encode_features_list(self, X, features):
        feature_list = []
        for f in features:
            if f.get_name() in self.cols:
                f = ft.Feature([f], primitive=HashingEnc(self))
            feature_list.append(f)
        return feature_list

    def get_hash_method(self):
        return self.hash_method

    def get_n_components(self):
        return self.n_components
