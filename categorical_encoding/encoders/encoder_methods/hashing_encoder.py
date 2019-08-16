import featuretools as ft
from category_encoders import HashingEncoder as Hashing

from categorical_encoding.primitives import HashingEnc


class HashingEncoder():
    """Maps each categorical value to several columns using a specific hash function.

    Parameters:
        cols: [str]
            list of column names to encode.
        hash_method: str
            str for hash_method name to use. Any method from hashlib works.
        n_components: int
            integer for the number of columns to map to.
    """
    name = 'hashing'

    def __init__(self, cols=None, hash_method='md5', n_components=8):
        self.encoder = {}
        self.cols = cols
        self.hash_method = 'md5'
        self.n_components = n_components

    def fit(self, X, features, y=None):
        """Fits encoder to data table.
        returns self
        """
        self.cols = Hashing(cols=self.cols, hash_method=self.hash_method, n_components=self.n_components).fit(X, y).cols
        for col in self.cols:
            self.encoder[col] = Hashing(cols=[col], hash_method=self.hash_method, n_components=self.n_components)
            self.encoder[col].fit(X[col], y)
        self.features = self.encode_features_list(X, features)
        return self

    def transform(self, X):
        """Encodes matrix and updates features accordingly.
        returns encoded matrix (dataframe)
        """
        X_new = X.copy()
        index = 0
        for col in self.cols:
            new_columns = self.encoder[col].transform(X[col])
            index = X_new.columns.get_loc(col)
            X_new.drop([col], axis=1, inplace=True)
            for col_enc in new_columns.iloc[:, ::-1]:
                X_new.insert(index, col_enc, new_columns[col_enc], allow_duplicates=True)

        feature_names = []
        for feature in self.features:
            for fname in feature.get_feature_names():
                feature_names.append(fname)
        X_new.columns = feature_names

        return X_new

    def fit_transform(self, X, features, y=None):
        """Fits, then transforms matrix.
        returns encoded matrix (dataframe)
        """
        return self.fit(X, features, y).transform(X)

    def encode_features_list(self, X, features):
        feature_list = []
        for f in features:
            if f.get_name() in self.cols:
                f = ft.Feature([f], primitive=HashingEnc(self))
            feature_list.append(f)
        return feature_list

    def get_hash_method(self):
        """Gets the hash_method of the encoder.
        return hash_method (str)"""
        return self.hash_method

    def get_n_components(self):
        """Gets the number of columns used in the encoder.
        returns n_components (int)"""
        return self.n_components

    def get_features(self):
        return self.features

    def get_name(self):
        return self.name
