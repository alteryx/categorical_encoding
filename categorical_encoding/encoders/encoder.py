from .encoder_methods import (
    BinaryEncoder,
    HashingEncoder,
    OneHotEncoder,
    OrdinalEncoder
)


class Encoder():
    """
        Encodes specified columns of categorical values.

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
            gets the mapping for the encoder (binary, ordinal only)
        get_hash_method:
            gets the hash_method of the encoder (hashing only)
            return hash_method (str)
        get_n_components:
            gets the number of columns used in the encoder (hashing only)
            returns n_components (int)
    """

    def __init__(self, method='ordinal', to_encode=None):
        encoder_list = {'ordinal': OrdinalEncoder(cols=to_encode),
                        'binary': BinaryEncoder(cols=to_encode),
                        'hashing': HashingEncoder(cols=to_encode),
                        'one_hot': OneHotEncoder(cols=to_encode)}
        if method in encoder_list:
            method = encoder_list[method]
        elif isinstance(method, str):
            raise ValueError("'%s' is not a supported encoder. The list of supported String encoder method names is: %s" % (method, encoder_list.keys()))

        self.method = method
        self.features = []

    def fit(self, X, features, y=None):
        self.method.fit(X, y)
        self._encode_features_list(X, features)
        return self

    def transform(self, X):
        return self.method.transform(X, self.features)

    def fit_transform(self, X, features, y=None):
        return self.fit(X, features, y).transform(X)

    def get_features(self):
        return self.features

    def get_mapping(self, category=0):
        return self.method.get_mapping(category)

    def get_hash_method(self):
        if not isinstance(self.method, HashingEncoder):
            raise TypeError("Must be HashingEncoder")
        return self.method.hash_method

    def get_n_components(self):
        if not isinstance(self.method, HashingEncoder):
            raise TypeError("Must be HashingEncoder")
        return self.method.n_components

    def _encode_features_list(self, X, features):
        self.features = self.method.encode_features_list(X, features)
