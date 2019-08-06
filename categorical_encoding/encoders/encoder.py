from .encoder_methods import BinaryEncoder, HashingEncoder, OrdinalEncoder


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

    def __init__(self, method=OrdinalEncoder, to_encode=None):
        encoder_list = {'ordinal': OrdinalEncoder,
                        'binary': BinaryEncoder,
                        'hashing': HashingEncoder}
        if method in encoder_list:
            method = encoder_list[method]

        self.method = method(cols=to_encode)
        self.features = []

    def fit(self, X, y=None):
        self.method.fit(X, y)
        return self

    def transform(self, X, features):
        self.features = features
        self._encode_features_list(X)
        return self.method.transform(X, self.features)

    def fit_transform(self, X, features, y=None):
        return self.fit(X, y).transform(X, features)

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

    def _encode_features_list(self, X):
        self.features = self.method.encode_features_list(X, self.features)
