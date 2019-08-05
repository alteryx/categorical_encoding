import featuretools as ft
from category_encoders import (
    BinaryEncoder,
    HashingEncoder,
    JamesSteinEncoder,
    LeaveOneOutEncoder,
    MEstimateEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder
)

from .ordinal_enc import OrdinalEnc
from .binary_enc import BinaryEnc


class Encoder():
    def __init__(self, method=OneHotEncoder, to_encode=None, top_n=10):
        encoder_list = {'ordinal': OrdinalEncoder,
                        'one_hot': OneHotEncoder,
                        'binary': BinaryEncoder,
                        'hashing': HashingEncoder,
                        'target': TargetEncoder,
                        'leave_one_out': LeaveOneOutEncoder,
                        'james-stein': JamesSteinEncoder,
                        'm_estimate': MEstimateEncoder}
        if method in encoder_list:
            method = encoder_list[method]

        self.method = method(cols=to_encode)
        self.features = []
        # top_n only for OneHotEncoder
        self.top_n = top_n

    def fit(self, X, features, y=None):
        self.features = features
        return self.method.fit(X, y)

    def transform(self, X):
        self._encode_features_list(X)
        return self.method.transform(X)

    def fit_transform(self, X, features, y=None):
        self.features = features
        self.fit(X=X, y=y, features=features)
        return self.transform(X)

    def get_features(self):
        return self.features
    
    # perhaps better name than col_index
    def get_mapping(self, col_index=0):
        if isinstance(self.method, BinaryEncoder):
            return self._get_mapping_helper(self.method.base_n_encoder, col_index), \
                    self._get_mapping_helper(self.method.base_n_encoder.ordinal_encoder, col_index)
        return self._get_mapping_helper(self.method, col_index)
    
    def _get_mapping_helper(self, method, col_index):
        if isinstance(col_index, str):
            for map in method.mapping:
                if map['col'] == col_index:
                    return map['mapping']
        return method.mapping[col_index]['mapping']
    
    def _encode_features_list(self, X):
        if isinstance(self.method, OneHotEncoder):
            self.features = self._encode_fl_one_hot(X)
        elif isinstance(self.method, OrdinalEncoder):
            self.features = self._encode_fl_ordinal(X)
        elif isinstance(self.method, BinaryEncoder):
            self.features = self._encode_fl_binary(X)
   
    def _encode_fl_one_hot(self, X):
        feature_list = []
        for f in self.features:
            if f.get_name() in self.method.cols:
                val_counts = X[f.get_name()].value_counts().to_frame()
                val_counts.sort_values(f.get_name(), ascending=False)
                unique = val_counts.index.tolist()
                for label in unique:
                    add = f == label
                    feature_list.append(add)
            else:
                feature_list.append(f)
        return feature_list
    
    def _encode_fl_ordinal(self, X):
        feature_list = []
        for f in self.features:
            index = 0
            if f.get_name() in self.method.cols:
                f = ft.Feature([f], primitive=OrdinalEnc(self, index))
                index += 1
            feature_list.append(f)
        return feature_list
    
    def _encode_fl_binary(self, X):
        feature_list = []
        for f in self.features:
            index = 0
            if f.get_name() in self.method.base_n_encoder.cols:
                f = ft.Feature([f], primitive=BinaryEnc(self, index))
                index += 1
            feature_list.append(f)
        return feature_list
