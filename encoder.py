from category_encoders import (
    OneHotEncoder,
    OrdinalEncoder,
    BinaryEncoder,
    HashingEncoder,
    TargetEncoder,
    LeaveOneOutEncoder,
    JamesSteinEncoder,
    MEstimateEncoder
)
# instead of using category_encoders, perhaps we should wrap it ourselves in order for method to work

class Encoder():
    def __init__(self, method=OneHotEncoder, to_encode=None, top_n=10):
        encoder_list = {'Ordinal': OrdinalEncoder,
                        'OneHot': OneHotEncoder,
                        'Binary': BinaryEncoder,
                        'Hashing': HashingEncoder,
                        'Target': TargetEncoder,
                        'LeaveOneOut': LeaveOneOutEncoder,
                        'James-Stein': JamesSteinEncoder,
                        'MEstimate': MEstimateEncoder}
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
        feature_list = []
        if isinstance(self.method, OneHotEncoder):
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
        else:
            for f in self.features:
                if f.get_name() in self.method.cols: # this isn't supporting multiple output features yet
                    f = f.rename(f.get_name() + '_' + str(self.method)[:str(self.method).find('Encoder')].lower())
                print(f)
                feature_list.append(f)
            
        self.features = feature_list
        
        return self.method.transform(X)
    
    def fit_transform(self, X, features, y=None):
        self.features = features
        self.fit(X=X, y=y, features=features)
        return self.transform(X)
    
    def get_features(self):
        return self.features
    
# potential challenge in calculate feature matrix from feature list
'''
from featuretools.encoding import OneHotEncoding, TargetEncoding, HashEncoding, BinaryEncoding

fm_train, fm_test = split(fm)


# option 1
fm_enc, fl_enc = ft.encode_features(fm_train, fl)
fm_enc, fl_enc = ft.encode_features(fm_train, fl, method=TargetEncoding(target_col="label", alpha=.5))


fm_enc, fl_enc, fm_test_enc = ft.encode_features(
    fm_train,
    fl,
    use_cols=["cat_1", "cat_2", "cat_3"],
    method=[(TargetEncoding(target_col="label", alpha=.5), ["cat_1"]),
            OneHotEncoding(top_n=3), #apply to all, limited by use_cols
            (TargetEncoding(target_col="label", alpha=.3), ["cat_1", "cat_2"])]
    transform_data=fm_test
)
fl.save_features(fl_enc, "features.json")


# option 2
enc = ft.Encoder(method=TargetEncoding(alpha=.5), leave_one_out=True, k_folds=5)
enc = ft.Encoder(method='onehot', leave_one_out=True, k_folds=5)
fm_train_enc = enc.fit_transform(X=fm_train, features=fl, y=y_train)
fm_test_enc = enc.transform(fm_test)
fl_enc = enc.get_encoded_features() # only works if FL is provided
fl.save_features(fl_enc, "features.json")


# transform
fl_enc = fl.load_features("features.json")
ft.calculate_feature_matrix(
    features=fl_enc,
    entity_set=es_prod
)
'''