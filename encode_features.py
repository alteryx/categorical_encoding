import category_encoders as ce
'''
Things to implement still:
verbose
inplace
top_n? (don't see how this is necessary for anything besides OneHot). we could do this still, but what should we default to?
drop_first?
returning encoded feature names (don't think this is necessary without top_n
because we know all the features will be encoded)
'''

def encode_features(feature_matrix, features, top_n=10, include_unknown=True,
                    to_encode=None, inplace=False, drop_first=False, verbose=False,
                    encoder=None, target_value=None, is_fitted=False):
    # for bayesian encoders (or possibly other), if already fitted, we'd want to use
    # the fitted encoder to transform
    if is_fitted:
        fm_encoded = encoder.transform(feature_matrix)
        return fm_encoded, features, encoder
    
    classic_encoders = {'Ordinal': ce.OrdinalEncoder(cols = to_encode),
                        'OneHot': ce.OneHotEncoder(cols = to_encode),
                        'Binary': ce.BinaryEncoder(cols = to_encode),
                        'Hashing': ce.HashingEncoder(cols = to_encode)}
    bayesian_encoders = {'Target': ce.TargetEncoder(cols = to_encode),
                         'LeaveOneOut': ce.LeaveOneOutEncoder(cols = to_encode),
                         'James-Stein': ce.JamesSteinEncoder(cols = to_encode),
                         'MEstimate': ce.MEstimateEncoder(cols = to_encode)}
    is_bayesian = False
    
    if isinstance(encoder, str):
        if encoder in classic_encoders:
            encoder = classic_encoders[encoder]
        if encoder in bayesian_encoders:
            encoder = bayesian_encoders[encoder]
            is_bayesian = True
    
    if isinstance(encoder, str):
        # raise error about incorrect category_encoder
    
    if is_bayesian:
        fm_encoded = encoder.fit_transform(feature_matrix, target_value)
    else:
        fm_encoded = encoder.fit_transform(feature_matrix)
    
    # need to handle inplace somewhere somehow
    
    # previously we're handling feature names and how they change with encoding
    # this does not seem as necessary for anything except one-hot encoder
    # also brings up the question, is top_n necessary for the new encode_features
    
    return fm_encoded, features, encoder

'''
def transform_features():
    then, perhaps the functions wouldn't get as confused if we need to call it twice for
    encoding a feature matrix here
'''