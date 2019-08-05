import featuretools as ft
import pandas as pd
from featuretools.tests.testing_utils import make_ecommerce_entityset

from categorical_encoding.encoder import Encoder
from categorical_encoding import OrdinalEnc, BinaryEnc

def create_feature_matrix():
    es = make_ecommerce_entityset()
    f1 = ft.Feature(es["log"]["product_id"])
    f2 = ft.Feature(es["log"]["purchased"])
    f3 = ft.Feature(es["log"]["value"])
    f4 = ft.Feature(es["log"]["countrycode"])
    features = [f1, f2, f3, f4]
    ids = [0, 1, 2, 3, 4, 5]
    feature_matrix = ft.calculate_feature_matrix(features, es, instance_ids=ids)
    
    return feature_matrix, features, f1, f2, f3, f4, es, ids

def test_ordinal_encoding():
    feature_matrix, features, f1, f2, f3, f4, es, ids = create_feature_matrix()
    
    enc = Encoder(method='ordinal')
    enc.fit_transform(feature_matrix, features)

    encoder = OrdinalEnc(fitted_encoder=enc, category='product_id')
    encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
    encoded_results = [2, 3, 1, 1]
    assert (encoded == encoded_results).all()

    product_feature = ft.Feature([f1], primitive=OrdinalEnc(enc, 0))
    cc_feature = ft.Feature([f4], primitive=OrdinalEnc(enc, 1))
    features = [product_feature, f2, f3, cc_feature]
    assert features == enc.get_features()
    
    feature_matrix = ft.calculate_feature_matrix(features, es, instance_ids=ids)
    result = pd.DataFrame([[1, True, 0.0, 1],
                           [1, True, 5.0, 1],
                           [1, True, 10.0, 1],
                           [2, True, 15.0, 1],
                           [2, True, 20.0, 1],
                           [3, True, 0.0, 2]],
                          columns=['PRODUCT_ID_ordinal', 'purchased', 'value', 'COUNTRYCODE_ordinal'])
    assert (result == feature_matrix).all().all()


def test_binary_encoding():
    feature_matrix, features, f1, f2, f3, f4, es, ids = create_feature_matrix()
    
    enc = Encoder(method='binary')
    enc.fit_transform(feature_matrix, features)
    
    encoder = BinaryEnc(fitted_encoder=enc, category='product_id')
    encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
    result = [[0, 0, 0, 0],
              [1, 1, 0, 0],
              [0, 1, 1, 1]]
    assert encoded == result
    
    product_feature = ft.Feature([f1], primitive=BinaryEnc(enc, 0))
    cc_feature = ft.Feature([f4], primitive=BinaryEnc(enc, 1))
    features = [product_feature, f2, f3, cc_feature]
    assert len(features) == len(enc.get_features())
    for i in range(len(features)):
        assert features[i].unique_name() == enc.get_features()[i].unique_name()
              
    # __eq__ does not support this multioutput columns yet
    
    feature_matrix = ft.calculate_feature_matrix(features, es, instance_ids=ids)
    fm_result = pd.DataFrame([[0, 0, 1, True, 0.0, 0, 1],
                              [0, 0, 1, True, 5.0, 0, 1],
                              [0, 0, 1, True, 10.0, 0, 1],
                              [0, 1, 0, True, 15.0, 0, 1],
                              [0, 1, 0, True, 20.0, 0, 1],
                              [0, 1, 1, True, 0.0, 1, 0]],
                              columns=['PRODUCT_ID_binary__0', 'PRODUCT_ID_binary__1', 'PRODUCT_ID_binary__2', 'purchased', 'value', 'COUNTRYCODE_binary__0', 'COUNTRYCODE_binary__1'])
    assert (fm_result == feature_matrix).all().all()
