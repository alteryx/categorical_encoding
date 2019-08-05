import featuretools as ft
import pandas as pd
from featuretools.tests.testing_utils import make_ecommerce_entityset

from categorical_encoding.encoder import Encoder
from categorical_encoding import OrdinalEnc, BinaryEnc, HashingEnc

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
    fm_encoded = enc.fit_transform(feature_matrix, features)

    encoder = OrdinalEnc(fitted_encoder=enc, category='product_id')
    encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
    encoded_results = [2, 3, 1, 1]
    assert (encoded == encoded_results).all()

    product_feature = ft.Feature([f1], primitive=OrdinalEnc(enc, 0))
    cc_feature = ft.Feature([f4], primitive=OrdinalEnc(enc, 1))
    features = [product_feature, f2, f3, cc_feature]
    assert features == enc.get_features()
    
    feature_matrix_new = ft.calculate_feature_matrix(features, es, instance_ids=ids)
    assert (fm_encoded == feature_matrix_new).all().all()


def test_binary_encoding():
    feature_matrix, features, f1, f2, f3, f4, es, ids = create_feature_matrix()
    
    enc = Encoder(method='binary')
    fm_encoded = enc.fit_transform(feature_matrix, features)
    
    encoder = BinaryEnc(fitted_encoder=enc, category='product_id')
    encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
    result = [[0, 0, 0, 0],
              [1, 1, 0, 0],
              [0, 1, 1, 1]]
    assert (encoded == result).all()
    
    product_feature = ft.Feature([f1], primitive=BinaryEnc(enc, 0))
    cc_feature = ft.Feature([f4], primitive=BinaryEnc(enc, 1))
    features = [product_feature, f2, f3, cc_feature]
    assert len(features) == len(enc.get_features())
    # __eq__ does not support multioutput columns yet
    for i in range(len(enc.get_features())):
        assert features[i].unique_name() == enc.get_features()[i].unique_name()
    
    feature_matrix = ft.calculate_feature_matrix(features, es, instance_ids=ids)
    assert (fm_encoded == feature_matrix).all().all()

def test_hashing_encoding():
    feature_matrix, features, f1, f2, f3, f4, es, ids = create_feature_matrix()
    
    enc = Encoder(method='hashing')
    enc.fit_transform(feature_matrix, features)
    
    encoder = HashingEnc(fitted_encoder=enc)
    encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
    result = [[0, 0, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 1],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]
    assert (encoded == result).all()
    
    product_feature = ft.Feature([f1], primitive=HashingEnc(enc))
    cc_feature = ft.Feature([f4], primitive=HashingEnc(enc))
    features = [product_feature, f2, f3, cc_feature]
    assert len(features) == len(enc.get_features())
    for i in range(len(features)):
        assert features[i].unique_name() == enc.get_features()[i].unique_name()

    feature_matrix = ft.calculate_feature_matrix(features, es, instance_ids=ids)    
    