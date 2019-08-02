import featuretools as ft
import pandas as pd
from featuretools.tests.testing_utils import make_ecommerce_entityset

from categorical_encoding.encoder import Encoder
from categorical_encoding import OrdinalEnc, BinaryEnc

def test_ordinal_encoding():
    es = make_ecommerce_entityset()
    f1 = ft.Feature(es["log"]["product_id"])
    f2 = ft.Feature(es["log"]["purchased"])
    f3 = ft.Feature(es["log"]["value"])
    f4 = ft.Feature(es["log"]["countrycode"])
    features = [f1, f2, f3, f4]
    ids = [0, 1, 2, 3, 4, 5]
    feature_matrix = ft.calculate_feature_matrix(features, es, instance_ids=ids)
    
    enc = Encoder(method='ordinal')
    enc.fit_transform(feature_matrix, features)
    mapping = enc.get_mapping('product_id')

    encoder = OrdinalEnc(mapping)
    encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
    encoded_results = [2, 3, 1, 1]
    assert (encoded == encoded_results).all()

    product_feature = ft.Feature([f1], primitive=OrdinalEnc(mapping))
    cc_feature = ft.Feature([f4], primitive=OrdinalEnc(enc.get_mapping(1)))
    features = [product_feature, f2, f3, cc_feature]
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
    es = make_ecommerce_entityset()
    f1 = ft.Feature(es["log"]["product_id"])
    f2 = ft.Feature(es["log"]["purchased"])
    f3 = ft.Feature(es["log"]["value"])
    f4 = ft.Feature(es["log"]["countrycode"])
    features = [f1, f2, f3, f4]
    ids = [0, 1, 2, 3, 4, 5]
    feature_matrix = ft.calculate_feature_matrix(features, es, instance_ids=ids)
    
    enc = Encoder(method='binary')
    enc.fit_transform(feature_matrix, features)
    mapping, mapping_ord = enc.get_mapping('product_id')
    
    encoder = BinaryEnc(mapping=mapping, mapping_ord=mapping_ord)
    encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
    result = [[0, 0, 0, 0],
              [1, 1, 0, 0],
              [0, 1, 1, 1]]
    assert encoded == result
    
    product_feature = ft.Feature([f1], primitive=BinaryEnc(mapping, mapping_ord))
    mapping_cc, mapping_ord_cc = enc.get_mapping(1)
    # this returns a tuple, perhaps we can work with that tuple within the function
    # that would retain API functionality but may make it more confusing
    cc_feature = ft.Feature([f4], primitive=BinaryEnc(mapping_cc, mapping_ord_cc))
    features = [product_feature, f2, f3, cc_feature]
    feature_matrix = ft.calculate_feature_matrix(features, es, instance_ids=ids)
    fm_result = pd.DataFrame([[0, 0, 1, True, 0.0, 0, 1],
                              [0, 0, 1, True, 5.0, 0, 1],
                              [0, 0, 1, True, 10.0, 0, 1],
                              [0, 1, 0, True, 15.0, 0, 1],
                              [0, 1, 0, True, 20.0, 0, 1],
                              [0, 1, 1, True, 0.0, 1, 0]],
                              columns=['PRODUCT_ID_binary__0', 'PRODUCT_ID_binary__1', 'PRODUCT_ID_binary__2', 'purchased', 'value', 'COUNTRYCODE_binary__0', 'COUNTRYCODE_binary__1'])
    assert (fm_result == feature_matrix).all().all()
