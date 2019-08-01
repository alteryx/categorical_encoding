import numpy as np
import pandas as pd
import pytest

import featuretools as ft
from categorical_encoding.ordinal_enc import OrdinalEnc
import category_encoders as ce
from featuretools.tests.testing_utils import make_ecommerce_entityset

def test_ordinal_encoding():
    es = make_ecommerce_entityset()
    f1 = ft.Feature(es["log"]["product_id"])
    f2 = ft.Feature(es["log"]["purchased"])
    f3 = ft.Feature(es["log"]["value"])
    f4 = ft.Feature(es["log"]["countrycode"])

    features = [f1, f2, f3, f4]
    ids = [0, 1, 2, 3, 4, 5]
    feature_matrix = ft.calculate_feature_matrix(features, es, instance_ids=ids)

    enc = ce.OrdinalEncoder()
    enc.fit_transform(feature_matrix)
    mapping = enc.mapping[0]['mapping']

    encoder = OrdinalEnc(mapping)
    encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
    encoded_results = [2, 3, 1, 1]
    assert (encoded == encoded_results).all()

    product_feature = ft.Feature([f1], primitive=OrdinalEnc(mapping))
    cc_feature = ft.Feature([f4], primitive=OrdinalEnc(enc.mapping[1]['mapping']))
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

