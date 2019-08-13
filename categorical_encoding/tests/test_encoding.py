import featuretools as ft
import numpy as np
import pandas as pd

from .testing_utils import create_feature_matrix

from categorical_encoding.encoders import Encoder
from categorical_encoding.primitives import (
    BinaryEnc,
    HashingEnc,
    LeaveOneOutEnc,
    OneHotEnc,
    OrdinalEnc,
    TargetEnc
)


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

    features = enc.get_features()
    feature_matrix_new = ft.calculate_feature_matrix(features, es, instance_ids=ids)
    assert (fm_encoded == feature_matrix_new).all().all()


def test_binary_encoding():
    feature_matrix, features, f1, f2, f3, f4, es, ids = create_feature_matrix()

    enc = Encoder(method='binary')
    fm_encoded = enc.fit_transform(feature_matrix, features)

    encoder = BinaryEnc(fitted_encoder=enc, category='product_id')
    encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
    encoded_results = [[0, 0, 0, 0],
                       [1, 1, 0, 0],
                       [0, 1, 1, 1]]
    assert (encoded == encoded_results).all()

    product_feature = ft.Feature([f1], primitive=BinaryEnc(enc, 0))
    cc_feature = ft.Feature([f4], primitive=BinaryEnc(enc, 1))
    features = [product_feature, f2, f3, cc_feature]
    assert len(features) == len(enc.get_features())
    # __eq__ does not support multioutput columns yet
    for i in range(len(enc.get_features())):
        assert features[i].unique_name() == enc.get_features()[i].unique_name()

    features = enc.get_features()
    feature_matrix = ft.calculate_feature_matrix(features, es, instance_ids=ids)
    assert (fm_encoded == feature_matrix).all().all()


def test_hashing_encoding():
    feature_matrix, features, f1, f2, f3, f4, es, ids = create_feature_matrix()

    enc = Encoder(method='hashing')
    fm_encoded = enc.fit_transform(feature_matrix, features)

    encoder = HashingEnc(fitted_encoder=enc)
    encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
    encoded_results = [[0, 0, 0, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 1],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]]
    assert (encoded == encoded_results).all()

    product_feature = ft.Feature([f1], primitive=HashingEnc(enc))
    cc_feature = ft.Feature([f4], primitive=HashingEnc(enc))
    features = [product_feature, f2, f3, cc_feature]
    assert len(features) == len(enc.get_features())
    for i in range(len(features)):
        assert features[i].unique_name() == enc.get_features()[i].unique_name()

    features = enc.get_features()
    feature_matrix = ft.calculate_feature_matrix(features, es, instance_ids=ids)
    assert (fm_encoded == feature_matrix).all().all()


def test_one_hot_encoding():
    feature_matrix, features, f1, f2, f3, f4, es, ids = create_feature_matrix()

    feature_matrix['countrycode'][0] = np.nan
    enc = Encoder(method='one_hot')
    fm_encoded = enc.fit_transform(feature_matrix, features)

    encoder = OneHotEnc(value='coke zero')
    encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
    encoded_results = [0, 0, 1, 1]
    assert (encoded == encoded_results).all()

    encoder = OneHotEnc(value=np.nan)
    encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero', np.nan])
    encoded_results = [0, 0, 0, 0, 1]
    assert (encoded == encoded_results).all()

    f1_1 = ft.Feature([f1], primitive=OneHotEnc('coke zero'))
    f1_2 = ft.Feature([f1], primitive=OneHotEnc('car'))
    f1_3 = ft.Feature([f1], primitive=OneHotEnc('toothpaste'))

    f4_1 = ft.Feature([f4], primitive=OneHotEnc('US'))
    f4_2 = ft.Feature([f4], primitive=OneHotEnc('AL'))
    f4_3 = ft.Feature([f4], primitive=OneHotEnc(np.nan))
    features_encoded = [f1_1, f1_2, f1_3, f2, f3, f4_1, f4_2, f4_3]
    assert len(features_encoded) == len(enc.get_features())
    for i in range(len(features_encoded)):
        assert features_encoded[i].unique_name() == enc.get_features()[i].unique_name()

    features_encoded = enc.get_features()
    feature_matrix = ft.calculate_feature_matrix(features_encoded, es, instance_ids=[6, 7])
    data = {'product_id = coke zero': [0, 0],
            'product_id = car': [0, 0],
            'product_id = toothpaste': [1, 1],
            'purchased': [True, True],
            'value': [1.0, 2.0],
            'countrycode = US': [0, 0],
            'countrycode = AL': [1, 1],
            'countrycode = nan': [0, 0]}
    fm_encoded = pd.DataFrame(data, index=[6, 7])
    assert feature_matrix.eq(fm_encoded).all().all()


def test_target_encoding():
    feature_matrix, features, f1, f2, f3, f4, es, ids = create_feature_matrix()

    enc = Encoder(method='target')
    fm_encoded = enc.fit_transform(feature_matrix, features, feature_matrix['value'])

    encoder = TargetEnc(fitted_encoder=enc, category='product_id')
    encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
    encoded_results = [15.034704, 8.333333, 5.397343, 5.397343]
    np.testing.assert_almost_equal(encoded, encoded_results, decimal=5)

    product_feature = ft.Feature([f1], primitive=TargetEnc(enc, 'product_id'))
    cc_feature = ft.Feature([f4], primitive=TargetEnc(enc, 'countrycode'))
    features = [product_feature, f2, f3, cc_feature]
    assert features == enc.get_features()

    features = enc.get_features()
    feature_matrix_new = ft.calculate_feature_matrix(features, es, instance_ids=ids)
    assert (fm_encoded == feature_matrix_new).all().all()


def test_leave_one_out_encoding():
    feature_matrix, features, f1, f2, f3, f4, es, ids = create_feature_matrix()

    enc = Encoder(method='leave_one_out')
    fm_encoded = enc.fit_transform(feature_matrix, features, feature_matrix['value'])
    fm_encoded_result = [[7.50001, 5.00001, 2.50001, 20.00001, 15.00001, 8.33333],
                         [True, True, True, True, True, True],
                         [0.00001, 5.00001, 10.00001, 15.00001, 20.00001, 0.00001],
                         [12.50001, 11.250001, 10.00001, 8.750001, 7.50001, 8.33333]]
    fm_encoded_result = np.swapaxes(fm_encoded_result, 0, 1)
    np.testing.assert_almost_equal(fm_encoded.values, fm_encoded_result, decimal=1)

    encoder = LeaveOneOutEnc(fitted_encoder=enc, category='product_id')
    encoded = encoder(['car', 'toothpaste', 'coke zero', 'coke zero'])
    encoded_results = [17.5, 8.33333, 5, 5]
    np.testing.assert_almost_equal(encoded, encoded_results, decimal=4)

    product_feature = ft.Feature([f1], primitive=LeaveOneOutEnc(enc, 'product_id'))
    cc_feature = ft.Feature([f4], primitive=LeaveOneOutEnc(enc, 'countrycode'))
    features = [product_feature, f2, f3, cc_feature]
    assert features == enc.get_features()

    features = enc.get_features()
    feature_matrix_new = ft.calculate_feature_matrix(features, es, instance_ids=ids)
    new_data = [[5.00001, 5.00001, 5.00001, 17.50001, 17.50001, 8.33333],
                [True, True, True, True, True, True],
                [0.00001, 5.00001, 10.00001, 15.00001, 20.00001, 0.00001],
                [10.00001, 10.00001, 10.00001, 10.00001, 10.00001, 8.33333]]
    new_result = np.swapaxes(new_data, 0, 1)
    np.testing.assert_almost_equal(feature_matrix_new.values, new_result, decimal=1)
