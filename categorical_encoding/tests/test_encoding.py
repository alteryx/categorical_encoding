import featuretools as ft

from .testing_utils import create_feature_matrix

from categorical_encoding.encoders import Encoder
from categorical_encoding.primitives import BinaryEnc, HashingEnc, OrdinalEnc


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

    feature_matrix = ft.calculate_feature_matrix(features, es, instance_ids=ids)
    assert (fm_encoded == feature_matrix).all().all()
