# categorical-encoding

[![CircleCI](https://circleci.com/gh/FeatureLabs/categorical_encoding/tree/master.svg?style=shield)](https://circleci.com/gh/FeatureLabs/categorical_encoding/tree/master)
[![codecov](https://codecov.io/gh/FeatureLabs/categorical_encoding/branch/master/graph/badge.svg)](https://codecov.io/gh/FeatureLabs/categorical_encoding)
[![Documentation Status](https://readthedocs.org/projects/categorical_encoding/badge/?version=latest)](http://docs.compose.ml/en/latest/?badge=latest)

categorical-encoding is a Python library for encoding categorical data, intended for use with [Featuretools](https://github.com/Featuretools/featuretools). 
categorical-encoding allows for easy encoding of data and integration into Featuretools pipeline for automated feature engineering within the machine learning pipeline.

### Install
```shell
python -m pip install "featuretools[categorical_encoding]"
```

### Description
For more general questions regarding how to use categorical encoding in a machine learning pipeline, consult the guides located in the [categorical encoding github repository](https://github.com/FeatureLabs/categorical_encoding/tree/master/guides).

```py
import categorical_encoding as ce

encoder = ce.Encoder()
encoder.fit(feature_matrix, features)
fm_encoded = encoder.transform(feature_matrix, features)
```
feature_matrix
```py
    product_id  purchased  value countrycode
id                                          
0    coke zero       True    0.0          US
1    coke zero       True    5.0          US
2    coke zero       True   10.0          US
3          car       True   15.0          US
4          car       True   20.0          US
5   toothpaste       True    0.0          AL
```
fm_encoded
```py
    PRODUCT_ID_ordinal  purchased  value  COUNTRYCODE_ordinal
id                                                           
0                    1       True    0.0                    1
1                    1       True    5.0                    1
2                    1       True   10.0                    1
3                    2       True   15.0                    1
4                    2       True   20.0                    1
5                    3       True    0.0                    2
```
Supports easy integration into Featuretools through its support and use of features.
Learn features through fitting an encoder to data, and then use those features to easily generate new tables of encoded data.
```
>>> features = encoder.get_features()
[<Feature: PRODUCT_ID_ordinal>,
 <Feature: purchased>,
 <Feature: value>,
 <Feature: COUNTRYCODE_ordinal>]
>>> feature_matrix_2 = ft.calculate_feature_matrix(features, es)
```

## Feature Labs
<a href="https://www.featurelabs.com/">
    <img src="http://www.featurelabs.com/wp-content/uploads/2017/12/logo.png" alt="Featuretools" />
</a>

categorical-encoding is an open source project created by [Feature Labs](https://www.featurelabs.com/). To see the other open source projects we're working on visit Feature Labs [Open Source](https://www.featurelabs.com/open). If building impactful data science pipelines is important to you or your business, please [get in touch](https://www.featurelabs.com/contact/).
