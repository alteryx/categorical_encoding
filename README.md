# categorical-encoding

![Tests](https://github.com/FeatureLabs/categorical_encoding/workflows/Tests/badge.svg)


categorical-encoding is a Python library for encoding categorical data, intended for use with [Featuretools](https://github.com/Featuretools/featuretools).
categorical-encoding allows for seamless encoding of data and integration into Featuretools pipeline for automated feature engineering within the machine learning pipeline.

### Install
```shell
python -m pip install "featuretools[categorical_encoding]"
```

### Description

#### Install Demo Guide Requirements
```shell
python -m pip install demo-requirements.txt
```

For more general questions regarding how to use categorical encoding in a machine learning pipeline, consult the guides located in the [categorical encoding github repository](https://github.com/FeatureLabs/categorical_encoding/tree/main/guides).

```py
>>> feature_matrix
    product_id  purchased  value countrycode
id
0    coke zero       True    0.0          US
1    coke zero       True    5.0          US
2    coke zero       True   10.0          US
3          car       True   15.0          US
4          car       True   20.0          US
5   toothpaste       True    0.0          AL
```
Integrates into standard procedure of train/test split within applied machine learning processes.
```py
>>> train_data = feature_matrix.iloc[[0, 1, 4, 5]]
>>> train_data
    product_id  purchased  value countrycode
id
0    coke zero       True    0.0          US
1    coke zero       True    5.0          US
4          car       True   20.0          US
5   toothpaste       True    0.0          AL
>>> test_data = feature_matrix.iloc[[2, 3]]
>>> test_data
   product_id  purchased  value countrycode
id
2   coke zero       True   10.0          US
3         car       True   15.0          US
```
```py
>>> import categorical_encoding as ce
>>> encoder = ce.Encoder(method='leave_one_out')
>>> train_enc = encoder.fit_transform(train_data, features, train_data['value'])
>>> test_enc = encoder.transform(test_data)
```
Encoder fits and transforms to train data, and then transforms test data using its learned fitted encoding.
```py
>>> train_enc
    PRODUCT_ID_leave_one_out  purchased  value  COUNTRYCODE_leave_one_out
id
0                       5.00       True    0.0                      12.50
1                       0.00       True    5.0                      10.00
4                       6.25       True   20.0                       2.50
5                       6.25       True    0.0                       6.25
>>> test_enc
    PRODUCT_ID_leave_one_out  purchased  value  COUNTRYCODE_leave_one_out
id
2                       2.50       True   10.0                   8.333333
3                       6.25       True   15.0                   8.333333
```
Supports easy integration into Featuretools through its support and use of features.
First, learn features through fitting an encoder to data. Then, when new data comes in, easily prepare it for your trained machine learning model by using those features to seamlessly generate new tables of encoded data.
```py
>>> features = encoder.get_features()
[<Feature: PRODUCT_ID_leave_one_out>,
 <Feature: purchased>,
 <Feature: value>,
 <Feature: COUNTRYCODE_leave_one_out>]
>>> features_encoded = enc.get_features()
>>> fm2_encoded = ft.calculate_feature_matrix(features_encoded, es, instance_ids=[6,7])
>>> fm2_encoded
    PRODUCT_ID_leave_one_out  purchased  value  COUNTRYCODE_leave_one_out
id
6                       6.25       True    1.0                       6.25
7                       6.25       True    2.0                       6.25
```


## Built at Alteryx Innovation Labs

<a href="https://www.alteryx.com/innovation-labs">
    <img src="https://evalml-web-images.s3.amazonaws.com/alteryx_innovation_labs.png" alt="Alteryx Innovation Labs" />
</a>
