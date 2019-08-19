import featuretools as ft
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
import categorical_encoding as ce
import time

def load_entityset(data_dir):
    # We first load our data.
    air_reserve = pd.read_csv(os.path.join(data_dir, "air_reserve.csv"))
    hpg_reserve = pd.read_csv(os.path.join(data_dir, "hpg_reserve.csv"))
    air_store_info = pd.read_csv(os.path.join(data_dir, "air_store_info.csv"))
    hpg_store_info = pd.read_csv(os.path.join(data_dir, "hpg_store_info.csv"))
    store_id_relation = pd.read_csv(os.path.join(data_dir, 'store_id_relation.csv'))
    air_visit_data = pd.read_csv(os.path.join(data_dir, "air_visit_data.csv"))
    date_info = pd.read_csv(os.path.join(data_dir, "date_info.csv"))
    
    # First, we want to save the id relations between air and hpg 
    # so that we can link them in the same table together later.
    air_store_info = air_store_info.merge(store_id_relation,
                                     on='air_store_id',
                                     how='outer')
    # We rename the hpg_store's columns so it doesn't conflict with the air_store.
    hpg_store_info = hpg_store_info.rename(columns={'latitude': 'hpg_latitude',
                                                'longitude': 'hpg_longitude'})
    # For some reason, there are stores in the reservations table that are not in the store table, 
    # so we fix that quickly right now.
    hpg_store_info = hpg_store_info.merge(hpg_reserve[['hpg_store_id']].drop_duplicates(), how='outer')
    # We merge the two data tables for air/hpg together.
    store_info = air_store_info.merge(hpg_store_info,
                                  on='hpg_store_id',
                                  how='outer')
    # We want to fill missing air information with hpg information.
    store_info['store_id'] = store_info['air_store_id'].fillna(store_info['hpg_store_id'])
    store_info['latitude'] = store_info['latitude'].fillna(store_info['hpg_latitude'])
    store_info['longitude'] = store_info['longitude'].fillna(store_info['hpg_longitude'])
    store_info['area_name'] = store_info['air_area_name'].fillna(store_info['hpg_area_name'])
    store_info['genre_name'] = store_info['air_genre_name'].fillna(store_info['hpg_genre_name'])
    # Now that we've filled the the missing information, we want to get rid of any extraneous columns.
    store_info.drop(['hpg_latitude',
                 'hpg_longitude',
                 'hpg_area_name',
                 'air_area_name',
                 'air_store_id',
                 'air_genre_name',
                 'hpg_genre_name'],
                 axis=1, inplace=True)
    # We want to create a latlong field for later.
    store_info['latlong'] = store_info[['latitude', 'longitude']].apply(tuple, axis=1)
    
    # It is easier to incorporate the date_info table in here as opposed to later linking it
    # as an entity (either way works), as there is only one important attribute (holiday flag). 
    # We can technically compute all of these values using premium primitives, but if we don't 
    # use premium primitives, we will need the holiday flag values.
    date_info['holiday_flg'] = date_info['holiday_flg'].astype(bool)
    date_info.drop(['day_of_week'], axis=1, inplace=True)
    date_info = date_info.sort_values('calendar_date')

    air_visit_data = air_visit_data.merge(date_info[['calendar_date', 'holiday_flg']],
                                          left_on='visit_date',
                                          right_on='calendar_date',
                                          how='left')
    air_visit_data = air_visit_data.rename(columns={'air_store_id': 'store_id'})
    
    # We want to make each visit unique, so we'll create a new column that concatenates 
    # the store_id with the visit_date.
    air_visit_data['store_calendar_id'] = air_visit_data['store_id'].str.cat(
                                                                air_visit_data['visit_date'].astype(str))
    air_visit_data = air_visit_data.sort_values(['visit_date', 'store_id'])
    visit_data = air_visit_data
    
    # We combine the hpg and air data (rather similarly to what we did before with the store_info)
    hpg_reserve = hpg_reserve.merge(store_info[['store_id', 'hpg_store_id']],
                                on='hpg_store_id',
                                how='left')
    hpg_reserve.drop(['hpg_store_id'], axis=1, inplace=True)
    store_info.drop(['hpg_store_id'], axis=1, inplace=True)
    air_reserve = air_reserve.rename(columns={'air_store_id': 'store_id'})
    
    # We combine and create the unique visit id from the store id and datetime.
    combined_reserve = (pd.concat([air_reserve, hpg_reserve], ignore_index=True, sort=True)
                    .reset_index(drop=True)
                    .sort_values(['reserve_datetime']))
    combined_reserve['visit_date'] = pd.to_datetime(pd.to_datetime(combined_reserve['visit_datetime']).dt.date)
    combined_reserve['store_calendar_id'] = combined_reserve['store_id'].str.cat(
            combined_reserve['visit_date'].astype(str))
    
    date_info['calendar_date'] = pd.to_datetime(date_info['calendar_date'])
    combined_reserve = combined_reserve.merge(date_info[['calendar_date', 'holiday_flg']],
                                          left_on='visit_date',
                                          right_on='calendar_date',
                                          how='left')
    combined_reserve['reserve_date'] = pd.to_datetime(combined_reserve['reserve_datetime']).dt.date

    es = ft.EntitySet(id="customer_data")
    
    es.entity_from_dataframe(entity_id='store_info',
                         dataframe=store_info,
                         index='store_id',
                         variable_types={"latlong": ft.variable_types.LatLong})
    es.entity_from_dataframe(entity_id='visit_data',
                             dataframe=visit_data,
                             index='store_calendar_id',
                             time_index='visit_date')
    es.entity_from_dataframe(entity_id='reservations',
                             dataframe=combined_reserve,
                             index='reservation_id',
                             make_index=True,
                             time_index='reserve_datetime')
    # This isn't important for the entityset, but we'll load it so we can visualize the data
    es.entity_from_dataframe(entity_id='date_info',
                             dataframe=date_info,
                             index='calendar_date')
    
    es.normalize_entity(base_entity_id='store_info',
                        new_entity_id='genres',
                        index='genre_name')
    es.add_relationships([
        ft.Relationship(es['store_info']['store_id'],
                        es['visit_data']['store_id']),
        ft.Relationship(es['visit_data']['store_calendar_id'],
                        es['reservations']['store_calendar_id']),
    ])
    
    return es


def bayesian_encoder_results(feature_matrix, features):
    X = feature_matrix.drop('visitors', axis=1)
    X = X.fillna(0)
    y = feature_matrix['visitors']

    bayesian_encoders = [ce.Encoder(method='target'),
                         ce.Encoder(method='leave_one_out')]
    bayesian_results = pd.DataFrame(columns=['Encoder', 'Score', '# Columns', ])
    for encoder in bayesian_encoders:
        encoder_name = str(encoder)[:str(encoder).find('(')]
        start_time = time.time()
        
        scores = []
        for i in range(3):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

            features_old = features.copy()
            features.remove('visitors')
            encoder.fit(X_train, features, y_train)
            print(len(features))
            features = features_old
            print(len(features))

            X_train_encoded = encoder.transform(X_train)
            X_test_encoded = encoder.transform(X_test)

            model = create_xgb_model(X_train_encoded, X_test_encoded, y_train, y_test)
            dtest = xgb.DMatrix(X_test_encoded)
            preds = model.predict(dtest)
            scores.append(r2_score(y_test, preds, multioutput='variance_weighted'))
        score = "SCORE: %.2f +/- %.2f" % (scores.mean(), scores.std())

        bayesian_results = bayesian_results.append({'Encoder': encoder_name,
                                                    'Score': score,
                                                    '# Columns': len(X_train.columns),
                                                    'Average Elapsed Time': (time.time() - start_time) / 3},
                                                   ignore_index=True)
    print(len(features))
    return bayesian_results


def classic_encoder_results(feature_matrix, features):
    print(len(features))
    classic_encoders = [ce.Encoder(method='ordinal'), ce.Encoder(method='one_hot'), ce.Encoder(method='binary'), ce.Encoder(method='hashing')]
    classic_results = pd.DataFrame(columns=['Encoder', 'Score', '# Columns', 'Elapsed Time'])
    for encoder in classic_encoders:
        encoder_name = str(encoder)[:str(encoder).find('(')]
        start_time = time.time()
        
        X_encoded = encoder.fit_transform(feature_matrix, features)

        X_encoded = X_encoded.drop('visitors', axis=1)
        X_encoded = X_encoded.fillna(0)
        y = feature_matrix['visitors']

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3)
        model = create_xgb_model(X_train, X_test, y_train, y_test)
        dtest = xgb.DMatrix(X_test)
        preds = model.predict(dtest)
        score = "SCORE: %.2f" % (r2_score(y_test, preds, multioutput='variance_weighted'))

        classic_results = classic_results.append({'Encoder': encoder_name,
                                                  'Score': score,
                                                  '# Columns': len(X_train.columns),
                                                  'Average Elapsed Time': (time.time() - start_time) / 3},
                                                  ignore_index=True)
    return classic_results


def create_xgb_model(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    evals = [(dtrain, 'train'), (dvalid, 'valid')]

    params = {
        'min_child_weight': 1, 'eta': 0.166,
        'colsample_bytree': 0.4, 'max_depth': 9,
        'subsample': 1.0, 'lambda': 57.93,
        'booster': 'gbtree', 'gamma': 0.5,
        'silent': 1, 'eval_metric': 'rmse',
        'objective': 'reg:linear',
    }

    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=227,
                    evals=evals, early_stopping_rounds=60, maximize=False, verbose_eval=False)
    
    return model