
import fforma
import fforma.base_models
import fforma.base_models_r
from fforma.m4_data import prepare_m4_data, seas_dict

dataset = 'Monthly'
validation_periods = seas_dict[dataset]['output_size']
seasonality = seas_dict[dataset]['seasonality']


X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset, './m4', 100)

# optimal params by hyndman
optimal_params = {'n_estimators': 94,
                  'eta': 0.58,
                  'max_depth': 14,
                  'subsample': 0.92,
                  'colsample_bytree': 0.77}

h = validation_periods

base_models = {'SeasonalNaive': fforma.base_models.SeasonalNaive(h, seasonality),
               'Naive2': fforma.base_models.Naive2(h, seasonality),
               'RandomWalkDrift': fforma.base_models.RandomWalkDrift(h),
               #'MovingAverage': fforma.base_models.MovingAverage(h, seasonality),
               'ETS': fforma.base_models_r.ETS(freq=seasonality)
               #'ARIMA': fforma.base_models_r.ARIMA(freq=seasonality)
               }

fforma = fforma.FFORMA(params=optimal_params,
                       h=h,
                       seasonality=seasonality,
                       base_models=base_models,
                       metric='smape',
                       early_stopping_rounds=10,
                       threads=None,
                       random_seed=1)

fforma.fit(X_train_df, y_train_df)

preds = fforma.predict(X_test_df)
