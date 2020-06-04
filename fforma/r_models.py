#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import rpy2.robjects as robjects

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import IntVector
from copy import deepcopy

forecast = importr('forecast')

def forecast_object_to_dict(forecast_object):
    """Transform forecast_object into a python dictionary."""
    dict_ = zip(forecast_object.names,
                list(forecast_object))
    dict_ = dict(dict_)

    return dict_

def get_forecast(fitted_model, h):
    """Calculate forecast from a fitted model."""
    y_hat = forecast.forecast(fitted_model, h=h)
    y_hat = forecast_object_to_dict(y_hat)
    y_hat = y_hat['mean']

    return y_hat

def get_fitted_forecast_model(y, freq, model, **kwargs):
    """Wrapper of the following flow:
        - Load _forecast_ package.
        - Transform data into a ts object.
        - Train the model.

    Parameters
    ----------
    model: str
        Name of a model included in the
        _forecast_ package. Ej. 'auto.arima'.
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    kwargs:
        Arguments of the model function.

    Returns
    -------
    rpy2 object
        Fitted model
    """

    freq = deepcopy(freq)
    if isinstance(freq, int):
        freq = freq
    else:
        freq = IntVector(freq)

    rstring = """
     function(y, freq, ...){
         library(forecast)
         y_ts <- msts(y, seasonal.periods=freq)
         fitted_model<-%s(y_ts, ...)
         fitted_model
     }
    """ % (model)
    pandas2ri.activate()
    rfunc = robjects.r(rstring)
    fitted_model = rfunc(y, freq, **kwargs)

    return fitted_model

class ForecastModel(BaseEstimator, RegressorMixin):
    """Wrapper for models in the R package _forecast_ that returns a model.

    Parameters
    ----------
    model: str
        Name of a model included in the
        _forecast_ package. Ej. 'auto.arima'.
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    kwargs:
        Arguments of the model function.
    """

    def __init__(self, model, freq, **kwargs):
        self.freq = freq
        self.model = model
        self.kwargs = kwargs

    def fit(self, y):
        self.fitted_model_ = get_fitted_forecast_model(y, self.freq, self.model, **self.kwargs)

        return self

    def predict(self, h):
        check_is_fitted(self, 'fitted_model_')
        y_hat = get_forecast(self.fitted_model_, h)

        return y_hat

class ForecastObject(BaseEstimator, RegressorMixin):
    """Wrapper for models in the R package _forecast_ that returns a forecast object.

    Parameters
    ----------
    model: rpy2 object
        Name of a model included in the
        _forecast_ package. Ej. 'auto.arima'.
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    kwargs:
        Arguments of the model function.
    """

    def __init__(self, model, freq, **kwargs):
        self.freq = freq
        self.model = model
        self.kwargs = kwargs

    def fit(self, y):
        self.y_ts_ = y

        return self

    def predict(self, h):
        check_is_fitted(self, 'y_ts_')
        fitted_model = get_fitted_forecast_model(self.y_ts_, self.freq, self.model, h=h, **self.kwargs)

        y_hat = get_forecast(fitted_model, h)

        return y_hat

class ARIMA(ForecastModel):
    """Wrapper of forecast::auto.arima from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq, **kwargs):
        super().__init__(model='auto.arima', freq=freq, **kwargs)

class ETS(ForecastModel):
    """Wrapper of forecast::ets from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq, **kwargs):
        super().__init__(model='ets', freq=freq, **kwargs)

class NNETAR(ForecastModel):
    """Wrapper of forecast::nnetar from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq, **kwargs):
        super().__init__(model='nnetar', freq=freq, **kwargs)

class TBATS(ForecastModel):
    """Wrapper of forecast::tbats from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq, **kwargs):
        super().__init__(model='tbats', freq=freq, **kwargs)

class STLM(ForecastModel):
    """Wrapper of forecast::stlm from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq=7, **kwargs):
        super().__init__(model='stlm', freq=freq, **kwargs)

class RandomWalk(ForecastModel, BaseEstimator, RegressorMixin):
    """Wrapper of forecast::rwf from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq, **kwargs):
        super().__init__(model='rwf', freq=freq, **kwargs)

class ThetaF(ForecastObject):
    """Wrapper of forecast::thetaf from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq, **kwargs):
        super().__init__(model='thetaf', freq=freq, **kwargs)

class Naive(ForecastObject):
    """Wrapper of forecast::naive from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq, **kwargs):
        super().__init__(model='naive', freq=freq, **kwargs)

class SeasonalNaive(ForecastObject):
    """Wrapper of forecast::snaive from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq, **kwargs):
        super().__init__(model='snaive', freq=freq, **kwargs)
