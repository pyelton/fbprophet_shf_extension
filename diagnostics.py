# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.






import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(name)-8s: %(levelname)-8s %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from functools import reduce
from fbprophet import Prophet

def _cutoffs(df, horizon, k, period,time_buffer=None,first_cutoff=None):
    """Generate cutoff dates

    Parameters
    ----------
    df: pd.DataFrame with historical data
    horizon: pd.Timedelta.
        Forecast horizon
    k: Int number.
        The number of forecasts point.
    period: pd.Timedelta.
        Simulated Forecast will be done at every this period.

    Returns
    -------
    list of pd.Timestamp
    """
    if first_cutoff == None:
        # Last cutoff is 'latest date in data - horizon' date
        cutoff = df['ds'].max() - horizon
        if cutoff < df['ds'].min():
            raise ValueError('Less data than horizon.')
        result = [cutoff]
        for i in range(1, k):
            cutoff -= period
            # If data does not exist in data range (cutoff, cutoff + horizon]
            if not (((df['ds'] > cutoff) & (df['ds'] <= cutoff + horizon)).any()):
                # Next cutoff point is 'last date before cutoff in data - horizon'
                closest_date = df[df['ds'] <= cutoff].max()['ds']
                cutoff = closest_date - horizon
            if cutoff < df['ds'].min():
                logger.warning(
                'Not enough data for requested number of cutoffs! '
                'Using {}.'.format(i))
                break
            result.append(cutoff)
    else:
        cutoff = pd.to_datetime(first_cutoff)
        result = [cutoff]
        if cutoff + time_buffer + horizon <= df.ds.max():
            cutoff += period
            result.append(cutoff)       
                    
    # Sort lines in ascending order
    return reversed(result)


def simulated_historical_forecasts(model, input_df, horizon, k, period=None, initial=None, time_buffer='0 days', first_cutoff=None):
    """Simulated Historical Forecasts.

    Make forecasts from k historical cutoff points, working backwards from
    (end - horizon) with a spacing of period between each cutoff.

    Parameters
    ----------
    model: Prophet class object.
        Fitted Prophet model
    horizon: string with pd.Timedelta compatible style, e.g., '5 days',
        '3 hours', '10 seconds'.
    k: Int number of forecasts point.
    period: Optional string with pd.Timedelta compatible style. Simulated
        forecast will be done at every this period. If not provided,
        0.5 * horizon is used.

    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.
    """
    time_buffer = pd.Timedelta(time_buffer)
    if type(model) == Prophet:
        df = model.history.copy().reset_index(drop=True)
    else:
        df = input_df
    horizon = pd.Timedelta(horizon)
    period = 0.5 * horizon if period is None else pd.Timedelta(period)
    cutoffs = _cutoffs(df, horizon, k, period,time_buffer,first_cutoff)
    predicts = []
    indices = []
    for cutoff in cutoffs:
        logger.info('cutoff is %s' % cutoff)
        if initial != None:
            df_cutoff = df[(df['ds'] <= cutoff) & (df['ds'] >= cutoff - initial)]
        else:
            df_cutoff = df[df['ds'] <= cutoff]
        logger.info('min and max of training df %s %s' % (df_cutoff.ds.min(),df_cutoff.ds.max()))
        index_predicted = (df['ds'] > cutoff + time_buffer) & (df['ds'] <= cutoff + time_buffer + horizon)
        indices += list(df[(df.ds > cutoff + time_buffer) & (df.ds <= cutoff + time_buffer + horizon)].index)
        logger.info('min and max of prediction df %s %s' % (df[index_predicted].ds.min(),df[index_predicted].ds.max()))
        if type(model) == Prophet:
            # Generate new object with copying fitting options
            m = model.copy(cutoff)
        else:
            m = model
        if type(model) == Prophet:
            # Train model
            m.fit(df_cutoff)
            # Calculate yhat
           
            columns = ['ds']
            if m.growth == 'logistic':
                columns.append('cap')
                if m.logistic_floor:
                    columns.append('floor')
            columns.extend(m.extra_regressors.keys())
            yhat_df = m.predict(df[index_predicted][columns])
            predicts.append(pd.concat([
            yhat_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            df[index_predicted][['y']].reset_index(drop=True),
            pd.DataFrame({'cutoff': [cutoff] * yhat_df.shape[0]})
            ]   , axis=1))
        else:
            m.fit(df_cutoff.drop(['y','ds'],axis=1),df_cutoff['y']) 
            yhat = m.predict(df[index_predicted].drop(['y','ds'],axis=1))
            yhat_df = pd.DataFrame({'yhat':yhat})
            predicts.append(pd.concat([yhat_df,
            pd.DataFrame({'cutoff': [cutoff] * len(yhat)})
            ], axis=1))
    # Combine all predicted pd.DataFrame into one pd.DataFrame
    if type(model) == Prophet:
        output_df = reduce(lambda x, y: x.append(y), predicts).reset_index(drop=True)
    else:
        model_df = reduce(lambda x, y: x.append(y), predicts).reset_index(drop=True)
        input_df = input_df.iloc[list(indices),:].reset_index()
        output_df = pd.concat([input_df,model_df],axis=1)
    return output_df

def cross_validation(model, horizon, period=None, initial=None, time_buffer='0 days', first_cutoff=None, input_df=None):
    """Cross-Validation for time series.

    Computes forecasts from historical cutoff points. Beginning from initial,
    makes cutoffs with a spacing of period up to (end - horizon).

    When period is equal to the time interval of the data, this is the
    technique described in https://robjhyndman.com/hyndsight/tscv/ .

    Parameters
    ----------
    model: Prophet class object. Fitted Prophet model
    horizon: string with pd.Timedelta compatible style, e.g., '5 days',
        '3 hours', '10 seconds'.
    period: string with pd.Timedelta compatible style. Simulated forecast will
        be done at every this period. If not provided, 0.5 * horizon is used.
    initial: string with pd.Timedelta compatible style. The first training
        period will begin here. If not provided, 3 * horizon is used.

    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.
    """
    horizon = pd.Timedelta(horizon)
    period = 0.5 * horizon if period is None else pd.Timedelta(period)
    initial = 3 * horizon if initial is None else pd.Timedelta(initial)
    if type(model) == Prophet:
        te = model.history['ds'].max()
        ts = model.history['ds'].min()
    else:
        te = input_df['ds'].max()
        ts = input_df['ds'].min()
    k = int(np.ceil(((te - horizon) - (ts + initial)) / period))
    if k < 1:
        raise ValueError(
            'Not enough data for specified horizon, period, and initial.')
    return simulated_historical_forecasts(model, input_df, horizon, k, period, initial, time_buffer, first_cutoff)

def plot_timeseries_predictions(df_plot,ylabel):
    plt.figure(figsize=(25,10))
    colors = {}
    palette = ['red','blue','green','orange','purple']*10
    df_plot.cutoff = df_plot.cutoff.map(str)
    ax = df_plot.plot('ds',y=['y'],ax = plt.gca(),color='black',alpha=0.5,style='.-')
    grouped = df_plot.groupby('cutoff')
    for i in range(len(df_plot.cutoff.unique())):
        colors[df_plot.cutoff.unique()[i]] = palette[i]

    for key, group in grouped:
        print('cutoff',key)
        group.plot(ax=plt.gca(), x='ds', y='yhat', label=key, color=colors[key],alpha=0.2,
                   style='.-')

    ax.set_ylabel(ylabel)
    ax.set_xlabel('Timestamp')
