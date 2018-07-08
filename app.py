# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 15:26:34 2018

@author: evanc
"""

import dash
import dash_core_components as dcc
print(dcc.__version__)
import dash_html_components as html
import pandas as pd
from sklearn.externals import joblib
import numpy as np
import plotly.graph_objs as go
from xgboost import XGBRegressor
import xgboost
import plotly.plotly as py
import plotly.tools as tls


app = dash.Dash()
app.config['suppress_callback_exceptions'] = True
server = app.server
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

dtyp = {'Assortment':object,
        'CompDist':object,
        'CompOpen':object,
        'Customers':int,
        'DayOfWeek':int,
        'Open':int,
        'Promo':int,
        'Promo2':int,
        'Sales':float,
        'Store':int,
        'StoreType':object,
        'month':int,
        'Date':object
        }
obs = pd.read_csv('train_for_app2.csv', dtype = dtyp)
pred = pd.read_csv('test_with_preds2.csv', dtype = dtyp)
obs['Set'] = 'Observed'
pred['Set'] = 'Predicted'

comb = obs.append(pred)
comb.Date = comb.Date.astype('datetime64[ns]')
comb['AvgPerCust'] = comb.Sales / comb.Customers
model = joblib.load('sales2.joblib.dat')
mod = model.best_estimator_
feat_imp = pd.DataFrame({'Features':model.Features,
                         'Importances':mod.feature_importances_})
feat_imp = feat_imp.sort_values(by = 'Importances', ascending = False)

date_agg = comb.groupby(['Date', 'Set'], as_index = False)[['Sales', 'Preds']].sum()

#trace1 = go.Scatter(
#    x=date_agg.index,
#    y=date_agg.Sales,
#    name = "Aggregated Sales",
#    line = dict(color = '#17BECF'),
#    opacity = 0.8)
#
#data1 = [trace1]
#layout1 = dict(
#    title='Aggregated Sales',
#    xaxis=dict(
#        rangeselector=dict(
#            buttons=list([
#                dict(count=1,
#                     label='1m',
#                     step='month',
#                     stepmode='backward'),
#                dict(count=6,
#                     label='6m',
#                     step='month',
#                     stepmode='backward'),
#                dict(step='all')
#            ])
#        ),
#        rangeslider=dict(),
#        type='date'
#    )
#)

store_agg = comb.groupby(['Date', 'Store'], as_index = False)[['Sales', 'Preds']].sum()

month_agg = comb.groupby(['month'], as_index = False)['Sales', 'Preds'].mean()

day_agg = comb.groupby(['DayOfWeek'], as_index = False)['Sales', 'Preds'].mean()

promo_agg = comb[comb.Set == 'Observed'].groupby(['Promo'], as_index = False)['Sales', 'AvgPerCust'].mean()

del obs
del pred

app.layout = html.Div([
        html.H1('Analyzing Sales Volume'),

        html.Hr(),

        html.H2('Total Sales Volume'),
        html.H4('Observed vs Predicted'),
        html.Div([dcc.Graph(
               id = 'sales-plot',
               figure = {
                       'data' :[
                               go.Scatter(
                                    x = date_agg[date_agg.Set == 'Observed']['Date'],
                                    y = date_agg[date_agg.Set == 'Observed']['Sales'],
                                    opacity = .8,
                                    name = 'Observed'
                                    ),
                               go.Scatter(
                                    x = date_agg.Date,
                                    y = date_agg.Preds,
                                    opacity = .8,
                                    name = 'Predicted'
                                    )
                               ],
                       'layout': go.Layout(
                               xaxis=dict(
                                    title = 'Date - Use the slider to Change View',
                                    rangeselector=dict(
                                        buttons=list([
                                            dict(count=1,
                                                 label='1m',
                                                 step='month',
                                                 stepmode='backward'),
                                            dict(count=6,
                                                 label='6m',
                                                 step='month',
                                                 stepmode='backward'),
                                            dict(step='all')
                                        ])
                                    ),
                                    rangeslider=dict(),
                                    type='date'
                                    ),
                               yaxis={'title':'Sales'},
                               margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                               legend={'x': 0, 'y': 1},
                               hovermode='closest'
                               )
                       }
               ), '**Use the slider to change your view of the graph'], style = {'width':'80%', 'margin': '0 auto'}),

               html.Hr(),

               html.H2('Analyze By Store'),
               html.H4('Observed vs Predicted'),
               html.Div([html.Label('Select Store'),dcc.Dropdown(
                       id = 'store',
                       options=[{'label': i, 'value': i} for i in comb.Store.unique()],
                       value=262
                       )], style = {'width': '20%'}),
               html.Div([dcc.Graph( id = 'store-plot')], style = dict(width = '80%', margin = '0 auto')),

               html.H2('Analyze Past Sales'),
               html.H3('Relative Predictive Power of Data Categories'),
               html.Div([
                        dcc.Graph( id = 'imprt-plot', figure = {
                        'data' :[ go.Bar(
                                x = feat_imp.Features,
                                y = feat_imp.Importances,
                                name = 'Feature Importances'
                                )],
                        'layout' : go.Layout(
                                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                xaxis = {'title': 'Features'},
                                yaxis = {'title': 'Relative Importances'}
                                )
                        })], style = {'width' : '80%', 'margin': '0 auto'}),
                html.H3('Average Usage - Time Of Year'),
                html.Div([
                        html.Div([html.Div(dcc.Graph( id = 'month-plot', figure = {
                        'data' : [go.Scatter(
                                x = month_agg.month,
                                y = month_agg[i],
                                name = i
                                ) for i in ['Sales', 'Preds']],
                       'layout' : go.Layout(
                               xaxis = {'title':'Month'},
                               yaxis = {'title': 'Average Sales Volume'},
                               margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                               legend={'x': 0, 'y': 1},
                               hovermode='closest'
                               )
                        }), className="six columns"),
                       html.Div(dcc.Graph( id = 'dayofweek-plot', figure = {
                       'data' : [go.Scatter(
                               x = day_agg.DayOfWeek,
                               y = day_agg[i],
                               name = i
                               ) for i in ['Sales', 'Preds']],
                       'layout' : go.Layout(
                               xaxis = {'title':'Day Of Week'},
                               yaxis = {'title': 'Average Sales Volume'},
                               margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                               legend={'x': 0, 'y': 1},
                               hovermode='closest'
                               )
                       }), className="six columns")
                   ], className="row")]),
               html.H3('The Power of the Promotion'),
               html.Div(
                       [dcc.Graph( id = 'promo-plot', figure = {
                       'data' : [go.Bar(
                               x = promo_agg.Promo,
                               y = promo_agg['Sales'],
                               name = 'Average Sales'
                               ),
                               go.Scatter(
                               x = promo_agg.Promo,
                               y = promo_agg.AvgPerCust,
                               name= 'Average Per Customer',
                               yaxis = 'y2'
                               )],
                       'layout' : go.Layout(
                               barmode = 'group',
                               margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                               yaxis = {'title': 'Sales Volume'},
                               yaxis2 = dict(
                                title='Sales Per Customer',
                                titlefont=dict(
                                    color='rgb(148, 103, 189)'
                                ),
                                tickfont=dict(
                                    color='rgb(148, 103, 189)'
                                ),
                                range = [0, promo_agg.AvgPerCust.max()+1] ,
                                overlaying='y',
                                side='right'
                               )
                        )
                       })], style = {'width': '80%', 'margin':'0 auto'})
], style = {'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'})

@app.callback(
        dash.dependencies.Output('store-plot', 'figure'),
        [dash.dependencies.Input('store', 'value')]
        )
def generate_plot_by_store(value):
    return {
            'data': [
                    go.Scatter(
                            x = store_agg[store_agg.Store == value]['Date'],
                            y = store_agg[store_agg.Store == value]['Sales'],
                            opacity = .8,
                            name = 'Observed Sales - Store ' + str(value)
                    ),
                    go.Scatter(
                            x = store_agg[store_agg.Store == value]['Date'],
                            y = store_agg[store_agg.Store == value]['Preds'],
                            opacity = .8,
                            name = 'Observed Sales - Store ' + str(value)
                            )],
            'layout': go.Layout(
                            xaxis=dict(
                            title = 'Date - Use the slider to Change View',
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=1,
                                         label='1m',
                                         step='month',
                                         stepmode='backward'),
                                    dict(count=6,
                                         label='6m',
                                         step='month',
                                         stepmode='backward'),
                                    dict(step='all')
                                ])
                            ),
                            rangeslider=dict(),
                            type='date'
                            ),
                            yaxis={'title':'Sales'},
                            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                            legend={'x': 0, 'y': 1},
                            hovermode='closest'
                    )
            }


if __name__ == '__main__':
    app.run_server(debug = False)