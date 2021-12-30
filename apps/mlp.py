import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from app import app
from dash.dependencies import Input, Output, State, MATCH, ALL
from util_classes.MNIST import MLP, MNISTBase
import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import confusion_matrix
from pathlib import Path
import os
from datetime import datetime

ds_options = [
    {'label': 'Fashion', 'value': 'fashion'},
    {'label': 'Numbers', 'value': 'numbers'},
    {'label': 'Cifar10', 'value': 'cifar10'},
]

ac_options = [
    {'label': 'Sigmoid  ', 'value': 'sigmoid'},
    {'label': 'ReLU', 'value': 'relu'},
]

ep_options = [
    {'label': '10  ', 'value': 10},
    {'label': '20  ', 'value': 20},
    {'label': '50  ', 'value': 50},
    {'label': '100  ', 'value': 100},
]


layout = html.Div([
    dbc.Row([
        dbc.Col(html.H6("Data set"), width={"size": 1, "offset": 0}),
        dbc.Col(dcc.Dropdown(id="ds-dd", options=ds_options, className='dash-bootstrap', value='fashion'), width={"size": 2, "offset": 1})
    ]),
    dbc.Row(html.Br()),
    dbc.Row([
        dbc.Col(html.H6("Activation"), width={"size": 1, "offset": 0}),
        dbc.Col(dcc.RadioItems(id="ra-ac", options=ac_options, value='relu'), width={"size": 2, "offset": 1}),
    ]),
    dbc.Row(html.Br()),
    dbc.Row([
        dbc.Col(html.H6("Epochs"), width={"size": 1, "offset": 0}),
        dbc.Col(dcc.RadioItems(id="ra-ep", options=ep_options, value=10), width={"size": 2, "offset": 1}),
    ]),
    dbc.Row(html.Br()),
    dbc.Row([
        dbc.Col(html.H6("Neurons per layer"), width={"size": 1, "offset": 0}),
        dbc.Col(dbc.Input(id='ip-la', type='text', value='128,64', className='dash-bootstrap'), width={"size": 2, "offset": 1}),
        dbc.Col(html.P("Enter neurons per layer, separated by commas")),
    ]),
    dbc.Row(html.Br()),
    dbc.Row([
        dbc.Col(html.H6("Filename (Optional):"), width={"size": 1, "offset": 0}),
        dbc.Col(dbc.Input(id='ip-fi', type='text', className='dash-bootstrap'), width={"size": 2, "offset": 1}),
        dbc.Col(html.P("Enter a filename, alternatively let the system generate a unique one for you based on parameters and datetime"))

    ]),
    dbc.Row([
        dbc.Col(dcc.Loading(html.P(id='op-fi')), width={"size": 6, "offset": 2})
    ]),
    dbc.Row(html.Br()),
    dbc.Row([
        dbc.Col(dcc.Loading(dbc.Button("Train, test & Save", id='bu-tts', color="success")), width={"size": 2, "offset": 0}),
    ]),
])

@app.callback(Output('op-fi', 'children'),
              Input('bu-tts', 'n_clicks'),
              [State('ds-dd', 'value'),
               State('ra-ac', 'value'),
               State('ra-ep', 'value'),
               State('ip-la', 'value'),
               State('ip-fi', 'value')])
def mlp(click, dataset, activation, epochs, layers, filename):
    if click==None:
        return filename

    layer_list = [int(x) for x in layers.split(',') if x.strip().isdigit()]
    if filename==None:
        now = datetime.now()
        l = ""
        for num in layer_list:
            fs = "_" + str(num)
            l+=fs
        filename="MLP" + l + dataset + activation + now.strftime("%Y%m%d_%H%M%S") + ".pickle"

    mn = MNISTBase(dataset)
    perceptron = MLP(mn, layer_list, activation)
    perceptron.compile_model()
    perceptron.train_model(epochs)
    perceptron.evaluate_model()
    perceptron.confusion_mat(True, filename)
    perceptron.save_model()

    return "Results file is: " + filename