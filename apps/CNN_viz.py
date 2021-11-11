import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from app import app
from dash.dependencies import Input, Output, State, MATCH, ALL
from util_classes.MNIST import MLP, MNISTBase, NN_layers
import numpy as np
import pickle
from pathlib import Path
import os
import dash_table as dt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas as pd

layout = html.Div([
    dbc.Row([
        dbc.Col([html.H5("Upload .hdf5 file")],width={"size": 1, "offset": 1, "align": "center"}),
        dbc.Col(dcc.Upload(
                id='upload-hdf5',
                children=html.Div([
                    'Drag & drop or ',
                    html.A('select ".hdf5" file')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=True
            ),width={"size": 2, "offset": 0}),
        dbc.Col(html.P("Layer 1 Channel")),
        dbc.Col([dcc.Input(id='l1-ip', type='number', min=0, max=64, step=1, value=10)]),
        dbc.Col(html.P("Layer 2 Channel")),
        dbc.Col([dcc.Input(id='l2-ip', type='number', min=0, max=128, step=1, value=5)]),
        dbc.Col([dbc.Button("Display", id='disp-but', n_clicks=0, color="success")]),
    ]),
    dbc.Row([
        dbc.Col([html.H4(id='cnn-images')], width=12, style={'textAlign': 'center'}),
        dcc.Loading(html.Div(id='view-cnn', children=[]), type='cube', fullscreen=True, className='dash-bootstrap'),
    ])
])

@app.callback([Output('cnn-images', 'children'),
               Output('view-cnn', 'children')],
                [Input('upload-hdf5', 'filename'),
                 Input('disp-but', 'n_clicks')],
                [State('l1-ip', 'value'),
                State('l2-ip', 'value')],
              )
def update_output(fname, click, l1, l2):

    if fname == None:
        fname = ['fashionCNN20211108_095455.hdf5']

    fn = fname[0]
    title = ""
    images = dbc.Row([])

    if 'CNN' not in fn:
        title = "Need to supply a ConvNet (must have 'CNN' in filename). Drag & Drop a 'CNN' .hdf5 file."
        return title, images

    title = fn
    fhdf5 = os.path.join(os.getcwd(), "assets", fn)
    nn = NN_layers(fhdf5)
    cl = nn.conv_layers
    l0_filter_list=[]
    size=2
    im_row_l = []

    l = cl[0]


    for i in range(l.shape[3]):
        f = l[:, :, :, i]
        fig = make_subplots(1, 1)
        img = np.array([[[255*s, 255*s, 255*s] for s in r] for r in f[:,:,0]], dtype="u1")
        fig.add_trace(go.Image(z=img), 1, 1)
        fig.update_layout(height=np.sqrt(size) * 180, width=np.sqrt(size) * 180,
                          plot_bgcolor='#222', paper_bgcolor='#222', font=dict(size=12, color='white'))
        l0_filter_list.append(dcc.Graph(id='ret-fig', figure=fig))

    itxt = dbc.Row(dbc.Col(html.H1("Layer0"), width={"size": 6, "offset": 5}))
    images = dbc.Row(l0_filter_list)
    iRow = dbc.Row([itxt, images])
    im_row_l.append(iRow)

    l1_filter_list = []
    l = cl[1]
    for i in range(l.shape[3]):
        f = l[:, :, :, i]
        fig = make_subplots(1, 1)
        img = np.array([[[255*s, 255*s, 255*s] for s in r] for r in f[:,:,l1]], dtype="u1")
        fig.add_trace(go.Image(z=img), 1, 1)
        fig.update_layout(height=np.sqrt(size) * 180, width=np.sqrt(size) * 180,
                          plot_bgcolor='#222', paper_bgcolor='#222', font=dict(size=12, color='white'))
        l1_filter_list.append(dcc.Graph(id='ret-fig', figure=fig))

    itxt = dbc.Row(dbc.Col(html.H1("Layer1"), width={"size": 6, "offset": 5}))
    images = dbc.Row(l1_filter_list)
    iRow = dbc.Row([itxt, images])
    im_row_l.append(iRow)

    l = cl[2]
    l3_filter_list = []
    for i in range(l.shape[3]):
        f = l[:, :, :, i] * 255
        fig = make_subplots(1, 1)
        img = np.array([[[255-s, 255-s, 255-s] for s in r] for r in f[:,:,l2]], dtype="u1")
        fig.add_trace(go.Image(z=img), 1, 1)
        fig.update_layout(height=np.sqrt(size) * 180, width=np.sqrt(size) * 180,
                          plot_bgcolor='#222', paper_bgcolor='#222', font=dict(size=12, color='white'))
        l3_filter_list.append(dcc.Graph(id='ret-fig', figure=fig))

    itxt = dbc.Row(dbc.Col(html.H1("Layer2"), width={"size": 2, "offset": 5}))
    images = dbc.Row(l3_filter_list)
    iRow = dbc.Row([itxt, images])
    im_row_l.append(iRow)


    return title, im_row_l