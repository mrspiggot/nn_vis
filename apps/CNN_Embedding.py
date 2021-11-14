import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import base64
import dash_daq as daq
from app import app
from dash.dependencies import Input, Output, State, MATCH, ALL
from util_classes.MNIST import MLP, MNISTBase, NN_layers
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import numpy as np
from numpy import expand_dims
import pickle
from pathlib import Path
import os
import dash_table as dt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas as pd

layout = html.Div([
    dbc.Row([
        dbc.Col([html.H5("Upload .hdf5 file")],width={"size": 1, "offset": 0, "align": "center"}),
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
        dbc.Col([html.H5("Upload a test image")], width={"size": 1, "offset": 0, "align": "center"}),
        dbc.Col(dcc.Upload(
            id='upload-png',
            children=html.Div([
                'Drag & drop or ',
                html.A('select ".png" file')
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
        ), width={"size": 2, "offset": 0}),
        dcc.Loading(html.Div(id='view-sample', children=[]), type='cube', fullscreen=True, className='dash-bootstrap'),
        dbc.Col(html.P("Choose layer for embedding")),
        dbc.Col([dcc.Input(id='layer-ip', type='number', min=0, max=2, step=1, value=0)]),
        dbc.Col([dbc.Button("Display", id='embed-but', n_clicks=0, color="success")]),
    ]),
    dbc.Row([
        dbc.Col([html.H4(id='embed-images')], width=12, style={'textAlign': 'center'}),
        dcc.Loading(html.Div(id='view-embed', children=[]), type='cube', fullscreen=True, className='dash-bootstrap'),
    ])
])

@app.callback([Output('embed-images', 'children'),
               Output('view-embed', 'children'),
               Output('view-sample', 'children')],
                [Input('upload-hdf5', 'filename'),
                Input('upload-png', 'filename'),
                 Input('embed-but', 'n_clicks')],
                State('layer-ip', 'value'),
              )
def update_output(fname, iname, click, l1):
    if fname == None:
        fname = ['fashionCNN20211108_095455.hdf5']

    if iname == None:
        iname = ['T-shirt_4108_test.png']

    fn = fname[0]
    im = iname[0]
    title = ""
    images = dbc.Row([])

    if 'CNN' not in fn:
        title = "Need to supply a ConvNet (must have 'CNN' in filename). Drag & Drop a 'CNN' .hdf5 file."
        return title, images

    title = fn
    fhdf5 = os.path.join(os.getcwd(), "assets", fn)
    im_handle = os.path.join(os.getcwd(), "assets", "Sample Images", im)
    encoded_image = base64.b64encode(open(im_handle, 'rb').read())
    img = html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), width=64, height=64)
    np_img = load_img(im_handle, target_size=(28,28))
    np_img = img_to_array(np_img)
    np_img = expand_dims(np_img, axis=0) #Input to NN expects a tensor of images. This creates a single image tensor
    np_img = np.dot(np_img[...,:3], [0.2989, 0.5870, 0.1140]).astype(int) #RGB to grayscale
    title = im_handle
    nn = NN_layers(fhdf5)

    model = nn.model
    ixs = []
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if 'conv' not in layer.name:
            continue
        ixs.append(i)

    print(ixs)
    outputs = [model.layers[i].output for i in ixs]
    model = Model(inputs=model.inputs, outputs=outputs)
    feature_maps = model.predict(np_img)
    # print(feature_maps.shape, type(feature_maps))

    title_list = []
    n = 0
    for fmap in feature_maps:
        print(fmap.shape, type(fmap))
        fa = fmap.max()
        fi = fmap.min()
        fmap = 255*(fmap-fi)/(fa-fi)
        fmap = fmap.astype(int)
        l_tit = "Layer " + str(n) + " with " + str(fmap.shape[3]) + " filters"
        tit = dbc.Row(dbc.Col(html.H5(l_tit), width={"size": 6, "offset": 2, "align": "center"}))
        n+=1
        embed_list = []
        for i in range(fmap.shape[3]):

            fig = make_subplots(1,1)
            embed_img = fmap[0, :, :, i]
            print(embed_img.shape)
            img = np.array([[[s, s, s] for s in r] for r in embed_img[:, :]], dtype="u1")
            im_tit = "Output from filter " + str(i)
            # img = np.array(embed_img, dtype="u1")
            fig.add_trace(go.Image(z=img), 1, 1)
            fig.update_layout(height=256, width=256, title=im_tit,
                              plot_bgcolor='#222', paper_bgcolor='#222', font=dict(size=12, color='white'))
            embed_list.append(dcc.Graph(id='ret-fig', figure=fig))

        title_list.append(tit)
        title_list.append(dbc.Row(embed_list))

    images = dbc.Row(title_list)

    return title, images, img