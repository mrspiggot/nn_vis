import dash_core_components as dcc
from util_classes.MNIST import MNISTBase
import dash_html_components as html
import dash_bootstrap_components as dbc
import os
from dash.dependencies import Input, Output, State
from app import app
import random
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from util_classes.MNIST import MNISTBase

mn = MNISTBase('fashion')

slider_dict = {1: 'Tiny', 2: "V-Small", 3: 'Small', 4: 'Med', 5: 'Large', 6: 'Huge'}
amount_dict = {1: '10', 2: '20', 3: '50', 4: '100', 5: '200'}
type_list= [{'label': 'Numbers', 'value': 'numbers'}, {'label': 'Fashion', 'value': 'fashion'}, {'label': 'CIFAR 10', 'value': 'cifar10'}]
random_dict = {1: 'Ordered', 2: 'Random'}

options=[]
for item in mn.class_names:
    d = {'label': item, 'value': item}
    options.append(d)



layout = html.Div([
    dbc.Row([
        dbc.Col([html.P("Amount to display")], width=2, style={'textAlign': 'center'}, lg={'size': 2,  "offset": 4}),
        dbc.Col([html.P("Img. Size")], width=2, style={'textAlign': 'center'}),
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Slider(min=1, max=5, marks=amount_dict, id="amount-slider", value=2),
            width=2,lg={'size': 2,  "offset": 4}
        ),
        dbc.Col(
            dcc.Slider(min=1, max=6, marks=slider_dict, id="size-slider", value=3),
            width=3, lg={'size': 2,  "offset": 0}
        ),
    ]),
    dbc.Row([dbc.Col([html.P("Sample Images:")], width=12, style={'textAlign': 'center'}),]),
    html.Div(id='view-content', children=[]),
])


@app.callback(Output('view-content', 'children'),
              [Input('amount-slider', 'value'),
               Input('size-slider', 'value')])
def draw_garments(amount, size):
    print("Hello world")
    ti = mn.test_images
    matching_images = []
    print(size)
    for i in range(int(amount_dict[amount])):
        fig = make_subplots(1, 1)
        img = np.array([[[255 - s, 255 - s, 255 - s] for s in r] for r in ti[i]], dtype="u1")
        fig.add_trace(go.Image(z=img), 1, 1)
        fig.update_layout(height=np.sqrt(size)*180, width=np.sqrt(size)*180)
        matching_images.append(dcc.Graph(id='ret-fig', figure=fig))

        im_row = dbc.Row(matching_images)


    return im_row



















    # to_display = int(amount_dict[amount])
    # to_scale = size/1.5
    # per_image = int(to_display / len(images))
    # print("To display: ", to_display)
    #
    # mn = MNISTBase(im_type)
    # N = [i for i in range(len(mn.class_names)) if mn.class_names[i] in images]
    #
    # image_mask = np.isin(mn.train_labels, N)
    # im = image_mask.reshape(len(image_mask,))
    # X = mn.train_images[im]
    #
    # print("Images", len(X), X.shape)
    #
    # print(N)
    #
    # matching_images = []
    # for n in N:
    #     print(n)
    #     display_mask = np.where(mn.train_labels == n)
    #     match = mn.train_images[display_mask]
    #     display_range = match[0:to_display]
    #     for i in display_range:
    #         img = np.array([[[255 - s, 255 - s, 255 - s] for s in r] for r in display_range[i]], dtype="u1")
    #         fig = go.Image(z=img)
    #         # fig.update_layout(height=250, width=250)
    #         matching_images.append(dcc.Graph(id='ret-fig', figure=fig))
    #
    # im_row = dbc.Row(matching_images)
    #
    # return im_row








