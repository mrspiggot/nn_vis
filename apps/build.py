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
import base64
import datetime
import io
import dash_table
import os
import dash_table as dt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas as pd

MAX_NEURONS = 784
HDF5_DIRECTORY = Path("assets")
amount_dict = {1: '10', 2: '20', 3: '50', 4: '100', 5: '200'}


layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H4("Confusion matrix Uploader"),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag & drop or ',
                    html.A('select ".pickle" file')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=True
            ),
            html.P("Results to display"),
            dcc.Slider(min=1, max=5, marks=amount_dict, id="limit-slider", value=2),
            ], width=2
        ),
        dbc.Col([
            dt.DataTable(id='output-data-upload', style_data_conditional=[
                             {
                                 'if': {'row_index': 'odd'},
                                 'backgroundColor': 'rgb(40, 40, 40)',
                             },
                            {
                                'if': {'row_index': 'even'},
                                'backgroundColor': 'rgb(50, 50, 50)',
                            }
                         ],
                         style_header={
                             'backgroundColor': 'rgb(10, 10, 10)',
                             'color': 'white'
                         },
                         ),
            dbc.Alert("Drag a pickle file to the drop-site on the left and then click a cell on the table", id='tbl_out'),
        ], width={"size": 8, "offset": 0})
    ]),
    dbc.Row([
        dbc.Col([html.H4(id='conf-result')], width=12, style={'textAlign': 'center'}),
        dcc.Loading(html.Div(id='view-results', children=[]), type='cube', fullscreen=True, className='dash-bootstrap'),
    ])
])

@app.callback([Output('tbl_out', 'children'),
               Output('view-results', 'children'),
               Output('conf-result', 'children')],
               [Input('output-data-upload', 'active_cell'),
               Input('limit-slider', 'value')],
               State('upload-data', 'filename'))
def update_graphs(active_cell, limit, fname):
    if active_cell == None:
        active_cell['row'] = 0
        active_cell['column'] = 0

    fname = os.path.join(os.getcwd(), "assets", fname[0])

    with open(fname, 'rb') as file:
        nn_d = pickle.load(file)

    mn = MNISTBase(nn_d['Type'])
    conf_df = pd.DataFrame(nn_d['Confusion'], columns=nn_d['Labels'], index=nn_d['Labels'])
    conf_df.insert(0, "id", nn_d['Labels'])
    reality = nn_d['Labels'][active_cell['row']]
    estimate = nn_d['Labels'][active_cell['column']-1]
    predictions = nn_d['Prediction']
    tl = mn.test_labels
    ti = mn.test_images
    mc = mn.class_names
    n_est = active_cell['row']
    r_est = active_cell['column']-1

    i = 0
    match = 0
    image_list = []
    for pred in predictions:
        nn_pred = np.argmax(pred)
        if nn_pred == r_est:
            # print("Prediction: ", mn.class_names[n_est])
            if tl[i] == n_est:
                if match <= int(amount_dict[limit]):
                    image_list.append(i)
                match += 1
        i += 1


    matching_images = []
    if nn_d['Type'] == 'cifar10':
        for i in image_list:
            colors = ['blue'] * 10
            colors[n_est] = 'green'
            colors[r_est] = 'red'
            fig = make_subplots(1, 2)
            fig.add_trace(go.Image(z=ti[i]), 1, 1)
            fig.add_trace(go.Bar(x=mc, y=predictions[i], marker_color=colors), 1, 2)
            fig.update_layout(height=300, width=500, plot_bgcolor='#222', paper_bgcolor='#222', font_color='#29E')
            matching_images.append(dcc.Graph(id='ret-fig', figure=fig))
    elif nn_d['Type'] == 'fashion':
        for i in image_list:
            colors = ['blue'] * 10
            colors[n_est] = 'green'
            colors[r_est] = 'red'
            fig = make_subplots(1, 2)
            img = np.array([[[255 - s, 255 - s, 255 - s] for s in r] for r in ti[i]], dtype="u1")

            fig.add_trace(go.Image(z=img), 1, 1)
            # px.imshow(mn.test_images[i], color_continuous_scale='gray')
            fig.add_trace(go.Bar(x=mc, y=predictions[i], marker_color=colors), 1, 2)
            fig.update_layout(height=250, width=480, plot_bgcolor='#222', paper_bgcolor='#222', font_color='#29E')
            matching_images.append(dcc.Graph(id='ret-fig', figure=fig))
    else:
        for i in image_list:
            colors = ['blue'] * 10
            colors[n_est] = 'green'
            colors[r_est] = 'red'
            fig = make_subplots(1, 2)
            img = np.array([[[s, s, s] for s in r] for r in ti[i]], dtype="u1")

            fig.add_trace(go.Image(z=img), 1, 1)
            # px.imshow(mn.test_images[i], color_continuous_scale='gray')
            fig.add_trace(go.Bar(x=mc, y=predictions[i], marker_color=colors), 1, 2)
            fig.update_layout(height=250, width=480, plot_bgcolor='#222', paper_bgcolor='#222', font_color='#29E')
            matching_images.append(dcc.Graph(id='ret-fig', figure=fig))

    im_row = dbc.Row(matching_images)

    title_str = "Model predicted " + estimate + ", actual image: " + reality + ". " + str(
        match) + " items incorrectly predicted. Prediction red, actual image green; in probability charts below."

    return str(active_cell), im_row, title_str

@app.callback([Output('output-data-upload', 'data'),
               Output('output-data-upload', 'columns')],
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    fname = os.path.join(os.getcwd(), "assets", list_of_names[0])

    with open(fname, 'rb') as file:
        nn_d = pickle.load(file)

    mn = MNISTBase(nn_d['Type'])
    conf_df = pd.DataFrame(nn_d['Confusion'], columns=nn_d['Labels'], index=nn_d['Labels'])
    conf_df.insert(0, "id", nn_d['Labels'])
    data = conf_df.to_dict('records')
    columns= [{"name": i, "id": i} for i in conf_df.columns]

    return data, columns

