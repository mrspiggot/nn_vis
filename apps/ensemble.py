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
import pandas as pd
from numpy import expand_dims
import pickle
from pathlib import Path
import os
import dash_table as dt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


# fname = os.path.join(os.getcwd(), "assets", "numbersOutput_ENS__7_20220103_102353_0.9836.pickle")
# print(fname)
# print(os.getcwd())
# with open(fname, 'rb') as pkl:
#     p_dict = pickle.load(pkl)
#
# print(p_dict)

tabs_styles = {
    'height': '44px',
    'width': '244px'
}
tab_style = {
    'borderBottom': '1px solid #222',
    'backgroundColor': '#222',
    'padding': '6px',
    # 'fontWeight': 'bold',
    'color': '#29E'
}

tab_selected_style = {
    'borderTop': '1px solid #222',
    'borderBottom': '1px solid #222',
    'backgroundColor': 'lightblue',
    'color': 'white',
    'padding': '6px'
}
amount_dict = {1: '10', 2: '20', 3: '50', 4: '100', 5: '200'}

layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H5("Upload *ENS*.pickle file")
        ], width={"size": 2, "offset": 0, "align": "center"}),
        dbc.Col(
            dcc.Upload(
                id='upload-ens',
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
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=True
            ),width={"size": 3, "offset": 0}),
        dbc.Col(daq.BooleanSwitch(id='ens-bool', on=True, label="Hard Vote", labelPosition="top", color="#29E"), width=1),
        dbc.Col(html.P("Max. Images"), width = 1),
        dbc.Col(dcc.Slider(min=1, max=5, marks=amount_dict, id="ens-slider", value=2),width=2),
        ]),
        dbc.Row([
            dbc.Col(
            dcc.Loading(html.Div(id='view-tabs', children=[], className='dash-bootstrap'), type='cube',
                        fullscreen=True, className='dash-bootstrap'), width={"size": 10, "offset": 1}
            )
        ])
    ])

def get_component_content(nn_d, order, fname):
    # # print(nn_d['Accuracy'], type(nn_d['Accuracy']))
    # print(nn_d, type(nn_d))
    # print(sorted(nn_d, key=lambda x: x['Accuracy'], reverse=True))
    if any(isinstance(el, list) for el in nn_d['Layer code']):
        # tstring = "[" + [str(i) for i in nn_d['Layer code'][0]] + "], "
        layer_string = [str(i) for i in nn_d['Layer code'][0]]
        string_layers = "[" + ",".join(layer_string) + "], "

        layer_count = 1
        for layers in nn_d['Layer code'][1:]:
            layer_string = [str(i) for i in layers]
            string_layers += "[" + ",".join(layer_string) + "], "
            layer_count+=1


        title = "CNN: {} ".format(layer_count) + string_layers[:-2]
    else:
        layer_string = str(nn_d['Layer code'][0])
        layer_count = 1
        for num in nn_d['Layer code'][1:]:
            layer_string += ", " + str(num)
            layer_count+=1
        title = "Layers {}: ".format(layer_count ) + " [" + layer_string + "]"

    conf_df = pd.DataFrame(nn_d['Confusion'], columns=nn_d['Labels'], index=nn_d['Labels'])
    conf_df.insert(0, "id", nn_d['Labels'])
    data = conf_df.to_dict('records')
    columns = [{"name": i, "id": i} for i in conf_df.columns]
    title = title + " Acc. " + str(round(nn_d['Test Accuracy'] * 100, 2)) + "%"
    ens_dt = dt.DataTable(id='ens-dt', data=data, columns=columns, style_data_conditional=[
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
                          })
    return ens_dt, title


def get_ensemble_content(nn_d, vote, quantity):
    mn = MNISTBase(nn_d['Type'])
    if vote==True:
        conf_type = "Confusion Votes"
        acc_type = "Vote Accuracy"
    else:
        conf_type = "Confusion"
        acc_type = "Test Accuracy"

    conf_df = pd.DataFrame(nn_d[conf_type], columns=nn_d['Labels'], index=nn_d['Labels'])
    conf_df.insert(0, "id", nn_d['Labels'])
    data = conf_df.to_dict('records')
    columns = [{"name": i, "id": i} for i in conf_df.columns]
    title = "Ensemble Accuracy " + str(round(nn_d[acc_type] * 100, 2)) + "%"
    ens_dt = dt.DataTable(id='ens-dt', data=data, columns=columns, style_data_conditional=[
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
                         })
    return ens_dt, title

def get_accuracy_rank(d):
    return d['Accuracy']

@app.callback([
               Output('view-tabs', 'children')
               ],
              [Input('upload-ens', 'contents'),
              Input('ens-slider', 'value'),
              Input('ens-bool', 'on')],
              State('upload-ens', 'filename'),
              State('upload-ens', 'last_modified'))
def update_output(list_of_contents, quantity, vote, list_of_names, list_of_dates):

    if list_of_names == None:
        d = {'Loss': [0], 'Accuracy': [0], 'Validation Loss': [0], 'Validation Accuracy': [0]}
        title = "Load a '*ENS*Pickle' file"
        # tab_layout=dcc.Tabs(id='null', children=[dcc.Tab(label=title)])
        tab_layout = " "
    else:

        fname = os.path.join(os.getcwd(), "assets", list_of_names[0])

        with open(fname, 'rb') as pkl:
            p_dict = pickle.load(pkl)

        # print("Type", p_dict, type(p_dict))
        n=0
        label_list = []
        accuracy_order = []
        for tb in p_dict['Ensemble files']:
            if n == 0:
                tc, title = get_ensemble_content(p_dict, vote, quantity)


            else:
                with open(tb[3:], 'rb') as comp_f:
                    c_dict = pickle.load(comp_f)
                    # accuracy_order.append(c_dict)
                # accuracy_order.sort(key=get_accuracy_rank)
                # for c_d in accuracy_order:
                tc, title = get_component_content(c_dict, n, tb[3:])

            n+=1

            my_tab = dcc.Tab(label=title, style=tab_style, children=[tc], selected_style=tab_selected_style, className='dash-bootstrap')
            label_list.append(my_tab)

        tab_layout = dcc.Tabs(id='dyn-tabs', value='tab-master', children=label_list)

    return [tab_layout]
