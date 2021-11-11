import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from app import app
from dash.dependencies import Input, Output, State
from util_classes.Dimensionality import DisplayTSNE
import plotly.graph_objects as go
import pickle


layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H4("Pickle Uploader"),
            dcc.Upload(
                id='upload-tsne',
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
            html.P('Animation speed:'),
            dcc.Slider(id='speed-slider', min=0, max=3, marks={0: "Fast", 1: "Med", 2: "Slow", 3: "Crawl"}, className='dash-bootstrap', value=0),
            html.Div(id='tsne-params', children=[])
            ], width=2
        ),
        dbc.Col([
            dcc.Loading(dcc.Graph(id='output-data-tsne'), type='cube'),
        ], width={"size": 8, "offset": 0})
    ]),
])

@app.callback([Output('output-data-tsne', 'figure'),
               Output('tsne-params', 'children')],
              [Input('upload-tsne', 'filename'),
               Input('speed-slider', 'value')]
              )
def display_tse_bubble(fname, speed):
    print(fname)
    print(speed)
    if fname == None:
        fname = 'assets/tsne_p_30_e_12.0_d_2.pickle'
    else:
        fname = 'assets/' + fname[0]

    frame_speed = 20*(speed) + 1
    transition_speed = 20*(speed) + 1

    print((frame_speed, transition_speed))

    with open(fname, 'rb') as pkl:
        p_dict = pickle.load(pkl)

    df = p_dict['anim_df']
    pe = p_dict['perplexity']
    ee = p_dict['early_exaggeration']
    di = p_dict['dimensions']
    df = df.reset_index()
    df['labels'] = df['labels'].astype(str)

    xmin = df['x'].min()
    xmax = df['x'].max()
    ymin = df['y'].min()
    ymax = df['y'].max()

    if di == 2:
        fig = px.scatter(df, x='x', y='y', animation_frame='iteration', color='labels', height=950, width=1600,
                     range_x=[xmin, xmax], range_y=[ymin, ymax])
    else:
        zmin = df['z'].min()
        zmax = df['z'].max()
        fig = px.scatter_3d(df, x='x', y='y', z='z', animation_frame='iteration', color='labels', height=950, width=1600,
                     range_x=[xmin, xmax], range_y=[ymin, ymax], range_z=[zmin, zmax])
        fig.update_layout(scene_aspectmode='cube', scene=dict(
                          xaxis=dict(
                              backgroundcolor="#222",
                              gridcolor="white",
                              title="X",
                              # nticks=50,
                              showbackground=True,
                              zerolinecolor="white", ),
                          yaxis=dict(
                              backgroundcolor="#222",
                              gridcolor="white",
                              title="Y",
                              # nticks=50,
                              showbackground=True,
                              zerolinecolor="white"),
                          zaxis=dict(
                              backgroundcolor="#222",
                              gridcolor="white",
                              title="3D t_SNE Transformation",
                              showbackground=True,
                              zerolinecolor="white", ), ))

    fig.update_layout(plot_bgcolor='#222', paper_bgcolor='#222', font=dict(size=12, color='white'))
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = frame_speed
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = transition_speed
    dim = dbc.Row(dbc.Col(html.H6("Dimensions: " + str(di))))
    per = dbc.Row(dbc.Col(html.H6("Perplexity: " + str(pe))))
    eex = dbc.Row(dbc.Col(html.H6("Early Exaggeration: " + str(ee))))

    tsne_div = html.Div([dim, per, eex])

    return fig, tsne_div