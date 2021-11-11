import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
from app import app

from apps import show_images, build, tsne, CNN_viz



navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="assets/lucidate.png", height="90px")),
                        dbc.Col(dbc.NavbarBrand("AI Dashboard", className="ml-2", href="/home")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="https://websites.looka.com/preview/06d736f4?device=desktop",
            ),
            dbc.Collapse(
                dbc.Nav(
                    [dbc.NavItem(dbc.NavLink("Training images", href="/show_images", disabled=False)),
                     dbc.NavItem(dbc.NavLink("CNN Filters", href="/cnn", disabled=False)),
                     dbc.NavItem(dbc.NavLink("CNN Embeddings", href="/embed", disabled=False)),
                    dbc.NavItem(dbc.NavLink("Output Layer", href="/build", disabled=False)),
                     dbc.NavItem(dbc.NavLink("Show t-SNE", href="/tsne", disabled=False)),
                     ], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            ),

        ]
    ),
    color="dark",
    dark=True,
    className="mb-5",
)




app.layout = html.Div([
    dcc.Location(id='page-url'),
    navbar,
    html.Div(id='page-content', children=[]),

])

@app.callback(Output('page-content', 'children'),
              [Input('page-url', 'pathname')])
def display_page(pathname):
    if pathname == '/show_images':
        return show_images.layout
    if pathname == '/build':
        return build.layout
    if pathname == '/tsne':
        return tsne.layout
    if pathname == '/cnn':
        return CNN_viz.layout
    # if pathname == '/generate':
    #     return generate.layout(layers)

if __name__ == '__main__':
    app.run_server(debug=True, port=8025)
