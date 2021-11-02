import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
from app import app

from apps import show_images



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
        print("Hello")
        return show_images.layout
    # if pathname == '/results':
    #     return results.layout
    # if pathname == '/generate':
    #     return generate.layout(layers)

if __name__ == '__main__':
    app.run_server(debug=True, port=8025)
