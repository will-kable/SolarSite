from dash import Dash, dcc, html
import dash
import dash_bootstrap_components as dbc
import dash_daq as daq

model_name = 'test'
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.layout = html.Div(
            [
                html.Div(
                    [
                        html.Div([
                            html.Button('Run', id='run', style={'width': '25%', 'font-size': '20px'}),
                            dcc.Input(id='model_name', value='Model1', type="text", style={'width': '75%', 'font-size': '20px'})
                        ], style={'display': 'flex', 'height': '5vh', 'margin': '10px'}
                        ),
                    ], style={'width': '30%', "border": "2px black solid", 'display': 'block', 'vertical-align': 'top',
                              'border-radius': '15px', 'background-color': 'beige', 'margin': '10px'}
                ),
            ]
        )

app.run_server(debug=False)