from dash import Dash, dcc, html
import dash
import dash_bootstrap_components as dbc
import dash_daq as daq

model_name = 'test'
app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
app.layout = html.Div([html.Button('Run', id='run'), dcc.Input(id='model_name', value='Model1', type="text")])
app.run_server(debug=False)