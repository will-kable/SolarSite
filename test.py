from dash import Dash, dcc, html
import dash
import dash_bootstrap_components as dbc
import dash_daq as daq

model_name = 'test'
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.layout = html.Div([
    html.P(f'County: ',
           id={'type': 'summary_data_table_label', 'name': model_name},
           style={'text-align': 'center', 'width': '100%', "font-weight": "bold", 'height': '10vh',
                  'border': '2px black solid', 'padding-top': '5vh', 'font-size': '200px'}
           )
],
)
app.run_server(debug=False)