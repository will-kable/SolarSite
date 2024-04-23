from dash import Dash, dcc, html
import dash
import dash_bootstrap_components as dbc
import dash_daq as daq

model_name = 'test'
app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
app.layout =                         html.Div(
                            [
                                dcc.Tabs(id='sumamry-tabs', value='Summary', children=[
                                    dcc.Tab(label='Summary',
                                            value='Summary',
                                            children=html.Div([
                                                html.Div(dash.dash_table.DataTable(id={'type': 'summary_data_table', 'name': model_name}, export_format='csv', style_table={'overflowX': 'scroll'}), style={'height': '65vh', 'width': '50%', 'display': 'inline-block'}),
                                                dcc.Graph(id={'type': 'x-graph', 'name': model_name}, style={'height': '65vh', 'width': '50%', 'display': 'inline-block'}),
                                                ], style={'width': '100%', 'display': 'inline-block', 'border': '2px black solid'}
                                            )
                                            ),
                                    dcc.Tab(label='Cashflows',
                                            value='Cashflows',
                                            children=html.Div(dash.dash_table.DataTable(id={'type': 'data_table', 'name': model_name}, export_format='csv', style_table={'overflowX': 'scroll'}), style={'background-color': 'whitesmoke', 'height': '65vh'})
                                            )]),
                                html.Button('Delete Tab', id={'type': 'delete', 'name': model_name})
                            ]
                        )
app.run_server(debug=False)