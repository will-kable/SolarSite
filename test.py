from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import dash_daq as daq

app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
app.layout = html.Div(
    [
        html.Div(
            [
                html.P('Analysis Parameters', style={'textAlign': 'center', 'width': '100%', "font-weight": "bold"}),
                html.Div(
                    [
                        html.P('Start Year', style={'display': 'inline-block', 'width': '35%'}),
                        dcc.Input(id={'type': 'param', 'name': 'StartYear'}, type='number', value=self.StartYear, step=1, style={'display': 'inline-block', 'width': '60%'}),
                    ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh', 'margin': '5px'}
                ),
                html.Div(
                    [
                        html.P('Analysis Period (years)', style={'display': 'inline-block', 'width': '35%'}),
                        dcc.Input(id={'type': 'param', 'name': 'Term'}, type='number', value=self.Term, step=1, style={'display': 'inline-block', 'width': '60%'}),
                    ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh', 'margin': '5px'}
                ),
                html.Div(
                    [
                        html.P('Infaltion Rate (%)', style={'display': 'inline-block', 'width': '35%'}),
                        dcc.Input(id={'type': 'param', 'name': 'InfaltionRate'}, type='number', value=self.InfaltionRate, step=0.1,
                                  style={'display': 'inline-block', 'width': '60%'}),
                    ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh', 'margin': '5px'}
                ),
                html.Div(
                    [
                        html.P('Discount Rate (%)', style={'display': 'inline-block', 'width': '35%'}),
                        dcc.Input(id={'type': 'param', 'name': 'DiscountRate'}, type='number',
                                  value=self.DiscountRate, step=0.1,
                                  style={'display': 'inline-block', 'width': '60%'}),
                    ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh', 'margin': '5px'}
                ),
                ], style={'width': '50%', "border": "2px black solid", 'display': 'block', 'vertical-align': 'top', 'border-radius':'15px', 'background-color': 'beige', 'margin': '10px'}
            ),
        html.Div(
            [
                html.P('Tax Parameters', style={'textAlign': 'center', 'width': '100%', "font-weight": "bold"}),
                html.Div(
                    [
                        html.P('Federal Tax Rate (%)', style={'display': 'inline-block', 'width': '35%'}),
                        dcc.Input(id={'type': 'param', 'name': 'FederalTaxRate'}, type='number', value=self.FederalTaxRate, step=1, style={'display': 'inline-block', 'width': '60%'}),
                    ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh', 'margin': '5px'}
                ),
                html.Div(
                    [
                        html.P('State Tax Rate (%)', style={'display': 'inline-block', 'width': '35%'}),
                        dcc.Input(id={'type': 'param', 'name': 'StateTaxRate'}, type='number', value=self.StateTaxRate,
                                  step=1, style={'display': 'inline-block', 'width': '60%'}),
                    ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh', 'margin': '5px'}
                ),
                html.Div(
                    [
                        html.P('State Sales Tax Rate (%)', style={'display': 'inline-block', 'width': '35%'}),
                        dcc.Input(id={'type': 'param', 'name': 'SalesTaxRate'}, type='number', value=self.SalesTaxRate,
                                  step=1, style={'display': 'inline-block', 'width': '60%'}),
                    ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh', 'margin': '5px'}
                ),
            ], style={'width': '50%', "border": "2px black solid", 'display': 'block', 'vertical-align': 'top',
                      'border-radius': '15px', 'background-color': 'beige', 'margin': '10px'}
        ),
    ]
)



app.run_server(debug=False)