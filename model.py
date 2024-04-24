import datetime
import os
from copy import deepcopy

import PySAM.Merchantplant as MERC
import PySAM.Pvwattsv8 as PVW
import PySAM.ResourceTools as RT
import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import geopy.distance
import matplotlib.pyplot as plt
import numpy
import pandas
import requests
from dash import Dash, dcc, html, ALL, MATCH, no_update
from dash.dash_table.Format import Format
from dateutil import relativedelta
from multiprocessing import Pool

from common import closest_key, timeshape
from config import *

timeout = 60

plt.interactive(True)


class BaseModel:
    def __init__(self, models=[], update_sub_parmas=True):
        self.S1 = datetime.datetime.now()
        self.update_model(models, update_sub_parmas)
        self.ModelsIn = models

    def __repr__(self):
        return self.__class__.__name__

    @property
    def ExecutionTime(self):
        return (self.S2 - self.S1).total_seconds()

    @property
    def get_dict(self):
        return dict(vars(self))

    def update_model(self, models, update_sub_parmas=True):
        for model in models:
            setattr(self, model.__class__.__name__, model)
            if update_sub_parmas:
                for key, value in model.__dict__.items():
                    if key not in ['S1', 'S2']:
                        setattr(self, key, value)


class LocationModel(BaseModel):
    def __init__(self, lat, long):
        super().__init__()
        self.Lat = lat
        self.Long = long
        self.Index = ((COUNTIES['X (Lat)'] - lat).abs() + (COUNTIES['Y (Long)'] - long).abs()).idxmin()
        self.Info = COUNTIES.loc[self.Index]
        self.Name = self.Info['CNTY_NM']
        self.FIPS = self.Info['FIPS']
        self.TimeZone = FINDER.timezone_at(lng=self.Long, lat=self.Lat)
        self.Zone = self.Info['ZONE'] or closest_key(ZONAL_MAP, (lat, long))
        self.IsERCOT = True if self.Info['ZONE'] else False
        self.LandPrice = LAND_PRICES[self.Name]['Data']['2023']
        self.PropertyTax = PROP_TAXES[self.Name]
        self.LandRegion = LAND_PRICES[self.Name]['Region']
        self.TransDistanceDict = self.trans_distance()
        self.S2 = datetime.datetime.now()

    def __repr__(self):
        return self.Name

    def land_cost(self, area):
        return self.LandPrice * area

    def land_tax(self, area):
        return self.land_cost(area) * self.PropertyTax / 100

    def trans_distance(self):
        dct = {}
        for voltage in {'220-287', '100-161', '345', '500'}:
            key = f'{voltage.replace("-", "_")}_kV'
            temp_df = TRANS_CORDS[TRANS_CORDS['Voltage'] == voltage]
            if len(temp_df):
                idx = ((temp_df.Lat - self.Lat).pow(2) + (temp_df.Lon - self.Long).pow(2)).pow(0.5).idxmin()
                cord = temp_df.loc[idx, ['Lat', 'Lon']].tolist()
                dct[key] = geopy.distance.geodesic(cord, (self.Lat, self.Long)).miles
            else:
                dct[key] = 9e9
            setattr(self, key, dct[key])
        return {}


class PeriodModel(BaseModel):
    def __init__(self, sd, ed, timezone):
        super().__init__()
        self.StartDate = sd
        self.EndDate = ed
        self.TimeZone = timezone
        self.Forward = pandas.date_range(sd, ed + datetime.timedelta(days=1), freq='h', tz=self.TimeZone)[1:]
        self.ForwardMerge = self.attributes(self.Forward)
        self.Years = list(range(self.StartDate.year, self.EndDate.year + 1))
        self.CashFlowYears = list(range(self.StartDate.year - 1, self.EndDate.year + 1))
        self.S2 = datetime.datetime.now()

    def __repr__(self):
        return f'{self.__class__.__name__} {self.StartDate} {self.EndDate} {self.TimeZone}'

    def attributes(self, df):
        if not isinstance(df, pandas.DataFrame):
            df = df.to_frame('index').set_axis(['index'], axis=1)
        hb = df.shift(freq='-1h').index
        attr = {
            'Year': hb.year,
            'Month': hb.month,
            'Day': hb.day,
            'Hour': hb.hour,
            'HourEnding': hb.hour + 1,
            'Date': hb.date,
            'DateMonth': pandas.to_datetime({'year': hb.year, 'month': hb.month, 'day': 1}).dt.date,
            'TimeShape': hb.map(timeshape),
        }
        df = df.reset_index().assign(**attr)
        df = df.assign(Idx=df.groupby(['Date', 'TimeShape'])['Year'].cumcount())
        df = df.assign(DST=df.groupby(['Date', 'TimeShape'])['Year'].transform('count').isin([7, 9]))
        return df.set_index(df.columns[0]).rename_axis('index')


class TechnologyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.Capacity = 100
        self.DCRatio = 1.28
        self.Azimuth = 180
        self.InvEff = 96
        self.Losses = 14.07
        self.ArrayType = '1 Axis Tracker'
        self.GCR = 0.3
        self.DegradationRate = 0.5
        self.ModuleType = 'Premium'
        self.S2 = datetime.datetime.now()

    @property
    def ModuleEfficiency(self):
        return MODULE_EFF[self.ModuleType]

    @property
    def CapacityAC(self):
        return self.Capacity / self.DCRatio

    @property
    def LandArea(self):
        return self.Capacity * 1000 / self.ModuleEfficiency * 0.000247105 / self.GCR

    def render(self):
        return html.Div(
            [
                html.Div(
                    [
                        html.P('System Parameters',
                               style={'textAlign': 'center', 'width': '100%', "font-weight": "bold"}),
                        html.Div(
                            [
                                html.P('Capacity (MW)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'Capacity'}, type='number', value=self.Capacity,
                                          style={'display': 'inline-block', 'width': '60%'}, step=10),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Module Type', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Dropdown(id={'type': 'param', 'name': 'ModuleType'},
                                             options=['Standard', 'Premium', 'Thin Film'],
                                             value=self.ModuleType, style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('DC Ratio', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'DCRatio'}, type='number', value=self.DCRatio,
                                          step=0.01,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Inverter Efficiency (%)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'InvEff'}, type='number', value=self.InvEff,
                                          step=1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                    ], style={'width': '50%', "border": "2px black solid", 'display': 'block', 'vertical-align': 'top',
                              'border-radius': '15px', 'background-color': 'beige', 'margin': '10px'}
                ),
                html.Div(
                    [
                        html.P('System Orientation',
                               style={'textAlign': 'center', 'width': '100%', "font-weight": "bold"}),
                        html.Div(
                            [
                                html.P('Array Type', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Dropdown(id={'type': 'param', 'name': 'ArrayType'},
                                             options=['Fixed', 'Fixed Roof', '1 Axis Tracker', '1 Axis Backtracked',
                                                      '2 Axis Tracker'],
                                             value=self.ArrayType, style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Azimuth', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'Azimuth'}, type='number', value=self.Azimuth,
                                          step=1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Ground Coverage Ratio', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'GCR'}, type='number', value=self.GCR, step=0.01,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),

                    ], style={'width': '50%', "border": "2px black solid", 'display': 'block', 'vertical-align': 'top',
                              'border-radius': '15px', 'background-color': 'beige', 'margin': '10px'}
                ),
                html.Div(
                    [
                        html.P('System Losses', style={'textAlign': 'center', 'width': '100%', "font-weight": "bold"}),
                        html.Div(
                            [
                                html.P('Total Losses (%)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'Losses'}, type='number', value=self.Losses,
                                          step=0.01,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Annual Degradation (%/year)',
                                       style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'DegredationRate'}, type='number',
                                          step=0.1, value=self.DegradationRate,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                    ], style={'width': '50%', "border": "2px black solid", 'display': 'block', 'vertical-align': 'top',
                              'border-radius': '15px', 'background-color': 'beige', 'margin': '10px'}
                )
            ], style={'background-color': 'whitesmoke'}
        )


class RevenueModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.IncludeCannib = True
        self.IncludeCurtailment = True
        self.IncludeRECs = True
        self.FederalPTC = 0.0275
        self.FederalPTCTerm = 10
        self.StatePTC = 0
        self.StatePTCTerm = 0
        self.S2 = datetime.datetime.now()

    def render(self):
        return html.Div(
            [
                html.Div(
                    [
                        html.P('Revenue Adjustments',
                               style={'textAlign': 'center', 'width': '100%', "font-weight": "bold"}),

                        dcc.Checklist(
                            [
                                {
                                    "label": html.Div(['Include Cannibalization Adjustment'],
                                                      style={
                                                          'color': 'Black',
                                                          'font-size': 18,
                                                          'justify-content': 'center',
                                                          "align-items": "center",
                                                          'margin': '10px',
                                                      }),
                                    "value": 'Include Cannibalization Adjustment',
                                },
                            ],
                            value=['Include Cannibalization Adjustment'] if self.IncludeCannib else [],
                            id={'type': 'param', 'name': 'IncludeCannib'},
                            labelStyle={"display": "flex", 'justify-content': 'center', "align-items": "center"}
                        ),

                        dcc.Checklist(
                            [
                                {
                                    "label": html.Div(['Include Curtailment Adjustment'],
                                                      style={
                                                          'color': 'Black',
                                                          'font-size': 18,
                                                          'justify-content': 'center',
                                                          "align-items": "center",
                                                          'margin': '10px',
                                                      }),
                                    "value": 'Include Curtailment Adjustment',
                                },
                            ],
                            value=['Include Curtailment Adjustment'] if self.IncludeCurtailment else [],
                            id={'type': 'param', 'name': 'IncludeCurtailment'},
                            labelStyle={"display": "flex", 'justify-content': 'center', "align-items": "center"}
                        ),

                        dcc.Checklist(
                            [
                                {
                                    "label": html.Div(['Include REC Value'],
                                                      style={
                                                          'color': 'Black',
                                                          'font-size': 18,
                                                          'justify-content': 'center',
                                                          "align-items": "center",
                                                          'margin': '10px',
                                                      }),
                                    "value": 'Include REC Value',
                                },
                            ],
                            value=['Include REC Value'] if self.IncludeRECs else [],
                            id={'type': 'param', 'name': 'IncludeRECs'},
                            labelStyle={"display": "flex", 'justify-content': 'center', "align-items": "center"}
                        ),
                    ], style={'width': '50%', "border": "2px black solid", 'display': 'block', 'vertical-align': 'top',
                              'border-radius': '15px', 'background-color': 'beige', 'margin': '10px', "align-items": "center"}
                ),
                html.Div(
                    [
                        html.P('Production Incentives',
                               style={'textAlign': 'center', 'width': '100%', "font-weight": "bold"}),
                        html.Div(
                            [
                                html.P('Federal Production Tax Credit ($/MWh)',
                                       style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'FederalPTC'}, type='number',
                                          value=self.FederalPTC, step=0.0001,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Federal Production Tax Credit Term (Years)',
                                       style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'FederalPTCTerm'}, type='number',
                                          value=self.FederalPTCTerm, step=1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('State Production Tax Credit ($/MWh)',
                                       style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'StatePTC'}, type='number', value=self.StatePTC,
                                          step=0.0001, style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('State Production Tax Credit Term (Years)',
                                       style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'StatePTCTerm'}, type='number', value=self.StatePTCTerm,
                                          step=1, style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                    ], style={'width': '50%', "border": "2px black solid", 'display': 'block', 'vertical-align': 'top',
                              'border-radius': '15px', 'background-color': 'beige', 'margin': '10px'}
                ),
            ]
        )


class CapitalCostModel(BaseModel):
    def __init__(self):
        super().__init__()
        # https://atb.nrel.gov/electricity/2023/utility-scale_pv#tech-innovations-table-upv
        self.ModuleCost = 0.33
        self.InverterCost = 0.04
        self.EquipmentCost = 0.18
        self.LaborCost = 0.1
        self.OverheadCost = 0.05
        self.ContingencyCost = 0.03
        self.PermittingCost = 0
        self.GridConnectionCost = 0.03
        self.EngineeringCost = 0.08
        # https://www.nrel.gov/docs/fy22osti/83586.pdf
        self.TransmissionCost = 600734

        self.S2 = datetime.datetime.now()

    def direct_cost(self, capacity):
        direct = (self.ModuleCost + self.InverterCost + self.EquipmentCost + self.LaborCost + self.OverheadCost)
        direct += (direct * self.ContingencyCost)
        return direct * 1e6 * capacity

    def indirect_cost(self, capacity):
        indirect = (self.PermittingCost + self.GridConnectionCost + self.EngineeringCost)
        return indirect * 1e6 * capacity

    def render(self):
        return html.Div(
            [
                html.Div(
                    [
                        html.P('Direct Costs', style={'textAlign': 'center', 'width': '100%', "font-weight": "bold"}),
                        html.Div(
                            [
                                html.P('Module Cost ($/Wdc)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'ModuleCost'}, type='number',
                                          value=self.ModuleCost, step=0.01,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Inverter Cost ($/Wdc)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'InverterCost'}, type='number',
                                          value=self.InverterCost, step=0.01,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Equipment Cost ($/Wdc)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'EquipmentCost'}, type='number',
                                          value=self.EquipmentCost, step=0.01,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Labor Cost ($/Wdc)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'LaborCost'}, type='number',
                                          value=self.LaborCost, step=0.01,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Overhead Cost ($/Wdc)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'OverheadCost'}, type='number',
                                          value=self.OverheadCost, step=0.01,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Contingency Cost (% of total Direct Costs)',
                                       style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'ContingencyCost'}, type='number',
                                          value=self.ContingencyCost, step=0.01,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Transmission Cost ($/mile)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'TransmissionCost'}, type='number',
                                          value=self.TransmissionCost,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                    ], style={'width': '50%', "border": "2px black solid", 'display': 'block', 'vertical-align': 'top',
                              'border-radius': '15px', 'background-color': 'beige', 'margin': '10px'}
                ),
                html.Div(
                    [
                        html.P('Indirect Costs', style={'textAlign': 'center', 'width': '100%', "font-weight": "bold"}),
                        html.Div(
                            [
                                html.P('Permitting Cost ($/Wdc)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'PermittingCost'}, type='number',
                                          value=self.PermittingCost, step=0.01,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Grid Connection Cost ($/Wdc)',
                                       style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'GridConnectionCost'}, type='number',
                                          value=self.GridConnectionCost, step=0.01,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Engineering Overhead Cost ($/Wdc)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'EngineeringCost'}, type='number',
                                          value=self.EngineeringCost, step=0.01,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                    ], style={'width': '50%', "border": "2px black solid", 'display': 'block', 'vertical-align': 'top',
                              'border-radius': '15px', 'background-color': 'beige', 'margin': '10px'}
                ),
            ]
        )


class OperatingCostModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.AnnualCost = 0
        self.AnnualCostByCapacity = 23
        self.VariableCostGen = 0
        self.S2 = datetime.datetime.now()

    def render(self):
        return html.Div(
            [
                html.Div(
                    [
                        html.P('Operation and Maintenance Costs',
                               style={'textAlign': 'center', 'width': '100%', "font-weight": "bold"}),
                        html.Div(
                            [
                                html.P('Annual Fixed Cost ($/yr)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'AnnualCost'}, type='number',
                                          value=self.AnnualCost, step=1e6,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Annual Fixed Cost by Capacity ($/(kWac-yr))',
                                       style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'AnnualCostByCapacity'}, type='number',
                                          value=self.AnnualCostByCapacity, step=1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Variable Cost by Generation ($/MWhac)',
                                       style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'VariableCostGen'}, type='number',
                                          value=self.VariableCostGen, step=1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                    ], style={'width': '60%', "border": "2px black solid", 'display': 'block', 'vertical-align': 'top',
                              'border-radius': '15px', 'background-color': 'beige', 'margin': '10px'}
                ),
            ]
        )


class FinanceModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.Term = 25
        self.StartYear = datetime.date.today().year + 1
        self.InflationRate = 2.5
        self.DiscountRate = 6.4
        # https://www.nrel.gov/docs/fy22osti/80694.pdf
        self.DebtFraction = 71.8
        self.DebtUpfrontFee = 2.75
        self.DebtInterestRate = 7
        self.DebtTenor = 18
        self.InsuranceRate = 0
        self.FederalTaxRate = 21
        self.StateTaxRate = 7
        self.SalesTaxRate = 6.25
        self.FederalInvestmentTaxCredit = 0
        self.EquipmentReserve = 0.1
        self.EquipmentReserveFreq = 15
        self.ReserveInterest = 1.75
        self.ReserveOperatingMonths = 6
        self.StateInvestmentTaxCredit = 0
        self.S2 = datetime.datetime.now()

    @property
    def StartDate(self):
        return datetime.date(self.StartYear, 1, 1)

    @property
    def EndDate(self):
        return self.StartDate + relativedelta.relativedelta(years=self.Term, days=-1)

    @property
    def Years(self):
        return list(range(self.StartYear, self.EndDate.year + 1))

    def render(self):
        return html.Div(
            [
                html.Div(
                    [
                        html.P('Analysis Parameters',
                               style={'textAlign': 'center', 'width': '100%', "font-weight": "bold"}),
                        html.Div(
                            [
                                html.P('Start Year', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'StartYear'}, type='number',
                                          value=self.StartYear, step=1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Analysis Period (years)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'Term'}, type='number', value=self.Term, step=1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Infaltion Rate (%)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'InfaltionRate'}, type='number',
                                          value=self.InflationRate, step=0.1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Discount Rate (%)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'DiscountRate'}, type='number',
                                          value=self.DiscountRate, step=0.1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                    ], style={'width': '50%', "border": "2px black solid", 'display': 'block', 'vertical-align': 'top',
                              'border-radius': '15px', 'background-color': 'beige', 'margin': '10px'}
                ),
                html.Div(
                    [
                        html.P('Debt Parameters',
                               style={'textAlign': 'center', 'width': '100%', "font-weight": "bold"}),
                        html.Div(
                            [
                                html.P('Debt Fraction (%)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'DebtFraction'}, type='number',
                                          value=self.DebtFraction, step=0.1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Debt Tenor (Years)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'DebtTenor'}, type='number',
                                          value=self.DebtTenor, step=1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Debt Interest Rate (%)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'DebtInterestRate'}, type='number',
                                          value=self.DebtInterestRate, step=1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Debt Upfront Fee (%)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'DebtUpfrontFee'}, type='number',
                                          value=self.DebtUpfrontFee, step=0.01,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Insurance Rate (%)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'InsuranceRate'}, type='number',
                                          value=self.InsuranceRate, step=1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                    ], style={'width': '50%', "border": "2px black solid", 'display': 'block', 'vertical-align': 'top',
                              'border-radius': '15px', 'background-color': 'beige', 'margin': '10px'}

                ),
                html.Div(
                    [
                        html.P('Tax Parameters', style={'textAlign': 'center', 'width': '100%', "font-weight": "bold"}),
                        html.Div(
                            [
                                html.P('Federal Tax Rate (%)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'FederalTaxRate'}, type='number',
                                          value=self.FederalTaxRate, step=1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('State Tax Rate (%)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'StateTaxRate'}, type='number',
                                          value=self.StateTaxRate,
                                          step=1, style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('State Sales Tax Rate (%)', style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'SalesTaxRate'}, type='number',
                                          value=self.SalesTaxRate,
                                          step=0.01, style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                    ], style={'width': '50%', "border": "2px black solid", 'display': 'block', 'vertical-align': 'top',
                              'border-radius': '15px', 'background-color': 'beige', 'margin': '10px'}
                ),
                html.Div(
                    [
                        html.P('Investment Incentives',
                               style={'textAlign': 'center', 'width': '100%', "font-weight": "bold"}),
                        html.Div(
                            [
                                html.P('Federal Investment Tax Credit (%)',
                                       style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'FederalInvestmentTaxCredit'}, type='number',
                                          value=self.FederalInvestmentTaxCredit, step=1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('State Investment Tax Credit (%)',
                                       style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'StateInvestmentTaxCredit'}, type='number',
                                          value=self.StateInvestmentTaxCredit, step=1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                    ], style={'width': '50%', "border": "2px black solid", 'display': 'block', 'vertical-align': 'top',
                              'border-radius': '15px', 'background-color': 'beige', 'margin': '10px'}
                ),
                html.Div(
                    [
                        html.P('Cash Reserves',
                               style={'textAlign': 'center', 'width': '100%', "font-weight": "bold"}),
                        html.Div(
                            [
                                html.P('Months of Operating Cost Reserve (Months)',
                                       style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'ReserveOperatingMonths'}, type='number',
                                          value=self.ReserveOperatingMonths, step=1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Interest on Reserves (%/year)',
                                       style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'ReserveInterest'}, type='number',
                                          value=self.ReserveInterest, step=0.01,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Equipment Reserve ($/Wdc)',
                                       style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'EquipmentReserve'}, type='number',
                                          value=self.EquipmentReserve, step=0.1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                        html.Div(
                            [
                                html.P('Equipment Replacement Frequency (years)',
                                       style={'display': 'inline-block', 'width': '35%'}),
                                dcc.Input(id={'type': 'param', 'name': 'EquipmentReserveFreq'}, type='number',
                                          value=self.EquipmentReserveFreq, step=1,
                                          style={'display': 'inline-block', 'width': '60%'}),
                            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'height': '5vh',
                                      'margin': '5px'}
                        ),
                    ], style={'width': '50%', "border": "2px black solid", 'display': 'block', 'vertical-align': 'top',
                              'border-radius': '15px', 'background-color': 'beige', 'margin': '10px'}
                ),
            ]
        )


class DiscountModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.DiscountCurve = self.discount_curve()
        self.S2 = datetime.datetime.now()

    def discount_curve(self):
        df = pandas.read_csv('Assets/discount_curve.csv', index_col=0).rename(pandas.to_datetime).resample('d').interpolate()
        return df.set_axis(df.index.date)['Discount']

    def discount(self, date):
        return self.DiscountCurve.loc[date]


class WeatherModel(BaseModel):
    def __init__(self, models):
        super().__init__(models)
        self.SolarResourceFile = self.resource_file()
        self.Fixed = self.pull_historical()
        self.AverageDNI = self.Fixed['DNI'].mean()
        self.AverageDHI = self.Fixed['DHI'].mean()
        self.S2 = datetime.datetime.now()

    def __repr__(self):
        return self.__class__.__name__ + ' ' + self.LocationModel.Name

    def resource_file(self):
        lon, lat = self.LocationModel.Long, self.LocationModel.Lat
        fetcher = RT.FetchResourceFiles(
            'solar',
            API_KEY_NREL,
            'will.kable@axpo.com',
            resource_dir='data\\Weather',
            resource_type='psm3-2-2-tmy',
            workers=512,
            verbose=False,
        )

        fetcher = fetcher.fetch([(lon, lat)])
        return fetcher.resource_file_paths[0]

    def pull_historical(self):
        info = pandas.read_csv(self.SolarResourceFile, nrows=1)
        timezone, elevation = info['Local Time Zone'], info['Elevation']
        self.TimeZone = timezone[0]
        self.Elevation = elevation[0]
        df = pandas.read_csv(
            self.SolarResourceFile,
            skiprows=2,
            usecols=lambda x: 'Unnamed' not in x
        )
        return df


class SolarModel(BaseModel):
    def __init__(self, models):
        super().__init__(models, False)
        self.Tilt = self.optimal_angle()
        self.LCOEReal = 0
        self.LCOENom = 0
        self.Unfixed = self.simulate()
        self.HourlyProfile = self.Unfixed.groupby(['Month', 'HourEnding'])['MW'].mean().unstack()
        self.MonthlyProfile = self.Unfixed.groupby('Month')['MW'].mean()
        self.AverageMW = self.Unfixed.MW.mean()
        self.S2 = datetime.datetime.now()

    def __repr__(self):
        return self.__class__.__name__ + ' ' + self.LocationModel.Name

    def optimal_angle(self):
        lat, lon = self.LocationModel.Lat, self.LocationModel.Long
        if os.path.exists(f'Config/Angles/{lon}_{lat}.csv'):
            return pandas.read_csv(f'Config/Angles/{lon}_{lat}.csv', names=[0], skiprows=1).iloc[0, 0]
        angle = requests.get(f'https://api.globalsolaratlas.info/data/lta?loc={lat},{lon}').json()['annual']['data']['OPTA']
        pandas.Series([angle]).to_csv(f'Config/Angles/{lon}_{lat}.csv')
        return angle

    def total_installed_cost(self):
        self.DirectCost = self.CapitalCostModel.direct_cost(self.TechnologyModel.Capacity)
        self.TransmissionCost = self.LocationModel.get_dict['100_161_kV'] * self.CapitalCostModel.TransmissionCost
        self.IndirectCosts = self.CapitalCostModel.indirect_cost(self.TechnologyModel.Capacity) + self.TransmissionCost
        self.LandCost = self.LocationModel.land_cost(self.TechnologyModel.LandArea)
        self.SalesTax = self.FinanceModel.SalesTaxRate / 100 * (self.DirectCost)
        self.TotalInstalledCost = self.DirectCost + self.IndirectCosts + self.LandCost + self.SalesTax
        self.TotalInstalledCostPerCapacity = self.TotalInstalledCost / (self.TechnologyModel.Capacity * 1E6)
        return self.TotalInstalledCost

    def total_construction_financing_cost(self):
        princ = self.total_installed_cost()
        interest = princ * 6.5 / 100 / 12 * 6 / 2
        upfront = princ * 1 / 100
        return interest + upfront

    def simulate(self):
        print('Simulating: ', self.LocationModel.Name, self.LocationModel.Index)
        df = self.WeatherModel.Fixed

        # Initialize the Models
        PV_MODEL = PVW.default('PVWattsMerchantPlant')
        MERC_MODEL = MERC.default('PVWattsMerchantPlant')

        PV_MODEL.assign(
            {
                'Lifetime': {
                    'analysis_period': self.FinanceModel.Term,
                    'system_use_lifetime_output': 0,
                    'dc_degradation': [0],
                },
                'SystemDesign': {
                    'system_capacity': self.TechnologyModel.Capacity * 1000,
                    'dc_ac_ratio': self.TechnologyModel.DCRatio,
                    'tilt': 0 if ARRAY_MAP[self.TechnologyModel.ArrayType] >= 2 else self.Tilt,
                    'azimuth': self.TechnologyModel.Azimuth,
                    'inv_eff': self.TechnologyModel.InvEff,
                    'losses': self.TechnologyModel.Losses,
                    'array_type': ARRAY_MAP[self.TechnologyModel.ArrayType],
                    'module_type': MODULE_MAP[self.TechnologyModel.ModuleType],
                    'gcr': self.TechnologyModel.GCR,
                },
                'SolarResource': {
                    'solar_resource_file': self.WeatherModel.SolarResourceFile
                },
            }
        )

        PV_MODEL.execute()
        out_solar = PV_MODEL.Outputs.export()

        # Build the Solar Models
        gen = pandas.concat([df.assign(Year=year) for year in self.PeriodModel.Years])
        idx = pandas.to_datetime(gen[['Year', 'Month', 'Day', 'Hour']]).dt.tz_localize(f'Etc/GMT+{-self.WeatherModel.TimeZone}').dt.tz_convert('US/Central')
        gen = gen.assign(HourEnding=idx.dt.hour + 1).set_index(idx).sort_index().shift(freq='1h').assign(Gen=out_solar['gen'] * self.FinanceModel.Term)

        # Curtail the generation if turned on
        curtailed_gen = pandas.Series(out_solar['gen'])
        if bool(self.RevenueModel.IncludeCurtailment):
            curtailment_perc = df.merge(self.CurtailmentModel.Unfixed)['ERCOT']
            curtailed_gen = (1 - curtailment_perc) * curtailed_gen

        # Build the price models
        energy_prices = self.EnergyModel.Unfixed.reindex(gen.index).fillna(0)
        cannib_prices = self.CannibalizationModel.Unfixed.reindex(gen.index).fillna(0) * bool(self.RevenueModel.IncludeCannib)
        rec_prices = self.RECModel.Unfixed.reindex(gen.index).fillna(0) * bool(self.RevenueModel.IncludeRECs)
        final_prices = energy_prices + cannib_prices + rec_prices

        MERC_MODEL.assign(
            {
                'SystemOutput': {
                    'gen': curtailed_gen.tolist(),
                    'system_pre_curtailment_kwac': out_solar['gen'],
                    'degradation': [self.TechnologyModel.DegradationRate],
                    'system_capacity': self.TechnologyModel.Capacity * 1000,
                },

                'Lifetime': {'system_use_lifetime_output': 0},

                'FinancialParameters': {
                    'analysis_period': len(self.PeriodModel.Years),

                    'real_discount_rate': self.FinanceModel.DiscountRate,
                    'inflation_rate': self.FinanceModel.InflationRate,
                    'state_tax_rate': [self.FinanceModel.StateTaxRate],
                    'federal_tax_rate': [self.FinanceModel.FederalTaxRate],
                    'property_tax_rate': self.LocationModel.PropertyTax,
                    'insurance_rate': self.FinanceModel.InsuranceRate,
                    'construction_financing_cost': self.total_construction_financing_cost(),
                    'system_capacity': self.TechnologyModel.Capacity * 1000,

                    'debt_option': 0,
                    'debt_percent': self.FinanceModel.DebtFraction,
                    'term_int_rate': self.FinanceModel.DebtInterestRate,
                    'term_tenor': self.FinanceModel.DebtTenor,
                    'cost_debt_fee': self.FinanceModel.DebtUpfrontFee,
                    'months_working_reserve': self.FinanceModel.ReserveOperatingMonths,
                    'reserves_interest': self.FinanceModel.ReserveInterest,
                    'equip1_reserve_freq': self.FinanceModel.EquipmentReserveFreq,
                    'equip1_reserve_cost': self.FinanceModel.EquipmentReserve
                },

                'Depreciation': {
                    'depr_alloc_macrs_5_percent': 100,
                    'depr_alloc_macrs_15_percent': 0,
                    'depr_alloc_sl_15_percent': 0,
                    'depr_alloc_sl_20_percent': 0,
                    'depr_bonus_fed_macrs_15': 1,
                    'depr_bonus_sta_macrs_15': 1,
                },

                'Revenue': {
                    'mp_energy_market_revenue_single': final_prices.to_frame().to_records(index=False).tolist(),
                    'mp_market_percent_gen': 100,
                },

                'LandLease': {
                    'land_area': self.TechnologyModel.LandArea,
                },

                'SystemCosts': {
                    'om_fixed': [self.OperatingCostModel.AnnualCost],
                    'om_capacity': [self.OperatingCostModel.AnnualCostByCapacity / self.TechnologyModel.DCRatio],
                    'om_production': [self.OperatingCostModel.VariableCostGen],
                    'total_installed_cost': self.total_installed_cost(),
                },

                'TaxCreditIncentives': {
                    'itc_fed_percent': [self.FinanceModel.FederalInvestmentTaxCredit],
                    'itc_sta_percent': [self.FinanceModel.StateInvestmentTaxCredit],
                    'ptc_fed_amount': [self.RevenueModel.FederalPTC],
                    'ptc_sta_amount': [self.RevenueModel.StatePTC],
                    'ptc_fed_escal': self.FinanceModel.InflationRate,
                    'ptc_sta_escal': self.FinanceModel.InflationRate,
                    'ptc_fed_term': self.RevenueModel.FederalPTCTerm,
                    'ptc_sta_term': self.RevenueModel.StatePTCTerm,
                }
            }
        )

        MERC_MODEL.execute()
        out_merc = MERC_MODEL.Outputs.export()

        mws = numpy.array(out_merc['mp_total_cleared_capacity'])
        gen = gen.assign(MW=mws)
        total_revenue = numpy.array(out_merc['mp_energy_market_generated_revenue'])
        energy_revenue = total_revenue * energy_prices/final_prices
        cannib_revenue = total_revenue * cannib_prices/final_prices
        rec_revenue = total_revenue * rec_prices/final_prices

        self.PPAPrice = sum(total_revenue) / (sum(out_merc['cf_energy_net']) / 1000)
        self.LCOEReal = out_merc['lcoe_real'] * 10
        self.LCOENom = out_merc['lcoe_nom'] * 10
        self.NPV = out_merc["project_return_aftertax_npv"]
        self.IRR = out_merc["analysis_period_irr"]
        self.CapacityFactor = sum(out_solar['gen']) / (8760 * self.TechnologyModel.Capacity * 1000) * 100

        self.RevenueTable = pandas.DataFrame({
            'Energy Revenue ($)': energy_revenue,
            'REC Revenue ($)': rec_revenue,
            'Cannibalization Adjustment ($)': cannib_revenue,
        }, index=gen.index).shift(freq='-1min').resample('YS').sum().rename(lambda x: x.year).reindex(self.PeriodModel.CashFlowYears)

        self.CashFlowTable = pandas.Series({
            'Energy': [],
            'Net Energy Production (MWh)': numpy.array(out_merc['cf_energy_net']) / 1000,
            'Curtailed Energy (MWh)': numpy.array(out_merc['cf_energy_net']) / 1000,
            'Energy Revenue ($)': self.RevenueTable['Energy Revenue ($)'].tolist(),
            'REC Revenue ($)': self.RevenueTable['REC Revenue ($)'].tolist(),
            'Cannibalization Adjustment ($)': self.RevenueTable['Cannibalization Adjustment ($)'].tolist(),
            'Total Revenue ($)': self.RevenueTable.sum(axis=1).tolist(),
            'O&M Cost': [],
            'Operating Cost Capacity ($)': numpy.array(out_merc['cf_om_capacity_expense']) * -1,
            'Operating Cost Fixed ($)': numpy.array(out_merc['cf_om_fixed_expense']) * -1,
            'Operating Cost Production ($)': numpy.array(out_merc['cf_om_production_expense']) * -1,
            'Property Tax Cost ($)': numpy.array(out_merc['cf_property_tax_expense']) * -1,
            'Total Operating Cost ($)': numpy.array(out_merc['cf_property_tax_expense']) * -1,
            'Operating Revenues': [],
            'EBITA ($)': out_merc['cf_ebitda'],
            'Interest on Reserves ($)': out_merc['cf_reserve_interest'],
            'Interest on Debt Payment ($)': out_merc['cf_debt_payment_interest'],
            'Operating Cash Flow ($)': out_merc['cf_project_operating_activities'],
            'Debt Terms': [],
            'Total Installed Cost ($)': [self.total_installed_cost()],
            'Debt Fee ($)': [MERC_MODEL.Outputs.cost_debt_upfront],
            'Construction Financing Cost ($)': [self.total_construction_financing_cost()],
            'Total Debt Cost ($)': self.total_installed_cost() + MERC_MODEL.Outputs.cost_debt_upfront + self.total_construction_financing_cost(),
            'Debt Service Decrease/Reserve Increase ($)': out_merc['cf_project_dsra'],
            'Working Capital Decrease/Reserve Increase ($)': out_merc['cf_project_wcra'],
            'Total Debt Fraction ($)': out_merc['size_of_debt'],
            'Total Equity Fraction ($)': out_merc['size_of_equity'],
            'Returns': [],
            'Total Pretax Returns ($)': out_merc['cf_project_return_pretax'],
            'Federal ITC Total Returns ($)': out_merc['cf_itc_fed'],
            'Federal PTC Total Returns ($)': out_merc['cf_ptc_fed'],
            'State ITC Total Returns ($)': out_merc['cf_itc_sta'],
            'State PTC Total Returns ($)': out_merc['cf_ptc_sta'],
            'Total After-tax Returns ($)': out_merc['cf_project_return_aftertax'],
            'Total After-tax NPV ($)': out_merc['cf_project_return_aftertax_npv'],
            'Depreciation': [],
            'Total Tax Deprecation - MARC 5 year': out_merc['cf_feddepr_macrs_5']
        }).apply(pandas.Series).round(2).set_axis(self.PeriodModel.CashFlowYears, axis=1).T

        self.SummaryTable = pandas.Series({
            'Year 1 Energy Production (MWh)': sum(out_solar['gen']) / 1000,
            'Year 1 Energy Curtailed (MWh)': (sum(out_solar['gen']) - curtailed_gen.sum())/1000,
            'Capacity Factor (%)': self.CapacityFactor,
            'Energy Revenue ($)': sum(energy_revenue.fillna(0)),
            'REC Revenue ($)': sum(rec_revenue.fillna(0)),
            'Cannibalization Adjustment ($)': sum(cannib_revenue.fillna(0)),
            'Total Revenue': sum(total_revenue),
            'Solar Capture Rate (%)': ((gen['MW'] * energy_prices).sum() / gen['MW'].sum()) / energy_prices.mean() * 100,
            'PPA Price ($/MWh)': sum(total_revenue) / (sum(out_merc['cf_energy_net']) / 1000),
            'LCOE Real ($/MWh)': out_merc['lcoe_real'] * 10,
            'LCOE Nominal ($/MWh)': out_merc['lcoe_nom'] * 10,
            'NPV ($)': out_merc["project_return_aftertax_npv"],
            'IRR(%)': out_merc["analysis_period_irr"],
            'IRR Invert Year': self.FinanceModel.StartYear + out_merc['flip_actual_year'],
            'Total Capital Cost($)': out_merc["adjusted_installed_cost"],
            'Size of Equity ($)': out_merc['size_of_equity'],
            'Size of Debt ($)': out_merc['size_of_debt'],
            'Land Area (Acres)': self.TechnologyModel.LandArea,
            'Direct Capital Cost ($)': self.DirectCost,
            'Transmission Cost ($)': self.TransmissionCost,
            'Land Purchase Cost ($)': self.LandCost,
            'Indirect Capital Cost ($)': self.IndirectCosts,
        }).round(2).reset_index().set_axis(['Metric', 'Value'], axis=1)

        self.SolarWeightedAverage = gen.assign(Px=energy_prices).groupby('Year').apply(lambda x: numpy.average(x['Px'], weights=x['MW']), include_groups=False)
        self.BaseLoadAverage = gen.assign(Px=energy_prices).groupby('Year')['Px'].apply(numpy.mean, include_groups=False)
        self.CaptureTable = pandas.concat([
            self.SolarWeightedAverage.to_frame('Solar Weighted Price'),
            self.BaseLoadAverage.to_frame('Average Price'),
            (self.SolarWeightedAverage/self.BaseLoadAverage).to_frame('Capture Rate (%)'),
        ], axis=1)
        return gen

    def projected(self):
        grp = self.Fixed.groupby(['Month', 'HourEnding'])['MW'].mean().reset_index()
        fwd = self.PeriodModel.ForwardMerge.merge(grp)
        fwd = fwd.merge(grp).set_index('index')['MW']
        deg_rate = self.TechnologyModel.DegradationRate / 100
        min_year = min(fwd.index.year)
        deg = fwd.shift(freq='-1h').index.map(lambda x: (1 - deg_rate) ** (x.year - min_year))
        return fwd * deg


class CannibalizationModel(BaseModel):
    def __init__(self, zone, models):
        super().__init__(models)
        self.Zone = zone
        self.Unfixed = self.fwd_vgr()
        self.S2 = datetime.datetime.now()

    def data(self):
        df = pandas.read_csv('Assets/renewable_data.csv')
        df.index = pandas.to_datetime(df["index"], utc='True').dt.tz_convert('US/Central')
        df = df.drop(['index'], axis=1)
        df = self.PeriodModel.attributes(df)
        df = df.join(self.EnergyModel.Fixed.set_axis(['Price'], axis=1)).sort_index().dropna()
        df = df.assign(
            FixedSolar=df.groupby(['Month', 'HourEnding'])['Solar'].transform('mean'),
            FixedWind=df.groupby(['Month', 'HourEnding'])['Wind'].transform('mean'),
        )
        temp_data = pandas.read_csv('Assets/temperature_data.csv')
        temp_data.index = pandas.to_datetime(temp_data["index"], utc='True').dt.tz_convert('US/Central')
        temp_data = temp_data.drop(['index'], axis=1)
        df = df.join(temp_data)
        return df

    def capacity_data(self):
        return pandas.read_csv('Assets/forward_capacity.csv', index_col=0).rename(
            lambda x: pandas.to_datetime(x).date())

    def hist_vgr(self):
        df = self.Data
        fixed = df.groupby(['DateMonth', 'TimeShape']).apply(lambda x: numpy.average(x['Price'], weights=x['FixedSolar'].clip(0.001), axis=0))
        volum = df.groupby(['DateMonth', 'TimeShape']).apply(lambda x: numpy.average(x['Price'], weights=x['Solar'].clip(0.001), axis=0))
        vgr = (volum / fixed).unstack()
        return vgr

    def fwd_vgr(self):
        df = pandas.read_csv(f'data/Prices/Cannib/{self.Zone}_vgr.csv', index_col=0).rename(pandas.to_datetime).rename(lambda x: x.date())
        vgr = df.reset_index().melt(id_vars='index').set_axis(['DateMonth', 'TimeShape', 'Scalar'], axis=1)
        out = self.PeriodModel.ForwardMerge.merge(vgr).set_index('index')['Scalar'] - 1
        return out * self.EnergyModel.Unfixed.loc[out.index]
        # df = self.Data
        # hist_vgr = self.Fixed
        # forw_cap = self.CapacityData['Forecast'].dropna()
        # hist_cap = df['Solar Cap'].shift(freq='-1min').groupby(lambda x: x.date().replace(day=1)).max()
        # mont_vgr = hist_vgr.groupby(lambda x: x.month).mean()
        # out = []
        # for month in range(1,13):
        #     temp_month_cap = hist_cap[hist_cap.index.map(lambda x: x.month==month)]
        #     temp_month_vgr = hist_vgr[hist_vgr.index.map(lambda x: x.month==month)]
        #     temp_month_fwd = forw_cap[forw_cap.index.map(lambda x: x.month==month)]
        #     temp = []
        #     for timeshape in ['5x16', '2x16', '7x8']:
        #         model = LinearRegression()
        #         model.fit(temp_month_cap.to_frame(), temp_month_vgr[timeshape])
        #         predi = model.predict(temp_month_fwd.to_frame('Solar Cap'))
        #         temp.append(pandas.Series(predi, index=temp_month_fwd.index).to_frame(timeshape))
        #     out.append(pandas.concat(temp, axis=1))
        # out = pandas.concat(out).sort_index()
        # out = out.reset_index().melt(id_vars='index').set_axis(['DateMonth', 'TimeShape', 'Scalar'], axis=1)
        # out = self.PeriodModel.ForwardMerge.merge(out).set_index('index')['Scalar'] - 1
        # return out * self.EnergyModel.Unfixed.loc[out.index]


class CurtailmentModel(BaseModel):
    def __init__(self, zone, models):
        super().__init__(models)
        self.Zone = zone
        self.Fixed = self.hist_curtailment()
        self.Unfixed = self.forward_curtailment()
        self.S2 = datetime.datetime.now()

    def hist_curtailment(self):
        return pandas.read_csv('data/Curtailment/Curtailment.csv')

    def forward_curtailment(self):
        hist = self.Fixed
        return hist.groupby('Month').mean()['ERCOT'].reset_index()


class RECModel(BaseModel):
    def __init__(self, models):
        super().__init__(models)
        self.RECPrices = self.rec_prices()
        self.Unfixed = self.forward_rec_prices()
        self.S2 = datetime.datetime.now()

    def rec_prices_vintages(self):
        return pandas.read_csv(f'data/Prices/RECS/Forward/RECS_VINTAGE.csv', index_col=[0, 1, 2]).reset_index()

    def rec_prices(self):
        return pandas.read_csv(f'data/Prices/RECS/Forward/RECS.csv')

    def forward_rec_prices(self):
        fwd = self.PeriodModel.ForwardMerge
        prices = self.RECPrices
        prices = pandas.concat([prices] + [prices[prices['Year'] == prices['Year'].max()].assign(Year=year) for year in self.PeriodModel.Years if year not in set(prices['Year'].astype(int))])
        return fwd.merge(prices).ffill().set_index('index')['Price']


class EnergyModel(BaseModel):
    def __init__(self, zone, models):
        super().__init__(models)
        self.Zone = zone
        self.Fixed = self.settles()
        self.Forwards = self.forwards()
        self.Scalars = self.scalars()
        self.Unfixed = self.datacurves()
        self.S2 = datetime.datetime.now()

    def __repr__(self):
        return self.__class__.__name__ + ' ' + self.Zone

    def settles(self):
        df = pandas.read_csv(f'data/Prices/Energy/Settled/{self.Zone}_settles.csv', index_col=0)
        return df.set_axis(pandas.to_datetime(df.index, utc=True)).tz_convert('US/Central')

    def forwards(self):
        df = pandas.read_csv(f'data/Prices/Energy/Forward/{self.Zone}_forwards.csv', index_col=0)
        df = df.set_axis(pandas.to_datetime(df.index))
        fill = pandas.concat(
            [df] + [df[df.index.year == max(df.index.year)].rename(lambda x: x.replace(year=i)) for i in
                    range(max(df.index.year) + 1, self.PeriodModel.EndDate.year + 1)])
        return fill.rename(lambda x: x.date()).loc[self.PeriodModel.StartDate: self.PeriodModel.EndDate]

    def scalars(self):
        return pandas.read_csv(f'data/Prices/Energy/Scalars/{self.Zone}_scalars.csv', index_col=0)

    def datacurves(self):
        fwds = self.Forwards
        fwds_merge = fwds.reset_index().melt(id_vars='index').set_axis(['DateMonth', 'TimeShape', 'Price'],
                                                                       axis=1).dropna()
        mrg = self.PeriodModel.ForwardMerge
        mrg = mrg.merge(fwds_merge, how='left')
        scalars = self.Scalars
        mrg = mrg.merge(scalars).set_index('index')
        return (mrg['Price'] * mrg['Scalar']).sort_index()


class Model(BaseModel):
    def __init__(self, models):
        super().__init__(models)
        self.Optimal = False
        self.WeatherModel = WeatherModel([self.LocationModel, self.PeriodModel])
        self.SolarModel = SolarModel([
            self.LocationModel,
            self.PeriodModel,
            self.RevenueModel,
            self.EnergyModel,
            self.WeatherModel,
            self.TechnologyModel,
            self.FinanceModel,
            self.OperatingCostModel,
            self.CapitalCostModel,
            self.CannibalizationModel,
            self.CurtailmentModel,
            self.RECModel
        ])
        self.update_model([self.WeatherModel, self.SolarModel, self.RevenueModel])
        self.S2 = datetime.datetime.now()

    def __repr__(self):
        return self.__class__.__name__ + ' ' + self.LocationModel.Name

class Models(BaseModel):
    def __init__(self):
        super().__init__()
        self.Lats = COUNTIES['X (Lat)'].tolist()
        self.Longs = COUNTIES['Y (Long)'].tolist()
        self.Locations = [LocationModel(lat, long) for lat, long in zip(self.Lats, self.Longs)]
        self.Zones = set([i.Zone for i in self.Locations])
        self.TimeZones = set([i.TimeZone for i in self.Locations])
        self.TechnologyModel = TechnologyModel()
        self.DiscountModel = DiscountModel()
        self.RevenueModel = RevenueModel()
        self.OperatingCostModel = OperatingCostModel()
        self.CapitalCostModel = CapitalCostModel()
        self.FinanceModel = FinanceModel()
        self.RunModel = RunModel([self])
        self.S2 = datetime.datetime.now()


class RunModel(BaseModel):
    def __init__(self, models):
        super().__init__(models)
        self.RunTacker = 0
        self.S2 = datetime.datetime.now()

    def build_mode(self, location):
        return Model(
            [
                location,
                self.PeriodModel,
                self.RevenueModel,
                self.EnergyModels[location.Zone],
                self.TechnologyModel,
                self.OperatingCostModel,
                self.CapitalCostModel,
                self.FinanceModel,
                self.CannibModels[location.Zone],
                self.CurtailModels[location.Zone],
                self.RECModel
            ]
        )

    def reset(self):
        self.__init__(self.ModelsIn)

    def rebuild(self):
        self.S1 = datetime.datetime.now()
        self.PeriodModel = PeriodModel(self.FinanceModel.StartDate, self.FinanceModel.EndDate, 'US/Central')
        self.EnergyModels = {i: EnergyModel(i, [self.PeriodModel]) for i in self.Zones}
        self.CannibModels = {i: CannibalizationModel(i, [self.EnergyModels[i]]) for i in self.Zones}
        self.CurtailModels = {i: CurtailmentModel(i, [self.TechnologyModel, self.EnergyModels[i]]) for i in self.Zones}
        self.RECModel = RECModel([self.PeriodModel])
        with Pool() as pool:
            self.Models = pool.map(self.build_mode, self.Locations)
        # self.Models = [self.build_mode(i) for i in self.Locations]
        self.optimal_county()
        self.DF = self.build_dataframe()
        self.execution_time_profile()
        self.S2 = datetime.datetime.now()

    def build_dataframe(self):
        cols = [
            'Zone', 'Optimal', 'NPV', 'IRR', 'LCOEReal', 'LCOENom', 'PPAPrice', 'AverageDHI', 'AverageDNI', 'AverageMW',
            'LandPrice', 'PropertyTax', '100_161_kV', '220_287_kV', '345_kV', '500_kV', 'Name', 'FIPS', 'Lat', 'Long'
        ]
        return pandas.DataFrame({col: [getattr(i, col) for i in self.Models] for col in cols}).round(2)

    def optimal_county(self, metric='NPV'):
        self.OptimalCountyModel = self.Models[pandas.Series([getattr(i, metric) for i in self.Models]).idxmax()]
        self.OptimalCountyModel.Optimal = True
        self.OptimalCountyName = self.OptimalCountyModel.Name
        self.OptimalCountySummary = self.OptimalCountyModel.SummaryTable.loc[[0, 2, 7, 8, 9, 10, 11, 12]]
        tab_data = self.OptimalCountySummary.round(2).to_dict('records')
        tab_cols = [{"name": str(i), "id": str(i), 'type': 'numeric', 'format': Format(group=',')} for i in self.OptimalCountySummary.columns]
        return tab_data, tab_cols

    def execution_time_profile(self):
        models = {}
        def add_time(val):
            if hasattr(val, 'ExecutionTime'):
                models[(val.__class__.__name__, '')] = val.ExecutionTime
            if isinstance(val, list):
                for model in val:
                    if hasattr(model, 'ExecutionTime'):
                        models[(model.__class__.__name__, str(model))] = model.ExecutionTime
            if isinstance(val, dict):
                for key, model in val.items():
                    if hasattr(model, 'ExecutionTime'):
                        models[(model.__class__.__name__, str(model))] = model.ExecutionTime

        for key, val in vars(self).items():
            add_time(val)
            if isinstance(val, type):
                for key, val_sub in vars(val).items():
                    if isinstance(val, type):
                        add_time(val_sub)

        self.ExecutionTimeProfile = pandas.Series(models)
        return pandas.Series(models)

    def render(self):
        return html.Div(
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

def setup_app(model):
    app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
    app.config.suppress_callback_exceptions = True
    tables = ['Technology', 'CapitalCost', 'OperatingCost', 'Revenue', 'Finance', 'Run']
    model_cache = {}
    app.layout = html.Div([
        html.P(id='banner_text', children='Configure Model Parameters', style={'background-color': 'whitesmoke'}),
        dcc.Tabs(id='tabs', value='Technology', children=[
            dcc.Tab(label=i, value=i, children=getattr(model, f'{i}Model').render(),
                    style={'background-color': 'whitesmoke'}) for i in tables]),
        html.Div(id='tabs-content', style={"height": "100vh", 'background-color': 'whitesmoke'}),
    ], style={'height': '100vh', 'padding': 10, 'background-color': 'whitesmoke'})

    @app.callback(
        dash.dependencies.Output('banner_text', 'children'),
        dash.dependencies.Input({"type": "param", "name": ALL}, 'value'),
        dash.dependencies.Input({"type": "dates", "name": ALL}, 'date'),
        dash.dependencies.State('tabs', 'value'),
        prevent_initial_call=True
    )
    def update_model(*values):
        tab = values[-1]
        triggered = dash.callback_context.triggered
        if triggered:
            name = json.loads(triggered[0]['prop_id'].split('.')[0])['name']
            value = triggered[0]['value']
            if value is not None and tab in tables:
                active_model = getattr(model, f'{tab}Model')
                setattr(active_model, name, value)
                print(active_model.get_dict)
        return tab

    @app.callback(
        dash.dependencies.Output('tabs', 'children'),
        dash.dependencies.Input('run', 'n_clicks'),
        dash.dependencies.State('model_name', 'value'),
        dash.dependencies.State('tabs', 'children'),
        prevent_initial_call=True
    )
    def run(value, model_name, children):
        triggered = dash.callback_context.triggered
        if 'run' in triggered[0]['prop_id'] and value > model.RunModel.RunTacker:
            assert model_name not in model_cache
            model.RunModel.RunTacker = value
            model.RunModel.rebuild()
            model_cache[model_name] = deepcopy(model)
            opt_tab, opt_col = model.RunModel.optimal_county()
            new_tab = dcc.Tab(
                label=model_name,
                value=model_name,
                children=html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            dcc.Dropdown(model.RunModel.DF.columns, 'Zone', id={'type': 'map-select', 'name': model_name}),
                                            style={'width': '50%', 'display': 'inline-block', 'height': '4vh'}
                                        ),
                                        html.Div(
                                            dcc.Dropdown(['None', 'All', '100-161 kV', '220-287 kV', '345 kV', '500 kV'], 'None', id={'type': 'trans-select', 'name': model_name}),
                                            style={'width': '20%', 'display': 'inline-block', 'height': '4vh'}
                                        ),
                                        html.Div(
                                            dcc.Checklist(
                                                [
                                                    {
                                                        "label": html.Div(['Show Load Zone'], style={'color': 'Black', 'font-size': 20, "font-weight": "bold"}),
                                                        "value": "Show Load Zone",
                                                    },
                                                ],
                                                value=[],
                                                id={'type': 'load-zone-select', 'name': model_name},
                                                labelStyle={"display": "flex", "align-items": "center"}
                                            ),
                                            style={'width': '20%', 'display': 'inline-block', 'height': '4vh'}
                                        ),
                                    ],
                                    style={'display': 'flex', 'justify-content': 'space-between', 'height': '4vh'}

                                ),
                                dcc.Graph(
                                    id={'type': 'map', 'name': model_name},
                                    figure=px.choropleth(
                                        model.RunModel.DF,
                                        geojson=GEO_JSON,
                                        locations='FIPS',
                                        color='Zone',
                                        hover_data=model.RunModel.DF.columns,
                                        custom_data=['Name', 'Lat', 'Long'],
                                        scope='usa',
                                        fitbounds="locations",
                                        color_continuous_scale='Viridis',
                                        basemap_visible=False,
                                        center={"lat": 31.391489, "lon": -99.174304}
                                    ),
                                    style={'height': '65vh', 'vertical-align': 'bottom'}
                                )
                            ], style={'width': '75%', 'display': 'inline-block', 'vertical-align': 'top', 'height': '70vh'}),
                            html.Div(
                                [
                                    html.P(f'Optimal County: {model.RunModel.OptimalCountyName}',
                                           style={'text-align': 'center', 'width': '100%', "font-weight": "bold", 'height': '4vh', 'padding-top': '1.5vh', 'font-size': '20px'}
                                           ),
                                    dash.dash_table.DataTable(
                                        data=opt_tab,
                                        columns=opt_col,
                                        id={'type': 'recommend_data_table', 'name': model_name},
                                        style_cell={'width': '50%'},
                                        style_table={'height': '20vh'},
                                    ),
                                    html.P(f'County: ',
                                           id={'type': 'summary_data_table_label', 'name': model_name},
                                           style={'text-align': 'center', 'width': '100%', "font-weight": "bold", 'height': '4vh', 'padding-top': '1.5vh', 'font-size': '20px'}
                                           ),
                                    dash.dash_table.DataTable(
                                        id={'type': 'summary_data_table', 'name': model_name},
                                        style_cell={'width': '50%'},
                                        style_table={'overflowY': 'scroll', 'height': '38.5vh'},
                                    )
                                ], style={'width': '25%', 'display': 'inline-block', 'height': '70vh'}
                            ),
                        html.Div(
                            [
                                dcc.Tabs(id='sumamry-tabs', value='Summary', children=[
                                    dcc.Tab(label='Summary',
                                            value='Summary',
                                            children=html.Div([
                                                dcc.Graph(
                                                    id={'type': 'hourly-x-graph', 'name': model_name},
                                                    style={'height': '65vh', 'width': '50%', 'display': 'inline-block'}
                                                ),
                                                dcc.Graph(
                                                    id={'type': 'monthly-x-graph', 'name': model_name},
                                                    style={'height': '65vh', 'width': '50%', 'display': 'inline-block'}
                                                ),
                                                dcc.Graph(
                                                    id={'type': 'capture-x-graph', 'name': model_name},
                                                    style={'height': '65vh', 'width': '50%', 'display': 'inline-block'}
                                                ),
                                                dcc.Graph(
                                                    id={'type': 'cashflow-x-graph', 'name': model_name},
                                                    style={'height': '65vh', 'width': '50%', 'display': 'inline-block'}
                                                )
                                            ])
                                            ),
                                    dcc.Tab(label='Cashflows',
                                            value='Cashflows',
                                            children=html.Div(
                                                dash.dash_table.DataTable(id={'type': 'data_table', 'name': model_name},
                                                                          export_format='csv',
                                                                          style_table={'overflowX': 'scroll'}),
                                                style={'background-color': 'whitesmoke', 'height': '65vh'}
                                            ))
                                ]),
                                html.Button('Delete Tab', id={'type': 'delete', 'name': model_name})
                            ]
                        )
                    ]
                )
            )
            model.RunModel.reset()
            return children + [new_tab]
        return no_update

    @app.callback(
        dash.dependencies.Output({"type": "map", "name": MATCH}, 'figure'),
        dash.dependencies.Input({"type": "map-select", "name": MATCH}, "value"),
        dash.dependencies.Input({"type": "trans-select", "name": MATCH}, "value"),
        dash.dependencies.Input({"type": "load-zone-select", "name": MATCH}, "value"),
    )
    def update_map(value, add_trans, add_load_zone):
        name = dash.callback_context.args_grouping[0]['id']['name']
        temp_model = model_cache.get(name, model)
        fig = px.choropleth(
            temp_model.RunModel.DF,
            geojson=GEO_JSON,
            locations='FIPS',
            color=value,
            hover_data=temp_model.RunModel.DF.columns,
            custom_data=['Name', 'Lat', 'Long'],
            scope='usa',
            fitbounds="locations",
            color_continuous_scale='Viridis',
            basemap_visible=False,
            center={"lat": 31.391489, "lon": -99.174304}
        )
        if len(add_load_zone):
            fig.add_traces(LOAD_ZONE_TRACES)
        if add_trans != 'None':
            temp_cords = TRANS_CORDS
            if add_trans != 'All':
                temp_cords = temp_cords[(temp_cords.Voltage == add_trans.replace(' kV', '')) | temp_cords.Voltage.isna()]
            temp_trans_data = px.scatter_geo(temp_cords, 'Lat', 'Lon', text='Voltage').update_traces(mode="lines", line_color="red").data
            fig.add_traces(temp_trans_data)
        return fig

    @app.callback(
        dash.dependencies.Output({"type": 'summary_data_table_label', "name": MATCH}, 'children'),
        dash.dependencies.Output({"type": 'summary_data_table', "name": MATCH}, 'data'),
        dash.dependencies.Output({"type": 'summary_data_table', "name": MATCH}, 'columns'),
        dash.dependencies.Output({"type": 'data_table', "name": MATCH}, 'data'),
        dash.dependencies.Output({"type": 'data_table', "name": MATCH}, 'columns'),
        dash.dependencies.Output({"type": 'hourly-x-graph', "name": MATCH}, 'figure'),
        dash.dependencies.Output({"type": 'monthly-x-graph', "name": MATCH}, 'figure'),
        dash.dependencies.Output({"type": 'capture-x-graph', "name": MATCH}, 'figure'),
        dash.dependencies.Output({"type": 'cashflow-x-graph', "name": MATCH}, 'figure'),
        dash.dependencies.Input({"type": "map", "name": MATCH}, "clickData"),
    )
    def update_county_profile(clickData):
        if clickData is None or clickData['points'][0].get('customdata') is None:
            return no_update

        name = dash.callback_context.args_grouping['id']['name']
        temp_model = model_cache.get(name, model)
        county = clickData['points'][0]['customdata'][0]
        county_model = [i for i in temp_model.RunModel.Models if i.Name == county][0]
        county_label = f'County: {county_model.Name}'
        hourly_graph = county_model.HourlyProfile.T
        monthly_graph = county_model.MonthlyProfile.T
        capture_graph = county_model.CaptureTable['Capture Rate (%)'] * 100
        cashflow_graph = county_model.CashFlowTable['Total After-tax Returns ($)']

        # Update the Graphs
        hourly_plt_dct = {
            'data': [{'x': hourly_graph.index, 'y': hourly_graph[i], 'type': 'line', 'name': i} for i in hourly_graph.columns],
            'layout': {
                'title': {'text': f'{county} Hourly Generation Profile'},
                'xaxis': {'title': 'Hourending', 'dtick': 1},
                'yaxis': {'range': [0, hourly_graph.max(axis=1).max() * 1.1], 'title': 'Hourly Generation (MWh)'}
            }

        }

        monthly_plt_dct = {
            'data': [{'x': monthly_graph.index, 'y': monthly_graph.tolist(), 'type': 'line', 'name': 'MW'}],
            'layout': {
                'title': {'text': f'{county} Monthly Generation Profile'},
                'xaxis': {'title': 'Month', 'range': [1, 12], 'dtick': 1},
                'yaxis': {'range': [monthly_graph.min() * 0.9, monthly_graph.max() * 1.1], 'title': 'Monthly Generation (MWh)'}
            }

        }

        capture_plt_dct = {
            'data': [{'x': capture_graph.index, 'y': capture_graph.tolist(), 'type': 'line', 'name': 'Capture'}],
            'layout': {
                'title': {'text': f'{county} Solar Capture Rate (%)'},
                'xaxis': {'title': 'Year', 'dtick': 1},
                'yaxis': {'range': [capture_graph.min() * 0.9, capture_graph.max() * 1.1], 'title': 'Solar Capture Rate (%)'}
            }

        }

        cashflow_plt_dct = {
            'data': [{'x': cashflow_graph.index, 'y': cashflow_graph.tolist(), 'type': 'bar', 'name': 'Cashflow'}],
            'layout': {
                'title': {'text': f'{county} Yearly Cash Flows ($)'},
                'xaxis': {'title': 'Year', 'dtick': 1},
                'yaxis': {'range': [cashflow_graph.min() * 0.9, cashflow_graph.max() * 1.1], 'title': 'Cash Flow ($)'}
            }

        }

        # Update the DataTable
        data = county_model.CashFlowTable.T.reset_index()
        tab_data = data.round(2).to_dict('records')
        tab_cols = [{"name": str(i), "id": str(i), 'type': 'numeric', 'format': Format(group=',')} for i in data.columns]

        sum_data = county_model.SummaryTable
        sum_tab_data = sum_data.round(2).to_dict('records')
        sum_tab_cols = [{"name": str(i), "id": str(i), 'type': 'numeric', 'format': Format(group=',')} for i in sum_data.columns]

        return county_label, sum_tab_data, sum_tab_cols, tab_data, tab_cols, hourly_plt_dct, monthly_plt_dct, capture_plt_dct, cashflow_plt_dct

    @app.callback(
        dash.dependencies.Output('tabs', 'children', allow_duplicate=True),
        dash.dependencies.Input({'type': 'delete', 'name': ALL}, 'n_clicks'),
        dash.dependencies.State('tabs', 'children'),
        dash.dependencies.State('tabs', 'value'),
        prevent_initial_call=True
    )
    def delete_tab(run, children, name):
        triggered = dash.callback_context.triggered
        if 'delete' in triggered[0]['prop_id'] and triggered[0]['value']:
            del model_cache[name]
            return [i for i in children if i['props']['label'] != name]
        return no_update

    app.run_server(debug=True, use_reloader=False)


if __name__ == "__main__":
    model = Models()
    # model.RunModel.rebuild()
    # self = model.RunModel.Models[0].SolarModel
    setup_app(model)
