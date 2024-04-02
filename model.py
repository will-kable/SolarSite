import datetime
import io
import json
import os

import PySAM.PySSC as pssc
import PySAM.Pvwattsv8 as PVW
import PySAM.Merchantplant as MERC
import PySAM.Grid as GRID
import itertools
import PySAM.ResourceTools as RT
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy
import pandas
import requests
import dash
import plotly.express as px
from dash import Dash, dcc, html, callback, ALL, MATCH, Input, State, Output, Patch, no_update
import dash_bootstrap_components as dbc
from common import closest_key, timeshape
from config import COUNTIES, API_KEY_NREL, FINDER, ZONAL_MAP, GEO_JSON, LAND_PRICES, ARRAY_MAP, MODULE_MAP, MODULE_EFF, PROP_TAXES
import dash_daq as daq
from dateutil import relativedelta
from flask_caching import Cache
from copy import deepcopy
from multiprocessing import Pool


timeout = 60

plt.interactive(True)

class BaseModel:
    def __init__(self):
        self.S1 = datetime.datetime.now()

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
        self.Zone = closest_key(ZONAL_MAP, (lat, long))
        self.LandPrice = LAND_PRICES[self.Name]['Data']['2023']
        self.PropertyTax = PROP_TAXES[self.Name]/100
        self.LandRegion = LAND_PRICES[self.Name]['Region']
        self.S2 = datetime.datetime.now()

    def __repr__(self):
        return self.Name

    def LandCost(self, area):
        return self.LandPrice * area

    def LandTax(self, area):
        return self.LandCost(area) * self.PropertyTax/100


class PeriodModel(BaseModel):
    def __init__(self, sd, ed, timezone):
        super().__init__()
        self.StartDate = sd
        self.EndDate = ed
        self.TimeZone = timezone
        self.Forward = pandas.date_range(sd, ed + datetime.timedelta(days=1), freq='h', tz=self.TimeZone)[1:]
        self.ForwardMerge = self.attributes(self.Forward)
        self.Years = list(range(self.StartDate.year, self.EndDate.year + 1))
        self.CashFlowYears = list(range(self.StartDate.year-1, self.EndDate.year + 1))
        self.S2 = datetime.datetime.now()

    def __repr__(self):
        return f'{self.__class__.__name__} {self.StartDate} {self.EndDate} {self.TimeZone}'

    def attributes(self, df, cols=None):
        if not isinstance(df, pandas.DataFrame):
            df = df.to_frame('index').set_axis(['index'], axis=1)
        hb = df.shift(freq='-1h').index
        funcs = {
            'Year': lambda x: x.year.values,
            'Month': lambda x: x.month.values,
            'Mon': lambda x: x.strftime('%b').values,
            'Day': lambda x: x.day.values,
            'Hour': lambda x: x.hour.values,
            'HourEnding': lambda x: (x.hour + 1).values,
            'Date': lambda x: x.date,
            'DayOfYear': lambda x: x.dayofyear.values,
            'DateMonth': lambda x: x.to_period('M').to_timestamp().date,
            'TimeShape': lambda x: x.map(timeshape).values,
        }
        attr = {key: funcs[key](hb) for key in (cols if cols else funcs)}
        df = df.reset_index().assign(**attr)
        df = df.assign(Idx=df.groupby(['Date', 'TimeShape'])['Year'].cumcount())
        return df.set_index(df.columns[0]).rename_axis('index')


class TechnologyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.Capacity = 100
        self.DCRatio = 1.1
        self.Azimuth = 180
        self.InvEff = 96
        self.Losses = 14.07
        self.ArrayType = 'Fixed'
        self.GCR = 0.4
        self.DegradationRate = 0.5
        self.ModuleType = 'Standard'
        self.S2 = datetime.datetime.now()

    def __repr__(self):
        return self.__class__.__name__

    @property
    def ModuleEfficiency(self):
        return MODULE_EFF[self.ModuleType]

    @property
    def LandArea(self):
        return self.Capacity * 1000 / self.ModuleEfficiency * 0.000247105 / self.GCR

    def render(self):
        return html.Div(
            [
                html.P('Capacity (MW)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'Capacity'}, type='number', placeholder=self.Capacity),
                html.P('DC Ratio'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'DCRatio'}, type='number', placeholder=self.DCRatio),
                html.P('Azimuth'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'Azimuth'}, type='number', placeholder=self.Azimuth),
                html.P('Inverter Efficiency'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'InvEff'}, type='number', placeholder=self.InvEff),
                html.P('Losses (%)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'Losses'}, type='number', placeholder=self.Losses),
                html.P('Array Type'),
                html.Br(),
                dcc.Dropdown(id={'type': 'param', 'name': 'ArrayType'},
                             options=['Fixed', 'Fixed Roof', '1 Axis Tracker', '1 Axis Backtracked', '2 Axis Tracker'],
                             placeholder=self.ArrayType),
                html.P('Module Type'),
                html.Br(),
                dcc.Dropdown(id={'type': 'param', 'name': 'ModuleType'},
                             options=['Standard', 'Premium', 'Thin Film'],
                             placeholder=self.ModuleType),
                html.P('GCR'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'GCR'}, type='number', placeholder=self.GCR),
                html.P('Annual Degradation (%/year)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'DegredationRate'}, type='number', placeholder=self.DegradationRate),
            ]
        )


class MarketModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.IncludeCannib = False
        self.IncludeCurtailment = False
        self.IncludeRECs = False
        self.FederalPTC = 0.275
        self.StatePTC = 0
        self.S2 = datetime.datetime.now()

    def render(self):
        return html.Div(
            [
                html.Br(),
                daq.ToggleSwitch(label='Include Cannibilization Adjustment', id={'type': 'param', 'name': 'IncludeCannib'}),
                html.Br(),
                daq.ToggleSwitch(label='Include Curtailment Adjustment', id={'type': 'param', 'name': 'IncludeCurtailment'}),
                html.Br(),
                daq.ToggleSwitch(label='Include REC Value', id={'type': 'param', 'name': 'IncludeRECs'}),
                html.P('Federal PTC ($/MWh)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'FederalPTC'}, type='number', placeholder=self.FederalPTC),
                html.P('State PTC ($/MWh)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'StatePTC'}, type='number', placeholder=self.StatePTC),
            ]
        )


class CapitalCostModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.ModuleCost = 0.39
        self.InverterCost = 0.05
        self.EquipmentCost = 0.29
        self.LaborCost = 0.18
        self.OverheadCost = 0.12
        self.ContingencyCost = 0.03
        self.PermittingCost = 0
        self.GridConnectionCost = 0.02
        self.EngineeringCost = 0.02
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
                html.P('ModuleCost ($/Wdc)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'ModuleCost'}, type='number', placeholder=self.ModuleCost),
                html.P('Inverter Cost ($/Wdc)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'InverterCost'}, type='number', placeholder=self.InverterCost),
                html.P('Equipment Cost ($/Wdc)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'EquipmentCost'}, type='number', placeholder=self.EquipmentCost),
                html.P('Labor Cost ($/Wdc)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'LaborCost'}, type='number', placeholder=self.LaborCost),
                html.P('Overhead Cost ($/Wdc)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'OverheadCost'}, type='number', placeholder=self.OverheadCost),
                html.P('Contingency Cost ($/Wdc)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'ContingencyCost'}, type='number', placeholder=self.ContingencyCost),
                html.P('Permitting Cost ($/Wdc)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'PermittingCost'}, type='number', placeholder=self.PermittingCost),
                html.P('Grid Connection Cost ($/Wdc)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'GridConnectionCost'}, type='number', placeholder=self.GridConnectionCost),
                html.P('Engineering Cost ($/Wdc)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'EngineeringCost'}, type='number', placeholder=self.EngineeringCost),
            ]
        )


class OperatingCostModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.AnnualCost = 0.01
        self.AnnualCostByCapacity = 15
        self.S2 = datetime.datetime.now()

    def render(self):
        return html.Div(
            [
                html.P('Annual Fixed Cost ($/yr)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'AnnualCost'}, type='number', placeholder=self.AnnualCost),
                html.P('Annual Fixed Cost by Capacity ($/(kw-yr))'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'AnnualCostByCapacity'}, type='number', placeholder=self.AnnualCostByCapacity),
            ]
        )


class FinanceModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.Term = 10
        self.StartYear = datetime.date.today().year + 1
        self.InflationRate = 2.5
        self.FederalTaxRate = 21
        self.StateTaxRate = 7
        self.PropertyTaxRate = 1.5
        self.SalesTaxRate = 6.25
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
                html.P('Start Yera'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'StartYear'}, type='number', placeholder=self.StartYear),
                # dcc.DatePickerSingle(id={'type': 'dates', 'name': 'Date'}, date=self.Date),
                html.P('Analysis Period (years)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'Term'}, type='number', placeholder=self.Term),
                html.P('Inflation Rate (%)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'InflationRate'}, type='number', placeholder=self.Term),
                html.P('Federal Tax Rate (%)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'FederalTaxRate'}, type='number', placeholder=self.FederalTaxRate),
                html.P('State Tax Rate (%)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'StateTaxRate'}, type='number', placeholder=self.StateTaxRate),
                html.P('Property Tax Rate (%)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'PropertyTaxRate'}, type='number', placeholder=self.PropertyTaxRate),
                html.P('Sales Tax Rate (%)'),
                html.Br(),
                dcc.Input(id={'type': 'param', 'name': 'SalesTaxRate'}, type='number',placeholder=self.SalesTaxRate),

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
        super().__init__()
        self.Models = models
        self.update_model(self.Models)
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
        super().__init__()
        self.update_model(models, False)
        self.Tilt = self.optimal_angle()
        self.LCOEReal = 0
        self.LCOENom = 0
        self.Unfixed = self.simulate()
        self.CapacityFactor = self.Unfixed.MW.sum() / self.Unfixed.MW.count() / self.TechnologyModel.Capacity
        self.HourlyProfile = self.Unfixed.groupby(['Month', 'HourEnding'])['MW'].sum().unstack()
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
        direct = self.CapitalCostModel.direct_cost(self.TechnologyModel.Capacity)
        indirect = self.CapitalCostModel.indirect_cost(self.TechnologyModel.Capacity)
        land = self.LocationModel.LandCost(self.TechnologyModel.LandArea)
        sales = self.FinanceModel.SalesTaxRate/100 * (direct)
        return direct + indirect + land + sales

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
                    'system_use_lifetime_output': 1,
                    'dc_degradation': [self.TechnologyModel.DegradationRate],
                },
                'SystemDesign': {
                    'system_capacity': self.TechnologyModel.Capacity*1000,
                    'dc_ac_ratio': self.TechnologyModel.DCRatio,
                    'tilt': self.Tilt,
                    'azimuth': self.TechnologyModel.Azimuth,
                    'inv_eff': self.TechnologyModel.InvEff,
                    'losses': self.TechnologyModel.Losses,
                    'array_type': ARRAY_MAP[self.TechnologyModel.ArrayType],
                    'module_type': MODULE_MAP[self.TechnologyModel.ModuleType],
                    'gcr': self.TechnologyModel.GCR,
                },
                'SolarResource': {
                    'solar_resource_data': {
                        'lat': self.LocationModel.Lat,
                        'lon': self.LocationModel.Long,
                        'elev': self.WeatherModel.Elevation,
                        'tz': -6,
                        'dn': tuple(df['DNI']),
                        'df': tuple(df['DHI']),
                        'gh': tuple(df['GHI']),
                        'tdry': tuple(df['Temperature']),
                        'wspd': tuple(df['Wind Speed']),
                        'year': tuple(df.Year - 20),
                        'month': tuple(df.Month),
                        'day': tuple(df.Day),
                        'hour': tuple(df.Hour),
                        'minute': tuple(df.Hour * 0),
                        'albedo': tuple(df['Surface Albedo']),
                        'use_wf_albedo': 1,
                    }
                },
            }
        )

        PV_MODEL.execute()
        out_solar = PV_MODEL.Outputs.export()

        # Build the Solar Models
        gen = pandas.concat([df.assign(Year=year) for year in self.PeriodModel.Years])
        idx = pandas.to_datetime(gen[['Year', 'Month', 'Day', 'Hour']]).dt.tz_localize(f'Etc/GMT+{-self.WeatherModel.TimeZone}').dt.tz_convert('US/Central')
        gen = gen.assign(HourEnding=idx.dt.hour+1).set_index(idx).sort_index().shift(freq='1h').assign(Gen=out_solar['gen'])

        # Build the price models
        energy_prices = self.EnergyModel.Unfixed.reindex(gen.index).fillna(0)
        cannib_prices = self.CannibalizationModel.Unfixed.reindex(gen.index).fillna(0) * self.MarketModel.IncludeCannib
        rec_prices = self.RECModel.Unfixed.reindex(gen.index).fillna(0) * self.MarketModel.IncludeRECs
        final_prices = energy_prices + cannib_prices + rec_prices

        MERC_MODEL.assign(
            {
                'SystemOutput': {
                    'gen': tuple(gen['Gen']),
                    'system_pre_curtailment_kwac': tuple(gen['Gen']),
                    'degradation': [self.TechnologyModel.DegradationRate],
                    'system_capacity': self.TechnologyModel.Capacity*1000,
                },

                'FinancialParameters': {
                    'analysis_period': len(self.PeriodModel.Years),
                    'property_tax_rate': self.LocationModel.PropertyTax * 100,
                    'inflation_rate': self.FinanceModel.InflationRate,
                    'state_tax_rate': [self.FinanceModel.StateTaxRate],
                    'federal_tax_rate': [self.FinanceModel.FederalTaxRate],
                    'system_capacity': self.TechnologyModel.Capacity * 1000,
                    'debt_option': 0,
                    'term_tenor': 5,
                },

                'Revenue': {
                    'mp_energy_market_revenue_single': final_prices.to_frame().to_records(index=False).tolist(),
                    'mp_market_percent_gen': 100,
                },

                'Lifetime': {
                    'system_use_lifetime_output': 1
                },

                'SystemCosts': {
                    'om_capacity': [self.OperatingCostModel.AnnualCost * 1000],
                    'total_installed_cost': self.total_installed_cost(),
                }
            }
        )

        MERC_MODEL.execute()
        out_merc = MERC_MODEL.Outputs.export()

        self.FairValue = sum(out_merc['cf_energy_market_revenue']) / (sum(out_merc['cf_energy_net']) / 1000)
        self.LCOEReal = out_merc['lcoe_real']
        self.LCOENom = out_merc['lcoe_nom']
        self.CashFlowTable = pandas.concat([
            pandas.Series(out_merc['cf_energy_net']).to_frame('Net Energy (MWh)')/1000,
            pandas.Series(out_merc['cf_energy_market_revenue']).to_frame('Net Energy Revenue ($)'),
            pandas.Series(out_merc['cf_om_capacity_expense']).to_frame('O/M Expense ($)') * -1,
            pandas.Series(out_merc['cf_property_tax_expense']).to_frame('Property Tax Expense ($)') * -1,
            pandas.Series(out_merc['cf_operating_expenses']).to_frame('Total Operating Expense ($)') * -1,
            pandas.Series(out_merc['cf_ebitda']).to_frame('EBITDA ($)'),
            pandas.Series(out_merc['cf_reserve_interest']).to_frame('Interest on Reserves ($)'),
            pandas.Series(out_merc['cf_debt_payment_interest']).to_frame('Debt Interest Payment ($)') * -1,
            pandas.Series(out_merc['cf_project_operating_activities']).to_frame('Cash Flow Operating Activities ($)'),
            pandas.Series([self.total_installed_cost()]).to_frame('Total Installed Cost ($)'),
            pandas.Series([MERC_MODEL.Outputs.cost_debt_upfront]).to_frame('Debt up-front Cost ($)'),
            pandas.Series([self.total_construction_financing_cost()]).to_frame('Construction Financing Cost ($)'),
            pandas.Series(out_merc['cf_project_dsra']).to_frame('Debt Service Decrease/Reserve Increase ($)'),
            pandas.Series(out_merc['cf_project_wcra']).to_frame('Working Capital Decrease/Reserve Increase ($)'),
            pandas.Series(out_merc['cf_project_investing_activities']).to_frame('Cash Flow Investing ($)'),
            pandas.Series(out_merc['cf_project_financing_activities']).to_frame('Cash Flow Financing ($)'),
            pandas.Series(out_merc['cf_pretax_cashflow']).to_frame('Total Pre-tax Cash Flow ($)'),
        ], axis=1).set_axis(self.PeriodModel.CashFlowYears)

        mws = numpy.array(out_solar['gen'])/1000
        gen = gen.assign(MW=mws)
        return gen

    def projected(self):
        grp = self.Fixed.groupby(['Month', 'HourEnding'])['MW'].mean().reset_index()
        fwd = self.PeriodModel.ForwardMerge.merge(grp)
        fwd = fwd.merge(grp).set_index('index')['MW']
        deg_rate = self.TechnologyModel.DegradationRate/100
        min_year = min(fwd.index.year)
        deg = fwd.shift(freq='-1h').index.map(lambda x: (1-deg_rate)**(x.year - min_year))
        return fwd * deg

class CannibalizationModel(BaseModel):
    def __init__(self, zone, models):
        super().__init__()
        self.Models = models
        self.Zone = zone
        self.update_model(models)
        self.Data = self.data()
        self.CapacityData = self.capacity_data()
        self.Fixed = self.hist_vgr()
        self.Unfixed = self.fwd_vgr()
        self.S2 = datetime.datetime.now()

    def data(self):
        df = pandas.read_csv('Assets/renewable_data.csv')
        df.index = pandas.to_datetime(df['index'], utc='True').dt.tz_convert('US/Central')
        df = df.drop(['index'], axis=1)
        df = self.PeriodModel.attributes(df)
        df = df.join(self.EnergyModel.Fixed.set_axis(['Price'], axis=1)).sort_index().dropna()
        df = df.assign(
            FixedSolar=df.groupby(['Month', 'HourEnding'])['Solar'].transform('mean'),
            FixedWind=df.groupby(['Month', 'HourEnding'])['Wind'].transform('mean'),
        )

        temp_data = pandas.read_csv('Assets/temperature_data.csv')
        temp_data.index = pandas.to_datetime(temp_data['index'], utc='True').dt.tz_convert('US/Central')
        temp_data = temp_data.drop(['index'], axis=1)

        df = df.join(temp_data)
        return df

    def capacity_data(self):
        return pandas.read_csv('Assets/forward_capacity.csv', index_col=0).rename(lambda x: pandas.to_datetime(x).date())

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

class RECModel(BaseModel):
    def __init__(self, models):
        super().__init__()
        self.Models = models
        self.update_model(models)
        self.RECMap = self.rec_map()
        self.RECPrices = self.rec_prices()
        self.Unfixed = self.forward_rec_prices()
        self.S2 = datetime.datetime.now()

    def rec_map(self):
        return set(self.PeriodModel.ForwardMerge.DateMonth.apply(lambda x: f'Credit REC TX CRS Solar {"Front" if x.month < 6 else "Back"} Half {x.year}'))

    def rec_prices(self):
        return {i: pandas.read_csv(f'data/Prices/RECS/{i}.csv', index_col=0).rename(lambda x: pandas.to_datetime(x).date()) for i in self.RECMap}

    def forward_rec_prices(self):
        fwd = self.PeriodModel.ForwardMerge
        prices = self.RECPrices
        def get_price(date):
            label = f'Credit REC TX CRS Solar {"Front" if date.month < 6 else "Back"} Half {date.year}'
            return prices[label].loc[date, 'value']
        out = fwd['DateMonth'].apply(get_price).set_axis(fwd.index)
        return out

class EnergyModel(BaseModel):
    def __init__(self, zone, models):
        super().__init__()
        self.Zone = zone
        self.update_model(models)
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
        fill = pandas.concat([df] + [df[df.index.year == max(df.index.year)].rename(lambda x: x.replace(year=i)) for i in range(max(df.index.year) + 1, self.PeriodModel.EndDate.year + 1)])
        return fill.rename(lambda x: x.date()).loc[self.PeriodModel.StartDate.replace(day=1): self.PeriodModel.EndDate.replace(day=1)]

    def scalars(self):
        return json.loads(open(f'data/Prices/Energy/Scalars/{self.Zone}_scalars.json').read())

    def datacurves(self):
        fwds = self.Forwards
        fwds_merge1 = fwds.reset_index().melt(id_vars='index').set_axis(['Date', 'TimeShape', 'Price'], axis=1).dropna()
        fwds_merge2 = fwds.reset_index().melt(id_vars='index').set_axis(['DateMonth', 'TimeShape', 'Price'], axis=1).dropna()
        mrg = self.PeriodModel.ForwardMerge
        mrg = mrg.assign(Price=mrg.merge(fwds_merge1, how='left')['Price'].fillna(mrg.merge(fwds_merge2, how='left')['Price']).values)
        scalars = self.Scalars
        last_year = max([int(i.split('-')[-1]) for i in scalars])

        def map_forward(idx, scalars):
            date, shape = idx.name
            if len(idx) == 7:
                scalars = scalars.get('Spring_DST', scalars.get(f'Spring_DST-{date.year}', scalars.get(f'Spring_DST-{last_year}')))
            elif len(idx) == 9:
                scalars = scalars.get('Fall_DST', scalars.get(f'Fall_DST-{date.year}', scalars.get(f'Spring_DST-{last_year}')))
            else:
                scalars = scalars.get(
                    date.strftime('%b'),
                    scalars.get(
                        date.strftime('%d%b-%Y'),
                        scalars.get(
                            date.strftime('%b-%Y'),
                            scalars.get(date.replace(year=last_year).strftime('%b-%Y')))))
            return idx * scalars[shape]

        fwds_curve = mrg.groupby(['Date', 'TimeShape'])['Price'].transform(map_forward, *[scalars]).set_axis(mrg['index'])
        return fwds_curve


class Model(BaseModel):
    def __init__(self, models):
        super().__init__()
        self.Models = models
        self.update_model(self.Models)
        self.WeatherModel = WeatherModel([self.LocationModel, self.PeriodModel])
        self.SolarModel = SolarModel([
            self.LocationModel,
            self.PeriodModel,
            self.MarketModel,
            self.EnergyModel,
            self.WeatherModel,
            self.TechnologyModel,
            self.FinanceModel,
            self.OperatingCostModel,
            self.CapitalCostModel,
            self.CannibalizationModel,
            self.RECModel
        ])
        self.update_model([self.WeatherModel, self.SolarModel, self.MarketModel])
        self.S2 = datetime.datetime.now()


class Models(BaseModel):
    def __init__(self):
        super().__init__()
        self.Term = 5
        self.StartDate = datetime.datetime.now()
        self.End = datetime.date.today() + datetime.timedelta(days=365)
        self.Lats = COUNTIES['X (Lat)'].tolist()
        self.Longs = COUNTIES['Y (Long)'].tolist()
        self.Locations = [LocationModel(lat, long) for lat, long in zip(self.Lats, self.Longs)][:5]
        self.Zones = set([i.Zone for i in self.Locations])
        self.TimeZones = set([i.TimeZone for i in self.Locations])
        self.TechnologyModel = TechnologyModel()
        self.DiscountModel = DiscountModel()
        self.MarketModel = MarketModel()
        self.OperatingCostModel = OperatingCostModel()
        self.CapitalCostModel = CapitalCostModel()
        self.FinanceModel = FinanceModel()
        self.RunModel = RunModel([self])
        self.S2 = datetime.datetime.now()


class RunModel(BaseModel):
    def __init__(self, models):
        super().__init__()
        self.ModelsIn = models
        self.update_model(models)
        self.RunTacker = 0
        self.S2 = datetime.datetime.now()

    def build_mode(self, location):
        return Model(
            [
                location,
                self.PeriodModel,
                self.MarketModel,
                self.EnergyModels[location.Zone],
                self.TechnologyModel,
                self.OperatingCostModel,
                self.CapitalCostModel,
                self.FinanceModel,
                self.CannibModels[location.Zone],
                self.RECModel
             ]
        )

    def reset(self):
        self.__init__(self.ModelsIn)

    def rebuild(self):
        self.PeriodModel = PeriodModel(self.FinanceModel.StartDate, self.FinanceModel.EndDate, 'US/Central')
        self.EnergyModels = {i: EnergyModel(i, [self.PeriodModel, self.DiscountModel, self.MarketModel]) for i in self.Zones}
        self.CannibModels = {i: CannibalizationModel(i, [self.PeriodModel, self.DiscountModel, self.EnergyModels[i]]) for i in self.Zones}
        self.RECModel = RECModel([self.PeriodModel])
        with Pool() as pool:
            self.Models = pool.map(self.build_mode, self.Locations)
        # self.Models = [self.build_mode(i) for i in self.Locations]
        self.DF = self.build_dataframe()

    def build_dataframe(self):
        cols = ['Name', 'FIPS', 'Lat', 'Long', 'Zone', 'LandPrice', 'PropertyTax', 'FairValue', 'AverageDNI', 'AverageDHI', 'AverageMW', 'LCOEReal', 'LCOENom']
        return pandas.DataFrame({col: [getattr(i, col) for i in self.Models] for col in cols})

    def render(self):
        return html.Div([html.Button('Run', id='run'), dcc.Input(id='model_name', value='Model1', type="text")])


def setup_app(model):
    app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
    app.config.suppress_callback_exceptions = True
    tables = ['Technology', 'CapitalCost', 'OperatingCost', 'Market', 'Finance', 'Run']
    model_cache = {}
    app.layout = html.Div([
        html.P(id='banner_text', children='Configure Model Parameters'),
        dcc.Tabs(id='tabs', value='Technology', children=[dcc.Tab(label=i, value=i, children=getattr(model, f'{i}Model').render()) for i in tables]),
        html.Div(id='tabs-content',  style={"height": "100vh"}),
    ], style={'height': '100vh', 'padding': 10})

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
        print(triggered)
        if 'run' in triggered[0]['prop_id'] and value > model.RunModel.RunTacker:
            assert model_name not in model_cache
            model.RunModel.RunTacker = value
            model.RunModel.rebuild()
            model_cache[model_name] = deepcopy(model)
            new_tab = dcc.Tab(
                label=model_name,
                value=model_name,
                children=html.Div(
                    [
                        html.Div(
                            [
                                dcc.Dropdown(model.RunModel.DF.columns, 'Zone', id={'type': 'map-select', 'name': model_name}),
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
                                    )
                                )
                            ], style={'width': '49%', 'display': 'inline-block'}
                        ),

                        html.Div(
                            [
                                dcc.Graph(id={'type': 'x-graph', 'name': model_name}),
                            ], style={'width': '49%', 'display': 'inline-block'}
                        ),

                        html.Div(
                            [
                                dash.dash_table.DataTable(id={'type': 'data_table', 'name': model_name}, export_format='csv'),
                                html.Button('Delete Tab', id={'type': 'delete', 'name': model_name})
                            ]
                        )
                    ],
                )
            )
            model.RunModel.reset()
            return children + [new_tab]
        return no_update

    @app.callback(
        dash.dependencies.Output({"type": "map", "name": MATCH}, 'figure'),
        dash.dependencies.Input({"type": "map-select", "name": MATCH}, "value"),
    )
    def update_map(value):
        name = dash.callback_context.args_grouping['id']['name']
        temp_model = model_cache.get(name, model)
        return px.choropleth(
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


    @app.callback(
        dash.dependencies.Output({"type": 'data_table', "name": MATCH}, 'data'),
        dash.dependencies.Output({"type": 'data_table', "name": MATCH}, 'columns'),
        dash.dependencies.Output({"type": 'x-graph', "name": MATCH}, 'figure'),
        dash.dependencies.Input({"type": "map", "name": MATCH}, "clickData"),
    )
    def update_county_profile(clickData):
        if clickData is None:
            return no_update
        # patched_table = Patch()
        # patched_figure = Patch()


        name = dash.callback_context.args_grouping['id']['name']
        temp_model = model_cache.get(name, model)
        county = clickData['points'][0]['customdata'][0]
        county_model = [i for i in temp_model.RunModel.Models if i.Name == county][0]
        graph = county_model.HourlyProfile.T

        #Update the Graph
        plt_dct = {
            'data': [{'x': graph.index, 'y': graph[i], 'type': 'line', 'name': i} for i in graph.columns],
            'layout': {
                'title': {'text': f'{county} Hourly Generation Profile'},
                'xaxis': {'title': 'Hourending'},
                'yaxis': {'range': [0, graph.max(axis=1).max() * 1.1], 'title': 'Avg Generation (MWh)'}
            }

        }
        # patched_figure['data'] = [{'x': graph.index, 'y': graph[i], 'type': 'line', 'name': i} for i in graph.columns]
        # patched_figure['layout']['yaxis'].update({'range': [0, graph.max(axis=1).max() * 1.1]})

        # Update the DataTable
        data = county_model.CashFlowTable.T.reset_index()
        tab_data = data.round(2).to_dict('records')
        tab_cols = [{"name": str(i), "id": str(i)} for i in data.columns]
        return tab_data, tab_cols, plt_dct

    @app.callback(
        dash.dependencies.Output('tabs', 'children', allow_duplicate=True),
        dash.dependencies.Input({'type': 'delete', 'name': ALL}, 'n_clicks'),
        dash.dependencies.State('tabs', 'children'),
        dash.dependencies.State('tabs', 'value'),
        prevent_initial_call=True
    )
    def delete_tab(run, children, name):
        triggered = dash.callback_context.triggered
        print(triggered)
        if 'delete' in triggered[0]['prop_id'] and triggered[0]['value']:
            del model_cache[name]
            return [i for i in children if i['props']['label'] != name]
        return no_update

    app.run_server(debug=True, use_reloader=False)



if __name__ == "__main__":
    app = Dash(__name__)
    model = Models()
#     setup_app(model)
    model.RunModel.rebuild()
    self = model.RunModel.Models[0].SolarModel
