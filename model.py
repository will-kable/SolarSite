import datetime
import io
import json
import os

import PySAM.PySSC as pssc
import PySAM.Pvwattsv8 as PVW
import PySAM.Merchantplant as MERC
import PySAM.Utilityrate5 as UTL
import PySAM.Grid as GRID
import PySAM.ResourceTools as RT

import matplotlib.pyplot as plt
import numpy
import pandas
import requests
import dash
import plotly.express as px
from dash import Dash, dcc, html, callback, ALL, MATCH, Input, State, Output, Patch, no_update
import dash_bootstrap_components as dbc
from common import closest_key, timeshape
from config import COUNTIES, API_KEY_NREL, FINDER, ZONAL_MAP, GEO_JSON, LAND_PRICES, ARRAY_MAP
import dash_daq as daq
from dateutil import relativedelta
from flask_caching import Cache
from copy import deepcopy
from multiprocessing import Pool


timeout = 60

plt.interactive(True)

class BaseModel:
    @property
    def get_dict(self):
        return dict(vars(self))

    def update_model(self, models, update_sub_parmas=True):
        for model in models:
            setattr(self, model.__class__.__name__, model)
            if update_sub_parmas:
                for key, value in model.__dict__.items():
                    setattr(self, key, value)


class LocationModel(BaseModel):
    def __init__(self, lat, long):
        self.Lat = lat
        self.Long = long
        self.Index = ((COUNTIES['X (Lat)'] - lat).abs() + (COUNTIES['Y (Long)'] - long).abs()).idxmin()
        self.Info = COUNTIES.loc[self.Index]
        self.Name = self.Info['CNTY_NM']
        self.FIPS = self.Info['FIPS']
        self.TimeZone = FINDER.timezone_at(lng=self.Long, lat=self.Lat)
        self.Zone = closest_key(ZONAL_MAP, (lat, long))
        self.LandPrice = LAND_PRICES[self.Name]['Data']['2023']
        self.LandRegion = LAND_PRICES[self.Name]['Region']

    def __repr__(self):
        return self.Name


class PeriodModel(BaseModel):
    def __init__(self, sd, ed, timezone):
        self.StartDate = sd
        self.EndDate = ed
        self.TimeZone = timezone
        self.Forward = pandas.date_range(sd, ed + datetime.timedelta(days=1), freq='h', tz=self.TimeZone)[1:]
        self.ForwardMerge = self.attributes(self.Forward)
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
            # 'DateMonth': hb.to_period('M').to_timestamp().date,
            # 'Minute': hb.minute.values,
            'TimeShape': lambda x: x.map(timeshape).values,
        }
        attr = {key: funcs[key](hb) for key in (cols if cols else funcs)}
        df = df.reset_index().assign(**attr)
        df = df.assign(Idx=df.groupby(['Date', 'TimeShape'])['Year'].cumcount())
        return df.set_index(df.columns[0]).rename_axis('index')


class TechnologyModel(BaseModel):
    def __init__(self):
        self.Capacity = 100
        self.DCRatio = 1.1
        self.Azimuth = 180
        self.InvEff = 96
        self.Losses = 14.07
        self.ArrayType = 'Fixed'
        self.GCR = 0.4
        self.DegradationRate = 0.5

    def __repr__(self):
        return self.__class__.__name__

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
        self.IncludeCannib = True
        self.IncludeCurtailment = True
        self.IncludeRECs = True
        self.IncludeAncillary = True
        self.FederalPTC = 0.275
        self.StatePTC = 0


    def render(self):
        return html.Div(
            [
                html.Br(),
                daq.ToggleSwitch(label='Include Cannibilization Adjustment', id={'type': 'param', 'name': 'IncludeCannib'}),
                html.Br(),
                daq.ToggleSwitch(label='Include Curtailment Adjustment', id={'type': 'param', 'name': 'IncludeCurtailment'}),
                html.Br(),
                daq.ToggleSwitch(label='Include REC Value', id={'type': 'param', 'name': 'IncludeRECs'}),
                html.Br(),
                daq.ToggleSwitch(label='Include Include Ancillary Value',
                                 id={'type': 'param', 'name': 'IncludeAncillary'}),
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
        self.ModuleCost = 0.39
        self.InverterCost = 0.05
        self.EquipmentCost = 0.29
        self.LaborCost = 0.18
        self.OverheadCost = 0.12
        self.ContingencyCost = 0.03
        self.PermittingCost = 0
        self.GridConnectionCost = 0.02
        self.EngineeringCost = 0.02

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
        self.AnnualCost = 0.01
        self.AnnualCostByCapacity = 15

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
        self.Term = 10
        self.Date = (datetime.date.today() + relativedelta.relativedelta(months=1)).replace(day=1)
        self.InflationRate = 2.5
        self.FederalTaxRate = 21
        self.StateTaxRate = 7
        self.PropertyTaxRate = 0

    @property
    def StartDate(self):
        return pandas.to_datetime(self.Date).date()

    @property
    def EndDate(self) -> datetime:
        return self.StartDate + relativedelta.relativedelta(years=self.Term, days=-1)

    def render(self):
        return html.Div(
            [
                html.P('Start Date'),
                html.Br(),
                dcc.DatePickerSingle(id={'type': 'dates', 'name': 'Date'}, date=self.Date),
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



            ]
        )


class DiscountModel(BaseModel):
    def __init__(self):
        self.DiscountCurve = self.discount_curve()

    def discount_curve(self):
        df = pandas.read_csv('Assets/discount_curve.csv', index_col=0).rename(pandas.to_datetime).resample('d').interpolate()
        return df.set_axis(df.index.date)['Discount']

    def discount(self, date):
        return self.DiscountCurve.loc[date]


class WeatherModel(BaseModel):
    def __init__(self, models):
        self.Models = models
        self.update_model(self.Models)
        self.Fixed = self.pull_historical()
        self.AverageDNI = self.Fixed['DNI'].mean()
        self.AverageDHI = self.Fixed['DHI'].mean()
        self.Unfixed = self.projected()

    def __repr__(self):
        return self.__class__.__name__ + ' ' + self.LocationModel.Name

    def projected(self):
        df = self.Fixed
        cols = ['GHI', 'DHI', 'DNI', 'Wind Speed', 'Temperature', 'Surface Albedo']
        grp = df.groupby(['Month', 'HourEnding'])[cols].mean().reset_index()
        fwd = self.PeriodModel.ForwardMerge.merge(grp)
        return fwd.set_index('index').sort_index()

    def pull_historical(self):
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

        info = pandas.read_csv(fetcher.resource_file_paths[0], nrows=1)
        timezone, elevation = info['Local Time Zone'], info['Elevation']
        self.TimeZone = timezone[0]
        self.Elevation = elevation[0]
        df = pandas.read_csv(
            fetcher.resource_file_paths[0],
            skiprows=2,
            usecols=lambda x: 'Unnamed' not in x
        )

        idx = pandas.to_datetime(df[['Year', 'Month', 'Day', 'Hour']]).dt.tz_localize(f'Etc/GMT+{-self.TimeZone}').dt.tz_convert('US/Central')
        df = df.set_index(idx).sort_index().shift(freq='1h')
        df = self.PeriodModel.attributes(df).sort_index()
        return df


class SolarModel(BaseModel):
    def __init__(self, models):
        self.update_model(models, False)
        self.Tilt = self.optimal_angle()
        self.Unfixed = self.simulate()
        self.CapacityFactor = self.Unfixed.MW.sum() / self.Unfixed.MW.count() / self.TechnologyModel.Capacity
        self.HourlyProfile = self.Unfixed.groupby(['Month', 'HourEnding'])['MW'].sum().unstack()
        self.AverageMW = self.Unfixed.MW.mean()

    def __repr__(self):
        return self.__class__.__name__ + ' ' + self.LocationModel.Name

    def optimal_angle(self):
        lat, lon = self.LocationModel.Lat, self.LocationModel.Long
        if os.path.exists(f'data/Angles/{lon}_{lat}.csv'):
            return pandas.read_csv(f'data/Angles/{lon}_{lat}.csv', names=[0], skiprows=1).iloc[0, 0]
        angle = requests.get(f'https://api.globalsolaratlas.info/data/lta?loc={lat},{lon}').json()['annual']['data']['OPTA']
        pandas.Series([angle]).to_csv(f'data/Angles/{lon}_{lat}.csv')
        return angle

    def simulate(self):
        print('Simulating: ', self.LocationModel.Name, self.LocationModel.Index)
        df = self.WeatherModel.Unfixed.copy()

        ssc = pssc.PySSC()
        wfd = ssc.data_create()
        new = PVW.default('PVWattsMerchantPlant')
        PV_MODEL = PVW.new()

        PV_MODEL.assign(
            {
                'SystemDesign': {
                    'system_capacity': self.TechnologyModel.Capacity,
                    'dc_ac_ratio': self.TechnologyModel.DCRatio,
                    'tilt': self.Tilt,
                    'azimuth': self.TechnologyModel.Azimuth,
                    'inv_eff': self.TechnologyModel.InvEff,
                    'losses': self.TechnologyModel.Losses,
                    'array_type': ARRAY_MAP[self.TechnologyModel.ArrayType],
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
        out = PV_MODEL.Outputs.export()
        mws = numpy.array(out['gen'])
        df = df.assign(MW=mws)
        return

    def projected(self):
        grp = self.Fixed.groupby(['Month', 'HourEnding'])['MW'].mean().reset_index()
        fwd = self.PeriodModel.ForwardMerge.merge(grp)
        fwd = fwd.merge(grp).set_index('index')['MW']
        deg_rate = self.TechnologyModel.DegradationRate/100
        min_year = min(fwd.index.year)
        deg = fwd.shift(freq='-1h').index.map(lambda x: (1-deg_rate)**(x.year - min_year))
        return fwd * deg


class PricingModel(BaseModel):
    def __init__(self, zone, models):
        self.Zone = zone
        self.update_model(models)
        self.Fixed = self.settles()
        self.Forwards = self.forwards()
        self.Scalars = self.scalars()
        self.Unfixed = self.datacurves()

    def __repr__(self):
        return self.__class__.__name__ + ' ' + self.Zone

    def settles(self):
        df = pandas.read_csv(f'data/Prices/Settled/{self.Zone}_settles.csv', index_col=0)
        return df.set_axis(pandas.to_datetime(df.index, utc=True)).tz_convert('US/Central')

    def forwards(self):
        df = pandas.read_csv(f'data/Prices/Forward/{self.Zone}_forwards.csv', index_col=0)
        df = df.set_axis(pandas.to_datetime(df.index))
        fill = pandas.concat([df] + [df[df.index.year == max(df.index.year)].rename(lambda x: x.replace(year=i)) for i in range(max(df.index.year) + 1, self.PeriodModel.EndDate.year + 1)])
        return fill.rename(lambda x: x.date()).loc[self.PeriodModel.StartDate.replace(day=1): self.PeriodModel.EndDate.replace(day=1)]

    def scalars(self):
        return json.loads(open(f'data/Prices/Scalars/{self.Zone}_scalars.json').read())

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


class ValueModel(BaseModel):
    def __init__(self, models):
        self.Models = models
        self.update_model(self.Models, False)
        self.Value = self.PricingModel.Unfixed * self.SolarModel.Unfixed.MW
        self.IncomeDF = self.IncomeDF()
        self.FairValue = self.IncomeDF['Total Revenue ($)'].sum() / self.IncomeDF['MWh'].sum()

    def __repr__(self):
        return self.__class__.__name__ + ' ' + self.LocationModel.Name

    def IncomeDF(self, grouping='YS'):
        mws = self.SolarModel.Unfixed.MW
        pxs = self.PricingModel.Unfixed

        def price_value(px):
            temp = mws * px
            temp = temp.groupby(temp.shift(freq='-1min').index.date).sum()
            return temp * temp.index.map(self.PricingModel.DiscountModel.discount)

        daily_mws = price_value(1)
        ene_value = price_value(pxs)
        rec_value = ene_value * 0
        anc_value = ene_value * 0
        can_value = ene_value * 0
        ptc_fed_value = daily_mws * self.MarketModel.FederalPTC
        ptc_sta_value = daily_mws * self.MarketModel.StatePTC
        total_rev = ene_value + rec_value + anc_value + can_value + ptc_sta_value
        df = pandas.concat([daily_mws, ene_value, rec_value, anc_value, can_value, ptc_fed_value, ptc_sta_value, total_rev], axis=1)
        df = df.rename(pandas.to_datetime).resample(grouping).sum().set_axis([
            'MWh',
            'Energy Value ($)',
            'REC Value ($)',
            'Ancillary Value ($)',
            'Cannibalization Value ($)',
            'Federal PTC Value ($)',
            'State PTC Value ($)',
            'Total Revenue ($)'
        ], axis=1)
        return df

    def ExpensesDF(self):
        return 1

class Model(BaseModel):
    def __init__(self, models):
        self.Models = models
        self.update_model(self.Models)
        self.WeatherModel = WeatherModel([self.LocationModel, self.PeriodModel])
        self.SolarModel = SolarModel([
            self.LocationModel,
            self.PeriodModel,
            self.MarketModel,
            self.PricingModel,
            self.WeatherModel,
            self.TechnologyModel,
            self.FinanceModel
        ])
        # self.ValueModel = ValueModel([
        #     self.LocationModel,
        #     self.TechnologyModel,
        #     self.SolarModel,
        #     self.MarketModel,
        #     self.PricingModel,
        #     self.FinanceModel,
        #     self.OperatingCostModel,
        #     self.CapitalCostModel
        # ])
        # self.update_model([self.WeatherModel, self.SolarModel, self.MarketModel, self.ValueModel])


class Models(BaseModel):
    def __init__(self):
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


class RunModel(BaseModel):
    def __init__(self, models):
        self.ModelsIn = models
        self.update_model(models)
        self.RunTacker = 0

    def build_mode(self, location):
        return Model(
            [location,
             self.PeriodModel,
             self.MarketModel,
             self.PricingModels[location.Zone],
             self.TechnologyModel,
             self.OperatingCostModel,
             self.CapitalCostModel,
             self.FinanceModel,
             ]
        )

    def reset(self):
        self.__init__(self.ModelsIn)

    def rebuild(self):
        self.PeriodModel = PeriodModel(self.FinanceModel.StartDate, self.FinanceModel.EndDate, 'US/Central')
        self.PricingModels = {i: PricingModel(i, [self.PeriodModel, self.DiscountModel, self.MarketModel]) for i in self.Zones}
        with Pool() as pool:
            self.Models = pool.map(self.build_mode, self.Locations)
        # self.Models = [self.build_mode(i) for i in self.Locations]
        # self.DF = self.build_dataframe()

    def build_dataframe(self):
        cols = ['Name', 'FIPS', 'Lat', 'Long', 'Zone', 'LandPrice', 'FairValue', 'AverageDNI', 'AverageDHI', 'AverageMW']
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
        dct = {
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
        df = county_model.IncomeDF.reset_index().to_dict('records')

        return df, dct

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
    model = Models()
    # setup_app(model)
    model.RunModel.rebuild()
    self = model.RunModel.Models[0].SolarModel