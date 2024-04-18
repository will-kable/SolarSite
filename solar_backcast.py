import os

import PySAM.Pvwattsv8 as PVW
import PySAM.ResourceTools as RT
import pandas


API_KEY = 'me1DbZYqNf6JyoeT9XlnhQhHuzdjVsY4xbXzXzg0'

LOCATIONS = pandas.read_csv('Config/uscounties.csv')

def fetch_year(year):
    locations = LOCATIONS[LOCATIONS.state_id == 'TX']
    cords = [(x['lng'], x['lat']) for idx, x in locations.iterrows()]
    fetcher = RT.FetchResourceFiles(
        'solar',
        API_KEY,
        'will.kable@axpo.com',
        resource_dir='data_weather\\Weather',
        workers=500,
        resource_type='psm3-2-2',
        resource_year=str(year),
        verbose=True,
    )
    fetcher = fetcher.fetch(cords)
    for (idx, row), file in zip(locations.iterrows(), fetcher.resource_file_paths):
        print(idx, row.county)
        if file:
            info = pandas.read_csv(file).iloc[:1].loc[0]
            df = pandas.read_csv(file, skiprows=2).assign(TimeZone=info['Time Zone'], Elevation=info.Elevation).dropna(axis=1)
            os.makedirs(f'/solar_weather_data/counties/{row.state_name}/{year}/', exist_ok=True)
            df.to_csv(f'/solar_weather_data/counties/{row.state_name}/{year}/{row.county}.csv')


def build_pv(data, cap, tilt, tz='US/Central'):
    PV_MODEL = PVW.default('PVWattsMerchantPlant')
    PV_MODEL.assign(
        {
            'SystemDesign': {
                'system_capacity': cap * 1000,
                'dc_ac_ratio': 1.3,
                'tilt': tilt,
                'azimuth': 180,
                'inv_eff': 96,
                'losses': 14.07,
                'array_type': 0,
                'module_type': 0,
                'gcr': 0.4,
            },
            'SolarResource': {
                'solar_resource_data': {
                    'lat': data.Lat.iloc[0],
                    'lon': data.Long.iloc[0],
                    'elev': data.Elv.iloc[0],
                    'tz': data.TZ.iloc[0],
                    'dn': tuple(data['DNI']),
                    'df': tuple(data['DHI']),
                    'gh': tuple(data['GHI']),
                    'tdry': tuple(data['Temperature']),
                    'wspd': tuple(data['Wind Speed']),
                    'year': tuple(data.Year),
                    'month': tuple(data.Month),
                    'day': tuple(data.Day),
                    'hour': tuple(data.Hour),
                    'minute': tuple(data.Hour),
                    'albedo': tuple(data['Surface Albedo']),
                    'use_wf_albedo': 1,
                    }
                },
            }
    )
    PV_MODEL.execute()
    gen = PV_MODEL.Outputs.gen
    idx = pandas.to_datetime(data[['Year', 'Month', 'Day', 'Hour']]).dt.tz_localize(f'Etc/GMT+{-data.TZ.iloc[0]}').dt.tz_convert(tz)
    df = pandas.DataFrame([
        idx.dt.date.values,
        (idx.dt.hour + 1).values,
        gen,
        [i/1000 for i in gen]
    ]).T.set_axis(idx).shift(freq='1h').set_axis(['Date', 'HE', 'KW', 'MW'], axis=1)
    return df

if __name__ == '__main__':
    pass