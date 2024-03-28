import pandas
from geopy.geocoders import Nominatim
import json
from timezonefinder import TimezoneFinder

COUNTIES = pandas.read_csv("Config/Texas_Counties_Centroid_Map.csv")

LAND_PRICES = json.load(open('Config/land_prices.json'))

FINDER = TimezoneFinder()

GEO_JSON = json.load(open('Config/geo_json.json'))

PROP_TAXES = json.load(open('Config/property_tax.json'))

API_KEY_NREL = 'me1DbZYqNf6JyoeT9XlnhQhHuzdjVsY4xbXzXzg0'

ARRAY_MAP = {
    'Fixed': 0,
    'Fixed Roof': 1,
    '1 Axis Tracker': 2,
    '1 Axis Backtracked': 3,
    '2 Axis Tracker': 4,
}

MODULE_MAP = {
    'Standard': 0,
    'Premium': 1,
    'Thin Film': 2
}

MODULE_EFF = {
    'Standard': 0.19,
    'Premium': 0.21,
    'Thin Film': 0.18
}

ZONAL_MAP = {
    'West': (32.381910, -100.922995),
    'North': (32.768378, -97.327781), # Dallas
    'South': (29.433180, -98.515710), # San Antonio
    'Houston': (29.754269, -95.365714), # Houston
}
