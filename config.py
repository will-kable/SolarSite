import pandas
import json
from timezonefinder import TimezoneFinder
import plotly.express as px


COUNTIES = pandas.read_csv("Config/Texas_Counties_Centroid_Map.csv").fillna('')

LAND_PRICES = json.load(open('Config/land_prices.json'))

FINDER = TimezoneFinder()

GEO_JSON = json.load(open('Config/geo_json.json'))

PROP_TAXES = json.load(open('Config/property_tax.json'))

API_KEY_NREL = ""

TRANS_CORDS = pandas.read_csv("Config/transmission_coords_texas_ercot_only.csv")

LOAD_ZONE_CORDS = pandas.read_csv("Config/loadzone_coords.csv")

LOAD_ZONE_TRACES = px.scatter_geo(LOAD_ZONE_CORDS, 'Lat', 'Long', text='Zone').update_traces(mode="lines", line=dict(color="Black", width=1)).data

MAPBOX_TOKEN = "pk.eyJ1Ijoid2lsbGthYmxlIiwiYSI6ImNsdXl0NWdqNjBidjIyams3bHVsOWhlZGUifQ.fhRt22a1eXsYVRecPpiWyw"

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