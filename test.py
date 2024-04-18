import datetime
import time
import geojson
import plotly.express as px
import pandas
import geojson
import geopandas
from dash import Dash, dcc, html, callback, ALL, MATCH, Input, State, Output, Patch, no_update
import plotly.graph_objects as go
import numpy as np
from config import COUNTIES, API_KEY_NREL, FINDER, ZONAL_MAP, GEO_JSON, LAND_PRICES, ARRAY_MAP, MODULE_MAP, MODULE_EFF, PROP_TAXES, MAPBOX_TOKEN

px.set_mapbox_access_token(MAPBOX_TOKEN)

with open('Config/transmission.geojson') as f:
    poly_json = geojson.load(f)

df = geopandas.GeoDataFrame.from_features(poly_json)
df = df[json_string_column].apply(lambda x: poly_json.loads(x)['features'][0])

fig1 = px.choropleth(COUNTIES, geojson=GEO_JSON, locations='FIPS', scope='usa', fitbounds="locations", color_continuous_scale='Viridis',basemap_visible=False,center={"lat": 31.391489, "lon": -99.174304})
# fig1.update_layout(mapbox_layers=[{
#             "sourcetype": "geojson",
#             "type": "line",
#             "source": poly_json
#         }])
for feature in poly_json['features']:
    cordinates = feature['geometry']['coordinates']
    fig.add_trace(
        go.Scattergeo(
            lon = [df_flight_paths['start_lon'][i], df_flight_paths['end_lon'][i]],
            lat = [df_flight_paths['start_lat'][i], df_flight_paths['end_lat'][i]],
            mode = 'lines',
            line = dict(width = 1,color = 'red'),
            opacity = float(df_flight_paths['cnt'][i]) / float(df_flight_paths['cnt'].max()),
        )
    )
fig1.show()
