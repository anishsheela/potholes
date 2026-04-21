"""
Step 3 of the road quality mapping pipeline.

Reads road_quality.gpkg produced by snap_to_roads.py and renders an
interactive Folium HTML map with road segments coloured by quality.

Features
--------
- Two toggle-able layers: Pessimistic and Majority-vote quality
- Hover tooltip: street name, quality, observation count, score
- Colour-coded legend
- Observation count layer (line weight scales with obs_count)
- Clean CartoDB Positron base map

Usage
-----
    python src/mapping/render_map.py \
        --input processed_data/mapping/road_quality.gpkg \
        --output processed_data/mapping/road_quality_map.html

    # Choose aggregation shown by default:
    python src/mapping/render_map.py \
        --input processed_data/mapping/road_quality.gpkg \
        --default-layer majority \
        --output processed_data/mapping/road_quality_map.html
"""

import os
import argparse
import json

import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MousePosition

# ── Colour scheme ─────────────────────────────────────────────────────────────
QUALITY_COLOUR = {
    # 3-class merged model
    'Bad':       '#d73027',   # red
    'Good':      '#a8c92e',   # yellow-green
    # 5-class full model
    'Poor':      '#d73027',   # red
    'Fair':      '#f4a11d',   # amber
    'Excellent': '#1a9850',   # dark green
}
FALLBACK_COLOUR = '#95a5a6'   # grey — should not appear if data is clean

# Line weight: base + per-observation scaling (capped)
BASE_WEIGHT   = 3
OBS_WEIGHT_MAX = 7
OBS_SCALE_CAP  = 50          # obs_count beyond this gives max weight


def quality_colour(label: str) -> str:
    return QUALITY_COLOUR.get(label, FALLBACK_COLOUR)


def obs_weight(obs_count: int) -> float:
    """Scale line weight linearly from BASE_WEIGHT to OBS_WEIGHT_MAX."""
    t = min(obs_count, OBS_SCALE_CAP) / OBS_SCALE_CAP
    return BASE_WEIGHT + t * (OBS_WEIGHT_MAX - BASE_WEIGHT)


def clean_name(name_val) -> str:
    """OSMnx sometimes stores street names as Python lists — flatten them."""
    if pd.isna(name_val):
        return 'Unnamed road'
    if isinstance(name_val, list):
        return ', '.join(str(n) for n in name_val if n)
    return str(name_val)


# ── GeoJSON feature builders ──────────────────────────────────────────────────

def build_geojson(gdf: gpd.GeoDataFrame, label_col: str) -> dict:
    """
    Convert a GeoDataFrame of road edges to a GeoJSON FeatureCollection,
    embedding all properties needed for styling and tooltips.
    """
    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        label = row.get(label_col, '')
        props = {
            'quality':    label,
            'colour':     quality_colour(label),
            'weight':     obs_weight(int(row.get('obs_count', 1))),
            'name':       clean_name(row.get('name')),
            'highway':    str(row.get('highway', '')),
            'obs_count':  int(row.get('obs_count', 0)),
            'score':      float(row.get('quality_score', 0)),
            'majority':   str(row.get('quality_majority', '')),
            'pessimistic':str(row.get('quality_pessimistic', '')),
            'n_excellent':int(row.get('count_excellent', 0)),
            'n_good':     int(row.get('count_good',      0)),
            'n_fair':     int(row.get('count_fair',      0)),
            'n_poor':     int(row.get('count_poor',      0)),
        }
        features.append({
            'type': 'Feature',
            'geometry': geom.__geo_interface__,
            'properties': props,
        })

    return {'type': 'FeatureCollection', 'features': features}


# ── Folium layer builders ─────────────────────────────────────────────────────

def make_quality_layer(geojson: dict, layer_name: str, show: bool) -> folium.GeoJson:
    tooltip = folium.GeoJsonTooltip(
        fields=['name', 'quality', 'obs_count', 'score',
                'n_excellent', 'n_good', 'n_fair', 'n_poor'],
        aliases=['Street', 'Quality', 'Observations', 'Avg score',
                 'Excellent', 'Good', 'Fair', 'Poor'],
        localize=True,
        sticky=True,
        style=(
            'background-color: white; color: #333; font-family: sans-serif;'
            'font-size: 12px; padding: 6px; border-radius: 4px;'
        ),
    )

    return folium.GeoJson(
        geojson,
        name=layer_name,
        show=show,
        style_function=lambda f: {
            'color':   f['properties']['colour'],
            'weight':  f['properties']['weight'],
            'opacity': 0.85,
        },
        highlight_function=lambda f: {
            'color':   f['properties']['colour'],
            'weight':  f['properties']['weight'] + 2,
            'opacity': 1.0,
        },
        tooltip=tooltip,
    )


def make_obs_count_layer(gdf: gpd.GeoDataFrame, show: bool = False) -> folium.GeoJson:
    """A separate layer that shows observation density by line weight."""
    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        obs = int(row.get('obs_count', 1))
        features.append({
            'type': 'Feature',
            'geometry': geom.__geo_interface__,
            'properties': {
                'obs_count': obs,
                'weight':    obs_weight(obs),
                'name':      clean_name(row.get('name')),
            },
        })

    tooltip = folium.GeoJsonTooltip(
        fields=['name', 'obs_count'],
        aliases=['Street', 'Observations'],
        sticky=True,
    )

    return folium.GeoJson(
        {'type': 'FeatureCollection', 'features': features},
        name='Observation density',
        show=show,
        style_function=lambda f: {
            'color':   '#2980b9',
            'weight':  f['properties']['weight'],
            'opacity': 0.70,
        },
        tooltip=tooltip,
    )


# ── Legend HTML ───────────────────────────────────────────────────────────────

# Label display order: worst → best
LABEL_ORDER = ['Bad', 'Poor', 'Fair', 'Good', 'Excellent']


def build_legend_html(labels: list) -> str:
    """Build legend HTML for only the labels present in this dataset."""
    entries = ''
    for lbl in LABEL_ORDER:
        if lbl in labels:
            colour = QUALITY_COLOUR.get(lbl, FALLBACK_COLOUR)
            entries += (
                f'<span style="color:{colour};font-size:18px;">&#9644;</span>'
                f' {lbl}<br>'
            )
    return f"""
<div style="
    position: fixed;
    bottom: 40px; left: 10px;
    z-index: 1000;
    background: white;
    padding: 12px 16px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    font-family: sans-serif;
    font-size: 13px;
    line-height: 1.6;
">
  <b>Road Quality</b><br>
  {entries}
  <hr style="margin:6px 0">
  <span style="color:#2980b9;font-size:18px;">&#9644;</span> Observation density
</div>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Render an interactive road quality map from road_quality.gpkg'
    )
    parser.add_argument('--input', type=str,
                        default='processed_data/mapping/road_quality.gpkg',
                        help='GeoPackage produced by snap_to_roads.py')
    parser.add_argument('--output', type=str,
                        default='processed_data/mapping/road_quality_map.html',
                        help='Output HTML file path')
    parser.add_argument('--default-layer', type=str,
                        default='pessimistic',
                        choices=['pessimistic', 'majority'],
                        help='Which quality layer is visible on load')
    parser.add_argument('--min-obs', type=int, default=1,
                        help='Hide edges with fewer observations than this')
    parser.add_argument('--tile', type=str,
                        default='CartoDB positron',
                        help='Folium tile layer name (default: CartoDB positron)')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f'Loading {args.input}...')
    gdf = gpd.read_file(args.input, layer='road_quality')
    gdf = gdf.to_crs('EPSG:4326')
    print(f'  {len(gdf):,} road edges loaded.')

    if args.min_obs > 1:
        before = len(gdf)
        gdf = gdf[gdf['obs_count'] >= args.min_obs]
        print(f'  Filtered to {len(gdf):,} edges with >= {args.min_obs} observations '
              f'(removed {before - len(gdf):,}).')

    if gdf.empty:
        print('No edges to render. Exiting.')
        return

    # ── Map centre ────────────────────────────────────────────────────────────
    bounds = gdf.total_bounds          # (minx, miny, maxx, maxy)
    centre_lat = (bounds[1] + bounds[3]) / 2
    centre_lon = (bounds[0] + bounds[2]) / 2

    print(f'\nRendering map centred at ({centre_lat:.5f}, {centre_lon:.5f})...')

    # ── Build Folium map ──────────────────────────────────────────────────────
    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=14,
        tiles=args.tile,
        control_scale=True,
    )

    # Fit map to data extent
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # ── Build GeoJSON payloads ─────────────────────────────────────────────────
    pess_geojson = build_geojson(gdf, 'quality_pessimistic')
    maj_geojson  = build_geojson(gdf, 'quality_majority')

    show_pess = (args.default_layer == 'pessimistic')
    show_maj  = (args.default_layer == 'majority')

    pess_layer = make_quality_layer(pess_geojson, 'Quality (pessimistic)', show=show_pess)
    maj_layer  = make_quality_layer(maj_geojson,  'Quality (majority)',    show=show_maj)
    obs_layer  = make_obs_count_layer(gdf, show=False)

    pess_layer.add_to(m)
    maj_layer.add_to(m)
    obs_layer.add_to(m)

    # ── UI extras ─────────────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(m)
    MousePosition(
        position='bottomright',
        separator=' | ',
        prefix='Lat/Lon:',
    ).add_to(m)

    # Dynamic legend — only labels present in this dataset
    present_labels = gdf['quality_pessimistic'].unique().tolist()
    m.get_root().html.add_child(folium.Element(build_legend_html(present_labels)))

    # ── Summary stats in map title ────────────────────────────────────────────
    total = len(gdf)
    pess_counts = gdf['quality_pessimistic'].value_counts()
    stats_lines = ' &nbsp;|&nbsp; '.join(
        f'<span style="color:{quality_colour(lbl)};font-weight:bold;">'
        f'{lbl}</span> {pess_counts.get(lbl, 0):,}'
        for lbl in LABEL_ORDER
        if lbl in present_labels
    )
    title_html = f"""
    <div style="
        position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
        z-index: 1000; background: white; padding: 8px 18px;
        border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        font-family: sans-serif; font-size: 13px; white-space: nowrap;
    ">
      <b>Road Quality Map</b> &nbsp;—&nbsp; {total:,} segments
      &nbsp;&nbsp; {stats_lines}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    m.save(args.output)

    print(f'\n{"="*55}')
    print(f'Map saved to {args.output}')
    print(f'{"="*55}')
    print(f'  Total road segments : {total:,}')
    print(f'\nQuality breakdown (pessimistic):')
    for lbl in LABEL_ORDER:
        if lbl in present_labels:
            n = pess_counts.get(lbl, 0)
            pct = 100 * n / max(total, 1)
            print(f'  {lbl:<12} {n:>5,}  ({pct:.1f}%)')
    print(f'\nOpen in a browser: {os.path.abspath(args.output)}')


if __name__ == '__main__':
    main()
