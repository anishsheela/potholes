"""
Quality-aware route comparison.

Computes two routes between a start and end point:
  1. Shortest path  — minimises total distance (standard routing)
  2. Smoothest path — minimises a quality-penalised distance

     weight = length × (5 − quality_score)

  quality_score is 1–4 (Poor=1 … Excellent=4).
  An Excellent road (score 4) gets weight = 1×length.
  A Poor road      (score 1) gets weight = 4×length.
  Unrated edges use a neutral fallback score of 2.5.

The output is an HTML Folium map overlaying:
  • Road quality layer (coloured segments from road_quality.gpkg)
  • Shortest route   (blue dashed)
  • Smoothest route  (purple solid)
  • Info panel comparing the two routes

Usage
-----
    python src/mapping/route_quality.py

    # Custom endpoints:
    python src/mapping/route_quality.py \
        --start "8.5530,76.9815" \
        --end   "8.6450,77.0200"

    # Use pessimistic quality instead of majority:
    python src/mapping/route_quality.py --quality-col quality_pessimistic
"""

import os
import argparse
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MousePosition
import osmnx as ox
import networkx as nx

# ── Constants ─────────────────────────────────────────────────────────────────

QUALITY_SCORE = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4, 'Bad': 1}
FALLBACK_SCORE = 2.5   # neutral score for unrated edges
UNRATED_PENALTY = 1.2  # extra multiplier for unrated edges (slight preference for rated)

QUALITY_COLOUR = {
    'Poor':      '#d73027',
    'Fair':      '#f4a11d',
    'Good':      '#a8c92e',
    'Excellent': '#1a9850',
    'Bad':       '#d73027',
}
FALLBACK_COLOUR = '#95a5a6'

ROUTE_SHORTEST_COLOUR  = '#2471a3'   # blue
ROUTE_SMOOTHEST_COLOUR = '#7d3c98'   # purple

DEFAULT_GPKG  = 'processed_data/mapping/5class/road_quality.gpkg'
DEFAULT_GRAPH = 'processed_data/mapping/osm_graph.graphml'
DEFAULT_OUT   = 'processed_data/mapping/route_comparison.html'

# Default endpoints chosen to force a route through the Fair/Poor corridor
# around Kunjukonam Road (well-covered area).
DEFAULT_START = (8.5530, 76.9815)
DEFAULT_END   = (8.6450, 77.0230)


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_name(v) -> str:
    if pd.isna(v):
        return 'Unnamed road'
    if isinstance(v, list):
        return ', '.join(str(x) for x in v if x)
    return str(v)


def quality_colour(label: str) -> str:
    return QUALITY_COLOUR.get(label, FALLBACK_COLOUR)


def obs_weight(obs: int, cap: int = 50, base: float = 3, top: float = 7) -> float:
    t = min(obs, cap) / cap
    return base + t * (top - base)


def route_coords(G, nodes: list) -> list[tuple[float, float]]:
    """Return [(lat, lon), ...] for a node list."""
    return [(G.nodes[n]['y'], G.nodes[n]['x']) for n in nodes]


def route_stats(G, nodes: list, quality_lookup: dict) -> dict:
    """Compute distance and quality breakdown for a route."""
    total_dist = 0.0
    scored_dist = 0.0
    weighted_score_sum = 0.0
    label_dist: dict[str, float] = {}

    for u, v in zip(nodes[:-1], nodes[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            continue
        # pick key 0 or the only key available
        k = min(edge_data.keys())
        length = edge_data[k].get('length', 0.0)
        total_dist += length

        key = (u, v, k)
        rev_key = (v, u, k)
        entry = quality_lookup.get(key) or quality_lookup.get(rev_key)
        if entry:
            score = entry['score']
            label = entry['label']
            weighted_score_sum += score * length
            scored_dist += length
            label_dist[label] = label_dist.get(label, 0.0) + length

    avg_score = weighted_score_sum / scored_dist if scored_dist > 0 else None
    rated_pct = 100 * scored_dist / total_dist if total_dist > 0 else 0

    return {
        'distance_m':  round(total_dist),
        'avg_score':   round(avg_score, 2) if avg_score else None,
        'rated_pct':   round(rated_pct, 1),
        'label_dist':  label_dist,
    }


# ── Graph weight injection ────────────────────────────────────────────────────

def inject_quality_weights(G, quality_lookup: dict):
    """Add 'quality_weight' to every edge in G (in-place)."""
    for u, v, k, data in G.edges(keys=True, data=True):
        length = data.get('length', 1.0)
        key = (u, v, k)
        rev_key = (v, u, k)
        entry = quality_lookup.get(key) or quality_lookup.get(rev_key)
        if entry:
            score = entry['score']
            penalty = 5.0 - score          # Poor→4, Fair→3, Good→2, Excellent→1
        else:
            penalty = 5.0 - FALLBACK_SCORE  # = 2.5, times UNRATED_PENALTY
            penalty *= UNRATED_PENALTY
        data['quality_weight'] = length * penalty


# ── Map rendering ─────────────────────────────────────────────────────────────

def make_quality_layer(gdf: gpd.GeoDataFrame, label_col: str,
                       name: str, show: bool) -> folium.GeoJson:
    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        label = row.get(label_col, '')
        obs   = int(row.get('obs_count', 1))
        features.append({
            'type': 'Feature',
            'geometry': geom.__geo_interface__,
            'properties': {
                'quality':   label,
                'colour':    quality_colour(label),
                'weight':    obs_weight(obs),
                'name':      clean_name(row.get('name')),
                'obs_count': obs,
                'score':     float(row.get('quality_score', 0)),
            },
        })
    geojson = {'type': 'FeatureCollection', 'features': features}

    tooltip = folium.GeoJsonTooltip(
        fields=['name', 'quality', 'obs_count', 'score'],
        aliases=['Street', 'Quality', 'Observations', 'Avg score'],
        sticky=True,
        style='background-color:white;font-family:sans-serif;font-size:12px;padding:6px;border-radius:4px;',
    )

    return folium.GeoJson(
        geojson, name=name, show=show,
        style_function=lambda f: {
            'color':   f['properties']['colour'],
            'weight':  f['properties']['weight'],
            'opacity': 0.75,
        },
        highlight_function=lambda f: {
            'color':   f['properties']['colour'],
            'weight':  f['properties']['weight'] + 2,
            'opacity': 1.0,
        },
        tooltip=tooltip,
    )


def score_to_label(score: float | None) -> str:
    if score is None:
        return 'N/A'
    if score >= 3.5:
        return f'{score:.2f} (Excellent)'
    if score >= 2.5:
        return f'{score:.2f} (Good)'
    if score >= 1.5:
        return f'{score:.2f} (Fair)'
    return f'{score:.2f} (Poor)'


def build_info_html(stats_short: dict, stats_smooth: dict,
                    start: tuple, end: tuple) -> str:
    def dist_km(m):
        return f'{m/1000:.2f} km' if m else '—'

    def breakdown(ld: dict) -> str:
        order = ['Poor', 'Fair', 'Bad', 'Good', 'Excellent']
        parts = []
        for lbl in order:
            d = ld.get(lbl, 0)
            if d > 0:
                col = quality_colour(lbl)
                parts.append(
                    f'<span style="color:{col};font-weight:bold;">{lbl}</span>'
                    f' {d/1000:.2f} km'
                )
        return ' &nbsp;·&nbsp; '.join(parts) if parts else '—'

    shared_row = (
        f'<tr><td style="padding:4px 8px 4px 0"><b>Rated road coverage</b></td>'
        f'<td style="padding:4px 8px">{stats_short["rated_pct"]}%</td>'
        f'<td style="padding:4px 8px">{stats_smooth["rated_pct"]}%</td></tr>'
    )

    return f"""
<div style="
    position:fixed; bottom:40px; right:10px; z-index:1000;
    background:white; padding:14px 18px; border-radius:10px;
    box-shadow:0 2px 10px rgba(0,0,0,0.25);
    font-family:sans-serif; font-size:12px; max-width:360px;
">
  <b style="font-size:14px;">Route Comparison</b>
  <table style="width:100%;margin-top:8px;border-collapse:collapse;">
    <tr style="border-bottom:1px solid #eee;">
      <th style="padding:4px 8px 4px 0;text-align:left;"></th>
      <th style="padding:4px 8px;color:{ROUTE_SHORTEST_COLOUR};">Shortest</th>
      <th style="padding:4px 8px;color:{ROUTE_SMOOTHEST_COLOUR};">Smoothest</th>
    </tr>
    <tr>
      <td style="padding:4px 8px 4px 0"><b>Distance</b></td>
      <td style="padding:4px 8px">{dist_km(stats_short['distance_m'])}</td>
      <td style="padding:4px 8px">{dist_km(stats_smooth['distance_m'])}</td>
    </tr>
    <tr>
      <td style="padding:4px 8px 4px 0"><b>Avg quality score</b></td>
      <td style="padding:4px 8px">{score_to_label(stats_short['avg_score'])}</td>
      <td style="padding:4px 8px">{score_to_label(stats_smooth['avg_score'])}</td>
    </tr>
    {shared_row}
    <tr>
      <td style="padding:4px 8px 4px 0;vertical-align:top"><b>Quality breakdown</b></td>
      <td style="padding:4px 8px" colspan="2">&nbsp;</td>
    </tr>
    <tr>
      <td style="padding:2px 8px 2px 16px">Shortest</td>
      <td style="padding:2px 8px" colspan="2">{breakdown(stats_short['label_dist'])}</td>
    </tr>
    <tr>
      <td style="padding:2px 8px 2px 16px">Smoothest</td>
      <td style="padding:2px 8px" colspan="2">{breakdown(stats_smooth['label_dist'])}</td>
    </tr>
  </table>
  <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
  <span style="color:#666;font-size:11px;">
    Start: {start[0]:.5f}, {start[1]:.5f}<br>
    End: &nbsp;{end[0]:.5f}, {end[1]:.5f}
  </span>
</div>
"""


def build_legend_html() -> str:
    entries = ''
    for lbl, col in [('Poor', '#d73027'), ('Fair', '#f4a11d'),
                     ('Good', '#a8c92e'), ('Excellent', '#1a9850')]:
        entries += f'<span style="color:{col};font-size:18px;">&#9644;</span> {lbl}<br>'
    return f"""
<div style="
    position:fixed; bottom:40px; left:10px; z-index:1000;
    background:white; padding:12px 16px; border-radius:8px;
    box-shadow:0 2px 8px rgba(0,0,0,0.25);
    font-family:sans-serif; font-size:13px; line-height:1.7;
">
  <b>Road Quality</b><br>
  {entries}
  <hr style="margin:6px 0;border:none;border-top:1px solid #eee;">
  <span style="color:{ROUTE_SHORTEST_COLOUR};font-weight:bold;">- - -</span> Shortest route<br>
  <span style="color:{ROUTE_SMOOTHEST_COLOUR};font-weight:bold;">———</span> Smoothest route
</div>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Compare shortest vs quality-aware routes')
    p.add_argument('--start', type=str, default=None,
                   help='Start coordinate as "lat,lon" (default: built-in demo point)')
    p.add_argument('--end', type=str, default=None,
                   help='End coordinate as "lat,lon" (default: built-in demo point)')
    p.add_argument('--gpkg', type=str, default=DEFAULT_GPKG,
                   help='Road quality GeoPackage (from snap_to_roads.py)')
    p.add_argument('--graph', type=str, default=DEFAULT_GRAPH,
                   help='Cached OSM GraphML (from snap_to_roads.py)')
    p.add_argument('--quality-col', type=str, default='quality_majority',
                   choices=['quality_majority', 'quality_pessimistic'],
                   help='Which quality column to use for routing weights')
    p.add_argument('--output', type=str, default=DEFAULT_OUT,
                   help='Output HTML path')
    return p.parse_args()


def main():
    args = parse_args()

    start = tuple(float(x) for x in args.start.split(',')) if args.start else DEFAULT_START
    end   = tuple(float(x) for x in args.end.split(','))   if args.end   else DEFAULT_END

    # ── Load graph ────────────────────────────────────────────────────────────
    print(f'Loading OSM graph from {args.graph}...')
    G = ox.load_graphml(args.graph)
    print(f'  {len(G.nodes):,} nodes, {len(G.edges):,} edges')

    # ── Load road quality ─────────────────────────────────────────────────────
    print(f'Loading road quality from {args.gpkg}...')
    gdf = gpd.read_file(args.gpkg, layer='road_quality').to_crs('EPSG:4326')
    print(f'  {len(gdf):,} rated road segments (using column: {args.quality_col})')

    # Build (u, v, key) → {score, label} lookup
    quality_lookup: dict[tuple, dict] = {}
    for _, row in gdf.iterrows():
        label = str(row.get(args.quality_col, ''))
        score = QUALITY_SCORE.get(label, FALLBACK_SCORE)
        quality_lookup[(int(row['u']), int(row['v']), int(row['key']))] = {
            'score': score,
            'label': label,
        }

    # ── Inject quality weights ─────────────────────────────────────────────────
    print('Injecting quality weights into graph...')
    inject_quality_weights(G, quality_lookup)

    # ── Find nearest graph nodes to start/end ─────────────────────────────────
    print(f'\nStart: {start}  →  End: {end}')
    orig = ox.nearest_nodes(G, X=start[1], Y=start[0])
    dest = ox.nearest_nodes(G, X=end[1],   Y=end[0])
    print(f'  Snapped to nodes: {orig} → {dest}')

    # ── Compute routes ────────────────────────────────────────────────────────
    print('\nComputing shortest route (distance)...')
    try:
        route_short = nx.shortest_path(G, orig, dest, weight='length')
    except nx.NetworkXNoPath:
        print('  ERROR: No path found. Try different start/end coordinates.')
        return

    print('Computing smoothest route (quality-weighted)...')
    try:
        route_smooth = nx.shortest_path(G, orig, dest, weight='quality_weight')
    except nx.NetworkXNoPath:
        print('  ERROR: No quality-weighted path found.')
        return

    # ── Compute stats ─────────────────────────────────────────────────────────
    stats_short  = route_stats(G, route_short,  quality_lookup)
    stats_smooth = route_stats(G, route_smooth, quality_lookup)

    print(f'\n{"="*55}')
    print(f'Shortest route:   {stats_short["distance_m"]:,} m  '
          f'| avg score: {stats_short["avg_score"]}  '
          f'| rated: {stats_short["rated_pct"]}%')
    print(f'Smoothest route:  {stats_smooth["distance_m"]:,} m  '
          f'| avg score: {stats_smooth["avg_score"]}  '
          f'| rated: {stats_smooth["rated_pct"]}%')

    extra_dist = stats_smooth['distance_m'] - stats_short['distance_m']
    if extra_dist > 0:
        print(f'Quality detour:   +{extra_dist:,} m ({100*extra_dist/max(stats_short["distance_m"],1):.1f}% longer)')
    else:
        print(f'Routes identical in distance (or smoothest is shorter).')
    print(f'{"="*55}')

    # ── Build map ─────────────────────────────────────────────────────────────
    coords_short  = route_coords(G, route_short)
    coords_smooth = route_coords(G, route_smooth)

    all_lats = [c[0] for c in coords_short + coords_smooth]
    all_lons = [c[1] for c in coords_short + coords_smooth]
    centre   = ((min(all_lats)+max(all_lats))/2, (min(all_lons)+max(all_lons))/2)

    print(f'\nRendering map...')
    m = folium.Map(location=centre, zoom_start=14, tiles='CartoDB positron',
                   control_scale=True)
    m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

    # Road quality background layer
    quality_layer = make_quality_layer(gdf, args.quality_col,
                                       name='Road quality', show=True)
    quality_layer.add_to(m)

    # Shortest route — blue dashed
    folium.PolyLine(
        coords_short,
        color=ROUTE_SHORTEST_COLOUR,
        weight=5,
        opacity=0.9,
        dash_array='10 6',
        tooltip=f'Shortest: {stats_short["distance_m"]/1000:.2f} km',
        name='Shortest route',
    ).add_to(m)

    # Smoothest route — purple solid
    folium.PolyLine(
        coords_smooth,
        color=ROUTE_SMOOTHEST_COLOUR,
        weight=5,
        opacity=0.9,
        tooltip=f'Smoothest: {stats_smooth["distance_m"]/1000:.2f} km',
        name='Smoothest route',
    ).add_to(m)

    # Start/end markers
    folium.Marker(
        location=start,
        tooltip='Start',
        icon=folium.Icon(color='green', icon='play', prefix='fa'),
    ).add_to(m)
    folium.Marker(
        location=end,
        tooltip='End',
        icon=folium.Icon(color='red', icon='flag', prefix='fa'),
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    MousePosition(position='bottomright', separator=' | ', prefix='Lat/Lon:').add_to(m)

    m.get_root().html.add_child(folium.Element(build_legend_html()))
    m.get_root().html.add_child(folium.Element(
        build_info_html(stats_short, stats_smooth, start, end)
    ))

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    m.save(args.output)
    print(f'Map saved → {os.path.abspath(args.output)}')


if __name__ == '__main__':
    main()
