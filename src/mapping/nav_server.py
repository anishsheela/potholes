"""
Live quality-aware navigation demo server.

Loads the OSM graph and road quality data once at startup, then serves:
  GET  /              → Leaflet.js navigation UI
  GET  /api/quality   → GeoJSON of all rated road segments
  POST /api/route     → Dual route (shortest vs smoothest) between two points

Run from project root:
    python src/mapping/nav_server.py

Then open http://localhost:5050
"""

import os
import argparse

import pandas as pd
import geopandas as gpd
from flask import Flask, jsonify, request, render_template
import osmnx as ox
import networkx as nx
from shapely import wkt as shapely_wkt

# ── Constants ─────────────────────────────────────────────────────────────────

QUALITY_SCORE  = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4, 'Bad': 1}
FALLBACK_SCORE = 2.5
UNRATED_MULT   = 1.2   # slight penalty for roads with no quality data

QUALITY_COLOUR = {
    'Poor':      '#d73027',
    'Fair':      '#f4a11d',
    'Good':      '#a8c92e',
    'Excellent': '#1a9850',
    'Bad':       '#d73027',
}

DEFAULT_GPKG  = 'processed_data/mapping/5class/road_quality.gpkg'
DEFAULT_GRAPH = 'processed_data/mapping/osm_graph.graphml'

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_name(v):
    if pd.isna(v) if not isinstance(v, list) else False:
        return 'Unnamed road'
    if isinstance(v, list):
        return ', '.join(str(x) for x in v if x) or 'Unnamed road'
    s = str(v).strip()
    return s if s and s != 'nan' else 'Unnamed road'


def build_quality_lookup(gdf, quality_col):
    lookup = {}
    for _, row in gdf.iterrows():
        label = str(row.get(quality_col, ''))
        score = QUALITY_SCORE.get(label, FALLBACK_SCORE)
        lookup[(int(row['u']), int(row['v']), int(row['key']))] = {
            'score': score, 'label': label,
        }
    return lookup


def inject_quality_weights(G, lookup):
    for u, v, k, data in G.edges(keys=True, data=True):
        length = data.get('length', 1.0)
        entry  = lookup.get((u, v, k)) or lookup.get((v, u, k))
        if entry:
            penalty = 5.0 - entry['score']
        else:
            penalty = (5.0 - FALLBACK_SCORE) * UNRATED_MULT
        data['quality_weight'] = length * penalty


def route_coords(G, nodes):
    """Return [[lat, lon], ...] for a node sequence, using edge geometries where available."""
    coords = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            continue
        k    = min(edge_data.keys())
        data = edge_data[k]
        geom = data.get('geometry')

        if geom is not None:
            if isinstance(geom, str):
                try:
                    geom = shapely_wkt.loads(geom)
                except Exception:
                    geom = None
            if geom is not None:
                pts = [[lat, lon] for lon, lat in geom.coords]
                if coords and coords[-1] == pts[0]:
                    pts = pts[1:]
                coords.extend(pts)
                continue

        # Fallback: straight line between nodes
        u_pt = [G.nodes[u]['y'], G.nodes[u]['x']]
        v_pt = [G.nodes[v]['y'], G.nodes[v]['x']]
        if not coords:
            coords.append(u_pt)
        elif coords[-1] == v_pt:
            continue
        coords.append(v_pt)

    return coords


def route_edge_qualities(G, nodes, lookup):
    """Per-edge quality list for the 'upcoming road quality' feature."""
    edges = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            continue
        k    = min(edge_data.keys())
        data = edge_data[k]
        entry = lookup.get((u, v, k)) or lookup.get((v, u, k))
        edges.append({
            'length':  round(data.get('length', 0), 1),
            'name':    clean_name(data.get('name', '')),
            'quality': entry['label'] if entry else None,
            'score':   round(entry['score'], 2) if entry else None,
        })
    return edges


def route_stats(G, nodes, lookup):
    total = scored = wsum = 0.0
    label_dist: dict[str, float] = {}

    for u, v in zip(nodes[:-1], nodes[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            continue
        length = edge_data[min(edge_data.keys())].get('length', 0.0)
        total += length
        entry = lookup.get((u, v, min(edge_data.keys()))) or lookup.get((v, u, min(edge_data.keys())))
        if entry:
            wsum   += entry['score'] * length
            scored += length
            label_dist[entry['label']] = label_dist.get(entry['label'], 0.0) + length

    return {
        'distance_m': round(total),
        'avg_score':  round(wsum / scored, 2) if scored > 0 else None,
        'rated_pct':  round(100 * scored / total, 1) if total > 0 else 0,
        'label_dist': {k: round(v) for k, v in label_dist.items()},
    }


def build_quality_geojson(gdf, quality_col):
    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        label = str(row.get(quality_col, ''))
        obs   = int(row.get('obs_count', 1))
        t     = min(obs, 50) / 50
        features.append({
            'type': 'Feature',
            'geometry': geom.__geo_interface__,
            'properties': {
                'quality':   label,
                'colour':    QUALITY_COLOUR.get(label, '#95a5a6'),
                'weight':    round(3 + t * 4, 1),
                'name':      clean_name(row.get('name', '')),
                'obs_count': obs,
                'score':     round(float(row.get('quality_score', 0)), 2),
            },
        })
    return {'type': 'FeatureCollection', 'features': features}


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder='templates')

_G              = None
_quality_lookup = None
_quality_geojson = None
_map_centre     = None


@app.route('/')
def index():
    return render_template('nav.html', centre=_map_centre)


@app.route('/api/quality')
def get_quality():
    return jsonify(_quality_geojson)


@app.route('/api/route', methods=['POST'])
def get_route():
    body  = request.get_json()
    start = body.get('start')   # [lat, lon]
    end   = body.get('end')     # [lat, lon]

    if not start or not end:
        return jsonify({'error': 'start and end are required'}), 400

    try:
        orig = ox.nearest_nodes(_G, X=start[1], Y=start[0])
        dest = ox.nearest_nodes(_G, X=end[1],   Y=end[0])
    except Exception as e:
        return jsonify({'error': f'Node lookup failed: {e}'}), 500

    try:
        nodes_short  = nx.shortest_path(_G, orig, dest, weight='length')
        nodes_smooth = nx.shortest_path(_G, orig, dest, weight='quality_weight')
    except nx.NetworkXNoPath:
        return jsonify({'error': 'No path found between these points. Try points closer to mapped roads.'}), 404

    return jsonify({
        'shortest': {
            'coords': route_coords(_G, nodes_short),
            'stats':  route_stats(_G, nodes_short, _quality_lookup),
            'edges':  route_edge_qualities(_G, nodes_short, _quality_lookup),
        },
        'smoothest': {
            'coords': route_coords(_G, nodes_smooth),
            'stats':  route_stats(_G, nodes_smooth, _quality_lookup),
            'edges':  route_edge_qualities(_G, nodes_smooth, _quality_lookup),
        },
    })


# ── Startup ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Quality-aware live navigation demo server')
    p.add_argument('--gpkg',        default=DEFAULT_GPKG)
    p.add_argument('--graph',       default=DEFAULT_GRAPH)
    p.add_argument('--quality-col', default='quality_majority',
                   choices=['quality_majority', 'quality_pessimistic'])
    p.add_argument('--port', type=int, default=5050)
    p.add_argument('--host', default='127.0.0.1')
    return p.parse_args()


def main():
    global _G, _quality_lookup, _quality_geojson, _map_centre
    args = parse_args()

    print(f'Loading OSM graph from {args.graph}...')
    _G = ox.load_graphml(args.graph)
    print(f'  {len(_G.nodes):,} nodes, {len(_G.edges):,} edges')

    print(f'Loading road quality from {args.gpkg}...')
    gdf = gpd.read_file(args.gpkg, layer='road_quality').to_crs('EPSG:4326')
    print(f'  {len(gdf):,} rated segments (column: {args.quality_col})')

    _quality_lookup = build_quality_lookup(gdf, args.quality_col)

    print('Injecting quality weights...')
    inject_quality_weights(_G, _quality_lookup)

    print('Building quality GeoJSON...')
    _quality_geojson = build_quality_geojson(gdf, args.quality_col)

    b = gdf.total_bounds
    _map_centre = [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]

    print(f'\nReady →  http://{args.host}:{args.port}\n')
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
