"""
Step 2 of the road quality mapping pipeline.

Reads the classified-frame CSV produced by classify_dataset.py, downloads the
OSM drive network for the covered area, snaps every GPS point to its nearest
road edge, and aggregates quality labels per edge.

Aggregation modes
-----------------
pessimistic (default)
    The worst label observed on a segment wins.
    One Poor observation beats ten Excellents.
    Conservative — good for safety-critical reporting.

majority
    The most-common label wins (ties broken by worst).

Both labels are stored in the output so render_map.py can choose.

Outputs
-------
  processed_data/road_quality.gpkg   — edge GeoDataFrame (WGS84)
  processed_data/road_quality.csv    — same data without geometry

Usage
-----
    python src/mapping/snap_to_roads.py \
        --predictions processed_data/mapping/predictions.csv

    # Stricter options:
    python src/mapping/snap_to_roads.py \
        --predictions processed_data/mapping/predictions.csv \
        --max-snap-dist 25 \
        --min-obs 5 \
        --drop-invalid
"""

import os
import argparse
import math
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
from tqdm import tqdm
import osmnx as ox

# ── Quality ordering ──────────────────────────────────────────────────────────
# Higher score = better road.  Used for pessimistic (min) and scoring.
QUALITY_SCORE = {
    # 3-class merged model
    'Bad':       1,
    'Good':      3,
    # 5-class full model
    'Poor':      1,
    'Fair':      2,
    'Excellent': 4,
}
SCORE_TO_LABEL = {v: k for k, v in QUALITY_SCORE.items()}

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_coord_col(series: pd.Series) -> pd.Series:
    """Parse lat/lon columns that may be stored as 'N8.5010' strings or floats."""
    def _parse(v):
        if pd.isna(v):
            return float('nan')
        s = str(v).strip()
        if s and s[0] in ('N', 'S', 'E', 'W'):
            sign = -1 if s[0] in ('S', 'W') else 1
            return sign * float(s[1:])
        return float(s)
    return series.apply(_parse)


def bbox_with_buffer(lats, lons, buffer_m=500):
    """
    Return (south, west, north, east) bounding box of the GPS points,
    expanded by buffer_m metres on each side.
    """
    lat_deg = buffer_m / 111_320
    north = lats.max() + lat_deg
    south = lats.min() - lat_deg
    mid_lat = (lats.max() + lats.min()) / 2
    lon_deg = buffer_m / (111_320 * math.cos(math.radians(mid_lat)))
    east  = lons.max() + lon_deg
    west  = lons.min() - lon_deg
    return south, west, north, east


def pessimistic_label(labels):
    """Return the worst quality label from a list."""
    return SCORE_TO_LABEL[min(QUALITY_SCORE[l] for l in labels)]


def majority_label(labels):
    """Return the most common label; break ties by choosing the worst."""
    counts = Counter(labels)
    max_count = max(counts.values())
    candidates = [l for l, c in counts.items() if c == max_count]
    # among tied candidates pick the worst (lowest score)
    return SCORE_TO_LABEL[min(QUALITY_SCORE[l] for l in candidates)]


def quality_score_avg(labels):
    """Mean quality score (1–4 float)."""
    return sum(QUALITY_SCORE[l] for l in labels) / len(labels)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Snap classified GPS points to OSM road edges and aggregate quality labels'
    )
    parser.add_argument('--predictions', type=str,
                        default='processed_data/mapping/predictions.csv',
                        help='CSV produced by classify_dataset.py')
    parser.add_argument('--output-gpkg', type=str,
                        default='processed_data/mapping/road_quality.gpkg',
                        help='Output GeoPackage path (includes geometry)')
    parser.add_argument('--output-csv', type=str,
                        default='processed_data/mapping/road_quality.csv',
                        help='Output CSV path (no geometry column)')
    parser.add_argument('--max-snap-dist', type=float, default=35.0,
                        help='Max distance (metres) to accept a road snap. Default: 35')
    parser.add_argument('--min-obs', type=int, default=3,
                        help='Min observations per edge to include in output. Default: 3')
    parser.add_argument('--conf-threshold', type=float, default=0.0,
                        help='Drop predictions below this confidence before snapping')
    parser.add_argument('--drop-invalid', action='store_true',
                        help='Exclude Invalid-class predictions before snapping')
    parser.add_argument('--network-type', type=str, default='drive',
                        choices=['drive', 'drive_service', 'all'],
                        help='OSM network type to download')
    parser.add_argument('--buffer-m', type=float, default=500.0,
                        help='Buffer (metres) added around GPS bbox when fetching OSM network')
    parser.add_argument('--cache-graph', type=str,
                        default='processed_data/mapping/osm_graph.graphml',
                        help='Save/load the OSM graph from this path to avoid re-downloading')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load predictions ──────────────────────────────────────────────────────
    print(f'Loading predictions from {args.predictions}...')
    df = pd.read_csv(args.predictions)
    print(f'  {len(df):,} rows loaded.')

    # Parse lat/lon (handle both float and 'N8.5010' string formats)
    df['latitude']  = parse_coord_col(df['latitude'])
    df['longitude'] = parse_coord_col(df['longitude'])

    # Drop rows without coordinates
    before = len(df)
    df = df.dropna(subset=['latitude', 'longitude'])
    if len(df) < before:
        print(f'  Dropped {before - len(df):,} rows with missing GPS.')

    # Apply confidence filter
    if args.conf_threshold > 0 and 'confidence' in df.columns:
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
        before = len(df)
        df = df[df['confidence'] >= args.conf_threshold]
        print(f'  Dropped {before - len(df):,} rows below conf={args.conf_threshold}.')

    # Drop Invalid class
    if args.drop_invalid:
        before = len(df)
        df = df[df['predicted_class'] != 'Invalid']
        print(f'  Dropped {before - len(df):,} Invalid-class rows.')

    # Keep only road quality labels (not Invalid)
    valid_labels = set(QUALITY_SCORE.keys())
    before = len(df)
    df = df[df['predicted_class'].isin(valid_labels)]
    if len(df) < before:
        print(f'  Dropped {before - len(df):,} rows with unrecognised class labels.')

    # Drop GPS outliers using IQR to remove OCR misreads near (0, 0)
    before = len(df)
    for col in ('latitude', 'longitude'):
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        iqr = q3 - q1
        df = df[df[col].between(q1 - 3 * iqr, q3 + 3 * iqr)]
    if len(df) < before:
        print(f'  Dropped {before - len(df):,} GPS outlier rows (IQR filter).')

    if df.empty:
        print('No valid predictions to process. Exiting.')
        return

    lats = df['latitude'].values
    lons = df['longitude'].values
    labels_all = df['predicted_class'].values
    print(f'\n  {len(df):,} predictions remain after filtering.')

    # ── Download / load OSM road network ─────────────────────────────────────
    if args.cache_graph and os.path.exists(args.cache_graph):
        print(f'\nLoading cached OSM graph from {args.cache_graph}...')
        G = ox.load_graphml(args.cache_graph)
    else:
        south, west, north, east = bbox_with_buffer(
            pd.Series(lats), pd.Series(lons), buffer_m=args.buffer_m
        )
        print(f'\nDownloading OSM {args.network_type} network ...')
        print(f'  Bbox: N={north:.5f}  S={south:.5f}  E={east:.5f}  W={west:.5f}')
        G = ox.graph_from_bbox(
            bbox=(north, south, east, west),
            network_type=args.network_type,
            simplify=True,
        )
        if args.cache_graph:
            os.makedirs(os.path.dirname(args.cache_graph) or '.', exist_ok=True)
            ox.save_graphml(G, args.cache_graph)
            print(f'  Saved graph to {args.cache_graph}')

    print(f'  Graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges')

    # ── Project graph and GPS points to UTM for metre-accurate distances ──────
    print('\nProjecting graph to UTM...')
    G_proj = ox.project_graph(G)
    edges_proj = ox.graph_to_gdfs(G_proj, nodes=False)
    crs_proj = edges_proj.crs

    # Project GPS points
    transformer = Transformer.from_crs('EPSG:4326', crs_proj, always_xy=True)
    xs_proj, ys_proj = transformer.transform(lons, lats)   # lon → x, lat → y

    # ── Snap GPS points to nearest edge ───────────────────────────────────────
    print(f'Snapping {len(df):,} points to nearest edges (max {args.max_snap_dist}m)...')
    ne, dists = ox.nearest_edges(
        G_proj,
        X=xs_proj,
        Y=ys_proj,
        return_dist=True,
    )

    # ── Aggregate quality labels per edge ─────────────────────────────────────
    # ne is an array of (u, v, key) tuples; dists is an array of distances in metres
    edge_observations = defaultdict(list)   # (u,v,key) -> [label, ...]

    skipped_dist = 0
    for (u, v, k), dist, label in zip(ne, dists, labels_all):
        if dist > args.max_snap_dist:
            skipped_dist += 1
            continue
        edge_observations[(u, v, k)].append(label)

    print(f'  Snapped to {len(edge_observations):,} unique edges.')
    print(f'  Skipped {skipped_dist:,} points farther than {args.max_snap_dist}m from any road.')

    # ── Build result rows ──────────────────────────────────────────────────────
    quality_counts = Counter()
    rows = []
    skipped_min_obs = 0

    for (u, v, k), obs_labels in edge_observations.items():
        if len(obs_labels) < args.min_obs:
            skipped_min_obs += 1
            continue

        pess  = pessimistic_label(obs_labels)
        maj   = majority_label(obs_labels)
        score = quality_score_avg(obs_labels)
        cls_counts = Counter(obs_labels)

        rows.append({
            'u':                   u,
            'v':                   v,
            'key':                 k,
            'quality_pessimistic': pess,
            'quality_majority':    maj,
            'quality_score':       round(score, 3),
            'obs_count':           len(obs_labels),
            'count_excellent':     cls_counts.get('Excellent', 0),
            'count_good':          cls_counts.get('Good',      0),
            'count_bad':           cls_counts.get('Bad',       0),
            'count_fair':          cls_counts.get('Fair',      0),
            'count_poor':          cls_counts.get('Poor',      0),
        })
        quality_counts[pess] += 1

    print(f'  Kept {len(rows):,} edges with >= {args.min_obs} observations.')
    print(f'  Skipped {skipped_min_obs:,} edges below minimum observation threshold.')

    if not rows:
        print('No edges passed all filters. Exiting.')
        return

    result_df = pd.DataFrame(rows)

    # ── Join edge geometries (in WGS84) ───────────────────────────────────────
    print('\nJoining edge geometries...')
    edges_wgs84 = ox.graph_to_gdfs(G, nodes=False).reset_index()

    # edges_wgs84 index is (u, v, key); reset gives columns u, v, key
    edges_wgs84 = edges_wgs84[['u', 'v', 'key', 'name', 'highway', 'length', 'geometry']]

    result_gdf = result_df.merge(edges_wgs84, on=['u', 'v', 'key'], how='left')
    result_gdf = gpd.GeoDataFrame(result_gdf, geometry='geometry', crs='EPSG:4326')

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_gpkg) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv)  or '.', exist_ok=True)

    print(f'\nSaving GeoPackage to {args.output_gpkg}...')
    result_gdf.to_file(args.output_gpkg, driver='GPKG', layer='road_quality')

    print(f'Saving CSV to {args.output_csv}...')
    result_gdf.drop(columns='geometry').to_csv(args.output_csv, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_edges = len(result_gdf)
    print(f'\n{"="*55}')
    print(f'Snapping complete')
    print(f'{"="*55}')
    print(f'  Input predictions       : {len(df):,}')
    print(f'  Snapped (within {args.max_snap_dist}m)     : {len(df) - skipped_dist:,}')
    print(f'  Unique edges with data  : {total_edges:,}')
    print(f'\nRoad quality distribution (pessimistic):')
    for label in ['Poor', 'Fair', 'Good', 'Excellent']:
        n = quality_counts.get(label, 0)
        pct = 100 * n / max(total_edges, 1)
        bar = '█' * int(pct / 2)
        print(f'  {label:<12} {n:>5,}  ({pct:4.1f}%)  {bar}')
    print(f'\nOutputs:')
    print(f'  {args.output_gpkg}')
    print(f'  {args.output_csv}')


if __name__ == '__main__':
    main()
