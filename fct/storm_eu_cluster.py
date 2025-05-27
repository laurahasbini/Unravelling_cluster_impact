from collections import defaultdict
import math
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import xarray as xr 
import datetime 
import pytz
import csv
import seaborn as sns
from scipy.stats import kde
from scipy.stats import gaussian_kde
from haversine import haversine
import netCDF4
import xskillscore as xs
import geopandas as gpd
from scipy import stats
import rioxarray 
import shapely.geometry as sgeom
from shapely.geometry import Point, box
from shapely.ops import unary_union
from collections import defaultdict

def assign_clusters_allow_multiple(df_track_info, df_tracks, r=700, nb_hours_diff=72, is_mask=False, mask=None):
    # Convert landing dates to datetime and create geometries
    df_track_info = df_track_info.copy()
    df_track_info['storm_landing_date'] = pd.to_datetime(df_track_info['storm_landing_date'])
    gdf_tracks = gpd.GeoDataFrame(
        df_tracks,
        geometry=gpd.points_from_xy(df_tracks.lon, df_tracks.lat),
        crs="EPSG:4326"
    )
    gdf_tracks['buffer'] = gdf_tracks.to_crs(epsg=3395).geometry.buffer(r * 1000).to_crs(epsg=4326)
    spatial_index = gdf_tracks.sindex

    # Initialize clusters: each storm starts with its own unique cluster
    clusters = {storm_id: {cluster_id} for cluster_id, storm_id in enumerate(df_track_info['storm_id'])}
    
    for _, storm_row in df_track_info.iterrows():
        storm_id = storm_row['storm_id']
        storm_date = storm_row['storm_landing_date']
        storm_buffer = gdf_tracks.loc[gdf_tracks['storm_id'] == storm_id, 'buffer'].values
        union_buffer = unary_union(storm_buffer)
        combined_union = union_buffer
        
        #Apply an extra filter over the region
        if is_mask : 
            for geom in mask.geometry :
                union_buffer = union_buffer.intersection(geom)
                if union_buffer.is_empty:
                    break
                
        if not union_buffer.is_empty:
            # Find potential spatial matches
            bounding_box = box(*union_buffer.bounds)
            possible_matches_idx = list(spatial_index.query(bounding_box))
            possible_matches = gdf_tracks.iloc[possible_matches_idx]
            spatial_matches = set()

            # Check for intersection with each buffer and aggregate intersections
            for match_row in possible_matches.itertuples():
                match_id = match_row.storm_id
                if match_id == storm_id:
                    continue
                match_date = df_track_info.loc[df_track_info['storm_id'] == match_id, 'storm_landing_date'].values[0]
                if abs((match_date - storm_date).total_seconds()) <= nb_hours_diff * 3600:    
                    match_buffer = match_row.buffer
                    if combined_union.intersects(match_buffer):
                        combined_union = combined_union.intersection(match_buffer)
                        spatial_matches.add(match_id)
        
            # Check temporal condition and link clusters
            for match_id in spatial_matches:
                if match_id == storm_id:
                    continue
                clusters[storm_id].update(clusters[storm_id])
                clusters[match_id].update(clusters[storm_id]) 
    
    # Assign back to the DataFrame
    df_track_info.loc[:,'clust_ids'] = df_track_info['storm_id'].apply(lambda x: list(clusters[x]))
    
    return df_track_info

def filter_clusters(df_track_info):
    """
    Filters the clusters to:
    1. Remove clusters that are subsets of others.
    2. Exclude clusters with only one storm.
    3. Remove duplicate (identical) clusters.
    4. Exclude storms not part of any cluster.
    """
    # Step 1: Group clusters and their storms
    cluster_to_storms = defaultdict(set)
    for storm_id, cluster_ids in zip(df_track_info['storm_id'], df_track_info['clust_ids']):
        for clust_id in cluster_ids:
            cluster_to_storms[clust_id].add(storm_id)

    # Step 2: Remove subset clusters
    cluster_ids = list(cluster_to_storms.keys())
    for i, clust_a in enumerate(cluster_ids):
        for j, clust_b in enumerate(cluster_ids):
            if i != j and cluster_to_storms.get(clust_a, set()) < cluster_to_storms.get(clust_b, set()):
                # If clust_a is a subset of clust_b, mark clust_a for removal
                cluster_to_storms.pop(clust_a, None)

    # Step 3: Remove clusters with only one storm
    cluster_to_storms = {k: v for k, v in cluster_to_storms.items() if len(v) > 1}

    # Step 4: Remove duplicate (identical) clusters
    unique_clusters = {}
    for clust_id, storms in cluster_to_storms.items():
        storms_tuple = tuple(sorted(storms))  # Sort storms to detect duplicates
        if storms_tuple not in unique_clusters.values():
            unique_clusters[clust_id] = storms_tuple

    cluster_to_storms = {k: set(v) for k, v in unique_clusters.items()}

    # Step 5: Reassign valid clusters to storms
    valid_clusters = set(cluster_to_storms.keys())
    storm_to_clusters = defaultdict(set)
    for clust_id, storms in cluster_to_storms.items():
        for storm_id in storms:
            storm_to_clusters[storm_id].add(clust_id)

    # Step 6: Filter storms that are not part of any valid cluster
    df_track_info = df_track_info[df_track_info['storm_id'].isin(storm_to_clusters.keys())].copy()

    # Step 7: Assign back the filtered clusters
    df_track_info['clust_ids'] = df_track_info['storm_id'].apply(lambda x: sorted(list(storm_to_clusters[x])))

    return df_track_info

def filter_clusters_explode(df_track_info):
    """
    Filters the clusters to:
    1. Remove clusters that are subsets of others.
    2. Exclude storms not part of any cluster.
    3. Explode the clusters so each storm-cluster pair is a separate row.
    """
    from itertools import chain
    # Step 1: Filter subset clusters and storms
    df_filtered = filter_clusters(df_track_info)

    # Step 2: Explode the clust_ids column
    df_exploded = (
        df_filtered
        .explode('clust_ids')  # Explode clust_ids into separate rows
        .rename(columns={'clust_ids': 'clust_id'})  # Rename column to clust_id for clarity
        .reset_index(drop=True)  # Reset index for clean output
    )
    return df_exploded

def filter_clusters_exploded(df_track_info):
    """
    Filters the clusters to:
    1. Remove clusters that are subsets of others.
    2. Exclude clusters with only one storm.
    3. Remove duplicate (identical) clusters.
    4. Exclude storms not part of any cluster.
    
    Assumes clust_id in df_track_info is an integer, not a list.
    """
    from collections import defaultdict

    # Step 1: Group clusters and their storms
    cluster_to_storms = defaultdict(set)
    for storm_id, clust_id in zip(df_track_info['storm_id'], df_track_info['clust_id']):
        cluster_to_storms[clust_id].add(storm_id)

    # Step 2: Remove subset clusters
    cluster_ids = list(cluster_to_storms.keys())
    for i, clust_a in enumerate(cluster_ids):
        for j, clust_b in enumerate(cluster_ids):
            if i != j and cluster_to_storms.get(clust_a, set()) < cluster_to_storms.get(clust_b, set()):
                # If clust_a is a subset of clust_b, mark clust_a for removal
                cluster_to_storms.pop(clust_a, None)
    
    # Step 3: Remove clusters with only one storm
    cluster_to_storms = {k: v for k, v in cluster_to_storms.items() if len(v) > 1}
    
    # Step 4: Remove duplicate (identical) clusters
    unique_clusters = {}
    for clust_id, storms in cluster_to_storms.items():
        storms_tuple = tuple(sorted(storms))  # Sort storms to detect duplicates
        if storms_tuple not in unique_clusters.values():
            unique_clusters[clust_id] = storms_tuple

    cluster_to_storms = {k: set(v) for k, v in unique_clusters.items()}
    
    # Step 5: Reassign valid clusters to storms
    valid_clusters = set(cluster_to_storms.keys())
    storm_to_clusters = defaultdict(set)
    for clust_id, storms in cluster_to_storms.items():
        for storm_id in storms:
            storm_to_clusters[storm_id].add(clust_id)

    # Step 6: Filter storms that are not part of any valid cluster
    df_track_info = df_track_info[df_track_info['storm_id'].isin(storm_to_clusters.keys())].copy()

    # Step 7: Assign back the filtered clusters (single clust_id per row)
    df_track_info['clust_ids'] = df_track_info['storm_id'].apply(lambda x: list(storm_to_clusters.get(x, set())))#.apply(lambda x: next(iter(storm_to_clusters[x]), None))
    df_track_info = df_track_info.drop(["clust_id"], axis=1)
    
    # Step 8 : Explode back
    df_track_info = (
        df_track_info
        .explode('clust_ids')  # Explode clust_ids into separate rows
        .rename(columns={'clust_ids': 'clust_id'})  # Rename column to clust_id for clarity
        .reset_index(drop=True)  # Reset index for clean output
    )
    
    # Step 9: Remove duplicate rows
    df_track_info = df_track_info.drop_duplicates()
    return df_track_info
