import numpy as np
import pandas as pd
import xarray as xr
import datetime
import rioxarray 
import pytz

from rechunker import rechunk
from rasterio import features
from affine import Affine

import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import Point
from scipy.signal import find_peaks

import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from paths import *

#FOR PLOTS
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cf
proj = ccrs.PlateCarree()

#################### Spatial grouping at ERA5 level #########################
from scipy.interpolate import griddata
def convert_era5_resolution(sinclim, latitude_name='num_lat', longitude_name='num_lon') : 
    """
    """
    grid_lat = np.arange(40, 60.25, 0.25)
    grid_lon = np.arange(-10, 15.25, 0.25)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    #Interpolate 
    interpolated_lat = griddata(
        points=(sinclim['num_lon'], sinclim['num_lat']),
        values=sinclim['num_lat'],
        xi=(grid_lon_mesh, grid_lat_mesh),
        method='linear'
    )
    interpolated_lon = griddata(
        points=(sinclim['num_lon'], sinclim['num_lat']),
        values=sinclim['num_lon'],
        xi=(grid_lon_mesh, grid_lat_mesh),
        method='linear'
    )
    interpolated_lat_flat = interpolated_lat.flatten()
    interpolated_lon_flat = interpolated_lon.flatten()

    def add_era5_lat(row) : 
        try : 
            return(grid_lat[np.abs(grid_lat - row.num_lat).argmin()])
        except (KeyError, ValueError, AttributeError):
            return np.nan

    def add_era5_lon(row) : 
        try : 
            return(grid_lon[np.abs(grid_lon - row.num_lon).argmin()])
        except (KeyError, ValueError, AttributeError):
            return np.nan

    sinclim['latitude']  = sinclim.apply(add_era5_lat, axis=1)
    sinclim['longitude'] = sinclim.apply(add_era5_lon, axis=1)
    sinclim['region_era5']    = sinclim['longitude'].astype(str) + "_" + sinclim['latitude'].astype(str)
    
    #Add into the new coordinates
    sinclim_grp = sinclim.groupby(['region_era5', 'storm_id']).agg(nb_claims = ('num_chg_brut_cst', 'count'), 
                                                               longitude = ('longitude', 'first'),
                                                               latitude = ('latitude', 'first'),
                                                               mean_chg_brut_cst = ('num_chg_brut_cst', 'sum'), 
                                                               wgust = ('wgust_max', "first")
                                                              )
    sinclim_grp = sinclim_grp.reset_index()
    return sinclim_grp

#################### PLOTING FUNCTIONS #########################

letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)','(k)','(l)']

def weighted_hist2d(x, y, weights, bins):
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights)
    return H, xedges, yedges

def plot_nbclaims(sinclim, days_list, save=False, path_save_fig=PATH_FIGURE):        
    lat_range = [40, 55]
    lon_range = [-5, 15]
    lat_bins_05 = int((lat_range[1] - lat_range[0]) / 0.5)
    lon_bins_05 = int((lon_range[1] - lon_range[0]) / 0.5)
    
    ncol = len(days_list)
    fig, ax = plt.subplots(1, ncol, figsize=(5*ncol, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    for j in range(ncol): 
        ax[j]._autoscaleXon = False
        ax[j]._autoscaleYon = False

    vmin, vmax = 5, 50
    ticks_wgust = np.arange(4, 52, 2)
    cmap_wgust = "gist_ncar"

    contour_indivs = []  # Store contour objects for second-row plots

    for n_col, day in enumerate(days_list):
        cost_stormi = sinclim.loc[sinclim.dat_sin == day]
        H, xedges, yedges = weighted_hist2d(cost_stormi['num_lon'], cost_stormi['num_lat'], np.ones(len(cost_stormi)), bins=[lon_bins_05, lat_bins_05])
        Z = H.T
        xgrid = (xedges[:-1] + xedges[1:]) / 2
        ygrid = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(xgrid, ygrid)

        bounds_storm = [1, 2, 5, 10, 50, 100, 200, 500, 2000, 6000]
        norm_storm = mcolors.BoundaryNorm(bounds_storm, ncolors=256)

        ax[n_col].set_extent([-5, 10, 40, 55], crs=ccrs.PlateCarree())
        contour_indiv = ax[n_col].contourf(X, Y, Z, levels=bounds_storm, cmap="YlOrRd", norm=norm_storm, transform=ccrs.PlateCarree())
        contour_indivs.append(contour_indiv)
        ax[n_col].coastlines(alpha=0.4)
        ax[n_col].add_feature(cf.BORDERS, linestyle='-', alpha=0.4)
        ax[n_col].set_title(f"{day}\n{str(len(cost_stormi))} Claims", fontsize=14)

    # Add a single colorbar for the second row
    cbar_ax2 = fig.add_axes([0.15, 0.07, 0.7, 0.05])  # Adjust position for second-row colorbar
    cbar2 = fig.colorbar(contour_indivs[0], cax=cbar_ax2, orientation='horizontal')
    cbar2.set_ticks(bounds_storm)
    cbar2.ax.tick_params(labelsize=14)
    cbar2.set_label('Number of claims', fontsize=16)    
    plt.subplots_adjust(hspace=0.5, wspace=0.4)  # Increase vertical spacing
    
    if save : 
        fig.savefig(path_save_fig+"Claims_count_"+days_list[0]+"-"+days_list[-1]+".png",
                    transparent=True, bbox_inches='tight', dpi=400)
        fig.savefig(path_save_fig+"Claims_count"+days_list[0]+"-"+days_list[-1]+".svg", 
                    format="svg", transparent=True, bbox_inches='tight', dpi=400)
    return fig

def plot_storm_wind_losses(df_storm, sinclim_storm_grp, sinclim, stormi, path_footprints, save=False, path_save_fig="/home/user/lhasbini/", method="d-3_d+3_unique-wgust_min50_priestley_ALL_1979-2024WIN_r1300"):        
    lat_range = [40, 55]
    lon_range = [-5, 15]
    lat_bins_05 = int((lat_range[1] - lat_range[0]) / 0.5)
    lon_bins_05 = int((lon_range[1] - lon_range[0]) / 0.5)
    
    fig, ax = plt.subplots(2, 1, figsize=(5, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    for i in range(2):
        ax[i]._autoscaleXon = False
        ax[i]._autoscaleYon = False

    vmin, vmax = 5, 50
    ticks_wgust = np.arange(4, 52, 2)
    cmap_wgust = "gist_ncar"#"gist_ncar"#"inferno_r"

    contour_indivs = []  # Store contour objects for second-row plots

    footprint_stormi = xr.open_mfdataset(path_footprints + stormi + "_max_r1300.nc")
#     data = footprint_stormi.max_wind_gust.sel(expver=1).isel(time=0).sel(longitude=slice(-5, 10), latitude=slice(40, 55))
    data = footprint_stormi.max_wind_gust.isel(time=0).sel(longitude=slice(-5, 10), latitude=slice(40, 55))

    ax[0].set_extent([-5, 10, 40, 55], crs=ccrs.PlateCarree())
    plot = ax[0].contourf(data.longitude, data.latitude, data, ticks_wgust, cmap=cmap_wgust, transform=ccrs.PlateCarree())
    stormi_parts = stormi.split('_')
    stormi_formated = '_'.join([stormi_parts[0]] + ["{0:.2f}".format(float(x)) for x in stormi_parts[1:]])
    ax[0].set_title(f"{stormi_formated}", fontsize=13)
    ax[0].add_feature(cf.BORDERS, linestyle='--', edgecolor='black', linewidth=0.5)
    ax[0].add_feature(cf.COASTLINE, edgecolor='black', linewidth=0.5)

    storm_plot = df_storm.loc[df_storm.storm_id == stormi]
    ax[0].plot(storm_plot['lon'], storm_plot['lat'], '-o', alpha=0.8, color="black", linewidth=4, label=stormi)

    cost_stormi = sinclim.loc[sinclim.storm_id == stormi]
    H, xedges, yedges = weighted_hist2d(cost_stormi['num_lon'], cost_stormi['num_lat'], cost_stormi['num_chg_brut_cst'], bins=[lon_bins_05, lat_bins_05])
    Z = H.T
    xgrid = (xedges[:-1] + xedges[1:]) / 2
    ygrid = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xgrid, ygrid)

    bounds_storm = [0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    norm_storm = mcolors.BoundaryNorm(bounds_storm, ncolors=256)

    ax[1].set_extent([-5, 10, 40, 55], crs=ccrs.PlateCarree())
    contour_indiv = ax[1].contourf(X, Y, Z/1e6, levels=bounds_storm, cmap="YlOrRd", norm=norm_storm, transform=ccrs.PlateCarree())
    contour_indivs.append(contour_indiv)
    ax[1].coastlines(alpha=0.4)
    ax[1].add_feature(cf.BORDERS, linestyle='-', alpha=0.4)
    cost_title = cost_stormi.num_chg_brut_cst.sum()/1e6
    ax[1].set_title("{0:.2f} m€".format(float(cost_title)), fontsize=13)

    # Add a single colorbar for the second row
    cbar_ax2 = fig.add_axes([0.15, 0.07, 0.7, 0.02])  # Adjust position for second-row colorbar
    cbar2 = fig.colorbar(contour_indivs[0], cax=cbar_ax2, orientation='horizontal', label='Losses [m€cst 2023]')
    cbar2.set_ticks(bounds_storm)
    cbar2.ax.tick_params(labelsize=14)
    cbar2.set_label('Losses [m€cst 2023]', fontsize=16)

    # Add a colorbar for the first row
    cbar_ax1 = fig.add_axes([0.15, 0.52, 0.7, 0.02])  # Adjust for better positioning
    cbar1 = fig.colorbar(plot, cax=cbar_ax1, orientation='horizontal')
    cbar1.ax.tick_params(labelsize=14)
    cbar1.set_label('Max Wind Gust [m/s]', fontsize=16)
    
    plt.subplots_adjust(hspace=0.5, wspace=0.4)  # Increase vertical spacing
    if save : 
        fig.savefig(path_save_fig+"Losses_storm_"+str(stormi)+"_"+method+".png", transparent=True, bbox_inches='tight', dpi=400)
        fig.savefig(path_save_fig+"Losses_storm_"+str(stormi)+"_"+method+".svg", format="svg", transparent=True, bbox_inches='tight', dpi=400)
    return fig

def plot_storm_wind_nbclaims(df_storm, sinclim_storm_grp, sinclim, stormi, path_footprints, save=False, path_save_fig="/home/user/lhasbini/", method="d-3_d+3_unique-wgust_min50_priestley_ALL_1979-2024WIN_r1300"):        
    lat_range = [40, 55]
    lon_range = [-5, 15]
    lat_bins_05 = int((lat_range[1] - lat_range[0]) / 0.5)
    lon_bins_05 = int((lon_range[1] - lon_range[0]) / 0.5)
    
    fig, ax = plt.subplots(2, 1, figsize=(5, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    for i in range(2):
        ax[i]._autoscaleXon = False
        ax[i]._autoscaleYon = False

    vmin, vmax = 5, 50
    ticks_wgust = np.arange(4, 52, 2)
    cmap_wgust = "gist_ncar"#"gist_ncar"#"inferno_r"

    contour_indivs = []  # Store contour objects for second-row plots

    footprint_stormi = xr.open_mfdataset(path_footprints + stormi + "_max_r1300.nc")
#     data = footprint_stormi.max_wind_gust.sel(expver=1).isel(time=0).sel(longitude=slice(-5, 10), latitude=slice(40, 55))
    data = footprint_stormi.max_wind_gust.isel(time=0).sel(longitude=slice(-5, 10), latitude=slice(40, 55))

    ax[0].set_extent([-5, 10, 40, 55], crs=ccrs.PlateCarree())
    plot = ax[0].contourf(data.longitude, data.latitude, data, ticks_wgust, cmap=cmap_wgust, transform=ccrs.PlateCarree())
    stormi_parts = stormi.split('_')
    stormi_formated = '_'.join([stormi_parts[0]] + ["{0:.2f}".format(float(x)) for x in stormi_parts[1:]])
    ax[0].set_title(f"{stormi_formated}", fontsize=13)
    ax[0].add_feature(cf.BORDERS, linestyle='--', edgecolor='black', linewidth=0.5)
    ax[0].add_feature(cf.COASTLINE, edgecolor='black', linewidth=0.5)

    storm_plot = df_storm.loc[df_storm.storm_id == stormi]
    ax[0].plot(storm_plot['lon'], storm_plot['lat'], '-o', alpha=0.8, color="black", linewidth=4, label=stormi)

    
    cost_stormi = sinclim.loc[sinclim.storm_id == stormi]
    H, xedges, yedges = weighted_hist2d(cost_stormi['num_lon'], cost_stormi['num_lat'], np.ones(len(cost_stormi)), bins=[lon_bins_05, lat_bins_05])
    Z = H.T
    xgrid = (xedges[:-1] + xedges[1:]) / 2
    ygrid = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xgrid, ygrid)

    bounds_storm = [1, 2, 5, 10, 50, 100, 200, 500, 2000]
    norm_storm = mcolors.BoundaryNorm(bounds_storm, ncolors=256)

    ax[1].set_extent([-5, 10, 40, 55], crs=ccrs.PlateCarree())
    contour_indiv = ax[1].contourf(X, Y, Z, levels=bounds_storm, cmap="YlOrRd", norm=norm_storm, transform=ccrs.PlateCarree())
    contour_indivs.append(contour_indiv)
    ax[1].coastlines(alpha=0.4)
    ax[1].add_feature(cf.BORDERS, linestyle='-', alpha=0.4)
    ax[1].set_title(f"{str(len(cost_stormi))} Claims", fontsize=13)
    
    # Add a single colorbar for the second row
    cbar_ax2 = fig.add_axes([0.15, 0.07, 0.7, 0.02])  # Adjust position for second-row colorbar
    cbar2 = fig.colorbar(contour_indivs[0], cax=cbar_ax2, orientation='horizontal', label='Losses [m€cst 2023]')
    cbar2.set_ticks(bounds_storm)
    cbar2.ax.tick_params(labelsize=14)
    cbar2.set_label('Number of claims', fontsize=16)

    # Add a colorbar for the first row
    cbar_ax1 = fig.add_axes([0.15, 0.52, 0.7, 0.02])  # Adjust for better positioning
    cbar1 = fig.colorbar(plot, cax=cbar_ax1, orientation='horizontal')
    cbar1.ax.tick_params(labelsize=14)
    cbar1.set_label('Max Wind Gust [m/s]', fontsize=16)
    
    plt.subplots_adjust(hspace=0.5, wspace=0.4)  # Increase vertical spacing
    if save : 
        fig.savefig(path_save_fig+"Claims_storm_"+str(stormi)+"_"+method+".png", transparent=True, bbox_inches='tight', dpi=400)
        fig.savefig(path_save_fig+"Claims_storm_"+str(stormi)+"_"+method+".svg", format="svg", transparent=True, bbox_inches='tight', dpi=400)
    return fig


def plot_cluster_wind_losses(df_storm, sinclim_storm_grp, sinclim, clust_nb, path_footprints, save=False, path_save_fig="/home/user/lhasbini/", method="d-3_d+3_unique-wgust_min50_priestley_ALL_1979-2024WIN_r1300"):
    sinclim_clust_plot = sinclim_storm_grp.loc[sinclim_storm_grp.clust_id == clust_nb]
        
    lat_range = [40, 55]
    lon_range = [-5, 15]
    lat_bins_05 = int((lat_range[1] - lat_range[0]) / 0.5)
    lon_bins_05 = int((lon_range[1] - lon_range[0]) / 0.5)
    
    ncol_tot = len(sinclim_clust_plot)
    fig, ax = plt.subplots(2, ncol_tot, figsize=(5*ncol_tot, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    for i in range(2):
        for j in range(ncol_tot): 
            ax[i, j]._autoscaleXon = False
            ax[i, j]._autoscaleYon = False

    vmin, vmax = 5, 50
    ticks_wgust = np.arange(4, 52, 2)
    cmap_wgust = "gist_ncar"#"gist_ncar"#"inferno_r"

    contour_indivs = []  # Store contour objects for second-row plots

    for n_col, stormi in enumerate(sinclim_clust_plot.storm_id):
        footprint_stormi = xr.open_mfdataset(path_footprints + stormi + "_max_r1300.nc")
        data = footprint_stormi.max_wind_gust.isel(time=0).sel(longitude=slice(-5, 10), latitude=slice(40, 55))
#         data = footprint_stormi.max_wind_gust.sel(expver=1).isel(time=0).sel(longitude=slice(-5, 10), latitude=slice(40, 55))

        ax[0, n_col].set_extent([-5, 10, 40, 55], crs=ccrs.PlateCarree())
        plot = ax[0, n_col].contourf(data.longitude, data.latitude, data, ticks_wgust, cmap=cmap_wgust, transform=ccrs.PlateCarree())
        stormi_parts = stormi.split('_')
        stormi_formated = '_'.join([stormi_parts[0]] + ["{0:.2f}".format(float(x)) for x in stormi_parts[1:]])
        ax[0, n_col].set_title(f"{stormi_formated}", fontsize=13)
        ax[0, n_col].add_feature(cf.BORDERS, linestyle='--', edgecolor='black', linewidth=0.5)
        ax[0, n_col].add_feature(cf.COASTLINE, edgecolor='black', linewidth=0.5)

        storm_plot = df_storm.loc[df_storm.storm_id == stormi]
        ax[0, n_col].plot(storm_plot['lon'], storm_plot['lat'], '-o', alpha=0.8, color="black", linewidth=4, label=stormi)
        ax[0, n_col].text(0.5, 0.95, letters[n_col], transform=ax[0, n_col].transAxes, fontsize=20, va='center', ha='center')

        cost_stormi = sinclim.loc[sinclim.storm_id == stormi]
        H, xedges, yedges = weighted_hist2d(cost_stormi['num_lon'], cost_stormi['num_lat'], cost_stormi['num_chg_brut_cst'], bins=[lon_bins_05, lat_bins_05])
        Z = H.T
        xgrid = (xedges[:-1] + xedges[1:]) / 2
        ygrid = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(xgrid, ygrid)

        bounds_storm = [0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
#         bounds_storm = [0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]*1e-3
        bounds_storm = [i*1e-3 for i in bounds_storm]
        norm_storm = mcolors.BoundaryNorm(bounds_storm, ncolors=256)

        ax[1, n_col].set_extent([-5, 10, 40, 55], crs=ccrs.PlateCarree())
        contour_indiv = ax[1, n_col].contourf(X, Y, Z/1e6, levels=bounds_storm, cmap="YlOrRd", norm=norm_storm, transform=ccrs.PlateCarree())
        contour_indivs.append(contour_indiv)
        ax[1, n_col].coastlines(alpha=0.4)
        ax[1, n_col].add_feature(cf.BORDERS, linestyle='-', alpha=0.4)
#         "{0:.2f}".format(float(x))
        cost_title = cost_stormi.num_chg_brut_cst.sum()/1e6
        ax[1, n_col].set_title("{0:.2f} m€".format(float(cost_title)), fontsize=13)
        ax[1, n_col].text(0.5, 0.95, letters[n_col+ncol_tot], transform=ax[1, n_col].transAxes, fontsize=20, va='center', ha='center')
    # Add a single colorbar for the second row
    cbar_ax2 = fig.add_axes([0.15, 0.07, 0.7, 0.02])  # Adjust position for second-row colorbar
    cbar2 = fig.colorbar(contour_indivs[0], cax=cbar_ax2, orientation='horizontal', label='Losses [m€cst 2023]')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    cbar2.set_ticks(bounds_storm)
    cbar2.ax.xaxis.set_major_formatter(formatter)
    cbar2.ax.tick_params(labelsize=14)
    cbar2.set_label('Losses [m€cst 2015]', fontsize=16)

    # Add a colorbar for the first row
    cbar_ax1 = fig.add_axes([0.15, 0.52, 0.7, 0.02])  # Adjust for better positioning
    cbar1 = fig.colorbar(plot, cax=cbar_ax1, orientation='horizontal')
    cbar1.ax.tick_params(labelsize=14)
    cbar1.set_label('Max Wind Gust [m/s]', fontsize=16)
    
    plt.subplots_adjust(hspace=0.5, wspace=0.4)  # Increase vertical spacing
    if save : 
        fig.savefig(path_save_fig+"Losses_clust_"+str(clust_nb)+"_"+method+".png", transparent=True, bbox_inches='tight', dpi=400)
        fig.savefig(path_save_fig+"Losses_clust_"+str(clust_nb)+"_"+method+".svg", format="svg", transparent=True, bbox_inches='tight', dpi=400)
    return fig

def plot_cluster_wind_nbclaims(df_storm, sinclim_storm_grp, sinclim, clust_nb, path_footprints, save=False, path_save_fig="/home/user/lhasbini/", method="d-3_d+3_unique-wgust_min50_priestley_ALL_1979-2024WIN_r1300"):
    sinclim_clust_plot = sinclim_storm_grp.loc[sinclim_storm_grp.clust_id == clust_nb]
        
    lat_range = [40, 55]
    lon_range = [-5, 15]
    lat_bins_05 = int((lat_range[1] - lat_range[0]) / 0.5)
    lon_bins_05 = int((lon_range[1] - lon_range[0]) / 0.5)
    
    ncol_tot = len(sinclim_clust_plot)
    fig, ax = plt.subplots(2, ncol_tot, figsize=(5*ncol_tot, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    for i in range(2):
        for j in range(ncol_tot): 
            ax[i, j]._autoscaleXon = False
            ax[i, j]._autoscaleYon = False

    vmin, vmax = 5, 50
    ticks_wgust = np.arange(4, 52, 2)
    cmap_wgust = "gist_ncar"

    contour_indivs = []  # Store contour objects for second-row plots

    for n_col, stormi in enumerate(sinclim_clust_plot.storm_id):
        footprint_stormi = xr.open_mfdataset(path_footprints + stormi + "_max_r1300.nc")
        data = footprint_stormi.max_wind_gust.isel(time=0).sel(longitude=slice(-5, 10), latitude=slice(40, 55))
#         data = footprint_stormi.max_wind_gust.sel(expver=1).isel(time=0).sel(longitude=slice(-5, 10), latitude=slice(40, 55))

        ax[0, n_col].set_extent([-5, 10, 40, 55], crs=ccrs.PlateCarree())
        plot = ax[0, n_col].contourf(data.longitude, data.latitude, data, ticks_wgust, cmap=cmap_wgust, transform=ccrs.PlateCarree())
        stormi_parts = stormi.split('_')
        stormi_formated = '_'.join([stormi_parts[0]] + ["{0:.2f}".format(float(x)) for x in stormi_parts[1:]])
        ax[0, n_col].set_title(f"{stormi_formated}", fontsize=13)
        ax[0, n_col].add_feature(cf.BORDERS, linestyle='--', edgecolor='black', linewidth=0.5)
        ax[0, n_col].add_feature(cf.COASTLINE, edgecolor='black', linewidth=0.5)

        storm_plot = df_storm.loc[df_storm.storm_id == stormi]
        ax[0, n_col].plot(storm_plot['lon'], storm_plot['lat'], '-o', alpha=0.8, color="black", linewidth=4, label=stormi)
        ax[0, n_col].text(0.5, 0.95, letters[n_col], transform=ax[0, n_col].transAxes, fontsize=20, va='center', ha='center')

        cost_stormi = sinclim.loc[sinclim.storm_id == stormi]
        H, xedges, yedges = weighted_hist2d(cost_stormi['num_lon'], cost_stormi['num_lat'], np.ones(len(cost_stormi)), bins=[lon_bins_05, lat_bins_05])
        Z = H.T
        xgrid = (xedges[:-1] + xedges[1:]) / 2
        ygrid = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(xgrid, ygrid)

        bounds_storm = [1, 2, 5, 10, 50, 100, 200, 500, 1000, 5000]
        norm_storm = mcolors.BoundaryNorm(bounds_storm, ncolors=256)

        ax[1, n_col].set_extent([-5, 10, 40, 55], crs=ccrs.PlateCarree())
        contour_indiv = ax[1, n_col].contourf(X, Y, Z, levels=bounds_storm, cmap="YlOrRd", norm=norm_storm, transform=ccrs.PlateCarree())
        contour_indivs.append(contour_indiv)
        ax[1, n_col].coastlines(alpha=0.4)
        ax[1, n_col].add_feature(cf.BORDERS, linestyle='-', alpha=0.4)
        ax[1, n_col].set_title(f"{str(len(cost_stormi))} Claims", fontsize=13)
        ax[1, n_col].text(0.5, 0.95, letters[n_col+ncol_tot], transform=ax[1, n_col].transAxes, fontsize=20, va='center', ha='center')

    # Add a single colorbar for the second row
    cbar_ax2 = fig.add_axes([0.15, 0.07, 0.7, 0.02])  # Adjust position for second-row colorbar
    cbar2 = fig.colorbar(contour_indivs[0], cax=cbar_ax2, orientation='horizontal', label='Losses [m€cst 2023]')
    cbar2.set_ticks(bounds_storm)
    cbar2.ax.tick_params(labelsize=14)
    cbar2.set_label('Number of claims', fontsize=16)

    # Add a colorbar for the first row
    cbar_ax1 = fig.add_axes([0.15, 0.52, 0.7, 0.02])  # Adjust for better positioning
    cbar1 = fig.colorbar(plot, cax=cbar_ax1, orientation='horizontal')
    cbar1.ax.tick_params(labelsize=14)
    cbar1.set_label('Max Wind Gust [m/s]', fontsize=16)
    
    plt.subplots_adjust(hspace=0.5, wspace=0.4)  # Increase vertical spacing
    
    if save : 
        fig.savefig(path_save_fig+"Claims_clust_"+str(clust_nb)+"_"+method+".png", transparent=True, bbox_inches='tight', dpi=400)
        fig.savefig(path_save_fig+"Claims_clust_"+str(clust_nb)+"_"+method+".svg", format="svg", transparent=True, bbox_inches='tight', dpi=400)
    return fig