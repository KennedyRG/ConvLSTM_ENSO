# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 19:22:14 2022

@author: DELL
"""
import numpy as np
import os.path
os.getcwd()
import sys
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt

"""###creando matrix para graficar######usr cuando usas [redicion6m]
anom_pred_total.shape
anom_real.shape

act_toplot1 = np.copy(anom_real) # proviene de prediccion6m
pred_toplot1 = np.copy(anom_pred_total) # proviene de prediccion6m
"""#######USAR CUANDO GUARDASTE LOS NPZ###########################################
import os
os.chdir(r'E:\DocumentsD\ENSO\grid\keras_model')
os.getcwd()

with np.load('pred_toplot_1516ej.npz') as npz:
    to_pred = np.ma.MaskedArray(**npz)
    
with np.load('real_toplot_1516ej.npz') as npz:
    to_real = np.ma.MaskedArray(**npz)
###sst
with np.load('sst_pred_1516ej.npz') as npz:
    sst_pred = np.ma.MaskedArray(**npz)

with np.load('sst_real_1516ej.npz') as npz:
    sst_real = np.ma.MaskedArray(**npz)    

###################################################
import pandas as pd
import numpy as np
import xarray as xr
#from netCDF4 import Dataset as netcdf       # netcdf4-python module
#from netCDF4 import num2date
##abrir ubicacion
import os
os.chdir(r'E:\DocumentsD\ENSO\grid\keras_model')
os.getcwd()

nc_f = 'E:\DocumentsD\pyDoc\sst.mnmean.nc'  # Your filename
a = xr.open_dataset(nc_f)  #para que funcione se tiene que instalar prompt: "pip install netCDF4", instalo la v.1.5.8 funciono
a
data_sst = a.sst
lons=a.lon #esta cada 2
lats=a.lat #esta cada 2
####lon y lan en array
lon = lons.lon.values
lat = lats.lat.values
#recortando area
lon2 = lon[69:144,]#67,143
lat2 = lat[39:51,]#39,51
lon2.shape
lat2.shape

###########################
####################GRAFICA todo en 1: para anomalias sst vs predicion - RECORTADO##############
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec

proj180 = ccrs.PlateCarree(central_longitude=180)
proj = ccrs.PlateCarree()
corre = 6
n = 0 #0 para juni, 6 para ene
fig = plt.figure(figsize=(13, 13), dpi = 300) #figsize=(13, 13) para seis*2 mapas
gs = gridspec.GridSpec(nrows=6, ncols=2)

for i in range(corre):
    #fig = plt.figure(figsize=(20, 30))
    # Use gridspec to help size elements of plot; small top plot and big bottom plot
    #gs = gridspec.GridSpec(nrows=1, ncols=2)
    ax1 = fig.add_subplot(gs[i, 0], projection=proj180)
    # Top plot for geographic reference (makes small map)
    ##op. 1 para centrar  colores
    #clevs1 = np.arange(-6, 6.1, 0.25)# plotear el funcion del rango de analisis de min max -5 5 demas
    clevs1 = np.arange(-6, 6.1, 0.5)# plotear el funcion del rango de analisis de min max -6 6 exacto
    cf1 = ax1.contourf(lon2, lat2, to_pred[i+n,::,::], clevs1, transform=proj, cmap = 'seismic') #afmhot RdBu_r RdYlBu seismic
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=1) #pred_toplot1
    ax1.add_feature(cfeature.BORDERS, linewidths=0.4) # para agregar bordes de paises
    #ax1.stock_img() # para graficar mar y tierra
    os.environ["CARTOPY_USER_BACKGROUNDS"] = "C:/Users/HP/anaconda3/Lib/site-packages/cartopy/BG"
    ax1.background_img(name='etopo', resolution='high') # para topografia
    ax1.set_xticks(np.arange(120,300,20), crs=proj)
    ax1.set_yticks(np.arange(-15,15+5,5), crs=proj180)
    ## seleccionando el solo el area a analisis
    minlon = 138; maxlon = 286 ; minlat = -12; maxlat = 10
    ax1.set_extent([minlon, maxlon, minlat, maxlat],crs=proj)
    ax1.grid(linestyle='dashed', linewidth=0.01, alpha=0.3)
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    #ax1.plt.imshow(cf1)
    cbar = plt.colorbar(cf1, orientation='horizontal', pad=0.25, aspect=50)
    
    # Bottom
    ax2 = fig.add_subplot(gs[i, 1], projection=proj180)
    # Plot of chosen variable averaged over latitude and slightly smoothed
    clevs2 = np.arange(-6, 6.1, 0.5)# plotear el funcion del rango de analisis de min max -6 6 exacto
    cf2 = ax2.contourf(lon2, lat2, to_real[i+n,::,::], clevs2, transform=proj, cmap = 'seismic') #afmhot RdBu_r RdYlBu seismic
    ax2.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=1)
    ax2.add_feature(cfeature.BORDERS, linewidths=0.4) # para agregar bordes de paises
    #ax2.stock_img() # para graficar mar y tierra
    os.environ["CARTOPY_USER_BACKGROUNDS"] = "C:/Users/HP/anaconda3/Lib/site-packages/cartopy/BG"
    ax2.background_img(name='etopo', resolution='high') # para topografia
    ax2.set_xticks(np.arange(120,300,20), crs=proj)
    ax2.set_yticks(np.arange(-15,15+5,5), crs=proj180)
    ## seleccionando el solo el area a analisis
    minlon = 138; maxlon = 286 ; minlat = -12; maxlat = 10
    ax2.set_extent([minlon, maxlon, minlat, maxlat],crs=proj)
    ax2.grid(linestyle='dashed', linewidth=0.01, alpha=0.3)
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    #ax2.plt.imshow(cf2)
    cbar = plt.colorbar(cf2, orientation='horizontal', pad=0.25, aspect=50)
fig.savefig('pred1516ej.pdf', format='pdf')

########################################################################################
####################GRAFICA todo en 1: para TSM sst vs predicion - RECORTADO##############
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec

proj180 = ccrs.PlateCarree(central_longitude=180)
proj = ccrs.PlateCarree()
corre = 6
n = 0 #0 para juni, 6 para ene
fig = plt.figure(figsize=(13, 13), dpi = 300) #figsize=(13, 13) para seis*2 mapas
gs = gridspec.GridSpec(nrows=6, ncols=2)

for i in range(corre):
    #fig = plt.figure(figsize=(20, 30))
    # Use gridspec to help size elements of plot; small top plot and big bottom plot
    #gs = gridspec.GridSpec(nrows=1, ncols=2)
    ax1 = fig.add_subplot(gs[i, 0], projection=proj180)
    # Top plot for geographic reference (makes small map)
    ##op. 1 para centrar  colores
    #clevs1 = np.arange(-6, 6.1, 0.25)# plotear el funcion del rango de analisis de min max -5 5 demas
    clevs1 = np.arange(16, 31.5, 0.5)# plotear el funcion del rango de analisis de min max 17 31 exacto
    cf1 = ax1.contourf(lon2, lat2, sst_pred[i+n,::,::],clevs1 , transform=proj, cmap = 'jet') #afmhot RdBu_r RdYlBu seismic
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=1) #pred_toplot1
    ax1.add_feature(cfeature.BORDERS, linewidths=0.4) # para agregar bordes de paises
    #ax1.stock_img() # para graficar mar y tierra
    os.environ["CARTOPY_USER_BACKGROUNDS"] = "C:/Users/HP/anaconda3/Lib/site-packages/cartopy/BG"
    ax1.background_img(name='etopo', resolution='high') # para topografia
    ax1.set_xticks(np.arange(120,300,20), crs=proj)
    ax1.set_yticks(np.arange(-15,15+5,5), crs=proj180)
    ## seleccionando el solo el area a analisis
    minlon = 138; maxlon = 286 ; minlat = -12; maxlat = 10
    ax1.set_extent([minlon, maxlon, minlat, maxlat],crs=proj)
    ax1.grid(linestyle='dashed', linewidth=0.01, alpha=0.3)
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    #ax1.plt.imshow(cf1)
    cbar = plt.colorbar(cf1, orientation='horizontal', pad=0.25, aspect=50)
    
    # Bottom
    ax2 = fig.add_subplot(gs[i, 1], projection=proj180)
    # Plot of chosen variable averaged over latitude and slightly smoothed
    clevs2 = np.arange(16, 31.5, 0.5)# plotear el funcion del rango de analisis de min max 17 31 exacto
    cf2 = ax2.contourf(lon2, lat2, sst_real[i+n,::,::],clevs2 , transform=proj, cmap = 'jet') #afmhot RdBu_r RdYlBu seismic
    ax2.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=1)
    ax2.add_feature(cfeature.BORDERS, linewidths=0.4) # para agregar bordes de paises
    #ax2.stock_img() # para graficar mar y tierra
    os.environ["CARTOPY_USER_BACKGROUNDS"] = "C:/Users/HP/anaconda3/Lib/site-packages/cartopy/BG"
    ax2.background_img(name='etopo', resolution='high') # para topografia
    ax2.set_xticks(np.arange(120,300,20), crs=proj)
    ax2.set_yticks(np.arange(-15,15+5,5), crs=proj180)
    ## seleccionando el solo el area a analisis
    minlon = 138; maxlon = 286 ; minlat = -12; maxlat = 10
    ax2.set_extent([minlon, maxlon, minlat, maxlat],crs=proj)
    ax2.grid(linestyle='dashed', linewidth=0.01, alpha=0.3)
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    #ax2.plt.imshow(cf2)
    cbar = plt.colorbar(cf2, orientation='horizontal', pad=0.25, aspect=50)
fig.savefig('sst1516ej.pdf', format='pdf')
