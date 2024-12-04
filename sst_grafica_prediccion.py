# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:50:24 2021

@author: KRGT
"""
import pandas as pd
import numpy as np

#graficas de SST predicciones
len_frame = 12
map_height, map_width = 10, 50
#sabemos que nuestro data tiene para este ejem 170*12*10*50*1
#el primero es el anio 170, el segundo es el numero de meses 12, el resto dimensiones de la cuadricula
data_sst = train_X_raw
data_sst.shape
#Ahora, vamos a extraer SST durante el período que nos interesa.
anioI = 112 # anio de inicio (colocar el periodo de analisis) 1851-1850=1
anioF = 170 # anio final (colocar el periodo de analisis) 1852-1850+1=3
# Extraer los meses de analisis
mes = 0 #colocar el mes inicial (ejem Ene=0)
#extraendo 
sst_cal = np.mean(data_sst[anioI:anioF,mes,:,:], axis=0) #promedio de los anios y meses pedidos
sst_cal_1 = pp.inverse_normalization(sst_cal[::, ::,0])
#promedio de todos los anios pedidos por meses
sst_cal_meses = np.mean(data_sst[anioI:anioF,:,:,:], axis=0) #promedio de los anios y meses pedidos
sst_cal_meses_1 = np.zeros((len_frame,int(map_height),int(map_width))) # creando array
for i in range(len_frame):
    sst_cal_meses_1[i,:,:] = pp.inverse_normalization(sst_cal_meses[i,::, ::,0])
# calculo de anomalias por ejem de 97-98
anio_anom = 165 # colocar el año a analizar de la anomalia 2015
#anom_sst = np.zeros((len_frame,int(map_height),int(map_width))) # creando array
anom = np.zeros((len_frame,int(map_height),int(map_width))) # creando array
anom1 = np.zeros((len_frame,int(map_height),int(map_width))) # creando array
for ian in range(len_frame): #ian = 0
    anom[ian,:,:] = data_sst[anio_anom,ian,:,:,0] #promedio de los anios y meses pedidos
    anom1[ian,:,:] = pp.inverse_normalization(anom[ian,:,:])
anom_sst =  anom1[mes,:,:] - sst_cal_1 #matriz de año pedido anom_sst
"""  
anom = data_sst[anio_anom,mes,:,:,0] #promedio de los anios y meses pedidos
anom1 = pp.inverse_normalization(anom[::, ::])
anom_sst = sst_cal_2 - anom1 #matriz de año pedido anom_sst
"""
# data de run_rm_2g_6m_54.py
pred_sequence.shape
sst_cal_rna = pred_sequence
anom2 = np.zeros((len_frame,int(map_height),int(map_width))) # creando array
for ian in range(len_frame): #ian = 0
    anom2[ian,:,:] = pp.inverse_normalization(sst_cal_rna[ian,:,:])
anom_sst_rna = anom2[mes,:,:] - sst_cal_1  #matriz de año pedido anom_sst

#graficando
for i in range(len_frame):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(321)
    ax.text(1, 3, 'Prediction', fontsize=12)
    pred_toplot =  anom2[i,:,:] - sst_cal_meses_1[i,:,:] #matriz de año pedido anom_sst
    plt.imshow(pred_toplot)
    cbar = plt.colorbar(plt.imshow(pred_toplot), orientation='horizontal')
    cbar.set_label('°C',fontsize=12)

    ax = fig.add_subplot(322)
    ax.text(1, 3, 'Real', fontsize=12)
    act_toplot = anom1[i,:,:] - sst_cal_meses_1[i,:,:]
    plt.imshow(act_toplot)
    cbar = plt.colorbar(plt.imshow(act_toplot), orientation='horizontal')
    cbar.set_label('°C',fontsize=12)


