# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:34:59 2022

@author: KRGT
"""
import scipy.io as sio
import numpy as np

train_length = 336# 144
# monthly sst setting
len_year = 336#144#336#168
len_seq = 6
map_height, map_width = 12, 75
MAX = 36.008137#31.017838 para 1950#36.008137
MIN = 13.765629#14.252443 para 1950#13.765629
MEAN = 27.125076135105203#27.329489664082686 para 1950#27.125076135105203

# 0~1 Normalization
def normalization(data):
    normalized_data = np.zeros((map_height, map_width), dtype=np.float)
    for i in range(len(data)):
        for j in range(len(data[0])):
            normalized_data[i][j] = (data[i][j]- MIN)/(MAX - MIN)
    return normalized_data

def inverse_normalization(data):
    inverse_data = np.zeros((map_height, map_width), dtype=np.float)
    for i in range(len(data)):
        for j in range(len(data[0])):
            inverse_data[i][j] = data[i][j]*(MAX - MIN) + MIN
    return inverse_data

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))
#####################################################
####usar para cargar data mensual para nuestro ejemplo
def load_data_convlstm_monthly(train_length):
    # load data
    import os
    os.chdir(r'E:\DocumentsD\ENSO\grid\keras_model')
    os.getcwd()
    with np.load('test2g.npz') as npz:  #desde 1854-01 hasta 2021-12
        sst_data = np.ma.MaskedArray(**npz)
    sst_data1 = sst_data[::,::,::]
    sst_data = np.array(sst_data, dtype=float)
    print("Shape of origin Dataset: ", sst_data.shape)

    # (180 * 360 * 2004) --> (10 * 50 * 2004)
    # the NINO3.4 region (5W~5N, 170W~120W)
    sst_data = sst_data[::,39:51,69:144]#68:144 76#desde ene 1950 sera "1152" ,caso contrario para ene 1854 es "::"
    sst_data2 = sst_data1[::,39:51,69:144]
    mini = sst_data2.min()
    maxi = sst_data2.max()
    media = sst_data2.mean()
    print('=' * 50)
    print("min:", mini, "max:", maxi)
    print("mean:", media)
    # sst min:20.33 / max:31.18

    normalized_sst = np.zeros((len_year,len_seq,map_height,map_width,1), dtype = np.float64)
    for i in range(len_year):
        for k in range(len_seq):
            # Year * 12 + currentMonth
            normalized_sst[i,k,::,::,0] = normalization(sst_data2[i*len_seq+k,::,::])
    train_X_raw = normalized_sst[:train_length]
   
    train_Y_raw = np.zeros((train_length,len_seq,map_height,map_width,1), dtype = np.float64)
    for i in range(train_length):
        for k in range(len_seq):
            if(k != len_seq-1):
                train_Y_raw[i,k,::,::,0] = train_X_raw[i,k+1,::,::,0]
            # Se emite diciembre de cada año y enero del año.
            else:
                if(i != train_length-1):
                    train_Y_raw[i,k,::,::,0] = train_X_raw[i+1,0,::,::,0]
                else:
                # El último mes del año pasado utiliza el marco actual como salida
                    train_Y_raw[i,k,::,::,0] = train_X_raw[i,k,::,::,0]

    print("Whole Shape: ", normalized_sst.shape)
    print("Train_X Shape: ", train_X_raw.shape)
    print("Train_Y Shape: ", train_Y_raw.shape)
    return normalized_sst, train_X_raw, train_Y_raw
normalized_sst, train_X_raw, train_Y_raw = load_data_convlstm_monthly(train_length)
