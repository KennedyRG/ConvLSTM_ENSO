# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:36:03 2022

@author: KRGT
"""
import pylab as plt
import numpy as np  # para tensorflow 2.5 py3.8 funciona con numpy v.1.19.5
import preprocessing_2g_6m_54 as pp
import metric
import ConvLSTM2D
import os.path
os.getcwd()
import sys
from matplotlib import pyplot
from keras.models import load_model
from keras import optimizers
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from contextlib import redirect_stdout
sys.setrecursionlimit(100000000)
 
def CovLSTM2D_model():
    seq = ConvLSTM2D.model()
    print('=' * 10)
    print(seq.summary())
    print('=' * 10)
    return seq
# monthly sst parameters setting
epochs =5000
batch_size = 100
validation_split = 0.1#se usara el 10% de anios de entrenamiento 144semestres*0.1= 160=16 *2semes=30 aprox o 160anios*0.1=16anios*2se=32
##de 168 le quitamos 8 anios apra el test quedando 160 datos*2=320 del cual 144 es para cali y 16 para valida
train_length = 320 #desde 1854 hasta anio dic 2014 (288):1854+144-1=1997 formula para saber el anio 160*.9=144 anios*2=288
len_seq = 320 #
len_frame = 6#12
### PARA EL PRONOSTICO USAMOS LOS ANIOS QUE DESEAMOS
start_seq = 286 #130 #el anio que deseo se calcula: 1997-1854=143*2=286 para :1997-EJ ya que empieza en cero train_X_raw[0,:,:,:,:]
end_seq = 288 #132 solo cambiar el anio (eje 1998) 1998-1854=144*2=288 para :1998-EJ ya que empieza en cero train_X_raw[0,:,:,:,:]
#pronostica 287: 1997-JD y 288:1998-EJ
point_x, point_y = 2, 2
map_height, map_width = 12, 75

fold_name = "modelo_1854_"+str(epochs)+"_epochs_"+str(len_frame)

os.makedirs(fold_name)
# fit model
file_path = fold_name +'/'+fold_name +".h5"
log_file_path = fold_name+'/'+fold_name +".log"
log = open(log_file_path,'w')
# model setting
seq = CovLSTM2D_model()
with redirect_stdout(log):
    seq.summary()
#todo
sst_grid_raw = np.copy(normalized_sst)
# normalization, data for ConvLSTM Model -n ahead -5 dimension
train_X = np.zeros((len_seq, len_frame, map_height, map_width, 1), dtype=np.float64)
train_Y = np.zeros((len_seq, len_frame, map_height, map_width, 1), dtype=np.float64)
sst_grid = np.zeros((len_seq+len_frame, len_frame, map_height, map_width, 1), dtype=np.float64)
for i in range(len_seq): # i = 149
    for k in range(len_frame):
        train_X[i,k,::,::,0] = (train_X_raw[i,k,::,::,0])
        train_Y[i,k,::,::,0] = (train_Y_raw[i,k,::,::,0])
        sst_grid[i,k,::,::,0] = (sst_grid_raw[i,k,::,::,0])
###cambiar los nan por 0
train_X[np.isnan(train_X)] = 0
train_Y[np.isnan(train_Y)] = 0
sst_grid_raw[np.isnan(sst_grid_raw)] = 0
seq.compile(loss="mse", optimizer='adam')#, metrics=["categorical_accuracy"])
####para corregir el tensorflow con nan o masked o cero#######
################################
#train_X= np.ma.masked_invalid(train_X)
#train_Y = np.ma.masked_invalid(train_Y)
###################################
if not os.path.exists(file_path):
    # ConvLSTM Model
    history = seq.fit(train_X[:train_length], train_Y[:train_length],
                      batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    seq.save(file_path)
    pyplot.plot(history.history['loss'], 'c')
    log.write("\n train_loss=========")
    log.write("\n %s" % history.history['loss'])
    #pyplot.plot(history.history['val_loss'],'orange')
    log.write("\n\n\n val_loss=========")
    log.write("\n %s" % history.history['val_loss'])
    pyplot.title('Pérdidas del modelo')
    pyplot.ylabel('Pérdidas')
    pyplot.xlabel('Épocas')
    pyplot.legend(['Entrenamiento'], loc='upper left')
    #pyplot.legend(['Validación'], loc='upper left')
    pyplot.grid(linestyle='dotted', linewidth=0.5)
    #pyplot.legend(['Entrenamiento', 'Validación'], loc='upper left')
    pyplot.savefig(fold_name + '/%i_epoch_loss_train1854_pro.png' % epochs)
    ######segunda grafica#########################
    pyplot.plot(history.history['categorical_accuracy'])
    log.write("\n train_categorical_accuracy=========")
    log.write("\n %s" % history.history['categorical_accuracy'])
    pyplot.plot(history.history['val_categorical_accuracy'])
    log.write("\n\n\n val_categorical_accuracy=========")
    log.write("\n %s" % history.history['val_categorical_accuracy'])
    pyplot.title('Precisión del modelo')
    pyplot.ylabel('Presición')
    pyplot.xlabel('Épocas')
    pyplot.legend(['Entrenamiento', 'Validación'], loc='upper left')
    #pyplot.savefig(fold_name + '/%i_epoch_loss_valid1950_pro.png' % epochs)

else:
    seq = load_model(file_path)
#####ahora vamos a predecir    ###########################################
model_sum_rmse = 0
base_sum_rmse = 0
model_sum_mae = 0
base_sum_mae = 0
model_sum_mape = 0
base_sum_mape = 0

single_point_model_sum_rmse = 0
single_point_base_sum_rmse = 0
###creando matrix para graficar######
ians = 6*(end_seq - start_seq)
sst_pred = np.zeros((ians,map_height, map_width)) # creando array
sst_act = np.zeros((ians,map_height, map_width)) # creando array
#"""############################prediccion ################ 
for k in range(start_seq, end_seq): # k = start_seq
    model_sum_rmse_current = 0
    base_sum_rmse_current = 0
    model_sum_mae_current = 0
    base_sum_mae_current = 0
    model_sum_mape_current = 0
    base_sum_mape_current = 0

    pred_sequence_raw = sst_grid_raw[k][::, ::, ::, ::]
    pred_sequence = sst_grid_raw[k][::, ::, ::, ::]
    act_sequence = sst_grid_raw[k+int(len_frame/6)][::, ::, ::, ::] #ojo con este dato 12 de len_frame

    for j in range(len_frame):
        new_frame = seq.predict(pred_sequence[np.newaxis, ::, ::, ::, ::])
        
        new = new_frame[::, -1, ::, ::, ::]
        pred_sequence = np.concatenate((pred_sequence, new), axis=0)

        baseline_frame = pp.inverse_normalization(pred_sequence_raw[j, ::, ::, 0])
        pred_toplot = pp.inverse_normalization(pred_sequence[-1, ::, ::, 0])
        act_toplot = pp.inverse_normalization(act_sequence[j, ::, ::, 0])

        pred_sequence = pred_sequence[1:len_frame+1, ::, ::, ::]
###############para dos anios consecutivos, si es un anio desactivar####               
    if k < (end_seq-1):
        for ian in range(ians-6):
            sst_pred[ian,::,::] = pred_sequence[ian,::,::,0]
            sst_act[ian,::,::] = act_sequence[ian,::,::,0]
    else:
        for ian in range(6,ians):
            sst_pred[ian,::,::] = pred_sequence[ian-6,::,::,0]
            sst_act[ian,::,::] = act_sequence[ian-6,::,::,0]
#"""
###############para dos anios consecutivos, si es un anio desactivar####       
    if k < (end_seq-1):
        for ian in range(ians-6):
            sst_pred[ian,::,::] = pred_sequence[ian,::,::,0]
            sst_act[ian,::,::] = act_sequence[ian,::,::,0]
    else:
        for ian in range(6,ians):
            sst_pred[ian,::,::] = pred_sequence[ian-6,::,::,0]
            sst_act[ian,::,::] = act_sequence[ian-6,::,::,0]
#"""
###########normalizando la data##########################
to_pred = np.zeros((ians,map_height, map_width))
to_act = np.zeros((ians,map_height, map_width))
    
for i in range(ians):
    to_pred[i,::,::] = pp.inverse_normalization(sst_pred[i, ::, ::])
    to_act[i,::,::] = pp.inverse_normalization(sst_act[i, ::, ::])
    
to_pred[to_pred==13.765629 ] = np.nan###14.252443 para 1950 ##13.765629 
to_act[to_act==13.765629 ] = np.nan
###exportando la data act_pred
np.save('to_pred6m97_98_1854', to_pred) #120 122
np.save('to_real6m97_98_1854', to_act)
    
for i in range(ians):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(321)
    ax.text(1, 3, 'Prediction', fontsize=12)
    pred_toplot = to_pred[i, ::, ::]
    plt.imshow(pred_toplot)
    cbar = plt.colorbar(plt.imshow(pred_toplot), orientation='horizontal')
    cbar.set_label('°C',fontsize=12)

    # Secuencial 12 linea base
    ax = fig.add_subplot(322)
    plt.text(1, 3, 'Ground truth', fontsize=12) # valor verdadero
    act_toplot = to_act[i, ::, ::]
    plt.imshow(act_toplot)
    cbar = plt.colorbar(plt.imshow(act_toplot), orientation='horizontal')
    cbar.set_label('°C',fontsize=12)
#"""###################esto lo hice yo hasta aqui######################### 
#graficando la sst prediccion y base y la diferencia entre ellos
for i in range(len_frame):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(321)
    ax.text(1, 3, 'Prediction', fontsize=12)
    pred_toplot = pp.inverse_normalization(pred_sequence[i, ::, ::, 0])
    plt.imshow(pred_toplot)
    cbar = plt.colorbar(plt.imshow(pred_toplot), orientation='horizontal')
    cbar.set_label('°C',fontsize=12)

    # Secuencial 12 linea base
    ax = fig.add_subplot(322)
    plt.text(1, 3, 'Ground truth', fontsize=12) # valor verdadero
    act_toplot = pp.inverse_normalization(act_sequence[i, ::, ::, 0])
    plt.imshow(act_toplot)
    cbar = plt.colorbar(plt.imshow(act_toplot), orientation='horizontal')
    cbar.set_label('°C',fontsize=12)
####################################################################
###############codigo para guardar el HISTORY#######################
####### save to json: 1ra forma##########################
# convert the history.history dict to a pandas DataFrame:     
import pandas as pd
hist_df = pd.DataFrame(history.history) 

hist_json_file = 'E:\DocumentsD\ENSO\grid\keras_model\modelo_1854_5000_epochs_6\history.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)
##########cargando el archivo json de history    
load_json = open('E:\DocumentsD\ENSO\grid\keras_model\modelo_1854_5000_epochs_6\history.json')    
import json
load_json = json.load(load_json)
history_df = pd.DataFrame(load_json)
print (history_df)
############# or save to csv: 2da forma####################
hist_csv_file = 'E:\DocumentsD\ENSO\grid\keras_model\modelo_1854_5000_epochs_6\history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
##########cargando el archivo csv de history    
import pandas as pd
load_csv = pd.read_csv(r'E:\DocumentsD\ENSO\grid\keras_model\modelo_1854_5000_epochs_6\history.csv')
history_df = pd.DataFrame(load_csv, columns= ['loss','val_loss'])
print (history_df)

############graficando history###########################
import matplotlib.pyplot as plt
a, b = 200, 300 #RANGO DE GRAFICAS segun las epocas
fig = plt.figure(figsize=(8, 6), dpi=300) # por defecto es 8, 6,#### 6, 4 para word
epocas = range(a,b)
plt.plot(epocas, history_df['loss'][a:b], 'c', label='Entrenamiento')
plt.plot(epocas, history_df['val_loss'][a:b], 'orange', label='Validación')
plt.title('Entrenamiento y Validación - Pérdidas del modelo')
pyplot.grid(linestyle='dotted', linewidth=0.5)
plt.xlabel('Épocas')
plt.ylabel('Pérdidas')
plt.legend()
plt.show()
print(history_df['loss'][4900])
print(history_df['val_loss'][4900])
########################################
# Creación del modelo utilizando matrices como en scikitlearn
# ==============================================================================
# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
flafla = train_X.flatten()
flaflay = train_Y.flatten()
x_sm = sm.add_constant(flafla, prepend=True)
modelo1 = sm.OLS(endog=flaflay, exog=flafla)
modelo1 = modelo1.fit()
print(modelo1.summary())
import scipy.stats as stats
r,p=stats.pearsonr(flafla ,flaflay) #para pearson

datos = pd.DataFrame({'equipos': flafla, 'bateos': flaflay})
datos.head(2)
fig, ax = plt.subplots(figsize=(6, 3.84))
datos.plot(
    x    = 'bateos',
    y    = 'runs',
    c    = 'firebrick',
    kind = "scatter",
    ax   = ax
)
ax.set_title('Distribución de bateos y runs');








history.conf_int(alpha=0.05)

predicciones = seq.get_prediction(exog = flafla).summary_frame(alpha=0.05)
predicciones.head(4)

train_X[:train_length], train_Y[:train_length]