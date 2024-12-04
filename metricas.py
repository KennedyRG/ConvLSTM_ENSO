# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 11:26:25 2022

@author: KRGT
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
"""
#"""#######USAR CUANDO GUARDASTE LOS NPZ###########################################
import os
os.chdir(r'E:\DocumentsD\ENSO\grid\keras_model')
os.getcwd()

with np.load('pred_toplot_1516jj.npz') as npz:
    to_pred = np.ma.MaskedArray(**npz)
    
with np.load('real_toplot_1516jj.npz') as npz:
    to_real = np.ma.MaskedArray(**npz)
###sst
with np.load('sst_pred_1516jj.npz') as npz:
    sst_pred = np.ma.MaskedArray(**npz)

with np.load('sst_real_1516jj.npz') as npz:
    sst_real = np.ma.MaskedArray(**npz)    
#"""
act_toplot1 = np.copy(to_real) # proviene de prediccion6m
pred_toplot1 = np.copy(to_pred) # proviene de prediccion6m

#######solo para nino 82/83############
#anio82_2 = np.copy(anom_pred_total[5,:,:]) # este dato es de anio 97/98 ==> anom_pred_total[2,:,:]
#anio82_2[anio82_2 < -0.6] = -0.5 #para aminorar temp negativa
#pred_toplot1[5,:,:] = anio82_2
#####################################
act_toplot = np.copy(act_toplot1)
act_toplot[np.isnan(act_toplot)] = 0
act_toplot.min(),act_toplot.max()

pred_toplot = np.copy(pred_toplot1)
pred_toplot[np.isnan(pred_toplot)] = 0
pred_toplot.min(),pred_toplot.max()

len_frame = len(act_toplot[:,0,0])
#####################################
model_sum_mse = 0
model_sum_rmse = 0
model_sum_rmspe = 0
model_sum_mae = 0
model_sum_mape = 0

single_point_model_sum_rmse = 0

model_sum_mse_current = 0
model_sum_rmse_current = 0
model_sum_rmspe_current = 0
model_sum_mae_current = 0
model_sum_mape_current = 0

model_mse_mes = []
model_rmse_mes = []
model_rmspe_mes = []
model_mae_mes = []
model_mape_mes = []
####para ubicar un punto en el oceano
point_x, point_y = 2, 2

pred_toplot2 = pred_toplot1.reshape(12,900)
pred_toplot3 = pred_toplot2.T
pred_toplot3 = pred_toplot3[~np.isnan(pred_toplot3).any(axis=1)]
#pred_toplot3 = pred_toplot3.T

act_toplot2 = act_toplot1.reshape(12,900)
act_toplot3 = act_toplot2.T
act_toplot3 = act_toplot3[~np.isnan(act_toplot3).any(axis=1)]
#act_toplot3 = act_toplot3.T
#model_rmspe_mes = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100

#j=0
for j in range(len_frame):
    model_mse = mean_squared_error(act_toplot[j,:,:], pred_toplot[j,:,:])
    model_rmse = sqrt(model_mse)
    model_mse_mes.append(model_mse)
    model_rmse_mes.append(model_rmse)
    
    model_rmspe = np.sqrt(np.mean(np.square((act_toplot3[:,j] - pred_toplot3[:,j]) / act_toplot3[:,j])))
    model_rmspe_mes.append(model_rmspe)
        
    model_mae = mean_absolute_error(act_toplot[j,:,:], pred_toplot[j,:,:])
    model_mae_mes.append(model_mae)
    
    model_mape = mean_absolute_percentage_error(act_toplot[j,:,:], pred_toplot[j,:,:])
    model_mape_mes.append(model_mape)

    model_sum_mse = model_sum_mse + model_mse    
    model_sum_rmse = model_sum_rmse + model_rmse
    model_sum_rmspe = model_sum_rmspe + model_rmspe
    model_sum_mae = model_sum_mae + model_mae
    model_sum_mape = model_sum_mape + model_mape

    model_sum_mse_current = model_sum_mse_current + model_mse
    model_sum_rmse_current = model_sum_rmse_current + model_rmse
    model_sum_rmspe_current = model_sum_rmspe_current + model_rmspe
    model_sum_mae_current = model_sum_mae_current + model_mae
    model_sum_mape_current = model_sum_mape_current + model_mape

    single_model_rmse = (act_toplot[:,point_x, point_y] - pred_toplot[:,point_x, point_y])**2
    
    single_point_model_sum_rmse = single_point_model_sum_rmse + single_model_rmse

#ahora correlacionamos con estadisticos pearson, spearman y kendall
import numpy as np
import scipy.stats
#NINO vector
nino_pred_vec = np.copy(pred_toplot1)
nino_real_vec = np.copy(act_toplot1)

nino_pred_vec = nino_pred_vec.reshape(12,900)
nino_pred_vec  = nino_pred_vec.T

nino_real_vec = nino_real_vec.reshape(12,900)
nino_real_vec = nino_real_vec.T
##eliminamos el nan
nino_pred_vec = nino_pred_vec[~np.isnan(nino_pred_vec).any(axis=1)]
nino_real_vec = nino_real_vec[~np.isnan(nino_real_vec).any(axis=1)]

nino_pred_vec.shape
nino_real_vec.shape
r_p = np.zeros((12,2)) #(12,2) colocar a (12,1) cuando quiero solo valores de r y activar if de los tres
r_s = np.zeros((12,2)) #(12,2) colocar a (12,1) cuando quiero solo valores de r y activar if de los tres
r_k = np.zeros((12,2)) #(12,2) colocar a (12,1) cuando quiero solo valores de r y activar if de los tres
for j in range(len_frame): #j=0
    r1, p1 = scipy.stats.pearsonr(nino_real_vec[:,j], nino_pred_vec[:,j])
    r2, p2 = scipy.stats.spearmanr(nino_real_vec[:,j], nino_pred_vec[:,j])
    r3, p3 = scipy.stats.kendalltau(nino_real_vec[:,j], nino_pred_vec[:,j])
    #"""op. 1 activar cuando es r_p o similar es (12,2)
    r_p[j,0] = r1
    r_p[j,1] = p1
    r_s[j,0] = r2
    r_s[j,1] = p2
    r_k[j,0] = r3
    r_k[j,1] = p3
    """op 2. activar cuando es r_p o similar es (12,1)
    if p1 < 0.05:
        r_p34[j] = r1
    else:
        r_p34[j] = np.NaN
    if p2 < 0.05:
        r_s34[j] = r2
    else:
        r_s34[j] = np.NaN
    if p3 < 0.05:
        r_k34[j] = r3
    else:
        r_k34[j] = np.NaN
"""
# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
#from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')
#############################################################################
############################################################
# Gráficos regression
# ==============================================================================
##cordenar en funcion a los que deseas los meses
titulo = ["Julio","Agosto","Setiembre","Octubre","Noviembre","Diciembre"]
#titulo = ["Enero","Febreo","Marzo","Abril","Mayo", "Junio"]

from scipy import stats
import seaborn as sns
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 10))

y_train = nino_real_vec[:,0]
prediccion_train = nino_pred_vec[:,0]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 12, color = 'r') # s = size
#axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax00 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci=95, ax=axes[0,0], color='black')
axes[0, 0].set_title(titulo[0], fontsize = 12, fontweight = "bold")
axes[0, 0].set_xlabel('Observado', fontsize = 10)
axes[0, 0].set_ylabel('Pronóstico', fontsize = 10)
axes[0, 0].tick_params(labelsize = 7)
axes[0, 0].text(y_train.min()+0.2, prediccion_train.max()-0.2, '$r = %0.3f$' % r_value, fontsize = 12)
axes[0, 0].text(y_train.min()+0.2, prediccion_train.max()-0.5, '$p = %0.3e$' % p_value, fontsize = 12)
axes[0, 0].text(1.2, -0.3,'$Y=%0.5s \cdot X %0.6s$' % (slope, intercept), fontsize = 12)
#axes[0, 0].text(2, 0, '$r = aa$', fontsize = 12, bbox = dict(facecolor = 'k', alpha = 0.2)) 

y_train = nino_real_vec[:,1]
prediccion_train = nino_pred_vec[:,1]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[0, 1].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 12, color = 'r') # s = size
#axes[0, 1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax01 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci = 95, ax=axes[0,1], color = 'black')#, order=2)
axes[0, 1].set_title(titulo[1], fontsize = 12, fontweight = "bold")
axes[0, 1].set_xlabel('Observado', fontsize = 10)
axes[0, 1].set_ylabel('Pronóstico', fontsize = 10)
axes[0, 1].tick_params(labelsize = 7)
axes[0, 1].text(y_train.min()+0.2, prediccion_train.max()-0.2, '$r = %0.3f$' % r_value, fontsize = 12)
axes[0, 1].text(y_train.min()+0.2, prediccion_train.max()-0.5, '$p = %0.3e$' % p_value, fontsize = 12)
axes[0, 1].text(0.5, -0.9,'$Y=%0.5s \cdot X %0.6s$' % (slope, intercept), fontsize = 12)

y_train = nino_real_vec[:,2]
prediccion_train = nino_pred_vec[:,2]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[0, 2].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 12, color = 'r') # s = size
#axes[0, 2].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax02 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci = 95, ax=axes[0,2], color = 'black')
axes[0, 2].set_title(titulo[2], fontsize = 12, fontweight = "bold")
axes[0, 2].set_xlabel('Observado', fontsize = 10)
axes[0, 2].set_ylabel('Pronóstico', fontsize = 10)
axes[0, 2].tick_params(labelsize = 7)
axes[0, 2].text(y_train.min()+0.2, prediccion_train.max()-0.2, '$r = %0.3f$' % r_value, fontsize = 12)
axes[0, 2].text(y_train.min()+0.2, prediccion_train.max()-0.5, '$p = %0.3e$' % p_value, fontsize = 12)
axes[0, 2].text(0.9, -1,'$Y=%0.5s \cdot X %0.6s$' % (slope, intercept), fontsize = 12)

y_train = nino_real_vec[:,3]
prediccion_train = nino_pred_vec[:,3]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[1, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 12, color = 'r') # s = size
#axes[1, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax10 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci = 95, ax=axes[1,0], color = 'black')
axes[1, 0].set_title(titulo[3], fontsize = 12, fontweight = "bold")
axes[1, 0].set_xlabel('Observado', fontsize = 10)
axes[1, 0].set_ylabel('Pronóstico', fontsize = 10)
axes[1, 0].tick_params(labelsize = 7)
axes[1, 0].text(y_train.min()+0.2, prediccion_train.max()-0.2, '$r = %0.3f$' % r_value, fontsize = 12)
axes[1, 0].text(y_train.min()+0.2, prediccion_train.max()-0.5, '$p = %0.3e$' % p_value, fontsize = 12)
axes[1, 0].text(1, -0.8,'$Y=%0.5s \cdot X+ %0.5s$' % (slope, intercept), fontsize = 12)

y_train = nino_real_vec[:,4]
prediccion_train = nino_pred_vec[:,4]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[1, 1].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 12, color = 'r') # s = size
#axes[1, 1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax11 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci = 95, ax=axes[1,1], color = 'black')
axes[1, 1].set_title(titulo[4], fontsize = 12, fontweight = "bold")
axes[1, 1].set_xlabel('Observado', fontsize = 10)
axes[1, 1].set_ylabel('Pronóstico', fontsize = 10)
axes[1, 1].tick_params(labelsize = 7)
axes[1, 1].text(y_train.min()+0.2, prediccion_train.max()-0.2, '$r = %0.3f$' % r_value, fontsize = 12)
axes[1, 1].text(y_train.min()+0.2, prediccion_train.max()-0.5, '$p = %0.3e$' % p_value, fontsize = 12)
axes[1, 1].text(1.3, -0.95,'$Y=%0.5s \cdot X %0.6s$' % (slope, intercept), fontsize = 12)

y_train = nino_real_vec[:,5]
prediccion_train = nino_pred_vec[:,5]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[1, 2].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 12, color = 'r') # s = size
#axes[1, 2].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax12 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci = 95, ax=axes[1,2], color = 'black')
axes[1, 2].set_title(titulo[5], fontsize = 12, fontweight = "bold")
axes[1, 2].set_xlabel('Observado', fontsize = 10)
axes[1, 2].set_ylabel('Pronóstico', fontsize = 10)
axes[1, 2].tick_params(labelsize = 7)
axes[1, 2].text(y_train.min()+0.2, prediccion_train.max()-0.2, '$r = %0.3f$' % r_value, fontsize = 12)
axes[1, 2].text(y_train.min()+0.2, prediccion_train.max()-0.5, '$p = %0.3e$' % p_value, fontsize = 12)
axes[1, 2].text(1.5, -0.95,'$Y=%0.5s \cdot X %0.6s$' % (slope, intercept), fontsize = 12)

fig.savefig('reg1516jd.pdf', format='pdf')

## graficando estadisticos #####################
for i in range(6):
    y_train = nino_real_vec[:,i]
    prediccion_train = nino_pred_vec[:,i]
    residuos_train   = prediccion_train - y_train
    # Gráficos
    # ==============================================================================
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    
    axes[0, 0].scatter(list(range(len(y_train))), residuos_train, edgecolors=(0, 0, 0), alpha = 0.7, s=30, color = 'k')
    axes[0, 0].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[0, 0].set_title('Residuos del modelo Observado', fontsize = 12, fontweight = "bold")
    axes[0, 0].set_xlabel('Observado')
    axes[0, 0].set_ylabel('Residuo')
    axes[0, 0].tick_params(labelsize = 7)
    
    sns.histplot(data=residuos_train, stat="density", kde=True, line_kws= {'linewidth': 2}, 
                 color="firebrick", alpha=0.5, ax=axes[1, 0])
    axes[0, 1].scatter(prediccion_train, residuos_train, edgecolors=(0, 0, 0), alpha = 0.7, s=30, color = 'r')
    axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[0, 1].set_title('Residuos del modelo Pronóstico', fontsize = 12, fontweight = "bold")
    axes[0, 1].set_xlabel('Pronóstico')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].tick_params(labelsize = 7)
        
    axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 12, fontweight = "bold")
    axes[1, 0].set_xlabel("Residuo")
    axes[1, 0].tick_params(labelsize = 7)
    
    sm.qqplot(residuos_train, fit = True, line = 'q', ax = axes[1, 1], color = 'firebrick', alpha = 0.5, lw = 2)
    axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 12, fontweight = "bold")
    axes[1, 1].tick_params(labelsize = 7)
    # Se eliminan los axes vacíos
    #fig.delaxes(axes[2,1])
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle(titulo[i], fontsize = 14, fontweight = "bold")

##pplot(iris, x="sepal_length", y=gamma, hue="species", kind='qq', height=4, aspect=2, 
 #     display_kws={"identity":False, "fit":True, "reg":True, "ci":0.025})

##############################################################
####para ubicar un region en el oceano##########################
nino34_real = np.copy(act_toplot1[:,2:9,26:52]) #seleccionar solo region nino 3.4
nino34_pred = np.copy(pred_toplot1[:,2:9,26:52]) #seleccionar solo region nino 3.4

nino12_real = np.copy(act_toplot1[:,5:11,66:72]) #seleccionar solo region nino 1+2
nino12_pred = np.copy(pred_toplot1[:,5:11,66:72]) #seleccionar solo region nino 1+2

act_toplot34_1 = np.copy(nino34_real) # proviene de prediccion6m
pred_toplot34_1 = np.copy(nino34_pred) # proviene de prediccion6m

act_toplot12_1 = np.copy(nino12_real) # proviene de prediccion6m
pred_toplot12_1 = np.copy(nino12_pred) # proviene de prediccion6m
#####################################
act_toplot34 = np.copy(act_toplot34_1)
act_toplot34[np.isnan(act_toplot34)] = 0
act_toplot34.min(),act_toplot34.max()

pred_toplot34 = np.copy(pred_toplot34_1)
pred_toplot34[np.isnan(pred_toplot34)] = 0
pred_toplot34.min(),pred_toplot34.max()

act_toplot12 = np.copy(act_toplot12_1)
act_toplot12[np.isnan(act_toplot12)] = 0
act_toplot12.min(),act_toplot12.max()

pred_toplot12 = np.copy(pred_toplot12_1)
pred_toplot12[np.isnan(pred_toplot12)] = 0
pred_toplot12.min(),pred_toplot12.max()
#######################################
len_frame = len(act_toplot34[:,0,0])

model_sum_mse = 0
model_sum_rmse = 0
model_sum_rmspe = 0
model_sum_mae = 0
model_sum_mape = 0

single_point_model_sum_rmse = 0

model_sum_mse_current = 0
model_sum_rmse_current = 0
model_sum_rmspe_current = 0
model_sum_mae_current = 0
model_sum_mape_current = 0

model_mse_mes = []
model_rmse_mes = []
model_rmspe_mes = []
model_mae_mes = []
model_mape_mes = []

## para nino34
for j in range(len_frame):
    model_mse = mean_squared_error(act_toplot34[j,:,:], pred_toplot34[j,:,:])
    model_rmse = sqrt(model_mse)
    model_mse_mes.append(model_mse)
    model_rmse_mes.append(model_rmse)
    
    model_rmspe = np.sqrt(np.mean(np.square((act_toplot34[j,:,:] - pred_toplot34[j,:,:]) / act_toplot34[j,:,:])))
    model_rmspe_mes.append(model_rmspe)
    
    model_mae = mean_absolute_error(act_toplot34[j,:,:], pred_toplot34[j,:,:])
    model_mae_mes.append(model_mae)
    
    model_mape = mean_absolute_percentage_error(act_toplot34[j,:,:], pred_toplot34[j,:,:])
    model_mape_mes.append(model_mape)

    model_sum_mse = model_sum_mse + model_mse    
    model_sum_rmse = model_sum_rmse + model_rmse
    model_sum_rmspe = model_sum_rmspe + model_rmspe
    model_sum_mae = model_sum_mae + model_mae
    model_sum_mape = model_sum_mape + model_mape

    model_sum_mse_current = model_sum_mse_current + model_mse
    model_sum_rmse_current = model_sum_rmse_current + model_rmse
    model_sum_rmspe_current = model_sum_rmspe_current + model_rmspe
    model_sum_mae_current = model_sum_mae_current + model_mae
    model_sum_mape_current = model_sum_mape_current + model_mape

## para nino12

##############
act_toplot12[act_toplot12==0 ] = 0.00001
pred_toplot12[pred_toplot12==0 ] = 0.00001

model_sum_mse = 0
model_sum_rmse = 0
model_sum_rmspe = 0
model_sum_mae = 0
model_sum_mape = 0

single_point_model_sum_rmse = 0

model_sum_mse_current = 0
model_sum_rmse_current = 0
model_sum_rmspe_current = 0
model_sum_mae_current = 0
model_sum_mape_current = 0

model_mse_mes = []
model_rmse_mes = []
model_rmspe_mes = []
model_mae_mes = []
model_mape_mes = []

for j in range(len_frame):
    model_mse = mean_squared_error(act_toplot12[j,:,:], pred_toplot12[j,:,:])
    model_rmse = sqrt(model_mse)
    model_mse_mes.append(model_mse)
    model_rmse_mes.append(model_rmse)
        
    model_rmspe = np.sqrt(np.mean(np.square((act_toplot12[j,:,:] - pred_toplot12[j,:,:]) / act_toplot12[j,:,:])))
    model_rmspe_mes.append(model_rmspe)
    
    model_mae = mean_absolute_error(act_toplot12[j,:,:], pred_toplot12[j,:,:])
    model_mae_mes.append(model_mae)
    
    model_mape = mean_absolute_percentage_error(act_toplot12[j,:,:], pred_toplot12[j,:,:])
    model_mape_mes.append(model_mape)

    model_sum_mse = model_sum_mse + model_mse    
    model_sum_rmse = model_sum_rmse + model_rmse
    model_sum_rmspe = model_sum_rmspe + model_rmspe
    model_sum_mae = model_sum_mae + model_mae
    model_sum_mape = model_sum_mape + model_mape

    model_sum_mse_current = model_sum_mse_current + model_mse
    model_sum_rmse_current = model_sum_rmse_current + model_rmse
    model_sum_rmspe_current = model_sum_rmspe_current + model_rmspe
    model_sum_mae_current = model_sum_mae_current + model_mae
    model_sum_mape_current = model_sum_mape_current + model_mape
    
#################################################################################################
### ahora correlacionamos con estadisticos pearson, spearman y kendall para nino 3.4 y 1+2 ######
#################################################################################################
import numpy as np
import scipy.stats
#NINO 3.4 vector
nino34_pred_vec = np.copy(nino34_pred)
nino34_real_vec = np.copy(nino34_real)

nino34_pred_vec = nino34_pred_vec.reshape(12,182)
nino34_pred_vec  = nino34_pred_vec.T
nino34_real_vec = nino34_real_vec.reshape(12,182)
nino34_real_vec = nino34_real_vec.T

r_p34 = np.zeros((12,2)) #(12,2) colocar a (12,1) cuando quiero solo valores de r y activar if de los tres
r_s34 = np.zeros((12,2)) #(12,2) colocar a (12,1) cuando quiero solo valores de r y activar if de los tres
r_k34 = np.zeros((12,2)) #(12,2) colocar a (12,1) cuando quiero solo valores de r y activar if de los tres
for j in range(len_frame): #j=0
    r1, p1 = scipy.stats.pearsonr(nino34_real_vec[:,j], nino34_pred_vec[:,j])
    r2, p2 = scipy.stats.spearmanr(nino34_real_vec[:,j], nino34_pred_vec[:,j])
    r3, p3 = scipy.stats.kendalltau(nino34_real_vec[:,j], nino34_pred_vec[:,j])
    #"""op. 1 activar cuando es r_p o similar es (12,2)
    r_p34[j,0] = r1
    r_p34[j,1] = p1
    r_s34[j,0] = r2
    r_s34[j,1] = p2
    r_k34[j,0] = r3
    r_k34[j,1] = p3
    """op 2. activar cuando es r_p o similar es (12,1)
    if p1 < 0.05:
        r_p34[j] = r1
    else:
        r_p34[j] = np.NaN
    if p2 < 0.05:
        r_s34[j] = r2
    else:
        r_s34[j] = np.NaN
    if p3 < 0.05:
        r_k34[j] = r3
    else:
        r_k34[j] = np.NaN
    """
### graficos regresion 3.4 ###############
from scipy import stats
import seaborn as sns
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 10))

y_train = nino34_real_vec[:,0]
prediccion_train = nino34_pred_vec[:,0]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 20, color = 'm') # s = size
#axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax00 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci=95, ax=axes[0,0], color='black')
axes[0, 0].set_title(titulo[0], fontsize = 12, fontweight = "bold")
axes[0, 0].set_xlabel('Observado', fontsize = 10)
axes[0, 0].set_ylabel('Pronóstico', fontsize = 10)
axes[0, 0].tick_params(labelsize = 7)
axes[0, 0].text(y_train.min()+0.1, prediccion_train.max()-0.05, '$r = %0.3f$' % r_value, fontsize = 12)
axes[0, 0].text(y_train.min()+0.1, prediccion_train.max()-0.15, '$p = %0.3e$' % p_value, fontsize = 12)
#axes[0, 0].text(1.3, 0.4,'$Y=%0.5s \cdot X %0.6s$' % (slope, intercept), fontsize = 12)
#axes[0, 0].text(2, 0, '$r = aa$', fontsize = 12, bbox = dict(facecolor = 'k', alpha = 0.2)) 

y_train = nino34_real_vec[:,1]
prediccion_train = nino34_pred_vec[:,1]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[0, 1].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 20, color = 'm') # s = size
#axes[0, 1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax01 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci = 95, ax=axes[0,1], color = 'black')
axes[0, 1].set_title(titulo[1], fontsize = 12, fontweight = "bold")
axes[0, 1].set_xlabel('Observado', fontsize = 10)
axes[0, 1].set_ylabel('Pronóstico', fontsize = 10)
axes[0, 1].tick_params(labelsize = 7)
axes[0, 1].text(y_train.min()+0.1, prediccion_train.max()-0.05, '$r = %0.3f$' % r_value, fontsize = 12)
axes[0, 1].text(y_train.min()+0.1, prediccion_train.max()-0.15, '$p = %0.3e$' % p_value, fontsize = 12)

#axes[0, 1].text(1.5, 0.2,'$Y=%0.5s \cdot X %0.6s$' % (slope, intercept), fontsize = 12)

y_train = nino34_real_vec[:,2]
prediccion_train = nino34_pred_vec[:,2]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[0, 2].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 20, color = 'm') # s = size
#axes[0, 2].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax02 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci = 95, ax=axes[0,2], color = 'black')
axes[0, 2].set_title(titulo[2], fontsize = 12, fontweight = "bold")
axes[0, 2].set_xlabel('Observado', fontsize = 10)
axes[0, 2].set_ylabel('Pronóstico', fontsize = 10)
axes[0, 2].tick_params(labelsize = 7)
axes[0, 2].text(y_train.min()+0.1, prediccion_train.max()-0.05, '$r = %0.3f$' % r_value, fontsize = 12)
axes[0, 2].text(y_train.min()+0.1, prediccion_train.max()-0.15, '$p = %0.3e$' % p_value, fontsize = 12)
#axes[0, 2].text(1.7, -0.35,'$Y=%0.5s \cdot X %0.6s$' % (slope, intercept), fontsize = 12)

y_train = nino34_real_vec[:,3]
prediccion_train = nino34_pred_vec[:,3]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[1, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 20, color = 'm') # s = size
#axes[1, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax10 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci = 95, ax=axes[1,0], color = 'black')
axes[1, 0].set_title(titulo[3], fontsize = 12, fontweight = "bold")
axes[1, 0].set_xlabel('Observado', fontsize = 10)
axes[1, 0].set_ylabel('Pronóstico', fontsize = 10)
axes[1, 0].tick_params(labelsize = 7)
axes[1, 0].text(y_train.min()+0.1, prediccion_train.max()-0.05, '$r = %0.3f$' % r_value, fontsize = 12)
axes[1, 0].text(y_train.min()+0.1, prediccion_train.max()-0.15, '$p = %0.3e$' % p_value, fontsize = 12)
#axes[1, 0].text(1.8, 0.1,'$Y=%0.5s \cdot X+ %0.5s$' % (slope, intercept), fontsize = 12)

y_train = nino34_real_vec[:,4]
prediccion_train = nino34_pred_vec[:,4]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[1, 1].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 20, color = 'm') # s = size
#axes[1, 1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax11 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci = 95, ax=axes[1,1], color = 'black')
axes[1, 1].set_title(titulo[4], fontsize = 12, fontweight = "bold")
axes[1, 1].set_xlabel('Observado', fontsize = 10)
axes[1, 1].set_ylabel('Pronóstico', fontsize = 10)
axes[1, 1].tick_params(labelsize = 7)
axes[1, 1].text(y_train.min()+0.1, prediccion_train.max()-0.05, '$r = %0.3f$' % r_value, fontsize = 12)
axes[1, 1].text(y_train.min()+0.1, prediccion_train.max()-0.15, '$p = %0.3e$' % p_value, fontsize = 12)
#axes[1, 1].text(2.1, 0.4,'$Y=%0.5s \cdot X %0.6s$' % (slope, intercept), fontsize = 12)

y_train = nino34_real_vec[:,5]
prediccion_train = nino34_pred_vec[:,5]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[1, 2].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 20, color = 'm') # s = size
#axes[1, 2].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax12 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci = 95, ax=axes[1,2], color = 'black')
axes[1, 2].set_title(titulo[5], fontsize = 12, fontweight = "bold")
axes[1, 2].set_xlabel('Observado', fontsize = 10)
axes[1, 2].set_ylabel('Pronóstico', fontsize = 10)
axes[1, 2].tick_params(labelsize = 7)
axes[1, 2].text(y_train.min()+0.1, prediccion_train.max()-0.05, '$r = %0.3f$' % r_value, fontsize = 12)
axes[1, 2].text(y_train.min()+0.1, prediccion_train.max()-0.15, '$p = %0.3e$' % p_value, fontsize = 12)
#axes[1, 2].text(2, 0.5,'$Y=%0.5s \cdot X %0.6s$' % (slope, intercept), fontsize = 12)
fig.savefig('reg34_1516jd.pdf', format='pdf')
####graficando estadisticos #############
for i in range(len_frame):
    y_train = nino34_real_vec[:,i]
    prediccion_train = nino34_pred_vec[:,i]
    residuos_train   = prediccion_train - y_train
    # Gráficos
    # ==============================================================================
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 10))
    
    axes[0, 0].scatter(list(range(len(y_train))), residuos_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[0, 0].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[0, 0].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[0, 0].set_xlabel('id')
    axes[0, 0].set_ylabel('Residuo')
    axes[0, 0].tick_params(labelsize = 7)
    
    sns.histplot(
        data    = residuos_train,
        stat    = "density",
        kde     = True,
        line_kws= {'linewidth': 1},
        color   = "firebrick",
        alpha   = 0.3,
        ax      = axes[1, 0])
    
    axes[0, 1].scatter(prediccion_train, residuos_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[0, 1].set_title('Residuos del modelo vs predicción', fontsize = 10, fontweight = "bold")
    axes[0, 1].set_xlabel('Predicción')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].tick_params(labelsize = 7)
        
    axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[1, 0].set_xlabel("Residuo")
    axes[1, 0].tick_params(labelsize = 7)
    
    sm.qqplot(
        residuos_train,
        fit   = True,
        line  = 'q',
        ax    = axes[1, 1], 
        color = 'firebrick',
        alpha = 0.4,
        lw    = 2)
    
    axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[1, 1].tick_params(labelsize = 7)
    
    # Se eliminan los axes vacíos
    #fig.delaxes(axes[2,1])
    
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('Diagnóstico residuos', fontsize = 12, fontweight = "bold")

#################################################################
### estadisticos para nino 12 vector ############################
nino12_pred_vec = np.copy(nino12_pred)
nino12_real_vec = np.copy(nino12_real)

nino12_pred_vec = nino12_pred_vec.reshape(12,36)
nino12_pred_vec = nino12_pred_vec.T
nino12_real_vec = nino12_real_vec.reshape(12,36)
nino12_real_vec = nino12_real_vec.T
##eliminamos el nan
nino12_pred_vec = np.delete(nino12_pred_vec, 17, axis=0)
nino12_real_vec = np.delete(nino12_real_vec, 17, axis=0)

r_p12 = np.zeros((12,2)) #(12,2) colocar a (12,1) cuando quiero solo valores de r y activar if de los tres
r_s12 = np.zeros((12,2)) #(12,2) colocar a (12,1) cuando quiero solo valores de r y activar if de los tres
r_k12 = np.zeros((12,2)) #(12,2) colocar a (12,1) cuando quiero solo valores de r y activar if de los tres
for j in range(len_frame): #j=0
    r1, p1 = scipy.stats.pearsonr(nino12_real_vec[:,j], nino12_pred_vec[:,j])
    r2, p2 = scipy.stats.spearmanr(nino12_real_vec[:,j], nino12_pred_vec[:,j])
    r3, p3 = scipy.stats.kendalltau(nino12_real_vec[:,j], nino12_pred_vec[:,j])
    #"""op. 1 activar cuando es r_p o similar es (12,2)
    r_p12[j,0] = r1
    r_p12[j,1] = p1
    r_s12[j,0] = r2
    r_s12[j,1] = p2
    r_k12[j,0] = r3
    r_k12[j,1] = p3
    """op 2. activar cuando es r_p o similar es (12,1)
    if p1 < 0.05:
        r_p12[j] = r1
    else:
        r_p12[j] = np.NaN
    if p2 < 0.05:
        r_s12[j] = r2
    else:
        r_s12[j] = np.NaN
    if p3 < 0.05:
        r_k12[j] = r3
    else:
        r_k12[j] = np.NaN
    """
### graficos regresion 1+2 ###############
from scipy import stats
import seaborn as sns
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 10))

y_train = nino12_real_vec[:,0]
prediccion_train = nino12_pred_vec[:,0]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 20, color = 'b') # s = size
#axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax00 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci=95, ax=axes[0,0], color='black')
axes[0, 0].set_title(titulo[0], fontsize = 12, fontweight = "bold")
axes[0, 0].set_xlabel('Observado', fontsize = 10)
axes[0, 0].set_ylabel('Pronóstico', fontsize = 10)
axes[0, 0].tick_params(labelsize = 7)
axes[0, 0].text(y_train.min()+0.1, prediccion_train.max()-0.05, '$r = %0.3f$' % r_value, fontsize = 12)
axes[0, 0].text(y_train.min()+0.1, prediccion_train.max()-0.15, '$p = %0.3e$' % p_value, fontsize = 12)
#axes[0, 0].text(2.1, 0.75,'$Y=%0.5s \cdot X %0.6s$' % (slope, intercept), fontsize = 12)
#axes[0, 0].text(2, 0, '$r = aa$', fontsize = 12, bbox = dict(facecolor = 'k', alpha = 0.2)) 

y_train = nino12_real_vec[:,1]
prediccion_train = nino12_pred_vec[:,1]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[0, 1].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 20, color = 'b') # s = size
#axes[0, 1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax01 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci = 95, ax=axes[0,1], color = 'black')
axes[0, 1].set_title(titulo[1], fontsize = 12, fontweight = "bold")
axes[0, 1].set_xlabel('Observado', fontsize = 10)
axes[0, 1].set_ylabel('Pronóstico', fontsize = 10)
axes[0, 1].tick_params(labelsize = 7)
axes[0, 1].text(y_train.min()+0.1, prediccion_train.max()-0.05, '$r = %0.3f$' % r_value, fontsize = 12)
axes[0, 1].text(y_train.min()+0.1, prediccion_train.max()-0.15, '$p = %0.3e$' % p_value, fontsize = 12)
#axes[0, 1].text(1.3, 0.8,'$Y=%0.5s \cdot X+ %0.5s$' % (slope, intercept), fontsize = 12)

y_train = nino12_real_vec[:,2]
prediccion_train = nino12_pred_vec[:,2]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[0, 2].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 20, color = 'b') # s = size
#axes[0, 2].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax02 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci = 95, ax=axes[0,2], color = 'black')
axes[0, 2].set_title(titulo[2], fontsize = 12, fontweight = "bold")
axes[0, 2].set_xlabel('Observado', fontsize = 10)
axes[0, 2].set_ylabel('Pronóstico', fontsize = 10)
axes[0, 2].tick_params(labelsize = 7)
axes[0, 2].text(y_train.min()+0.1, prediccion_train.max()-0.05, '$r = %0.3f$' % r_value, fontsize = 12)
axes[0, 2].text(y_train.min()+0.1, prediccion_train.max()-0.15, '$p = %0.3e$' % p_value, fontsize = 12)
#axes[0, 2].text(1.7, 0.3,'$Y=%0.5s \cdot X+ %0.5s$' % (slope, intercept), fontsize = 12)
fig.savefig('reg12_1516jd.pdf', format='pdf')

y_train = nino12_real_vec[:,3]
prediccion_train = nino12_pred_vec[:,3]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[1, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 20, color = 'b') # s = size
#axes[1, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax10 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci = 95, ax=axes[1,0], color = 'black')
axes[1, 0].set_title(titulo[3], fontsize = 12, fontweight = "bold")
axes[1, 0].set_xlabel('Observado', fontsize = 10)
axes[1, 0].set_ylabel('Pronóstico', fontsize = 10)
axes[1, 0].tick_params(labelsize = 7)
axes[1, 0].text(y_train.min()+0.1, prediccion_train.max()-0.05, '$r = %0.3f$' % r_value, fontsize = 12)
axes[1, 0].text(y_train.min()+0.1, prediccion_train.max()-0.15, '$p = %0.3e$' % p_value, fontsize = 12)
#axes[1, 0].text(1.85, 0.95,'$Y=%0.5s \cdot X+ %0.5s$' % (slope, intercept), fontsize = 12)

y_train = nino12_real_vec[:,4]
prediccion_train = nino12_pred_vec[:,4]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[1, 1].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 20, color = 'b') # s = size
#axes[1, 1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax11 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci = 95, ax=axes[1,1], color = 'black')
axes[1, 1].set_title(titulo[4], fontsize = 12, fontweight = "bold")
axes[1, 1].set_xlabel('Observado', fontsize = 10)
axes[1, 1].set_ylabel('Pronóstico', fontsize = 10)
axes[1, 1].tick_params(labelsize = 7)
axes[1, 1].text(y_train.min()+0.1, prediccion_train.max()-0.05, '$r = %0.3f$' % r_value, fontsize = 12)
axes[1, 1].text(y_train.min()+0.1, prediccion_train.max()-0.15, '$p = %0.3e$' % p_value, fontsize = 12)
#axes[1, 1].text(1.8, 1.95,'$Y=%0.6s \cdot X+ %0.5s$' % (slope, intercept), fontsize = 12)

y_train = nino12_real_vec[:,5]
prediccion_train = nino12_pred_vec[:,5]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(y_train, prediccion_train)
predict_y = slope * y_train + intercept
axes[1, 2].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.7, s= 20, color = 'b') # s = size
#axes[1, 2].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', color = 'black', lw=2)
ax12 = sns.regplot(x=y_train, y=prediccion_train,scatter=False, ci = 95, ax=axes[1,2], color = 'black')
axes[1, 2].set_title(titulo[5], fontsize = 12, fontweight = "bold")
axes[1, 2].set_xlabel('Observado', fontsize = 10)
axes[1, 2].set_ylabel('Pronóstico', fontsize = 10)
axes[1, 2].tick_params(labelsize = 7)
axes[1, 2].text(y_train.min()+0.1, prediccion_train.max()-0.05, '$r = %0.3f$' % r_value, fontsize = 12)
axes[1, 2].text(y_train.min()+0.1, prediccion_train.max()-0.15, '$p = %0.3e$' % p_value, fontsize = 12)
#axes[1, 2].text(1.6, -0.7,'$Y=%0.5s \cdot X %0.6s$' % (slope, intercept), fontsize = 12)
fig.savefig('reg12_1516jd.pdf', format='pdf')
####graficando estadisticos #############
for i in range(len_frame):
    y_train = nino12_real_vec[:,i]
    prediccion_train = nino12_pred_vec[:,i]
    residuos_train   = prediccion_train - y_train
    # Gráficos
    # ==============================================================================
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 10))
    
    axes[0, 0].scatter(list(range(len(y_train))), residuos_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[0, 0].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[0, 0].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[0, 0].set_xlabel('id')
    axes[0, 0].set_ylabel('Residuo')
    axes[0, 0].tick_params(labelsize = 7)
    
    sns.histplot(
        data    = residuos_train,
        stat    = "density",
        kde     = True,
        line_kws= {'linewidth': 1},
        color   = "firebrick",
        alpha   = 0.3,
        ax      = axes[1, 0])
    
    axes[0, 1].scatter(prediccion_train, residuos_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[0, 1].set_title('Residuos del modelo vs predicción', fontsize = 10, fontweight = "bold")
    axes[0, 1].set_xlabel('Predicción')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].tick_params(labelsize = 7)
        
    axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[1, 0].set_xlabel("Residuo")
    axes[1, 0].tick_params(labelsize = 7)
    
    sm.qqplot(
        residuos_train,
        fit   = True,
        line  = 'q',
        ax    = axes[1, 1], 
        color = 'firebrick',
        alpha = 0.4,
        lw    = 2)
    
    axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[1, 1].tick_params(labelsize = 7)
    
    # Se eliminan los axes vacíos
    #fig.delaxes(axes[2,1])
    
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('Diagnóstico residuos', fontsize = 12, fontweight = "bold")

##############################################
###para cada 6 meses
from scipy import stats
r_val_jd = np.zeros((len(anom_real[0,:,0]),len(anom_real[0,0,:])))
r_val_ej = np.zeros((len(anom_real[0,:,0]),len(anom_real[0,0,:])))

for ixx in range(len(anom_real[0,:,0])): #ixx=0
    for iyy in range(len(anom_real[0,0,:])): #iyy=1
        #r,p=stats.pearsonr(act_toplot[0:6,ixx,iyy],pred_toplot[0:6,ixx,iyy]) #para pearson
        #r,p=stats.spearmanr(act_toplot[0:6,ixx,iyy],pred_toplot[0:6,ixx,iyy]) #para pearson
        #r,p=stats.kendalltau(act_toplot[0:6,ixx,iyy],pred_toplot[0:6,ixx,iyy]) #para pearson
        if p < 0.10:
            r_val_jd[ixx,iyy] = r
        else:
            r_val_jd[ixx,iyy] = 0

for ixx in range(len(anom_real[0,:,0])): #ixx=0
    for iyy in range(len(anom_real[0,0,:])): #iyy=1
        r,p=stats.pearsonr(act_toplot[6:12,ixx,iyy],pred_toplot[6:12,ixx,iyy]) #para pearson
        #r,p=stats.spearmanr(act_toplot[6:12,ixx,iyy],pred_toplot[6:12,ixx,iyy]) #para pearson
        #r,p=stats.kendalltau(act_toplot[6:12,ixx,iyy],pred_toplot[6:12,ixx,iyy]) #para pearson
        if p < 0.10:
            r_val_ej[ixx,iyy] = r
        else:
            r_val_ej[ixx,iyy] = 0
###graficando r_val en el mundo################################
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

plt.figure(figsize=(10,12))#10,30
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
#ax.set_global()
#ax.coastlines()
#para colocar el colobar con centro igual cero codigo 'norm='
clevs = np.arange(-1, 1.05, 0.05)# plotear el funcion del rango de analisis de min max -6 6 exacto
cf = ax.contourf(lon2, lat2, r_val_jd, clevs, transform=ccrs.PlateCarree(), cmap = 'RdBu_r') #afmhot RdBu_r RdYlBu, jet
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=1)
#ax.add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.1)
ax.add_feature(cfeature.BORDERS, linewidths=0.4) # para agregar bordes de paises
ax.stock_img() # para graficar mar y tierra
#import os
#os.environ["CARTOPY_USER_BACKGROUNDS"] = "C:/Users/HP/anaconda3/Lib/site-packages/cartopy/BG"
#ax.background_img(name='etopo', resolution='high') # para topografia
ax.set_xticks(np.arange(120,300,20), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(-15,15+5,5), crs=ccrs.PlateCarree(central_longitude=180))
## seleccionando el solo el area a analisis
minlon = 138; maxlon = 286 ; minlat = -12; maxlat = 10
ax.set_extent([minlon, maxlon, minlat, maxlat],crs=ccrs.PlateCarree())
ax.grid(linestyle='dotted', linewidth=1)
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.07, aspect=50)#, extendrect=False) # para poner recto o triangular
cbar.set_label('r valor')
#plot the x-y-axis label
plt.xlabel("Longitud")
plt.ylabel("Latitud")
# Set some titles
plt.title('Promedio Mensual de la TSM', loc='left', fontsize=10)
plt.title('Periodo: {0:%Y} - {1:%Y}'.format(vtimes[0], vtimes[-1]), 
          loc='right', fontsize=10)
plt.show()
###graficando
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

plt.figure(figsize=(10,12))#10,30
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
#ax.set_global()
#ax.coastlines()
#para colocar el colobar con centro igual cero codigo 'norm='
clevs = np.arange(-1, 1.05, 0.05)# plotear el funcion del rango de analisis de min max -6 6 exacto
cf = ax.contourf(lon2, lat2, r_val_ej, clevs, transform=ccrs.PlateCarree(), cmap = 'RdBu_r') #afmhot RdBu_r RdYlBu, jet
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=1)
#ax.add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.1)
ax.add_feature(cfeature.BORDERS, linewidths=0.4) # para agregar bordes de paises
ax.stock_img() # para graficar mar y tierra
#import os
#os.environ["CARTOPY_USER_BACKGROUNDS"] = "C:/Users/HP/anaconda3/Lib/site-packages/cartopy/BG"
#ax.background_img(name='etopo', resolution='high') # para topografia
ax.set_xticks(np.arange(120,300,20), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(-15,15+5,5), crs=ccrs.PlateCarree(central_longitude=180))
## seleccionando el solo el area a analisis
minlon = 138; maxlon = 286 ; minlat = -12; maxlat = 10
ax.set_extent([minlon, maxlon, minlat, maxlat],crs=ccrs.PlateCarree())
ax.grid(linestyle='dotted', linewidth=1)
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.07, aspect=50)#, extendrect=False) # para poner recto o triangular
cbar.set_label('Temperatura °C')
#plot the x-y-axis label
plt.xlabel("Longitud")
plt.ylabel("Latitud")
# Set some titles
plt.title('Promedio Mensual de la TSM', loc='left', fontsize=10)
plt.title('Periodo: {0:%Y} - {1:%Y}'.format(vtimes[0], vtimes[-1]), 
          loc='right', fontsize=10)
plt.show()
