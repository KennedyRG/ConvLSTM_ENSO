# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:32:39 2021

@author: KRGT
"""
#codigo para crear "npz", en este ejemplo sera para test.npz

###########
import os
os.chdir(r'E:\DocumentsD\ENSO\grid')
os.getcwd()

import numpy as np
# guardando 'masked array'como array test.npz para su corrida en tensorflow :
#sst.data se carga del codigo "sst_mean.py" 
#del codigo ==> sst = a.variables['sst'][:] #desde 1850/1 - 2019/12  = 2040
np.savez_compressed('test1.npz', data=sst.data, mask=sst.mask)

with np.load('test.npz') as npz:
    arr = np.ma.MaskedArray(**npz)
    
    
sst_max = np.amax(arr)
sst_min = np.amin(arr)
sst_mean = np.mean(arr)
##########para 2 grados
import os
os.chdir(r'E:\DocumentsD\ENSO\grid')
os.getcwd()

import numpy as np
# guardando 'masked array'como array test.npz para su corrida en tensorflow :
#sst.data se carga del codigo "sst_mean.py" 
#del codigo ==> sst = a.variables['sst'][:] #desde 1850/1 - 2019/12  = 2040
np.savez_compressed('test2g.npz', data=sst.data, mask=sst.mask)

import os
os.chdir(r'E:\DocumentsD\ENSO\grid\keras_model')
os.getcwd()


##guardando matrix 82-83 de prediccion
import numpy as np
np.savez_compressed('pred_toplot_8283.npz', data=pred_toplot1)
##abriendo matrix 82-83 de prediccion
with np.load('pred_toplot_8283.npz') as npz:
    arr = np.ma.MaskedArray(**npz)

##creando archivo nps de 1854-01 hasta 2022-05
import os
os.chdir(r'E:\DocumentsD\ENSO\grid')
os.getcwd()

import numpy as np
# guardando 'masked array'como array test.npz para su corrida en tensorflow :
#sst.data se carga del codigo "sst_mean.py" 
#del codigo ==> sst = a.variables['sst'][:] #desde 1850/1 - 2019/12  = 2040
np.savez_compressed('test2g_2022_05.npz', data=sst.data, mask=sst.mask)
#####################################################
##creando archivo nps de 1854-01 hasta 2022-06
import os
os.chdir(r'E:\DocumentsD\ENSO\grid')
os.getcwd()

import numpy as np
# usar previamentesst_mean_2g_2022_05_06.py, ubicado en E:\DocumentsD\pyDoc
# guardando 'masked array'como array test.npz para su corrida en tensorflow :
#sst.data se carga del codigo "sst_mean.py" 
#del codigo ==> sst = a.variables['sst'][:] #desde 1850/1 - 2022/06 = 2040
np.savez_compressed('test2g_2022_06.npz', data=sst.data, mask=sst.mask)
