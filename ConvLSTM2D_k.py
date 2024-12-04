'''
Desc: ConvLSTM2D.
'''

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import Dropout
from keras.models import load_model

height, width = 12, 75

def model():
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=16, kernel_size=(3, 3),
                       input_shape=(None, height, width, 1),
                       padding='same', return_sequences=True))
    seq.add(ConvLSTM2D(filters=16, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(ConvLSTM2D(filters=16, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(ConvLSTM2D(filters=16, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(Conv3D(filters=1, kernel_size=(1, 1, 1),
                   activation='relu',
                   padding='same', data_format='channels_last'))
    return seq
