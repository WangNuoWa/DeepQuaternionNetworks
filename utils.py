#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras.backend as K
from keras.layers import Lambda

def get_r(x):
    image_format = K.image_data_format()
    ndim = K.ndim(x)
    input_shape = K.shape(x)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        input_dim = input_shape[1] // 4
        return x[:, :input_dim]

    input_dim = input_shape[1] // 4
    if ndim == 3:
        return x[:, :, :input_dim]
    elif ndim == 4:
        return x[:, :, :, :input_dim]
    elif ndim == 5:
        return x[:, :, :, :, :input_dim]


def get_i(x):
    image_format = K.image_data_format()
    ndim = K.ndim(x)
    input_shape = K.shape(x)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        input_dim = input_shape[1] // 4
        return x[:, input_dim:input_dim*2]

    input_dim = input_shape[1] // 4
    if ndim == 3:
        return x[:, :, input_dim:input_dim*2]
    elif ndim == 4:
        return x[:, :, :, input_dim:input_dim*2]
    elif ndim == 5:
        return x[:, :, :, :, input_dim:input_dim*2]


def get_j(x):
    image_format = K.image_data_format()
    ndim = K.ndim(x)
    input_shape = K.shape(x)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        input_dim = input_shape[1] // 4
        return x[:, input_dim*2:input_dim*3]

    input_dim = input_shape[1] // 4
    if ndim == 3:
        return x[:, :, input_dim*2:input_dim*3]
    elif ndim == 4:
        return x[:, :, :, input_dim*2:input_dim*3]
    elif ndim == 5:
        return x[:, :, :, :, input_dim*2:input_dim*3]


def get_k(x):
    image_format = K.image_data_format()
    ndim = K.ndim(x)
    input_shape = K.shape(x)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        input_dim = input_shape[1] // 4
        return x[:, input_dim*3:]

    input_dim = input_shape[1] // 4
    if ndim == 3:
        return x[:, :, input_dim*3:]
    elif ndim == 4:
        return x[:, :, :, input_dim*3:]
    elif ndim == 5:
        return x[:, :, :, :, input_dim*3:]


def getpart_output_shape(input_shape):
    returned_shape = list(input_shape[:])
    image_format = K.image_data_format()
    ndim = len(returned_shape)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        axis = 1
    else:
        axis = -1

    returned_shape[1] = returned_shape[1] // 4

    return tuple(returned_shape)

GetR = Lambda(get_r, output_shape=getpart_output_shape)
GetI = Lambda(get_i, output_shape=getpart_output_shape)
GetJ = Lambda(get_j, output_shape=getpart_output_shape)
GetK = Lambda(get_k, output_shape=getpart_output_shape)
