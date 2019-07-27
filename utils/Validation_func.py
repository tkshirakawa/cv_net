'''
    Copyright (c) 2019, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com
    
    
    Released under the BSD license.
    URL: https://opensource.org/licenses/BSD-2-Clause
    
    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    
    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''


# Import modules
# import tensorflow as tf
from keras import backend as K
from keras.losses import mean_squared_error, mean_absolute_error, hinge, squared_hinge, logcosh



def mean_iou(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(K.clip(K.abs(y_true), 0., 1.))
    y_pred_f = K.flatten(K.clip(K.abs(y_pred), 0., 1.))
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def mean_iou_loss(y_true, y_pred):
    return 1. - mean_iou(y_true, y_pred)


def mean_iou_MSE_loss(y_true, y_pred):
    return mean_iou_loss(y_true, y_pred) + mean_squared_error(y_true, y_pred)


def mean_iou_MAE_loss(y_true, y_pred):
    return mean_iou_loss(y_true, y_pred) + mean_absolute_error(y_true, y_pred)


def mean_iou_SVM_loss(y_true, y_pred):
    return mean_iou_loss(y_true, y_pred) + hinge(y_true, y_pred)


def mean_iou_SSVM_loss(y_true, y_pred):
    return mean_iou_loss(y_true, y_pred) + squared_hinge(y_true, y_pred)


def mean_iou_LGC_loss(y_true, y_pred):
    return mean_iou_loss(y_true, y_pred) + logcosh(y_true, y_pred)




def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(K.clip(K.abs(y_true), 0., 1.))
    y_pred_f = K.flatten(K.clip(K.abs(y_pred), 0., 1.))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def dice_coef_MSE_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred) + mean_squared_error(y_true, y_pred)


def dice_coef_MAE_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred) + mean_absolute_error(y_true, y_pred)


def dice_coef_SVM_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred) + hinge(y_true, y_pred)


def dice_coef_SSVM_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred) + squared_hinge(y_true, y_pred)


def dice_coef_LGC_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred) + logcosh(y_true, y_pred)


