'''    
    Copyright (c) 2019, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com

    Released under the BSD license.
    URL: https://opensource.org/licenses/BSD-2-Clause


    <<< About this network >>>
    U-net model
    Revised from
    URL: http://ni4muraano.hatenablog.com/entry/2017/08/10/101053
'''


from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, DepthwiseConv2D, Conv2DTranspose, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Maximum, Concatenate, Add
# from keras.layers.noise import GaussianDropout, AlphaDropout  : Not supported in coremltools, Mar 18, 2019
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.initializers import Constant



def Model_Name():
    return 'U-net'



def Model_Description():
    return         'U-net model for A.I.Segmentation\n\
                    Revised from\n\
                    URL: http://ni4muraano.hatenablog.com/entry/2017/08/10/101053'



def ActivationBy(activation='relu', alpha=0.2):

    if activation == 'leakyrelu':
        return LeakyReLU(alpha=alpha)
    
    elif activation == 'prelu':
        return PReLU(alpha_initializer=Constant(alpha), shared_axes=[1, 2])
    
    elif activation == 'elu':
        return ELU(alpha=alpha)
    
    else:
        return Activation(activation)



############################# ORIGINAL COPY RIGHTS #############################
##    U-net model
##    Revised from
##    URL: http://ni4muraano.hatenablog.com/entry/2017/08/10/101053
################################################################################

def Build_Model():

    def add_encoding_layer(filter_count, sequence):
        new_sequence = ActivationBy('leakyrelu', alpha=0.2)(sequence)
        new_sequence = ZeroPadding2D(1)(new_sequence)
        new_sequence = Conv2D(filter_count, 4, strides=2)(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        return new_sequence

    def add_decoding_layer(filter_count, add_drop_layer, sequence):
        new_sequence = ActivationBy('relu')(sequence)
        new_sequence = Conv2DTranspose(filter_count, 2, strides=2, kernel_initializer='he_uniform')(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        if add_drop_layer:
            new_sequence = Dropout(0.25)(new_sequence)
        return new_sequence


    first_layer_filter_count = 64


    # OpenCV(grayscale) = HEIGHT x WIDTH
    # Keras = HEIGHT x WIDTH x CHANNEL
    inputs = Input(shape=(200, 200, 1), name='input')
    zpad = ZeroPadding2D(28)(inputs)


    # エンコーダーの作成
    # (128 x 128 x N)
    enc1 = ZeroPadding2D(1)(zpad)
    enc1 = Conv2D(first_layer_filter_count, 4, strides=2)(enc1)

    # (64 x 64 x 2N)
    filter_count = first_layer_filter_count*2
    enc2 = add_encoding_layer(filter_count, enc1)

    # (32 x 32 x 4N)
    filter_count = first_layer_filter_count*4
    enc3 = add_encoding_layer(filter_count, enc2)

    # (16 x 16 x 8N)
    filter_count = first_layer_filter_count*8
    enc4 = add_encoding_layer(filter_count, enc3)

    # (8 x 8 x 8N)
    enc5 = add_encoding_layer(filter_count, enc4)

    # (4 x 4 x 8N)
    enc6 = add_encoding_layer(filter_count, enc5)

    # (2 x 2 x 8N)
    enc7 = add_encoding_layer(filter_count, enc6)

    # (1 x 1 x 8N)
    enc8 = add_encoding_layer(filter_count, enc7)

    # デコーダーの作成
    # (2 x 2 x 8N)
    dec1 = add_decoding_layer(filter_count, True, enc8)
    dec1 = Concatenate(axis=-1)([dec1, enc7])

    # (4 x 4 x 8N)
    dec2 = add_decoding_layer(filter_count, True, dec1)
    dec2 = Concatenate(axis=-1)([dec2, enc6])

    # (8 x 8 x 8N)
    dec3 = add_decoding_layer(filter_count, True, dec2)
    dec3 = Concatenate(axis=-1)([dec3, enc5])

    # (16 x 16 x 8N)
    dec4 = add_decoding_layer(filter_count, False, dec3)
    dec4 = Concatenate(axis=-1)([dec4, enc4])

    # (32 x 32 x 4N)
    filter_count = first_layer_filter_count*4
    dec5 = add_decoding_layer(filter_count, False, dec4)
    dec5 = Concatenate(axis=-1)([dec5, enc3])

    # (64 x 64 x 2N)
    filter_count = first_layer_filter_count*2
    dec6 = add_decoding_layer(filter_count, False, dec5)
    dec6 = Concatenate(axis=-1)([dec6, enc2])

    # (128 x 128 x N)
    filter_count = first_layer_filter_count
    dec7 = add_decoding_layer(filter_count, False, dec6)
    dec7 = Concatenate(axis=-1)([dec7, enc1])

    # (256 x 256 x output_channel_count)
    dec8 = Activation(activation='relu')(dec7)
    dec8 = Conv2DTranspose(1, 2, strides=2)(dec8)
    dec8 = Activation(activation='sigmoid')(dec8)


    outputs = Cropping2D(28, name='output')(dec8)


    # Generate model
    return Model(inputs=inputs, outputs=outputs)

