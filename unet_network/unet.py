""" U-Net Model """

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Dropout, \
#     UpSampling2D, Cropping2D, BatchNormalization, Activation, Conv2DTranspose
# from tensorflow.keras.optimizers import Adam

from keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                          MaxPooling2D, Conv2DTranspose, concatenate, Dropout, UpSampling2D, Cropping2D)
from keras.models import Model
from keras.optimizers import Adam

# print("Tensorflow", tf.__version__)
# print("Keras", keras.__version__)


def unet(pretrained_weights=None, input_size=(256, 256, 1), n=64):
    inputss = Input(shape=input_size)

    double = n * 2  # 128
    triple = double * 2  # 256
    quadruple = triple * 2  # 512
    penta = quadruple * 2  # 1024

    conv1 = Conv2D(n, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputss)
    conv1 = Conv2D(n, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(double, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(double, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(triple, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(triple, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(quadruple, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(quadruple, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(penta, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(penta, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(quadruple, 2, activation='relu', padding='same', kernel_initializer='he_normal') \
        (UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(quadruple, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(quadruple, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(triple, 2, activation='relu', padding='same', kernel_initializer='he_normal') \
        (UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(triple, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(triple, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(double, 2, activation='relu', padding='same', kernel_initializer='he_normal') \
        (UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(double, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(double, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(n, 2, activation='relu', padding='same', kernel_initializer='he_normal') \
        (UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(n, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(n, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputss, conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def unet2(input_size=(572, 572, 1)):
    inputss = Input(shape=input_size, name='input_image')
    # first contracting block
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', name='CB1_Layer1', activation='relu')(
        inputss)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='valid', name='CB1_Layer2', activation='relu')(
        conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='valid', name='CB1_Layer3', activation='relu')(
        conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same', name='CB1_Pool1')(conv3)
    # second contracting block
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', name='CB2_Layer1', activation='relu')(
        pool1)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='valid', name='CB2_Layer2', activation='relu')(
        conv4)
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='valid', name='CB2_Layer3', activation='relu')(
        conv5)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same', name='CB2_Pool1')(conv6)
    # third contracting block
    conv7 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', name='CB3_Layer1', activation='relu')(
        pool2)
    conv8 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='valid', name='CB3_Layer2', activation='relu')(
        conv7)
    conv9 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='valid', name='CB3_Layer3', activation='relu')(
        conv8)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same', name='CB3_Pool1')(conv9)
    # forth contracting block
    conv10 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', name='CB4_Layer1', activation='relu')(
        pool3)
    conv11 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='valid', name='CB4_Layer2', activation='relu')(
        conv10)
    conv12 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='valid', name='CB4_Layer3', activation='relu')(
        conv11)  # copy and crop
    pool4 = MaxPooling2D(pool_size=(2, 2), padding='same', name='CB4_Pool1')(conv12)
    # connecting block
    conv13 = Conv2D(filters=1024, kernel_size=(3, 3), strides=1, padding='same', name='CB5_Layer1', activation='relu')(
        pool4)
    conv14 = Conv2D(filters=1024, kernel_size=(3, 3), strides=1, padding='valid', name='CB5_Layer2', activation='relu')(
        conv13)
    conv15 = Conv2D(filters=1024, kernel_size=(3, 3), strides=1, padding='valid', name='CB5_Layer3', activation='relu')(
        conv14)
    # first expansive block
    up1 = UpSampling2D(size=(2, 2))(conv15)
    crop1 = Cropping2D(cropping=((4, 4), (4, 4)))(conv12)
    concat1 = concatenate([crop1, up1], axis=3)
    conv16 = Conv2D(filters=1024, kernel_size=(3, 3), strides=1, padding='same', name='EB1_layer1', activation='relu')(
        concat1)
    conv17 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='valid', name='EB1_layer2', activation='relu')(
        conv16)
    conv18 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='valid', name='EB1_layer3', activation='relu')(
        conv17)

    model = Model(inputss, conv18)

    return model


def conv_block(input, num_filters):
    x = Conv2D(filters=num_filters, kernel_size=3, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


# encoder block
def encoder_block(input, num_filters):
    s = conv_block(input, num_filters)
    p = MaxPooling2D(pool_size=(2, 2))(s)
    return s, p


# decoder block
# skip features gets input from encoder for concatenation
def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, kernel_size=(2, 2), strides=2, padding='same')(input)
    x = concatenate([x, skip_features], axis=3)
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(d4)

    model = Model(inputs, outputs, name='U-Net')
    return model

# model = build_unet(input_shape=(256, 256, 1))
# model.summary()
