import numpy as np
from keras.models import Input, Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Activation
from keras.layers import Add, Multiply, concatenate, Lambda, conv_utils
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

def unet(num_classes, metrics, loss, lr, pretrained_weights=None, input_size=(224, 224, 3), bn = True):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1) if bn else conv1
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1) if bn else conv1
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2) if bn else conv2
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2) if bn else conv2
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3) if bn else conv3
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3) if bn else conv3
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4) if bn else conv4
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4) if bn else conv4
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5) if bn else conv5
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5) if bn else conv5
    drop5 = Dropout(0.5)(conv5)

    upsampling6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, upsampling6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6) if bn else conv6
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6) if bn else conv6

    upsampling7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, upsampling7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7) if bn else conv7
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7) if bn else conv7

    upsampling8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, upsampling8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8) if bn else conv8
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8) if bn else conv8

    upsampling9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, upsampling9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9) if bn else conv9
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9) if bn else conv9
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9) if bn else conv9

    output = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=output)
    compile_model(model, num_classes, metrics, loss, lr)

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def attention_gating_unet2D(num_classes, metrics, loss, lr, pretrained_weights=None, input_size=(224, 224, 3), bn = True):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1) if bn else conv1
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1) if bn else conv1
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2) if bn else conv2
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2) if bn else conv2
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3) if bn else conv3
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3) if bn else conv3
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4) if bn else conv4
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4) if bn else conv4
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # center
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5) if bn else conv5
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5) if bn else conv5
    drop5 = Dropout(0.5)(conv5)

    # gating
    gating = Conv2D(1024, 1, activation='relu', padding='same', kernel_initializer='he_normal')(drop5)

    att6, att_weight6 = grid_attention_block_2D(drop4, gating, 512)
    upsampling6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([att6, upsampling6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6) if bn else conv6
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6) if bn else conv6

    att7, att_weight7 = grid_attention_block_2D(conv3, gating, 256)
    upsampling7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([att7, upsampling7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7) if bn else conv7
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7) if bn else conv7

    att8, att_weight8 = grid_attention_block_2D(conv2, gating, 128)
    upsampling8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([att8, upsampling8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8) if bn else conv8
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8) if bn else conv8

    att9, att_weight9 = grid_attention_block_2D(conv1, gating, 64)
    upsampling9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([att9, upsampling9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9) if bn else conv9
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9) if bn else conv9
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9) if bn else conv9

    output = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=output)
    compile_model(model, num_classes, metrics, loss, lr)

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def compile_model(model, num_classes, metrics, loss, lr):
    from keras.losses import binary_crossentropy
    from keras.losses import categorical_crossentropy

    from keras.metrics import binary_accuracy
    from keras.metrics import categorical_accuracy

    from keras.optimizers import Adam

    from metrics import dice_coeff
    from metrics import jaccard_index
    from metrics import class_jaccard_index
    from metrics import pixelwise_precision
    from metrics import pixelwise_sensitivity
    from metrics import pixelwise_specificity
    from metrics import pixelwise_recall

    if isinstance(loss, str):
        if loss in {'ce', 'crossentropy'}:
            if num_classes == 1:
                loss = binary_crossentropy
            else:
                loss = categorical_crossentropy
        else:
            raise ValueError('unknown loss %s' % loss)

    if isinstance(metrics, str):
        metrics = [metrics, ]

    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'acc':
            metrics[i] = binary_accuracy if num_classes == 1 else categorical_accuracy
        elif metric == 'jaccard_index':
            metrics[i] = jaccard_index(num_classes)
        elif metric == 'jaccard_index0':
            metrics[i] = class_jaccard_index(0)
        elif metric == 'jaccard_index1':
            metrics[i] = class_jaccard_index(1)
        elif metric == 'jaccard_index2':
            metrics[i] = class_jaccard_index(2)
        elif metric == 'jaccard_index3':
            metrics[i] = class_jaccard_index(3)
        elif metric == 'jaccard_index4':
            metrics[i] = class_jaccard_index(4)
        elif metric == 'jaccard_index5':
            metrics[i] = class_jaccard_index(5)
        elif metric == 'dice_coeff':
            metrics[i] = dice_coeff(num_classes)
        elif metric == 'pixelwise_precision':
            metrics[i] = pixelwise_precision(num_classes)
        elif metric == 'pixelwise_sensitivity':
            metrics[i] = pixelwise_sensitivity(num_classes)
        elif metric == 'pixelwise_specificity':
            metrics[i] = pixelwise_specificity(num_classes)
        elif metric == 'pixelwise_recall':
            metrics[i] = pixelwise_recall(num_classes)
        else:
            raise ValueError('metric %s not recognized' % metric)

    model.compile(optimizer=Adam(lr=lr),
                  loss=loss,
                  metrics=metrics)

def grid_attention_block_2D(x, g, out_filter, dim=2, sub_sample_factor=(2,2), bn= True):    

    """
    x: Input features before downsampling
    g: Gating signal from the coarser features
    """
    assert isinstance(sub_sample_factor, tuple) 
    assert isinstance(dim, int)
    # preserve the filters in input x
    num_filter = int(x.shape[3])

    # linear transformation using 1x1 convolution
    gating_signal = Conv2D(filters=num_filter,
                           kernel_size=(1,1),
                           strides=1,
                           use_bias=True,
                           padding='same',
                           kernel_initializer='he_normal')(g)

    sub_sample_kernel_size = sub_sample_factor
    # linear transformation using 2x2 convolution, strides = 2 
    input_feature = Conv2D(filters=num_filter,
                           kernel_size=sub_sample_kernel_size,
                           strides=2,
                           use_bias=False,
                           padding='same',
                           kernel_initializer='he_normal')(x)

    input_feature_size = conv_utils.normalize_tuple(value=int(input_feature.shape[1]),
                                                    n=dim,
                                                    name="input_feature_size")
    gating_signal = UpSampling2DBilinear(input_feature_size)(gating_signal)
    additive_attention = Add()([input_feature, gating_signal])
    relu = Activation('relu')(additive_attention)

    conv_attention = Conv2D(filters = num_filter,
                            kernel_size=(1,1),
                            strides=1,
                            use_bias=True,
                            padding='same',
                            kernel_initializer='he_normal',
                            activation='sigmoid')(relu)

    upsample_size = conv_utils.normalize_tuple(value=int(x.shape[1]),
                                               n=dim ,
                                               name='upsample_size')
    upsample_attention_weight = UpSampling2DBilinear(upsample_size)(conv_attention)
    element_wise_multiplication = Multiply()([upsample_attention_weight, x])

    conv_out = Conv2D(filters= out_filter,
                      kernel_size=(1,1),
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal')(element_wise_multiplication)
    conv_out = BatchNormalization()(conv_out) if bn else conv_out
    return conv_out, upsample_attention_weight

def UpSampling2DBilinear(size):
    return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True))

