import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Add, MaxPooling2D, concatenate, UpSampling2D
from tensorflow.keras.layers import ZeroPadding2D, Cropping2D
from typing import Tuple


def unet(
    H: int, W: int, Hpad: int = 3, Wpad: int = 3, kshape: Tuple[int] = (3, 3), channels: int = 24
) -> tf.keras.Model:
    """U-net reconstruction model.

    It receives as input the channel-wise zero-filled reconstruction.
    Reference: Jin et al., "Deep Convolutional Neural Network for Inverse Problems in Imaging", IEEE Tran Img Proc, 2017

    :param H: Spatial height of image
    :type H: int
    :param W: Spatial width of image.
    :type W: int
    :param Hpad: Height zero padding to apply around images, defaults to 3.
    :type Hpad: int, optional
    :param Wpad: Width zero padding to apply around images, defaults to 3.
    :type Wpad: int, optional
    :param kshape: Shape of kernel to use in 2D convolutional layers, defaults to (3, 3)
    :type kshape: Tuple[int], optional
    :param channels: Number of input channels, defaults to 24.
    :type channels: int, optional
    :return: Unet reconstruction model
    :rtype: tf.keras.Model
    """
    inputs = Input(shape=(H, W, channels))
    input_padded = ZeroPadding2D(padding=(Hpad, Wpad))(inputs)  # Pad to compensate for the max-poolings

    conv1 = Conv2D(64, kshape, activation="relu", padding="same")(input_padded)
    conv1 = Conv2D(64, kshape, activation="relu", padding="same")(conv1)
    conv1 = Conv2D(64, kshape, activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, kshape, activation="relu", padding="same")(pool1)
    conv2 = Conv2D(128, kshape, activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, kshape, activation="relu", padding="same")(pool2)
    conv3 = Conv2D(256, kshape, activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, kshape, activation="relu", padding="same")(pool3)
    conv4 = Conv2D(512, kshape, activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, kshape, activation="relu", padding="same")(pool4)
    conv5 = Conv2D(1024, kshape, activation="relu", padding="same")(conv5)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)

    conv6 = Conv2D(512, kshape, activation="relu", padding="same")(up1)
    conv6 = Conv2D(512, kshape, activation="relu", padding="same")(conv6)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)

    conv7 = Conv2D(256, kshape, activation="relu", padding="same")(up2)
    conv7 = Conv2D(256, kshape, activation="relu", padding="same")(conv7)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)

    conv8 = Conv2D(128, kshape, activation="relu", padding="same")(up3)
    conv8 = Conv2D(128, kshape, activation="relu", padding="same")(conv8)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)

    conv9 = Conv2D(128, kshape, activation="relu", padding="same")(up4)
    conv9 = Conv2D(128, kshape, activation="relu", padding="same")(conv9)

    conv10 = Conv2D(channels, (1, 1), activation="linear")(conv9)

    res = Add()([conv10, input_padded])  # Residual

    out = Cropping2D(cropping=(Hpad, Wpad))(res)  # Crop to go back to desired image dimensions

    model = Model(inputs=inputs, outputs=out)

    return model
