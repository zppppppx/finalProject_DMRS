from tensorflow import keras
from tensorflow.keras import layers


def add_common_layers(x):
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    return x


def CE_net(x):
    x = add_common_layers(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1152)(x)
    x = layers.Reshape((48, 12, 2))(x)
    x = keras.layers.LeakyReLU()(x)
    short_cut = x
    for i in range(27):
        x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
        x = add_common_layers(x)
    x = layers.Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = keras.layers.Add()([short_cut, x])
    return x
