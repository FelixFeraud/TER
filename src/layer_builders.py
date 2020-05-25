from keras import backend as K
from keras.layers import Dense, GlobalAveragePooling2D, Lambda


def basic_layers_builder(inputs, nb_classes):
    x = GlobalAveragePooling2D()(inputs)
    return Dense(nb_classes, activation='softmax')(x)


def l2_layer_builder(inputs, nb_classes):
    x = GlobalAveragePooling2D()(inputs)
    x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)
    x = Dense(1024, activation='relu')(x)
    return Dense(nb_classes, activation='softmax')(x)
