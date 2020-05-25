from keras import callbacks
from keras import optimizers
from keras.applications import resnet

from func_utils import build_tl_model, freeze_n_layers, setup_data_generators, train_model, makedirs_if
from layer_builders import basic_layers_builder

data_path = 'drive/My Drive/TER/'

image_size = (224, 224)

batch_size = 32
epochs = 30

layers_to_freeze = 81

train_gen, val_gen = setup_data_generators(data_path, image_size, batch_size, resnet.preprocess_input)

base_model = resnet.ResNet50(weights = 'imagenet', include_top = False, input_shape = (*image_size, 3))
freeze_n_layers(base_model, len(base_model.layers))

model = build_tl_model(base_model, basic_layers_builder, image_size, 200)

early_stop_tl = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
early_stop_ft = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=15, min_delta = 0.01)
opt_tl = optimizers.Adam(lr = 1e-3)
opt_ft = optimizers.Adam(lr = 1e-6)

train_model(model, train_gen, val_gen, epochs, opt_tl, callbacks = [early_stop_tl])

freeze_n_layers(base_model, layers_to_freeze)

train_model(model, train_gen, val_gen, epochs, opt_ft, callbacks = [early_stop_ft])

makedirs_if(data_path + 'models/')
model.save(data_path + 'models/test.h5')