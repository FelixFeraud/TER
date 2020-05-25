from keras import callbacks
from keras import optimizers
from keras.applications import resnet

from func_utils import build_tl_model, freeze_n_layers, setup_data_generators, train_model
from layer_builders import basic_layers_builder

data_path = 'drive/My Drive/TER/'

image_size = (224, 224)
train_gen, val_gen = setup_data_generators(data_path, image_size, 32, resnet.preprocess_input)

base_model = resnet.ResNet50(weights = 'imagenet', include_top = False, input_shape = (*image_size, 3))
freeze_n_layers(base_model, len(base_model.layers))

model = build_tl_model(base_model, basic_layers_builder, image_size, 200)

early_stop_tl = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10, min_delta = 0.01)
early_stop_ft = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=15, min_delta = 0.01)
opt_tl = optimizers.Adam(lr = 1e-3)
opt_ft = optimizers.Adam(lr = 1e-6)

train_model(model, train_gen, val_gen, 15, opt_tl, callbacks = [early_stop_tl])

freeze_n_layers(base_model, 81)
# K.set_value(opt.lr, 1e-6)



train_model(model, train_gen, val_gen, 15, opt_ft, callbacks = [early_stop_ft])

model.save(data_path + 'models/test.h5')